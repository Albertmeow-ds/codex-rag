"""End-to-end pipeline wiring: ingest -> index -> retrieve -> rerank -> assemble -> answer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from rag.agentic import AgenticConfig, AgenticRAG, RetrievalGrader
from rag.chunking import chunk_document
from rag.config import RAGConfig
from rag.embed.base import Embedder
from rag.embed.cache import EmbeddingCache
from rag.embed.factory import build_embedder
from rag.embed.lsa import LSAEmbedder
from rag.generate import build_answerer
from rag.index.bm25 import BM25Index
from rag.index.dense import DenseIndex
from rag.index.late import LateInteractionIndex
from rag.index.store import ChunkStore
from rag.llm import LLM, build_llm
from rag.usage import CountingLLM, Usage
from rag.rerank import build_reranker
from rag.retrieve.hybrid import DEFAULT_WEIGHTS, WEIGHT_PROFILES, HybridRetriever
from rag.retrieve.web import WebSearchRetriever
from rag.text import content_tokens
from rag.transform import build_transform
from rag.types import Answer, Chunk, Document, Retrieved, RetrievedUnit

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".json", ".csv", ".html"}


@dataclass(slots=True)
class IndexStats:
    documents: int = 0
    chunks: int = 0
    skipped: int = 0
    embedded: int = 0
    embedder: str = ""
    elapsed_s: float = 0.0


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    documents: list[Document] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = sorted(p for p in path.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES and p.is_file())
        else:
            candidates = [path]
        for file_path in candidates:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            title = file_path.stem
            if file_path.suffix.lower() in {".md", ".markdown"}:
                for line in text.split("\n"):
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
            documents.append(
                Document(
                    doc_id=str(file_path).replace("\\", "/"),
                    text=text,
                    source=str(file_path),
                    title=title,
                    metadata={"source": str(file_path), "suffix": file_path.suffix.lower()},
                )
            )
    return documents


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.store = ChunkStore(self.config.db_path)
        self.embedder: Embedder = build_embedder(
            self.config.embedder, dim=self.config.embed_dim, ollama_host=self.config.ollama_host
        )
        self.llm: LLM | None = build_llm(
            self.config.llm,
            ollama_model=self.config.ollama_model,
            ollama_host=self.config.ollama_host,
            base_url=self.config.llm_base_url,
            model=self.config.llm_model,
            wire_api=self.config.llm_wire_api,
        )
        if self.llm is not None and getattr(self.llm, "name", "") == "none":
            self.llm = None
        if self.config.retrieval.weights is None:
            profile = getattr(self.embedder, "weight_profile", "lexical")
            self.config.retrieval.weights = WEIGHT_PROFILES.get(profile, DEFAULT_WEIGHTS)
        self.bm25 = BM25Index()
        self.cache = EmbeddingCache(self.config.embed_cache_path)
        self.dense = DenseIndex()
        self.late = LateInteractionIndex()
        self.retriever: HybridRetriever | None = None
        web_config = self.config.web
        self.web = WebSearchRetriever(
            timeout=web_config.timeout,
            max_results=web_config.max_results,
            fetch_pages=web_config.fetch_pages,
            backend=web_config.backend,
            prefer_snippet=web_config.prefer_snippet,
            min_snippet_chars=web_config.min_snippet_chars,
        )
        self.stats = IndexStats()

    def index_paths(self, paths: Sequence[str | Path]) -> IndexStats:
        return self.index_documents(load_documents(paths))

    def index_documents(self, documents: Sequence[Document]) -> IndexStats:
        import time

        started = time.time()
        new_chunks: list[Chunk] = []
        skipped = 0
        for document in documents:
            if self.store.document_fingerprint(document.doc_id) == document.fingerprint:
                skipped += 1
                continue
            chunks = chunk_document(document, self.config.chunking)
            if isinstance(self.embedder, LSAEmbedder):
                new_chunks.extend(chunks)
            self.store.upsert_document(document)
            self.store.replace_chunks(document.doc_id, chunks)
        self.store.commit()

        if isinstance(self.embedder, LSAEmbedder):
            corpus = [c.text for c in self._all_chunks()] or [c.text for c in new_chunks]
            self.embedder.fit(corpus or [""])

        self._embed_indexable()
        self._rebuild_indexes()

        counts = self.store.counts()
        self.stats = IndexStats(
            documents=counts["documents"],
            chunks=counts["chunks"],
            skipped=skipped,
            embedded=counts["vectors"],
            embedder=self.embedder.name,
            elapsed_s=round(time.time() - started, 3),
        )
        return self.stats

    def _all_chunks(self) -> list[Chunk]:
        return self.store.load_chunks(indexable_only=False)

    def _stored_vector_ids(self) -> set[str]:
        return {row[0] for row in self.store.conn.execute("SELECT chunk_id FROM vectors")}

    def _indexable_chunks(self) -> list[Chunk]:
        """Selection is driven by the chunking strategy, never by what is already embedded.

        Filtering by stored vectors first would deadlock incremental indexing: a new
        chunk has no vector yet, so it would be excluded before it could ever be embedded.
        """

        return self._select_indexable(self._all_chunks())

    def _select_indexable(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Small-to-big: only children are search units; parents are context, not candidates."""

        if self.config.chunking.strategy == "hierarchical":
            children = [c for c in chunks if c.parent_id is not None]
            if children:
                return children
        return list(chunks)

    def _embed_chunks(self, chunks: Sequence[Chunk]) -> dict[str, np.ndarray]:
        """Reuse cached vectors so re-indexing only pays for genuinely new text.

        LSA is corpus-fitted, so its vectors are not reusable across indexes.
        """

        if not chunks:
            return {}
        cacheable = not isinstance(self.embedder, LSAEmbedder)
        dim = int(getattr(self.embedder, "dim", 0) or 0)
        vectors: dict[str, np.ndarray] = {}
        pending: list[Chunk] = []
        if cacheable and dim:
            cached = self.cache.get(self.embedder.name, dim, [c.text for c in chunks])
            for chunk in chunks:
                hit = cached.get(chunk.text)
                if hit is not None and hit.shape[0] == dim:
                    vectors[chunk.chunk_id] = hit
                else:
                    pending.append(chunk)
        else:
            pending = list(chunks)
        if pending:
            fresh = self.embedder.embed([c.text for c in pending])
            for chunk, vector in zip(pending, fresh):
                vectors[chunk.chunk_id] = np.asarray(vector, dtype=np.float32)
            if cacheable:
                self.cache.put(self.embedder.name, dim, [c.text for c in pending], np.asarray(fresh, dtype=np.float32))
        return vectors

    def _embed_indexable(self) -> None:
        stored = self._stored_vector_ids()
        missing = [c for c in self._indexable_chunks() if c.chunk_id not in stored]
        if not missing:
            return
        for chunk_id, vector in self._embed_chunks(missing).items():
            self.store.put_vector(chunk_id, self.embedder.name, vector)
        self.store.commit()

    def _rebuild_indexes(self) -> None:
        chunks = self._all_chunks()
        self._embed_indexable()
        indexable = self._indexable_chunks()
        stored_vectors = dict(zip(*self.store.load_vectors(self.embedder.name))) if self.store.counts()["vectors"] else {}
        usable = [c for c in indexable if c.chunk_id in stored_vectors] or indexable
        if not usable:
            raise RuntimeError("no chunks available to index")
        missing_vectors = [c for c in usable if c.chunk_id not in stored_vectors]
        if missing_vectors:
            stored_vectors.update(self._embed_chunks(missing_vectors))
            self.store.commit()

        ids = [c.chunk_id for c in usable]
        matrix = np.vstack([stored_vectors[c.chunk_id] for c in usable])

        self.bm25.build([(c.chunk_id, content_tokens(c.text)) for c in usable])
        self.dense.build(ids, matrix)
        self.late.build([(c.chunk_id, self.embedder.embed_tokens(c.text)) for c in usable])
        self.retriever = HybridRetriever(
            self.bm25,
            self.dense,
            self.late,
            chunks,
            self.embedder,
            self.config.retrieval,
            reranker=build_reranker(
                self.config.reranker, late_index=self.late, embedder=self.embedder, llm=self.llm
            ),
        )

    def augment(self, documents: Sequence[Document]) -> None:
        """Add live web results into the same index mid-query."""

        for document in documents:
            self.store.upsert_document(document)
            self.store.replace_chunks(document.doc_id, chunk_document(document, self.config.chunking))
        self.store.commit()
        if isinstance(self.embedder, LSAEmbedder):
            self.embedder.fit([c.text for c in self._all_chunks()] or [""])
        self._embed_indexable()
        self._rebuild_indexes()

    def retrieve(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> Retrieved:
        self._require_retriever()
        assert self.retriever is not None
        return self.retriever.retrieve(query, top_k=top_k, filters=filters)

    def answer(self, query: str, agentic: bool | None = None, filters: dict[str, Any] | None = None) -> Answer:
        self._require_retriever()
        assert self.retriever is not None
        use_agentic = self.config.agentic if agentic is None else agentic
        usage = Usage()
        # One ledger for the whole answer: the grader, the rewriter and the generator all
        # draw from the same budget, which is the only way "how much did this cost" is answerable.
        run_llm = CountingLLM(self.llm, usage) if self.llm is not None else None
        answerer = build_answerer(run_llm, self.config.generation)

        if not use_agentic:
            retrieved = self.retrieve(query, filters=filters)
            answer = answerer.answer(query, retrieved.units)
            answer.trace = retrieved.trace
            answer.trace.record("answer", answerer=answerer.name, sources=len(answer.units))
            answer.trace.usage = usage.as_dict()
            return answer

        transform = build_transform(self.config.transform, run_llm)
        agent = AgenticRAG(
            self.retriever,
            answerer,
            config=self.config.agentic_config,
            grader=RetrievalGrader(run_llm),
            transform=transform,
            web=self.web if self.config.agentic_config.allow_web_fallback else None,
            augment=self.augment,
            usage=usage,
        )
        answer = agent.run(query, filters=filters)
        return answer

    def explain(self, query: str, chunk_id: str) -> dict[str, Any]:
        self._require_retriever()
        assert self.retriever is not None
        return self.retriever.explain(query, chunk_id)

    def _require_retriever(self) -> None:
        if self.retriever is None:
            self._rebuild_indexes()
        if self.retriever is None or self.dense.size == 0:
            raise RuntimeError("no indexed content; call index_paths() first")

    def doctor(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.name,
            "embed_dim": getattr(self.embedder, "dim", 0),
            "llm": getattr(self.llm, "name", "none"),
            "web_search": self.web.available(),
            "web_reachable": self.web.reachable_backends(),
            "store": self.store.counts(),
            "reranker": self.config.reranker,
            "fusion": self.config.retrieval.fusion,
            "weights": self.config.retrieval.resolved_weights(),
            "transform": self.config.transform,
            "chunking": self.config.chunking.strategy,
            "agentic": {
                "enabled": self.config.agentic_config.enabled,
                "max_rounds": self.config.agentic_config.max_rounds,
                "max_llm_calls": self.config.agentic_config.max_llm_calls,
                "max_total_tokens": self.config.agentic_config.max_total_tokens,
                "evidence": self.config.agentic_config.evidence,
                "evidence_gate": self.config.agentic_config.evidence_gate,
            },
        }

    def close(self) -> None:
        self.cache.close()
        self.store.close()


def build_pipeline(config_path: str | Path | None = None, **overrides: Any) -> RAGPipeline:
    config = RAGConfig.from_file(config_path) if config_path else RAGConfig()
    if overrides:
        config = RAGConfig.from_dict({**config.to_dict(), **overrides})
    return RAGPipeline(config)