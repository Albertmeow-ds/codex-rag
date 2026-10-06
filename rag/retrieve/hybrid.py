"""Hybrid retrieval: BM25 + dense (+ optional late interaction) fused by RRF or weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from rag.chunking import sentence_window
from rag.index.bm25 import BM25Index
from rag.index.dense import DenseIndex
from rag.index.late import LateInteractionIndex
from rag.text import content_tokens, tokenize
from rag.types import Chunk, Retrieved, RetrievedUnit, Trace


DEFAULT_WEIGHTS = {"bm25": 1.0, "dense": 1.0, "late": 0.5}

# Measured on examples/corpus + examples/eval.jsonl (see examples/embedder_comparison.py).
# Lexical embedders (hashed n-gram, LSA) live in the same space as BM25, so BM25 should lead.
# Real embedding models carry cross-lingual signal BM25 cannot see, so dense should lead.
WEIGHT_PROFILES = {
    "lexical": {"bm25": 1.2, "dense": 0.8, "late": 0.5},
    "semantic": {"bm25": 0.5, "dense": 1.5, "late": 0.5},
}


@dataclass(slots=True)
class RetrievalConfig:
    candidate_k: int = 40
    top_k: int = 8
    channels: tuple[str, ...] = ("bm25", "dense")
    fusion: str = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] | None = None
    sentence_window_radius: int = 0
    expand_parents: bool = False
    filters: dict[str, Any] = field(default_factory=dict)
    max_per_document: int = 0

    def resolved_weights(self) -> dict[str, float]:
        """Explicit weights always win; None means 'pick the profile for the built embedder'."""

        return self.weights or DEFAULT_WEIGHTS


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        late: LateInteractionIndex,
        chunks: Sequence[Chunk],
        embedder: Any,
        config: RetrievalConfig | None = None,
        reranker: Any = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.late = late
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.reranker = reranker
        self.chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self.indexable_chunks = list(chunks)
        self.parents_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks if c.parent_id is None}
        self.trace = Trace()

    def _passes_filter(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        if not filters:
            return True
        include = filters.get("doc_ids")
        if include and chunk.doc_id not in include:
            return False
        exclude = filters.get("exclude_doc_ids")
        if exclude and chunk.doc_id in exclude:
            return False
        heading_contains = filters.get("heading_contains")
        if heading_contains and heading_contains.lower() not in " ".join(chunk.heading_path).lower():
            return False
        metadata = filters.get("metadata") or {}
        for key, value in metadata.items():
            if chunk.metadata.get(key) != value:
                return False
        return True

    def _channel_scores(self, query: str, candidate_k: int) -> dict[str, dict[str, float]]:
        channels: dict[str, dict[str, float]] = {}
        tokens = content_tokens(query)
        if "bm25" in self.config.channels and self.bm25.doc_ids:
            hits = self.bm25.search(tokens, candidate_k)
            if hits:
                channels["bm25"] = {chunk_id: score for score, chunk_id in hits}
        if "dense" in self.config.channels and self.dense.size:
            vectors = self.embedder.embed([query])
            hits = self.dense.search(vectors[0], candidate_k)
            if hits:
                channels["dense"] = {chunk_id: score for score, chunk_id in hits}
        if "late" in self.config.channels and self.late.enabled:
            query_tokens = self.embedder.embed_tokens(query)
            universe = set()
            for scores in channels.values():
                universe.update(scores)
            if universe:
                channels["late"] = self.late.score_many(query_tokens, sorted(universe))
        return channels

    def _fuse(self, channels: dict[str, dict[str, float]]) -> dict[str, tuple[float, dict[str, float]]]:
        if not channels:
            return {}
        weights = self.config.resolved_weights()
        if self.config.fusion == "rrf":
            fused: dict[str, float] = {}
            contributions: dict[str, dict[str, float]] = {}
            for name, scores in channels.items():
                weight = weights.get(name, 1.0)
                ordered = sorted(scores.items(), key=lambda kv: -kv[1])
                for rank, (chunk_id, _score) in enumerate(ordered, start=1):
                    value = weight / (self.config.rrf_k + rank)
                    fused[chunk_id] = fused.get(chunk_id, 0.0) + value
                    contributions.setdefault(chunk_id, {})[name] = value
            return {cid: (score, contributions[cid]) for cid, score in fused.items()}

        fused = {}
        contributions = {}
        for name, scores in channels.items():
            if not scores:
                continue
            values = np.asarray(list(scores.values()), dtype=np.float64)
            lo, hi = float(values.min()), float(values.max())
            span = (hi - lo) or 1.0
            weight = weights.get(name, 1.0)
            for chunk_id, score in scores.items():
                normalized = (score - lo) / span
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight * normalized
                contributions.setdefault(chunk_id, {})[name] = normalized
        return {cid: (score, contributions[cid]) for cid, score in fused.items()}

    def _expand(self, chunk: Chunk) -> str:
        if self.config.expand_parents and chunk.parent_id and chunk.parent_id in self.parents_by_id:
            return self.parents_by_id[chunk.parent_id].text
        if self.config.sentence_window_radius > 0:
            return sentence_window(self.indexable_chunks, chunk, self.config.sentence_window_radius)
        return ""

    def retrieve(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> Retrieved:
        cfg = self.config
        self.trace = Trace()
        limit = top_k or cfg.top_k
        merged_filters = {**cfg.filters, **(filters or {})}
        trace_channels: dict[str, int] = {}

        if self.dense.size and self.embedder is not None and "dense" in cfg.channels:
            pass

        channels = self._channel_scores(query, cfg.candidate_k)
        trace_channels = {name: len(scores) for name, scores in channels.items()}

        fused = self._fuse(channels)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1][0])

        units: list[RetrievedUnit] = []
        per_document: dict[str, int] = {}
        cap = cfg.max_per_document
        for chunk_id, (score, contributions) in ranked:
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is None or not self._passes_filter(chunk, merged_filters):
                continue
            # A case library wants distinct cases, not three chunks of the best-matching one.
            if cap and per_document.get(chunk.doc_id, 0) >= cap:
                continue
            per_document[chunk.doc_id] = per_document.get(chunk.doc_id, 0) + 1
            units.append(
                RetrievedUnit(
                    chunk=chunk,
                    score=float(score),
                    rank=len(units) + 1,
                    channels=contributions,
                    window_text=self._expand(chunk),
                )
            )
            if len(units) >= limit:
                break

        retrieved = Retrieved(query=query, units=units, trace=self.trace)
        retrieved.trace.record("retrieve", channels=trace_channels, fusion=cfg.fusion, returned=len(units))

        if self.reranker is not None and units:
            units = self.reranker.rerank(query, units)
            for rank, unit in enumerate(units, start=1):
                unit.rank = rank
            retrieved.units = units
            retrieved.trace.record("rerank", reranker=getattr(self.reranker, "name", "unknown"))

        return retrieved

    def explain(self, query: str, chunk_id: str) -> dict[str, Any]:
        tokens = tokenize(query)
        return {
            "bm25_terms": self.bm25.explain(content_tokens(query), chunk_id),
            "chunk": self.chunks_by_id.get(chunk_id).text[:240] if self.chunks_by_id.get(chunk_id) else "",
            "raw_tokens": tokens[:24],
        }