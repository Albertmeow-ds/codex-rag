import unittest

from rag.chunking import chunk_document
from rag.embed.hashed import HashedNGramEmbedder
from rag.index.bm25 import BM25Index
from rag.index.dense import DenseIndex
from rag.index.late import LateInteractionIndex
from rag.llm import NullLLM
from rag.rerank import CompositeReranker, HeuristicReranker, LateInteractionReranker, LLMListwiseReranker, build_reranker
from rag.retrieve.hybrid import HybridRetriever, RetrievalConfig
from rag.text import content_tokens
from rag.types import Document

DOCS = [
    Document("on", "# Rerank\n\nLate interaction reranking compares token level embeddings with maxsim after first stage retrieval.", "Rerank"),
    Document("off", "# Unrelated\n\nCooking a stew requires slow heat and patience over several hours.", "Unrelated"),
]


def units_for(query: str):
    chunks = [c for d in DOCS for c in chunk_document(d)]
    embedder = HashedNGramEmbedder(256)
    bm25 = BM25Index().build([(c.chunk_id, content_tokens(c.text)) for c in chunks])
    dense = DenseIndex().build([c.chunk_id for c in chunks], embedder.embed([c.text for c in chunks]))
    late = LateInteractionIndex().build([(c.chunk_id, embedder.embed_tokens(c.text)) for c in chunks])
    retriever = HybridRetriever(bm25, dense, late, chunks, embedder, RetrievalConfig(top_k=4, candidate_k=12))
    return retriever.retrieve(query).units, late, embedder


class RerankTests(unittest.TestCase):
    def test_heuristic_promotes_topical_passage(self):
        units, _late, _emb = units_for("maxsim token level reranking")
        ranked = HeuristicReranker().rerank("maxsim token level reranking", units)
        self.assertEqual(ranked[0].chunk.doc_id, "on")

    def test_late_reranker_uses_final_score_as_base(self):
        units, late, embedder = units_for("maxsim token level reranking")
        for unit in units:
            unit.score = 5.0
        ranked = LateInteractionReranker(late, embedder).rerank("maxsim token level reranking", units)
        self.assertTrue(all(u.rerank_score is not None and u.rerank_score >= 5.0 for u in ranked))

    def test_composite_chains_preserve_ordering_signal(self):
        units, late, embedder = units_for("maxsim token level reranking")
        chain = CompositeReranker([HeuristicReranker(), LateInteractionReranker(late, embedder, weight=0.1)])
        ranked = chain.rerank("maxsim token level reranking", units)
        self.assertEqual(ranked[0].chunk.doc_id, "on")

    def test_llm_reranker_falls_back_when_model_fails(self):
        units, _late, _emb = units_for("maxsim token level reranking")
        ranked = LLMListwiseReranker(NullLLM()).rerank("maxsim token level reranking", units)
        self.assertEqual(len(ranked), len(units))

    def test_build_reranker_comma_chain(self):
        units, late, embedder = units_for("maxsim token level reranking")
        reranker = build_reranker("heuristic,late", late_index=late, embedder=embedder)
        self.assertIsInstance(reranker, CompositeReranker)
        self.assertEqual(reranker.name, "heuristic+late-interaction")

    def test_build_reranker_none(self):
        self.assertIsNone(build_reranker("none"))


if __name__ == "__main__":
    unittest.main()