import unittest

from rag.chunking import chunk_document
from rag.embed.hashed import HashedNGramEmbedder
from rag.index.bm25 import BM25Index
from rag.index.dense import DenseIndex
from rag.index.late import LateInteractionIndex
from rag.retrieve.hybrid import HybridRetriever, RetrievalConfig
from rag.text import content_tokens
from rag.types import Document

DOCS = [
    Document("hybrid", "# Hybrid\n\nHybrid retrieval fuses BM25 lexical scores with dense vector scores using rank fusion.", "Hybrid"),
    Document("chunk", "# Chunking\n\nSemantic chunking chooses breakpoints by embedding similarity between sentences.", "Chunking"),
    Document("eval", "# Eval\n\nnDCG and MRR measure ranking quality of a retrieval system.", "Eval"),
]


def build(config: RetrievalConfig | None = None) -> HybridRetriever:
    chunks = [c for d in DOCS for c in chunk_document(d)]
    embedder = HashedNGramEmbedder(256)
    bm25 = BM25Index().build([(c.chunk_id, content_tokens(c.text)) for c in chunks])
    dense = DenseIndex().build([c.chunk_id for c in chunks], embedder.embed([c.text for c in chunks]))
    late = LateInteractionIndex().build([(c.chunk_id, embedder.embed_tokens(c.text)) for c in chunks])
    return HybridRetriever(bm25, dense, late, chunks, embedder, config or RetrievalConfig(top_k=3, candidate_k=12))


class HybridTests(unittest.TestCase):
    def test_rrf_rewards_agreement_between_channels(self):
        retriever = build(RetrievalConfig(top_k=3, candidate_k=12, fusion="rrf"))
        result = retriever.retrieve("bm25 dense vector scores")
        self.assertEqual(result.units[0].chunk.doc_id, "hybrid")
        self.assertIn("bm25", result.units[0].channels)
        self.assertIn("dense", result.units[0].channels)

    def test_weighted_fusion_normalizes_channels(self):
        retriever = build(RetrievalConfig(top_k=3, candidate_k=12, fusion="weighted"))
        result = retriever.retrieve("bm25 dense vector scores")
        self.assertEqual(result.units[0].chunk.doc_id, "hybrid")
        self.assertLessEqual(max(result.units[0].channels.values()), 1.0001)

    def test_max_per_document_keeps_results_diverse(self):
        many = Document(
            "case",
            "# Case\n\n" + "\n\n".join(
                f"## Section {i}\n\nHybrid retrieval fuses BM25 lexical scores with dense vectors in section {i}."
                for i in range(6)
            ),
            "Case",
        )
        chunks = [c for c in chunk_document(many)] + [c for d in DOCS for c in chunk_document(d)]
        embedder = HashedNGramEmbedder(256)
        bm25 = BM25Index().build([(c.chunk_id, content_tokens(c.text)) for c in chunks])
        dense = DenseIndex().build([c.chunk_id for c in chunks], embedder.embed([c.text for c in chunks]))
        late = LateInteractionIndex().build([(c.chunk_id, embedder.embed_tokens(c.text)) for c in chunks])

        uncapped = HybridRetriever(bm25, dense, late, chunks, embedder, RetrievalConfig(top_k=4, candidate_k=20))
        before = uncapped.retrieve("BM25 lexical dense vectors")
        self.assertGreater(
            sum(1 for u in before.units if u.chunk.doc_id == "case"), 1,
            "expected several chunks of one document without a cap",
        )

        capped = HybridRetriever(
            bm25, dense, late, chunks, embedder,
            RetrievalConfig(top_k=4, candidate_k=20, max_per_document=1),
        )
        after = capped.retrieve("BM25 lexical dense vectors")
        self.assertEqual(len({u.chunk.doc_id for u in after.units}), len(after.units))
        self.assertEqual(after.units[0].chunk.doc_id, "case")

    def test_metadata_filter_excludes_docs(self):
        retriever = build()
        result = retriever.retrieve("ranking quality nDCG MRR", filters={"exclude_doc_ids": ["eval"]})
        self.assertTrue(all(u.chunk.doc_id != "eval" for u in result.units))

    def test_heading_filter(self):
        retriever = build()
        result = retriever.retrieve("chunking similarity", filters={"heading_contains": "Chunking"})
        self.assertTrue(result.units)
        self.assertTrue(all("Chunking" in " ".join(u.chunk.heading_path) for u in result.units))

    def test_parent_expansion_uses_parent_text(self):
        chunks = [c for d in DOCS for c in chunk_document(d, __import__("rag.chunking", fromlist=["x"]).ChunkingConfig(strategy="hierarchical", parent_tokens=60, child_tokens=16))]
        embedder = HashedNGramEmbedder(128)
        indexable = [c for c in chunks if c.parent_id]
        bm25 = BM25Index().build([(c.chunk_id, content_tokens(c.text)) for c in indexable])
        dense = DenseIndex().build([c.chunk_id for c in indexable], embedder.embed([c.text for c in indexable]))
        late = LateInteractionIndex()
        retriever = HybridRetriever(bm25, dense, late, chunks, embedder, RetrievalConfig(top_k=3, candidate_k=12, expand_parents=True))
        result = retriever.retrieve("embedding similarity breakpoints")
        self.assertTrue(result.units)
        self.assertTrue(all(len(u.window_text) >= len(u.chunk.text) for u in result.units))

    def test_empty_query_returns_nothing(self):
        self.assertEqual(build().retrieve("").units, [])


if __name__ == "__main__":
    unittest.main()