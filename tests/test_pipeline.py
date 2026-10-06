import unittest
from pathlib import Path

from rag.config import RAGConfig
from rag.eval import evaluate_retrieval, load_dataset
from rag.pipeline import RAGPipeline, load_documents
from rag.types import Document

DB = Path("data/test_pipeline.db")


def make_pipeline(db_path: Path | None = None, **overrides) -> RAGPipeline:
    db = db_path or DB
    if db.exists():
        db.unlink()
    config = RAGConfig.from_file("config.yaml")
    config.db_path = str(db)
    config.llm = "none"
    for key, value in overrides.items():
        setattr(config, key, value)
    pipeline = RAGPipeline(config)
    pipeline.index_paths(["examples/corpus"])
    return pipeline


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = make_pipeline()

    @classmethod
    def tearDownClass(cls):
        cls.pipeline.close()

    def test_indexing_produced_chunks_and_vectors(self):
        counts = self.pipeline.store.counts()
        self.assertGreater(counts["documents"], 0)
        self.assertGreater(counts["chunks"], 0)
        self.assertGreaterEqual(counts["vectors"], 0)

    def test_hierarchical_only_indexes_children(self):
        indexed = {row[0] for row in self.pipeline.store.conn.execute("SELECT chunk_id FROM vectors")}
        self.assertTrue(indexed)
        self.assertTrue(all("#p" not in chunk_id for chunk_id in indexed))

    def test_retrieval_metrics_beat_random(self):
        metrics = evaluate_retrieval(load_dataset("examples/eval.jsonl"), lambda q: self.pipeline.retrieve(q).chunk_ids, k=5)
        self.assertGreater(metrics["hit@5"], 0.9)
        self.assertGreater(metrics["ndcg@5"], 0.7)

    def test_answer_without_llm_is_extractive_and_cited(self):
        answer = self.pipeline.answer("nDCG 和 MRR 分别衡量什么", agentic=False)
        self.assertTrue(answer.text)
        self.assertRegex(answer.text, r"\[\d+\]")
        self.assertGreater(answer.groundedness or 0.0, 0.5)

    def test_agentic_loop_records_trace_stages(self):
        answer = self.pipeline.answer("RRF 融合相比加权分数融合有什么优缺点", agentic=True)
        stages = [entry["stage"] for entry in answer.trace.stages]
        self.assertIn("retrieve", stages)
        self.assertIn("grade", stages)

    def test_incremental_reindex_skips_unchanged(self):
        stats = self.pipeline.index_paths(["examples/corpus"])
        self.assertEqual(stats.documents, len(load_documents(["examples/corpus"])))
        self.assertGreater(stats.skipped, 0)

    def test_new_document_gets_embedded_on_second_index_run(self):
        """Regression: indexable selection must not be filtered by already-stored vectors."""

        fresh = Path("data/test_incremental.db")
        if fresh.exists():
            fresh.unlink()
        config = RAGConfig.from_file("config.yaml")
        config.db_path = str(fresh)
        config.llm = "none"
        pipeline = RAGPipeline(config)
        pipeline.index_paths(["examples/corpus/01-hybrid-retrieval.md"])
        first_vectors = pipeline.store.counts()["vectors"]
        pipeline.index_paths(["examples/corpus/02-chunking.md", "examples/corpus/03-reranking.md"])
        second_vectors = pipeline.store.counts()["vectors"]
        self.assertGreater(second_vectors, first_vectors)
        hits = pipeline.retrieve("semantic chunking breakpoints embedding similarity", top_k=3)
        self.assertTrue(any("02-chunking" in u.chunk.doc_id for u in hits.units))
        pipeline.close()
        fresh.unlink(missing_ok=True)

    def test_augmented_web_document_is_retrievable(self):
        pipeline = make_pipeline(db_path=Path("data/test_augment.db"))
        before = pipeline.dense.size
        pipeline.augment(
            [
                Document(
                    doc_id="web:test-graphrag",
                    text="# GraphRAG\n\nGraphRAG builds an entity graph and retrieves by community detection and multi-hop traversal for aggregative questions.",
                    source="https://example.com/graphrag",
                    title="GraphRAG",
                    metadata={"origin": "web"},
                )
            ]
        )
        self.assertGreater(pipeline.dense.size, before)
        hits = pipeline.retrieve("graphrag entity graph community detection aggregative questions", top_k=5)
        self.assertTrue(any(u.chunk.doc_id == "web:test-graphrag" for u in hits.units))
        pipeline.close()
        Path("data/test_augment.db").unlink(missing_ok=True)

    def test_source_metadata_reaches_chunks_and_supports_filtering(self):
        pipeline = make_pipeline(db_path=Path("data/test_filter.db"))
        pipeline.augment(
            [
                Document(
                    doc_id="web:filter-target",
                    text="# Web source\n\nGraphRAG community detection for aggregative questions across documents.",
                    source="https://example.com/x",
                    title="Web source",
                    metadata={"origin": "web"},
                )
            ]
        )
        web_only = pipeline.retrieve("graphrag community detection aggregative", top_k=5, filters={"metadata": {"origin": "web"}})
        self.assertTrue(web_only.units)
        self.assertTrue(all(u.chunk.metadata.get("origin") == "web" for u in web_only.units))
        local_only = pipeline.retrieve("graphrag community detection aggregative", top_k=5, filters={"exclude_doc_ids": ["web:filter-target"]})
        self.assertTrue(all(u.chunk.doc_id != "web:filter-target" for u in local_only.units))
        pipeline.close()
        Path("data/test_filter.db").unlink(missing_ok=True)

    def test_doctor_reports_capabilities(self):
        report = self.pipeline.doctor()
        self.assertIn("embedder", report)
        self.assertIn("llm", report)
        self.assertIn("web_search", report)




class EmbeddingCacheTests(unittest.TestCase):
    """rag/embed/cache.py must actually short-circuit re-embedding across indexes."""

    CACHE = Path("data/test_embed_cache.db")

    def tearDown(self):
        self.CACHE.unlink(missing_ok=True)

    def _config(self, db_path: Path, embedder: str = "hashed"):
        db_path.unlink(missing_ok=True)
        config = RAGConfig.from_file("config.yaml")
        config.embedder = embedder
        config.embed_cache_path = str(self.CACHE)
        config.db_path = str(db_path)
        config.llm = "none"
        return config

    def test_second_index_reuses_cached_vectors(self):
        import rag.pipeline as pipeline_module
        from rag.embed.hashed import HashedNGramEmbedder

        first = RAGPipeline(self._config(Path("data/test_cache_first.db")))
        try:
            first.index_paths(["examples/corpus"])
            embedded = first.store.counts()["vectors"]
        finally:
            first.close()
        self.assertGreater(embedded, 0)

        class NoReembed(HashedNGramEmbedder):
            def embed(self, texts):
                raise AssertionError("every chunk should already be in the embedding cache")

        original = pipeline_module.build_embedder
        pipeline_module.build_embedder = lambda *args, **kwargs: NoReembed(dim=512)
        second = RAGPipeline(self._config(Path("data/test_cache_second.db")))
        try:
            second.index_paths(["examples/corpus"])
            counts = second.store.counts()
        finally:
            pipeline_module.build_embedder = original
            second.close()
        self.assertEqual(counts["vectors"], embedded)

    def test_lsa_vectors_are_not_cached_because_they_are_corpus_fitted(self):
        pipeline = RAGPipeline(self._config(Path("data/test_cache_lsa.db"), embedder="lsa"))
        try:
            pipeline.index_paths(["examples/corpus"])
            chunks = pipeline.store.counts()["chunks"]
            rows = pipeline.cache.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        finally:
            pipeline.close()
        self.assertGreater(chunks, 0)
        self.assertEqual(rows, 0)


if __name__ == "__main__":
    unittest.main()