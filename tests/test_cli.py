"""The CLI must not inject the bundled corpus into a purpose-built knowledge base."""

import unittest
from pathlib import Path

from rag.cli import _seed_default_corpus
from rag.config import RAGConfig
from rag.pipeline import RAGPipeline
from rag.types import Document

NOTE = Document(
    doc_id="notes/deploy.md",
    text="# 部署手册\n\n灰度先 5% 流量观察 30 分钟，错误率低于 0.3% 才放量。",
    source="notes/deploy.md",
    title="部署手册",
)


class DefaultCorpusSeedingTests(unittest.TestCase):
    def _pipeline(self, db_name: str) -> RAGPipeline:
        db = Path("data") / db_name
        db.unlink(missing_ok=True)
        config = RAGConfig.from_file("config.yaml")
        config.db_path = str(db)
        config.llm = "none"
        return RAGPipeline(config)

    def test_empty_store_is_seeded_so_the_cli_works_out_of_the_box(self):
        pipeline = self._pipeline("test_seed_empty.db")
        try:
            _seed_default_corpus(pipeline)
            self.assertGreater(pipeline.store.counts()["documents"], 1)
        finally:
            pipeline.close()

    def test_existing_knowledge_base_is_left_alone(self):
        pipeline = self._pipeline("test_seed_keep.db")
        try:
            pipeline.index_documents([NOTE])
            before = pipeline.store.counts()["documents"]
            _seed_default_corpus(pipeline)
            self.assertEqual(pipeline.store.counts()["documents"], before)
            hits = pipeline.retrieve("灰度发布要观察多久", top_k=3)
            self.assertTrue(hits.units)
            self.assertTrue(all(unit.chunk.doc_id.startswith("notes/") for unit in hits.units))
        finally:
            pipeline.close()


if __name__ == "__main__":
    unittest.main()
