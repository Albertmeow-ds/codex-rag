"""Config layering and embedder-aware fusion weight profiles."""

import shutil
import tempfile
import pathlib
import unittest
from pathlib import Path

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline
from rag.retrieve.hybrid import DEFAULT_WEIGHTS, WEIGHT_PROFILES

CHILD = Path("data/test_child_config.yaml")


def write_child(text: str) -> None:
    CHILD.parent.mkdir(parents=True, exist_ok=True)
    CHILD.write_text(text, encoding="utf-8")


class ExtendsChainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ragcfg-")
        self.root = pathlib.Path(self.tmp)
        (self.root / "base.yaml").write_text(
            "reranker: heuristic\nchunking:\n  strategy: hierarchical\n  parent_tokens: 1200\n",
            "utf-8",
        )
        (self.root / "mid.yaml").write_text("extends: base.yaml\ndb_path: mid.db\nretrieval:\n  top_k: 5\n", "utf-8")
        (self.root / "leaf.yaml").write_text("extends: mid.yaml\nagentic:\n  evidence_gate: true\n", "utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_grandparent_keys_survive_a_two_hop_chain(self):
        config = RAGConfig.from_file(self.root / "leaf.yaml")
        self.assertEqual(config.reranker, "heuristic")
        self.assertEqual(config.chunking.parent_tokens, 1200)
        self.assertEqual(config.db_path, "mid.db")
        self.assertEqual(config.retrieval.top_k, 5)
        self.assertTrue(config.agentic_config.evidence_gate)

    def test_child_still_wins_over_ancestor(self):
        (self.root / "leaf2.yaml").write_text("extends: mid.yaml\nretrieval:\n  top_k: 9\n", "utf-8")
        self.assertEqual(RAGConfig.from_file(self.root / "leaf2.yaml").retrieval.top_k, 9)

    def test_circular_chain_is_reported_not_hung(self):
        (self.root / "a.yaml").write_text("extends: b.yaml\n", "utf-8")
        (self.root / "b.yaml").write_text("extends: a.yaml\n", "utf-8")
        with self.assertRaises(ValueError):
            RAGConfig.from_file(self.root / "a.yaml")


class ExtendsTests(unittest.TestCase):
    def tearDown(self):
        CHILD.unlink(missing_ok=True)

    def test_child_override_keeps_sibling_keys_of_the_same_section(self):
        write_child("extends: ../config.yaml\nretrieval:\n  weights: { bm25: 0.5, dense: 1.5 }\n")
        config = RAGConfig.from_file(CHILD)
        self.assertEqual(config.retrieval.weights, {"bm25": 0.5, "dense": 1.5})
        # Losing these would silently switch the pipeline back to rrf / no parent expansion.
        self.assertEqual(config.retrieval.fusion, "weighted")
        self.assertTrue(config.retrieval.expand_parents)
        self.assertEqual(config.chunking.strategy, "hierarchical")

    def test_deep_merge_is_recursive_not_just_top_level(self):
        merged = RAGConfig.deep_merge(
            {"retrieval": {"fusion": "weighted", "top_k": 8}},
            {"retrieval": {"top_k": 12}},
        )
        self.assertEqual(merged["retrieval"], {"fusion": "weighted", "top_k": 12})


class WeightProfileTests(unittest.TestCase):
    def tearDown(self):
        Path("data/test_profile.db").unlink(missing_ok=True)

    def _pipeline(self, embedder_name: str):
        db = Path("data/test_profile.db")
        db.unlink(missing_ok=True)
        config = RAGConfig.from_file("config.yaml")
        config.embedder = embedder_name
        config.llm = "none"
        config.db_path = str(db)
        return RAGPipeline(config)

    def test_unset_weights_follow_the_embedder_profile(self):
        lexical = self._pipeline("hashed")
        try:
            lexical_weights = lexical.config.retrieval.resolved_weights()
        finally:
            lexical.close()
        semantic = self._pipeline("ollama:bge-m3")
        try:
            semantic_weights = semantic.config.retrieval.resolved_weights()
        finally:
            semantic.close()
        self.assertEqual(lexical_weights, WEIGHT_PROFILES["lexical"])
        self.assertEqual(semantic_weights, WEIGHT_PROFILES["semantic"])

    def test_explicit_weights_are_never_overwritten(self):
        pipeline = self._pipeline("ollama:bge-m3")
        try:
            pipeline.config.retrieval.weights = {"bm25": 2.0, "dense": 0.1, "late": 0.0}
            self.assertEqual(
                pipeline.config.retrieval.resolved_weights(),
                {"bm25": 2.0, "dense": 0.1, "late": 0.0},
            )
        finally:
            pipeline.close()

    def test_resolver_falls_back_to_neutral_weights(self):
        config = RAGConfig()
        config.retrieval.weights = None
        self.assertEqual(config.retrieval.resolved_weights(), DEFAULT_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
