"""Corpus corruption is deterministic, and the curve degrades monotonically-ish."""

import unittest

from rag.eval import EvalCase
from rag.generate import ExtractiveAnswerer, assemble_context, delivery_report
from rag.generate import GenerationConfig
from rag.stress import StressConfig, corrupt_documents, degradation_table, stress_curve
from rag.types import Chunk, RetrievedUnit

CORPUS_TEXT = (
    "# 检索调参记录\n\n"
    "BM25 与稠密通道的融合权重默认是 bm25 1.2 dense 0.8。\n\n"
    "嵌入缓存的键由嵌入名、维度和文本哈希组成，命中后不再重复编码。\n\n"
    "评测集 recall@5 基线为 0.8714，改动后必须重新跑一遍对比。\n"
)


class CorruptTest(unittest.TestCase):
    def setUp(self):
        from rag.types import Document
        self.docs = [Document("a/one.md", CORPUS_TEXT, source="a/one.md", title="one")]

    def test_level_zero_is_a_faithful_clone(self):
        out = corrupt_documents(self.docs, operators=(), intensity=0)
        self.assertEqual(out[0].text, CORPUS_TEXT)
        self.assertEqual(out[0].doc_id, "a/one.md")

    def test_same_seed_is_reproducible(self):
        first = corrupt_documents(self.docs, ("typo", "shuffle", "duplicate"), 2)
        second = corrupt_documents(self.docs, ("typo", "shuffle", "duplicate"), 2)
        self.assertEqual(first[0].text, second[0].text)

    def test_doc_id_survives_so_gold_ids_stay_addressable(self):
        out = corrupt_documents(self.docs, ("typo", "truncate"), 1)
        self.assertEqual(out[0].doc_id, self.docs[0].doc_id)

    def test_conflict_documents_contradict_a_number(self):
        out = corrupt_documents(self.docs, ("conflict",), 1, conflict_docs=1)
        self.assertEqual(len(out), 2)
        self.assertIn("更正说明", out[1].text)

    def test_noise_documents_are_extra_and_separate(self):
        out = corrupt_documents(self.docs, ("noise",), 1, noise_docs=3)
        self.assertEqual(len(out), 4)
        self.assertEqual(len(out[0].text), len(CORPUS_TEXT))

    def test_more_corruption_adds_more_documents(self):
        low = corrupt_documents(self.docs, ("noise", "conflict"), 1, noise_docs=2, conflict_docs=1)
        high = corrupt_documents(self.docs, ("noise", "conflict"), 3, noise_docs=6, conflict_docs=3)
        self.assertGreater(len(high), len(low))

    def test_curve_runs_one_row_per_level(self):
        cases = [EvalCase("融合权重是多少", gold_doc_ids=["a/one.md"])]
        rows = stress_curve(self.docs, cases, lambda corpus: {"recall@3": 0.5}, StressConfig(levels=(0, 1, 2)))
        self.assertEqual([row["level"] for row in rows], [0, 1, 2])

    def test_degradation_table_shows_delta(self):
        rows = [{"level": 0, "recall@5": 0.9}, {"level": 1, "recall@5": 0.7}]
        table = degradation_table(rows, "recall@5")
        self.assertIn("-0.2000", table)


class DeliveryTest(unittest.TestCase):
    def test_budget_drops_are_counted_not_silent(self):
        units = [
            RetrievedUnit(chunk=Chunk(f"c{i}", "doc", "x" * 400), score=1.0 - i * 0.01)
            for i in range(6)
        ]
        selected = assemble_context(units, GenerationConfig(max_context_tokens=500))
        report = delivery_report(units, selected)
        self.assertEqual(report["retrieved"], 6)
        self.assertLess(report["delivered"], 6)
        self.assertEqual(report["dropped"], 6 - report["delivered"])
        self.assertGreater(report["dropped_tokens"], 0)

    def test_extractive_answer_carries_the_report(self):
        units = [RetrievedUnit(chunk=Chunk("c1", "doc", "融合权重默认是 bm25 1.2。"), score=1.0)]
        answer = ExtractiveAnswerer().answer("融合权重是多少", units)
        self.assertEqual(answer.delivery["delivered"], 1)


if __name__ == "__main__":
    unittest.main()