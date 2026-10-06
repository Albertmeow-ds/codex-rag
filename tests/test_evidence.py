"""Evidence sufficiency is a different question from topical relevance."""

import unittest

from rag.evidence import EvidenceConfig, EvidenceGapChecker, analyze_query, gap_terms, split_cjk_run
from rag.types import Chunk, RetrievedUnit


def unit(chunk_id: str, text: str) -> RetrievedUnit:
    return RetrievedUnit(chunk=Chunk(chunk_id, "doc", text), score=1.0)


class AnalyzeQueryTest(unittest.TestCase):
    def test_interrogative_decides_answer_shape(self):
        self.assertEqual(analyze_query("为什么检索会失败").answer_type, "cause")
        self.assertEqual(analyze_query("配置里 max_tokens 是多少").answer_type, "count")
        self.assertEqual(analyze_query("什么时候上线的").answer_type, "date")
        self.assertEqual(analyze_query("怎么开启缓存").answer_type, "procedure")

    def test_technical_tokens_become_entities(self):
        spec = analyze_query("bge-m3 为什么版本冲突会导致检索失败")
        self.assertIn("bge-m3", spec.entities)
        self.assertNotIn("bge-m3 为什么版本冲突会导致检索失败", spec.entities)

    def test_cjk_run_split_on_cue_seams(self):
        self.assertEqual(split_cjk_run("为什么版本冲突会导致检索失败"), ["版本冲突", "检索失败"])


class EvidenceGapTest(unittest.TestCase):
    def test_relevant_but_insufficient_is_flagged(self):
        checker = EvidenceGapChecker()
        report = checker.check(
            "bge-m3 的向量维度是多少",
            [unit("c1", "bge-m3 是一个常用的嵌入模型，效果很好。")],
        )
        self.assertFalse(report.answerable)
        self.assertEqual(report.missing_slots, ("count",))

    def test_fact_bearing_unit_passes(self):
        checker = EvidenceGapChecker()
        report = checker.check(
            "bge-m3 的向量维度是多少",
            [unit("c1", "bge-m3 输出的向量维度是 1024，可直接用于稠密检索。")],
        )
        self.assertTrue(report.answerable)
        self.assertGreaterEqual(report.supporting_units, 1)

    def test_missing_entities_feed_rewriting(self):
        checker = EvidenceGapChecker()
        query = "sentence-window 为什么会影响召回"
        report = checker.check(query, [unit("c1", "检索权重可以按嵌入模型自动切换。")])
        terms = gap_terms(query, [unit("c1", "检索权重可以按嵌入模型自动切换。")], report)
        self.assertIn("sentence-window", terms)

    def test_disabled_checker_never_blocks(self):
        checker = EvidenceGapChecker(EvidenceConfig(enabled=False))
        report = checker.check("完全无关的问题", [unit("c1", "text")])
        self.assertTrue(report.answerable)

    def test_empty_candidates_is_not_answerable(self):
        self.assertFalse(EvidenceGapChecker().check("为什么", []).answerable)


if __name__ == "__main__":
    unittest.main()