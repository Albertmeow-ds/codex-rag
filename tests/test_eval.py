import unittest

from rag.eval import EvalCase, hit_rate, load_dataset, mrr, ndcg_at_k, precision_at_k, recall_at_k


class MetricTests(unittest.TestCase):
    def test_recall_counts_coverage(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 3), 2 / 3)

    def test_recall_ignores_beyond_k(self):
        self.assertAlmostEqual(recall_at_k(["x", "y", "a"], {"a"}, 2), 0.0)

    def test_precision(self):
        self.assertAlmostEqual(precision_at_k(["a", "x", "y"], {"a"}, 3), 1 / 3)

    def test_mrr_uses_first_hit(self):
        self.assertAlmostEqual(mrr(["x", "y", "a", "b"], {"a", "b"}), 1 / 3)
        self.assertEqual(mrr(["x"], {"a"}), 0.0)

    def test_ndcg_perfect_ranking_is_one(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a", "b"}, 2), 1.0)

    def test_ndcg_penalizes_bad_order(self):
        self.assertLess(ndcg_at_k(["x", "a"], {"a"}, 2), 1.0)

    def test_hit_rate(self):
        self.assertEqual(hit_rate(["x", "a"], {"a"}, 2), 1.0)
        self.assertEqual(hit_rate(["x", "a"], {"a"}, 1), 0.0)

    def test_dataset_loader(self):
        cases = load_dataset("examples/eval.jsonl")
        self.assertTrue(cases)
        self.assertTrue(all(c.question and c.gold_doc_ids for c in cases))


if __name__ == "__main__":
    unittest.main()