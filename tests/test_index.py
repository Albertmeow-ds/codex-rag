import unittest

import numpy as np

from rag.embed.hashed import HashedNGramEmbedder
from rag.index.bm25 import BM25Index
from rag.index.dense import DenseIndex
from rag.index.late import LateInteractionIndex
from rag.text import content_tokens

DOCS = [
    ("hybrid", "hybrid retrieval fuses bm25 lexical scores with dense vector scores"),
    ("chunk", "semantic chunking chooses breakpoints by embedding similarity"),
    ("eval", "ndcg and mrr measure the ranking quality of a retrieval system"),
]


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index().build([(i, content_tokens(t)) for i, t in DOCS])

    def test_relevant_doc_ranks_first(self):
        top_score, top_id = self.index.search(content_tokens("bm25 dense vector scores"), 1)[0]
        self.assertEqual(top_id, "hybrid")
        self.assertGreater(top_score, 0)

    def test_no_match_returns_empty(self):
        self.assertEqual(self.index.search(content_tokens("quantum chromodynamics"), 5), [])

    def test_term_frequency_increases_score(self):
        repeated = BM25Index().build([("a", content_tokens("retrieval retrieval retrieval")), ("b", content_tokens("retrieval"))])
        scores = {doc_id: score for score, doc_id in repeated.search(content_tokens("retrieval"), 2)}
        self.assertGreater(scores["a"], scores["b"])

    def test_explain_reports_terms(self):
        explained = self.index.explain(content_tokens("bm25 lexical"), "hybrid")
        self.assertIn("bm25", explained)


class DenseTests(unittest.TestCase):
    def test_semantic_match_beats_unrelated(self):
        embedder = HashedNGramEmbedder(256)
        index = DenseIndex().build([i for i, _ in DOCS], embedder.embed([t for _, t in DOCS]))
        hits = index.search(embedder.embed(["embedding similarity chunking"])[0], 1)
        self.assertEqual(hits[0][1], "chunk")

    def test_multi_query_takes_best_variant(self):
        embedder = HashedNGramEmbedder(256)
        index = DenseIndex().build([i for i, _ in DOCS], embedder.embed([t for _, t in DOCS]))
        variants = embedder.embed(["totally unrelated topic", "ndcg mrr ranking quality"])
        score, chunk_id = index.search_multi(variants, 1)[0]
        self.assertEqual(chunk_id, "eval")
        self.assertGreater(score, 0.1)


class LateInteractionTests(unittest.TestCase):
    def test_maxsim_is_discriminative_not_saturated(self):
        embedder = HashedNGramEmbedder(256)
        index = LateInteractionIndex().build([(i, embedder.embed_tokens(t)) for i, t in DOCS])
        scores = index.score_many(embedder.embed_tokens("bm25 dense vector scores"), [i for i, _ in DOCS])
        self.assertAlmostEqual(scores["hybrid"], 1.0, places=3)
        self.assertLess(scores["eval"], 0.5)
        self.assertLess(scores["chunk"], 0.5)

    def test_empty_query_scores_zero(self):
        embedder = HashedNGramEmbedder(64)
        index = LateInteractionIndex().build([("a", embedder.embed_tokens("alpha beta"))])
        self.assertEqual(index.score_many(np.zeros((0, 64), dtype=np.float32), ["a"]), {})


if __name__ == "__main__":
    unittest.main()