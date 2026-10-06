import unittest

from rag.agentic import AgenticConfig, AgenticRAG, RetrievalGrader, coverage_ratio
from rag.chunking import chunk_document
from rag.embed.hashed import HashedNGramEmbedder
from rag.generate import ExtractiveAnswerer, GenerationConfig
from rag.index.bm25 import BM25Index
from rag.index.dense import DenseIndex
from rag.index.late import LateInteractionIndex
from rag.llm import NullLLM
from rag.retrieve.hybrid import HybridRetriever, RetrievalConfig
from rag.rerank import HeuristicReranker
from rag.text import content_tokens
from rag.types import Document, Grade

DOCS = [
    Document("on", "# Rerank\n\nLate interaction reranking compares token level embeddings with maxsim after retrieval.", "Rerank"),
    Document("off", "# Cooking\n\nCooking a stew requires slow heat and patience over several hours.", "Cooking"),
]


def build_retriever() -> HybridRetriever:
    chunks = [c for d in DOCS for c in chunk_document(d)]
    embedder = HashedNGramEmbedder(256)
    bm25 = BM25Index().build([(c.chunk_id, content_tokens(c.text)) for c in chunks])
    dense = DenseIndex().build([c.chunk_id for c in chunks], embedder.embed([c.text for c in chunks]))
    late = LateInteractionIndex().build([(c.chunk_id, embedder.embed_tokens(c.text)) for c in chunks])
    return HybridRetriever(
        bm25, dense, late, chunks, embedder,
        RetrievalConfig(top_k=4, candidate_k=12), reranker=HeuristicReranker(),
    )


class GraderTests(unittest.TestCase):
    def test_lexical_grader_labels_relevant(self):
        retriever = build_retriever()
        units = retriever.retrieve("maxsim token level reranking").units
        grade = RetrievalGrader().grade("maxsim token level reranking", units[0])
        self.assertIn(grade.label, {"relevant", "ambiguous"})
        self.assertGreater(grade.score, 0.0)

    def test_batch_grading_covers_every_candidate(self):
        retriever = build_retriever()
        units = retriever.retrieve("maxsim token level reranking").units
        grades = RetrievalGrader().grade_batch("maxsim token level reranking", units)
        self.assertEqual(len(grades), len(units))
        self.assertTrue(all(isinstance(g, Grade) for g in grades))

    def test_coverage_ratio(self):
        retriever = build_retriever()
        units = retriever.retrieve("maxsim token level reranking").units
        self.assertGreater(coverage_ratio("maxsim token reranking", units), 0.5)
        self.assertEqual(coverage_ratio("", units), 0.0)


class LoopTests(unittest.TestCase):
    def test_loop_produces_cited_answer_and_trace(self):
        retriever = build_retriever()
        agent = AgenticRAG(
            retriever,
            ExtractiveAnswerer(GenerationConfig(), sentences=3),
            config=AgenticConfig(max_rounds=2),
            grader=RetrievalGrader(NullLLM()),
        )
        answer = agent.run("how does late interaction reranking work")
        self.assertTrue(answer.text)
        self.assertRegex(answer.text, r"\[\d+\]")
        stages = [entry["stage"] for entry in answer.trace.stages]
        self.assertIn("retrieve", stages)
        self.assertIn("grade", stages)
        self.assertIn("final", stages)

    def test_weak_query_triggers_rewrite_round(self):
        retriever = build_retriever()
        agent = AgenticRAG(
            retriever,
            ExtractiveAnswerer(GenerationConfig(), sentences=3),
            config=AgenticConfig(max_rounds=2, weak_coverage_threshold=0.99),
            grader=RetrievalGrader(NullLLM()),
        )
        answer = agent.run("quantum chromogenomics")
        rounds = [e for e in answer.trace.stages if e["stage"] == "grade"]
        self.assertGreaterEqual(len(rounds), 2)
        self.assertTrue(any(e["stage"] == "rewrite" for e in answer.trace.stages))

    def test_graded_candidate_cap_is_respected(self):
        retriever = build_retriever()
        agent = AgenticRAG(
            retriever,
            ExtractiveAnswerer(GenerationConfig(), sentences=2),
            config=AgenticConfig(max_rounds=1, max_graded_candidates=1),
            grader=RetrievalGrader(NullLLM()),
        )
        answer = agent.run("maxsim token level reranking")
        graded = [e for e in answer.trace.stages if e["stage"] == "grade"]
        self.assertTrue(graded)
        self.assertLessEqual(graded[0]["candidates"], 1)


if __name__ == "__main__":
    unittest.main()