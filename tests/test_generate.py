import unittest

from rag.chunking import chunk_document
from rag.generate import ExtractiveAnswerer, GenerationConfig, assemble_context, build_citations, citation_coverage, groundedness_score, render_context
from rag.text import split_sentences
from rag.types import Chunk, Document, RetrievedUnit

CHUNKS = [
    Chunk("a", "doc1", "Reciprocal rank fusion merges rankings and needs no score calibration."),
    Chunk("b", "doc1", "RRF is robust to channels with incomparable score ranges."),
    Chunk("c", "doc2", "Cooking a stew requires slow heat and patience."),
]


def units():
    return [RetrievedUnit(chunk=c, score=s, rank=i) for i, (c, s) in enumerate(zip(CHUNKS, (0.9, 0.8, 0.1)), start=1)]


class AssemblyTests(unittest.TestCase):
    def test_dedupe_removes_duplicate_text(self):
        duplicate = RetrievedUnit(chunk=Chunk("dup", "doc1", CHUNKS[0].text), score=0.95)
        selected = assemble_context(units() + [duplicate], GenerationConfig())
        texts = [u.best_text for u in selected]
        self.assertEqual(len(texts), len(set(texts)))

    def test_budget_trim_drops_lowest_scoring(self):
        cfg = GenerationConfig(max_context_tokens=12)
        selected = assemble_context(units(), cfg)
        self.assertTrue(all(u.chunk.chunk_id != "c" for u in selected))

    def test_u_shape_places_best_at_both_ends(self):
        many = [RetrievedUnit(chunk=Chunk(f"k{i}", "d", f"content number {i} " * 6), score=1.0 - i * 0.01) for i in range(6)]
        selected = assemble_context(many, GenerationConfig(order="u-shape", max_context_tokens=10_000))
        ids = [u.chunk.chunk_id for u in selected]
        self.assertEqual(ids[0], "k0")
        self.assertEqual(ids[-1], "k1")
        self.assertGreater(ids.index("k4"), ids.index("k0"))
        self.assertGreater(ids.index("k5"), ids.index("k0"))

    def test_render_numbers_sources(self):
        rendered = render_context(units())
        self.assertIn("[1] source=doc1", rendered)
        self.assertIn("[3] source=doc2", rendered)

    def test_citations_align_with_rendered_indices(self):
        citations = build_citations(units())
        self.assertEqual([c.index for c in citations], [1, 2, 3])


class GroundednessTests(unittest.TestCase):
    def test_supported_answer_scores_high(self):
        answer = "Reciprocal rank fusion merges rankings. It needs no score calibration."
        self.assertGreater(groundedness_score(answer, units()), 0.7)

    def test_hallucinated_claim_scores_low(self):
        answer = "Quantum tunneling accelerates nuclear fusion in stellar cores."
        self.assertLess(groundedness_score(answer, units()), 0.34)

    def test_citation_coverage_counts_sources(self):
        self.assertAlmostEqual(citation_coverage("claim one [1] claim two [3]", 3), 2 / 3)


class ExtractiveTests(unittest.TestCase):
    def test_extractive_answer_cites_sources(self):
        answer = ExtractiveAnswerer(sentences=2).answer("what is reciprocal rank fusion", units())
        self.assertTrue(answer.text)
        self.assertRegex(answer.text, r"\[\d\]")
        self.assertGreater(answer.groundedness, 0.5)

    def test_no_evidence_marks_unanswerable(self):
        answer = ExtractiveAnswerer().answer("quantum", [RetrievedUnit(chunk=Chunk("x", "d", "hi"), score=0.1)])
        self.assertFalse(answer.answerable)


if __name__ == "__main__":
    unittest.main()