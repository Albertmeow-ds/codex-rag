import unittest

from rag.chunking import ChunkingConfig, chunk_document, hierarchical_chunks, parse_sections, sentence_window, semantic_chunks
from rag.types import Document

DOC = Document(
    "doc",
    "# Title\n\nIntro sentence one. Intro sentence two.\n\n## Section A\n\nAlpha one. Alpha two. Alpha three.\n\n## Section B\n\nBeta one. Beta two.\n",
    "Title",
)


class SectionTests(unittest.TestCase):
    def test_heading_hierarchy(self):
        sections = parse_sections(DOC.text)
        self.assertEqual(sections[1].heading_path, ("Title", "Section A"))

    def test_no_headings_yields_single_section(self):
        self.assertEqual(len(parse_sections("plain text here")), 1)


class ChunkTests(unittest.TestCase):
    def test_structural_chunks_carry_heading_path(self):
        chunks = chunk_document(DOC, ChunkingConfig(strategy="structural", max_tokens=30))
        self.assertTrue(any("Section A" in c.heading_path for c in chunks))
        self.assertTrue(all(c.text.startswith("Title") for c in chunks))

    def test_hierarchical_children_reference_parents(self):
        chunks = hierarchical_chunks(DOC, ChunkingConfig(strategy="hierarchical", parent_tokens=60, child_tokens=20))
        children = [c for c in chunks if c.parent_id]
        parents = [c for c in chunks if not c.parent_id]
        self.assertTrue(children)
        self.assertTrue(parents)
        parent_ids = {p.chunk_id for p in parents}
        self.assertTrue(all(c.parent_id in parent_ids for c in children))

    def test_semantic_chunking_produces_chunks(self):
        chunks = semantic_chunks(DOC, ChunkingConfig(strategy="semantic", max_tokens=25))
        self.assertTrue(chunks)
        self.assertTrue(all(c.text for c in chunks))

    def test_sentence_window_expands_neighbours(self):
        chunks = chunk_document(DOC, ChunkingConfig(strategy="structural", max_tokens=12))
        target = chunks[1]
        window = sentence_window(chunks, target, radius=1)
        self.assertIn(target.text, window)
        self.assertGreater(len(window), len(target.text))

    def test_overlap_repeats_tail_content(self):
        cfg = ChunkingConfig(strategy="structural", max_tokens=12, overlap_tokens=6)
        chunks = chunk_document(DOC, cfg)
        self.assertTrue(len(chunks) > 1)


if __name__ == "__main__":
    unittest.main()