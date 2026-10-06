import unittest

from rag.text import content_tokens, count_tokens, lexical_overlap, split_sentences, tokenize


class TokenizerTests(unittest.TestCase):
    def test_cjk_uses_bigrams(self):
        self.assertEqual(tokenize("检索增强"), ("检索", "索增", "增强"))

    def test_single_cjk_char_stays_unigram(self):
        self.assertEqual(tokenize("猫"), ("猫",))

    def test_mixed_script(self):
        tokens = tokenize("混合 retrieval 生成")
        self.assertIn("混合", tokens)
        self.assertIn("retrieval", tokens)
        self.assertIn("生成", tokens)

    def test_uni_bi_mode(self):
        tokens = tokenize("检索", cjk_mode="uni_bi")
        self.assertEqual(tokens, ("检", "索", "检索"))

    def test_latin_tokens_keep_digits_and_hyphens(self):
        self.assertEqual(tokenize("BM25+ cross-encoder 2026"), ("bm25", "cross-encoder", "2026"))

    def test_case_insensitive(self):
        self.assertEqual(tokenize("RAG"), ("rag",))

    def test_content_tokens_drop_stopwords(self):
        self.assertNotIn("the", content_tokens("the quick brown fox"))

    def test_count_tokens_mixes_scripts(self):
        self.assertEqual(count_tokens("检索 RAG"), 3)


class SentenceTests(unittest.TestCase):
    def test_chinese_and_english_boundaries(self):
        parts = split_sentences("第一句。第二句！Third one? yes")
        self.assertEqual(parts[0], "第一句。")
        self.assertEqual(parts[1], "第二句！")

    def test_abbreviation_not_split(self):
        parts = split_sentences("we use models e.g. bge-m3 for embedding")
        self.assertEqual(len(parts), 1)

    def test_blank_lines_end_a_sentence(self):
        self.assertEqual(len(split_sentences("one\n\ntwo")), 2)

    def test_punctuation_only_fragments_do_not_crash(self):
        for text in ["!!! ???", "???", "a. ? b", "!!!", "  ?  ?  "]:
            parts = split_sentences(text)
            self.assertIsInstance(parts, tuple)

    def test_web_like_mixed_punctuation(self):
        parts = split_sentences("Result: !!! See [1]. Done???")
        self.assertTrue(any("Result" in p for p in parts))

    def test_lexical_overlap_bounds(self):
        self.assertEqual(lexical_overlap(tokenize("vector database"), "a vector database store"), 1.0)
        self.assertLess(lexical_overlap(tokenize("quantum tunneling"), "a vector database store"), 0.1)


if __name__ == "__main__":
    unittest.main()