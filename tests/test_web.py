import unittest

from rag.retrieve.web import WebHit, WebSearchRetriever, hashlib_digest

DDG_FIXTURE = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Frag&rut=abc">Hybrid RAG explained</a>
  <a class="result__snippet">A <b>hybrid</b> retriever fuses BM25 with dense vectors.</a>
</div>
<div class="result">
  <a class="result__a" href="https://other.example/late-interaction">Late interaction reranking</a>
  <a class="result__snippet">ColBERT computes maxsim over token embeddings.</a>
</div>
"""

BING_FIXTURE = """
<ol id="b_results">
<li class="b_algo" data-id iid=SERP.1><h2><a href="https://microsoft.github.io/graphrag/">Welcome - GraphRAG</a></h2>
<div class="b_caption"><p>GraphRAG is a structured, hierarchical approach to retrieval.</p></div></li>
<li class="b_algo" data-id iid=SERP.2><h2><a href="https://github.com/microsoft/graphrag">GitHub - microsoft/graphrag</a></h2>
<div class="b_caption"><p>A modular graph-based retrieval system.</p></div></li>
</ol>
"""

ARXIV_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2609.38473v1</id>
    <title>Re-ranking and Late Interaction Drive Retrieval Quality</title>
    <published>2026-09-29T00:00:00Z</published>
    <summary>A controlled comparison of RAG strategies over 463,971 arXiv papers.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2610.01767v1</id>
    <title>A Matryoshka Hierarchical RAG for Efficient Multi-Hop QA</title>
    <published>2026-10-01T00:00:00Z</published>
    <summary>Hierarchical retrieval for multi-hop question answering.</summary>
  </entry>
</feed>
"""

PAGE_FIXTURE = """
<html><head><title>Agentic RAG &amp; CRAG</title></head>
<body><script>var x = 1;</script><style>.a{}</style>
<nav>登录 注册 邀请码 隐私政策</nav>
<article><h1>Agentic RAG</h1>
<p>Corrective RAG grades each retrieved passage as relevant, ambiguous or irrelevant.</p>
<p>When quality falls below a threshold the system rewrites the query or falls back to web search.</p>
</article>
<footer>上一篇 下一篇 全部评论 举报</footer>
</body></html>
"""


class ParsingTests(unittest.TestCase):
    def test_duckduckgo_titles_and_urls(self):
        hits = WebSearchRetriever.parse_duckduckgo(DDG_FIXTURE, 5)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].url, "https://example.com/rag")
        self.assertEqual(hits[0].title, "Hybrid RAG explained")

    def test_duckduckgo_snippets(self):
        hits = WebSearchRetriever.parse_duckduckgo(DDG_FIXTURE, 5)
        self.assertIn("BM25", hits[0].snippet)
        self.assertIn("maxsim", hits[1].snippet)

    def test_bing_results(self):
        hits = WebSearchRetriever.parse_bing(BING_FIXTURE, 5)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].url, "https://microsoft.github.io/graphrag/")
        self.assertIn("GraphRAG", hits[0].title)
        self.assertIn("hierarchical", hits[0].snippet)

    def test_bing_limit(self):
        self.assertEqual(len(WebSearchRetriever.parse_bing(BING_FIXTURE, 1)), 1)

    def test_arxiv_entries(self):
        hits = WebSearchRetriever.parse_arxiv(ARXIV_FIXTURE, 5)
        self.assertEqual(len(hits), 2)
        self.assertIn("Late Interaction", hits[0].title)
        self.assertIn("463,971", hits[0].snippet)

    def test_arxiv_urls_use_https(self):
        hits = WebSearchRetriever.parse_arxiv(ARXIV_FIXTURE, 5)
        self.assertTrue(all(h.url.startswith("https://arxiv.org/") for h in hits))

    def test_arxiv_malformed_xml_returns_empty(self):
        self.assertEqual(WebSearchRetriever.parse_arxiv("<feed><broken", 5), [])

    def test_arxiv_query_passthrough_and_phrase_wrapping(self):
        self.assertEqual(
            WebSearchRetriever._arxiv_query('abs:"graph rag" AND cat:cs.IR'),
            'abs:"graph rag" AND cat:cs.IR',
        )
        self.assertEqual(
            WebSearchRetriever._arxiv_query("graphrag"),
            'ti:"graphrag" OR abs:"graphrag"',
        )

    def test_limit_is_respected(self):
        self.assertEqual(len(WebSearchRetriever.parse_duckduckgo(DDG_FIXTURE, 1)), 1)

    def test_html_to_text_strips_scripts_and_tags(self):
        text = WebSearchRetriever.html_to_text(PAGE_FIXTURE)
        self.assertIn("Corrective RAG grades each retrieved passage", text)
        self.assertNotIn("var x = 1", text)
        self.assertNotIn(".a{}", text)

    def test_boilerplate_lines_are_dropped(self):
        text = WebSearchRetriever.html_to_text(PAGE_FIXTURE)
        self.assertNotIn("邀请码", text)
        self.assertNotIn("隐私政策", text)
        self.assertNotIn("全部评论", text)

    def test_extract_main_region_prefers_article(self):
        retriever = WebSearchRetriever()
        region = retriever.extract_main_region(PAGE_FIXTURE)
        self.assertIn("Corrective RAG", region)
        self.assertNotIn("footer", region)

    def test_extract_main_region_falls_back_to_full_document(self):
        retriever = WebSearchRetriever()
        self.assertIn("no article tag here", retriever.extract_main_region("<div>no article tag here</div>"))

    def test_digest_is_stable(self):
        self.assertEqual(hashlib_digest("https://a"), hashlib_digest("https://a"))
        self.assertNotEqual(hashlib_digest("https://a"), hashlib_digest("https://b"))


class BackendSelectionTests(unittest.TestCase):
    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            WebSearchRetriever(backend="yandex").search("test")

    def test_tavily_without_key_raises_clearly(self):
        with self.assertRaises(RuntimeError):
            WebSearchRetriever(backend="tavily").search("test")

    def test_available_returns_bool(self):
        self.assertIsInstance(WebSearchRetriever(timeout=2.0).available(), bool)

    def test_fetch_document_returns_none_on_failure(self):
        self.assertIsNone(WebSearchRetriever(timeout=2.0).fetch_document("http://127.0.0.1:9/nope"))

    def test_search_documents_falls_back_to_snippet_document(self):
        retriever = WebSearchRetriever(timeout=2.0, fetch_pages=False)
        retriever.search = lambda query, max_results=None: WebSearchRetriever.parse_bing(BING_FIXTURE, max_results or 5)
        documents = retriever.search_documents("graphrag")
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["origin"], "web")
        self.assertTrue(documents[0].text)




ARXIV_ABS_FIXTURE = """
<html><head><title>A Matryoshka Hierarchical RAG for Efficient Multi-Hop QA</title></head>
<body>
<div class="primary"><h1>Computer Science &gt; Computation and Language</h1>
<span class="identifier">arXiv:2610.01767v1 (cs)</span>
<div class="dateline">[Submitted on 1 Oct 2026]</div>
<h1 class="title">A Matryoshka Hierarchical RAG for Efficient Multi-Hop QA</h1>
<blockquote class="abstract"><span class="descriptor">Abstract:</span>Hierarchical retrieval nests child chunks inside parent chunks so a multi-hop question can escalate only when the small blocks are insufficient.</blockquote>
<a href="/abs/2610.01767v2">v2</a> <a href="/list/cs/latest">list</a></div>
</body></html>
"""


class ArxivAndSnippetTests(unittest.TestCase):
    def test_arxiv_abstract_is_extracted_without_navigation(self):
        region = WebSearchRetriever.extract_arxiv_abstract(ARXIV_ABS_FIXTURE)
        text = WebSearchRetriever.html_to_text(region)
        self.assertIn("nests child chunks inside parent chunks", text)
        for noise in ("Submitted on", "arXiv:2610.01767v1", "Computation and Language"):
            self.assertNotIn(noise, text)

    def test_fetch_document_uses_abstract_region_for_arxiv(self):
        retriever = WebSearchRetriever(timeout=2.0)
        retriever._request = lambda url, headers=None: ARXIV_ABS_FIXTURE
        document = retriever.fetch_document("https://arxiv.org/abs/2610.01767v1")
        self.assertIsNotNone(document)
        assert document is not None
        self.assertIn("multi-hop question can escalate", document.text)
        self.assertNotIn("Submitted on 1 Oct 2026", document.text)

    def test_prefer_snippet_skips_page_fetch_when_snippet_is_substantial(self):
        retriever = WebSearchRetriever(backend="arxiv", prefer_snippet=True, fetch_pages=True)

        def refuse(url, headers=None):
            raise AssertionError("must not fetch when a substantial snippet is available")

        retriever._request = refuse
        retriever.search = lambda query, max_results=None: [
            WebHit("https://arxiv.org/abs/1", "T", "x" * 300),
            WebHit("https://arxiv.org/abs/2", "T2", "too short"),
        ]
        documents = retriever.search_documents("rag")
        self.assertEqual(len(documents), 2)
        self.assertIn("x" * 300, documents[0].text)

    def test_short_snippet_still_fetches_the_page(self):
        retriever = WebSearchRetriever(backend="arxiv", prefer_snippet=True, fetch_pages=True)
        retriever._request = lambda url, headers=None: ARXIV_ABS_FIXTURE
        retriever.search = lambda query, max_results=None: [
            WebHit("https://arxiv.org/abs/1", "T", "too short")
        ]
        documents = retriever.search_documents("rag")
        self.assertIn("nests child chunks", documents[0].text)


if __name__ == "__main__":
    unittest.main()