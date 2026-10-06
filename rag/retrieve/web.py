"""Live web retriever.

Backends are chosen by availability, not hardcoded: Tavily or Brave when an API key is
present, then Bing, then DuckDuckGo. Fetched pages become Documents so they flow through
the same chunking, indexing, reranking and citation path as local files.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Sequence
import xml.etree.ElementTree as ET

from rag.types import Document

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S)
_PROBE_URLS = (
    "https://www.bing.com/search?q=test",
    "https://duckduckgo.com/html/?q=test",
    "https://api.tavily.com/search",
)


@dataclass(slots=True)
class WebHit:
    url: str
    title: str
    snippet: str


def hashlib_digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class WebSearchRetriever:
    def __init__(
        self,
        timeout: float = 12.0,
        max_results: int = 6,
        fetch_pages: bool = True,
        backend: str = "auto",
        prefer_snippet: bool = False,
        min_snippet_chars: int = 160,
    ) -> None:
        self.timeout = timeout
        self.max_results = max_results
        self.fetch_pages = fetch_pages
        self.backend = backend
        self.prefer_snippet = prefer_snippet
        self.min_snippet_chars = min_snippet_chars
        self._reachable: list[str] = []

    def _request(self, url: str, headers: dict[str, str] | None = None) -> str:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
                **(headers or {}),
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8", "ignore")

    def available(self) -> bool:
        """Probe every backend instead of assuming one host is reachable."""

        if self._reachable:
            return True
        probes = list(_PROBE_URLS)
        if not os.environ.get("TAVILY_API_KEY"):
            probes = [u for u in probes if "tavily" not in u]
        # Search engines rate-limit aggressively; one failed probe is not evidence of no network.
        for attempt in range(2):
            for url in probes:
                try:
                    self._request(url)
                    self._reachable.append(url)
                except urllib.error.HTTPError as exc:
                    # A 401/403 still proves the route is reachable; close it to avoid leaking the body.
                    exc.close()
                    self._reachable.append(url)
                except Exception:
                    continue
            if self._reachable:
                break
            time.sleep(1.0)
        return bool(self._reachable)

    def reachable_backends(self) -> list[str]:
        self.available()
        return list(self._reachable)

    def search(self, query: str, max_results: int | None = None) -> list[WebHit]:
        limit = max_results or self.max_results
        backend = self.backend
        if backend == "auto":
            if os.environ.get("TAVILY_API_KEY"):
                return self._tavily(query, limit, os.environ["TAVILY_API_KEY"])
            if os.environ.get("BRAVE_API_KEY"):
                return self._brave(query, limit, os.environ["BRAVE_API_KEY"])
            errors: list[str] = []
            for name in ("bing", "duckduckgo"):
                try:
                    hits = getattr(self, f"_{name}")(query, limit)
                    if hits:
                        return hits
                    errors.append(f"{name}: no results")
                except Exception as exc:
                    errors.append(f"{name}: {type(exc).__name__}")
            raise RuntimeError("all web backends failed: " + "; ".join(errors))
        if backend in {"bing", "duckduckgo", "arxiv"}:
            return getattr(self, f"_{backend}")(query, limit)
        if backend == "tavily":
            return self._tavily(query, limit, os.environ.get("TAVILY_API_KEY", ""))
        if backend == "brave":
            return self._brave(query, limit, os.environ.get("BRAVE_API_KEY", ""))
        raise ValueError(f"unknown web backend: {backend}")

    def _tavily(self, query: str, limit: int, api_key: str) -> list[WebHit]:
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        body = json.dumps({"api_key": api_key, "query": query, "max_results": limit}).encode("utf-8")
        request = urllib.request.Request(
            "https://api.tavily.com/search", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8", "ignore"))
        return [
            WebHit(r.get("url", ""), r.get("title", ""), r.get("content", ""))
            for r in data.get("results", [])[:limit]
        ]

    def _brave(self, query: str, limit: int, api_key: str) -> list[WebHit]:
        if not api_key:
            raise RuntimeError("BRAVE_API_KEY is not set")
        url = "https://api.search.brave.com/res/v1/web/search?q=" + urllib.parse.quote(query)
        data = json.loads(self._request(url, {"X-Subscription-Token": api_key, "Accept": "application/json"}))
        return [
            WebHit(r.get("url", ""), html.unescape(r.get("title", "")), html.unescape(r.get("description", "")))
            for r in data.get("web", {}).get("results", [])[:limit]
        ]

    def _arxiv(self, query: str, limit: int) -> list[WebHit]:
        """arXiv Atom API: the most reliable way to pull the newest method papers."""

        url = (
            "http://export.arxiv.org/api/query?search_query="
            + urllib.parse.quote(self._arxiv_query(query))
            + f"&start=0&max_results={limit}&sortBy=submittedDate&sortOrder=descending"
        )
        return self.parse_arxiv(self._request(url), limit)

    @staticmethod
    def _arxiv_query(query: str) -> str:
        """Pass through explicit arXiv field queries; otherwise match title/abstract."""

        if ":" in query or " AND " in query.upper() or " OR " in query.upper():
            return query
        phrase = f'"{query}"'
        return f"ti:{phrase} OR abs:{phrase}"

    @staticmethod
    def parse_arxiv(raw: str, limit: int) -> list[WebHit]:
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return []
        hits: list[WebHit] = []
        for entry in root.findall("atom:entry", namespace):
            title = (entry.findtext("atom:title", "", namespace) or "").strip().replace("\n", " ")
            link = (entry.findtext("atom:id", "", namespace) or "").strip()
            if link.startswith("http://arxiv.org/"):
                link = "https://" + link[len("http://") :]
            summary = (entry.findtext("atom:summary", "", namespace) or "").strip().replace("\n", " ")
            published = (entry.findtext("atom:published", "", namespace) or "")[:10]
            if not link:
                continue
            hits.append(WebHit(link, f"[{published}] {title}", summary))
            if len(hits) >= limit:
                break
        return hits

    def _bing(self, query: str, limit: int) -> list[WebHit]:
        raw = self._request(
            "https://www.bing.com/search?q=" + urllib.parse.quote(query) + f"&count={limit}&setlang=en"
        )
        return self.parse_bing(raw, limit)

    @staticmethod
    def parse_bing(raw: str, limit: int) -> list[WebHit]:
        """Parse Bing result blocks. Split out so it is testable without network."""

        hits: list[WebHit] = []
        for block in re.finditer(r'<li class="b_algo"[^>]*>(.*?)</li>', raw, re.S):
            chunk = block.group(1)
            link = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', chunk, re.S)
            if not link:
                link = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', chunk, re.S)
            if not link:
                continue
            url = html.unescape(link.group(1))
            if not url.startswith("http"):
                continue
            title = html.unescape(_TAG_RE.sub("", link.group(2))).strip()
            snippet_match = re.search(r'<p[^>]*>(.*?)</p>', chunk, re.S) or re.search(
                r'<div class="b_caption"[^>]*>(.*?)</div>', chunk, re.S
            )
            snippet = html.unescape(_TAG_RE.sub("", snippet_match.group(1))).strip() if snippet_match else ""
            hits.append(WebHit(url, title, snippet))
            if len(hits) >= limit:
                break
        return hits

    def _duckduckgo(self, query: str, limit: int) -> list[WebHit]:
        raw = self._request("https://duckduckgo.com/html/?q=" + urllib.parse.quote(query))
        return self.parse_duckduckgo(raw, limit)

    @staticmethod
    def parse_duckduckgo(raw: str, limit: int) -> list[WebHit]:
        """Parse DuckDuckGo HTML results. Split out so it is testable without network."""

        hits: list[WebHit] = []
        for match in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', raw, re.S):
            href = html.unescape(match.group(1))
            if href.startswith("//duckduckgo.com/l/") or href.startswith("/l/"):
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = (parsed.get("uddg") or [href])[0]
            title = html.unescape(_TAG_RE.sub("", match.group(2))).strip()
            hits.append(WebHit(href, title, ""))
            if len(hits) >= limit:
                break
        snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', raw, re.S)
        for hit, snippet in zip(hits, snippets):
            hit.snippet = html.unescape(_TAG_RE.sub("", snippet)).strip()
        return hits

    def fetch_document(self, url: str) -> Document | None:
        try:
            raw = self._request(url)
        except Exception:
            return None
        region = self.extract_arxiv_abstract(raw) if "arxiv.org" in url else ""
        if not region:
            region = self.extract_main_region(raw)
        text = self.html_to_text(region)
        if not text:
            return None
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S)
        title = html.unescape(title_match.group(1)).strip() if title_match else url
        return Document(
            doc_id="web:" + hashlib_digest(url),
            text=text,
            source=url,
            title=title,
            metadata={"origin": "web"},
        )

    def search_documents(self, query: str, max_results: int | None = None) -> list[Document]:
        documents: list[Document] = []
        for hit in self.search(query, max_results):
            document = None
            # Search-engine snippets are already curated prose; fetched pages are not, so a
            # substantial snippet indexes better than the nav-heavy HTML around it.
            if self.prefer_snippet and len(hit.snippet) >= self.min_snippet_chars:
                document = self._snippet_document(hit)
            if document is None and self.fetch_pages:
                document = self.fetch_document(hit.url)
            if document is None:
                document = self._snippet_document(hit)
            documents.append(document)
        return documents

    @staticmethod
    def _snippet_document(hit: WebHit) -> Document:
        return Document(
            doc_id="web:" + hashlib_digest(hit.url),
            text=f"{hit.title}\n{hit.snippet}",
            source=hit.url,
            title=hit.title,
            metadata={"origin": "web"},
        )

    _MAIN_REGION_RE = re.compile(
        r"<(article|main|main-content)[^>]*>(.*?)</\1>", re.S | re.I
    )
    _BOILERPLATE_RE = re.compile(
        r"登录|注册|邀请码|Cookie|隐私政策|免责声明|全部评论|写评论|上一篇|下一篇|目录[：:]?$"
        r"|扫码关注|分享笔记|举报|广告|版权|关于我们|联系我们|网站导航|搜索一下|APP下载",
        re.I,
    )

    _ARXIV_ABSTRACT_RE = re.compile(
        r"<blockquote class=\"abstract[^\"]*\"[^>]*>(.*?)</blockquote>", re.S | re.I
    )

    @classmethod
    def extract_arxiv_abstract(cls, raw: str) -> str:
        """arXiv abs pages have no article/main region; take only the abstract blockquote."""

        match = cls._ARXIV_ABSTRACT_RE.search(raw)
        if not match:
            return ""
        return match.group(1)

    def extract_main_region(self, raw: str) -> str:
        """Prefer the article/main region so nav and login boilerplate never reaches the index."""

        matches = self._MAIN_REGION_RE.findall(raw)
        if not matches:
            return raw
        best = max(matches, key=lambda m: len(m[1]))
        return best[1]

    @staticmethod
    def html_to_text(raw: str) -> str:
        cleaned = _TAG_RE.sub(" ", raw)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        lines = [line.strip() for line in cleaned.split("\n")]
        kept = [line for line in lines if len(line) > 2 and not WebSearchRetriever._BOILERPLATE_RE.search(line)]
        return "\n".join(kept)