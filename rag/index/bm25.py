"""BM25+ inverted index with a CJK-friendly tokenizer upstream."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


class BM25Index:
    def __init__(self, k1: float = 1.4, b: float = 0.75, delta: float = 1.0) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_ids: list[str] = []
        self.doc_len: np.ndarray = np.empty(0, dtype=np.float32)
        self.avgdl: float = 1.0
        self.idf: dict[str, float] = {}

    def build(self, documents: Sequence[tuple[str, Iterable[str]]]) -> "BM25Index":
        self.postings = defaultdict(list)
        self.doc_ids = []
        lengths: list[int] = []
        doc_freq: dict[str, int] = defaultdict(int)

        for index, (doc_id, tokens) in enumerate(documents):
            self.doc_ids.append(doc_id)
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for term, count in counts.items():
                self.postings[term].append((index, count))
                doc_freq[term] += 1
            lengths.append(max(sum(counts.values()), 1))

        self.doc_len = np.asarray(lengths, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if lengths else 1.0
        total = len(self.doc_ids)
        self.idf = {
            term: math.log(1.0 + (total - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }
        return self

    def search(self, query_tokens: Sequence[str], k: int = 10) -> list[tuple[float, str]]:
        if not self.doc_ids or not query_tokens:
            return []
        scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        seen: set[str] = set()
        for term in query_tokens:
            if term in seen:
                continue
            seen.add(term)
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_index, tf in postings:
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_index] / self.avgdl)
                scores[doc_index] += idf * ((self.k1 + 1.0) * tf / (tf + norm)) + self.delta * idf

        k = max(1, min(k, scores.size))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.doc_ids[i]) for i in top if scores[i] > 0]

    def explain(self, query_tokens: Sequence[str], doc_id: str) -> dict[str, float]:
        if doc_id not in self.doc_ids:
            return {}
        doc_index = self.doc_ids.index(doc_id)
        contributions: dict[str, float] = defaultdict(float)
        for term in set(query_tokens):
            for index, tf in self.postings.get(term, []):
                if index != doc_index:
                    continue
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_index] / self.avgdl)
                contributions[term] += self.idf[term] * ((self.k1 + 1.0) * tf / (tf + norm)) + self.delta * self.idf[term]
        return dict(sorted(contributions.items(), key=lambda kv: -kv[1]))