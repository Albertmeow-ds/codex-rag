"""Exact dense ANN channel (batched cosine) with optional cached vectors."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rag.embed.base import cosine_topk, l2_normalize


class DenseIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.ids)

    def build(self, ids: Sequence[str], vectors: np.ndarray) -> "DenseIndex":
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        self.ids = list(ids)
        self.matrix = l2_normalize(vectors)
        return self

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[float, str]]:
        scores, idx = cosine_topk(query_vector, self.matrix, k)
        return [(float(score), self.ids[i]) for score, i in zip(scores, idx) if score > 0]

    def search_multi(self, query_vectors: np.ndarray, k: int = 10) -> list[tuple[float, str]]:
        """Multi-query channel: best score per chunk across all query variants."""

        if query_vectors.shape[0] == 0 or self.matrix.shape[0] == 0:
            return []
        best = np.max(l2_normalize(query_vectors) @ self.matrix.T, axis=0)
        k = max(1, min(k, best.size))
        top = np.argpartition(-best, k - 1)[:k]
        top = top[np.argsort(-best[top])]
        return [(float(best[i]), self.ids[i]) for i in top if best[i] > 0]