"""Embedder contract shared by every dense channel in the stack."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalized vectors."""

    def embed_tokens(self, text: str) -> np.ndarray:
        """Return an (t, dim) matrix used by late-interaction reranking."""


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int, batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k with batched matmul so large indexes stay memory-safe."""

    if matrix.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    query = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
    scores = np.empty(matrix.shape[0], dtype=np.float32)
    for start in range(0, matrix.shape[0], batch):
        stop = min(start + batch, matrix.shape[0])
        scores[start:stop] = (matrix[start:stop] @ query[0])
    k = max(1, min(k, matrix.shape[0]))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return scores[top_idx], top_idx