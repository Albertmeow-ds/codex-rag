"""ColBERT-style late interaction index used as a training-free reranker.

Score(query, chunk) = mean over query tokens of max cosine over chunk tokens.
Because it compares token-level representations after retrieval, it recovers
precision that single-vector retrieval loses on long chunks.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rag.embed.base import l2_normalize


class LateInteractionIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.offsets: list[tuple[int, int]] = []
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

    @property
    def enabled(self) -> bool:
        return bool(self.ids) and self.vectors.size > 0

    def build(self, token_matrices: Sequence[tuple[str, np.ndarray]]) -> "LateInteractionIndex":
        blocks: list[np.ndarray] = []
        self.ids = []
        self.offsets = []
        cursor = 0
        for chunk_id, matrix in token_matrices:
            if matrix.size == 0:
                continue
            normalized = l2_normalize(matrix)
            blocks.append(normalized)
            self.ids.append(chunk_id)
            self.offsets.append((cursor, cursor + normalized.shape[0]))
            cursor += normalized.shape[0]
        if blocks:
            self.vectors = np.concatenate(blocks, axis=0).astype(np.float32)
        return self

    def score(self, query_tokens: np.ndarray, chunk_index: int) -> float:
        if not self.enabled or query_tokens.size == 0:
            return 0.0
        start, end = self.offsets[chunk_index]
        return self._maxsim(l2_normalize(query_tokens), self.vectors[start:end])

    @staticmethod
    def _maxsim(query: np.ndarray, block: np.ndarray) -> float:
        """ColBERT MaxSim: per query token take the best matching chunk token, then average.

        Averaging (instead of taking one global maximum) is what keeps the score
        discriminative: a single shared token must not saturate a whole passage.
        """

        if block.shape[0] == 0:
            return 0.0
        similarities = query @ block.T
        return float(similarities.max(axis=1).mean())

    def score_many(self, query_tokens: np.ndarray, chunk_ids: Sequence[str]) -> dict[str, float]:
        if not self.enabled or query_tokens.size == 0:
            return {}
        normalized_query = l2_normalize(query_tokens)
        position = {chunk_id: i for i, chunk_id in enumerate(self.ids)}
        out: dict[str, float] = {}
        for chunk_id in chunk_ids:
            index = position.get(chunk_id)
            if index is None:
                continue
            start, end = self.offsets[index]
            block = self.vectors[start:end]
            if block.shape[0] == 0:
                continue
            out[chunk_id] = self._maxsim(normalized_query, block)
        return out