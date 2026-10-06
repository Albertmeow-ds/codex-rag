"""Feature-hashing embedder: deterministic, corpus-free, works fully offline."""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np

from rag.embed.base import l2_normalize
from rag.text import char_ngrams, tokenize


def _hash(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashedNGramEmbedder:
    """Random-projection embedding over word + character n-gram features.

    No training data and no network: features are hashed into a fixed-width space
    with signed collisions and sublinear term weighting. It is a genuine dense
    channel (shared subspaces for morphologically similar CJK/latin surface forms)
    and the default fallback when no embedding model is reachable.
    """

    def __init__(self, dim: int = 512, char_n: int = 3, use_char: bool = True) -> None:
        self.name = f"hashed-ngram-{dim}"
        self.weight_profile = "lexical"
        self.dim = dim
        self.char_n = char_n
        self.use_char = use_char

    def _features(self, text: str) -> Counter:
        counts: Counter = Counter()
        for token in tokenize(text, cjk_mode="uni_bi"):
            if len(token) > 1:
                counts[token] += 1
        if self.use_char:
            for gram in char_ngrams(text, self.char_n):
                counts["#" + gram] += 1
        return counts

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in self._features(text).items():
                idx, sign = _hash(feature, self.dim)
                out[row, idx] += sign * (1.0 + math.log(count))
        return l2_normalize(out)

    def embed_tokens(self, text: str) -> np.ndarray:
        tokens = [t for t in tokenize(text, cjk_mode="uni_bi") if len(t) > 1]
        if not tokens:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(tokens), self.dim), dtype=np.float32)
        for row, token in enumerate(tokens):
            idx, sign = _hash(token, self.dim)
            out[row, idx] = sign
            if self.use_char:
                for gram in char_ngrams(token, self.char_n):
                    gidx, gsign = _hash("#" + gram, self.dim)
                    out[row, gidx] += 0.5 * gsign
        return l2_normalize(out)