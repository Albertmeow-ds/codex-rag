"""Latent semantic analysis embedder: TF-IDF + randomized SVD, pure NumPy."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from rag.embed.base import l2_normalize
from rag.embed.hashed import HashedNGramEmbedder
from rag.text import STOPWORDS, tokenize


def randomized_svd(matrix: np.ndarray, k: int, oversample: int = 10, power_iter: int = 4, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = matrix.shape
    k = max(1, min(k, min(rows, cols)))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((cols, k + oversample), dtype=np.float32)
    approx = matrix @ omega
    for _ in range(power_iter):
        approx = matrix @ (matrix.T @ approx)
    basis, _ = np.linalg.qr(approx)
    small = basis.T @ matrix
    u_small, singular, vt = np.linalg.svd(small, full_matrices=False)
    return basis @ u_small, singular, vt


class LSAEmbedder:
    """Corpus-fitted dense embedder.

    Stronger than hashing for small/medium corpora because it captures co-occurrence
    structure, but it must be refit when the corpus changes. Token-level vectors fall
    back to hashed projections so late interaction still works.
    """

    def __init__(self, dim: int = 256, min_df: int = 1, max_df_ratio: float = 0.9) -> None:
        self.name = f"lsa-{dim}"
        self.weight_profile = "lexical"
        self.dim = dim
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.projection: np.ndarray | None = None
        self._token_embedder = HashedNGramEmbedder(dim=dim, use_char=False)
        self.fitted = False

    def _tokens(self, text: str) -> list[str]:
        return [t for t in tokenize(text, cjk_mode="uni_bi") if t not in STOPWORDS and len(t) > 1]

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        token_docs = [self._tokens(t) for t in texts]
        doc_freq: Counter = Counter()
        for tokens in token_docs:
            doc_freq.update(set(tokens))
        max_docs = max(len(texts), 1) * self.max_df_ratio
        vocab = {term: i for i, (term, df) in enumerate(sorted(doc_freq.items())) if doc_freq[term] >= self.min_df and doc_freq[term] <= max_docs}
        if not vocab:
            vocab = {term: i for i, term in enumerate(sorted(doc_freq))}
        self.vocab = vocab
        n_terms = len(vocab)
        matrix = np.zeros((len(texts), n_terms), dtype=np.float32)
        for row, tokens in enumerate(token_docs):
            counts = Counter(tokens)
            for term, count in counts.items():
                col = vocab.get(term)
                if col is not None:
                    matrix[row, col] = 1.0 + math.log(count)
        idf = np.log((1.0 + len(texts)) / (1.0 + np.array([doc_freq[t] for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])], dtype=np.float32))) + 1.0
        self.idf = idf
        matrix *= idf
        _u, singular, vt = randomized_svd(matrix, self.dim)
        self.projection = (vt.T * singular).astype(np.float32)
        self.fitted = True
        return self

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self.vocab), dtype=np.float32)
        counts = Counter(self._tokens(text))
        for term, count in counts.items():
            col = self.vocab.get(term)
            if col is not None:
                vector[col] = (1.0 + math.log(count)) * self.idf[col]
        return vector

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("LSAEmbedder.fit() must run before embed()")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        width = min(self.dim, self.projection.shape[1])
        for row, text in enumerate(texts):
            out[row, :width] = self._vectorize(text) @ self.projection[:, :width]
        return l2_normalize(out)

    def embed_tokens(self, text: str) -> np.ndarray:
        return self._token_embedder.embed_tokens(text)