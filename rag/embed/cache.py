"""Content-addressed embedding cache so re-indexing is incremental."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dim INTEGER, vector BLOB)"
        )
        self.conn.commit()

    @staticmethod
    def _key(embedder_name: str, dim: int, text: str) -> str:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        return f"{embedder_name}|{dim}|{digest}"

    def get(self, embedder_name: str, dim: int, texts: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        if not texts:
            return found
        keys = [self._key(embedder_name, dim, t) for t in texts]
        placeholders = ",".join("?" * len(keys))
        rows = self.conn.execute(
            f"SELECT key, dim, vector FROM embeddings WHERE key IN ({placeholders})", keys
        ).fetchall()
        lookup = {t: self._key(embedder_name, dim, t) for t in texts}
        by_key = {row[0]: row for row in rows}
        for text, key in lookup.items():
            row = by_key.get(key)
            if row:
                found[text] = np.frombuffer(row[2], dtype=np.float32)
        return found

    def put(self, embedder_name: str, dim: int, texts: Sequence[str], vectors: np.ndarray) -> None:
        if not texts:
            return
        rows = [
            (self._key(embedder_name, dim, text), int(vectors[i].shape[0]), vectors[i].tobytes())
            for i, text in enumerate(texts)
        ]
        self.conn.executemany("INSERT OR REPLACE INTO embeddings (key, dim, vector) VALUES (?,?,?)", rows)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()