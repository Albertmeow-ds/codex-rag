"""SQLite-backed persistence for documents, chunks and vectors."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from rag.types import Chunk, Document


class ChunkStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY, source TEXT, title TEXT,
                fingerprint TEXT, text TEXT, metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT, ordinal INTEGER,
                heading TEXT, heading_path TEXT, parent_id TEXT, text TEXT,
                start INTEGER, end INTEGER, metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id TEXT PRIMARY KEY, embedder TEXT, dim INTEGER, vector BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )
        self.conn.commit()

    def upsert_document(self, doc: Document) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, source, title, fingerprint, text, metadata) VALUES (?,?,?,?,?,?)",
            (doc.doc_id, doc.source, doc.title, doc.fingerprint, doc.text, json.dumps(doc.metadata, ensure_ascii=False)),
        )

    def document_fingerprint(self, doc_id: str) -> str | None:
        row = self.conn.execute("SELECT fingerprint FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return row[0] if row else None

    def replace_chunks(self, doc_id: str, chunks: Iterable[Chunk]) -> None:
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM vectors WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)", (doc_id,))
        self.conn.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, ordinal, heading, heading_path, parent_id, text, start, end, metadata) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    c.chunk_id, c.doc_id, c.ordinal, c.heading,
                    json.dumps(list(c.heading_path), ensure_ascii=False), c.parent_id, c.text,
                    c.start, c.end, json.dumps(c.metadata, ensure_ascii=False),
                )
                for c in chunks
            ],
        )
        self.conn.commit()

    def put_vector(self, chunk_id: str, embedder: str, vector: np.ndarray) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors (chunk_id, embedder, dim, vector) VALUES (?,?,?,?)",
            (chunk_id, embedder, int(vector.shape[0]), vector.astype(np.float32).tobytes()),
        )

    def commit(self) -> None:
        self.conn.commit()

    def load_documents(self) -> list[Document]:
        rows = self.conn.execute("SELECT doc_id, source, title, text, metadata FROM documents").fetchall()
        return [
            Document(doc_id=r[0], source=r[1] or "", title=r[2] or "", text=r[3], metadata=json.loads(r[4] or "{}"))
            for r in rows
        ]

    def load_chunks(self, indexable_only: bool = True) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, ordinal, heading, heading_path, parent_id, text, start, end, metadata "
            "FROM chunks ORDER BY doc_id, ordinal"
        ).fetchall()
        chunks = [
            Chunk(
                chunk_id=r[0], doc_id=r[1], ordinal=r[2], heading=r[3] or "",
                heading_path=tuple(json.loads(r[4] or "[]")), parent_id=r[5], text=r[6],
                start=r[7], end=r[8], metadata=json.loads(r[9] or "{}"),
            )
            for r in rows
        ]
        if not indexable_only:
            return chunks
        indexed = {row[0] for row in self.conn.execute("SELECT chunk_id FROM vectors")}
        return [c for c in chunks if c.chunk_id in indexed] or chunks

    def load_vectors(self, embedder: str) -> tuple[list[str], np.ndarray]:
        rows = self.conn.execute(
            "SELECT chunk_id, vector FROM vectors WHERE embedder = ? ORDER BY rowid", (embedder,)
        ).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        return [r[0] for r in rows], matrix

    def counts(self) -> dict[str, int]:
        return {
            "documents": self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "chunks": self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "vectors": self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0],
        }

    def close(self) -> None:
        self.conn.close()