"""Core data types shared across the retrieval stack."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    source: str = ""
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        import hashlib

        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int = 0
    heading: str = ""
    heading_path: tuple[str, ...] = ()
    start: int = 0
    end: int = 0
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_header(self) -> str:
        if not self.heading_path:
            return ""
        return " > ".join(self.heading_path)


@dataclass(slots=True)
class RetrievedUnit:
    chunk: Chunk
    score: float
    rank: int = 0
    channels: dict[str, float] = field(default_factory=dict)
    window_text: str = ""
    rerank_score: float | None = None

    @property
    def best_text(self) -> str:
        return self.window_text or self.chunk.text

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass(slots=True)
class Grade:
    label: str
    score: float
    reason: str = ""


@dataclass(slots=True)
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    source: str
    heading: str
    snippet: str


@dataclass(slots=True)
class Trace:
    stages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def record(self, stage: str, **payload: Any) -> None:
        entry = {"stage": stage, "t": round(time.time(), 4)}
        entry.update(payload)
        self.stages.append(entry)

    def summary(self) -> str:
        parts = []
        for item in self.stages:
            keys = [f"{k}={v}" for k, v in item.items() if k not in {"stage", "t"}]
            parts.append(item["stage"] + "(" + ", ".join(keys) + ")")
        return " -> ".join(parts)


@dataclass(slots=True)
class Answer:
    query: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    units: list[RetrievedUnit] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)
    groundedness: float | None = None
    answerable: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)
    delivery: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Retrieved:
    query: str
    units: list[RetrievedUnit] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)

    @property
    def chunk_ids(self) -> list[str]:
        return [unit.chunk.chunk_id for unit in self.units]