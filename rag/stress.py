"""Retrieval behaviour under a deliberately degrading knowledge base.

RAG benchmarks usually assume a clean corpus. Real corpora have typos, truncated
notes, duplicated paragraphs and documents that contradict each other. This module
corrupts a corpus in graded steps so the cost of each corruption is measurable
instead of anecdotal (arXiv:2610.04691, RAGStress).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from rag.eval import EvalCase, evaluate_retrieval
from rag.text import split_sentences
from rag.types import Document

CORRUPTIONS = ("typo", "truncate", "duplicate", "shuffle", "noise", "conflict")

_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;\n])")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

_NOISE_POOL = (
    "本周例会安排在会议室 B，纪要由行政部归档，与本项目无直接关系。",
    "The cafeteria on floor three serves lunch from eleven thirty to two in the afternoon.",
    "差旅报销需要附上发票原件，财务每月十号前完成审核打款。",
    "A stray note about gardening: water the ferns twice a week and keep them out of direct sun.",
    "打印机缺纸时先检查纸盒卡扣，再联系行政更换 A4 纸。",
    "Reminder: the building fire drill is scheduled for the last Friday of the month.",
)


@dataclass(slots=True)
class StressConfig:
    operators: tuple[str, ...] = ("typo", "truncate", "duplicate", "shuffle", "noise", "conflict")
    levels: tuple[int, ...] = (0, 1, 2, 3)
    seed: int = 20261006
    noise_docs_per_level: int = 2
    conflict_docs_per_level: int = 1


def _typo(text: str, rng: random.Random, rate: float) -> str:
    def swap(match: re.Match[str]) -> str:
        word = match.group(0)
        if rng.random() > rate:
            return word
        index = rng.randrange(1, len(word) - 1)
        return word[: index - 1] + word[index] + word[index - 1] + word[index + 1 :]

    return _WORD_RE.sub(swap, text)


def _truncate(text: str, keep: float) -> str:
    sentences = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        cut = max(1, int(len(text) * keep))
        return text[:cut]
    keep_count = max(1, round(len(sentences) * keep))
    return "".join(sentences[:keep_count])


def _duplicate(text: str, rng: random.Random, copies: int) -> str:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text
    for _ in range(copies):
        text = text.rstrip() + "\n\n" + rng.choice(paragraphs).strip()
    return text


def _shuffle(text: str, rng: random.Random) -> str:
    blocks: list[str] = []
    for paragraph in text.split("\n\n"):
        sentences = [s for s in _SENT_SPLIT_RE.split(paragraph) if s.strip()]
        rng.shuffle(sentences)
        blocks.append("".join(sentences))
    return "\n\n".join(blocks)


def _conflict_documents(documents: Sequence[Document], count: int, rng: random.Random) -> list[Document]:
    """Contradict the corpus: restate a fact-bearing sentence with a perturbed number."""

    donors: list[str] = []
    for document in documents:
        for sentence in split_sentences(document.text):
            if _NUMBER_RE.search(sentence) and len(sentence) > 20:
                donors.append(sentence.strip())
    rng.shuffle(donors)
    injected: list[Document] = []
    for index in range(min(count, len(donors))):
        sentence = donors[index]
        flipped = _NUMBER_RE.sub(lambda m: str(int(float(m.group(0)) * 3 + 7)), sentence, count=1)
        body = (
            "# 更正说明（自动生成）\n\n"
            f"此前记录有误：{sentence}\n"
            f"实际应为：{flipped}\n"
        )
        injected.append(
            Document(doc_id=f"stress/conflict-{index}", text=body, source="stress", title="更正说明")
        )
    return injected


def _noise_documents(count: int, rng: random.Random) -> list[Document]:
    injected: list[Document] = []
    for index in range(count):
        lines = [_NOISE_POOL[(index * 3 + offset) % len(_NOISE_POOL)] for offset in range(3)]
        injected.append(
            Document(doc_id=f"stress/noise-{index}", text="\n\n".join(lines), source="stress", title="无关记录")
        )
    return injected


def corrupt_documents(
    documents: Sequence[Document],
    operators: Sequence[str] = ("typo", "truncate", "duplicate", "shuffle"),
    intensity: int = 1,
    seed: int = 20261006,
    noise_docs: int = 0,
    conflict_docs: int = 0,
) -> list[Document]:
    """Return a degraded copy of the corpus. Level 0 / no operators returns clones."""

    rng = random.Random(seed)
    out: list[Document] = []
    for document in documents:
        text = document.text
        if "typo" in operators:
            text = _typo(text, rng, min(0.45, 0.08 * intensity))
        if "truncate" in operators:
            text = _truncate(text, max(0.25, 1.0 - 0.2 * intensity))
        if "shuffle" in operators:
            text = _shuffle(text, rng)
        if "duplicate" in operators:
            text = _duplicate(text, rng, intensity)
        out.append(
            Document(
                doc_id=document.doc_id,
                text=text,
                source=document.source,
                title=document.title,
                metadata=dict(document.metadata),
            )
        )
    if conflict_docs:
        out.extend(_conflict_documents(documents, conflict_docs, rng))
    if noise_docs:
        out.extend(_noise_documents(noise_docs, rng))
    return out


def stress_curve(
    documents: Sequence[Document],
    cases: Sequence[EvalCase],
    evaluate: Callable[[Sequence[Document]], dict[str, float]],
    config: StressConfig | None = None,
) -> list[dict[str, object]]:
    """Run `evaluate` once per corruption level and return one row per level."""

    cfg = config or StressConfig()
    rows: list[dict[str, object]] = []
    for level in cfg.levels:
        operators = tuple(op for op in cfg.operators if level > 0)
        corpus = corrupt_documents(
            documents,
            operators=operators,
            intensity=level,
            seed=cfg.seed,
            noise_docs=cfg.noise_docs_per_level * level,
            conflict_docs=cfg.conflict_docs_per_level * level,
        )
        metrics = evaluate(corpus)
        row: dict[str, object] = {"level": level, "documents": len(corpus)}
        row.update(metrics)
        rows.append(row)
    return rows


def degradation_table(rows: Sequence[dict[str, object]], metric: str) -> str:
    if not rows:
        return ""
    baseline = float(rows[0].get(metric, 0.0))
    lines = [f"{'level'.ljust(9)}{metric.ljust(14)}delta"]
    for row in rows:
        value = float(row.get(metric, 0.0))
        lines.append(f"level {row['level']}".ljust(9) + f"{value:<14.4f}{value - baseline:+.4f}")
    return "\n".join(lines)
