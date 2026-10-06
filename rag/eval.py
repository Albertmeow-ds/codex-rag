"""Retrieval and generation metrics (Recall@k, MRR, nDCG, faithfulness, answer relevancy)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from rag.generate import citation_coverage, groundedness_score
from rag.text import tokenize


@dataclass(slots=True)
class EvalCase:
    question: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_doc_ids: list[str] = field(default_factory=list)
    gold_answer: str = ""


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        cases.append(
            EvalCase(
                question=raw["question"],
                gold_chunk_ids=list(raw.get("gold_chunk_ids", [])),
                gold_doc_ids=list(raw.get("gold_doc_ids", [])),
                gold_answer=raw.get("gold_answer", ""),
            )
        )
    return cases


def _gold_set(case: EvalCase, chunk_doc: dict[str, str]) -> set[str]:
    gold = set(case.gold_chunk_ids)
    if case.gold_doc_ids:
        gold.update(cid for cid, doc_id in chunk_doc.items() if doc_id in case.gold_doc_ids)
    return gold


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & gold) / k


def mrr(retrieved: Sequence[str], gold: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    gains = [1.0 if chunk_id in gold else 0.0 for chunk_id in retrieved[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def hit_rate(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def evaluate_retrieval(cases: Sequence[EvalCase], retrieve, k: int = 8) -> dict[str, float]:
    totals = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0, "hit": 0.0}
    evaluated = 0
    for case in cases:
        retrieved = retrieve(case.question)
        chunk_doc = {cid: cid.split("#")[0] for cid in retrieved}
        gold = _gold_set(case, chunk_doc)
        if not gold:
            continue
        evaluated += 1
        totals["recall"] += recall_at_k(retrieved, gold, k)
        totals["precision"] += precision_at_k(retrieved, gold, k)
        totals["mrr"] += mrr(retrieved, gold)
        totals["ndcg"] += ndcg_at_k(retrieved, gold, k)
        totals["hit"] += hit_rate(retrieved, gold, k)
    if not evaluated:
        return {f"{key}@{k}": 0.0 for key in totals}
    return {f"{key}@{k}": round(totals[key] / evaluated, 4) for key in totals}


def answer_relevancy(question: str, answer: str, embedder) -> float:
    if not answer.strip():
        return 0.0
    vectors = embedder.embed([question, answer])
    return round(float(vectors[0] @ vectors[1]), 4)


def evaluate_generation(cases: Sequence[EvalCase], answer_fn, embedder) -> dict[str, float]:
    faithfulness: list[float] = []
    relevancy: list[float] = []
    citations: list[float] = []
    answerable: list[float] = []
    for case in cases:
        answer = answer_fn(case.question)
        faithfulness.append(groundedness_score(answer.text, answer.units))
        relevancy.append(answer_relevancy(case.question, answer.text, embedder))
        citations.append(citation_coverage(answer.text, len(answer.units)))
        answerable.append(1.0 if answer.answerable else 0.0)
    n = max(len(cases), 1)
    return {
        "faithfulness": round(sum(faithfulness) / n, 4),
        "answer_relevancy": round(sum(relevancy) / n, 4),
        "citation_coverage": round(sum(citations) / n, 4),
        "answerable_rate": round(sum(answerable) / n, 4),
    }


def report(metrics: dict[str, float]) -> str:
    width = max((len(k) for k in metrics), default=8)
    return "\n".join(f"{k.ljust(width)}  {v}" for k, v in metrics.items())