"""Rerankers: training-free late interaction, lexical heuristics, and LLM listwise ranking."""

from __future__ import annotations

from typing import Protocol, Sequence

from rag.index.late import LateInteractionIndex
from rag.llm import LLM, extract_numbers
from rag.text import content_tokens, tokenize
from rag.types import RetrievedUnit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, units: Sequence[RetrievedUnit]) -> list[RetrievedUnit]: ...


class LateInteractionReranker:
    name = "late-interaction"

    def __init__(self, index: LateInteractionIndex, embedder: object, weight: float = 1.0) -> None:
        self.index = index
        self.embedder = embedder
        self.weight = weight

    def rerank(self, query: str, units: Sequence[RetrievedUnit]) -> list[RetrievedUnit]:
        if not self.index.enabled:
            return list(units)
        query_tokens = self.embedder.embed_tokens(query)  # type: ignore[attr-defined]
        scores = self.index.score_many(query_tokens, [u.chunk.chunk_id for u in units])
        for unit in units:
            late = scores.get(unit.chunk.chunk_id)
            if late is not None:
                unit.channels["late"] = late
                unit.rerank_score = unit.final_score + self.weight * late
        return sorted(units, key=lambda u: u.final_score, reverse=True)


class HeuristicReranker:
    """No-model reranker: coverage, tight term proximity, title match, length shaping."""

    name = "heuristic"

    def __init__(self, title_weight: float = 0.35, proximity_weight: float = 0.45) -> None:
        self.title_weight = title_weight
        self.proximity_weight = proximity_weight

    @staticmethod
    def _proximity(query_tokens: Sequence[str], text: str) -> float:
        doc_tokens = tokenize(text)
        if not doc_tokens or not query_tokens:
            return 0.0
        wanted = set(query_tokens)
        present = [i for i, token in enumerate(doc_tokens) if token in wanted]
        if not present:
            return 0.0
        distinct = len({doc_tokens[i] for i in present})
        window = present[-1] - present[0] + 1
        density = distinct / max(window, 1)
        coverage = distinct / len(wanted)
        return coverage * 0.7 + density * 0.3

    def rerank(self, query: str, units: Sequence[RetrievedUnit]) -> list[RetrievedUnit]:
        query_tokens = content_tokens(query)
        for unit in units:
            text = unit.best_text
            coverage = sum(1 for t in query_tokens if t in set(tokenize(text))) / max(len(query_tokens), 1)
            title = " ".join(unit.chunk.heading_path)
            title_hit = sum(1 for t in query_tokens if t in set(tokenize(title))) / max(len(query_tokens), 1)
            proximity = self._proximity(query_tokens, text)
            length_bonus = 1.0 if 60 <= len(text) <= 2500 else 0.85
            unit.rerank_score = (
                unit.final_score
                + coverage
                + self.title_weight * title_hit
                + self.proximity_weight * proximity
            ) * length_bonus
        return sorted(units, key=lambda u: u.final_score, reverse=True)


class LLMListwiseReranker:
    """RankGPT-style listwise reranking: the model emits a relevance permutation."""

    name = "llm-listwise"

    def __init__(self, llm: LLM, window: int = 20, keep: int = 12, max_passage_chars: int = 700) -> None:
        self.llm = llm
        self.window = window
        self.keep = keep
        self.max_passage_chars = max_passage_chars

    def rerank(self, query: str, units: Sequence[RetrievedUnit]) -> list[RetrievedUnit]:
        candidates = list(units)[: self.keep]
        if not candidates:
            return list(units)
        ranked = self._rank_window(query, candidates)
        tail = list(units)[self.keep :]
        return ranked + tail

    def _rank_window(self, query: str, candidates: Sequence[RetrievedUnit]) -> list[RetrievedUnit]:
        payload = []
        for index, unit in enumerate(candidates, start=1):
            text = unit.best_text[: self.max_passage_chars].replace("\n", " ")
            payload.append(f"[{index}] {text}")
        prompt = (
            "You are a retrieval reranker. Rank the passages by how well they answer the question.\n"
            "Consider directness, factual coverage and specificity. Ignore length.\n\n"
            f"Question: {query}\n\n"
            + "\n".join(payload)
            + "\n\nReply with ONLY the passage numbers ordered from most to least relevant, "
            "separated by ' > '. Example: 3 > 1 > 5 > 2"
        )
        try:
            raw = self.llm.complete(prompt, system="You output compact rankings and nothing else.", temperature=0.0, max_tokens=300)
        except Exception:
            return list(candidates)
        order = extract_numbers(raw)
        seen: set[int] = set()
        ranked: list[RetrievedUnit] = []
        for number in order:
            if 1 <= number <= len(candidates) and number not in seen:
                seen.add(number)
                unit = candidates[number - 1]
                unit.rerank_score = unit.final_score + (len(candidates) - len(ranked)) / len(candidates)
                ranked.append(unit)
        for unit in candidates:
            if id(unit) not in {id(u) for u in ranked}:
                unit.rerank_score = unit.final_score
                ranked.append(unit)
        return ranked


class CompositeReranker:
    """Apply several rerankers in sequence, e.g. lexical first then late interaction."""

    def __init__(self, rerankers: Sequence[Reranker]) -> None:
        self.rerankers = list(rerankers)
        self.name = "+".join(r.name for r in self.rerankers)

    def rerank(self, query: str, units: Sequence[RetrievedUnit]) -> list[RetrievedUnit]:
        current = list(units)
        for reranker in self.rerankers:
            current = reranker.rerank(query, current)
        return current


def build_reranker(name: str, *, late_index: LateInteractionIndex | None = None, embedder: object = None, llm: LLM | None = None) -> Reranker | None:
    name = (name or "").lower()
    if not name or name in {"none", "off"}:
        return None
    if "," in name:
        chain = [
            r
            for r in (
                build_reranker(part.strip(), late_index=late_index, embedder=embedder, llm=llm)
                for part in name.split(",")
                if part.strip()
            )
            if r is not None
        ]
        if not chain:
            return None
        if len(chain) == 1:
            return chain[0]
        return CompositeReranker(chain)
    if name in {"late", "late-interaction"}:
        if late_index is None or not late_index.enabled:
            return None
        return LateInteractionReranker(late_index, embedder)
    if name in {"llm", "listwise", "llm-listwise"}:
        if llm is None:
            return HeuristicReranker()
        return LLMListwiseReranker(llm)
    if name == "heuristic":
        return HeuristicReranker()
    raise ValueError(f"unknown reranker: {name}")