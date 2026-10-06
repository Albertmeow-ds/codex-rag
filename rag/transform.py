"""Query transformation: HyDE, multi-query, step-back and decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag.llm import LLM, parse_json
from rag.text import content_tokens


@dataclass(slots=True)
class TransformResult:
    strategy: str
    variants: list[str]
    notes: str = ""


class Transform(Protocol):
    name: str

    def apply(self, query: str) -> TransformResult: ...


class NoTransform:
    name = "none"

    def apply(self, query: str) -> TransformResult:
        return TransformResult("none", [query])


class KeywordFallback:
    """Used when no LLM is available: strip question phrasing, keep entities."""

    name = "keyword"

    def apply(self, query: str) -> TransformResult:
        tokens = content_tokens(query)
        if not tokens:
            return TransformResult("keyword", [query])
        return TransformResult("keyword", [query, " ".join(tokens)])


class HyDE:
    """Hypothetical Document Embeddings: generate a plausible answer, embed that instead."""

    name = "hyde"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def apply(self, query: str) -> TransformResult:
        prompt = (
            f"Write a short factual passage (120-200 words) that would answer this question. "
            f"If you are unsure, write the most plausible technical explanation.\n\nQuestion: {query}"
        )
        try:
            passage = self.llm.complete(prompt, system="You write concise encyclopedic passages.", temperature=0.3, max_tokens=420)
        except Exception as exc:
            return TransformResult("hyde", [query], notes=f"llm failed: {exc}")
        if not passage.strip():
            return TransformResult("hyde", [query], notes="empty hypothetical document")
        return TransformResult("hyde", [query, passage], notes="hypothetical document appended")


class MultiQuery:
    name = "multi-query"

    def __init__(self, llm: LLM, n: int = 3) -> None:
        self.llm = llm
        self.n = n

    def apply(self, query: str) -> TransformResult:
        prompt = (
            f"Produce {self.n} paraphrased search queries that retrieve different aspects of this question. "
            f"Output a JSON array of strings only.\n\nQuestion: {query}"
        )
        try:
            raw = self.llm.complete(prompt, system="You output JSON only.", temperature=0.5, max_tokens=300)
        except Exception as exc:
            return TransformResult("multi-query", [query], notes=f"llm failed: {exc}")
        parsed = parse_json(raw, default=None)
        variants = [v.strip() for v in parsed if isinstance(v, str) and v.strip()] if isinstance(parsed, list) else []
        return TransformResult("multi-query", [query] + variants[: self.n], notes=f"{len(variants)} variants")


class StepBack:
    name = "step-back"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def apply(self, query: str) -> TransformResult:
        prompt = (
            "Rewrite the specific question below as a broader conceptual question whose answer provides "
            "the background needed to answer it. Output one question only.\n\n"
            f"Question: {query}"
        )
        try:
            abstract = self.llm.complete(prompt, system="You ask one abstract question.", temperature=0.3, max_tokens=160)
        except Exception as exc:
            return TransformResult("step-back", [query], notes=f"llm failed: {exc}")
        abstract = abstract.strip().split("\n")[0]
        return TransformResult("step-back", [query, abstract] if abstract else [query], notes="abstract question added")


class Decompose:
    name = "decompose"

    def __init__(self, llm: LLM, max_subquestions: int = 3) -> None:
        self.llm = llm
        self.max_subquestions = max_subquestions

    def apply(self, query: str) -> TransformResult:
        prompt = (
            f"Split this question into at most {self.max_subquestions} atomic sub-questions needed to answer it. "
            f"Output a JSON array of strings only.\n\nQuestion: {query}"
        )
        try:
            raw = self.llm.complete(prompt, system="You output JSON only.", temperature=0.2, max_tokens=300)
        except Exception as exc:
            return TransformResult("decompose", [query], notes=f"llm failed: {exc}")
        parsed = parse_json(raw, default=None)
        subs = [v.strip() for v in parsed if isinstance(v, str) and v.strip()] if isinstance(parsed, list) else []
        return TransformResult("decompose", [query] + subs[: self.max_subquestions], notes=f"{len(subs)} sub-questions")


def build_transform(name: str, llm: LLM | None) -> Transform:
    name = (name or "none").lower()
    if name in {"", "none", "off"}:
        return NoTransform()
    if llm is None:
        return KeywordFallback()
    if name == "hyde":
        return HyDE(llm)
    if name in {"multi", "multi-query"}:
        return MultiQuery(llm)
    if name == "step-back":
        return StepBack(llm)
    if name == "decompose":
        return Decompose(llm)
    raise ValueError(f"unknown transform: {name}")