"""Budget accounting for agentic retrieval.

arXiv:2610.05034 (Agentic RAG Evaluation: Budget Allocation Across Questions,
Trajectories, and Reads) shows the interesting axis is not accuracy alone but what
the loop spent to get there. Without a ledger, "it worked" and "it burned 9 model
calls to get there" look identical in the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.text import count_tokens


@dataclass(slots=True)
class Usage:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rounds: int = 0
    candidates_retrieved: int = 0
    graded_candidates: int = 0
    reranked_candidates: int = 0
    delivered_units: int = 0
    dropped_units: int = 0
    web_fetches: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: "Usage") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}

    def summary(self) -> str:
        parts = [f"{name}={value}" for name, value in self.as_dict().items() if value]
        parts.append(f"tokens={self.total_tokens}")
        return " ".join(parts)


@dataclass(slots=True)
class BudgetConfig:
    """Hard stops for one answer. 0 means unlimited."""

    max_llm_calls: int = 0
    max_tokens: int = 0
    max_graded_candidates: int = 24

    def exceeded(self, usage: Usage) -> str:
        if self.max_llm_calls and usage.llm_calls >= self.max_llm_calls:
            return f"llm_calls>={self.max_llm_calls}"
        if self.max_tokens and usage.total_tokens >= self.max_tokens:
            return f"tokens>={self.max_tokens}"
        if self.max_graded_candidates and usage.graded_candidates >= self.max_graded_candidates:
            return f"graded_candidates>={self.max_graded_candidates}"
        return ""


class CountingLLM:
    """Wraps any LLM adapter and keeps a running ledger of what the loop spent."""

    def __init__(self, inner: Any, usage: Usage | None = None) -> None:
        self.inner = inner
        self.usage = usage if usage is not None else Usage()
        self.name = f"counted({getattr(inner, 'name', 'llm')})"

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1200) -> str:
        self.usage.llm_calls += 1
        text = self.inner.complete(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        reported = getattr(self.inner, "last_usage", None) or {}
        reported_prompt = int(reported.get("prompt_tokens") or 0)
        reported_completion = int(reported.get("completion_tokens") or 0)
        self.usage.prompt_tokens += reported_prompt or count_tokens(prompt) + count_tokens(system)
        self.usage.completion_tokens += reported_completion or count_tokens(text)
        return text