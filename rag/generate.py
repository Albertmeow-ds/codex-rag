"""Context assembly and grounded generation with citations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence

from rag.llm import LLM
from rag.text import count_tokens, split_sentences, tokenize
from rag.types import Answer, Citation, RetrievedUnit

CITATION_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass(slots=True)
class GenerationConfig:
    max_context_tokens: int = 3200
    max_output_tokens: int = 1400
    max_claims: int = 24
    order: str = "u-shape"
    dedupe: bool = True
    require_citations: bool = True
    system_prompt: str = (
        "You answer strictly from the provided numbered sources. "
        "Every factual sentence must cite its source as [n]. "
        "If the sources are insufficient, say exactly what is missing instead of inventing facts."
    )


def _text_key(text: str) -> str:
    return hashlib.blake2b(text.strip().encode("utf-8"), digest_size=12).hexdigest()


def assemble_context(units: Sequence[RetrievedUnit], cfg: GenerationConfig) -> list[RetrievedUnit]:
    """Dedupe, expand, budget-trim, then reorder to fight the lost-in-the-middle effect."""

    selected: list[RetrievedUnit] = []
    seen_texts: set[str] = set()
    seen_ids: set[str] = set()
    budget = cfg.max_context_tokens

    for unit in sorted(units, key=lambda u: u.final_score, reverse=True):
        text = unit.best_text
        if not text:
            continue
        if cfg.dedupe:
            if unit.chunk.chunk_id in seen_ids:
                continue
            key = _text_key(text)
            if key in seen_texts:
                continue
            seen_ids.add(unit.chunk.chunk_id)
            seen_texts.add(key)
        cost = count_tokens(text)
        if selected and cost > budget:
            continue
        budget -= cost
        selected.append(unit)

    if cfg.order == "u-shape" and len(selected) > 2:
        left: list[RetrievedUnit] = []
        right: list[RetrievedUnit] = []
        for index, unit in enumerate(selected):
            (left if index % 2 == 0 else right).append(unit)
        selected = left + right[::-1]

    for index, unit in enumerate(selected, start=1):
        unit.rank = index
    return selected


def delivery_report(units: Sequence[RetrievedUnit], selected: Sequence[RetrievedUnit]) -> dict[str, int]:
    """How much of what retrieval found actually reached the prompt.

    arXiv:2610.06689 found supporting passages routinely retrieved but never delivered
    to the agent; without this counter the loss is invisible in the trace.
    """

    retrieved_tokens = sum(count_tokens(u.best_text) for u in units)
    delivered_tokens = sum(count_tokens(u.best_text) for u in selected)
    return {
        "retrieved": len(units),
        "delivered": len(selected),
        "dropped": max(0, len(units) - len(selected)),
        "context_tokens": delivered_tokens,
        "dropped_tokens": max(0, retrieved_tokens - delivered_tokens),
    }


def render_context(units: Sequence[RetrievedUnit]) -> str:
    blocks: list[str] = []
    for index, unit in enumerate(units, start=1):
        heading = unit.chunk.context_header or unit.chunk.doc_id
        blocks.append(f"[{index}] source={unit.chunk.doc_id} | {heading}\n{unit.best_text.strip()}")
    return "\n\n".join(blocks)


def build_citations(units: Sequence[RetrievedUnit]) -> list[Citation]:
    citations: list[Citation] = []
    for index, unit in enumerate(units, start=1):
        snippet = unit.best_text.strip().split("\n")
        body = " ".join(s for s in snippet if len(s) > 3)[:220]
        citations.append(
            Citation(
                index=index,
                chunk_id=unit.chunk.chunk_id,
                doc_id=unit.chunk.doc_id,
                source=unit.chunk.metadata.get("source", unit.chunk.doc_id) if unit.chunk.metadata else unit.chunk.doc_id,
                heading=unit.chunk.heading or unit.chunk.context_header,
                snippet=body,
            )
        )
    return citations


class GroundedAnswerer:
    name = "grounded-llm"

    def __init__(self, llm: LLM, cfg: GenerationConfig | None = None) -> None:
        self.llm = llm
        self.cfg = cfg or GenerationConfig()

    def answer(self, query: str, units: Sequence[RetrievedUnit]) -> Answer:
        selected = assemble_context(units, self.cfg)
        context = render_context(selected)
        citation_rule = (
            "Cite every factual claim with [n] referencing the numbered sources."
            if self.cfg.require_citations
            else "You may paraphrase the sources."
        )
        prompt = (
            f"Question: {query}\n\nNumbered sources:\n{context}\n\n"
            f"Answer in the same language as the question. {citation_rule}\n"
            "Prefer 3-6 dense points; add a short 'Gaps' line if evidence is partial."
        )
        text = self.llm.complete(
            prompt,
            system=self.cfg.system_prompt,
            temperature=0.1,
            max_tokens=self.cfg.max_output_tokens,
        )
        return Answer(
            query=query,
            text=text,
            citations=build_citations(selected),
            units=list(selected),
            groundedness=groundedness_score(text, selected),
            delivery=delivery_report(units, selected),
        )


class ExtractiveAnswerer:
    """Model-free fallback: MMR-selected evidence sentences with citations."""

    name = "extractive"

    def __init__(self, cfg: GenerationConfig | None = None, sentences: int = 6) -> None:
        self.cfg = cfg or GenerationConfig()
        self.sentences = sentences

    def answer(self, query: str, units: Sequence[RetrievedUnit]) -> Answer:
        selected = assemble_context(units, self.cfg)
        query_tokens = set(tokenize(query))
        candidates: list[tuple[float, str, int, str]] = []
        for index, unit in enumerate(selected, start=1):
            unit_tokens = set(tokenize(unit.best_text))
            for sentence in split_sentences(unit.best_text):
                sentence_tokens = set(tokenize(sentence))
                if len(sentence_tokens) < 3:
                    continue
                overlap = len(sentence_tokens & query_tokens) / max(len(query_tokens), 1)
                candidates.append((unit.final_score * 0.5 + overlap, unit.chunk.doc_id, index, sentence))

        chosen: list[tuple[float, str, int, str]] = []
        covered: set[str] = set()
        while candidates and len(chosen) < self.sentences:
            best_index = max(
                range(len(candidates)),
                key=lambda i: candidates[i][0] - 0.35 * len(set(tokenize(candidates[i][3])) & covered),
            )
            score, doc_id, source_index, sentence = candidates.pop(best_index)
            chosen.append((score, doc_id, source_index, sentence))
            covered.update(tokenize(sentence))

        if not chosen:
            return Answer(
                query=query,
                text="",
                citations=build_citations(selected),
                units=list(selected),
                answerable=False,
                groundedness=0.0,
                delivery=delivery_report(units, selected),
            )

        lines = [f"- {sentence} [{source_index}]" for _s, _d, source_index, sentence in chosen]
        text = "\n".join(lines)
        return Answer(
            query=query,
            text=text,
            citations=build_citations(selected),
            units=list(selected),
            groundedness=groundedness_score(text, selected),
            delivery=delivery_report(units, selected),
        )


def groundedness_score(answer: str, units: Sequence[RetrievedUnit], threshold: float = 0.34) -> float:
    """Lexical entailment proxy: share of answer claims supported by some source."""

    claims = [s for s in split_sentences(answer) if len(tokenize(s)) >= 4]
    if not claims:
        return 0.0
    source_token_sets = [set(tokenize(u.best_text)) for u in units]
    supported = 0
    for claim in claims[:24]:
        claim_tokens = {t for t in tokenize(claim) if len(t) > 1}
        if not claim_tokens:
            continue
        best = 0.0
        for source_tokens in source_token_sets:
            overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
            best = max(best, overlap)
            if best >= threshold:
                break
        if best >= threshold:
            supported += 1
    return supported / len(claims[:24])


def citation_coverage(answer: str, n_sources: int) -> float:
    cited = {int(n) for n in CITATION_RE.findall(answer) if 1 <= int(n) <= max(n_sources, 1)}
    return len(cited) / max(n_sources, 1)


def build_answerer(llm: LLM | None, cfg: GenerationConfig | None = None):
    if llm is None:
        return ExtractiveAnswerer(cfg)
    return GroundedAnswerer(llm, cfg)