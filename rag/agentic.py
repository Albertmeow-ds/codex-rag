"""Agentic control loop: adaptive retrieval, corrective grading, and self-checking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from rag.evidence import EvidenceConfig, EvidenceGapChecker, gap_terms
from rag.generate import delivery_report
from rag.llm import LLM, parse_json
from rag.retrieve.hybrid import HybridRetriever
from rag.retrieve.web import WebSearchRetriever
from rag.text import content_tokens, tokenize
from rag.transform import Transform
from rag.types import Answer, Document, Grade, RetrievedUnit, Trace
from rag.usage import BudgetConfig, Usage


@dataclass(slots=True)
class AgenticConfig:
    enabled: bool = True
    max_rounds: int = 3
    max_graded_candidates: int = 6
    relevance_threshold: float = 0.34
    weak_coverage_threshold: float = 0.35
    groundedness_threshold: float = 0.3
    allow_web_fallback: bool = False
    adaptive: bool = True
    # Budget stops (0 = unlimited). arXiv:2610.05034: report what the loop spent, not just what it found.
    max_llm_calls: int = 0
    max_total_tokens: int = 0
    # Evidence sufficiency (arXiv:2609.37469). Relevance grading alone keeps passing
    # passages that name the right entity but never carry the requested fact.
    evidence: bool = True
    evidence_drives_loop: bool = True
    evidence_gate: bool = False
    evidence_threshold: float = 0.5
    # arXiv:2610.06191: agents rarely turn "this retrieval was useless" into a stop decision.
    # When no candidate even names the queried entity, re-querying the same corpus cannot help.
    evidence_stop_on_no_entity: bool = False
    # A corpus with no lexical hit at all has nothing to say. Measured on the real
    # casebook corpus: 0/30 answerable questions stopped, 4/8 out-of-corpus caught.
    stop_when_no_lexical_hit: bool = False
    lexical_hit_min_score: float = 0.05


class RetrievalGrader:
    """CRAG-style grader labelling each retrieved passage relevant / ambiguous / irrelevant."""

    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm

    def grade_batch(self, query: str, units: Sequence[RetrievedUnit]) -> list[Grade]:
        """Grade every candidate in ONE model call.

        Per-candidate grading multiplies latency by the candidate count and is the
        usual reason agentic RAG loops feel unbearably slow.
        """

        if self.llm is None or not units:
            return [self._grade_lexical(query, unit) for unit in units]
        numbered = "\n".join(
            f"[{index}] {unit.best_text[:600].replace(chr(10), ' ')}" for index, unit in enumerate(units, start=1)
        )
        prompt = (
            "Label whether each passage helps answer the question.\n"
            'Return a JSON array with one object per passage: '
            '{"index": <int>, "label": "relevant" | "ambiguous" | "irrelevant", "score": <0.0-1.0>}\n\n'
            f"Question: {query}\n\nPassages:\n{numbered}"
        )
        try:
            raw = self.llm.complete(prompt, system="You output JSON only.", temperature=0.0, max_tokens=400)
        except Exception:
            return [self._grade_lexical(query, unit) for unit in units]
        parsed = parse_json(raw, default=None)
        if not isinstance(parsed, list):
            return [self._grade_lexical(query, unit) for unit in units]
        by_index: dict[int, Grade] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).lower()
            if label not in {"relevant", "ambiguous", "irrelevant"}:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(units):
                by_index[index] = Grade(label, score)
        return [by_index.get(i) or self._grade_lexical(query, unit) for i, unit in enumerate(units, start=1)]

    def grade(self, query: str, unit: RetrievedUnit) -> Grade:
        if self.llm is not None:
            graded = self._grade_llm(query, unit)
            if graded is not None:
                return graded
        return self._grade_lexical(query, unit)

    def _grade_lexical(self, query: str, unit: RetrievedUnit) -> Grade:
        query_tokens = set(content_tokens(query))
        text_tokens = set(tokenize(unit.best_text))
        coverage = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        if coverage >= 0.5:
            return Grade("relevant", coverage, f"lexical coverage {coverage:.2f}")
        if coverage >= self.self_lexical_threshold():
            return Grade("ambiguous", coverage, f"lexical coverage {coverage:.2f}")
        return Grade("irrelevant", coverage, f"lexical coverage {coverage:.2f}")

    def _grade_llm(self, query: str, unit: RetrievedUnit) -> Grade | None:
        prompt = (
            "Label whether the passage helps answer the question.\n"
            'Respond with JSON only: {"label": "relevant" | "ambiguous" | "irrelevant", "score": 0.0-1.0}\n\n'
            f"Question: {query}\nPassage: {unit.best_text[:900]}"
        )
        try:
            raw = self.llm.complete(prompt, system="You output JSON only.", temperature=0.0, max_tokens=80)
        except Exception:
            return None
        parsed = parse_json(raw, default=None)
        if not isinstance(parsed, dict):
            return None
        label = str(parsed.get("label", "")).lower()
        if label not in {"relevant", "ambiguous", "irrelevant"}:
            return None
        try:
            score = float(parsed.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return Grade(label, score)

    @staticmethod
    def self_lexical_threshold() -> float:
        return 0.2


def coverage_ratio(query: str, units: Sequence) -> float:
    query_tokens = set(content_tokens(query))
    if not query_tokens:
        return 0.0
    text_tokens: set[str] = set()
    for unit in units:
        text_tokens.update(tokenize(unit.best_text))
    return len(query_tokens & text_tokens) / len(query_tokens)


class AgenticRAG:
    """Wraps retrieve+answer with query rewriting, corrective retrieval and groundedness retry."""

    def __init__(
        self,
        retriever: HybridRetriever,
        answerer: Any,
        config: AgenticConfig | None = None,
        grader: RetrievalGrader | None = None,
        transform: Transform | None = None,
        web: WebSearchRetriever | None = None,
        augment: Callable[[Sequence[Document]], None] | None = None,
        usage: Any = None,
        evidence: EvidenceGapChecker | None = None,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.config = config or AgenticConfig()
        self.grader = grader or RetrievalGrader()
        self.transform = transform
        self.web = web
        self.augment = augment
        self.usage = usage
        self.evidence = evidence or EvidenceGapChecker(
            EvidenceConfig(
                enabled=config.evidence if config is not None else True,
                gate_answer=config.evidence_gate if config is not None else False,
                support_threshold=config.evidence_threshold if config is not None else 0.5,
            )
        )

    def run(self, query: str, filters: dict[str, Any] | None = None) -> Answer:
        cfg = self.config
        usage = self.usage if self.usage is not None else Usage()
        budget = BudgetConfig(max_llm_calls=cfg.max_llm_calls, max_tokens=cfg.max_total_tokens)
        best: Answer | None = None
        loop_trace = Trace()

        for round_index in range(1, cfg.max_rounds + 1):
            usage.rounds = round_index
            variants = [query]
            if self.transform is not None:
                variants = self.transform.apply(query).variants

            units: list[RetrievedUnit] = []
            for variant in variants:
                retrieved = self.retriever.retrieve(variant, filters=filters)
                loop_trace.stages.extend(retrieved.trace.stages)
                units.extend(retrieved.units)
            units = _dedupe_units(units)
            usage.candidates_retrieved += len(units)

            units = units[: cfg.max_graded_candidates]
            grades = self.grader.grade_batch(query, units)
            usage.graded_candidates += len(units)
            relevant = [g for g in grades if g.label == "relevant"]
            coverage = coverage_ratio(query, units)
            quality = (len(relevant) / max(len(grades), 1)) * 0.6 + coverage * 0.4

            report = self.evidence.check(query, units) if cfg.evidence else None
            if report is not None:
                loop_trace.record("evidence", round=round_index, **report.as_dict())

            loop_trace.record(
                "grade",
                round=round_index,
                variants=len(variants),
                candidates=len(units),
                relevant=len(relevant),
                coverage=round(coverage, 3),
                quality=round(quality, 3),
            )

            if cfg.adaptive and round_index == 1 and coverage < 0.15 and not cfg.enabled:
                break

            stop_reason = budget.exceeded(usage)
            if stop_reason:
                loop_trace.record("budget", round=round_index, reason=stop_reason)
                break

            weak = quality < cfg.weak_coverage_threshold
            gap = report is not None and not report.answerable
            web_available = cfg.allow_web_fallback and self.web is not None
            if cfg.stop_when_no_lexical_hit and not web_available:
                lexical_hits = sum(
                    1 for u in units if u.channels.get("bm25", 0.0) > cfg.lexical_hit_min_score
                )
                if lexical_hits == 0:
                    loop_trace.record("stop", round=round_index, reason="no_lexical_hit", candidates=len(units))
                    break
            if (
                cfg.evidence_stop_on_no_entity
                and report is not None
                and report.entity_coverage == 0.0
                and weak
                and not web_available
            ):
                loop_trace.record(
                    "stop", round=round_index, reason="no_entity_coverage", missing=list(report.missing_entities)[:6]
                )
                break
            if cfg.enabled and (weak or (gap and cfg.evidence_drives_loop)) and round_index < cfg.max_rounds:
                loop_trace.record(
                    "retry",
                    round=round_index,
                    weak=weak,
                    evidence_gap=gap,
                    entity_coverage=round(report.entity_coverage, 3) if report is not None else 1.0,
                )
                query = self._rewrite(query, units, grades, loop_trace, report)
                if cfg.allow_web_fallback and self.web is not None:
                    usage.web_fetches += 1
                    self._web_augment(query, filters)
                continue

            answer = self.answerer.answer(query, units)
            if report is not None:
                answer.evidence = report.as_dict()
                if cfg.evidence_gate and not report.answerable:
                    answer.answerable = False
            answer.delivery = delivery_report(units, answer.units)
            usage.delivered_units = len(answer.units)
            usage.dropped_units = int(answer.delivery.get("dropped", 0))
            answer.trace = loop_trace
            loop_trace.record("answer", round=round_index, sources=len(answer.units), **answer.delivery)
            if best is None or (answer.groundedness or 0.0) > (best.groundedness or 0.0):
                best = answer

            if (answer.groundedness or 0.0) >= cfg.groundedness_threshold:
                loop_trace.usage = usage.as_dict()
                loop_trace.record("final", groundedness=round(answer.groundedness or 0.0, 3))
                return answer
            if round_index >= cfg.max_rounds:
                break
            query = self._rewrite(query, units, grades, loop_trace, report)

        if best is None:
            best = Answer(query=query, text="", citations=[], units=[], answerable=False, groundedness=0.0)
        best.trace = loop_trace
        loop_trace.usage = usage.as_dict()
        loop_trace.record("final", groundedness=round(best.groundedness or 0.0, 3))
        return best

    def _rewrite(
        self,
        query: str,
        units: Sequence[RetrievedUnit],
        grades: list[Grade],
        trace: Trace,
        report: Any = None,
    ) -> str:
        if report is not None:
            missing = gap_terms(query, units, report)
        else:
            missing = sorted(
                {t for t in content_tokens(query)} - {t for u in units for t in tokenize(u.best_text)}
            )
        strong = sorted(
            {t for u in units if u.final_score > 0 for t in tokenize(u.best_text) if len(t) > 2}
        )[:8]
        rewritten = " ".join(missing[:6] or strong[:6] or [query])
        trace.record("rewrite", round=len(trace.stages), missing_terms=missing[:8], expanded=rewritten[:120])
        return f"{query} {rewritten}".strip()

    def _web_augment(self, query: str, filters: dict[str, Any] | None) -> None:
        if self.web is None:
            return
        try:
            documents = self.web.search_documents(query, max_results=3)
        except Exception as exc:
            self.retriever.trace.record("web", status="failed", error=str(exc)[:120])
            return
        if documents:
            self.retriever.trace.record("web", fetched=len(documents))
            if self.augment is not None:
                self.augment(documents)


def _dedupe_units(units: Sequence[RetrievedUnit]) -> list[RetrievedUnit]:
    best: dict[str, RetrievedUnit] = {}
    for unit in units:
        current = best.get(unit.chunk.chunk_id)
        if current is None or unit.final_score > current.final_score:
            best[unit.chunk.chunk_id] = unit
    return sorted(best.values(), key=lambda u: u.final_score, reverse=True)