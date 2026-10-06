"""Evidence sufficiency: a passage can be topically relevant and still not answer the question.

Motivated by arXiv:2609.37469 (Relevance Is Not Sufficient Evidence) and
arXiv:2609.03482 (From Topical Relevance to Answerability). A grader that only asks
"is this on topic?" keeps passing evidence that names the right entity but never
carries the fact the question asks for, so the generator answers anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from rag.text import STOPWORDS, ZH_STOPWORDS, content_tokens, tokenize
from rag.types import RetrievedUnit


@dataclass(slots=True)
class EvidenceConfig:
    enabled: bool = True
    drive_loop: bool = True
    gate_answer: bool = False
    support_threshold: float = 0.5
    entity_coverage_min: float = 0.5


@dataclass(slots=True)
class QuerySpec:
    query: str
    answer_type: str = "entity"
    entities: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    key_terms: tuple[str, ...] = ()


@dataclass(slots=True)
class EvidenceReport:
    spec: QuerySpec
    answerable: bool = True
    support: float = 0.0
    entity_coverage: float = 1.0
    term_coverage: float = 1.0
    slot_found: bool = True
    supporting_units: int = 0
    missing_entities: tuple[str, ...] = ()
    missing_slots: tuple[str, ...] = ()
    per_unit: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer_type": self.spec.answer_type,
            "answerable": self.answerable,
            "support": round(self.support, 3),
            "entity_coverage": round(self.entity_coverage, 3),
            "term_coverage": round(self.term_coverage, 3),
            "slot_found": self.slot_found,
            "supporting_units": self.supporting_units,
            "missing_entities": list(self.missing_entities)[:8],
            "missing_slots": list(self.missing_slots),
        }


_QUOTED_RE = re.compile(r"[\"'\u201c\u2018\u300c\u300e]([^\u201d\u2019\u300d\u300f\"']{2,40})[\u201d\u2019\u300d\u300f\"']")
_CODE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{3,}")
# Order matters: the first matching cue wins, so specific shapes come before generic ones.
_ANSWER_TYPE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # The interrogative, not the topic noun, decides the shape of the required fact.
    ("count", ("多少", "几个", "几条", "几次", "数量", "总数", "how many", "how much", "number of", "count")),
    ("date", ("什么时候", "何时", "日期", "哪年", "几月", "when", "what date", "which year")),
    ("cause", ("为什么", "为何", "原因", "导致", "因为", "why", "cause", "reason", "due to")),
    ("procedure", ("怎么", "如何", "怎样", "步骤", "方法", "流程", "how to", "how do", "how can", "steps", "procedure")),
    ("yesno", ("是否", "能不能", "可不可以", "能否", "吗", "does ", "can ", "is it", "should ", "could ")),
    ("comparison", ("哪个更", "更快", "更好", "区别", "差异", "对比", "compare", "difference", " faster", " vs ")),
    ("definition", ("是什么", "什么是", "定义", "含义", "what is", "what are", "define", "meaning of")),
    ("version", ("版本号", "哪个版本", "什么版本", "which version", "what version")),
    ("entity", ("哪个", "哪些", "谁", "哪里", "which", "who", "where", "what")),
)

# A slot marker is a weak cue that the passage carries the *shape* of the requested fact.
_SLOT_MARKERS: dict[str, tuple[str, ...]] = {
    "cause": ("因为", "由于", "原因", "导致", "因为", "because", "since ", "due to", "caused", "leads to", "results in"),
    "procedure": ("先", "然后", "接着", "首先", "步骤", "运行", "执行", "配置", "安装", "step ", "first", "then ", "finally", "run ", "set ", "add "),
    "comparison": ("更", "比", "高于", "低于", "优于", "相比", "than ", "faster", "slower", "better", "worse", "higher", "lower", "instead"),
    "yesno": ("可以", "不能", "支持", "不支持", "需要", "无需", "禁止", "can ", "cannot", "supports", "does not", "required", "optional"),
    "definition": ("是指", "指的是", "称为", "即", "用于", "refers to", "means", "is defined", "called", "consists of"),
}

_STANDALONE = r"(?<![A-Za-z0-9_\-])\d+(?:[.,]\d+)*(?![A-Za-z0-9_\-])"
_NUMERIC_RE = re.compile(_STANDALONE)
_VERSION_RE = re.compile(r"[vV]?\d+\.\d+(?:\.\d+)?")
_DATE_RE = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}|\d{4}\s*年|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2},? \d{4}", re.IGNORECASE)

# answer types whose requested fact has a hard surface shape (a number, a date, a version)
_FACT_SHAPE_RE: dict[str, re.Pattern[str]] = {
    "count": _NUMERIC_RE,
    "date": _DATE_RE,
    "version": _VERSION_RE,
}


def _is_entity(token: str) -> bool:
    stripped = token.strip()
    return len(stripped) >= 3 and stripped.lower() not in STOPWORDS


def _looks_technical(token: str) -> bool:
    """Hard entities are identifiers, not ordinary vocabulary.

    Accepting any 5+ letter lowercase word turned "cache"/"embeddings" into a hard
    requirement and produced false evidence-gap verdicts.
    """

    return bool(re.search(r"[A-Z]", token)) or "_" in token or "-" in token or "." in token or any(ch.isdigit() for ch in token)


def analyze_query(query: str) -> QuerySpec:
    lowered = query.lower()
    answer_type = "entity"
    for candidate, cues in _ANSWER_TYPE_CUES:
        if any(cue in lowered for cue in cues):
            answer_type = candidate
            break

    # Hard entities are things that must appear verbatim: quoted spans and technical
    # tokens. Long CJK segments are phrase fragments, so they are soft terms instead -
    # treating them as hard entities produced false "evidence gap" verdicts on 4/10
    # answerable questions in examples/eval.jsonl.
    entities: list[str] = []
    terms: list[str] = []
    for span in _QUOTED_RE.findall(query):
        if _is_entity(span):
            entities.append(span.strip())
    for token in _CODE_TOKEN_RE.findall(query):
        if _is_entity(token) and _looks_technical(token):
            entities.append(token)
    for run in _CJK_RUN_RE.findall(query):
        for piece in split_cjk_run(run):
            if _is_entity(piece):
                terms.append(piece)

    def dedupe(items: list[str]) -> tuple[str, ...]:
        kept: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item.lower() in seen:
                continue
            seen.add(item.lower())
            kept.append(item)
        return tuple(kept[:8])

    return QuerySpec(
        query=query,
        answer_type=answer_type,
        entities=dedupe(entities),
        terms=dedupe(terms),
        key_terms=content_tokens(query),
    )


def has_expected_slot(text: str, answer_type: str) -> bool:
    """True when the passage carries the surface shape of the requested fact.

    Types with no declared shape (plain "which/who" questions) are unconstrained, so
    entity coverage alone decides them.
    """

    shape = _FACT_SHAPE_RE.get(answer_type)
    markers = _SLOT_MARKERS.get(answer_type)
    if shape is None and not markers:
        return True
    if shape is not None and shape.search(text):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in (markers or ()))


# Interrogatives and function words only. Topic nouns (e.g. a version cue like
# "版本号") must not appear here or they cut the noun phrase they belong to.
_SPLIT_WORDS = tuple(
    sorted(
        {
            "什么", "为什么", "是什么", "什么是", "什么时候", "何时", "哪年", "几月", "日期",
            "多少", "几个", "几条", "几次", "数量", "总数", "哪个", "哪些", "哪里", "谁",
            "如何", "怎么", "怎样", "为何", "因为", "由于", "原因", "导致", "能否", "是否",
            "能不能", "可不可以", "可以", "定义", "含义", "区别", "差异", "对比", "更好",
            "更快", "步骤", "方法", "流程", "进行", "通过", "以及", "并且", "但是", "所以",
            "因此", "然而",
        }
        | {word for word in ZH_STOPWORDS if len(word) >= 2}
        | {"的", "会", "是", "了", "把", "被", "和", "与", "就", "都", "而", "或", "要", "让", "使", "从", "到", "给"},
        key=len,
        reverse=True,
    )
)


def split_cjk_run(run: str) -> list[str]:
    """Cut a long CJK question phrase into noun-ish segments on cue/stopword seams."""

    segments: list[str] = []
    buffer: list[str] = []
    index = 0
    while index < len(run):
        seam = next((word for word in _SPLIT_WORDS if run.startswith(word, index)), None)
        if seam is None:
            buffer.append(run[index])
            index += 1
            continue
        if buffer:
            segments.append("".join(buffer))
            buffer = []
        index += len(seam)
    if buffer:
        segments.append("".join(buffer))
    return [segment for segment in segments if 3 <= len(segment) <= 12]

class EvidenceGapChecker:
    """Scores whether the retrieved units actually carry the fact the question asks for."""

    def __init__(self, config: EvidenceConfig | None = None) -> None:
        self.config = config or EvidenceConfig()

    def check(self, query: str, units: Sequence[RetrievedUnit]) -> EvidenceReport:
        spec = analyze_query(query)
        if not self.config.enabled or not units:
            return EvidenceReport(spec=spec, answerable=bool(units), support=1.0 if units else 0.0)

        query_tokens = set(spec.key_terms)
        best_support = 0.0
        supporting = 0
        seen_entities: set[str] = set()
        seen_terms: set[str] = set()
        any_slot = False
        per_unit: list[dict[str, Any]] = []

        for unit in units:
            text = unit.best_text
            lowered = text.lower()
            hits = [entity for entity in spec.entities if entity.lower() in lowered]
            seen_entities.update(entity.lower() for entity in hits)
            term_hits = [term for term in spec.terms if term in text]
            seen_terms.update(term_hits)
            slot = has_expected_slot(text, spec.answer_type)
            any_slot = any_slot or slot
            entity_cov = (len(hits) / len(spec.entities)) if spec.entities else 1.0
            term_cov = (len(term_hits) / len(spec.terms)) if spec.terms else 1.0
            lexical = len(query_tokens & set(tokenize(text))) / max(len(query_tokens), 1)
            support = (
                0.35 * entity_cov
                + 0.20 * term_cov
                + 0.30 * (1.0 if slot else 0.0)
                + 0.15 * lexical
            )
            best_support = max(best_support, support)
            # With no hard identifier to anchor on, lexical overlap is the fallback anchor;
            # requiring a verbatim CJK phrase match produced false refusals.
            primary_cov = entity_cov if spec.entities else term_cov
            anchored = primary_cov >= self.config.entity_coverage_min or lexical >= 0.25
            if anchored and slot:
                supporting += 1
            per_unit.append(
                {
                    "chunk_id": unit.chunk.chunk_id,
                    "entity_coverage": round(entity_cov, 3),
                    "term_coverage": round(term_cov, 3),
                    "slot": slot,
                    "support": round(support, 3),
                }
            )

        entity_coverage = (len(seen_entities) / len(spec.entities)) if spec.entities else 1.0
        term_coverage = (len(seen_terms) / len(spec.terms)) if spec.terms else 1.0
        missing_entities = tuple(e for e in spec.entities if e.lower() not in seen_entities)
        answerable = best_support >= self.config.support_threshold and supporting > 0
        return EvidenceReport(
            spec=spec,
            answerable=answerable,
            support=best_support,
            entity_coverage=entity_coverage,
            term_coverage=term_coverage,
            slot_found=any_slot,
            supporting_units=supporting,
            missing_entities=missing_entities,
            missing_slots=() if any_slot else (spec.answer_type,),
            per_unit=per_unit[:12],
        )


def gap_terms(query: str, units: Sequence[RetrievedUnit], report: EvidenceReport, limit: int = 8) -> list[str]:
    """Terms the question needs but no candidate contains - fuel for query rewriting."""

    present = {term for unit in units for term in tokenize(unit.best_text)}
    missing = [term for term in report.spec.key_terms if term not in present]
    soft = [term for term in report.spec.terms if term not in {t for u in units for t in (u.best_text,)}]
    return (list(report.missing_entities) + soft + missing)[:limit]