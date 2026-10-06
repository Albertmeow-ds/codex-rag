"""Chunking strategies: structural, semantic, and hierarchical (small-to-big)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from rag.text import count_tokens, split_sentences
from rag.types import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")


@dataclass(slots=True)
class ChunkingConfig:
    strategy: str = "structural"
    max_tokens: int = 320
    min_tokens: int = 40
    overlap_tokens: int = 48
    parent_tokens: int = 700
    child_tokens: int = 140
    semantic_break_percentile: float = 0.22
    include_headings_in_text: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _Section:
    heading_path: tuple[str, ...]
    start: int
    end: int
    body: str


def parse_sections(text: str) -> list[_Section]:
    """Split on Markdown headings while tracking the heading hierarchy."""

    lines = text.split("\n")
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    heading_at: dict[int, tuple[int, str]] = {}
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line.strip())
        if match:
            heading_at[i] = (len(match.group(1)), match.group(2).strip())

    if not heading_at:
        return [_Section((), 0, len(text), text)]

    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    boundaries: list[tuple[int, int, str, tuple[str, ...]]] = []

    for i, line in enumerate(lines):
        if i not in heading_at:
            continue
        level, title = heading_at[i]
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        boundaries.append((i, level, title, tuple(t for _, t in stack)))

    for idx, (line_no, level, _title, path) in enumerate(boundaries):
        start = offsets[line_no]
        end = len(text)
        for nxt_line_no, nxt_level, _nxt_title, _nxt_path in boundaries[idx + 1 :]:
            if nxt_level <= level:
                end = offsets[nxt_line_no]
                break
        body_lines = lines[line_no + 1 :]
        body_end_line = len(lines)
        for nxt_line_no, nxt_level, _t, _p in boundaries[idx + 1 :]:
            if nxt_level <= level:
                body_end_line = nxt_line_no
                break
        body = "\n".join(body_lines[: body_end_line - line_no - 1])
        sections.append(_Section(path, start, end, body))

    return [s for s in sections if s.body.strip()]


def _group_atomic(body: str) -> list[str]:
    """Sentence units, but fenced code blocks stay atomic."""

    units: list[str] = []
    buffer: list[str] = []
    in_fence = False
    for line in body.split("\n"):
        if _FENCE_RE.match(line.strip()):
            if in_fence:
                buffer.append(line)
                units.append("\n".join(buffer))
                buffer = []
                in_fence = False
            else:
                if buffer:
                    units.extend(split_sentences(" ".join(buffer)))
                    buffer = []
                buffer.append(line)
                in_fence = True
            continue
        buffer.append(line)
    if buffer:
        units.extend(split_sentences(" ".join(buffer)))
    out: list[str] = []
    for unit in units:
        unit = unit.strip()
        if unit:
            out.extend(_hard_split(unit))
    return out


def _hard_split(unit: str, hard_limit: int = 900) -> list[str]:
    if count_tokens(unit) <= hard_limit:
        return [unit]
    parts: list[str] = []
    current: list[str] = []
    budget = 0
    for token in unit.split(" "):
        cost = max(count_tokens(token), 1)
        if budget + cost > hard_limit and current:
            parts.append(" ".join(current))
            current = [token]
            budget = cost
        else:
            current.append(token)
            budget += cost
    if current:
        parts.append(" ".join(current))
    return parts


def _pack(units: Sequence[str], cfg: ChunkingConfig, overlap_tokens: int) -> list[tuple[str, int, int]]:
    """Pack sentence units into token-budgeted chunks with sentence-level overlap."""

    packed: list[tuple[str, int, int]] = []
    current: list[str] = []
    budget = 0
    for unit in units:
        cost = max(count_tokens(unit), 1)
        if budget + cost > cfg.max_tokens and current:
            packed.append((" ".join(current), 0, 0))
            tail: list[str] = []
            tail_budget = 0
            for prev in reversed(current):
                prev_cost = max(count_tokens(prev), 1)
                if tail_budget + prev_cost > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_budget += prev_cost
            current = tail
            budget = tail_budget
        current.append(unit)
        budget += cost
    if current:
        packed.append((" ".join(current), 0, 0))

    merged: list[tuple[str, int, int]] = []
    for text, _s, _e in packed:
        if merged and count_tokens(text) < cfg.min_tokens and count_tokens(merged[-1][0]) + count_tokens(text) <= cfg.max_tokens * 1.4:
            merged[-1] = (merged[-1][0] + " " + text, merged[-1][1], merged[-1][2])
        else:
            merged.append((text, 0, 0))
    return merged


def chunk_document(doc: Document, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkingConfig()
    if cfg.strategy == "hierarchical":
        return hierarchical_chunks(doc, cfg)
    if cfg.strategy == "semantic":
        return semantic_chunks(doc, cfg)
    return structural_chunks(doc, cfg)


def structural_chunks(doc: Document, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkingConfig()
    chunks: list[Chunk] = []
    for section in parse_sections(doc.text):
        units = _group_atomic(section.body)
        for ordinal_in_section, (text, _s, _e) in enumerate(_pack(units, cfg, cfg.overlap_tokens)):
            heading = section.heading_path[-1] if section.heading_path else doc.title
            body = text
            if cfg.include_headings_in_text and section.heading_path:
                body = " > ".join(section.heading_path) + "\n" + text
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#s{len(chunks)}",
                    doc_id=doc.doc_id,
                    text=body,
                    ordinal=len(chunks),
                    heading=heading,
                    heading_path=section.heading_path,
                    start=section.start,
                    end=section.end,
                    metadata={**doc.metadata, **cfg.metadata},
                )
            )
    return chunks


def hierarchical_chunks(doc: Document, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    """Small-to-big: index small children, surface large parents at assembly time."""

    cfg = cfg or ChunkingConfig()
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for section in parse_sections(doc.text):
        units = _group_atomic(section.body)
        parent_cfg = ChunkingConfig(max_tokens=cfg.parent_tokens, min_tokens=cfg.min_tokens, overlap_tokens=0)
        child_cfg = ChunkingConfig(max_tokens=cfg.child_tokens, min_tokens=20, overlap_tokens=cfg.overlap_tokens)
        header = " > ".join(section.heading_path)
        for parent_text, _s, _e in _pack(units, parent_cfg, 0):
            parent_id = f"{doc.doc_id}#p{len(parents)}"
            parents.append(
                Chunk(
                    chunk_id=parent_id,
                    doc_id=doc.doc_id,
                    text=(header + "\n" + parent_text) if header else parent_text,
                    ordinal=len(parents),
                    heading=section.heading_path[-1] if section.heading_path else doc.title,
                    heading_path=section.heading_path,
                    metadata={**doc.metadata, **cfg.metadata},
                )
            )
            child_units = _group_atomic(parent_text)
            for child_text, _cs, _ce in _pack(child_units, child_cfg, cfg.overlap_tokens):
                children.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#c{len(children)}",
                        doc_id=doc.doc_id,
                        text=(header + "\n" + child_text) if header else child_text,
                        ordinal=len(children),
                        heading=section.heading_path[-1] if section.heading_path else doc.title,
                        heading_path=section.heading_path,
                        parent_id=parent_id,
                        metadata={**doc.metadata, **cfg.metadata},
                    )
                )
    return children + parents


def semantic_chunks(doc: Document, cfg: ChunkingConfig | None = None, embedder: object = None) -> list[Chunk]:
    """Embedding-based breakpoint chunking (TextTiling-flavoured).

    Adjacent sentences are compared by cosine similarity; low-similarity boundaries
    become chunk breaks, bounded by min/max token guards.
    """

    cfg = cfg or ChunkingConfig()
    if embedder is None:
        from rag.embed.factory import build_embedder

        embedder = build_embedder("hashed", dim=256)

    chunks: list[Chunk] = []
    for section in parse_sections(doc.text):
        sentences = _group_atomic(section.body)
        if len(sentences) < 3:
            groups = [s for s in _pack(sentences, cfg, cfg.overlap_tokens)]
        else:
            vectors = embedder.embed(sentences)  # type: ignore[attr-defined]
            sims = [
                float(vectors[i] @ vectors[i + 1]) for i in range(len(vectors) - 1)
            ]
            threshold = _percentile(sims, cfg.semantic_break_percentile)
            groups: list[tuple[str, int, int]] = []
            current: list[str] = []
            budget = 0
            for i, sentence in enumerate(sentences):
                cost = max(count_tokens(sentence), 1)
                boundary = i < len(sims) and sims[i] <= threshold
                if current and (budget + cost > cfg.max_tokens or (boundary and budget >= cfg.min_tokens)):
                    groups.append((" ".join(current), 0, 0))
                    current = []
                    budget = 0
                current.append(sentence)
                budget += cost
            if current:
                groups.append((" ".join(current), 0, 0))

        header = " > ".join(section.heading_path)
        for text, _s, _e in groups:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#m{len(chunks)}",
                    doc_id=doc.doc_id,
                    text=(header + "\n" + text) if (header and cfg.include_headings_in_text) else text,
                    ordinal=len(chunks),
                    heading=section.heading_path[-1] if section.heading_path else doc.title,
                    heading_path=section.heading_path,
                    start=section.start,
                    end=section.end,
                    metadata={**doc.metadata, **cfg.metadata},
                )
            )
    return chunks


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(pct * (len(ordered) - 1)))))
    return ordered[idx]


def sentence_window(chunks: Sequence[Chunk], target: Chunk, radius: int = 2) -> str:
    """Sentence-window retrieval: widen a hit with its neighbours in the same doc."""

    same_doc = sorted((c for c in chunks if c.doc_id == target.doc_id), key=lambda c: c.ordinal)
    try:
        pos = next(i for i, c in enumerate(same_doc) if c.chunk_id == target.chunk_id)
    except StopIteration:
        return target.text
    lo = max(0, pos - radius)
    hi = min(len(same_doc), pos + radius + 1)
    seen: set[str] = set()
    parts: list[str] = []
    for chunk in same_doc[lo:hi]:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        parts.append(chunk.text)
    return "\n".join(parts)