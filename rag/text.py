"""Tokenizer and sentence segmentation that work for Chinese and English."""

from __future__ import annotations

import re
from functools import lru_cache

_CJK_RANGES = (
    (0x2E80, 0x2EFF),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF66, 0xFF9F),
)

_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*|\d+(?:\.\d+)?")
_SENTENCE_ENDS = "。！？；!?;\n"
_ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "fig", "no", "dr", "mr", "mrs", "st", "approx"}

EN_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in is it its of on or
    that the their them they this to was were will with we you your not can could would should
    which what when who how all any some such than too very about into over under between
    do does did been being our us me my""".split()
)

ZH_STOPWORDS = frozenset(
    "的 了 和 是 在 我 有 也 就 不 人 都 一 与 及 或 但 而 于 其 这 那 他 她 它 们 个 上 下 中 里 为 以 被 能 会 可以 什么 怎么 如何 哪些 哪个 以及 并且 因为 所以 但是 然而 因此 通过 进行 以及".split()
)

STOPWORDS = EN_STOPWORDS | ZH_STOPWORDS


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def normalize(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text


@lru_cache(maxsize=40000)
def tokenize(text: str, cjk_mode: str = "bigram") -> tuple[str, ...]:
    """Mixed-script tokenizer.

    Latin text -> word tokens. CJK runs -> character bigrams (Lucene CJKBigramFilter
    behaviour) because Chinese has no whitespace boundaries and unigrams are too noisy
    for BM25. ``cjk_mode="uni_bi"`` also emits unigrams, which densifies embeddings.
    """

    text = normalize(text).lower()
    tokens: list[str] = []
    run: list[str] = []

    def flush_run() -> None:
        if not run:
            return
        if len(run) == 1:
            tokens.append(run[0])
        else:
            if cjk_mode == "uni_bi":
                tokens.extend(run)
            tokens.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
        run.clear()

    index = 0
    length = len(text)
    while index < length:
        ch = text[index]
        if is_cjk(ch):
            run.append(ch)
            index += 1
            continue
        match = _LATIN_TOKEN_RE.match(text, index)
        if match:
            flush_run()
            tokens.append(match.group(0))
            index = match.end()
            continue
        flush_run()
        index += 1

    flush_run()
    return tuple(tokens)


def content_tokens(text: str, cjk_mode: str = "bigram") -> tuple[str, ...]:
    return tuple(tok for tok in tokenize(text, cjk_mode) if tok not in STOPWORDS and len(tok) > 1)


@lru_cache(maxsize=20000)
def count_tokens(text: str) -> int:
    """Script-aware token estimate used to size chunks."""

    latin = len(_LATIN_TOKEN_RE.findall(text))
    cjk = sum(1 for ch in text if is_cjk(ch))
    return latin + cjk


@lru_cache(maxsize=20000)
def split_sentences(text: str) -> tuple[str, ...]:
    sentences: list[str] = []
    buffer: list[str] = []
    for raw_line in normalize(text).split("\n"):
        line = raw_line.strip()
        if not line:
            if buffer:
                sentences.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(line)
        last = line[-1]
        if last in _SENTENCE_ENDS:
            joined = " ".join(buffer)
            sentences.extend(_split_inline(joined))
            buffer = []
    if buffer:
        sentences.extend(_split_inline(" ".join(buffer)))
    return tuple(s for s in sentences if s.strip())


def _split_inline(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for token in re.split(r"([。！？；!?;])", text):
        if token in "。！？；!?;":
            current.append(token)
            candidate = "".join(current).strip()
            stem = candidate.rstrip("。！？；!?;").split()
            if not stem:
                current = []
                continue
            if stem[-1].lower().rstrip(".") in _ABBREVIATIONS and token in "!?;":
                continue
            if candidate:
                parts.append(candidate)
            current = []
        elif token:
            current.append(token)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def char_ngrams(text: str, n: int = 3) -> tuple[str, ...]:
    compact = re.sub(r"\s+", "", normalize(text).lower())
    if len(compact) < n:
        return (compact,) if compact else ()
    return tuple(compact[i : i + n] for i in range(len(compact) - n + 1))


def lexical_overlap(query_tokens: tuple[str, ...], text: str) -> float:
    """Coverage of query tokens in a passage, robust for CJK bigram tokens."""

    if not query_tokens:
        return 0.0
    doc_tokens = set(tokenize(text))
    hits = sum(1 for tok in query_tokens if tok in doc_tokens)
    return hits / len(query_tokens)