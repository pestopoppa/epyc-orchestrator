"""Embedding-drift excerpt selection for the coherence judge (``excerpt_mode="embed_drift"``).

Operator direction 2026-10-04 ("could a smart use of the embedders be leveraged here"):
approved as an OPTION next to the default ``head_tail_divergence`` excerpt; calibration
A/Bs the two and decides the default later.

When an output is over the judged-token cap, the default mode shows head + the window at
tier 0's first divergence + tail. ``embed_drift`` instead shows the passages where base
and candidate DRIFT APART semantically, measured on the BGE embedder pool:

1. **Chunker** (:func:`chunk_text`) — deterministic, sentence/line-bounded chunks of about
   :data:`CHUNK_TARGET_TOKENS` (hard max :data:`CHUNK_MAX_TOKENS`, well under BGE's 512).
   Atoms end at a newline or at ``.!?;:`` followed by blanks; atoms are packed greedily and
   a chunk also closes at a paragraph break (blank line) once it holds
   :data:`CHUNK_MIN_TOKENS`, so boundaries are content-defined and identical passages
   chunk identically. Both outputs are chunked from offset 0 with the SAME chars-per-token
   ratio (the base's), so the base chunks are candidate-independent (embedding-cache hits
   across candidates) and the shared prefix chunks byte-identically on both sides.
2. **Alignment** (:func:`align`) — ``difflib.SequenceMatcher`` over the chunk TEXTS
   (``autojunk=False``): ``equal`` blocks (the tier-0 identical prefix, and any passage
   where the outputs re-synchronise) cost nothing — similarity 1.0, never embedded.
   ``insert``/``delete`` blocks are unmatched chunks — maximal drift (similarity 0.0),
   never embedded. Only ``replace`` blocks are embedded; inside one, base and candidate
   chunks are paired by a monotone DP that maximises the sum of ``cosine - PAIR_FLOOR``
   (a pair below the floor is better left unmatched). Leftover chunks — a truncated
   candidate's missing tail, a longer candidate's extra tail — are unmatched, maximal
   drift.
3. **Selection** (:func:`select_units`) — per-side budget = the cap for an over-cap side,
   unlimited for a side that fits (it is shown whole, as in the default mode). First,
   per side, a small head (:data:`HEAD_SHARE` of the cap, at least the first chunk), then
   the unit holding each side's LAST chunk (so the judge sees where each output ends — a
   truncated or looping end is a verdict signal), each when it fits (a pair unit costs
   both sides; with a cap of a few chunks the ends can crowd out the drift — at the
   default 1536-token cap they take ~200 tokens per side). Then units by ascending similarity
   (ties: document order) while both side budgets allow; units at or above
   :data:`NEAR_IDENTICAL_SIM` are never selected for drift (they say nothing the base does
   not). Whatever is not selected is elided, in document order, with
   ``[... N tokens elided, similarity ≥ x ...]`` (x = the lowest similarity elided there).

Consequence for prefill (documented, measured by the calibration roll-up): the base
excerpt now DEPENDS ON THE CANDIDATE (which base passages are shown depends on where the
candidate drifts), so the KV prefix shared across candidates of one base stops at the
reference's head instead of running through the whole reference. Mitigations kept: the
fixed head, rubric, prompt and the reference's head chunk(s) are still candidate-
independent; the base CHUNKS (and so their cached embeddings) are candidate-independent.

This module is pure: no I/O, no pool access. The caller passes ``embed_fn``; any failure
raises :class:`DriftUnavailable` and the caller falls back to the default mode and
records why (never the full text, never nothing).
"""

from __future__ import annotations

import difflib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

HEAD_TAIL_DIVERGENCE = "head_tail_divergence"
EMBED_DRIFT = "embed_drift"
EXCERPT_MODES = (HEAD_TAIL_DIVERGENCE, EMBED_DRIFT)
DEFAULT_EXCERPT_MODE = HEAD_TAIL_DIVERGENCE

CHUNK_TARGET_TOKENS = 96
CHUNK_MAX_TOKENS = 128
CHUNK_MIN_TOKENS = 32
#: Share of the per-side cap reserved for the always-shown head.
HEAD_SHARE = 0.1
#: In a replace block, a base/candidate pair below this cosine is left unmatched.
PAIR_FLOOR = 0.5
#: Units at or above this cosine are never selected as drift.
NEAR_IDENTICAL_SIM = 0.99
#: Most chunk texts one call may embed (~25k tokens of divergent text); above it the
#: excerpt falls back to the default mode (``too_many_chunks``), recorded, so a 400k-char
#: request never floods the pool nor runs an O(n*m) alignment of thousands of chunks.
MAX_EMBED_CHUNKS = 256
#: Selected-unit detail kept on the verdict (the rest are counted, not listed).
MAX_RECORDED_SELECTIONS = 64

DRIFT_ELISION = "\n[... {n} tokens elided, similarity ≥ {sim} ...]\n"

#: Everything that shapes an embed_drift excerpt; part of the prompt-template identity.
PARAMS = {
    "chunk_target_tokens": CHUNK_TARGET_TOKENS,
    "chunk_max_tokens": CHUNK_MAX_TOKENS,
    "chunk_min_tokens": CHUNK_MIN_TOKENS,
    "head_share": HEAD_SHARE,
    "pair_floor": PAIR_FLOOR,
    "near_identical_sim": NEAR_IDENTICAL_SIM,
    "max_embed_chunks": MAX_EMBED_CHUNKS,
    "elision": DRIFT_ELISION,
    "aligner": "difflib.SequenceMatcher(autojunk=False)+monotone-dp",
}

_ATOM_END = re.compile(r"\n+|(?<=[.!?;:])[ \t]+")

#: texts -> (N, D) array of vectors (any norm); raise DriftUnavailable on failure.
EmbedFn = Callable[[Sequence[str]], Any]


class DriftUnavailable(Exception):
    """embed_drift could not run; the caller falls back to the default mode."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}{': ' + detail if detail else ''}")
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _atoms(text: str) -> list[tuple[int, int, bool]]:
    """(start, end, ends_paragraph) atoms covering ``text`` exactly."""
    out: list[tuple[int, int, bool]] = []
    cursor = 0
    for m in _ATOM_END.finditer(text):
        end = m.end()
        if end <= cursor:
            continue
        out.append((cursor, end, m.group().count("\n") >= 2))
        cursor = end
    if cursor < len(text):
        out.append((cursor, len(text), False))
    return out


def _hard_split(text: str, a: int, b: int, max_chars: int) -> list[tuple[int, int]]:
    """Split [a, b) into pieces <= max_chars, preferring the last blank before the limit."""
    out: list[tuple[int, int]] = []
    while b - a > max_chars:
        cut = text.rfind(" ", a + max_chars // 2, a + max_chars)
        cut = cut + 1 if cut > a else a + max_chars
        out.append((a, cut))
        a = cut
    if b > a:
        out.append((a, b))
    return out


def chunk_text(text: str, chars_per_token: float) -> list[tuple[int, int]]:
    """Deterministic chunk spans covering ``text`` (see the module docstring)."""
    if not text:
        return []
    cpt = chars_per_token if chars_per_token > 0 and math.isfinite(chars_per_token) else 4.0
    target = max(1, int(CHUNK_TARGET_TOKENS * cpt))
    max_chars = max(target, int(CHUNK_MAX_TOKENS * cpt))
    min_chars = max(1, int(CHUNK_MIN_TOKENS * cpt))
    chunks: list[tuple[int, int]] = []
    start: int | None = None
    end = 0

    def flush() -> None:
        nonlocal start
        if start is not None and end > start:
            chunks.append((start, end))
        start = None

    for a, b, para in _atoms(text):
        if b - a > max_chars:
            flush()
            chunks.extend(_hard_split(text, a, b, max_chars))
            continue
        if start is not None and b - start > max_chars:
            flush()
        if start is None:
            start = a
        end = b
        size = end - start
        if (para and size >= min_chars) or size >= target:
            flush()
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    """One aligned position: a base chunk, a candidate chunk, or both."""

    index: int
    base: tuple[int, int] | None
    candidate: tuple[int, int] | None
    kind: str  # equal | pair | base_only | candidate_only
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.index,
            "kind": self.kind,
            "similarity": round(self.similarity, 4),
            "base": list(self.base) if self.base else None,
            "candidate": list(self.candidate) if self.candidate else None,
        }


def _unit_vectors(raw: Any, n: int) -> np.ndarray:
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        raise DriftUnavailable("embedding_invalid", f"{type(exc).__name__}: {exc}") from exc
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] == 0 or not np.all(np.isfinite(arr)):
        raise DriftUnavailable("embedding_invalid", f"expected ({n}, D) finite, got {getattr(arr, 'shape', None)}")
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms <= 1e-9):
        raise DriftUnavailable("embedding_invalid", "zero-norm vector")
    return arr / norms[:, None]


def _pair_block(sims: np.ndarray) -> list[tuple[int | None, int | None]]:
    """Monotone alignment of a replace block maximising sum(cos - PAIR_FLOOR).

    Returns (i, j) steps in document order; None on one side = unmatched.
    """
    n, m = sims.shape
    gain = sims - PAIR_FLOOR
    score = np.zeros((n + 1, m + 1))
    move = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 diag, 2 up (base only), 3 left (cand only)
    move[1:, 0] = 2
    move[0, 1:] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, step = score[i - 1, j - 1] + gain[i - 1, j - 1], 1
            if score[i - 1, j] > best:  # strict: ties prefer pairing, then base-only
                best, step = score[i - 1, j], 2
            if score[i, j - 1] > best:
                best, step = score[i, j - 1], 3
            score[i, j], move[i, j] = best, step
    steps: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = move[i, j]
        if step == 1:
            steps.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == 2:
            steps.append((i - 1, None))
            i -= 1
        else:
            steps.append((None, j - 1))
            j -= 1
    steps.reverse()
    return steps


@dataclass
class Alignment:
    units: list[Unit]
    base_chunks: list[tuple[int, int]]
    candidate_chunks: list[tuple[int, int]]
    embedded_chunks: int = 0
    embed_ms: float = 0.0
    replace_blocks: int = 0


def align(
    base: str,
    candidate: str,
    *,
    chars_per_token: float,
    embed_fn: EmbedFn | None,
) -> Alignment:
    """Chunk both outputs, align them, embed ONLY replace blocks, return scored units."""
    bc = chunk_text(base, chars_per_token)
    cc = chunk_text(candidate, chars_per_token)
    btxt = [base[a:b] for a, b in bc]
    ctxt = [candidate[a:b] for a, b in cc]
    opcodes = difflib.SequenceMatcher(None, btxt, ctxt, autojunk=False).get_opcodes()
    blocks = [op for op in opcodes if op[0] == "replace"]
    texts: list[str] = []
    seen: dict[str, int] = {}
    for _, i1, i2, j1, j2 in blocks:
        for t in btxt[i1:i2] + ctxt[j1:j2]:
            if t not in seen:
                seen[t] = len(texts)
                texts.append(t)
    if len(texts) > MAX_EMBED_CHUNKS:
        raise DriftUnavailable("too_many_chunks", f"{len(texts)} divergent chunks > {MAX_EMBED_CHUNKS}")
    vectors = None
    embed_ms = 0.0
    if texts:
        if embed_fn is None:
            raise DriftUnavailable("no_embedder")
        started = time.perf_counter()
        try:
            raw = embed_fn(texts)
        except DriftUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any embedder failure is a recorded fallback
            raise DriftUnavailable("embedding_error", f"{type(exc).__name__}: {exc}") from exc
        embed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        vectors = _unit_vectors(raw, len(texts))
    units: list[Unit] = []

    def add(b: int | None, c: int | None, kind: str, sim: float) -> None:
        units.append(Unit(len(units), bc[b] if b is not None else None, cc[c] if c is not None else None, kind, sim))

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                add(i1 + k, j1 + k, "equal", 1.0)
        elif tag == "delete":
            for i in range(i1, i2):
                add(i, None, "base_only", 0.0)
        elif tag == "insert":
            for j in range(j1, j2):
                add(None, j, "candidate_only", 0.0)
        else:
            assert vectors is not None
            bv = vectors[[seen[t] for t in btxt[i1:i2]]]
            cv = vectors[[seen[t] for t in ctxt[j1:j2]]]
            sims = np.clip(bv @ cv.T, -1.0, 1.0)
            for i, j in _pair_block(sims):
                if i is not None and j is not None:
                    add(i1 + i, j1 + j, "pair", float(sims[i, j]))
                elif i is not None:
                    add(i1 + i, None, "base_only", 0.0)
                else:
                    add(None, j1 + j, "candidate_only", 0.0)  # type: ignore[operator]
    return Alignment(
        units=units,
        base_chunks=bc,
        candidate_chunks=cc,
        embedded_chunks=len(texts),
        embed_ms=embed_ms,
        replace_blocks=len(blocks),
    )


# ---------------------------------------------------------------------------
# Selection and rendering
# ---------------------------------------------------------------------------


@dataclass
class _Side:
    text: str
    tokens: int
    budget: float  # math.inf = shown whole

    def cost(self, span: tuple[int, int] | None) -> int:
        if span is None or not self.text:
            return 0
        return int(round(self.tokens * (span[1] - span[0]) / len(self.text)))


def select_units(units: Sequence[Unit], base: _Side, cand: _Side, cap: int) -> dict[int, str]:
    """unit index -> why selected (head | tail | drift)."""
    chosen: dict[int, str] = {}
    used = [0, 0]

    def fits(u: Unit) -> bool:
        return used[0] + base.cost(u.base) <= base.budget and used[1] + cand.cost(u.candidate) <= cand.budget

    def take(u: Unit, why: str) -> None:
        chosen[u.index] = why
        used[0] += base.cost(u.base)
        used[1] += cand.cost(u.candidate)

    # Head: each side's first chunk(s) up to HEAD_SHARE of the cap (the first always, when
    # it fits) — per side, so a pair diverging at byte 0 still shows both openings.
    head_budget = max(1, int(cap * HEAD_SHARE))
    for attr, side in (("base", base), ("candidate", cand)):
        spent = 0
        for u in units:
            span = getattr(u, attr)
            if span is None:
                continue
            c = side.cost(span)
            if spent and spent + c > head_budget:
                break
            if u.index not in chosen:
                if not fits(u):
                    break
                take(u, "head")
            spent += c
    for side in ("base", "candidate"):
        last = next((u for u in reversed(units) if getattr(u, side) is not None), None)
        if last is not None and last.index not in chosen and fits(last):
            take(last, "tail")
    for u in sorted(units, key=lambda x: (x.similarity, x.index)):
        if u.similarity >= NEAR_IDENTICAL_SIM or u.index in chosen:
            continue
        if fits(u):
            take(u, "drift")
    return chosen


def _floor2(x: float) -> str:
    return f"{math.floor(max(-1.0, min(1.0, x)) * 100.0) / 100.0:.2f}"


def _render_side(side: _Side, units: Sequence[Unit], chosen: dict[int, str], attr: str) -> tuple[str, list[list[int]], int]:
    """(rendered text, merged shown spans, shown chars) for one output."""
    if side.budget == math.inf:
        return side.text, [[0, len(side.text)]], len(side.text)
    parts: list[str] = []
    shown: list[list[int]] = []
    gap_chars = 0
    gap_sim = 1.0
    for u in units:
        span = getattr(u, attr)
        if span is None:
            continue
        if u.index in chosen:
            if gap_chars:
                parts.append(DRIFT_ELISION.format(n=side.cost((0, gap_chars)), sim=_floor2(gap_sim)))
                gap_chars, gap_sim = 0, 1.0
            parts.append(side.text[span[0] : span[1]])
            if shown and shown[-1][1] == span[0]:
                shown[-1][1] = span[1]
            else:
                shown.append([span[0], span[1]])
        else:
            gap_chars += span[1] - span[0]
            gap_sim = min(gap_sim, u.similarity)
    if gap_chars:
        parts.append(DRIFT_ELISION.format(n=side.cost((0, gap_chars)), sim=_floor2(gap_sim)))
    return "".join(parts), shown, sum(b - a for a, b in shown)


@dataclass
class DriftExcerpt:
    base: str
    candidate: str
    base_info: dict[str, Any] = field(default_factory=dict)
    candidate_info: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)


def drift_excerpt(
    base_text: str,
    cand_text: str,
    *,
    cap: int,
    base_tokens: int,
    cand_tokens: int,
    embed_fn: EmbedFn | None,
) -> DriftExcerpt:
    """Excerpt both outputs by embedding drift. Call only when an output is over ``cap``."""
    cpt = len(base_text) / base_tokens if base_tokens > 0 and base_text else 4.0
    alignment = align(base_text, cand_text, chars_per_token=cpt, embed_fn=embed_fn)
    base = _Side(base_text, base_tokens, float(cap) if base_tokens > cap else math.inf)
    cand = _Side(cand_text, cand_tokens, float(cap) if cand_tokens > cap else math.inf)
    chosen = select_units(alignment.units, base, cand, cap)
    rb, bspans, bchars = _render_side(base, alignment.units, chosen, "base")
    rc, cspans, cchars = _render_side(cand, alignment.units, chosen, "candidate")
    selected = [dict(u.to_dict(), why=chosen[u.index]) for u in alignment.units if u.index in chosen]
    elided = [u for u in alignment.units if u.index not in chosen]
    kinds: dict[str, int] = {}
    for u in alignment.units:
        kinds[u.kind] = kinds.get(u.kind, 0) + 1
    drift = {
        "chunks": {"base": len(alignment.base_chunks), "candidate": len(alignment.candidate_chunks)},
        "units": len(alignment.units),
        "unit_kinds": kinds,
        "replace_blocks": alignment.replace_blocks,
        "embedded_chunks": alignment.embedded_chunks,
        "embed_ms": alignment.embed_ms,
        "chars_per_token": round(cpt, 4),
        "selected_count": len(selected),
        "selected": selected[:MAX_RECORDED_SELECTIONS],
        "selected_truncated": len(selected) > MAX_RECORDED_SELECTIONS,
        "min_elided_similarity": round(min((u.similarity for u in elided), default=1.0), 4),
        "params": {k: v for k, v in PARAMS.items() if k not in ("elision",)},
    }

    def info(side: _Side, spans: list[list[int]], chars: int) -> dict[str, Any]:
        n = len(side.text)
        excerpted = side.budget != math.inf
        return {
            "excerpted": excerpted,
            "spans": spans,
            "judged_chars": chars,
            "judged_tokens": int(round(side.tokens * chars / n)) if n else 0,
            "divergence_window": None,
        }

    return DriftExcerpt(base=rb, candidate=rc, base_info=info(base, bspans, bchars),
                        candidate_info=info(cand, cspans, cchars), drift=drift)


__all__ = [
    "DEFAULT_EXCERPT_MODE",
    "DRIFT_ELISION",
    "DriftExcerpt",
    "DriftUnavailable",
    "EMBED_DRIFT",
    "EXCERPT_MODES",
    "HEAD_TAIL_DIVERGENCE",
    "PARAMS",
    "Unit",
    "align",
    "chunk_text",
    "drift_excerpt",
    "select_units",
]
