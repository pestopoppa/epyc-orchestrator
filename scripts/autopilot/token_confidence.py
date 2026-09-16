"""EV-CONF-2: per-token confidence capture and candidate confidence sources.

Why this exists
---------------
E7c measured math AUROC ~0.40 on both arms with the sequence-level
completion-probability geomean as confidence (ESC-7 draft section 5): the
geomean is anti-discriminative on math. The leading hypothesis is length
confounding. Long, correct derivations collect many low-probability tokens,
while short, confidently wrong answers score high. Testing a better source needs
the per-token vector, but until this module the eval sidecar persisted only the
aggregate. Offline re-scoring of E7c is therefore impossible.

The module has two halves:

1. **Capture (write side).** :func:`build_token_trace_record` compresses a
   llama.cpp ``completion_probabilities`` list (both the OpenAI
   ``{token, logprob, top_logprobs}`` shape and the legacy
   ``{content, probs:[{tok_str, prob}]}`` shape) into a bounded, JSON-safe
   record:

   * ``lp``: the sampled token's log-probability per token. Under the
     production deterministic profile this is the top-1 logprob. The encoding
     is ``uint16`` little-endian millinats of ``-logprob``, clamped to
     ``[0, 65.535]`` nats and base64-encoded, so each token costs 2 bytes.
   * ``ent``: the truncated top-k entropy per token in the same encoding, present
     only when the backend returned top-k alternatives (see
     :func:`topk_entropy`).
   * ``answer_span``: ``[start, end)`` token indices of the final answer. They
     are located by :func:`locate_answer_span`, which reuses the SCORE-03/16
     extractors in ``debug_scorer``.
   * Exact summaries computed before any truncation:
     ``full_mean_logprob`` (legacy parity), plus ``real_mean_logprob``,
     ``real_min_logprob`` and ``real_min_index``, which cover measured tokens
     only.
   * Speculative-decoding placeholders. The llama.cpp server (v7 and v9)
     reports ``prob=1.0`` with no top-k for every draft-accepted token. When
     top-k was requested, a token with ``logprob == 0`` and no alternatives is
     stored as the ``missing`` sentinel and counted in ``n_placeholder``.
     Under MTP serving this is most tokens, and the legacy geomean then
     collapses toward 1.0. The E7c sidecar shows 1335/1684 worker_general
     rows at ``confidence == 1.0``, which makes the length-confounding story
     suspect. The probe must therefore run with speculative decoding OFF.
   * Size bound: at most ``max_tokens`` tokens are stored (head quarter and
     tail remainder, since the answer sits at the end). ``kept`` lists the
     retained full-index ranges.

2. **Sources (read side).** Pure functions over a decoded :class:`TokenTrace`
   compute the EV-CONF-2 candidates: the full geomean (parity with the legacy
   ``confidence``), the ANSWER-SPAN geomean, the salient-token min-prob, and a
   high-entropy-token confidence. Each returns ``None`` when its input is
   unavailable, and never a fabricated placeholder.

No grading rule lives here. These are confidence *sources*, and AUROC/ECE are
computed by ``src.llm_primitives.stat_tests`` in the offline comparison tool
(``scripts/analysis/confidence_source_compare.py``).
"""

from __future__ import annotations

import base64
import math
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_TRACE_SCHEMA = "epyc.token_logprobs.v1"
ENCODING = "u16le_millinat_b64"
DEFAULT_MAX_TOKENS = 8192
_SCALE = 1000.0
_U16_MAX = 65535
# Sentinel for a token whose probability the backend did not compute. The
# llama.cpp server emits ``prob = 1.0`` with no top-k for every token accepted
# through speculative decoding (``server-context.cpp``: ``result.prob = 1.0f;
# // set later`` / ``// TODO: set result.probs``, present in v7 and v9).
_MISSING = _U16_MAX
_U16_REAL_MAX = _U16_MAX - 1
_PROB_FLOOR = 1e-12

_BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"


# ── encoding ──────────────────────────────────────────────────────────────


def encode_nonneg(values: Sequence[float | None]) -> str:
    """Encode non-negative floats (nats) as base64 uint16-LE millinats (``None`` -> sentinel)."""
    out = bytearray()
    for v in values:
        if v is None:
            q = _MISSING
        else:
            q = min(_U16_REAL_MAX, int(round(max(0.0, float(v)) * _SCALE)))
        out += struct.pack("<H", q)
    return base64.b64encode(bytes(out)).decode("ascii")


def decode_nonneg(blob: str) -> list[float | None]:
    raw = base64.b64decode(blob.encode("ascii"))
    if len(raw) % 2:
        raise ValueError("token trace blob has odd byte length")
    return [None if q == _MISSING else q / _SCALE for (q,) in struct.iter_unpack("<H", raw)]


# ── row parsing ───────────────────────────────────────────────────────────


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _candidate_probs(row: Mapping[str, Any]) -> list[float]:
    cands = row.get("top_logprobs") or row.get("probs") or []
    probs: list[float] = []
    if not isinstance(cands, list):
        return probs
    for c in cands:
        if not isinstance(c, Mapping):
            continue
        lp = _finite(c.get("logprob"))
        p = math.exp(lp) if lp is not None else _finite(c.get("probability", c.get("prob")))
        if p is not None:
            probs.append(min(1.0, max(0.0, p)))
    return probs


def _row_logprob(row: Mapping[str, Any]) -> float | None:
    """Sampled-token logprob. The selection order mirrors
    ``eval_tower._completion_probabilities_confidence`` exactly, so the full
    geomean reproduces the legacy aggregate."""
    prob = _finite(row.get("probability", row.get("prob")))
    logprob = _finite(row.get("logprob"))
    if prob is None:
        candidates = row.get("probs") or row.get("top_logprobs") or []
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            if isinstance(first, Mapping):
                prob = _finite(first.get("probability", first.get("prob")))
                logprob = _finite(first.get("logprob"))
    if prob is None and logprob is not None:
        prob = math.exp(logprob)
    if prob is None:
        return None
    return math.log(min(1.0, max(_PROB_FLOOR, prob)))


def _row_text_bytes(row: Mapping[str, Any]) -> bytes:
    raw = row.get("bytes")
    if isinstance(raw, list) and all(isinstance(b, int) and 0 <= b < 256 for b in raw):
        return bytes(raw)
    text = row.get("token", row.get("content", row.get("tok_str", "")))
    return str(text or "").encode("utf-8")


def topk_entropy(probs: Sequence[float]) -> float | None:
    """Truncated top-k entropy in nats, with the residual mass lumped into one bucket.

    ``H_k = -sum_i p_i ln p_i - r ln r`` where ``r = 1 - sum_i p_i``. This is a
    **lower bound** on the true next-token entropy, and it is exact only when
    k covers the whole support. It needs top-k alternatives from the backend
    (``n_probs`` / ``top_logprobs`` >= 2). With k <= 1 it returns ``None``,
    because a single probability carries no spread information beyond
    ``-ln p``.
    """
    ps = [min(1.0, max(0.0, float(p))) for p in probs]
    if len(ps) < 2:
        return None
    total = sum(ps)
    if total > 1.0:
        ps = [p / total for p in ps]
        total = 1.0
    h = -sum(p * math.log(p) for p in ps if p > 0.0)
    r = 1.0 - total
    if r > 1e-12:
        h -= r * math.log(r)
    return max(0.0, h)


# ── answer span (SCORE-03 / SCORE-16 extractors) ──────────────────────────


def _debug_scorer():
    if str(_BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(_BENCH_DIR))
    import debug_scorer  # type: ignore[import-not-found]

    return debug_scorer


def locate_answer_span(
    text: str, extract_pattern: str | None = None
) -> tuple[int, int, str] | None:
    """Return ``(char_start, char_end, source)`` for the final-answer payload.

    Precedence mirrors ``debug_scorer._score_exact_match``:

    1. The configured ``extract_pattern``, or ``<answer>...</answer>`` by default.
    2. The legacy ``#### value`` pattern.
    3. The last ``\\boxed{...}``, with nested braces handled by SCORE-16's
       ``_extract_boxed_answer``.
    4. The SCORE-03 ``_final_answer_region`` line.

    Every located span is checked against the extractor's own output, so a
    span can never disagree with what the scorer would read.
    """
    if not text:
        return None
    ds = _debug_scorer()
    for pattern, source in (
        (extract_pattern or r"<answer>(.*?)</answer>", "extract_pattern"),
        (r"####[ \t]*\n?(\S+)", "hash_marker"),
    ):
        try:
            compiled = ds._compile_single_group_pattern(pattern)
        except (ValueError, re.error):
            continue
        m = compiled.search(text)
        if m and m.group(1) and m.group(1).strip():
            s, e = m.span(1)
            inner = text[s:e]
            s += len(inner) - len(inner.lstrip())
            e -= len(inner) - len(inner.rstrip())
            return (s, e, source)
    boxed = ds._extract_boxed_answer(text)
    if boxed is not None and boxed:
        start = text.rfind("\\boxed{") + len("\\boxed{")
        idx = text.find(boxed, start)
        if idx >= 0:
            return (idx, idx + len(boxed), "boxed")
    region = ds._final_answer_region(text)
    if region:
        idx = text.rfind(region)
        if idx >= 0:
            return (idx, idx + len(region), "final_answer_region")
    return None


def _char_span_to_token_span(
    token_bytes: Sequence[bytes], text: str, span: tuple[int, int]
) -> tuple[int, int] | None:
    b_start = len(text[: span[0]].encode("utf-8"))
    b_end = len(text[: span[1]].encode("utf-8"))
    first = last = None
    offset = 0
    for i, tb in enumerate(token_bytes):
        t_start, t_end = offset, offset + len(tb)
        offset = t_end
        if t_end <= b_start or t_start >= b_end or t_start == t_end:
            continue
        if first is None:
            first = i
        last = i
    if first is None or last is None:
        return None
    return (first, last + 1)


# ── capture ───────────────────────────────────────────────────────────────


def build_token_trace_record(
    rows: Any,
    *,
    answer: str = "",
    extract_pattern: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any] | None:
    """Compress probability rows into a bounded sidecar record, or ``None``.

    ``answer`` is used only for provenance. The span is located in the text
    rebuilt from the token rows, because the chat ``answer`` may differ from
    the generated token stream (reasoning split, post-processing).
    ``answer_text_match`` records how the two relate.
    """
    if not isinstance(rows, list) or not rows:
        return None
    lps: list[float] = []
    cand_counts: list[int] = []
    ents: list[float | None] = []
    tbytes: list[bytes] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lp = _row_logprob(row)
        if lp is None:
            continue
        cands = _candidate_probs(row)
        lps.append(lp)
        cand_counts.append(len(cands))
        ents.append(topk_entropy(cands))
        tbytes.append(_row_text_bytes(row))
    n = len(lps)
    if n == 0:
        return None

    # Placeholder detection is possible only when the backend returned top-k
    # alternatives for at least one token. A token with logprob exactly 0 and
    # no alternatives in such a stream is a speculative-decoding placeholder,
    # not a measured certainty.
    topk = max(cand_counts)
    detectable = topk > 0
    real = [not (detectable and cand_counts[i] == 0 and lps[i] == 0.0) for i in range(n)]
    n_real = sum(real)

    record: dict[str, Any] = {
        "schema": TOKEN_TRACE_SCHEMA,
        "encoding": ENCODING,
        "n_tokens": n,
        "topk": topk,
        "placeholder_detection": "topk_absent_and_logprob_zero" if detectable else "unavailable",
        "n_placeholder": n - n_real,
        # Legacy parity: the aggregate eval_tower stores as `confidence` (placeholders included).
        "full_mean_logprob": sum(lps) / n,
    }
    real_idx = [i for i in range(n) if real[i]]
    if real_idx:
        min_idx = min(real_idx, key=lambda i: lps[i])
        record["real_mean_logprob"] = sum(lps[i] for i in real_idx) / len(real_idx)
        record["real_min_logprob"] = lps[min_idx]
        record["real_min_index"] = min_idx

    have_ent = detectable and all(ents[i] is not None for i in real_idx)

    cap = max(8, int(max_tokens))
    if n <= cap:
        kept = [[0, n]]
        idx = list(range(n))
    else:
        head = cap // 4
        tail = cap - head
        kept = [[0, head], [n - tail, n]]
        idx = list(range(head)) + list(range(n - tail, n))
    record["kept"] = kept
    record["truncated"] = n > cap
    record["lp"] = encode_nonneg([-lps[i] if real[i] else None for i in idx])
    if have_ent:
        record["ent"] = encode_nonneg([ents[i] if real[i] else None for i in idx])

    joined = b"".join(tbytes)
    text = joined.decode("utf-8", errors="replace")
    clean = True
    try:
        joined.decode("utf-8")
    except UnicodeDecodeError:
        clean = False
    ans = str(answer or "").strip()
    if not ans:
        match = "no_answer"
    elif text.strip() == ans:
        match = "exact"
    elif text.rstrip().endswith(ans):
        match = "suffix"
    elif ans in text:
        match = "contains"
    else:
        match = "mismatch"
    record["answer_text_match"] = match

    span_rec: dict[str, Any] | None = None
    if clean:
        loc = locate_answer_span(text, extract_pattern)
        if loc is not None:
            tok_span = _char_span_to_token_span(tbytes, text, (loc[0], loc[1]))
            if tok_span is not None:
                span_rec = {"start": tok_span[0], "end": tok_span[1], "source": loc[2]}
    record["answer_span"] = span_rec
    if not clean:
        record["answer_span_unavailable"] = "non_utf8_token_stream"
    return record


# ── read side ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TokenTrace:
    """Decoded per-token trace. ``positions[i]`` is the full-sequence index of kept token i."""

    n_tokens: int
    positions: tuple[int, ...]
    logprobs: tuple[float | None, ...]  # None = placeholder (probability never computed)
    entropies: tuple[float | None, ...] | None
    answer_span: tuple[int, int] | None
    full_mean_logprob: float | None
    real_mean_logprob: float | None
    real_min_logprob: float | None
    n_placeholder: int
    truncated: bool

    @property
    def placeholder_fraction(self) -> float:
        return self.n_placeholder / self.n_tokens if self.n_tokens else 0.0


def decode_token_trace(record: Mapping[str, Any] | None) -> TokenTrace | None:
    """Decode a sidecar ``token_logprobs`` record. Returns ``None`` for legacy rows."""
    if not isinstance(record, Mapping) or record.get("schema") != TOKEN_TRACE_SCHEMA:
        return None
    if record.get("encoding") != ENCODING:
        raise ValueError(f"unsupported token trace encoding {record.get('encoding')!r}")
    lps = tuple(None if v is None else -v for v in decode_nonneg(str(record.get("lp", ""))))
    positions: list[int] = []
    for lo, hi in record.get("kept") or []:
        positions.extend(range(int(lo), int(hi)))
    if len(positions) != len(lps):
        raise ValueError("token trace kept ranges disagree with vector length")
    ents = None
    if record.get("ent") is not None:
        ents = tuple(decode_nonneg(str(record["ent"])))
        if len(ents) != len(lps):
            raise ValueError("token trace entropy/logprob length mismatch")
    span = record.get("answer_span")
    span_t = (int(span["start"]), int(span["end"])) if isinstance(span, Mapping) else None
    return TokenTrace(
        n_tokens=int(record.get("n_tokens", len(lps))),
        positions=tuple(positions),
        logprobs=lps,
        entropies=ents,
        answer_span=span_t,
        full_mean_logprob=_finite(record.get("full_mean_logprob")),
        real_mean_logprob=_finite(record.get("real_mean_logprob")),
        real_min_logprob=_finite(record.get("real_min_logprob")),
        n_placeholder=int(record.get("n_placeholder", 0) or 0),
        truncated=bool(record.get("truncated", False)),
    )


def _geomean(lps: Sequence[float]) -> float | None:
    if not lps:
        return None
    return min(1.0, max(0.0, math.exp(sum(lps) / len(lps))))


def _real(values: Sequence[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def full_geomean(trace: TokenTrace) -> float | None:
    """Sequence geomean with placeholders counted as p=1.

    This is exactly the legacy E7c source (placeholders included) and stays
    exact even when the vector was truncated.
    """
    if trace.full_mean_logprob is not None:
        return min(1.0, max(0.0, math.exp(trace.full_mean_logprob)))
    return _geomean([0.0 if v is None else v for v in trace.logprobs])


def real_token_geomean(trace: TokenTrace) -> float | None:
    """Geomean over tokens whose probability was actually computed (placeholders excluded)."""
    if trace.real_mean_logprob is not None:
        return min(1.0, max(0.0, math.exp(trace.real_mean_logprob)))
    return _geomean(_real(trace.logprobs))


def _span_logprobs(trace: TokenTrace) -> list[float] | None:
    if trace.answer_span is None:
        return None
    s, e = trace.answer_span
    kept = {p: i for i, p in enumerate(trace.positions)}
    if any(p not in kept for p in range(s, e)):
        return None  # span fell in a truncated gap, so refuse rather than guess
    return _real(trace.logprobs[kept[p]] for p in range(s, e)) or None


def answer_span_geomean(trace: TokenTrace) -> float | None:
    """Geomean over the final-answer tokens only. This directly removes length confounding."""
    lps = _span_logprobs(trace)
    return _geomean(lps) if lps else None


def answer_span_min_prob(trace: TokenTrace) -> float | None:
    """The weakest token inside the answer span."""
    lps = _span_logprobs(trace)
    return math.exp(min(lps)) if lps else None


def salient_min_prob(trace: TokenTrace) -> float | None:
    """Min sampled-token probability over the whole generation (the single weakest decision).

    Uses the exact write-time minimum over real tokens, so it is unaffected by
    truncation. Placeholders are excluded.
    """
    if trace.real_min_logprob is not None:
        return math.exp(trace.real_min_logprob)
    real = _real(trace.logprobs)
    return math.exp(min(real)) if real else None


def high_entropy_confidence(trace: TokenTrace, *, top_frac: float = 0.1) -> float | None:
    """Mean sampled-token probability over the ``top_frac`` highest-entropy tokens.

    The high-entropy tokens are the decision points where the model actually
    chose between alternatives. Easy boilerplate tokens are excluded, which is
    the salient-token idea under a different selection rule. Requires the
    ``ent`` vector (top-k >= 2 at capture), otherwise ``None``. Over a truncated
    vector it uses only the kept tokens.
    """
    if trace.entropies is None:
        return None
    cand = [
        i
        for i, (e, lp) in enumerate(zip(trace.entropies, trace.logprobs))
        if e is not None and lp is not None
    ]
    if not cand:
        return None
    k = max(1, int(math.ceil(len(cand) * float(top_frac))))
    order = sorted(cand, key=lambda i: -trace.entropies[i])[:k]  # type: ignore[index,operator]
    return sum(math.exp(trace.logprobs[i]) for i in order) / len(order)  # type: ignore[arg-type]


def neg_mean_entropy_confidence(trace: TokenTrace) -> float | None:
    """``exp(-mean truncated entropy)`` in (0, 1]. A DeepConf-style whole-trace certainty."""
    ents = _real(trace.entropies or ())
    if not ents:
        return None
    return math.exp(-sum(ents) / len(ents))


CANDIDATE_SOURCES = {
    "full_geomean": full_geomean,
    "real_token_geomean": real_token_geomean,
    "answer_span_geomean": answer_span_geomean,
    "answer_span_min_prob": answer_span_min_prob,
    "salient_min_prob": salient_min_prob,
    "high_entropy_top10_prob": high_entropy_confidence,
    "neg_mean_entropy": neg_mean_entropy_confidence,
}
