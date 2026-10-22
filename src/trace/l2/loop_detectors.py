"""HIL-LB1: Locals' reply loop detectors, ported for offline scoring. Zero inference.

Source: Generated-Computers/Locals @0d39e16194e2c123ce10695e45619111eb0e45b0, ``locals/server.py``
``_looping`` (L2037), ``_line_looping`` (L2063), ``_past_grace``/``LOOP_GRACE`` (L2021-2034) and the
three threshold profiles chosen in ``Server._guard`` (L800-842) (intake-2386#record, claim 10). The
repeated-identical-call metric is intake-2395's process endpoint: same tool name and sorted-key JSON
arguments as an earlier call (intake-2395#record).

What is here, and what each detector reads:

* ``looping`` / ``looping_at`` — the token-period detector, a line-for-line port (including the
  short-period early return: the first period p < 8 whose 32-token tail matches decides the result).
  ``first_period_fire`` replays it STREAMING, once per token, as Locals calls ``_guard`` once per
  decoded token, with the optional grace window (``_past_grace``) of the profile.
* ``line_looping`` — the paraphrase detector: the last ``window`` sentence-length lines say at most
  ``distinct`` different things. The source compares exact stripped lines; ``normalize=True`` is an
  EPYC extension (lower-case, digits and punctuation dropped, whitespace collapsed) and is scored
  separately, never substituted for the source rule.
* Profiles: ``TOOLS`` (tools offered, outside a call: reps 3 over >= 24 tokens, blocks < 8 tokens
  must span 128; no grace), ``IN_GRAMMAR`` (inside a call's grammar-constrained arguments: 16 reps
  over >= 256 tokens, plus ``LOOP_GRACE`` = 1,024 tokens of continued looping), ``NO_TOOLS``
  (no tools, which in Locals includes schema/regex-constrained replies: 12 reps over >= 96, plus
  grace).
* ``test_after_edit_exempt`` — a verifying call (test/check/lint/build, or ``examine``/``look``/
  ``inventory`` in a text environment) that repeats an earlier identical call is LEGITIMATE when a
  mutating call ran in between (edit, then rerun the tests). Exempt calls are dropped from the
  repetition evidence before any detector reads it. A mutating call that repeats an identical
  earlier mutating call is never exempt (re-applying the same edit is the loop).

The tokenizer is injected. ``regex_tokenize`` is the deterministic default (words, single
punctuation marks, whitespace runs); scoring scripts may pass a Hugging Face ``tokenizer.json``
loaded with the ``tokenizers`` library (no model, no forward pass), whose ids are what Locals' token
thresholds were set on.

Decision record (HIL-LB1, 2026-10-22: does constrained output get a loop guard back?)
------------------------------------------------------------------------------------
Context: the orchestrator's streaming repetition guard (``src/llm_primitives/inference.py``
``_detect_streaming_repetition``) is switched OFF for json_schema/grammar requests
(``_request_is_grammar_constrained``) since TD-1d.2, because it cut a legitimate 24-question
schema-constrained JSON reply after 300 chunks. Locals keeps a guard on constrained output, made
tolerant by high repeat counts and a 1,024-token grace window.

Evidence (``scripts/trace/hil_lb1_loop_score.py``, Qwen3 tokenizer.json, 2026-10-22):
* Labelled loops (Who&When Pro text, injected mode A.4, N=53 of 6,257, all in the 311 alfagent
  traces): the in-grammar variant fires on 0/53 and 0/6,204 negatives — structurally, the
  1,024-token grace window is longer than a whole 30-step action stream — so it is a runaway-reply
  cap, not a behavioural loop detector. Best F1 is still the existing L2 detector (P 0.264,
  R 0.623, F1 0.371); Locals token-period on the action stream P 0.318 R 0.132; the source
  paraphrase rule fires on nothing (its 30-line window exceeds the trajectories); per-reply
  application of either never fires on a positive.
* Constrained-shaped fixtures: a TD-1-shaped 24-answer reply is cut by the current free-form
  guard and by no Locals profile. The in-grammar variant DOES cut legitimate closed arrays of
  identical rows once the periodic stretch passes about 1,400-1,600 tokens (48 identical compact
  objects, 1,682 tokens: cut at 1,585; 64 identical grid rows, 1,602 tokens: cut at 1,425), and
  stops a degenerate runaway at 1,729 / 2,198 tokens.
* Recorded EPYC replies (inference tap, 3,576 replies, 649 JSON-shaped): the in-grammar and
  no-tools profiles fire on 0, no reply ends in a periodic tail, and no JSON-shaped reply reached
  4,096 tokens. There is no observed constrained runaway to save tokens on.

So restoring a guard has a measured cost (legitimate long repetitive structured output is cut) and
no measured benefit; ``n_predict`` already ends a runaway.

Recommendation: do not restore a guard now (option B). If one is ever restored it must be the
in-grammar profile with grace (option A), never the free-form guard. Options:
  A. restore a guard for constrained replies using ``IN_GRAMMAR`` + grace behind a default-off
     FeatureSpec. Saves at most (n_predict - loop start - 1,024) tokens per runaway. Cost: cuts
     a legitimate reply whose identical periodic stretch exceeds ~1,300-1,600 tokens.
  B. (recommended) keep constrained output unguarded; ``n_predict`` bounds the waste. Reopen A when
     a census of constrained replies shows any ending at the token limit with a periodic tail
     (``looping(toks, 16, 256, 256, 256)`` on the final tokens is the census predicate; the request
     grammar flag would need to be recorded in the tap to make the census exact).
  C. restore the free-form guard. Rejected: it is the guard TD-1d.2 removed, and it reproduces that
     cut on the TD-1 fixture.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

LOOP_GRACE = 1024

_REGEX_TOKEN = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)


def regex_tokenize(text: str) -> list[str]:
    """Deterministic default tokenizer: word runs, single punctuation marks, whitespace runs."""
    return _REGEX_TOKEN.findall(text or "")


def hf_tokenizer(path: str) -> Callable[[str], list[int]]:
    """A tokenizer.json loaded with ``tokenizers`` (no model). Raises if the library is missing."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    return lambda text: tok.encode(text or "", add_special_tokens=False).ids


@dataclass(frozen=True)
class PeriodProfile:
    name: str
    reps: int
    min_span: int
    short_span: int
    max_period: int = 256
    grace: int | None = None


TOOLS = PeriodProfile("tools", reps=3, min_span=24, short_span=128)
IN_GRAMMAR = PeriodProfile("in_grammar", reps=16, min_span=256, short_span=256, grace=LOOP_GRACE)
NO_TOOLS = PeriodProfile("no_tools", reps=12, min_span=96, short_span=256, grace=LOOP_GRACE)
PROFILES = {p.name: p for p in (TOOLS, IN_GRAMMAR, NO_TOOLS)}


def looping_at(toks: Sequence, n: int, reps: int = 3, max_period: int = 256, min_span: int = 24,
               short_span: int = 128, _pos: dict | None = None) -> bool:
    """Locals ``_looping`` on the prefix ``toks[:n]`` (port; see module docstring).

    ``_pos`` (token -> sorted positions) only speeds up the candidate-period scan; the candidates
    it yields are exactly the periods for which ``toks[n - 1 - p] == toks[n - 1]``.
    """
    if n < min_span:
        return False
    for p in range(1, 8):
        if n >= 32 and toks[n - 32 + p:n] == toks[n - 32:n - p]:
            span = max(reps * p, min_span, short_span)
            return span <= n and toks[n - span + p:n] == toks[n - span:n - p]
    last = toks[n - 1]
    hi = min(max_period, n // reps)
    if hi < 8:
        return False
    if _pos is not None:
        plist = _pos.get(last, ())
        # positions j with 8 <= n-1-j <= hi  <=>  n-1-hi <= j <= n-9, ascending p = descending j
        lo_i = bisect_left(plist, n - 1 - hi)
        hi_i = bisect_left(plist, n - 8)
        cands = [n - 1 - j for j in reversed(plist[lo_i:hi_i])]
    else:
        cands = [p for p in range(8, hi + 1) if toks[n - 1 - p] == last]
    for p in cands:
        span = max(reps * p, min_span)
        if span <= n and toks[n - span + p:n] == toks[n - span:n - p]:
            return True
    return False


def looping(toks: Sequence, reps: int = 3, max_period: int = 256, min_span: int = 24,
            short_span: int = 128) -> bool:
    return looping_at(toks, len(toks), reps, max_period, min_span, short_span)


def first_period_fire(toks: Sequence, profile: PeriodProfile = TOOLS, start: int = 1) -> int | None:
    """Replay the detector once per token (as Locals' decode loop does). Returns the prefix length
    at which the guard would stop the reply, or None. With a grace window the guard stops only once
    the reply has looked like a loop for ``grace`` consecutive tokens (``_past_grace``)."""
    pos: dict = {}
    loop_at: int | None = None
    for n in range(1, len(toks) + 1):
        if n >= 2:
            pos.setdefault(toks[n - 2], []).append(n - 2)
        if n < start:
            continue
        hit = looping_at(toks, n, profile.reps, profile.max_period, profile.min_span,
                         profile.short_span, _pos=pos)
        if profile.grace is None:
            if hit:
                return n
            continue
        if not hit:
            loop_at = None
        else:
            if loop_at is None:
                loop_at = n
            if n - loop_at >= profile.grace:
                return n
    return None


_NORM_DROP = re.compile(r"[\d\W_]+", re.UNICODE)


def _norm_line(s: str) -> str:
    return " ".join(_NORM_DROP.sub(" ", s.lower()).split())


def line_looping(text: str, window: int = 30, distinct: int = 6, min_len: int = 12,
                 normalize: bool = False) -> bool:
    """Locals ``_line_looping`` (port); ``normalize`` is the EPYC extension (scored separately)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if len(ln.strip()) >= min_len]
    if normalize:
        lines = [_norm_line(ln) for ln in lines]
    return len(lines) >= window and len(set(lines[-window:])) <= distinct


def first_line_fire(text: str, window: int = 30, distinct: int = 6, min_len: int = 12,
                    normalize: bool = False) -> int | None:
    """Streaming paraphrase check evaluated at every line end. Returns the 1-based index of the
    qualifying line at which it first fires, or None. Between line ends the source's result can
    only change through the partial last line, which this replay counts once it is complete."""
    lines: list[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if len(s) < min_len:
            continue
        lines.append(_norm_line(s) if normalize else s)
        if len(lines) >= window and len(set(lines[-window:])) <= distinct:
            return len(lines)
    return None


# ------------------------------------------------------------------ repeated calls + exemption
MUTATING_TOOL_RE = re.compile(r"write|edit|patch|create|apply|delete|commit|replace|insert|append", re.I)
VERIFYING_TOOL_RE = re.compile(r"test|verify|check|lint|build|compile|assert|validate", re.I)
# Text-environment verbs (ALFWorld-style action strings) and shell bodies.
MUTATING_VERBS = frozenset({"take", "put", "move", "open", "close", "toggle", "heat", "cool", "clean",
                            "slice", "use", "drop", "place"})
VERIFYING_VERBS = frozenset({"examine", "look", "inventory", "check"})
SHELL_TOOLS = frozenset({"bash", "shell", "run", "run_shell", "terminal", "exec", "execute_command"})
VERIFYING_SHELL_RE = re.compile(r"\b(pytest|unittest|tox|nox|ctest|go test|cargo test|npm (run )?test|"
                                r"make (test|check)|ruff|mypy|flake8|lint)\b", re.I)


# Arguments that label a call for a human and change nothing it does (Claude Code's Bash
# ``description``). Dropped from the identity key; an EPYC normalisation of intake-2395's metric.
LABEL_ONLY_ARGS = frozenset({"description"})


@dataclass(frozen=True)
class CallUnit:
    tool: str | None
    args: object = None       # dict (preferred) or raw text

    @property
    def key(self) -> str:
        """Call identity: tool name + sorted-key JSON arguments (label-only arguments dropped)."""
        a = self.args
        if isinstance(a, dict):
            a = {k: v for k, v in a.items() if k not in LABEL_ONLY_ARGS}
        a = json.dumps(a, sort_keys=True, default=str) if isinstance(a, (dict, list)) else str(a or "")
        return f"{self.tool}\x1f{a}"

    def body(self) -> str:
        a = self.args
        if isinstance(a, dict):
            a = {k: v for k, v in a.items() if k not in LABEL_ONLY_ARGS}
            if len(a) == 1 and isinstance(next(iter(a.values())), str):
                return next(iter(a.values()))
        if isinstance(a, (dict, list)):
            return json.dumps(a, sort_keys=True, default=str)
        return str(a or "")


def classify_call(u: CallUnit) -> str:
    """``mutating`` | ``verifying`` | ``other``. Tool name first; then the verb of a text-environment
    action; then a shell body that runs tests or checks."""
    tool = (u.tool or "").strip()
    tl = tool.lower()
    if tl in SHELL_TOOLS:
        return "verifying" if VERIFYING_SHELL_RE.search(u.body()) else "other"
    if tool and MUTATING_TOOL_RE.search(tool):
        return "mutating"
    if tool and VERIFYING_TOOL_RE.search(tool):
        return "verifying"
    words = u.body().strip().split()
    verb = words[0].lower() if words else ""
    if verb in MUTATING_VERBS:
        return "mutating"
    if verb in VERIFYING_VERBS:
        return "verifying"
    return "other"


def test_after_edit_exempt(units: Sequence[CallUnit]) -> list[bool]:
    """True for a verifying call that repeats an identical earlier call with a mutating call in
    between (legitimate repetition: edit, then rerun the check)."""
    kinds = [classify_call(u) for u in units]
    last_seen: dict[str, int] = {}
    last_mut = -1
    out = []
    for i, (u, k) in enumerate(zip(units, kinds)):
        prev = last_seen.get(u.key)
        out.append(k == "verifying" and prev is not None and last_mut > prev)
        last_seen[u.key] = i
        if k == "mutating":
            last_mut = i
    return out


def repeated_calls(units: Sequence[CallUnit], exempt: Iterable[bool] | None = None) -> list[int]:
    """Indices of calls with the same tool and sorted-key arguments as ANY earlier call
    (intake-2395's process metric), skipping exempt calls."""
    ex = list(exempt) if exempt is not None else [False] * len(units)
    seen: set[str] = set()
    out = []
    for i, u in enumerate(units):
        if u.key in seen and not ex[i]:
            out.append(i)
        seen.add(u.key)
    return out


def action_stream(units: Sequence[CallUnit], exempt: Iterable[bool] | None = None) -> tuple[str, list[int]]:
    """The agent's emitted calls as one text, one call body per line, exempt calls dropped.
    Returns (text, unit index per kept line)."""
    ex = list(exempt) if exempt is not None else [False] * len(units)
    kept = [i for i in range(len(units)) if not ex[i]]
    return "\n".join(units[i].body() for i in kept) + ("\n" if kept else ""), kept
