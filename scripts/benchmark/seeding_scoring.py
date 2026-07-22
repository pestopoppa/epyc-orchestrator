"""Deterministic scoring, error classification, and timeout logic.

Pure functions — no network I/O or mutable state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from seeding_types import DEFAULT_TIMEOUT

__all__ = [
    "INFRA_PATTERNS",
    "_adaptive_timeout_s",
    "_bump_timeout_from_observed",
    "_classify_error",
    "_forced_role_serving_mismatch",
    "_inband_error_text",
    "_is_coding_task",
    "score_answer_deterministic",
    "score_answer_or_error",
]


# ── Scoring ──────────────────────────────────────────────────────────


_ORCH_SCORER_KEY = "epyc_orch_debug_scorer"


def _load_orchestrator_debug_scorer():
    """Load THIS repo's ``debug_scorer.py`` by path, under a private key.

    seeding_scoring lives beside the orchestrator copy of ``debug_scorer.py``,
    but the research repo ships a diverged copy of the *same* filename. A bare
    ``from debug_scorer import score_answer`` binds whichever copy won the
    ``sys.path`` insertion race, so scorer identity became import-order
    dependent (and could silently pick up the research copy's scoring rules).

    Load the sibling file explicitly via ``importlib`` under a private
    ``epyc_orch_debug_scorer`` ``sys.modules`` key so we always resolve the
    orchestrator copy regardless of import history — and without mutating
    ``sys.path`` or touching the research repo. Cached after first load.
    """
    import importlib.util
    import sys

    cached = sys.modules.get(_ORCH_SCORER_KEY)
    if cached is not None:
        return cached

    scorer_path = Path(__file__).resolve().parent / "debug_scorer.py"
    spec = importlib.util.spec_from_file_location(_ORCH_SCORER_KEY, scorer_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load debug_scorer from {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ORCH_SCORER_KEY] = module
    spec.loader.exec_module(module)
    return module


def score_answer_deterministic(
    answer: str,
    expected: str,
    scoring_method: str = "exact_match",
    scoring_config: dict[str, Any] | None = None,
) -> bool:
    """Score an answer deterministically."""
    score_answer = _load_orchestrator_debug_scorer().score_answer

    return score_answer(answer, expected, scoring_method, scoring_config or {})


def score_answer_or_error(
    answer: str,
    expected: str,
    scoring_method: str = "exact_match",
    scoring_config: dict[str, Any] | None = None,
) -> tuple[bool | None, str | None]:
    """Score, converting scorer-unavailability into an EXCLUDED row.

    The B7-hardened orchestrator scorer *raises* when it cannot produce a
    trustworthy verdict rather than silently mis-scoring:
    ``ScoringUnavailableError`` for entry_point-without-oracle, math_verify
    unavailable / bad gold, or an unreachable llm_judge (SCORE-04/05); and
    ``ValueError`` for an unknown scoring method or verifier (SCORE-25).

    The seeding call sites historically invoked ``score_answer_deterministic``
    with no ``try/except`` (seeding_eval.py ``_build_role_result``): a raise
    therefore CRASHED the entire seed run on the main 3-way path and was
    silently swallowed — the question vanishing with no row and no tally — on
    the debugger-retry path. Mirror the eval tower's per-question guard
    (eval_tower.py:2575-2591 → excluded from the denominator at :2991) by
    returning ``(None, reason)`` so the caller can stamp an EXCLUDED
    infrastructure row instead of crashing or dropping. A normal verdict
    returns ``(bool, None)``.
    """
    scorer = _load_orchestrator_debug_scorer()
    # ScoringUnavailableError lives in the dynamically-loaded scorer module.
    # Fall back to an empty tuple (a never-matching except target) if a future
    # scorer build lacks it, so this wrapper never raises on the guard itself.
    scoring_unavailable = getattr(scorer, "ScoringUnavailableError", ())
    try:
        return bool(
            score_answer_deterministic(answer, expected, scoring_method, scoring_config or {})
        ), None
    except scoring_unavailable as exc:  # type: ignore[misc]
        return None, f"scoring_unavailable: {exc}"
    except ValueError as exc:
        return None, f"scoring_error: {exc}"


# ── Error classification ─────────────────────────────────────────────


# REL-1 in-band error prefix (2026-07-21 EV-11c circuit-open incident).
# The orchestrator's llm primitives emit failures AS the answer string of the
# form ``[ERROR: <detail>]`` (src/llm_primitives/primitives.py::_call →
# ``return f"[ERROR: {e}]"``; the circuit-open detail is
# ``Backend unavailable (circuit open): <url>``). When the breaker opens and
# the /chat body is NOT run through server-side ``_annotate_error``, the client
# receives answer="[ERROR: Backend unavailable (circuit open): ...]" with
# error=None. Anchor to this REAL start-of-answer prefix, never a loose
# substring. Mirrors eval_tower.py::_inband_error_text (kept as a local copy;
# eval_tower is owned by another session — see report note on unification).
_INBAND_ERROR_PREFIX = "[ERROR:"


def _inband_error_text(answer: Any) -> str | None:
    """Return the in-band orchestrator error string when ``answer`` IS one.

    Anchored to the emitted ``[ERROR: ...]`` prefix at start-of-answer (after
    stripping leading whitespace), matching the primitives/inference emitters
    and the server-side ``_annotate_error`` convention. Returns ``None`` for a
    normal answer.
    """
    if not isinstance(answer, str):
        return None
    stripped = answer.lstrip()
    if stripped.startswith(_INBAND_ERROR_PREFIX):
        return stripped
    return None


def _forced_role_serving_mismatch(
    force_role: Any, resp: Mapping[str, Any]
) -> str | None:
    """Return the serving role when it differs from the forced role, else None.

    REL-1 Guard 2: when a seed config pins ``force_role`` for a role-attributed
    measurement *with delegation disabled* and the orchestrator silently serves
    it from a DIFFERENT role (the 2026-07-21 circuit_open fallback
    ``worker_math → worker_general``), the number is not a measurement of the
    forced role. Compare ``force_role`` against the response's ``routed_to``
    (the primary role that handled the request), falling back to the terminal
    ``role_history`` entry when ``routed_to`` is absent. Returns ``None`` when
    ``force_role`` is empty or the serving role cannot be determined — avoiding
    false positives on partial/legacy responses.

    NOTE: callers MUST gate this on ``allow_delegation is False``. On the
    seeding path the ARCHITECT config runs with delegation ENABLED, where
    ``routed_to != force_role`` is the expected, correct behavior (the architect
    delegates to workers); applying this guard there would wrongly exclude every
    delegated result. Mirrors eval_tower.py::_forced_role_serving_mismatch
    (local copy — see report note on unification).
    """
    forced = str(force_role or "").strip()
    if not forced:
        return None
    serving = str(resp.get("routed_to") or "").strip()
    if not serving:
        history = resp.get("role_history")
        if isinstance(history, (list, tuple)) and history:
            serving = str(history[-1] or "").strip()
    if not serving or serving == forced:
        return None
    return serving


INFRA_PATTERNS = [
    "timed out", "timeout", "connection", "refused",
    "unreachable", "502", "503", "504", "connecterror",
    "readtimeout", "backend down", "server error",
    "server disconnected without sending a response",
    "remoteprotocolerror", "connection reset", "broken pipe",
    "temporarily unavailable", "name or service not known",
    # REL-1 eval-honesty guards (2026-07-21 EV-11c circuit-open incident):
    # the orchestrator circuit breaker surfaces "[ERROR: Backend unavailable
    # (circuit open): <url>]" in-band; these must classify as INFRASTRUCTURE
    # (excluded from the quality denominator), never as a model task_failure.
    "circuit open", "backend unavailable",
    # eval_tower client-side REL-1 rejections (deadline-starvation floor and
    # forced-role integrity) are governance/infra exclusions, not model errors.
    "deadline_starved", "forced_role_fallback",
]


def _classify_error(error_str: str | None) -> str:
    """Classify error as infrastructure or task failure."""
    if error_str is None:
        return "none"
    # REL-1: an in-band "[ERROR: ...]" banner surfaced into the error field is
    # a backend/serving failure — the model never produced a real attempt at
    # the task. Classify it as INFRASTRUCTURE (excluded from scoring and from
    # MemRL reward emission), never as a task_failure (which would inject a
    # 0.0 reward and poison the learned router). This anchors the same way the
    # eval tower's Guard 1 does, so a generic in-band error (not matching an
    # INFRA_PATTERNS substring) is still excluded rather than scored WRONG.
    if error_str.lstrip().startswith(_INBAND_ERROR_PREFIX):
        return "infrastructure"
    error_lower = error_str.lower()
    if any(p in error_lower for p in INFRA_PATTERNS):
        return "infrastructure"
    return "task_failure"


# ── Coding-task heuristic ────────────────────────────────────────────


def _is_coding_task(prompt: str) -> bool:
    """Heuristic to determine if a task is coding-related.

    Used to choose between live architect-like roles when multiple exist.
    """
    coding_indicators = [
        "code", "function", "implement", "debug", "refactor",
        "class", "method", "algorithm", "bug", "error",
        "syntax", "compile", "runtime", "test", "unittest",
        "python", "javascript", "typescript", "rust", "go",
        "def ", "async ", "import ", "return ", "class ",
    ]
    prompt_lower = prompt.lower()
    return any(ind in prompt_lower for ind in coding_indicators)


# ── Timeout logic ────────────────────────────────────────────────────


def _adaptive_timeout_s(
    *,
    role: str,
    mode: str,
    prompt: str,
    is_vl: bool,
    hard_timeout_s: int,
) -> int:
    """Return a generous per-call timeout.

    Previous per-role caps (frontdoor=180, vision=240, etc.) caused premature
    INFRA classifications when the server was still generating.  The llama.cpp
    server keeps generating after client disconnect, so tight timeouts only
    waste the work.  Use a flat 600s ceiling; optimize later once we have
    solid per-role telemetry.
    """
    return max(60, int(hard_timeout_s or DEFAULT_TIMEOUT))


def _bump_timeout_from_observed(
    *,
    current_s: int,
    observed_s: float,
    factor: float,
    slack_s: int,
    hard_timeout_s: int,
    role_cap_s: int,
) -> int:
    """Increase timeout based on observed earlier stage runtime for this question.

    With the flat 600s ceiling from _adaptive_timeout_s, this function now
    only raises current_s if the observed time suggests it's too low.
    """
    if observed_s <= 0:
        return current_s
    observed_budget = int(observed_s * factor + slack_s)
    return max(current_s, min(observed_budget, max(60, int(hard_timeout_s or DEFAULT_TIMEOUT))))
