"""Deterministic scoring, error classification, and timeout logic.

Pure functions — no network I/O or mutable state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seeding_types import DEFAULT_TIMEOUT

__all__ = [
    "INFRA_PATTERNS",
    "_adaptive_timeout_s",
    "_bump_timeout_from_observed",
    "_classify_error",
    "_is_coding_task",
    "score_answer_deterministic",
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


# ── Error classification ─────────────────────────────────────────────


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
