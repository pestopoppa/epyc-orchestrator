"""Shared REL-1 measurement guards for role-attributed scoring.

Both the eval tower (`scripts/autopilot/eval_tower.py`) and the seeding path
(`scripts/benchmark/seeding_scoring.py`) must answer the same two questions
before a response is allowed to count as a measurement of a role:

  1. Is this "answer" actually an in-band orchestrator ERROR string?
  2. Was the request served by a DIFFERENT role than the one it pinned?

Until now each side carried its own copy. The copies were deliberate — eval_tower
was read-only to the session that wrote the seeding guards — and the audit that
created them filed their unification as a residual (`scorer-fork-drift-audit-
2026-07-22.md`). This module is that unification.

Why it matters that there is exactly one copy: these are not helpers, they are
the predicates that decide whether a number is admissible evidence. Two copies of
an admissibility rule drift silently, and the drift only shows up as a scorer
disagreement between two paths that are supposed to be measuring the same thing —
which is precisely the defect class the fork-drift audit exists to close.

Verified byte-equivalent in behaviour at unification time: both prior copies
parsed to identical ASTs once docstrings were stripped, so this move changes no
outcome on either path.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "INBAND_ERROR_PREFIX",
    "forced_role_serving_mismatch",
    "inband_error_text",
]


# The orchestrator emits in-band failures as `[ERROR: ...]` at start-of-answer:
# `src/llm_primitives/inference.py` returns `f"[ERROR: {e}]"`, and the
# circuit-open detail reads `Backend unavailable (circuit open): <url>`. When the
# breaker opens and the /chat body is NOT run through server-side
# `_annotate_error`, the client receives `answer="[ERROR: Backend unavailable
# (circuit open): ...]"` with `error=None` — i.e. a failure wearing a success's
# clothes.
#
# Anchor to this REAL start-of-answer prefix, never a loose substring: a model
# legitimately discussing "[ERROR:" mid-answer must not be discarded as infra.
INBAND_ERROR_PREFIX = "[ERROR:"


def inband_error_text(answer: Any) -> str | None:
    """Return the in-band orchestrator error string when ``answer`` IS one.

    Anchored to the emitted ``[ERROR: ...]`` prefix at start-of-answer (after
    stripping leading whitespace), matching the primitives/inference emitters
    and the server-side ``_annotate_error`` convention. Returns ``None`` for a
    normal answer.
    """
    if not isinstance(answer, str):
        return None
    stripped = answer.lstrip()
    if stripped.startswith(INBAND_ERROR_PREFIX):
        return stripped
    return None


def forced_role_serving_mismatch(
    force_role: Any, resp: Mapping[str, Any]
) -> str | None:
    """Return the serving role when it differs from the forced role, else None.

    REL-1 Guard 2: when a config pins ``force_role`` for a role-attributed
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
    delegated result.
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
