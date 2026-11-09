"""AP-29b(b): per-role token/elapsed capture for ``eval_details`` (observe-only).

Aggregates the per-question rows the eval tower already compacts (``route``,
``tokens_generated``, ``latency_ms``) so a replay can price a trial per role against
``baseline_tps`` (chapter 08 cost_ratio) without re-running anything. Nothing here feeds
SafetyGate, ParetoArchive or baseline mutation. Introduced with ``metric_schema_version`` 2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _role_of(row: Mapping[str, Any]) -> str:
    route = str(row.get("route") or "").strip().lower()
    return route or "unknown"


def role_cost_capture(
    question_results: Sequence[Mapping[str, Any]] | None,
    baseline_tps_by_role: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return ``per_role_tokens``, ``per_role_elapsed_s`` and ``baseline_tps`` (per role seen)."""
    tokens: dict[str, int] = {}
    elapsed: dict[str, float] = {}
    for row in question_results or []:
        if not isinstance(row, Mapping):
            continue
        role = _role_of(row)
        tokens[role] = tokens.get(role, 0) + int(row.get("tokens_generated") or 0)
        elapsed[role] = elapsed.get(role, 0.0) + max(0.0, float(row.get("latency_ms") or 0)) / 1000.0
    table = dict(baseline_tps_by_role or {})
    return {
        "per_role_tokens": tokens,
        "per_role_elapsed_s": {k: round(v, 6) for k, v in elapsed.items()},
        # None (not 0) when a role has no baseline: a missing prior must stay visible.
        "baseline_tps": {r: (float(table[r]) if table.get(r) else None) for r in tokens},
    }


def load_baseline_tps_by_role() -> dict[str, float]:
    """Registry t/s baselines; empty on any failure (capture then records None per role)."""
    try:
        from orchestration.repl_memory.q_scorer import registry_baseline_tps_by_role

        return dict(registry_baseline_tps_by_role())
    except Exception:  # noqa: BLE001 - observe-only; never break the journal write
        return {}
