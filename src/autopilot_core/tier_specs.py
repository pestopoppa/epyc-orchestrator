"""Per-tier scoring specifications for the autopilot Pareto archive + safety gate.

Quality (`fraction_correct * 3`) is computed on a per-tier question set and the tiers differ
in difficulty (T0 ~10 easy q saturates ~2.4; T1 ~50 mixed ~1.6-1.9; T2 ~480 incl. GPQA/olympiad
~1.16; T3 hard-only stress rows), so quality is NOT comparable across tiers. The archive and safety gate are therefore
tier-segregated: each eval tier >= MIN_FRONTIER_EVAL_TIER keeps its OWN frontier + baseline, and a
trial is only ever ranked / gated against the SAME tier.

A `TierSpec` defines how a tier's objective tuple is built and its hypervolume reference point.
This keeps per-tier scoring **pluggable**: a future, more complex tier can carry different quality
semantics (its own scale/metric) by registering a different spec — without confounding existing
tiers or re-plumbing the archive/gate. All CURRENT tiers share the 4D
`(quality, speed, -cost, reliability)` shape and the canonical reference point.

This module is the SINGLE source of truth for tier scoring and is imported the same way
(`from src.autopilot_core.tier_specs import ...`) by both `scripts/autopilot` (which puts
ORCH_ROOT on sys.path) and `src/api`, so `TIER_SPECS` is one shared registry object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# Canonical 4D objective shape + hypervolume reference point (worst acceptable values).
# (quality↑, speed↑, -cost↑ i.e. lower cost better, reliability↑)
DEFAULT_REFERENCE_POINT: tuple[float, ...] = (0.0, 0.0, -1.0, 0.0)
LEGACY_OBJECTIVE_POLICY = "legacy_4d_v1"
TASK_RATE_OBJECTIVE_POLICY = "task_rate_3d_v1"
TASK_RATE_REFERENCE_POINT: tuple[float, ...] = (0.0, 0.0, 0.0)

# T0 is a fast-reject sentinel tier (10q, quality saturates ~2.4 = 8/10) and never enters any
# frontier/baseline. Tiers >= this each keep their own segregated frontier + baseline.
MIN_FRONTIER_EVAL_TIER = 1
# The canonical "production" tier the controller optimizes and the dashboard shows by default.
# Other tiers (T2, T3) are periodic validation/stress lanes with their own frontiers.
DEFAULT_FRONTIER_TIER = 1


def _default_objectives_from(result: Any) -> tuple[float, ...]:
    """4D objective tuple from an EvalResult-like object: (quality, speed, -cost, reliability)."""
    return (
        float(getattr(result, "quality", 0.0) or 0.0),
        float(getattr(result, "speed", 0.0) or 0.0),
        -float(getattr(result, "cost", 0.0) or 0.0),
        float(getattr(result, "reliability", 0.0) or 0.0),
    )


def _default_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """4D objective tuple from a journal-row dict (dashboard reconstruction), or None if unusable."""
    try:
        return (
            float(row.get("quality") or 0.0),
            float(row.get("speed") or 0.0),
            -float(row.get("cost") or 0.0),
            float(row.get("reliability") or 0.0),
        )
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nested(row: dict, *path: str) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _task_rate_inputs_from_row(row: dict) -> tuple[float, float]:
    eval_details = row.get("eval_details") or {}
    n_questions = (
        row.get("n_questions")
        or _nested(eval_details, "details", "total")
        or _nested(eval_details, "details", "n_questions")
        or _nested(eval_details, "details", "per_suite_counts_total")
    )
    if not n_questions:
        counts = _nested(eval_details, "details", "per_suite_counts")
        if isinstance(counts, dict):
            n_questions = sum(int(v) for v in counts.values() if int(v) > 0)
    eval_wall_s = (
        row.get("eval_wall_s")
        or eval_details.get("eval_wall_s")
        or _nested(eval_details, "details", "eval_wall_s")
    )
    return _as_float(n_questions), _as_float(eval_wall_s)


def task_rate_qph_from(result: Any) -> float:
    """Questions completed per eval-wall-hour; 0 when wall/n are unavailable."""
    n_questions = _as_float(
        getattr(result, "n_questions", None)
        or _nested(getattr(result, "details", {}) or {}, "total")
    )
    eval_wall_s = _as_float(
        getattr(result, "eval_wall_s", None)
        or _nested(getattr(result, "details", {}) or {}, "eval_wall_s")
    )
    if n_questions <= 0 or eval_wall_s <= 0:
        return 0.0
    return n_questions / (eval_wall_s / 3600.0)


def task_rate_qph_from_row(row: dict) -> float:
    """Questions completed per eval-wall-hour from a journal row."""
    n_questions, eval_wall_s = _task_rate_inputs_from_row(row)
    if n_questions <= 0 or eval_wall_s <= 0:
        return 0.0
    return n_questions / (eval_wall_s / 3600.0)


def goodput_qph_from(result: Any) -> float:
    """Solved-question rate: quality-scaled task_rate on the 0-3 quality scale."""
    return (_as_float(getattr(result, "quality", 0.0)) / 3.0) * task_rate_qph_from(result)


def goodput_qph_from_row(row: dict) -> float:
    """Solved-question rate from a journal row."""
    return (_as_float(row.get("quality")) / 3.0) * task_rate_qph_from_row(row)


def task_rate_objectives_from(result: Any, tier: int | None = None) -> tuple[float, ...]:
    """Shadow 3D vector: (quality, task_rate_qph, reliability)."""
    _ = tier  # Reserved for future tier-specific rate semantics.
    return (
        _as_float(getattr(result, "quality", 0.0)),
        task_rate_qph_from(result),
        _as_float(getattr(result, "reliability", 0.0)),
    )


def task_rate_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """Shadow 3D vector from a journal row: (quality, task_rate_qph, reliability)."""
    try:
        return (
            float(row.get("quality") or 0.0),
            task_rate_qph_from_row(row),
            float(row.get("reliability") or 0.0),
        )
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TierSpec:
    """How one eval tier's quality is scored for the Pareto archive + safety gate."""
    tier: int
    label: str
    reference_point: tuple[float, ...] = DEFAULT_REFERENCE_POINT
    objectives_from: Callable[[Any], tuple[float, ...]] = _default_objectives_from
    objectives_from_row: Callable[[dict], tuple[float, ...] | None] = _default_objectives_from_row


# Registry: tier -> spec. All current tiers share the 4D shape. A future divergent-scoring tier is
# a new entry here, NOT a re-plumb of the archive/gate.
TIER_SPECS: dict[int, TierSpec] = {
    0: TierSpec(0, "T0 (10q sentinel, fast-reject)"),
    1: TierSpec(1, "T1 (50q gate)"),
    2: TierSpec(2, "T2 (480q comprehensive)"),
    3: TierSpec(3, "T3 (hard-only stress eval)"),
}


def spec_for(tier: int) -> TierSpec:
    """TierSpec for a tier; defaults to the 4D shape for any unregistered tier."""
    s = TIER_SPECS.get(int(tier))
    return s if s is not None else TierSpec(int(tier), f"T{int(tier)}")


def objectives_from(result: Any, tier: int | None = None) -> tuple[float, ...]:
    """Canonical objective construction — use this everywhere instead of `EvalResult.objectives`.

    `tier` defaults to `result.tier`. This is the single chokepoint so a future tier with
    different scoring semantics is handled by its spec, not by ad-hoc tuple-building at call sites.
    """
    t = int(tier if tier is not None else getattr(result, "tier", DEFAULT_FRONTIER_TIER))
    return spec_for(t).objectives_from(result)


def reference_point_for(tier: int) -> tuple[float, ...]:
    """Hypervolume reference point for a tier."""
    return spec_for(tier).reference_point
