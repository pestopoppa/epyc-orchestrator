"""Per-tier scoring specifications for the autopilot Pareto archive + safety gate.

Quality (`fraction_correct * 3`) is computed on a per-tier question set and the tiers differ
in difficulty (T0 ~10 easy q saturates ~2.4; T1 ~50 mixed ~1.6-1.9; T2 ~480 incl. GPQA/olympiad
~1.16), so quality is NOT comparable across tiers. The archive and safety gate are therefore
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

# T0 is a fast-reject sentinel tier (10q, quality saturates ~2.4 = 8/10) and never enters any
# frontier/baseline. Tiers >= this each keep their own segregated frontier + baseline.
MIN_FRONTIER_EVAL_TIER = 1
# The canonical "production" tier the controller optimizes and the dashboard shows by default.
# Other tiers (T2, future harder tiers) are periodic validation with their own frontiers.
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
