"""Deterministic RLVR reward contracts for eval-tower tiers.

AP-27 needs the eval tower to be usable as an RLVR environment without turning
current promotion metrics into training rewards by implication. This module is a
pure, observe-only contract: it maps EvalResult-like objects to deterministic
reward views and explicitly records missing calibration/process evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


RLVR_REWARD_POLICY = "ap27_rlvr_tier_reward_v1"


@dataclass(frozen=True)
class RLVRTierSpec:
    tier: int
    label: str
    reward_signal: str
    verifier: str = "deterministic_state_matching"
    required_metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class RLVRReward:
    tier: int
    policy: str
    reward_signal: str
    reward: float
    components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()

    @property
    def ready_for_training(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "policy": self.policy,
            "reward_signal": self.reward_signal,
            "reward": self.reward,
            "components": self.components,
            "metrics": self.metrics,
            "blockers": list(self.blockers),
            "ready_for_training": self.ready_for_training,
        }


RLVR_TIER_SPECS: dict[int, RLVRTierSpec] = {
    0: RLVRTierSpec(
        0,
        "T0 binary sentinel",
        "binary_outcome",
        required_metrics=("quality", "reliability"),
    ),
    1: RLVRTierSpec(
        1,
        "T1 calibrated continuous",
        "calibrated_continuous",
        required_metrics=("quality", "reliability", "ece", "auroc"),
    ),
    2: RLVRTierSpec(
        2,
        "T2 process-attributed",
        "process_attributed",
        required_metrics=("quality", "reliability", "ece", "auroc", "question_results"),
    ),
}


def spec_for_rlvr_tier(tier: int) -> RLVRTierSpec:
    """Return the RLVR reward contract for a tier.

    Tiers above T2 use the same process-attributed contract until they gain a
    separate verifier design.
    """
    tier_i = int(tier)
    return RLVR_TIER_SPECS.get(
        tier_i,
        RLVRTierSpec(
            tier_i,
            f"T{tier_i} process-attributed",
            "process_attributed",
            required_metrics=RLVR_TIER_SPECS[2].required_metrics,
        ),
    )


def rlvr_reward_from_result(result: Any) -> RLVRReward:
    """Build a deterministic reward view from an EvalResult-like object."""
    tier = int(getattr(result, "tier", 0) or 0)
    spec = spec_for_rlvr_tier(tier)
    quality = _as_float(getattr(result, "quality", None))
    reliability = _clamp01(_as_float(getattr(result, "reliability", None)))
    accuracy = _clamp01(quality / 3.0)
    ece = _as_float(getattr(result, "ece", None), math.nan)
    auroc = _as_float(getattr(result, "auroc", None), math.nan)
    question_results = getattr(result, "question_results", None)
    metrics = {
        "quality": quality,
        "accuracy": accuracy,
        "reliability": reliability,
        "ece": ece,
        "auroc": auroc,
    }
    blockers = list(_metric_blockers(spec, ece=ece, auroc=auroc, question_results=question_results))

    if spec.reward_signal == "binary_outcome":
        success = 1.0 if accuracy >= 1.0 and reliability >= 1.0 else 0.0
        return RLVRReward(
            tier=tier,
            policy=RLVR_REWARD_POLICY,
            reward_signal=spec.reward_signal,
            reward=success,
            components={"binary_success": success},
            metrics=metrics,
            blockers=tuple(blockers),
        )

    calibration = _calibration_component(ece)
    discrimination = _discrimination_component(auroc)
    if spec.reward_signal == "calibrated_continuous":
        reward = _clamp01(
            0.65 * accuracy + 0.20 * reliability + 0.10 * calibration + 0.05 * discrimination
        )
        return RLVRReward(
            tier=tier,
            policy=RLVR_REWARD_POLICY,
            reward_signal=spec.reward_signal,
            reward=round(reward, 6),
            components={
                "accuracy": accuracy,
                "reliability": reliability,
                "calibration": calibration,
                "discrimination": discrimination,
            },
            metrics=metrics,
            blockers=tuple(blockers),
        )

    process_integrity = _process_integrity(question_results)
    reward = _clamp01(
        0.55 * accuracy
        + 0.15 * reliability
        + 0.15 * calibration
        + 0.05 * discrimination
        + 0.10 * process_integrity
    )
    return RLVRReward(
        tier=tier,
        policy=RLVR_REWARD_POLICY,
        reward_signal=spec.reward_signal,
        reward=round(reward, 6),
        components={
            "accuracy": accuracy,
            "reliability": reliability,
            "calibration": calibration,
            "discrimination": discrimination,
            "process_integrity": process_integrity,
        },
        metrics={**metrics, "process_integrity": process_integrity},
        blockers=tuple(blockers),
    )


def _metric_blockers(
    spec: RLVRTierSpec,
    *,
    ece: float,
    auroc: float,
    question_results: Any,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if "ece" in spec.required_metrics and not _finite(ece):
        blockers.append("ece_missing")
    if "auroc" in spec.required_metrics and (not _finite(auroc) or auroc <= 0.0):
        blockers.append("auroc_missing_or_degenerate")
    if "question_results" in spec.required_metrics and not _question_rows(question_results):
        blockers.append("question_results_missing")
    return tuple(blockers)


def _process_integrity(question_results: Any) -> float:
    rows = _question_rows(question_results)
    if not rows:
        return 0.0
    clean = 0
    for row in rows:
        if row.get("error") or row.get("partial") or row.get("degraded"):
            continue
        clean += 1
    return round(clean / len(rows), 6)


def _question_rows(question_results: Any) -> list[Mapping[str, Any]]:
    if not isinstance(question_results, Sequence) or isinstance(question_results, (str, bytes)):
        return []
    return [row for row in question_results if isinstance(row, Mapping)]


def _calibration_component(ece: float) -> float:
    return 0.0 if not _finite(ece) else _clamp01(1.0 - ece)


def _discrimination_component(auroc: float) -> float:
    return 0.0 if not _finite(auroc) else _clamp01(auroc)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
