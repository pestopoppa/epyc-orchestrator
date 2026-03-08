"""Safety gate: quality floor, regression guards, rollback triggers.

Loads frozen baseline from autopilot_baseline.yaml and enforces constraints.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("autopilot.safety")

DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "autopilot_baseline.yaml"
)

# Hard-coded safety thresholds
QUALITY_FLOOR = 2.0  # Average quality >= 2.0/3.0
REGRESSION_THRESHOLD = -0.05  # Max quality drop vs baseline
PER_SUITE_REGRESSION = -0.1  # Max per-suite quality drop
ARCHITECT_ROUTING_CAP = 0.80  # Max fraction routed to architect-tier
MAX_CONSECUTIVE_FAILURES = 3  # Auto-rollback after this many T0 failures


@dataclass
class SafetyVerdict:
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class EvalResult:
    """Evaluation result from EvalTower."""
    tier: int
    quality: float  # Average quality 0-3
    speed: float  # Median tokens/sec
    cost: float  # Normalized cost 0-1
    reliability: float  # Fraction of non-error responses
    per_suite_quality: dict[str, float] = field(default_factory=dict)
    routing_distribution: dict[str, float] = field(default_factory=dict)
    n_questions: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def objectives(self) -> tuple[float, float, float, float]:
        return (self.quality, self.speed, -self.cost, self.reliability)


@dataclass
class Baseline:
    quality: float = 2.0
    speed: float = 10.0
    cost: float = 0.5
    reliability: float = 0.9
    per_suite_quality: dict[str, float] = field(default_factory=dict)
    frontdoor_speed: float = 10.0

    @classmethod
    def load(cls, path: Path | None = None) -> Baseline:
        path = path or DEFAULT_BASELINE_PATH
        if not path.exists():
            log.warning("No baseline file at %s, using defaults", path)
            return cls()
        data = yaml.safe_load(path.read_text())
        return cls(
            quality=data.get("quality", 2.0),
            speed=data.get("speed", 10.0),
            cost=data.get("cost", 0.5),
            reliability=data.get("reliability", 0.9),
            per_suite_quality=data.get("per_suite_quality", {}),
            frontdoor_speed=data.get("frontdoor_speed", 10.0),
        )

    def save(self, path: Path | None = None) -> None:
        path = path or DEFAULT_BASELINE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "quality": self.quality,
            "speed": self.speed,
            "cost": self.cost,
            "reliability": self.reliability,
            "per_suite_quality": self.per_suite_quality,
            "frontdoor_speed": self.frontdoor_speed,
        }
        path.write_text(yaml.dump(data, default_flow_style=False))


class SafetyGate:
    """Enforces safety constraints on trial results."""

    def __init__(self, baseline_path: Path | None = None):
        self.baseline = Baseline.load(baseline_path)
        self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def check(self, result: EvalResult) -> SafetyVerdict:
        """Run all safety checks on an eval result."""
        violations = []
        warnings = []

        # 1. Quality floor
        if result.quality < QUALITY_FLOOR:
            violations.append(
                f"Quality floor violation: {result.quality:.3f} < {QUALITY_FLOOR}"
            )

        # 2. Regression vs baseline
        delta = result.quality - self.baseline.quality
        if delta < REGRESSION_THRESHOLD:
            violations.append(
                f"Quality regression: {delta:+.3f} vs baseline "
                f"(threshold: {REGRESSION_THRESHOLD})"
            )
        elif delta < 0:
            warnings.append(f"Slight quality drop: {delta:+.3f} vs baseline")

        # 3. Per-suite regression
        for suite, quality in result.per_suite_quality.items():
            baseline_q = self.baseline.per_suite_quality.get(suite)
            if baseline_q is not None:
                suite_delta = quality - baseline_q
                if suite_delta < PER_SUITE_REGRESSION:
                    violations.append(
                        f"Suite '{suite}' regression: {suite_delta:+.3f} "
                        f"(threshold: {PER_SUITE_REGRESSION})"
                    )

        # 4. Routing diversity
        architect_frac = result.routing_distribution.get("architect", 0.0)
        if architect_frac > ARCHITECT_ROUTING_CAP:
            violations.append(
                f"Routing diversity violation: {architect_frac:.1%} architect-tier "
                f"(cap: {ARCHITECT_ROUTING_CAP:.0%})"
            )

        # 5. Throughput floor
        if result.speed < self.baseline.frontdoor_speed * 0.8:
            violations.append(
                f"Throughput floor: {result.speed:.1f} t/s < "
                f"{self.baseline.frontdoor_speed * 0.8:.1f} t/s "
                f"(80% of baseline {self.baseline.frontdoor_speed:.1f})"
            )
        elif result.speed < self.baseline.frontdoor_speed * 0.9:
            warnings.append(
                f"Speed marginal: {result.speed:.1f} t/s "
                f"({result.speed / self.baseline.frontdoor_speed:.0%} of baseline)"
            )

        passed = len(violations) == 0
        verdict = SafetyVerdict(passed=passed, violations=violations, warnings=warnings)

        # Track consecutive failures
        if not passed:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        return verdict

    def should_rollback(self) -> bool:
        """True if consecutive failures exceed threshold."""
        return self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    def update_baseline(self, result: EvalResult) -> None:
        """Update baseline with new production-best metrics."""
        self.baseline.quality = result.quality
        self.baseline.speed = result.speed
        self.baseline.cost = result.cost
        self.baseline.reliability = result.reliability
        self.baseline.per_suite_quality.update(result.per_suite_quality)
        self.baseline.save()
        log.info("Baseline updated: q=%.3f s=%.1f", result.quality, result.speed)

    def reset_failures(self) -> None:
        self._consecutive_failures = 0
