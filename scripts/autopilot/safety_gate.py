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
# T0 thresholds (10 sentinel questions — inflated scale, saturates at 3.0)
QUALITY_FLOOR_T0 = 2.0  # Average quality >= 2.0/3.0
# T1/T2 thresholds (50-500 real benchmark questions — honest signal)
QUALITY_FLOOR_T1 = 1.0  # ~33% correct minimum
REGRESSION_THRESHOLD = -0.05  # Max quality drop vs baseline (fraction of baseline)
PER_SUITE_REGRESSION = -0.1  # Max per-suite quality drop
ARCHITECT_ROUTING_CAP = 0.80  # Max fraction routed to architect-tier
MAX_CONSECUTIVE_FAILURES = 3  # Auto-rollback after this many failures


@dataclass
class SafetyVerdict:
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)  # AP-14: deficiency categories

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
    instruction_token_count: int = 0  # AP-16: per-request instruction overhead
    instruction_token_ratio: float = 0.0  # AP-16: instruction_tokens / total_input_tokens
    partial_count: int = 0  # Inference results with partial=True (read_timeout_partial)
    degraded_count: int = 0  # Inference results with degraded=True
    # AM KV compaction telemetry (populated when compact action is used)
    avg_prompt_tokens: float = 0.0  # Average context length across results
    compaction_events: int = 0  # Number of compacted slots in this eval
    # EV-2: Calibration metrics (from eval-tower-verification.md)
    ece: float = 0.0  # Expected Calibration Error (10-bin). Lower = better calibrated.
    auroc: float = 0.0  # Area Under ROC Curve. Higher = better discrimination. 0 if degenerate.
    calibration_violations: int = 0  # Questions where |confidence - correctness| > 0.5
    # Branching density: fraction of reasoning steps that are divergent/exploratory.
    # From intake-378 deep-dive: high branching (>0.30) = unproductive exploration.
    # 0.0 when no <think> blocks are present in eval answers.
    branching_density: float = 0.0

    @property
    def objectives(self) -> tuple[float, float, float, float]:
        return (self.quality, self.speed, -self.cost, self.reliability)

    def to_grep_lines(self, trial_id: int = 0, species: str = "") -> str:
        """AP-13: Grep-parseable key: value output.

        Designed for `grep 'METRIC' autopilot.log | awk -F': '` extraction.
        """
        lines = [
            f"METRIC trial: {trial_id}",
            f"METRIC species: {species}",
            f"METRIC tier: {self.tier}",
            f"METRIC quality: {self.quality:.4f}",
            f"METRIC speed: {self.speed:.2f}",
            f"METRIC cost: {self.cost:.4f}",
            f"METRIC reliability: {self.reliability:.4f}",
            f"METRIC n_questions: {self.n_questions}",
        ]
        for suite, q in sorted(self.per_suite_quality.items()):
            lines.append(f"METRIC suite_{suite}: {q:.4f}")
        for role, frac in sorted(self.routing_distribution.items()):
            lines.append(f"METRIC route_{role}: {frac:.4f}")
        # AP-16: Instruction token budget
        lines.append(f"METRIC instruction_tokens: {self.instruction_token_count}")
        lines.append(f"METRIC instruction_ratio: {self.instruction_token_ratio:.4f}")
        # Degradation metrics from refactored InferenceResult
        if self.partial_count > 0:
            lines.append(f"METRIC partial_count: {self.partial_count}")
        if self.degraded_count > 0:
            lines.append(f"METRIC degraded_count: {self.degraded_count}")
        # EV-2: Calibration metrics
        lines.append(f"METRIC ece: {self.ece:.4f}")
        if self.auroc > 0:
            lines.append(f"METRIC auroc: {self.auroc:.4f}")
        if self.calibration_violations > 0:
            lines.append(f"METRIC calibration_violations: {self.calibration_violations}")
        # Branching density (intake-378)
        if self.branching_density > 0:
            lines.append(f"METRIC branching_density: {self.branching_density:.4f}")
        # AM compaction telemetry
        if self.avg_prompt_tokens > 0:
            lines.append(f"METRIC avg_prompt_tokens: {self.avg_prompt_tokens:.0f}")
        if self.compaction_events > 0:
            lines.append(f"METRIC compaction_events: {self.compaction_events}")
        return "\n".join(lines)


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

    def __init__(
        self,
        baseline_path: Path | None = None,
        consecutive_failures: int = 0,
    ):
        self.baseline = Baseline.load(baseline_path)
        self._consecutive_failures = consecutive_failures

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def check(self, result: EvalResult) -> SafetyVerdict:
        """Run all safety checks on an eval result."""
        violations = []
        warnings = []
        categories = []  # AP-14: track which checks failed

        # 1. Quality floor (tier-aware)
        quality_floor = QUALITY_FLOOR_T0 if result.tier == 0 else QUALITY_FLOOR_T1
        if result.quality < quality_floor:
            violations.append(
                f"Quality floor violation: {result.quality:.3f} < {quality_floor} (tier {result.tier})"
            )
            categories.append("quality_floor")

        # 2. Regression vs baseline (relative: allow 5% drop from baseline)
        baseline_q = self.baseline.quality
        if baseline_q > 0:
            relative_delta = (result.quality - baseline_q) / baseline_q
            if relative_delta < REGRESSION_THRESHOLD:
                violations.append(
                    f"Quality regression: {result.quality:.3f} vs baseline {baseline_q:.3f} "
                    f"({relative_delta:+.1%}, threshold: {REGRESSION_THRESHOLD:+.0%})"
                )
                categories.append("regression")
            elif relative_delta < 0:
                warnings.append(
                    f"Slight quality drop: {result.quality:.3f} vs baseline {baseline_q:.3f} "
                    f"({relative_delta:+.1%})"
                )

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
                    if "per_suite_regression" not in categories:
                        categories.append("per_suite_regression")

        # 4. Routing diversity
        architect_frac = result.routing_distribution.get("architect", 0.0)
        if architect_frac > ARCHITECT_ROUTING_CAP:
            violations.append(
                f"Routing diversity violation: {architect_frac:.1%} architect-tier "
                f"(cap: {ARCHITECT_ROUTING_CAP:.0%})"
            )
            categories.append("routing_diversity")

        # 5. Throughput floor
        if result.speed < self.baseline.frontdoor_speed * 0.8:
            violations.append(
                f"Throughput floor: {result.speed:.1f} t/s < "
                f"{self.baseline.frontdoor_speed * 0.8:.1f} t/s "
                f"(80% of baseline {self.baseline.frontdoor_speed:.1f})"
            )
            categories.append("throughput")
        elif result.speed < self.baseline.frontdoor_speed * 0.9:
            warnings.append(
                f"Speed marginal: {result.speed:.1f} t/s "
                f"({result.speed / self.baseline.frontdoor_speed:.0%} of baseline)"
            )

        # 6. Proxy-only improvement detection (skeptical re-questioning)
        warnings.extend(self._proxy_check(result))

        passed = len(violations) == 0
        verdict = SafetyVerdict(passed=passed, violations=violations, warnings=warnings, categories=categories)

        # Track consecutive failures
        if not passed:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        return verdict

    def _proxy_check(self, result: EvalResult) -> list[str]:
        """Detect proxy-only improvements: quality up but concentrated in easy suites.

        Returns warnings (not violations) — these are suspicious but not blocking.
        Flags cases where overall quality improved but only 1 suite drove the gain
        while other suites declined.  (GPD "skeptical re-questioning" pattern.)
        """
        warnings: list[str] = []
        if not result.per_suite_quality or not self.baseline.per_suite_quality:
            return warnings

        improved: list[tuple[str, float]] = []
        declined: list[tuple[str, float]] = []
        for suite, q in result.per_suite_quality.items():
            bq = self.baseline.per_suite_quality.get(suite)
            if bq is None:
                continue
            delta = q - bq
            if delta > 0.05:
                improved.append((suite, delta))
            elif delta < -0.02:
                declined.append((suite, delta))

        # Flag if gains concentrated in ≤1 suite while others declined
        if improved and declined and len(improved) <= 1:
            imp_str = ", ".join(f"{s} +{d:.2f}" for s, d in improved)
            dec_str = ", ".join(f"{s} {d:+.2f}" for s, d in declined)
            warnings.append(
                f"Proxy-only improvement: gains in [{imp_str}] "
                f"but declines in [{dec_str}]"
            )
        return warnings

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

    @staticmethod
    def analyze_failure(result: EvalResult, verdict: SafetyVerdict) -> str:
        """Build a structured failure narrative from safety verdict and eval result.

        Pure rule-based (no LLM). Returns empty string if verdict passed.
        """
        if verdict.passed:
            return ""

        sections: list[str] = []

        # VIOLATIONS
        if verdict.violations:
            lines = ["VIOLATIONS:"]
            for v in verdict.violations:
                lines.append(f"  - {v}")
            sections.append("\n".join(lines))

        # DEGRADED SUITES (per-suite quality below floor)
        quality_floor = QUALITY_FLOOR_T0 if result.tier == 0 else QUALITY_FLOOR_T1
        degraded = [
            (suite, q)
            for suite, q in result.per_suite_quality.items()
            if q < quality_floor
        ]
        if degraded:
            lines = ["DEGRADED SUITES:"]
            for suite, q in sorted(degraded, key=lambda x: x[1]):
                lines.append(f"  - {suite}: {q:.3f} (floor: {quality_floor})")
            sections.append("\n".join(lines))

        # ROUTING IMBALANCE (>60% to one tier)
        for tier_name, frac in result.routing_distribution.items():
            if frac > 0.6:
                sections.append(
                    f"ROUTING IMBALANCE:\n  - {tier_name}: {frac:.1%} of requests"
                )

        # WARNINGS
        if verdict.warnings:
            lines = ["WARNINGS:"]
            for w in verdict.warnings:
                lines.append(f"  - {w}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
