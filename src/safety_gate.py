"""SafetyGate — eval-trust boundary for autopilot experiment acceptance.

Implements EV-8's amended two-tier warn/reject policy (NIB2-42,
handoffs/active/eval-tower-verification.md L201-226).

The gate never blocks inference; it decides whether a Pareto candidate
should be admitted into the frontier. Quality floor and per-suite
regression rules remain inviolate; the new diversity signals are
advisory until ``safety_gate_warn_only=False``.

Tier 1 WARN triggers when:
  - distinct2 drop vs baseline > 0.20 AND quality_delta <= 0

Tier 2 REJECT triggers when ALL of:
  - distinct2 drop  > 0.20
  - semantic_embedding_agreement drop > 0.10
  - quality_delta <= 0
  - VS recovery ratio < 0.50

Warn-only mode (``SAFETY_GATE_WARN_ONLY=1``, default) logs the WARN but
never actually rejects, until Verbalized Sampling replication on
Qwen3-30B-A3B produces baseline recovery data.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class Verdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    REJECT = "reject"


@dataclass
class EvalResult:
    """Metrics for a single evaluation trial.

    Core 4 metrics mirror the existing Pareto archive contract. NIB2-42
    adds 5 diversity fields, all NaN-safe (NaN means "signal unavailable
    this trial", not "zero diversity").
    """

    quality: float
    speed: float
    cost: float
    reliability: float

    # NIB2-42 diversity fields (EV-8 amended).
    diversity_entropy: float = math.nan
    diversity_distinct2: float = math.nan
    diversity_self_bleu: float = math.nan
    diversity_ttr: float = math.nan
    diversity_semantic_embedding_agreement: float = math.nan

    # Optional VS recovery ratio for this trial, in [0, 1]. NaN means
    # the probe was not run; SafetyGate treats that as "cannot confirm
    # recovery" → favours WARN over REJECT in multi-signal decisions.
    vs_recovery_ratio: float = math.nan

    # Free-form annotations (trial_id, commit, role, etc.).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateVerdict:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    deltas: dict[str, float] = field(default_factory=dict)
    warn_only_active: bool = False


# EV-8 amended thresholds (handoff L217-221).
DISTINCT2_DROP_THRESHOLD = 0.20
SEMANTIC_AGREEMENT_DROP_THRESHOLD = 0.10
VS_RECOVERY_THRESHOLD = 0.50


class SafetyGate:
    """Evaluate trials against a fixed baseline, emitting PASS / WARN / REJECT."""

    def __init__(
        self,
        baseline: EvalResult,
        warn_only: bool | None = None,
        distinct2_drop_threshold: float = DISTINCT2_DROP_THRESHOLD,
        semantic_drop_threshold: float = SEMANTIC_AGREEMENT_DROP_THRESHOLD,
        vs_recovery_threshold: float = VS_RECOVERY_THRESHOLD,
    ) -> None:
        self.baseline = baseline
        self.distinct2_drop_threshold = distinct2_drop_threshold
        self.semantic_drop_threshold = semantic_drop_threshold
        self.vs_recovery_threshold = vs_recovery_threshold

        if warn_only is None:
            env_val = os.environ.get("SAFETY_GATE_WARN_ONLY", "1").strip().lower()
            warn_only = env_val in ("1", "true", "yes", "on")
        self.warn_only = warn_only

    # ── public API ─────────────────────────────────────────────────

    def evaluate(self, result: EvalResult) -> GateVerdict:
        """Return PASS / WARN / REJECT for a trial result."""
        reasons: list[str] = []
        deltas = self._deltas(result)

        quality_up = deltas["quality_delta"] > 0
        distinct2_big_drop = self._is_drop(
            self.baseline.diversity_distinct2,
            result.diversity_distinct2,
            self.distinct2_drop_threshold,
        )
        semantic_big_drop = self._is_drop(
            self.baseline.diversity_semantic_embedding_agreement,
            result.diversity_semantic_embedding_agreement,
            self.semantic_drop_threshold,
        )

        # Tier 2 REJECT — multi-signal. Needs ALL four conditions.
        vs_recovery = result.vs_recovery_ratio
        vs_failed_to_recover = (
            not math.isnan(vs_recovery) and vs_recovery < self.vs_recovery_threshold
        )

        tier2 = (
            distinct2_big_drop
            and semantic_big_drop
            and not quality_up
            and vs_failed_to_recover
        )

        # Tier 1 WARN — single-signal with quality.
        tier1 = distinct2_big_drop and not quality_up

        if tier2:
            reasons.append(
                f"distinct-2 drop > {self.distinct2_drop_threshold:.0%}; "
                f"semantic-embedding-agreement drop > {self.semantic_drop_threshold:.0%}; "
                f"quality not up; VS recovery {vs_recovery:.2f} < {self.vs_recovery_threshold}"
            )
            verdict = Verdict.WARN if self.warn_only else Verdict.REJECT
            return GateVerdict(
                verdict=verdict,
                reasons=reasons,
                deltas=deltas,
                warn_only_active=self.warn_only,
            )

        if tier1:
            reasons.append(
                f"distinct-2 drop > {self.distinct2_drop_threshold:.0%} AND quality not up "
                f"(tier 1 warn; multi-signal check below rejection threshold)"
            )
            return GateVerdict(
                verdict=Verdict.WARN,
                reasons=reasons,
                deltas=deltas,
                warn_only_active=self.warn_only,
            )

        return GateVerdict(
            verdict=Verdict.PASS,
            reasons=reasons,
            deltas=deltas,
            warn_only_active=self.warn_only,
        )

    # ── helpers ────────────────────────────────────────────────────

    def _deltas(self, result: EvalResult) -> dict[str, float]:
        """Compute per-field Δ (result - baseline). NaN-safe."""
        out = {}
        for field_ in (
            "quality", "speed", "cost", "reliability",
            "diversity_entropy", "diversity_distinct2", "diversity_self_bleu",
            "diversity_ttr", "diversity_semantic_embedding_agreement",
        ):
            b = getattr(self.baseline, field_)
            r = getattr(result, field_)
            if math.isnan(b) or math.isnan(r):
                out[f"{field_}_delta"] = math.nan
            else:
                out[f"{field_}_delta"] = r - b
        return out

    @staticmethod
    def _is_drop(baseline: float, current: float, threshold: float) -> bool:
        """True when ``baseline`` is defined and ``current`` is at least
        ``threshold`` fractionally lower. NaN baseline → signal unavailable,
        returns False (signal can't fire)."""
        if math.isnan(baseline) or math.isnan(current):
            return False
        if baseline == 0:
            return False
        rel_drop = (baseline - current) / abs(baseline)
        return rel_drop > threshold
