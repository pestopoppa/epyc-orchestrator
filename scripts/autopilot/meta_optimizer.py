"""Meta-optimizer: self-improvement every N trials.

Rebalances species budgets, detects stagnation, adjusts search strategy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("autopilot.meta")

DEFAULT_INTERVAL = 50  # Rebalance every N trials
DIVERSITY_HISTORY_LIMIT = 50
DIVERSITY_DISTINCT2_RATIO_THRESHOLD = 0.80
DIVERSITY_SEMANTIC_DROP_THRESHOLD = 0.10
DIVERSITY_VS_RECOVERY_THRESHOLD = 0.50
DIVERSITY_STALL_STREAK_MIN = 10


@dataclass
class SpeciesBudget:
    """Budget allocation for each species (sums to 1.0)."""

    seeder: float = 0.40  # Default: 40% seeding (backbone)
    numeric_swarm: float = 0.25  # 25% numeric optimization
    prompt_forge: float = 0.20  # 20% prompt mutation
    structural_lab: float = 0.10  # 10% structural experiments
    evolution_manager: float = 0.05  # 5% knowledge distillation

    def as_dict(self) -> dict[str, float]:
        return {
            "seeder": self.seeder,
            "numeric_swarm": self.numeric_swarm,
            "prompt_forge": self.prompt_forge,
            "structural_lab": self.structural_lab,
            "evolution_manager": self.evolution_manager,
        }

    def normalize(self) -> None:
        d = self.as_dict()
        total = sum(d.values())
        if total > 0:
            for k, v in d.items():
                setattr(self, k, v / total)


# Per-species token budget caps (prevent runaway cost per iteration)
SPECIES_TOKEN_BUDGETS: dict[str, int] = {
    "seeder": 50_000,  # API-bound, not token-bound
    "numeric_swarm": 10_000,  # Optuna is cheap
    "prompt_forge": 200_000,  # Claude CLI mutation is expensive
    "structural_lab": 30_000,  # Moderate — training scripts
    "evolution_manager": 100_000,  # LLM summarization
}


def get_token_budget(species: str) -> int:
    """Return the token budget cap for a species."""
    return SPECIES_TOKEN_BUDGETS.get(species, 50_000)


class MetaOptimizer:
    """Rebalances species budgets and detects optimization stagnation."""

    def __init__(self, interval: int = DEFAULT_INTERVAL):
        self.interval = interval
        self.budget = SpeciesBudget()
        self.diversity_stall_state = self._default_diversity_stall_state()

    @staticmethod
    def _default_diversity_stall_state() -> dict[str, Any]:
        return {
            "schema_version": "ap37_diversity_stall.v1",
            "distinct2_baseline": None,
            "semantic_embedding_agreement_baseline": None,
            "distinct2_history": [],
            "consecutive_trigger_count": 0,
            "rebalance_recommended": False,
            "last_status": "unobserved",
            "last_reason": "",
        }

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    def restore_diversity_state(self, raw: dict[str, Any] | None) -> None:
        """Restore AP-37 diversity-stall state from persisted autopilot state."""
        state = self._default_diversity_stall_state()
        if isinstance(raw, dict):
            state.update({key: raw.get(key) for key in state if key in raw})
            history = raw.get("distinct2_history")
            state["distinct2_history"] = history if isinstance(history, list) else []
            state["consecutive_trigger_count"] = int(
                max(0, self._finite_float(raw.get("consecutive_trigger_count")) or 0)
            )
            state["rebalance_recommended"] = bool(raw.get("rebalance_recommended", False))
        self.diversity_stall_state = state

    def export_diversity_state(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of AP-37 diversity-stall state."""
        state = dict(self.diversity_stall_state)
        history = state.get("distinct2_history")
        state["distinct2_history"] = list(history) if isinstance(history, list) else []
        return state

    def should_rebalance(self, trial_id: int) -> bool:
        return trial_id > 0 and trial_id % self.interval == 0

    def observe_diversity(
        self,
        *,
        trial_id: int,
        distinct2: Any,
        semantic_embedding_agreement: Any,
        vs_recovery_ratio: Any = None,
        distinct2_baseline: Any = None,
        semantic_embedding_agreement_baseline: Any = None,
    ) -> dict[str, Any]:
        """Record AP-37 diversity-stall evidence and return the latest report.

        The trigger is intentionally multi-signal and baseline-gated:
        distinct-2 must drop below 80% of its baseline, semantic agreement must
        drop by more than 0.10, and the Verbalized Sampling recovery probe must
        fail to close at least half the diversity gap for 10 consecutive
        observations. Missing baselines or missing VS evidence record diagnostics
        but do not recommend a rebalance.
        """
        state = self.diversity_stall_state
        d2 = self._finite_float(distinct2)
        semantic = self._finite_float(semantic_embedding_agreement)
        vs_recovery = self._finite_float(vs_recovery_ratio)
        d2_baseline = self._finite_float(distinct2_baseline)
        semantic_baseline = self._finite_float(semantic_embedding_agreement_baseline)
        if d2_baseline is not None:
            state["distinct2_baseline"] = d2_baseline
        else:
            d2_baseline = self._finite_float(state.get("distinct2_baseline"))
        if semantic_baseline is not None:
            state["semantic_embedding_agreement_baseline"] = semantic_baseline
        else:
            semantic_baseline = self._finite_float(
                state.get("semantic_embedding_agreement_baseline")
            )

        status = "ok"
        reason = ""
        distinct2_ratio: float | None = None
        semantic_delta: float | None = None
        low_distinct2 = False
        low_semantic = False
        vs_failed = False
        trigger = False
        if d2 is None or semantic is None:
            status = "signal_missing"
            reason = "distinct2 or semantic_embedding_agreement unavailable"
        elif d2_baseline is None or semantic_baseline is None or d2_baseline <= 0.0:
            status = "baseline_missing"
            reason = "diversity baseline unavailable; EV-8 baseline remains inference-gated"
        elif vs_recovery is None:
            status = "vs_missing"
            reason = "VS recovery ratio unavailable; AP-37 trigger requires recovery failure"
            distinct2_ratio = d2 / d2_baseline
            semantic_delta = semantic - semantic_baseline
            low_distinct2 = distinct2_ratio < DIVERSITY_DISTINCT2_RATIO_THRESHOLD
            low_semantic = semantic_delta < -DIVERSITY_SEMANTIC_DROP_THRESHOLD
        else:
            distinct2_ratio = d2 / d2_baseline
            semantic_delta = semantic - semantic_baseline
            low_distinct2 = distinct2_ratio < DIVERSITY_DISTINCT2_RATIO_THRESHOLD
            low_semantic = semantic_delta < -DIVERSITY_SEMANTIC_DROP_THRESHOLD
            vs_failed = vs_recovery < DIVERSITY_VS_RECOVERY_THRESHOLD
            trigger = low_distinct2 and low_semantic and vs_failed
            if trigger:
                status = "trigger_observed"
                reason = "diversity stall trigger observed"

        if trigger:
            state["consecutive_trigger_count"] = (
                int(state.get("consecutive_trigger_count") or 0) + 1
            )
        else:
            state["consecutive_trigger_count"] = 0
        rebalance_recommended = (
            int(state.get("consecutive_trigger_count") or 0) >= DIVERSITY_STALL_STREAK_MIN
        )
        if rebalance_recommended:
            status = "rebalance_recommended"
            reason = (
                f"diversity stall trigger held for {state['consecutive_trigger_count']} "
                "consecutive observations"
            )

        observation = {
            "trial_id": int(trial_id),
            "distinct2": d2,
            "semantic_embedding_agreement": semantic,
            "vs_recovery_ratio": vs_recovery,
            "distinct2_baseline": d2_baseline,
            "semantic_embedding_agreement_baseline": semantic_baseline,
            "distinct2_ratio": distinct2_ratio,
            "semantic_delta": semantic_delta,
            "low_distinct2": low_distinct2,
            "low_semantic_embedding_agreement": low_semantic,
            "vs_failed_to_recover": vs_failed,
            "trigger": trigger,
            "consecutive_trigger_count": state["consecutive_trigger_count"],
            "rebalance_recommended": rebalance_recommended,
            "status": status,
            "reason": reason,
        }
        history = state.setdefault("distinct2_history", [])
        if not isinstance(history, list):
            history = []
            state["distinct2_history"] = history
        history.append(observation)
        del history[:-DIVERSITY_HISTORY_LIMIT]
        state["rebalance_recommended"] = rebalance_recommended
        state["last_status"] = status
        state["last_reason"] = reason
        return observation

    def diversity_rebalance_due(self) -> bool:
        """Whether the AP-37 diversity-stall streak currently recommends rebalance."""
        return bool(self.diversity_stall_state.get("rebalance_recommended", False))

    def rebalance(
        self,
        species_effectiveness: dict[str, dict[str, float]],
        hv_slope: float,
        memory_count: int,
        is_converged: bool,
        diversity_stall: bool | dict[str, Any] = False,
    ) -> SpeciesBudget:
        """Rebalance species budgets based on effectiveness and state.

        Args:
            species_effectiveness: {species: {total, pareto, rate, budget_rate}}.
                ``rate`` is the legacy Pareto-frontier rate; ``budget_rate`` is
                realized information gain from PEAF surprise when available,
                falling back to ``rate`` for legacy journals.
            hv_slope: Hypervolume trend slope (stagnation indicator)
            memory_count: Current routing memory count
            is_converged: Whether Q-values have converged
        """
        old = self.budget.as_dict()

        # Phase-based adjustments
        if memory_count < 500:
            # Phase: seeding priority
            self.budget.seeder = 0.55
            self.budget.numeric_swarm = 0.15
            self.budget.prompt_forge = 0.15
            self.budget.structural_lab = 0.10
            self.budget.evolution_manager = 0.05
            log.info("Meta: seeding phase (memories=%d < 500)", memory_count)

        elif is_converged and memory_count >= 500:
            # Phase: training + structural experiments
            self.budget.seeder = 0.15
            self.budget.numeric_swarm = 0.25
            self.budget.prompt_forge = 0.20
            self.budget.structural_lab = 0.30
            self.budget.evolution_manager = 0.10
            log.info("Meta: training phase (converged, memories=%d)", memory_count)

        else:
            # Phase: balanced optimization
            # Adjust based on effectiveness
            for species, stats in species_effectiveness.items():
                rate = stats.get("budget_rate", stats.get("rate", 0.0))
                if species == "seeder":
                    self.budget.seeder = max(0.15, 0.30 + rate * 0.2)
                elif species == "numeric_swarm":
                    self.budget.numeric_swarm = max(0.10, 0.20 + rate * 0.2)
                elif species == "prompt_forge":
                    self.budget.prompt_forge = max(0.10, 0.15 + rate * 0.2)
                elif species == "structural_lab":
                    self.budget.structural_lab = max(0.05, 0.10 + rate * 0.2)
                elif species == "evolution_manager":
                    self.budget.evolution_manager = max(0.03, 0.05 + rate * 0.1)

        # Stagnation boost: increase exploration
        if hv_slope < 0.001:
            log.info("Meta: stagnation detected (hv_slope=%.6f), boosting exploration", hv_slope)
            # Boost less-used species
            self.budget.prompt_forge = min(0.35, self.budget.prompt_forge + 0.10)
            self.budget.structural_lab = min(0.30, self.budget.structural_lab + 0.10)

        diversity_due = (
            bool(diversity_stall.get("rebalance_recommended"))
            if isinstance(diversity_stall, dict)
            else bool(diversity_stall)
        )
        if diversity_due:
            log.info("Meta: AP-37 diversity stall detected, boosting mutation exploration")
            self.budget.prompt_forge = min(0.40, self.budget.prompt_forge + 0.10)
            self.budget.structural_lab = min(0.35, self.budget.structural_lab + 0.05)

        self.budget.normalize()

        new = self.budget.as_dict()
        changes = {k: f"{old[k]:.2f} → {new[k]:.2f}" for k in old if abs(old[k] - new[k]) > 0.01}
        if changes:
            log.info("Meta: budget rebalanced: %s", changes)

        return self.budget

    def select_species(self, budget: SpeciesBudget | None = None) -> str:
        """Select next species based on budget weights (weighted random)."""
        import random

        b = budget or self.budget
        species = list(b.as_dict().keys())
        weights = list(b.as_dict().values())
        return random.choices(species, weights=weights, k=1)[0]

    def detect_stagnation(self, hv_slope: float, threshold: float = 0.001) -> bool:
        """True if hypervolume improvement is below threshold."""
        return hv_slope < threshold

    def parameter_importance(self, numeric_swarm) -> dict[str, dict[str, float]]:
        """Get parameter importance across all surfaces."""
        result = {}
        for surface in numeric_swarm.SURFACES:
            importance = numeric_swarm.importance(surface)
            if importance:
                result[surface] = importance
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "budget": self.budget.as_dict(),
            "interval": self.interval,
            "diversity_stall": self.export_diversity_state(),
        }
