"""Meta-optimizer: self-improvement every N trials.

Rebalances species budgets, detects stagnation, adjusts search strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("autopilot.meta")

DEFAULT_INTERVAL = 50  # Rebalance every N trials


@dataclass
class SpeciesBudget:
    """Budget allocation for each species (sums to 1.0)."""
    seeder: float = 0.40  # Default: 40% seeding (backbone)
    numeric_swarm: float = 0.25  # 25% numeric optimization
    prompt_forge: float = 0.20  # 20% prompt mutation
    structural_lab: float = 0.15  # 15% structural experiments

    def as_dict(self) -> dict[str, float]:
        return {
            "seeder": self.seeder,
            "numeric_swarm": self.numeric_swarm,
            "prompt_forge": self.prompt_forge,
            "structural_lab": self.structural_lab,
        }

    def normalize(self) -> None:
        total = self.seeder + self.numeric_swarm + self.prompt_forge + self.structural_lab
        if total > 0:
            self.seeder /= total
            self.numeric_swarm /= total
            self.prompt_forge /= total
            self.structural_lab /= total


class MetaOptimizer:
    """Rebalances species budgets and detects optimization stagnation."""

    def __init__(self, interval: int = DEFAULT_INTERVAL):
        self.interval = interval
        self.budget = SpeciesBudget()

    def should_rebalance(self, trial_id: int) -> bool:
        return trial_id > 0 and trial_id % self.interval == 0

    def rebalance(
        self,
        species_effectiveness: dict[str, dict[str, float]],
        hv_slope: float,
        memory_count: int,
        is_converged: bool,
    ) -> SpeciesBudget:
        """Rebalance species budgets based on effectiveness and state.

        Args:
            species_effectiveness: {species: {total, pareto, rate}}
            hv_slope: Hypervolume trend slope (stagnation indicator)
            memory_count: Current routing memory count
            is_converged: Whether Q-values have converged
        """
        old = self.budget.as_dict()

        # Phase-based adjustments
        if memory_count < 500:
            # Phase: seeding priority
            self.budget.seeder = 0.60
            self.budget.numeric_swarm = 0.15
            self.budget.prompt_forge = 0.15
            self.budget.structural_lab = 0.10
            log.info("Meta: seeding phase (memories=%d < 500)", memory_count)

        elif is_converged and memory_count >= 500:
            # Phase: training + structural experiments
            self.budget.seeder = 0.20
            self.budget.numeric_swarm = 0.25
            self.budget.prompt_forge = 0.20
            self.budget.structural_lab = 0.35
            log.info("Meta: training phase (converged, memories=%d)", memory_count)

        else:
            # Phase: balanced optimization
            # Adjust based on effectiveness
            for species, stats in species_effectiveness.items():
                rate = stats.get("rate", 0.0)
                if species == "seeder":
                    self.budget.seeder = max(0.15, 0.30 + rate * 0.2)
                elif species == "numeric_swarm":
                    self.budget.numeric_swarm = max(0.10, 0.20 + rate * 0.2)
                elif species == "prompt_forge":
                    self.budget.prompt_forge = max(0.10, 0.15 + rate * 0.2)
                elif species == "structural_lab":
                    self.budget.structural_lab = max(0.05, 0.10 + rate * 0.2)

        # Stagnation boost: increase exploration
        if hv_slope < 0.001:
            log.info("Meta: stagnation detected (hv_slope=%.6f), boosting exploration", hv_slope)
            # Boost less-used species
            self.budget.prompt_forge = min(0.35, self.budget.prompt_forge + 0.10)
            self.budget.structural_lab = min(0.30, self.budget.structural_lab + 0.10)

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
        }
