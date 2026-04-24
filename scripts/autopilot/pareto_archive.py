"""Pareto archive: 4D non-dominated sorting with hypervolume indicator.

Objectives: quality (↑), speed (↑), -cost (↑ i.e. lower cost is better), reliability (↑).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_STATE_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "autopilot_state.json"
)

# Reference point for hypervolume (worst acceptable values)
# Quality: 0, Speed: 0 t/s, Cost: -1.0 (high), Reliability: 0
REFERENCE_POINT = (0.0, 0.0, -1.0, 0.0)


@dataclass
class ParetoEntry:
    trial_id: int
    objectives: tuple[float, float, float, float]  # (quality, speed, -cost, reliability)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    git_tag: str = ""
    eval_tier: int = 0
    reasoning: str = ""
    parent_trial: int | None = None
    memory_count: int = 0
    active_flags: list[str] = field(default_factory=list)
    species: str = ""
    timestamp: str = ""
    is_production_best: bool = False

    def dominates(self, other: ParetoEntry) -> bool:
        """True if self dominates other (>= on all, > on at least one)."""
        dominated = False
        for a, b in zip(self.objectives, other.objectives):
            if a < b:
                return False
            if a > b:
                dominated = True
        return dominated

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["objectives"] = list(d["objectives"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParetoEntry:
        d = dict(d)
        d["objectives"] = tuple(d["objectives"])
        return cls(**d)


class ParetoArchive:
    """4D Pareto frontier with hypervolume tracking and genealogy."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or DEFAULT_STATE_PATH
        self._frontier: list[ParetoEntry] = []
        self._all_entries: list[ParetoEntry] = []
        self._hypervolume_history: list[tuple[int, float]] = []  # (trial_id, hv)
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        archive_data = data.get("pareto_archive", {})
        self._frontier = [
            ParetoEntry.from_dict(e) for e in archive_data.get("frontier", [])
        ]
        self._all_entries = [
            ParetoEntry.from_dict(e) for e in archive_data.get("all_entries", [])
        ]
        self._hypervolume_history = [
            tuple(h) for h in archive_data.get("hypervolume_history", [])
        ]

        # Integrity check: detect lost frontier
        trial_counter = data.get("trial_counter", 0)
        if trial_counter > 10 and not self._frontier and not self._all_entries:
            log.error(
                "PARETO FRONTIER LOST: trial_counter=%d but frontier is empty. "
                "This means autopilot_state.json was not checkpointed or was "
                "overwritten. Restore from a checkpoint that includes "
                "autopilot_state.json, or reconstruct from autopilot.log.",
                trial_counter,
            )
            raise RuntimeError(
                f"Pareto frontier empty at trial {trial_counter}. "
                f"Refusing to start — would discard all prior optimization. "
                f"Restore from checkpoint or reconstruct from logs."
            )

    def save(self, state: dict[str, Any] | None = None) -> None:
        """Save archive to state file, merging with existing state."""
        if self.state_path.exists():
            existing = json.loads(self.state_path.read_text())
        else:
            existing = {}
        if state:
            existing.update(state)
        existing["pareto_archive"] = {
            "frontier": [e.to_dict() for e in self._frontier],
            "all_entries": [e.to_dict() for e in self._all_entries],
            "hypervolume_history": list(self._hypervolume_history),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(existing, indent=2, default=str))

    # ── core operations ─────────────────────────────────────────

    def is_pareto_candidate(self, objectives: tuple[float, ...]) -> bool:
        """Check if objectives would be non-dominated by current frontier."""
        entry = ParetoEntry(trial_id=-1, objectives=objectives)
        for f in self._frontier:
            if f.dominates(entry):
                return False
        return True

    def update(self, entry: ParetoEntry) -> str:
        """Add entry to archive. Returns 'frontier', 'candidate', or 'dominated'."""
        self._all_entries.append(entry)

        # Check if dominated by any frontier entry
        if not self.is_pareto_candidate(entry.objectives):
            status = "dominated"
        else:
            # Remove entries dominated by the new one
            self._frontier = [
                f for f in self._frontier if not entry.dominates(f)
            ]
            self._frontier.append(entry)
            status = "frontier"

        # Update hypervolume
        hv = self.hypervolume()
        self._hypervolume_history.append((entry.trial_id, hv))
        return status

    def frontier(self) -> list[ParetoEntry]:
        return list(self._frontier)

    def frontier_size(self) -> int:
        return len(self._frontier)

    def production_best(self) -> ParetoEntry | None:
        for e in self._frontier:
            if e.is_production_best:
                return e
        return None

    def mark_production_best(self, trial_id: int) -> None:
        for e in self._frontier:
            e.is_production_best = e.trial_id == trial_id

    # ── hypervolume ──────────────────────────────────────────────

    def hypervolume(self, ref: tuple[float, ...] | None = None) -> float:
        """Compute hypervolume indicator for current frontier.

        Uses inclusion-exclusion for 4D (exact, fast enough for <1000 entries).
        """
        if not self._frontier:
            return 0.0
        ref = ref or REFERENCE_POINT
        return _hypervolume_4d(
            [e.objectives for e in self._frontier], ref
        )

    def hypervolume_trend(self, window: int | None = None) -> list[tuple[int, float]]:
        """Return (trial_id, hypervolume) history."""
        if window:
            return list(self._hypervolume_history[-window:])
        return list(self._hypervolume_history)

    def hypervolume_slope(self, window: int = 50) -> float:
        """Linear regression slope of hypervolume over last `window` entries."""
        hist = self._hypervolume_history[-window:]
        if len(hist) < 2:
            return 0.0
        n = len(hist)
        xs = list(range(n))
        ys = [h[1] for h in hist]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        return num / den if den > 0 else 0.0

    # ── genealogy ────────────────────────────────────────────────

    def children_of(self, trial_id: int) -> list[ParetoEntry]:
        return [e for e in self._all_entries if e.parent_trial == trial_id]

    def lineage(self, trial_id: int) -> list[ParetoEntry]:
        """Trace lineage back to root."""
        chain = []
        current = next((e for e in self._all_entries if e.trial_id == trial_id), None)
        while current:
            chain.append(current)
            if current.parent_trial is None:
                break
            current = next(
                (e for e in self._all_entries if e.trial_id == current.parent_trial),
                None,
            )
        return list(reversed(chain))

    # ── summary ──────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        if not self._frontier:
            return {"frontier_size": 0, "hypervolume": 0.0}
        best_quality = max(e.objectives[0] for e in self._frontier)
        best_speed = max(e.objectives[1] for e in self._frontier)
        best_cost = max(e.objectives[2] for e in self._frontier)  # -cost, higher is better
        return {
            "frontier_size": len(self._frontier),
            "total_entries": len(self._all_entries),
            "hypervolume": self.hypervolume(),
            "best_quality": best_quality,
            "best_speed": best_speed,
            "best_neg_cost": best_cost,
            "hv_slope_50": self.hypervolume_slope(50),
        }

    def summary_text(self) -> str:
        s = self.summary()
        lines = [
            f"Pareto frontier: {s['frontier_size']} entries "
            f"(of {s.get('total_entries', 0)} total)",
            f"Hypervolume: {s['hypervolume']:.4f}",
            f"HV slope (last 50): {s.get('hv_slope_50', 0):.6f}",
            f"Best quality: {s.get('best_quality', 0):.3f}",
            f"Best speed: {s.get('best_speed', 0):.1f} t/s",
        ]
        if self._frontier:
            lines.append("\nFrontier entries:")
            for e in sorted(self._frontier, key=lambda x: -x.objectives[0]):
                lines.append(
                    f"  #{e.trial_id} [{e.species}] "
                    f"q={e.objectives[0]:.3f} s={e.objectives[1]:.1f} "
                    f"c={-e.objectives[2]:.3f} r={e.objectives[3]:.2f}"
                    + (" [PROD]" if e.is_production_best else "")
                )
        return "\n".join(lines)


# ── hypervolume computation ──────────────────────────────────────


def _hypervolume_4d(
    points: list[tuple[float, ...]], ref: tuple[float, ...]
) -> float:
    """Exact 4D hypervolume via inclusion-exclusion.

    For small frontiers (<100 entries), this is fast enough.
    Falls back to 2D Monte Carlo approximation for very large frontiers.
    """
    n = len(points)
    if n == 0:
        return 0.0

    # Filter points that dominate the reference point
    valid = []
    for p in points:
        if all(pi > ri for pi, ri in zip(p, ref)):
            valid.append(p)
    if not valid:
        return 0.0

    if n > 100:
        # For large frontiers, use Monte Carlo approximation
        return _hypervolume_monte_carlo(valid, ref, samples=10000)

    # Inclusion-exclusion
    total = 0.0
    for size in range(1, len(valid) + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(valid, size):
            # Intersection box: min of each objective across subset
            box_min = tuple(min(p[d] for p in subset) for d in range(4))
            vol = 1.0
            for d in range(4):
                vol *= max(0.0, box_min[d] - ref[d])
            total += sign * vol
    return total


def _hypervolume_monte_carlo(
    points: list[tuple[float, ...]], ref: tuple[float, ...], samples: int = 10000
) -> float:
    """Monte Carlo hypervolume approximation."""
    import random

    dims = len(ref)
    # Bounding box
    upper = tuple(max(p[d] for p in points) for d in range(dims))
    box_vol = 1.0
    for d in range(dims):
        box_vol *= upper[d] - ref[d]

    hits = 0
    rng = random.Random(42)
    for _ in range(samples):
        sample = tuple(rng.uniform(ref[d], upper[d]) for d in range(dims))
        # Check if any point dominates this sample
        for p in points:
            if all(p[d] >= sample[d] for d in range(dims)):
                hits += 1
                break
    return box_vol * hits / samples
