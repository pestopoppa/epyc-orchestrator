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
        """Atomically save archive to state file, merging with existing state.

        2026-05-23 Phase 6a — atomic write via temp + os.replace. The
        autopilot_state.json holds pareto_archive as a sub-key alongside
        trial_counter et al.; a partial write would brick startup. Symmetric
        with state_store.save_state's atomic semantics.
        """
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
        import os as _os
        tmp = self.state_path.with_suffix(self.state_path.suffix + f".tmp.{_os.getpid()}")
        payload = json.dumps(existing, indent=2, default=str)
        with open(tmp, "w") as fh:
            fh.write(payload)
            fh.flush()
            _os.fsync(fh.fileno())
        _os.replace(tmp, self.state_path)

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

    def hv_slope_noise_floor(
        self,
        *,
        window_slopes: int = 50,
        slope_window: int = 10,
        floor_default: float = 1e-3,
        floor_min: float = 1e-5,
        k: float = 0.5,
    ) -> float:
        """Estimate the noise floor of hv_slope_<slope_window> from recent history.

        Used by the autopilot stagnation gate so STAGNATION_HV_EPS doesn't have
        to be hand-tuned. The rule: compute rolling hv_slope over the last
        `window_slopes` positions, take their standard deviation, scale by k.
        Below ~20 hypervolume entries (or near-zero variance), fall back to
        `floor_default` (the prior hardcoded constant) so behaviour is
        backward-compatible.

        Returns a value in [floor_min, floor_default] — never above the prior
        default, so auto-calibration can only TIGHTEN the gate (require flatter
        slopes than 1e-3 before declaring stagnation). False negatives missing
        real stagnation are recoverable on the next trial; false positives
        running the expensive rich prompt during a healthy exploit phase are
        not. The clip enforces the safer direction.
        """
        hist = self._hypervolume_history
        if len(hist) < max(slope_window + 5, 20):
            return floor_default
        # Compute rolling slope ending at each position from slope_window onward.
        slopes: list[float] = []
        for end in range(slope_window, len(hist)):
            chunk = hist[end - slope_window:end]
            n = len(chunk)
            if n < 2:
                continue
            xs = list(range(n))
            ys = [h[1] for h in chunk]
            mx = sum(xs) / n
            my = sum(ys) / n
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs)
            slopes.append(num / den if den > 0 else 0.0)
        # Only the most recent `window_slopes` matter — older noise regimes are
        # not representative if the system underwent a regime change.
        slopes = slopes[-window_slopes:]
        if len(slopes) < 5:
            return floor_default
        mean_s = sum(slopes) / len(slopes)
        var_s = sum((s - mean_s) ** 2 for s in slopes) / len(slopes)
        std_s = var_s ** 0.5
        floor = k * std_s
        # Clip to [floor_min, floor_default]; never let the gate go below the
        # known-good prior default — auto-calibration only tightens it.
        return max(floor_min, min(floor, floor_default))

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

    # ── frontier geometry (2026-05-23) ────────────────────────────────
    # Goes beyond summary_text() by surfacing structural info that helps the
    # planner choose WHICH axis to attack rather than just emitting blind
    # action proposals. Three pieces:
    #
    #   shape          — categorical: "empty" / "single" / "linear" /
    #                    "L_quality" / "L_speed" / "scattered". Tells the
    #                    planner whether the frontier is a smooth trade-off
    #                    curve or has obvious gaps.
    #   blocking       — per-axis: which frontier point dominates that axis
    #                    AND what it gives up on the other axes to do so.
    #                    Tells the planner where the current Pareto front is
    #                    capped + the cost of pushing further.
    #   gaps           — list of (q_range, s_range) regions in the
    #                    quality×speed projection where no frontier point
    #                    lives — concrete coordinates for "adjacent
    #                    possible" exploration.
    #
    # Output is a structured dict (for programmatic use) + a text rendering
    # helper for the controller prompt.

    def geometry(self) -> dict[str, Any]:
        """Compute structural info about the current frontier.

        Returns dict with keys: shape, blocking_quality, blocking_speed,
        gaps, hv_slope_10, suggested_attack.
        """
        out: dict[str, Any] = {
            "shape": "empty",
            "blocking_quality": None,
            "blocking_speed": None,
            "gaps": [],
            "hv_slope_10": self.hypervolume_slope(10),
            "suggested_attack": "no data — seed more trials first",
            "frontier_count": len(self._frontier),
        }
        if not self._frontier:
            return out
        if len(self._frontier) == 1:
            out["shape"] = "single"
            e = self._frontier[0]
            out["suggested_attack"] = (
                f"single frontier point trial #{e.trial_id} q={e.objectives[0]:.2f} "
                f"sp={e.objectives[1]:.1f} — propose explore actions to add "
                "diversity"
            )
            return out

        # Project to (quality, speed). Sort by quality ascending.
        pts = sorted(
            self._frontier, key=lambda x: (x.objectives[0], -x.objectives[1])
        )
        q_vals = [e.objectives[0] for e in pts]
        s_vals = [e.objectives[1] for e in pts]
        q_min, q_max = min(q_vals), max(q_vals)
        s_min, s_max = min(s_vals), max(s_vals)
        q_range = q_max - q_min
        s_range = s_max - s_min

        # Shape heuristics:
        #   linear      — points spread evenly along both axes
        #   L_quality   — bulk of points cluster at high quality (one outlier at high speed)
        #   L_speed     — bulk cluster at high speed (one outlier at high quality)
        #   scattered   — no clear pattern
        n = len(pts)
        if q_range < 1e-6 or s_range < 1e-6:
            out["shape"] = "linear"  # degenerate but useful: variation on one axis only
        else:
            # Coefficient of variation per axis as a roughness proxy.
            mean_q = sum(q_vals) / n
            mean_s = sum(s_vals) / n
            spread_q = (max(abs(v - mean_q) for v in q_vals)) / max(abs(mean_q), 1e-9)
            spread_s = (max(abs(v - mean_s) for v in s_vals)) / max(abs(mean_s), 1e-9)
            if spread_q > 2 * spread_s and n >= 3:
                out["shape"] = "L_speed"
            elif spread_s > 2 * spread_q and n >= 3:
                out["shape"] = "L_quality"
            else:
                out["shape"] = "linear" if n >= 3 else "scattered"

        # Blocking points per axis.
        blocking_q = max(self._frontier, key=lambda e: e.objectives[0])
        blocking_s = max(self._frontier, key=lambda e: e.objectives[1])
        out["blocking_quality"] = {
            "trial_id": blocking_q.trial_id,
            "objectives": list(blocking_q.objectives),
            "species": blocking_q.species,
            "gives_up_speed": s_max - blocking_q.objectives[1] if s_max > 0 else 0.0,
        }
        out["blocking_speed"] = {
            "trial_id": blocking_s.trial_id,
            "objectives": list(blocking_s.objectives),
            "species": blocking_s.species,
            "gives_up_quality": q_max - blocking_s.objectives[0] if q_max > 0 else 0.0,
        }

        # Gap detection: walk q-sorted points, look for consecutive pairs
        # where the (q, s) jump is large relative to per-axis range.
        gaps: list[dict[str, Any]] = []
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            dq = b.objectives[0] - a.objectives[0]
            ds = a.objectives[1] - b.objectives[1]  # frontier expected: speed drops as quality rises
            if dq > 0.15 * q_range or ds > 0.25 * s_range:
                gaps.append({
                    "between_trials": [a.trial_id, b.trial_id],
                    "q_window": [a.objectives[0], b.objectives[0]],
                    "s_window": [b.objectives[1], a.objectives[1]],
                    "size_q_frac": dq / q_range if q_range else 0.0,
                    "size_s_frac": ds / s_range if s_range else 0.0,
                })
        out["gaps"] = gaps[:5]  # cap at 5 most significant

        # Suggested attack.
        slope = out["hv_slope_10"]
        if abs(slope) < 1e-5 and len(pts) < 5:
            out["suggested_attack"] = (
                "frontier is sparse and hv-stagnant — try species rotation "
                "or seed_batch to break out of local minimum"
            )
        elif gaps:
            g = gaps[0]
            out["suggested_attack"] = (
                f"largest gap is between trials {g['between_trials'][0]} and "
                f"{g['between_trials'][1]} (q∈[{g['q_window'][0]:.2f},"
                f"{g['q_window'][1]:.2f}], sp∈[{g['s_window'][0]:.1f},"
                f"{g['s_window'][1]:.1f}]) — actions targeting this midpoint "
                "would expand the frontier most"
            )
        elif out["shape"] == "L_quality":
            out["suggested_attack"] = (
                "frontier is L-shaped along quality — speed is undersampled; "
                "propose numeric/structural actions tuned for throughput"
            )
        elif out["shape"] == "L_speed":
            out["suggested_attack"] = (
                "frontier is L-shaped along speed — quality is undersampled; "
                "propose prompt_mutation / gepa_optimize for accuracy gains"
            )
        elif slope > 1e-3:
            out["suggested_attack"] = (
                f"frontier is growing (hv_slope_10={slope:.4f}) — continue the "
                "current species mix"
            )
        else:
            out["suggested_attack"] = (
                "frontier mature with no obvious gaps — switch to exploit "
                "(rollback to best + small numeric perturbations) or invoke "
                "distill_knowledge to consolidate"
            )
        return out

    def geometry_text(self) -> str:
        """Render geometry() for inclusion in the controller prompt."""
        g = self.geometry()
        if g["frontier_count"] == 0:
            return "(frontier empty — no geometry to analyse)"
        lines = [
            f"Shape:   {g['shape']}  ({g['frontier_count']} frontier points)",
            f"HV slope last-10:  {g['hv_slope_10']:+.5f}",
        ]
        if g["blocking_quality"]:
            bq = g["blocking_quality"]
            lines.append(
                f"Blocking quality: trial #{bq['trial_id']} ({bq['species']}) "
                f"q={bq['objectives'][0]:.3f} sp={bq['objectives'][1]:.1f} — "
                f"to advance q here, would give up {bq['gives_up_speed']:.1f} t/s"
            )
        if g["blocking_speed"]:
            bs = g["blocking_speed"]
            lines.append(
                f"Blocking speed:   trial #{bs['trial_id']} ({bs['species']}) "
                f"q={bs['objectives'][0]:.3f} sp={bs['objectives'][1]:.1f} — "
                f"to advance sp here, would give up {bs['gives_up_quality']:.3f} q"
            )
        if g["gaps"]:
            lines.append("Gaps in frontier (q-sorted, largest first):")
            for gap in g["gaps"]:
                lines.append(
                    f"  - between #{gap['between_trials'][0]} and "
                    f"#{gap['between_trials'][1]}: "
                    f"q∈[{gap['q_window'][0]:.3f}, {gap['q_window'][1]:.3f}] "
                    f"sp∈[{gap['s_window'][0]:.1f}, {gap['s_window'][1]:.1f}] "
                    f"(size: {max(gap['size_q_frac'], gap['size_s_frac']):.0%} of axis range)"
                )
        lines.append(f"\nSuggested attack: {g['suggested_attack']}")
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
