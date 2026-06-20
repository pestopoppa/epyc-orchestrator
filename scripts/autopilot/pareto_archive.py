"""Pareto archive: 4D non-dominated sorting with hypervolume indicator.

Objectives: quality (↑), speed (↑), -cost (↑ i.e. lower cost is better), reliability (↑).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_STATE_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "autopilot_state.json"
)

# Per-tier scoring lives in the shared TierSpec module (single source of truth, imported the
# same way by scripts/autopilot and src/api). REFERENCE_POINT / MIN_FRONTIER_EVAL_TIER kept as
# back-compat aliases for any importer.
from src.autopilot_core.pareto_math import (  # noqa: E402
    dominates as pareto_dominates,
    hypervolume as pareto_hypervolume,
    hypervolume_monte_carlo as pareto_hypervolume_monte_carlo,
    median_objectives,
)
from src.autopilot_core.tier_specs import (  # noqa: E402
    DEFAULT_FRONTIER_TIER,
    DEFAULT_REFERENCE_POINT,
    LEGACY_OBJECTIVE_POLICY,
    MIN_FRONTIER_EVAL_TIER,
    spec_for,
)
REFERENCE_POINT = DEFAULT_REFERENCE_POINT
PARETO_STATUS_TIER_EXCLUDED = "fast_reject"


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
    # Representative-admission (2026-06-04 policy correction): a stable identity for the
    # DEPLOYED config a trial measured, so trusted within-noise reproductions dedup into a
    # single robust-median frontier point instead of N noisy per-trial points. Empty for
    # ordinary per-trial entries.
    config_fingerprint: str = ""
    n_reproductions: int = 1

    def dominates(self, other: ParetoEntry) -> bool:
        """True if self dominates other (>= on all, > on at least one)."""
        return pareto_dominates(self.objectives, other.objectives)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["objectives"] = list(d["objectives"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParetoEntry:
        # Tolerant load: ignore unknown keys so a state file written by a NEWER schema
        # (e.g. the 2026-06-04 config_fingerprint fields) never bricks an older reader,
        # and a missing key just falls back to the dataclass default.
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        d["objectives"] = tuple(d["objectives"])
        return cls(**d)


class ParetoArchive:
    """4D Pareto frontier with hypervolume tracking and genealogy."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or DEFAULT_STATE_PATH
        self._read_only = False
        # Tier-segregated: each eval tier >= MIN_FRONTIER_EVAL_TIER keeps its OWN frontier +
        # hypervolume history. Quality is never compared across tiers. T0 stays audit-only
        # (in _all_entries, never on any frontier).
        self._frontiers: dict[int, list[ParetoEntry]] = {}
        self._all_entries: list[ParetoEntry] = []
        self._hv_history_by_tier: dict[int, list[tuple[int, float]]] = {}
        # Per-"{tier}:{fingerprint}" raw within-noise measurement clusters, used to
        # recompute robust-median representative objectives across reproductions.
        self._repro_clusters: dict[str, list[list[float]]] = {}
        self._load()

    @classmethod
    def from_archive_payload(
        cls,
        archive_payload: dict[str, Any] | None,
        *,
        state_path: Path | None = None,
        read_only: bool = True,
    ) -> ParetoArchive:
        """Build an archive object from an already reconstructed archive payload.

        This deliberately bypasses ``__init__`` so journal-reconstructed diagnostics can
        expose the normal read API without reading or mutating ``autopilot_state.json``.
        """
        archive = cls.__new__(cls)
        archive.state_path = state_path or DEFAULT_STATE_PATH
        archive._read_only = bool(read_only)
        archive._frontiers = {}
        archive._all_entries = []
        archive._hv_history_by_tier = {}
        archive._repro_clusters = {}
        archive._load_archive_payload(archive_payload or {})
        return archive

    @property
    def read_only(self) -> bool:
        return self._read_only

    def _require_mutable(self) -> None:
        if self._read_only:
            raise RuntimeError("read-only ParetoArchive snapshot cannot be mutated")

    # ── per-tier access helpers ──────────────────────────────────
    @staticmethod
    def _tier(tier: int | None) -> int:
        return DEFAULT_FRONTIER_TIER if tier is None else int(tier)

    def _front(self, tier: int | None = None) -> list[ParetoEntry]:
        """Live (mutable) frontier list for a tier — created empty on first access."""
        return self._frontiers.setdefault(self._tier(tier), [])

    def _hv_hist(self, tier: int | None = None) -> list[tuple[int, float]]:
        return self._hv_history_by_tier.setdefault(self._tier(tier), [])

    def tiers(self) -> list[int]:
        """Eval tiers that currently have a frontier, ascending."""
        return sorted(self._frontiers)

    # ── persistence ──────────────────────────────────────────────

    def _load_archive_payload(self, archive_data: dict[str, Any]) -> None:
        self._all_entries = [
            ParetoEntry.from_dict(e) for e in archive_data.get("all_entries", [])
        ]
        # Hypervolume history: prefer the new per-tier field; else migrate the legacy flat list
        # to the canonical tier. JSON keys round-trip as strings → normalize to int.
        hv_by_tier = archive_data.get("hv_history_by_tier")
        if hv_by_tier is not None:
            self._hv_history_by_tier = {
                int(t): [tuple(h) for h in hist] for t, hist in hv_by_tier.items()
            }
        else:
            legacy_hv = [tuple(h) for h in archive_data.get("hypervolume_history", [])]
            self._hv_history_by_tier = (
                {DEFAULT_FRONTIER_TIER: legacy_hv} if legacy_hv else {}
            )
        # Reproduction clusters: raw within-noise measurements per "{tier}:{fingerprint}",
        # used to recompute robust-median representative objectives. Absent in pre-2026-06-04
        # state → empty (backward compatible).
        self._repro_clusters = {
            str(k): [[float(x) for x in obj] for obj in v]
            for k, v in (archive_data.get("repro_clusters", {}) or {}).items()
        }
        # Frontiers are always REBUILT per-tier from all_entries (ignores any legacy `frontier`
        # field + scrubs T0 pollution + auto-migrates old single-frontier state).
        self._rebuild_frontier()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        archive_data = data.get("pareto_archive", {})
        self._load_archive_payload(archive_data)

        # Integrity check: detect lost frontier
        trial_counter = data.get("trial_counter", 0)
        if trial_counter > 10 and not any(self._frontiers.values()) and not self._all_entries:
            import os as _os
            if data.get("_allow_empty_frontier_rebase") or _os.environ.get("AUTOPILOT_ALLOW_EMPTY_FRONTIER") == "1":
                # Deliberate rebase (e.g. the 2026-06-01 speed double-count correction):
                # the operator intentionally cleared the archive so it rebuilds on honest
                # measurements. Allow the empty start instead of refusing. Read from the
                # state file (survives re-launch/env-stripping) or an env override.
                log.warning(
                    "Pareto frontier empty at trial %d, but a deliberate-rebase flag is set "
                    "(_allow_empty_frontier_rebase / AUTOPILOT_ALLOW_EMPTY_FRONTIER) — starting "
                    "with an empty frontier; it will rebuild from new trials.",
                    trial_counter,
                )
                return
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

    def load(self, state: dict[str, Any]) -> None:
        """Load archive payload from an in-memory autopilot state dict."""
        self._require_mutable()
        self._load_archive_payload((state or {}).get("pareto_archive", {}) or {})

    def _replace_from_archive_payload(self, archive_data: dict[str, Any]) -> None:
        """Replace mutable runtime view from a journal-authoritative fold."""
        self._require_mutable()
        self._load_archive_payload(archive_data or {})

    # ── core operations ─────────────────────────────────────────

    @staticmethod
    def is_frontier_eligible(entry: ParetoEntry) -> bool:
        """True when an entry is allowed onto the production Pareto frontier.

        Tier-0 is a fast-reject sentinel eval. Its 10-question granularity makes
        q=2.4 mean "8/10 on the easy set", not a production-quality ceiling.
        Keep T0 entries in all_entries for audit, but never let them set the
        frontier, production_best, hypervolume, or archive-max baseline guard.
        """
        return entry.eval_tier >= MIN_FRONTIER_EVAL_TIER

    def _rebuild_frontier(self) -> None:
        """Recompute PER-TIER frontiers from all_entries (groups by eval_tier; dominance only
        within a tier; scrubs T0 + legacy single-frontier pollution)."""
        by_tier: dict[int, list[ParetoEntry]] = {}
        for entry in self._all_entries:
            if not self.is_frontier_eligible(entry):
                entry.is_production_best = False
                continue
            t = int(entry.eval_tier)
            rebuilt = by_tier.setdefault(t, [])
            if any(existing.dominates(entry) for existing in rebuilt):
                entry.is_production_best = False
                continue
            by_tier[t] = [existing for existing in rebuilt if not entry.dominates(existing)]
            by_tier[t].append(entry)
        self._frontiers = by_tier

    def is_pareto_candidate(self, objectives: tuple[float, ...], tier: int | None = None) -> bool:
        """Check if objectives would be non-dominated by the SAME-TIER frontier."""
        entry = ParetoEntry(
            trial_id=-1,
            objectives=objectives,
            eval_tier=self._tier(tier),
        )
        for f in self._front(tier):
            if f.dominates(entry):
                return False
        return True

    def update(self, entry: ParetoEntry) -> str:
        """Add entry to archive, routed to ITS tier's frontier. Returns 'frontier'/'dominated'/
        fast_reject. Dominance + hypervolume are strictly within `entry.eval_tier`."""
        self._require_mutable()
        self._all_entries.append(entry)
        if not self.is_frontier_eligible(entry):
            entry.is_production_best = False
            return PARETO_STATUS_TIER_EXCLUDED

        tier = int(entry.eval_tier)
        front = self._front(tier)
        if not self.is_pareto_candidate(entry.objectives, tier):
            status = "dominated"
        else:
            self._frontiers[tier] = [f for f in front if not entry.dominates(f)]
            self._frontiers[tier].append(entry)
            status = "frontier"

        # Update hypervolume (same tier)
        hv = self.hypervolume(tier=tier)
        self._hv_hist(tier).append((entry.trial_id, hv))
        return status

    def reproduction_count(self, tier: int, fingerprint: str) -> int:
        """Number of measurements folded into the (tier, fingerprint) representative."""
        return len(self._repro_clusters.get(f"{int(tier)}:{fingerprint}", []))

    def upsert_representative(
        self,
        fingerprint: str,
        tier: int,
        objectives: tuple[float, ...],
        *,
        trial_id: int,
        **entry_kwargs: Any,
    ) -> tuple[str, tuple[float, ...]]:
        """Fold a TRUSTED within-noise measurement into the per-(tier, fingerprint)
        reproduction cluster, recompute robust-MEDIAN objectives across the cluster, and
        admit/refresh a SINGLE representative entry for that config.

        Policy correction (2026-06-04): a within-quality-noise trial is excluded from
        strategy learning (AP-22), but it can still be NON-DOMINATED on speed/cost/
        reliability and therefore belongs on the multi-objective frontier. Dominance is
        tested on the cluster MEDIAN — never a lucky single-trial speed sample — so
        host-throughput variance cannot manufacture separate frontier geometry.
        Reproductions of the same config (even via different action types) dedup by
        ``fingerprint``, never by trial id.

        Returns ``(status, median_objectives)``. An empty fingerprint has no stable
        identity to dedup on, so it falls back to a plain per-trial :meth:`update`.
        """
        self._require_mutable()
        tier = int(tier)
        if not fingerprint:
            status = self.update(
                ParetoEntry(
                    trial_id=trial_id,
                    objectives=tuple(objectives),
                    eval_tier=tier,
                    **entry_kwargs,
                )
            )
            return status, tuple(objectives)

        key = f"{tier}:{fingerprint}"
        cluster = self._repro_clusters.setdefault(key, [])
        cluster.append([float(x) for x in objectives])
        median_objs = median_objectives(cluster)

        # Exactly one representative per (tier, fingerprint): drop the prior one before
        # re-adding with the refreshed median so the cluster can't accrete duplicates.
        self._all_entries = [
            e
            for e in self._all_entries
            if not (e.config_fingerprint == fingerprint and int(e.eval_tier) == tier)
        ]
        self._all_entries.append(
            ParetoEntry(
                trial_id=trial_id,
                objectives=median_objs,
                eval_tier=tier,
                config_fingerprint=fingerprint,
                n_reproductions=len(cluster),
                **entry_kwargs,
            )
        )
        self._rebuild_frontier()
        on_frontier = any(
            e.config_fingerprint == fingerprint and int(e.eval_tier) == tier
            for e in self._front(tier)
        )
        self._hv_hist(tier).append((trial_id, self.hypervolume(tier=tier)))
        return ("frontier" if on_frontier else "dominated"), median_objs

    def frontier(self, tier: int | None = None) -> list[ParetoEntry]:
        return list(self._front(tier))

    def frontier_size(self, tier: int | None = None) -> int:
        return len(self._front(tier))

    def production_best(self) -> ParetoEntry | None:
        """The single GLOBAL deployed-production config (across all tiers; only one is marked)."""
        for front in self._frontiers.values():
            for e in front:
                if e.is_production_best:
                    return e
        return None

    def mark_production_best(self, trial_id: int) -> bool:
        """Mark the single global production-best. R8 guard: refuse unless `trial_id` is on the
        CANONICAL (T1) frontier — a T2 validation entry must never become the deployed config.
        Returns True if marked, False if refused."""
        self._require_mutable()
        canonical = self._front(DEFAULT_FRONTIER_TIER)
        if not any(e.trial_id == trial_id for e in canonical):
            return False
        for front in self._frontiers.values():
            for e in front:
                e.is_production_best = e.trial_id == trial_id
        return True

    # ── hypervolume ──────────────────────────────────────────────

    def hypervolume(self, ref: tuple[float, ...] | None = None, tier: int | None = None) -> float:
        """Hypervolume indicator for one tier's frontier (its TierSpec reference point).

        Uses inclusion-exclusion for 4D (exact, fast enough for <1000 entries).
        """
        front = self._front(tier)
        if not front:
            return 0.0
        ref = ref or spec_for(self._tier(tier)).reference_point
        return _hypervolume_4d([e.objectives for e in front], ref)

    def hypervolume_trend(
        self, window: int | None = None, tier: int | None = None
    ) -> list[tuple[int, float]]:
        """Return (trial_id, hypervolume) history for a tier."""
        hist = self._hv_hist(tier)
        if window:
            return list(hist[-window:])
        return list(hist)

    def hypervolume_slope(self, window: int = 50, tier: int | None = None) -> float:
        """Linear regression slope of a tier's hypervolume over the last `window` entries."""
        hist = self._hv_hist(tier)[-window:]
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
        tier: int | None = None,
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
        hist = self._hv_hist(tier)
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

    # ── Bradley-Terry tiebreak — cheap axis-vote proxy (P17.BT-2) ──

    def bt_tiebreak_topk(self, k: int = 5, tier: int | None = None) -> dict[str, Any]:
        """Axis-vote Bradley-Terry tiebreak over top-K frontier entries (within ONE tier).

        IMPORTANT — what this is, and what this is NOT:

          This is a **cheap axis-vote proxy** that uses the BT engine on
          data we already have (the recorded 4D objectives). The pairwise
          inputs are mechanical Borda comparisons across the four
          objective axes — they are NOT independent model judgments of
          candidate outputs.

          The Fortytwo-style peer-ranked-consensus form described in
          intake-615 (arxiv:2510.24801) would use N judge models scoring
          each candidate pairwise. That is INFERENCE-GATED and tracked
          separately as P17.BT-4 in the autopilot handoff. This method
          is P17.BT-2: a strictly cheaper and weaker signal that runs
          purely off recorded objectives.

        Available as an offline diagnostic for hypervolume stagnation. The
        live planner prompt no longer injects this signal because J13 found
        the prompt hint cosmetic/non-certifying.

        Why this is still useful:
          Hypervolume scalarization collapses four axes into one number
          and can hide candidates that *consistently* beat peers across
          axes without being individually hypervolume-dominant.
          Axis-vote BT surfaces those candidates as alternative
          exploration seeds — a falsifiable signal at zero inference
          cost.

        Why pairwise scores come from axis comparison, not new inference:
          By design this is a code-only tiebreak (P17.BT-2) — it uses
          the objectives already recorded in archive entries. No
          eval-tower re-runs, no model calls. The accompanying
          falsification gate (P17.BT-3) is the only inference-dependent
          step.

        Top-K selection is range-normalized (fix landed 2026-05-27):
          The top-K candidates fed to BT are selected by a per-axis
          range-normalized sum: each axis (obj - ref) is divided by
          (max_e(obj) - ref) across the frontier before summing, so
          every axis contributes on a [0, 1] scale regardless of
          its physical magnitude. This prevents high-magnitude axes
          (e.g., speed in t/s, range 0-100+) from swamping low-magnitude
          axes (e.g., reliability in [0,1]) when deciding which
          candidates even enter the BT comparison.

        Parameters
        ----------
        k:
            Number of top frontier entries to compare. Defaults to 5;
            capped at the frontier size.

        Returns
        -------
        Dict with keys:
          - `ranking` — list of trial IDs ordered by BT log-skill (high to low)
          - `log_skills` — dict trial_id -> log-skill (lowest anchored at 0)
          - `top_k_trial_ids` — input set, in archive order
          - `warnings` — diagnostics from the BT fit (cycles, dominance skew, etc.)
          - `converged` — whether the BT iteration converged
          - `iterations` — Zermelo iteration count
          - `note` — short status string suitable for diagnostics/logging

        Returns an empty dict (with `note` set) when the frontier has <2
        entries — BT is undefined on a singleton.
        """
        # Local import keeps the BT module a leaf dependency.
        # bradley_terry lives in src/ (moved 2026-05-27 as part of DAR-6
        # scaffolding); ORCH_ROOT is on sys.path at autopilot runtime.
        from src.bradley_terry import bradley_terry_rank

        front = self._front(tier)
        ref = spec_for(self._tier(tier)).reference_point

        if len(front) < 2:
            return {
                "ranking": [e.trial_id for e in front],
                "log_skills": {e.trial_id: 0.0 for e in front},
                "top_k_trial_ids": [e.trial_id for e in front],
                "warnings": [],
                "converged": True,
                "iterations": 0,
                "note": f"BT tiebreak skipped (frontier_size={len(front)})",
            }

        # Pick top-K frontier entries by hypervolume contribution. Use the
        # current frontier as-is (no re-ranking); if K exceeds frontier
        # size, just compare the whole frontier.
        k = min(max(k, 2), len(front))
        # Rank frontier entries by a range-normalized sum of axis values minus
        # the reference. Earlier version used a raw sum, which made
        # high-magnitude axes (speed in t/s, range 0-100+) dominate vs
        # low-magnitude axes (reliability in [0,1]). Range-normalization
        # per axis across the frontier puts each axis on a [0, 1] scale
        # before summing so no axis can swamp the others purely on units.
        # Ties broken by quality.
        n_axes = len(front[0].objectives)
        # Per-axis frontier max; degenerate axes (max == ref) fall back to
        # 1.0 so we don't divide by zero and the contribution becomes
        # (obj - ref) / 1.0 = the raw delta (small for those axes).
        per_axis_max = [
            max(e.objectives[a] for e in front)
            for a in range(n_axes)
        ]
        per_axis_range = [
            (per_axis_max[a] - ref[a]) or 1.0
            for a in range(n_axes)
        ]

        def _normalized_axis_sum(e: ParetoEntry) -> float:
            return sum(
                (e.objectives[a] - ref[a]) / per_axis_range[a]
                for a in range(n_axes)
            )

        scored = sorted(
            front,
            key=lambda e: (_normalized_axis_sum(e), e.objectives[0]),
            reverse=True,
        )
        top_k = scored[:k]
        trial_ids = [e.trial_id for e in top_k]

        # Build pairwise win-scores from axis-wise Borda counting:
        #   pair_score[i, j] = fraction of objective axes where i > j
        #                    (ties count as 0.5)
        # This is symmetric: pair_score[i,j] + pair_score[j,i] = 1.0.
        # No axis normalization is needed because we only compare relative
        # ordering per axis, not magnitudes.
        n_axes = len(top_k[0].objectives)
        pairwise: dict[tuple[int, int], float] = {}
        for i, ei in enumerate(top_k):
            for j, ej in enumerate(top_k):
                if i == j:
                    continue
                wins = 0.0
                for a in range(n_axes):
                    if ei.objectives[a] > ej.objectives[a]:
                        wins += 1.0
                    elif ei.objectives[a] == ej.objectives[a]:
                        wins += 0.5
                pairwise[(ei.trial_id, ej.trial_id)] = wins / n_axes

        result = bradley_terry_rank(trial_ids, pairwise)

        # Build a short logging note. The lead candidate may differ from the
        # naive top-of-hypervolume entry — that disagreement is the signal
        # the controller cares about.
        naive_top = trial_ids[0]
        bt_top = result.ranking[0]
        if naive_top == bt_top:
            disagreement = f"BT agrees with hypervolume on trial #{naive_top}"
        else:
            disagreement = (
                f"BT picks trial #{bt_top} (rank-by-hv would pick #{naive_top}) "
                f"— pairwise consensus across {n_axes} axes disagrees with scalarization"
            )

        return {
            "ranking": result.ranking,
            "log_skills": result.log_skills,
            "top_k_trial_ids": trial_ids,
            "warnings": result.warnings,
            "converged": result.converged,
            "iterations": result.iterations,
            "note": disagreement,
        }

    # ── summary ──────────────────────────────────────────────────

    def summary(self, tier: int | None = None) -> dict[str, Any]:
        front = self._front(tier)
        if not front:
            return {"frontier_size": 0, "hypervolume": 0.0, "tier": self._tier(tier)}
        best_quality = max(e.objectives[0] for e in front)
        best_speed = max(e.objectives[1] for e in front)
        best_cost = max(e.objectives[2] for e in front)  # -cost, higher is better
        return {
            "tier": self._tier(tier),
            "frontier_size": len(front),
            "total_entries": len(self._all_entries),
            "hypervolume": self.hypervolume(tier=tier),
            "best_quality": best_quality,
            "best_speed": best_speed,
            "best_neg_cost": best_cost,
            "hv_slope_50": self.hypervolume_slope(50, tier=tier),
        }

    def tier_overview(self) -> str:
        """One compact line per tier (frontier size + best quality) for the planner — so it sees
        T2/harder-tier validation status without any cross-tier quality comparison."""
        if not self._frontiers:
            return "(no tier frontiers yet)"
        parts = []
        for t in self.tiers():
            front = self._frontiers[t]
            if not front:
                continue
            bq = max(e.objectives[0] for e in front)
            parts.append(f"T{t}: {len(front)} pts, best_q={bq:.3f}")
        return "Per-tier frontiers — " + "; ".join(parts) if parts else "(no tier frontiers yet)"

    def summary_text(self, tier: int | None = None) -> str:
        """Render ONE tier's frontier (default = canonical T1) + a per-tier overview line."""
        front = self._front(tier)
        t = self._tier(tier)
        s = self.summary(tier=tier)
        lines = [
            self.tier_overview(),
            f"\n[T{t}] Pareto frontier: {s['frontier_size']} entries "
            f"(of {s.get('total_entries', 0)} total)",
            f"Hypervolume: {s['hypervolume']:.4f}",
            f"HV slope (last 50): {s.get('hv_slope_50', 0):.6f}",
            f"Best quality: {s.get('best_quality', 0):.3f}",
            f"Best speed: {s.get('best_speed', 0):.1f} t/s",
        ]
        if front:
            lines.append("\nFrontier entries:")
            for e in sorted(front, key=lambda x: -x.objectives[0]):
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

    def geometry(self, tier: int | None = None) -> dict[str, Any]:
        """Compute structural info about ONE tier's frontier (default = canonical T1).

        Returns dict with keys: shape, blocking_quality, blocking_speed,
        gaps, hv_slope_10, suggested_attack.
        """
        front = self._front(tier)
        out: dict[str, Any] = {
            "shape": "empty",
            "blocking_quality": None,
            "blocking_speed": None,
            "gaps": [],
            "hv_slope_10": self.hypervolume_slope(10, tier=tier),
            "suggested_attack": "no data — seed more trials first",
            "frontier_count": len(front),
        }
        if not front:
            return out
        if len(front) == 1:
            out["shape"] = "single"
            e = front[0]
            out["suggested_attack"] = (
                f"single frontier point trial #{e.trial_id} q={e.objectives[0]:.2f} "
                f"sp={e.objectives[1]:.1f} — propose explore actions to add "
                "diversity"
            )
            return out

        # Project to (quality, speed). Sort by quality ascending.
        pts = sorted(
            front, key=lambda x: (x.objectives[0], -x.objectives[1])
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
        blocking_q = max(front, key=lambda e: e.objectives[0])
        blocking_s = max(front, key=lambda e: e.objectives[1])
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

    def geometry_text(self, tier: int | None = None) -> str:
        """Render geometry() for inclusion in the controller prompt (one tier)."""
        g = self.geometry(tier=tier)
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


def pareto_archive_from_journal_rows(
    rows: list[dict[str, Any]],
    session_start_ts: float | None = None,
    *,
    current_run_only: bool = False,
    max_trial_id: int | None = None,
    deinflate_before_ts: float | None = None,
    deinflate_factor: float = 1.0,
    objective_policy: str = LEGACY_OBJECTIVE_POLICY,
    state_path: Path | None = None,
) -> ParetoArchive | None:
    """Return a read-only ParetoArchive reconstructed from append-only journal rows."""
    from src.autopilot_core.journal_reconstruction import reconstruct_archive_from_journal_rows

    archive_payload = reconstruct_archive_from_journal_rows(
        rows,
        session_start_ts,
        current_run_only=current_run_only,
        max_trial_id=max_trial_id,
        deinflate_before_ts=deinflate_before_ts,
        deinflate_factor=deinflate_factor,
        objective_policy=objective_policy,
    )
    if archive_payload is None:
        return None
    return ParetoArchive.from_archive_payload(
        archive_payload,
        state_path=state_path,
        read_only=True,
    )


# ── hypervolume computation ──────────────────────────────────────


def _hypervolume_4d(
    points: list[tuple[float, ...]], ref: tuple[float, ...]
) -> float:
    """Exact 4D hypervolume via inclusion-exclusion.

    For small frontiers (<100 entries), this is fast enough.
    Falls back to 2D Monte Carlo approximation for very large frontiers.
    """
    return pareto_hypervolume(points, ref, exact_limit=100, samples=10000)


def _hypervolume_monte_carlo(
    points: list[tuple[float, ...]], ref: tuple[float, ...], samples: int = 10000
) -> float:
    """Monte Carlo hypervolume approximation."""
    return pareto_hypervolume_monte_carlo(points, ref, samples=samples)
