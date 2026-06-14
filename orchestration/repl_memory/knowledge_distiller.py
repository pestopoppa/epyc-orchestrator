"""Knowledge distillation pipeline for the AutoPilot strategy store (AP-29).

Implements the L1/L2/L3 hierarchy from intake-413 (HCC) with the MDL-style
compression check from intake-414 (Token Savior):

    L1 (raw)        — individual strategy from a single trial
    L2 (pattern)    — cluster of >= MDL_MIN_CLUSTER_SIZE similar L1 entries
                       within a single species
    L3 (convention) — pattern that recurs across >= MIN_SPECIES_FOR_CONVENTION
                       species (or accumulates >= MIN_SOURCES_FOR_CONVENTION
                       underlying source trials)

All three tiers live in the same ``strategies`` table, differentiated by
``entry_type`` (``raw`` / ``pattern`` / ``convention``). When a cluster of
raw entries promotes to a pattern, the source rows are quarantined (NIB2-41
validity counters bumped to drop them below the quarantine threshold) so
subsequent retrieve() calls surface the pattern rather than the source rows.

Triggered every N=25 trials by the autopilot main loop and on rebalance, but
the scheduling itself lives in autopilot.py — this module is purely the
consolidation engine.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# AP-29 tunables — kept module-level so tests can monkeypatch.
MDL_MIN_CLUSTER_SIZE = 3                  # min raw entries to merge into pattern
MIN_SPECIES_FOR_CONVENTION = 3            # cross-species threshold
MIN_SOURCES_FOR_CONVENTION = 10           # cumulative-source threshold
DEFAULT_CLUSTER_SIM_THRESHOLD = 0.75      # cosine threshold for greedy clustering
DEFAULT_PATTERN_SIM_THRESHOLD = 0.70      # looser for cross-species convention pass


def _dedupe_ints(values: list[Any]) -> list[int]:
    """Stable integer de-duplication for source/evidence IDs."""
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            continue
        if int_value in seen:
            continue
        seen.add(int_value)
        out.append(int_value)
    return out


@dataclass
class DistillationStats:
    """Per-cycle counters for the distillation pass."""

    patterns_created: int = 0
    conventions_created: int = 0
    raw_entries_consolidated: int = 0
    patterns_consolidated: int = 0
    skipped_below_mdl: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patterns_created": self.patterns_created,
            "conventions_created": self.conventions_created,
            "raw_entries_consolidated": self.raw_entries_consolidated,
            "patterns_consolidated": self.patterns_consolidated,
            "skipped_below_mdl": self.skipped_below_mdl,
            "notes": list(self.notes),
        }


class KnowledgeDistiller:
    """Periodic L1->L2->L3 consolidation of strategy memory entries.

    Construct with a ready ``StrategyStore`` instance; call ``distill(trial_id)``
    once per cycle. The distiller never owns the store — close() is the
    caller's responsibility.
    """

    def __init__(
        self,
        strategy_store: Any,
        cluster_sim_threshold: float = DEFAULT_CLUSTER_SIM_THRESHOLD,
        convention_sim_threshold: float = DEFAULT_PATTERN_SIM_THRESHOLD,
        min_validity: float = 0.10,
    ):
        self.store = strategy_store
        self.cluster_sim = cluster_sim_threshold
        self.convention_sim = convention_sim_threshold
        self.min_validity = min_validity

    # ── Public API ───────────────────────────────────────────────

    def distill(self, trial_id: int) -> DistillationStats:
        """Run one full L1->L2->L3 consolidation cycle.

        Safe to call repeatedly — entries below MDL or below the species/
        source thresholds are simply left alone.
        """
        stats = DistillationStats()

        raw_entries = self._fetch_entries_by_type("raw")
        if len(raw_entries) >= MDL_MIN_CLUSTER_SIZE:
            grouped = self._group_by(raw_entries, key="species")
            for species, entries in grouped.items():
                created, consolidated, skipped = self._extract_patterns(
                    entries, species, trial_id
                )
                stats.patterns_created += len(created)
                stats.raw_entries_consolidated += consolidated
                stats.skipped_below_mdl += skipped

        patterns = self._fetch_entries_by_type("pattern")
        if len(patterns) >= MDL_MIN_CLUSTER_SIZE:
            created, consolidated = self._extract_conventions(patterns, trial_id)
            stats.conventions_created += len(created)
            stats.patterns_consolidated += consolidated

        logger.info(
            "Distillation trial=%d: %d patterns, %d conventions "
            "(consolidated %d raw / %d pattern entries)",
            trial_id,
            stats.patterns_created,
            stats.conventions_created,
            stats.raw_entries_consolidated,
            stats.patterns_consolidated,
        )
        return stats

    # ── Internals ────────────────────────────────────────────────

    def _fetch_entries_by_type(self, entry_type: str) -> list[dict[str, Any]]:
        rows = self.store._conn.execute(
            "SELECT * FROM strategies WHERE entry_type = ?", (entry_type,)
        ).fetchall()
        # Filter by current validity score (skip already-quarantined / decayed)
        out: list[dict[str, Any]] = []
        for row in rows:
            sid = row["id"]
            validity = self.store._validity_score(sid)
            if validity < self.min_validity:
                continue
            out.append(
                {
                    "id": sid,
                    "description": row["description"],
                    "insight": row["insight"],
                    "source_trial_id": row["source_trial_id"],
                    "evidence_trial_ids": self.store._evidence_trial_ids_for_row(row),
                    "species": row["species"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "validity": validity,
                }
            )
        return out

    @staticmethod
    def _group_by(
        entries: list[dict[str, Any]], key: str
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in entries:
            grouped[e[key]].append(e)
        return dict(grouped)

    def _embed_all(self, entries: list[dict[str, Any]]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for e in entries:
            text = f"{e['description']} {e['insight']}"
            vectors.append(self.store._embed(text))
        return np.stack(vectors)

    @staticmethod
    def _greedy_cluster(
        emb_matrix: np.ndarray, sim_threshold: float
    ) -> list[list[int]]:
        """Greedy single-link clustering by cosine similarity.

        Returns a list of clusters, each a list of indices into the
        embedding matrix. Single-element clusters are included so the
        caller can decide whether to keep or drop them.
        """
        n = emb_matrix.shape[0]
        if n == 0:
            return []
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9
        normed = emb_matrix / norms
        sim = normed @ normed.T

        assigned = [False] * n
        clusters: list[list[int]] = []
        for i in range(n):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                if sim[i, j] >= sim_threshold:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        return clusters

    def _extract_patterns(
        self, entries: list[dict[str, Any]], species: str, trial_id: int
    ) -> tuple[list[str], int, int]:
        """L1 -> L2: cluster within a species, promote eligible clusters.

        Returns (new_pattern_ids, consolidated_count, skipped_below_mdl_count).
        """
        if len(entries) < MDL_MIN_CLUSTER_SIZE:
            return [], 0, 0

        emb_matrix = self._embed_all(entries)
        clusters = self._greedy_cluster(emb_matrix, self.cluster_sim)

        new_ids: list[str] = []
        consolidated = 0
        skipped = 0

        for cluster in clusters:
            if len(cluster) < MDL_MIN_CLUSTER_SIZE:
                continue
            cluster_entries = [entries[i] for i in cluster]

            # MDL check: representative description must compress the cluster.
            seed = max(cluster_entries, key=lambda e: e["validity"])
            seed_len = len(seed["description"]) + len(seed["insight"])
            total_len = sum(
                len(e["description"]) + len(e["insight"]) for e in cluster_entries
            )
            # Compression ratio = what we save by replacing N entries with 1.
            # Reject if pattern would not be < 60% of cluster total.
            if seed_len * 2 >= total_len:
                skipped += 1
                continue

            # Aggregate validity = mean of cluster, capped to leave room
            # for negative signal post-promotion (see update_validity).
            mean_validity = sum(e["validity"] for e in cluster_entries) / len(cluster_entries)
            source_ids = [e["id"] for e in cluster_entries]
            evidence_trial_ids = _dedupe_ints([
                trial_id
                for entry in cluster_entries
                for trial_id in entry.get("evidence_trial_ids", [])
            ])

            pattern_id = self.store.store(
                description=f"[PATTERN] {seed['description']}",
                insight=(
                    f"Consolidated from {len(cluster_entries)} trials "
                    f"(mean_validity={mean_validity:.2f}). {seed['insight']}"
                ),
                source_trial_id=trial_id,
                species=species,
                metadata={
                    "validity_score": min(0.9, mean_validity + 0.1),
                    "source_count": len(cluster_entries),
                    "source_ids": source_ids,
                    "compression_ratio": seed_len / max(total_len, 1),
                },
                entry_type="pattern",
                evidence_trial_ids=evidence_trial_ids,
            )

            # Quarantine the consolidated raw rows by aggressively bumping
            # their failure counter. Thirty bumps drop alpha/(alpha+beta)
            # below the 0.40 default quarantine threshold.
            for sid in source_ids:
                for _ in range(30):
                    self.store.update_validity(sid, failure=True)

            # Persist the convention summary in the dedicated NIB2-41 table
            # so structural_lab and other auditors can find it without
            # re-querying ``strategies``.
            try:
                trial_ids = [e["source_trial_id"] for e in cluster_entries]
                span = (min(trial_ids), max(trial_ids))
                self.store.add_convention(
                    representative=seed["description"],
                    member_ids=source_ids,
                    compression_ratio=seed_len / max(total_len, 1),
                    span_trials=span,
                )
            except Exception:
                # add_convention is best-effort — the pattern row is already
                # persisted, so a logging issue here must not abort the cycle.
                logger.exception("add_convention failed for pattern %s", pattern_id)

            new_ids.append(pattern_id)
            consolidated += len(cluster_entries)

        return new_ids, consolidated, skipped

    def _extract_conventions(
        self, patterns: list[dict[str, Any]], trial_id: int
    ) -> tuple[list[str], int]:
        """L2 -> L3: promote cross-species patterns to conventions."""
        emb_matrix = self._embed_all(patterns)
        clusters = self._greedy_cluster(emb_matrix, self.convention_sim)

        new_ids: list[str] = []
        consolidated = 0

        for cluster in clusters:
            if len(cluster) < MDL_MIN_CLUSTER_SIZE:
                continue
            cluster_entries = [patterns[i] for i in cluster]
            species_set = {e["species"] for e in cluster_entries}
            total_sources = sum(
                int(e["metadata"].get("source_count", 1)) for e in cluster_entries
            )
            evidence_trial_ids = _dedupe_ints([
                trial_id
                for entry in cluster_entries
                for trial_id in entry.get("evidence_trial_ids", [])
            ])

            if (
                len(species_set) < MIN_SPECIES_FOR_CONVENTION
                and total_sources < MIN_SOURCES_FOR_CONVENTION
            ):
                continue

            seed = max(cluster_entries, key=lambda e: e["validity"])
            description = seed["description"].replace("[PATTERN]", "[CONVENTION]").strip()
            convention_id = self.store.store(
                description=description if description.startswith("[CONVENTION]")
                else f"[CONVENTION] {description}",
                insight=(
                    f"Cross-species convention from {len(species_set)} species, "
                    f"{total_sources} total trials. {seed['insight']}"
                ),
                source_trial_id=trial_id,
                # Conventions live in the synthetic 'all' species so
                # species-filtered retrieves still surface them when callers
                # ask for them explicitly.
                species="all",
                metadata={
                    "validity_score": 0.9,
                    "species_sources": sorted(species_set),
                    "total_source_trials": total_sources,
                    "source_pattern_ids": [e["id"] for e in cluster_entries],
                },
                entry_type="convention",
                evidence_trial_ids=evidence_trial_ids,
            )
            new_ids.append(convention_id)
            consolidated += len(cluster_entries)

        return new_ids, consolidated
