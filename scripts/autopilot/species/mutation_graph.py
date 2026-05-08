"""Mutation Knowledge Graph for PromptForge (AP-31).

Tracks ``(mutation_type, failure_pattern, outcome, target_file)`` quadruples
across the autopilot trial history so PromptForge can:

    1. Avoid re-trying mutation types that consistently fail for a given
       failure pattern.
    2. Pick informed crossover sources — i.e. when assembling a new
       prompt from sections of two existing ones, prefer sections that
       have appeared on the Pareto frontier.

The graph is a single SQLite table with three indexes; we deliberately do
not pull in networkx because the queries we care about are all star-joins
on the four index columns. ``MutationGraph`` is a sidecar — PromptForge
can record outcomes and consult it for informed crossover, but the rest
of the species stays unchanged.

Schema::

    mutation_outcomes (
        id              INTEGER PRIMARY KEY,
        trial_id        INTEGER NOT NULL,
        mutation_type   TEXT    NOT NULL,
        failure_pattern TEXT    NOT NULL,   -- normalised tag (e.g. 'tool_compliance_low')
        target_file     TEXT    NOT NULL,
        outcome         TEXT    NOT NULL,   -- 'pareto_frontier' | 'accepted' | 'rejected' | 'safety_fail'
        delta_quality   REAL,
        delta_speed     REAL,
        section_ids     TEXT,               -- JSON list of prompt-section identifiers
        created_at      TEXT    NOT NULL
    )

The ``failure_pattern`` value is whatever the autopilot pipeline tags via
its existing failure analyzer; this module does not care about its
ontology, only about counting outcomes per (type, pattern) bucket.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/mutation_graph.db"
)

# Outcome categories — kept as plain strings rather than an Enum so the
# autopilot main loop can record any new category without a migration.
OUTCOME_PARETO = "pareto_frontier"
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"
OUTCOME_SAFETY_FAIL = "safety_fail"


@dataclass
class MutationOutcome:
    """One mutation_type × failure_pattern × outcome record."""

    trial_id: int
    mutation_type: str
    failure_pattern: str
    target_file: str
    outcome: str
    delta_quality: float = 0.0
    delta_speed: float = 0.0
    section_ids: list[str] = field(default_factory=list)


@dataclass
class TypePatternStats:
    """Aggregate counters for a single (mutation_type, failure_pattern) bucket."""

    mutation_type: str
    failure_pattern: str
    total: int
    pareto: int
    accepted: int
    rejected: int
    safety_fail: int
    mean_delta_quality: float
    mean_delta_speed: float

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.pareto + self.accepted) / self.total


class MutationGraph:
    """Sidecar SQLite store of PromptForge mutation outcomes (AP-31)."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS mutation_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id INTEGER NOT NULL,
                mutation_type TEXT NOT NULL,
                failure_pattern TEXT NOT NULL,
                target_file TEXT NOT NULL,
                outcome TEXT NOT NULL,
                delta_quality REAL,
                delta_speed REAL,
                section_ids TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mut_type_pattern
                ON mutation_outcomes(mutation_type, failure_pattern);
            CREATE INDEX IF NOT EXISTS idx_mut_target_file
                ON mutation_outcomes(target_file);
            CREATE INDEX IF NOT EXISTS idx_mut_outcome
                ON mutation_outcomes(outcome);
            """
        )
        self._conn.commit()

    # ── Recording ──────────────────────────────────────────────

    def record(self, outcome: MutationOutcome) -> int:
        """Persist a mutation outcome and return its row id."""
        cur = self._conn.execute(
            """INSERT INTO mutation_outcomes
               (trial_id, mutation_type, failure_pattern, target_file, outcome,
                delta_quality, delta_speed, section_ids, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(outcome.trial_id),
                outcome.mutation_type,
                outcome.failure_pattern,
                outcome.target_file,
                outcome.outcome,
                float(outcome.delta_quality),
                float(outcome.delta_speed),
                json.dumps(outcome.section_ids or []),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    # ── Aggregation ────────────────────────────────────────────

    def stats(
        self,
        mutation_type: Optional[str] = None,
        failure_pattern: Optional[str] = None,
    ) -> list[TypePatternStats]:
        """Aggregate counters per (mutation_type, failure_pattern) bucket.

        Filters apply when supplied. Buckets with zero rows are not returned.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if mutation_type is not None:
            clauses.append("mutation_type = ?")
            params.append(mutation_type)
        if failure_pattern is not None:
            clauses.append("failure_pattern = ?")
            params.append(failure_pattern)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        rows = self._conn.execute(
            f"""SELECT mutation_type, failure_pattern, outcome,
                       delta_quality, delta_speed
                FROM mutation_outcomes {where}""",
            tuple(params),
        ).fetchall()

        buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "pareto": 0,
                "accepted": 0,
                "rejected": 0,
                "safety_fail": 0,
                "delta_q_sum": 0.0,
                "delta_s_sum": 0.0,
            }
        )
        for row in rows:
            key = (row["mutation_type"], row["failure_pattern"])
            b = buckets[key]
            b["total"] += 1
            b["delta_q_sum"] += row["delta_quality"] or 0.0
            b["delta_s_sum"] += row["delta_speed"] or 0.0
            outcome = row["outcome"]
            if outcome == OUTCOME_PARETO:
                b["pareto"] += 1
            elif outcome == OUTCOME_ACCEPTED:
                b["accepted"] += 1
            elif outcome == OUTCOME_REJECTED:
                b["rejected"] += 1
            elif outcome == OUTCOME_SAFETY_FAIL:
                b["safety_fail"] += 1

        out: list[TypePatternStats] = []
        for (mt, fp), b in buckets.items():
            mean_q = b["delta_q_sum"] / b["total"] if b["total"] else 0.0
            mean_s = b["delta_s_sum"] / b["total"] if b["total"] else 0.0
            out.append(
                TypePatternStats(
                    mutation_type=mt,
                    failure_pattern=fp,
                    total=b["total"],
                    pareto=b["pareto"],
                    accepted=b["accepted"],
                    rejected=b["rejected"],
                    safety_fail=b["safety_fail"],
                    mean_delta_quality=mean_q,
                    mean_delta_speed=mean_s,
                )
            )
        out.sort(key=lambda s: s.success_rate, reverse=True)
        return out

    # ── Decision support ───────────────────────────────────────

    def best_mutation_for(
        self,
        failure_pattern: str,
        candidate_types: Optional[Iterable[str]] = None,
        min_trials: int = 3,
    ) -> Optional[str]:
        """Return the mutation_type with highest success rate for a pattern.

        ``min_trials`` guards against false-validation-confidence — a
        bucket needs at least ``min_trials`` rows before it can win.
        Returns ``None`` if no bucket meets the threshold; caller falls
        back to its default mutation-type sampling logic.
        """
        all_stats = self.stats(failure_pattern=failure_pattern)
        if candidate_types is not None:
            candidate_set = set(candidate_types)
            all_stats = [s for s in all_stats if s.mutation_type in candidate_set]
        eligible = [s for s in all_stats if s.total >= min_trials]
        if not eligible:
            return None
        return eligible[0].mutation_type

    def avoid_for(
        self,
        failure_pattern: str,
        max_success_rate: float = 0.10,
        min_trials: int = 5,
    ) -> set[str]:
        """Return mutation types we should AVOID for ``failure_pattern``.

        A type lands here if it has ``min_trials`` rows under that pattern
        and its success rate is below ``max_success_rate``.
        """
        avoid: set[str] = set()
        for s in self.stats(failure_pattern=failure_pattern):
            if s.total >= min_trials and s.success_rate <= max_success_rate:
                avoid.add(s.mutation_type)
        return avoid

    def pareto_best_sections(
        self, target_file: Optional[str] = None, top_n: int = 10
    ) -> list[tuple[str, int]]:
        """Most-frequent prompt-section ids that landed on the Pareto frontier.

        Used by informed crossover: when assembling a new prompt for
        ``target_file`` from sections of two siblings, prefer sections that
        themselves have a track record of contributing to a Pareto frontier
        outcome.

        Returns ``[(section_id, frequency), …]``, longest streaks first.
        ``target_file`` filters to mutations on that file when supplied.
        """
        clauses = ["outcome = ?"]
        params: list[Any] = [OUTCOME_PARETO]
        if target_file is not None:
            clauses.append("target_file = ?")
            params.append(target_file)
        where = "WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT section_ids FROM mutation_outcomes {where}", tuple(params)
        ).fetchall()
        counter: Counter[str] = Counter()
        for row in rows:
            try:
                ids = json.loads(row["section_ids"] or "[]")
            except json.JSONDecodeError:
                continue
            for sid in ids:
                if sid:
                    counter[str(sid)] += 1
        return counter.most_common(top_n)

    def informed_crossover_candidates(
        self,
        target_file: str,
        min_pareto_count: int = 1,
        top_n: int = 10,
    ) -> list[str]:
        """Suggest section ids for an informed crossover into ``target_file``.

        Returns just the section ids (without frequencies) with at least
        ``min_pareto_count`` Pareto-frontier appearances. Plug into
        ``PromptForge._build_mutation_prompt`` as a ``preferred_sections``
        hint when running ``crossover`` mutations.
        """
        ranked = self.pareto_best_sections(target_file=target_file, top_n=top_n)
        return [sid for sid, count in ranked if count >= min_pareto_count]

    # ── Lifecycle ──────────────────────────────────────────────

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "MutationGraph":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
