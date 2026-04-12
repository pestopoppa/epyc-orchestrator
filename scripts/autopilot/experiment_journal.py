"""Experiment journal: dual TSV + JSONL logging for AutoPilot trials.

Append-only with rotation (new file per 1000 trials).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any


class DeficiencyCategory(str, Enum):
    """Structured failure classification for safety gate violations (AP-14).

    Each category maps to a specific SafetyGate check or dispatch_action guard.
    Using str mixin for natural JSON serialization in JSONL journal.
    """
    QUALITY_FLOOR = "quality_floor"
    REGRESSION = "regression"
    PER_SUITE = "per_suite_regression"
    ROUTING_DIVERSITY = "routing_diversity"
    THROUGHPUT = "throughput"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    CODE_VALIDATION = "code_validation"
    SHRINKAGE = "shrinkage"
    REVERT = "revert"

DEFAULT_JOURNAL_DIR = Path(__file__).resolve().parents[2] / "orchestration"
MAX_TRIALS_PER_FILE = 1000

TSV_COLUMNS = [
    "trial_id",
    "timestamp",
    "species",
    "action_type",
    "tier",
    "quality",
    "speed",
    "cost",
    "reliability",
    "pareto_status",
    "git_tag",
    "reasoning_hash",
]


@dataclass
class JournalEntry:
    trial_id: int
    timestamp: str
    species: str
    action_type: str
    tier: int  # 0, 1, or 2
    quality: float
    speed: float
    cost: float
    reliability: float
    pareto_status: str  # "dominated", "candidate", "frontier"
    git_tag: str = ""
    reasoning_hash: str = ""
    # Full detail goes into JSONL only
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    config_diff: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    parent_trial: int | None = None
    memory_count: int = 0
    active_flags: list[str] = field(default_factory=list)
    eval_details: dict[str, Any] = field(default_factory=dict)
    failure_analysis: str = ""
    hypothesis: str = ""
    expected_mechanism: str = ""
    deficiency_category: str = ""  # AP-14: DeficiencyCategory value or empty
    instruction_token_count: int = 0  # AP-16: per-request instruction overhead
    instruction_token_ratio: float = 0.0  # AP-16: instruction_tokens / total_input
    self_criticism: str = ""  # AP-23: structured self-criticism from last trial
    keep_revert_decision: str = ""  # AP-24: "keep" | "revert" | ""
    optimization_directions: str = ""  # AP-24: forward-looking next-round guidance


class ExperimentJournal:
    """Append-only experiment log with TSV (human-readable) + JSONL (machine-readable)."""

    def __init__(self, journal_dir: Path | None = None):
        self.journal_dir = journal_dir or DEFAULT_JOURNAL_DIR
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[JournalEntry] = []
        self._load_existing()

    # ── persistence ──────────────────────────────────────────────

    def _tsv_path(self, batch: int = 0) -> Path:
        suffix = f"_{batch}" if batch > 0 else ""
        return self.journal_dir / f"autopilot_journal{suffix}.tsv"

    def _jsonl_path(self, batch: int = 0) -> Path:
        suffix = f"_{batch}" if batch > 0 else ""
        return self.journal_dir / f"autopilot_journal{suffix}.jsonl"

    def _current_batch(self) -> int:
        if not self._entries:
            return 0
        return self._entries[-1].trial_id // MAX_TRIALS_PER_FILE

    def _load_existing(self) -> None:
        """Load entries from all existing JSONL files."""
        batch = 0
        while True:
            jsonl = self._jsonl_path(batch)
            if not jsonl.exists():
                break
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entry = JournalEntry(
                        trial_id=data["trial_id"],
                        timestamp=data["timestamp"],
                        species=data["species"],
                        action_type=data["action_type"],
                        tier=data.get("tier", 0),
                        quality=data.get("quality", 0.0),
                        speed=data.get("speed", 0.0),
                        cost=data.get("cost", 0.0),
                        reliability=data.get("reliability", 0.0),
                        pareto_status=data.get("pareto_status", "dominated"),
                        git_tag=data.get("git_tag", ""),
                        reasoning_hash=data.get("reasoning_hash", ""),
                        config_snapshot=data.get("config_snapshot", {}),
                        config_diff=data.get("config_diff", {}),
                        reasoning=data.get("reasoning", ""),
                        parent_trial=data.get("parent_trial"),
                        memory_count=data.get("memory_count", 0),
                        active_flags=data.get("active_flags", []),
                        eval_details=data.get("eval_details", {}),
                        failure_analysis=data.get("failure_analysis", ""),
                        hypothesis=data.get("hypothesis", ""),
                        expected_mechanism=data.get("expected_mechanism", ""),
                        deficiency_category=data.get("deficiency_category", ""),
                        instruction_token_count=data.get("instruction_token_count", 0),
                        instruction_token_ratio=data.get("instruction_token_ratio", 0.0),
                    )
                    self._entries.append(entry)
            batch += 1

    # ── writing ──────────────────────────────────────────────────

    def record(self, entry: JournalEntry) -> None:
        """Append a trial entry to both TSV and JSONL."""
        batch = entry.trial_id // MAX_TRIALS_PER_FILE
        tsv = self._tsv_path(batch)
        jsonl = self._jsonl_path(batch)

        # TSV (human-readable subset)
        write_header = not tsv.exists()
        with open(tsv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerow({col: getattr(entry, col) for col in TSV_COLUMNS})

        # JSONL (full detail)
        with open(jsonl, "a") as f:
            f.write(json.dumps(asdict(entry), default=str) + "\n")

        self._entries.append(entry)

    # ── queries ──────────────────────────────────────────────────

    def recent(self, n: int = 20) -> list[JournalEntry]:
        """Return last n entries."""
        return self._entries[-n:]

    def all_entries(self) -> list[JournalEntry]:
        return list(self._entries)

    def count(self) -> int:
        return len(self._entries)

    def next_trial_id(self) -> int:
        if not self._entries:
            return 0
        return self._entries[-1].trial_id + 1

    def by_species(self, species: str) -> list[JournalEntry]:
        return [e for e in self._entries if e.species == species]

    def pareto_entries(self) -> list[JournalEntry]:
        return [e for e in self._entries if e.pareto_status == "frontier"]

    def summary(self) -> dict[str, Any]:
        """Compact summary for controller consumption."""
        if not self._entries:
            return {"total_trials": 0, "species_counts": {}, "pareto_size": 0}

        species_counts: dict[str, int] = {}
        pareto_count = 0
        for e in self._entries:
            species_counts[e.species] = species_counts.get(e.species, 0) + 1
            if e.pareto_status == "frontier":
                pareto_count += 1

        last = self._entries[-1]
        return {
            "total_trials": len(self._entries),
            "species_counts": species_counts,
            "pareto_size": pareto_count,
            "last_trial_id": last.trial_id,
            "last_species": last.species,
            "last_quality": last.quality,
            "last_speed": last.speed,
        }

    def summary_text(self, last_n: int = 20) -> str:
        """Human-readable summary for LLM controller prompt."""
        s = self.summary()
        lines = [
            f"Total trials: {s['total_trials']}",
            f"Pareto frontier size: {s['pareto_size']}",
            f"Species counts: {s.get('species_counts', {})}",
        ]
        recent = self.recent(last_n)
        if recent:
            lines.append(f"\nLast {len(recent)} trials:")
            for e in recent:
                line = (
                    f"  #{e.trial_id} [{e.species}/{e.action_type}] "
                    f"T{e.tier} q={e.quality:.3f} s={e.speed:.1f} "
                    f"c={e.cost:.3f} r={e.reliability:.2f} "
                    f"→ {e.pareto_status}"
                )
                if e.failure_analysis:
                    # Compact single-line failure summary for controller visibility
                    fa_oneline = e.failure_analysis.replace("\n", " | ")[:200]
                    line += f"  FAILED: {fa_oneline}"
                lines.append(line)
        return "\n".join(lines)

    def recent_failures(
        self, species: str | None = None, n: int = 10
    ) -> list[JournalEntry]:
        """Return the last n entries with non-empty failure_analysis.

        Optionally filter by species name.
        """
        failed = [
            e for e in self._entries
            if e.failure_analysis
            and (species is None or e.species == species)
        ]
        return failed[-n:]

    def suite_quality_trend(
        self, last_n: int = 10
    ) -> dict[str, list[tuple[int, float]]]:
        """Per-suite quality over the last n trials that have suite data.

        Returns {suite_name: [(trial_id, quality), ...]} sorted by trial_id.
        """
        entries_with_suites = [
            e for e in self._entries
            if e.eval_details.get("per_suite_quality")
        ][-last_n:]

        trends: dict[str, list[tuple[int, float]]] = {}
        for e in entries_with_suites:
            for suite, q in e.eval_details["per_suite_quality"].items():
                trends.setdefault(suite, []).append((e.trial_id, q))
        return trends

    # ── insights ──────────────────────────────────────────────────

    def insights_text(self, n: int = 10) -> str:
        """Synthesize actionable insights from recent trials.

        Extracts hypothesis + outcome from trials that either reached the
        Pareto frontier or failed safety gates — the two outcomes worth
        learning from.  Returns a compact text block suitable for injection
        into species prompts (cross-species fertilization).
        """
        interesting = [
            e for e in self._entries
            if e.pareto_status == "frontier" or e.failure_analysis
        ][-n:]
        if not interesting:
            return "(no insights yet)"

        lines: list[str] = []
        for e in interesting:
            tag = "SUCCESS" if e.pareto_status == "frontier" else "FAILED"
            hyp = e.hypothesis or e.action_type
            mechanism = e.expected_mechanism or ""
            detail = ""
            if e.pareto_status == "frontier":
                detail = f"q={e.quality:.3f} s={e.speed:.1f}"
            elif e.failure_analysis:
                # Compact single-line failure summary
                detail = e.failure_analysis.replace("\n", " | ")[:120]
            species_label = e.species
            lines.append(
                f"  [{tag}] #{e.trial_id} ({species_label}/{hyp})"
                + (f" [{mechanism}]" if mechanism else "")
                + f": {detail}"
            )
        return "\n".join(lines)

    # ── species effectiveness ────────────────────────────────────

    def species_effectiveness(
        self, window: int | None = None
    ) -> dict[str, dict[str, float]]:
        """Pareto improvement rate per species.

        Returns {species: {total: N, pareto: M, rate: M/N}} for each species.
        """
        entries = self._entries[-window:] if window else self._entries
        stats: dict[str, dict[str, float]] = {}
        for e in entries:
            if e.species not in stats:
                stats[e.species] = {"total": 0, "pareto": 0, "rate": 0.0}
            stats[e.species]["total"] += 1
            if e.pareto_status == "frontier":
                stats[e.species]["pareto"] += 1
        for sp in stats:
            total = stats[sp]["total"]
            stats[sp]["rate"] = stats[sp]["pareto"] / total if total > 0 else 0.0
        return stats
