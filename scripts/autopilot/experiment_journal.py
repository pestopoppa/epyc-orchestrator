"""Experiment journal: dual TSV + JSONL logging for AutoPilot trials.

Append-only with rotation (new file per 1000 trials).
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
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
    # 2026-06-04 non-executing-action residue (graph_router deadlock fix). An
    # action that never ran an eval — so it has no quality/speed evidence — but
    # whose failure reason is actionable and MUST be fed back to the planner
    # instead of being silently dropped (the "return None, increment, continue"
    # blind spot that let 119 identical invalid structural_experiments dispatch).
    #   INVALID_ACTION: failed pre-execution validation (e.g. a feature flag
    #     whose dependency is not enabled). Carries the validator reason.
    #   DISPATCH_SKIPPED: skipped at the dispatcher (AP-9 scope violation,
    #     dirty-tree fence, unknown action type, or a handler no-op).
    INVALID_ACTION = "invalid_action"
    DISPATCH_SKIPPED = "dispatch_skipped"
    # 2026-05-23 exogenous-restart resilience (handoff Phase 5).
    # EXOGENOUS_RELOAD: trial corrupted by an operator/external service reload
    #   detected via fleet markers; at least one question stayed unrecovered.
    #   bug_corrupted_by is set to "exogenous_operator_reload" so the planner's
    #   trustworthiness gate excludes it from hypothesis chains. SafetyGate +
    #   Pareto archive are explicitly SKIPPED for this trial.
    # AUTOPILOT_KILLED: placeholder JournalEntry written by the cmd_start
    #   recovery path when in_flight_trial indicates a crash between
    #   dispatch_action and journal.record. No eval evidence available.
    EXOGENOUS_RELOAD = "exogenous_reload"
    AUTOPILOT_KILLED = "autopilot_killed_mid_trial"
    # 2026-05-24: trial completed during a host page-cache flush + NUMA re-warm
    # window (`host_health.flush_cache_with_pause()` runs serial GGUF rewarms
    # taking ~30-60s during which decode throughput is suppressed). Like
    # EXOGENOUS_RELOAD, bug_corrupted_by gets set so the planner's
    # trustworthiness gate excludes the affected trial from hypothesis chains.
    EXOGENOUS_CACHE_FLUSH = "exogenous_cache_flush"

DEFAULT_JOURNAL_DIR = Path(__file__).resolve().parents[2] / "orchestration"
MAX_TRIALS_PER_FILE = 1000

_BASELINE_QUALITY_RE = re.compile(r"\bbaseline\s+([0-9]+(?:\.[0-9]+)?)")
_SUITE_REGRESSION_RE = re.compile(
    r"\bSuite\s+'[^']+'\s+regression:\s+-([0-9]+(?:\.[0-9]+)?)"
)
_MAX_QUALITY_SCALE = 3.0
_LEGACY_SCALE_FAILURE_SUMMARY = (
    "legacy-scale failure_analysis omitted: references impossible 0-3 quality "
    "baseline/per-suite regression; use recorded q/s/r fields instead"
)

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

SUPERSESSION_EVENT_TYPE = "supersession"
BASELINE_PROMOTION_EVENT_TYPE = "baseline_promotion"
JOURNAL_SNAPSHOT_EVENT_TYPE = "journal_snapshot"


def has_legacy_scale_failure_analysis(text: str) -> bool:
    """True when old failure text carries impossible 0-3 quality-scale values."""
    if not text:
        return False
    baseline_values = (float(m.group(1)) for m in _BASELINE_QUALITY_RE.finditer(text))
    if any(v > _MAX_QUALITY_SCALE for v in baseline_values):
        return True
    suite_deltas = (float(m.group(1)) for m in _SUITE_REGRESSION_RE.finditer(text))
    return any(v > _MAX_QUALITY_SCALE for v in suite_deltas)


def failure_analysis_for_prompt(entry: "JournalEntry", limit: int | None = None) -> str:
    """Render failure_analysis for controller-facing prompts without stale scale leaks."""
    if has_legacy_scale_failure_analysis(entry.failure_analysis):
        text = _LEGACY_SCALE_FAILURE_SUMMARY
    else:
        text = entry.failure_analysis.replace("\n", " | ")
    if limit is not None:
        return text[:limit]
    return text


def scrub_legacy_scale_text(text: str) -> str:
    """Redact any free-text field carrying impossible 0-3 quality-scale values.

    Used for controller-facing fields beyond failure_analysis (self_criticism,
    optimization_directions) that historically embedded the regression-gate
    string "… vs baseline 9.900". Returns the legacy-scale summary when the text
    references an impossible baseline/per-suite value, else the text unchanged —
    so the planner never re-surfaces a corrupt baseline that has since been fixed.
    """
    if has_legacy_scale_failure_analysis(text):
        return _LEGACY_SCALE_FAILURE_SUMMARY
    return text


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
    metric_schema_version: int = 1  # HLE-4 observe-only schema version.
    harness_metrics: dict[str, Any] = field(default_factory=dict)
    oracle_adequacy: dict[str, Any] = field(default_factory=dict)
    failure_analysis: str = ""
    hypothesis: str = ""
    expected_mechanism: str = ""
    deficiency_category: str = ""  # AP-14: DeficiencyCategory value or empty
    instruction_token_count: int = 0  # AP-16: per-request instruction overhead
    instruction_token_ratio: float = 0.0  # AP-16: instruction_tokens / total_input
    self_criticism: str = ""  # AP-23: structured self-criticism from last trial
    keep_revert_decision: str = ""  # AP-24: "keep" | "revert" | "excluded" | ""
    optimization_directions: str = ""  # AP-24: forward-looking next-round guidance
    predicted_objectives: dict[str, float] = field(default_factory=dict)  # PEAF: controller's pre-trial forecast (empty when disabled / unforecast)
    surprise_score: float | None = None  # PEAF: L1 distance in normalised objective space; None when no forecast
    # 2026-05-23: journal-pollution tracking. When a trial's outcome was
    # caused by an orchestrator bug that has since been fixed, the operator
    # runs scripts/autopilot/scrub_journal.py to set this to the short SHA of
    # the bug-fix commit. The planner's hypothesis-chain reasoning filters
    # bug_corrupted entries out of "trustworthy" trial counts so it doesn't
    # learn wrong lessons from buggy outcomes. Empty string == trustworthy.
    bug_corrupted_by: str = ""
    bug_corrupted_reason: str = ""  # free-text operator note (~80c) for context
    # 2026-05-23: constrained-creativity planner upgrade. `falsifier` holds the
    # one-line predicted outcome whose absence would invalidate the trial's
    # hypothesis (emitted by the controller in the rationale block).
    # `rubric_scores` holds the controller's self-scoring on info_gain /
    # coherence / usefulness plus an optional synthesis note. Both default
    # empty so legacy entries stay loadable.
    falsifier: str = ""
    rubric_scores: dict[str, Any] = field(default_factory=dict)
    # 2026-06-04: outcome status for non-executing trials. "ok" for a normal
    # metric-collecting trial; "invalid" when the action failed pre-execution
    # validation (e.g. a flag dependency); "skipped" when the dispatcher dropped
    # it (AP-9 scope / dirty-tree / unknown / handler no-op). Lets the planner and
    # audits distinguish "ran and scored 0" from "never ran" — the residue that
    # was previously discarded by the bare-None skip path.
    outcome_status: str = "ok"
    # 2026-05-24: stagnation gate signal that fired for this trial (empty for
    # lean-prompt trials, or e.g. "hv_slope_10=+0.00000 < eps=0.00100;
    # last 3 trials all action_type=seed_batch" when the rich prompt was used).
    # Enables retrospective behavioral analysis: action-type diversity under
    # rich vs lean prompts without running a separate experiment.
    stagnation_signal: str = ""


@dataclass
class SupersessionEvent:
    """Append-only ledger event superseding fields on prior trial rows.

    This is the Phase-3 event-sourcing bridge: new tooling can append a durable
    intent record without rewriting historical trial rows. Runtime consumers can
    fold these events later; until then the legacy scrub path remains available.
    """

    target_trial_ids: list[int]
    fields: dict[str, Any]
    reason: str
    policy_version: str
    actor: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = SUPERSESSION_EVENT_TYPE


@dataclass
class BaselinePromotionEvent:
    """Append-only ledger event recording an accepted production baseline move."""

    source_trial_id: int
    tier: int
    previous_quality: float | None
    new_quality: float
    reason: str
    proof: dict[str, Any]
    result_metrics: dict[str, Any]
    baseline_state: dict[str, Any]
    policy_version: str
    actor: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = BASELINE_PROMOTION_EVENT_TYPE


@dataclass
class JournalSnapshotEvent:
    """Append-only segment snapshot for bounded journal replay."""

    through_trial_id: int
    snapshot: dict[str, Any]
    policy_version: str
    actor: str
    parent_snapshot_hash: str = ""
    snapshot_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = JOURNAL_SNAPSHOT_EVENT_TYPE


def _snapshot_hash(
    *,
    through_trial_id: int,
    snapshot: dict[str, Any],
    policy_version: str,
    parent_snapshot_hash: str = "",
) -> str:
    payload = {
        "through_trial_id": int(through_trial_id),
        "snapshot": snapshot,
        "policy_version": policy_version,
        "parent_snapshot_hash": parent_snapshot_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ExperimentJournal:
    """Append-only experiment log with TSV (human-readable) + JSONL (machine-readable)."""

    def __init__(self, journal_dir: Path | None = None):
        self.journal_dir = journal_dir or DEFAULT_JOURNAL_DIR
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[JournalEntry] = []
        self._ledger_events_by_batch: dict[int, list[dict[str, Any]]] = {}
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
                    if data.get("type") and "trial_id" not in data:
                        self._ledger_events_by_batch.setdefault(batch, []).append(data)
                        continue
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
                        metric_schema_version=data.get(
                            "metric_schema_version",
                            data.get("eval_details", {}).get("metric_schema_version", 1),
                        ),
                        harness_metrics=data.get(
                            "harness_metrics",
                            data.get("eval_details", {}).get("harness_metrics", {}),
                        ),
                        oracle_adequacy=data.get(
                            "oracle_adequacy",
                            data.get("eval_details", {}).get("oracle_adequacy", {}),
                        ),
                        failure_analysis=data.get("failure_analysis", ""),
                        hypothesis=data.get("hypothesis", ""),
                        expected_mechanism=data.get("expected_mechanism", ""),
                        deficiency_category=data.get("deficiency_category", ""),
                        instruction_token_count=data.get("instruction_token_count", 0),
                        instruction_token_ratio=data.get("instruction_token_ratio", 0.0),
                        self_criticism=data.get("self_criticism", ""),
                        keep_revert_decision=data.get("keep_revert_decision", ""),
                        optimization_directions=data.get("optimization_directions", ""),
                        predicted_objectives=data.get("predicted_objectives", {}),
                        surprise_score=data.get("surprise_score", None),
                        bug_corrupted_by=data.get("bug_corrupted_by", ""),
                        bug_corrupted_reason=data.get("bug_corrupted_reason", ""),
                        falsifier=data.get("falsifier", ""),
                        rubric_scores=data.get("rubric_scores", {}),
                        stagnation_signal=data.get("stagnation_signal", ""),
                        outcome_status=data.get("outcome_status", "ok"),
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

    def append_ledger_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append a non-trial ledger event row to JSONL."""
        event = copy.deepcopy(event)
        if not event.get("type"):
            raise ValueError("ledger events require a type")
        if "trial_id" in event:
            raise ValueError("ledger events must not use the trial_id field")
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        batch = self._current_batch()
        jsonl = self._jsonl_path(batch)
        with open(jsonl, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        self._ledger_events_by_batch.setdefault(batch, []).append(event)
        return event

    def append_supersession_event(
        self,
        *,
        target_trial_ids: list[int],
        fields: dict[str, Any],
        reason: str,
        policy_version: str,
        actor: str,
    ) -> dict[str, Any]:
        """Append a supersession event row to JSONL without mutating trials."""
        return self.append_ledger_event(asdict(
            SupersessionEvent(
                target_trial_ids=target_trial_ids,
                fields=fields,
                reason=reason,
                policy_version=policy_version,
                actor=actor,
            )
        ))

    def append_baseline_promotion_event(
        self,
        *,
        source_trial_id: int,
        tier: int,
        previous_quality: float | None,
        new_quality: float,
        reason: str,
        proof: dict[str, Any],
        result_metrics: dict[str, Any],
        baseline_state: dict[str, Any],
        policy_version: str = "baseline-promotion-v1",
        actor: str = "autopilot.py",
    ) -> dict[str, Any]:
        """Append a baseline-promotion event row without changing baseline state."""
        return self.append_ledger_event(asdict(
            BaselinePromotionEvent(
                source_trial_id=source_trial_id,
                tier=tier,
                previous_quality=previous_quality,
                new_quality=new_quality,
                reason=reason,
                proof=proof,
                result_metrics=result_metrics,
                baseline_state=baseline_state,
                policy_version=policy_version,
                actor=actor,
            )
        ))

    def append_journal_snapshot_event(
        self,
        *,
        through_trial_id: int,
        snapshot: dict[str, Any],
        policy_version: str,
        actor: str,
        parent_snapshot_hash: str = "",
    ) -> dict[str, Any]:
        """Append a segment snapshot row without changing replay authority."""
        snapshot_hash = _snapshot_hash(
            through_trial_id=through_trial_id,
            snapshot=snapshot,
            policy_version=policy_version,
            parent_snapshot_hash=parent_snapshot_hash,
        )
        return self.append_ledger_event(asdict(
            JournalSnapshotEvent(
                through_trial_id=int(through_trial_id),
                snapshot=copy.deepcopy(snapshot),
                policy_version=policy_version,
                actor=actor,
                parent_snapshot_hash=parent_snapshot_hash,
                snapshot_hash=snapshot_hash,
            )
        ))

    # ── queries ──────────────────────────────────────────────────

    def recent(self, n: int = 20) -> list[JournalEntry]:
        """Return last n entries."""
        return self._entries[-n:]

    def all_entries(self) -> list[JournalEntry]:
        return list(self._entries)

    def count(self) -> int:
        return len(self._entries)

    def ledger_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Return loaded append-only ledger event rows, optionally filtered by type."""
        events: list[dict[str, Any]] = []
        for batch in sorted(self._ledger_events_by_batch):
            events.extend(
                event
                for event in self._ledger_events_by_batch[batch]
                if event_type is None or event.get("type") == event_type
            )
        return events

    def supersession_events(self) -> list[dict[str, Any]]:
        """Return loaded append-only supersession event rows."""
        return self.ledger_events(SUPERSESSION_EVENT_TYPE)

    def baseline_promotion_events(self) -> list[dict[str, Any]]:
        """Return loaded append-only baseline promotion event rows."""
        return self.ledger_events(BASELINE_PROMOTION_EVENT_TYPE)

    def journal_snapshot_events(self) -> list[dict[str, Any]]:
        """Return loaded append-only journal snapshot event rows."""
        return self.ledger_events(JOURNAL_SNAPSHOT_EVENT_TYPE)

    def latest_journal_snapshot_event(self) -> dict[str, Any] | None:
        """Return the newest snapshot event by ledger order, if any."""
        events = self.journal_snapshot_events()
        return events[-1] if events else None

    def _supersession_overrides_by_trial(self) -> dict[int, dict[str, Any]]:
        entry_fields = set(JournalEntry.__dataclass_fields__)
        overrides_by_trial: dict[int, dict[str, Any]] = {}
        for event in self.supersession_events():
            fields = event.get("fields")
            targets = event.get("target_trial_ids")
            if not isinstance(fields, dict) or not isinstance(targets, list):
                continue
            filtered_fields = {
                str(name): copy.deepcopy(value)
                for name, value in fields.items()
                if str(name) in entry_fields
            }
            if not filtered_fields:
                continue
            for target in targets:
                try:
                    trial_id = int(target)
                except (TypeError, ValueError):
                    continue
                overrides_by_trial.setdefault(trial_id, {}).update(
                    copy.deepcopy(filtered_fields)
                )
        return overrides_by_trial

    def entries_with_supersessions(self) -> list[JournalEntry]:
        """Return trial entries with append-only supersession events folded in.

        This is the runtime read view: persisted trial rows stay immutable, while
        planner-facing trust and prompt helpers see operator supersession events
        such as post-hoc resource-contention exclusions.
        """
        overrides_by_trial = self._supersession_overrides_by_trial()
        if not overrides_by_trial:
            return list(self._entries)
        entries: list[JournalEntry] = []
        for entry in self._entries:
            overrides = overrides_by_trial.get(entry.trial_id)
            if overrides:
                entries.append(replace(entry, **copy.deepcopy(overrides)))
            else:
                entries.append(entry)
        return entries

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
        entries = self.entries_with_supersessions()
        eligible_pareto_count = sum(
            1
            for e in entries
            if e.pareto_status == "frontier" and e.tier > 0 and not e.bug_corrupted_by
        )
        lines = [
            f"Total trials: {s['total_trials']}",
            f"Eligible Pareto frontier size (T1/T2, trustworthy): {eligible_pareto_count}",
            f"Species counts: {s.get('species_counts', {})}",
        ]
        recent = entries[-last_n:]
        if recent:
            lines.append(f"\nLast {len(recent)} trials:")
            for i, e in enumerate(recent):
                prefix = f"  #{e.trial_id} [{e.species}/{e.action_type}] "
                if e.bug_corrupted_by:
                    line = (
                        prefix
                        + f"CORRUPTED_BY={e.bug_corrupted_by} "
                        + "(metrics/reason hidden; excluded from planner trust)"
                    )
                    lines.append(line)
                    continue
                if e.tier == 0:
                    line = (
                        prefix
                        + "T0 audit-only sentinel "
                        + "(quality hidden; excluded from production frontier/baseline guards) "
                        + f"s={e.speed:.1f} c={e.cost:.3f} r={e.reliability:.2f} "
                        + f"→ {e.pareto_status}"
                    )
                else:
                    line = (
                        prefix
                        + f"T{e.tier} q={e.quality:.3f} s={e.speed:.1f} "
                        + f"c={e.cost:.3f} r={e.reliability:.2f} "
                        + f"→ {e.pareto_status}"
                    )
                if e.failure_analysis:
                    # Compact single-line failure summary for controller visibility.
                    # Prompt-budget trim (2026-06-10): full detail only for the most
                    # recent 6 failures (the ones the planner actually reasons about
                    # next); older failures show a short tag so the journal section
                    # stays small. Cap shortened 200→140.
                    if i >= len(recent) - 6:
                        fa_oneline = failure_analysis_for_prompt(e, limit=140)
                        line += f"  FAILED: {fa_oneline}"
                    else:
                        fa_oneline = failure_analysis_for_prompt(e, limit=60)
                        line += f"  FAILED({fa_oneline})"
                lines.append(line)
        return "\n".join(lines)

    # ── bug-corruption tracking (2026-05-23) ─────────────────────

    def trustworthy_entries(self) -> list[JournalEntry]:
        """All entries whose outcome is NOT marked bug_corrupted_by."""
        return [e for e in self.entries_with_supersessions() if not e.bug_corrupted_by]

    def trustworthiness_score(self) -> dict[str, Any]:
        """Counts of trustworthy vs bug-corrupted entries.

        Returned shape:
            {
                "total": N,
                "trustworthy": M,
                "corrupted": N - M,
                "ratio": M / N (or 1.0 when N == 0),
                "corrupted_by": {"de34dd4": K1, "b3895aa": K2, ...},
                "low_signal": bool,  # True when trustworthy < 5
            }
        """
        entries = self.entries_with_supersessions()
        total = len(entries)
        if total == 0:
            return {
                "total": 0, "trustworthy": 0, "corrupted": 0,
                "ratio": 1.0, "corrupted_by": {}, "low_signal": True,
            }
        corrupted_by: dict[str, int] = {}
        trustworthy = 0
        for e in entries:
            sha = e.bug_corrupted_by or ""
            if sha:
                corrupted_by[sha] = corrupted_by.get(sha, 0) + 1
            else:
                trustworthy += 1
        return {
            "total": total,
            "trustworthy": trustworthy,
            "corrupted": total - trustworthy,
            "ratio": trustworthy / total,
            "corrupted_by": corrupted_by,
            "low_signal": trustworthy < 5,
        }

    def recent_hypotheses(self, n: int = 3, exclude_bug_corrupted: bool = True) -> list[JournalEntry]:
        """Return the last n entries with a non-empty hypothesis field.

        When `exclude_bug_corrupted` is True, skip entries marked by
        scrub_journal so the planner's hypothesis-chain reasoning learns
        from real signal only.
        """
        pool = (
            self.trustworthy_entries() if exclude_bug_corrupted else self._entries
        )
        with_hyp = [e for e in pool if e.hypothesis]
        return with_hyp[-n:]

    def action_diversity_by_gate(
        self, window: int = 50
    ) -> dict[str, Any]:
        """Compare action-type diversity for rich-fragment trials (stagnation
        signal fired) vs lean-fragment trials (empty signal) over the last
        `window` controller-mode entries.

        Returns counts, distinct action_types, and Shannon entropy per bucket
        so operators can verify the gate is doing what it should: rich-prompt
        trials ought to explore more action_types than lean-prompt trials.
        Empty buckets return entropy=0.0 with count=0.
        """
        import math
        recent = self._entries[-window:]
        rich = [e for e in recent if e.stagnation_signal]
        lean = [e for e in recent if not e.stagnation_signal]

        def _stats(bucket: list[JournalEntry]) -> dict[str, Any]:
            if not bucket:
                return {"count": 0, "distinct_action_types": 0, "entropy_bits": 0.0}
            counts: dict[str, int] = {}
            for e in bucket:
                counts[e.action_type] = counts.get(e.action_type, 0) + 1
            total = sum(counts.values())
            h = 0.0
            for c in counts.values():
                p = c / total
                if p > 0:
                    h -= p * math.log2(p)
            return {
                "count": total,
                "distinct_action_types": len(counts),
                "entropy_bits": h,
                "histogram": counts,
            }

        return {"window": window, "rich": _stats(rich), "lean": _stats(lean)}

    def unfalsified_hypotheses(
        self, n: int = 5, exclude_bug_corrupted: bool = True
    ) -> list[tuple[int, str, str]]:
        """Return [(trial_id, hypothesis, falsifier)] for the last `n` trustworthy
        trials that carry both a hypothesis and an explicit falsifier.

        The planner consumes this list to grade new candidates against still-open
        claims — i.e. predictions that have not yet been resolved either way.
        Resolution-checking is intentionally minimal here (presence of the
        falsifier string only); semantic matching is deferred to the controller,
        which sees the trial's actual outcome via the journal summary.
        """
        pool = (
            self.trustworthy_entries() if exclude_bug_corrupted else self._entries
        )
        with_falsifier = [
            (e.trial_id, e.hypothesis, e.falsifier)
            for e in pool
            if e.hypothesis and e.falsifier
        ]
        return with_falsifier[-n:]

    def apply_scrub(
        self,
        commit_sha: str,
        reason: str,
        trial_id_min: int | None = None,
        trial_id_max: int | None = None,
        timestamp_min: str | None = None,
        timestamp_max: str | None = None,
    ) -> tuple[int, list[int]]:
        """Tag in-window entries as bug_corrupted_by=<commit_sha>.

        Filter semantics: an entry is tagged when it falls inside EVERY
        bound provided (AND, not OR). Bounds that are None match everything.

        Returns (n_tagged, [trial_ids_tagged]). Mutates self._entries in
        memory. The caller is responsible for re-persisting the JSONL +
        TSV via the scrub CLI (this method intentionally does not write to
        disk so the operator can preview the change first).
        """
        tagged: list[int] = []
        for e in self._entries:
            if trial_id_min is not None and e.trial_id < trial_id_min:
                continue
            if trial_id_max is not None and e.trial_id > trial_id_max:
                continue
            if timestamp_min is not None and (e.timestamp or "") < timestamp_min:
                continue
            if timestamp_max is not None and (e.timestamp or "") > timestamp_max:
                continue
            e.bug_corrupted_by = commit_sha
            e.bug_corrupted_reason = (reason or "")[:200]
            tagged.append(e.trial_id)
        return len(tagged), tagged

    def matching_trial_ids(
        self,
        *,
        trial_id_min: int | None = None,
        trial_id_max: int | None = None,
        timestamp_min: str | None = None,
        timestamp_max: str | None = None,
    ) -> list[int]:
        """Trial IDs matching the scrub filter without mutating entries."""
        matched: list[int] = []
        for e in self._entries:
            if trial_id_min is not None and e.trial_id < trial_id_min:
                continue
            if trial_id_max is not None and e.trial_id > trial_id_max:
                continue
            if timestamp_min is not None and (e.timestamp or "") < timestamp_min:
                continue
            if timestamp_max is not None and (e.timestamp or "") > timestamp_max:
                continue
            matched.append(e.trial_id)
        return matched

    def recent_failures(
        self,
        species: str | None = None,
        n: int = 10,
        exclude_bug_corrupted: bool = True,
    ) -> list[JournalEntry]:
        """Return the last n entries with non-empty failure_analysis.

        Optionally filter by species name.
        """
        entries = (
            self.entries_with_supersessions()
            if exclude_bug_corrupted
            else self._entries
        )
        failed = [
            e for e in entries
            if e.failure_analysis
            and (not exclude_bug_corrupted or not e.bug_corrupted_by)
            and (species is None or e.species == species)
        ]
        return failed[-n:]

    def failure_analysis_for_prompt(
        self, entry: JournalEntry, limit: int | None = None
    ) -> str:
        return failure_analysis_for_prompt(entry, limit=limit)

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
            e for e in self.entries_with_supersessions()
            if e.pareto_status == "frontier" or e.failure_analysis
            if not e.bug_corrupted_by
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
                detail = failure_analysis_for_prompt(e, limit=120)
            species_label = e.species
            lines.append(
                f"  [{tag}] #{e.trial_id} ({species_label}/{hyp})"
                + (f" [{mechanism}]" if mechanism else "")
                + f": {detail}"
            )
        return "\n".join(lines)

    def insights_structured(
        self,
        n: int = 30,
        exclude_bug_corrupted: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Group recent insightful trials by action_type to expose pattern + confidence.

        Each action_type bucket carries:
          - observation: one-line synthesis from the most-recent trial in
            that bucket (operator can scan; planner cites trial IDs as
            evidence).
          - trials_supporting: list[trial_id] in chronological order.
          - successes / failures: counts within the bucket.
          - confidence: 'high' (≥3 supporting, success_rate ≥ 0.66),
            'medium' (≥2 supporting), or 'low' (else).
          - latest_q / latest_s: quality + speed of the most-recent trial,
            for at-a-glance trend visibility.

        Bug-corrupted entries are excluded by default so the planner doesn't
        learn from poisoned signal.
        """
        pool = (
            self.trustworthy_entries() if exclude_bug_corrupted else self._entries
        )
        recent = pool[-n:]
        if not recent:
            return {}

        buckets: dict[str, list[JournalEntry]] = {}
        for e in recent:
            if not (e.pareto_status == "frontier" or e.failure_analysis):
                continue
            buckets.setdefault(e.action_type or "unknown", []).append(e)

        out: dict[str, dict[str, Any]] = {}
        for action_type, entries in buckets.items():
            entries.sort(key=lambda x: x.trial_id)
            successes = sum(1 for e in entries if e.pareto_status == "frontier")
            failures = sum(1 for e in entries if e.failure_analysis)
            n_sup = len(entries)
            success_rate = successes / n_sup if n_sup else 0.0
            if n_sup >= 3 and success_rate >= 0.66:
                confidence = "high"
            elif n_sup >= 2:
                confidence = "medium"
            else:
                confidence = "low"
            latest = entries[-1]
            observation = latest.hypothesis or latest.expected_mechanism or (
                failure_analysis_for_prompt(latest, limit=160)
            ) or "(no description)"
            out[action_type] = {
                "observation": observation[:240],
                "trials_supporting": [e.trial_id for e in entries],
                "successes": successes,
                "failures": failures,
                "confidence": confidence,
                "latest_trial_id": latest.trial_id,
                "latest_q": latest.quality,
                "latest_s": latest.speed,
                "latest_outcome": (
                    "frontier" if latest.pareto_status == "frontier" else "failed"
                ),
            }
        return out

    def action_distribution(self, last_n: int = 30) -> dict[str, int]:
        """Count of each action_type seen in the last n entries (newest-first window).

        Used by the Adjacent-Possible / tail-sampling section to identify
        action types that are over- vs under-represented in recent history.
        """
        counts: dict[str, int] = {}
        for e in self._entries[-last_n:]:
            t = e.action_type or "unknown"
            counts[t] = counts.get(t, 0) + 1
        return counts

    def tail_action_candidates(
        self,
        known_action_types: list[str],
        last_n: int = 30,
        n_sample: int = 3,
    ) -> list[str]:
        """Sample n_sample action types that are under-represented in recent history.

        "Tail" means actions that show up 0–1 times in the last `last_n`
        trials. Used to push the planner toward considering creative
        alternatives outside its recent comfort zone (breaks local-optimum
        traps).
        """
        import random
        dist = self.action_distribution(last_n=last_n)
        tail = [t for t in known_action_types if dist.get(t, 0) <= 1]
        random.shuffle(tail)
        return tail[:n_sample]

    def insights_structured_text(
        self, n: int = 30, exclude_bug_corrupted: bool = True
    ) -> str:
        """Render insights_structured() for inclusion in the controller prompt."""
        d = self.insights_structured(n=n, exclude_bug_corrupted=exclude_bug_corrupted)
        if not d:
            return "(no insights yet — seed more trials to build evidence)"
        # Sort by confidence desc, then n supporting desc
        rank = {"high": 0, "medium": 1, "low": 2}
        items = sorted(
            d.items(),
            key=lambda kv: (rank[kv[1]["confidence"]], -len(kv[1]["trials_supporting"])),
        )
        lines: list[str] = []
        for action_type, info in items:
            lines.append(
                f"{action_type} ({info['confidence']} confidence, "
                f"{info['successes']}+/{info['failures']}- across "
                f"trials {info['trials_supporting']}):"
            )
            lines.append(f"  Observation: {info['observation']}")
            lines.append(
                f"  Latest trial #{info['latest_trial_id']} → "
                f"{info['latest_outcome']} (q={info['latest_q']:.3f} "
                f"sp={info['latest_s']:.1f})"
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
