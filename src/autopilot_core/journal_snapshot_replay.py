"""Read-only diagnostics for journal snapshot replay readiness."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.autopilot_core.baseline_ledger import canonical_jsonable
from src.autopilot_core.journal_reconstruction import (
    objectives_from_journal_row,
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.learning_exclusions import WITHIN_NOISE_EXCLUSIONS
from src.autopilot_core.pareto_math import dominates, hypervolume
from src.autopilot_core.tier_specs import (
    DEFAULT_FRONTIER_TIER,
    LEGACY_OBJECTIVE_POLICY,
    TASK_RATE_OBJECTIVE_POLICY,
    TASK_RATE_REFERENCE_POINT,
    spec_for,
)


JOURNAL_SNAPSHOT_EVENT_TYPE = "journal_snapshot"


@dataclass(frozen=True)
class JournalSnapshotReplayDiagnostic:
    """Structured snapshot-readiness result for operator commands."""

    status: str
    bounded_replay_readiness: str = "not_ready"
    event_count: int = 0
    hash_status: str = "not_applicable"
    latest_event: dict[str, Any] | None = None
    through_trial_id: int | None = None
    policy_version: str = ""
    snapshot_hash: str = ""
    parent_snapshot_hash: str = ""
    tail_trial_count: int = 0
    tail_max_trial_id: int | None = None
    journal_max_trial_id: int | None = None
    post_snapshot_prefix_event_count: int = 0
    warnings: list[str] = field(default_factory=list)


def _trial_id_from_row(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("trial_id"))
    except (TypeError, ValueError):
        return None


def _snapshot_event_indices(events: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, event)
        for index, event in enumerate(events)
        if event.get("type") == JOURNAL_SNAPSHOT_EVENT_TYPE
        and "trial_id" not in event
    ]


def _event_targets_prefix(event: dict[str, Any], through_trial_id: int) -> bool:
    targets = event.get("target_trial_ids")
    if not isinstance(targets, list):
        return False
    for target in targets:
        try:
            if int(target) <= through_trial_id:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _expected_snapshot_hash(event: dict[str, Any]) -> str | None:
    try:
        through_trial_id = int(event["through_trial_id"])
    except (KeyError, TypeError, ValueError):
        return None
    snapshot = event.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    payload = {
        "through_trial_id": through_trial_id,
        "snapshot": snapshot,
        "policy_version": event.get("policy_version", ""),
        "parent_snapshot_hash": event.get("parent_snapshot_hash", ""),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _snapshot_archive_payload(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    archive = snapshot.get("archive")
    if isinstance(archive, dict):
        return archive
    archive_keys = {
        "frontier",
        "frontiers_by_tier",
        "all_entries",
        "hypervolume_history",
        "hv_history_by_tier",
    }
    if any(key in snapshot for key in archive_keys):
        return snapshot
    return None


def archive_payload_from_current_snapshot(
    rows: Iterable[dict[str, Any]],
    ledger_events: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the latest snapshot archive only when it is current and verified.

    This is the first bounded-replay consumption path: it avoids full journal
    replay only when there is no tail to fold and the existing diagnostic proves
    the snapshot archive matches current-policy prefix replay. Tailed snapshots
    still return ``None`` so callers keep using full replay until a tail-fold
    consumer exists.
    """
    diagnostic = build_snapshot_replay_diagnostic(rows, ledger_events)
    if diagnostic.bounded_replay_readiness != "current":
        return None
    event = diagnostic.latest_event or {}
    snapshot = event.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    archive = _snapshot_archive_payload(snapshot)
    return copy.deepcopy(archive) if isinstance(archive, dict) else None


def archive_payload_from_verified_snapshot(
    rows: Iterable[dict[str, Any]],
    ledger_events: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a verified snapshot archive, folding a safe post-snapshot tail.

    The tail path is deliberately conservative. It only folds ordinary trial
    rows whose objective contribution is self-contained. Rows that participate
    in within-noise representative clustering (for example ``seq_accumulating``)
    need raw prefix samples that a compact snapshot does not retain, so this
    helper returns ``None`` and leaves callers on full replay for those tails.
    """
    rows_list = list(rows)
    ledger_events_list = [
        event
        for event in ledger_events
        if event.get("type") and "trial_id" not in event
    ]
    current_payload = archive_payload_from_current_snapshot(
        rows_list,
        ledger_events_list,
    )
    if current_payload is not None:
        return current_payload

    diagnostic = build_snapshot_replay_diagnostic(rows_list, ledger_events_list)
    if diagnostic.bounded_replay_readiness != "tail_unverified":
        return None
    event = diagnostic.latest_event or {}
    snapshot = event.get("snapshot")
    if not isinstance(snapshot, dict) or diagnostic.through_trial_id is None:
        return None
    snapshot_archive = _snapshot_archive_payload(snapshot)
    if not isinstance(snapshot_archive, dict):
        return None

    snapshot_indices = _snapshot_event_indices(ledger_events_list)
    if not snapshot_indices:
        return None
    latest_snapshot_index, _latest_event = snapshot_indices[-1]
    post_snapshot_events = ledger_events_list[latest_snapshot_index + 1:]
    tail_events = [
        event
        for event in post_snapshot_events
        if event.get("type") == "supersession"
        and not _event_targets_prefix(event, diagnostic.through_trial_id)
    ]
    tail_rows = [
        row
        for row in rows_list
        if (
            (trial_id := _trial_id_from_row(row)) is not None
            and trial_id > diagnostic.through_trial_id
        )
    ]
    if not tail_rows:
        return None
    if any(_row_requires_prefix_raw_samples(row) for row in tail_rows):
        return None

    policy = str(
        snapshot_archive.get("objective_policy") or LEGACY_OBJECTIVE_POLICY
    )
    if policy not in {LEGACY_OBJECTIVE_POLICY, TASK_RATE_OBJECTIVE_POLICY}:
        return None
    for row in tail_rows:
        if objectives_from_journal_row(row, objective_policy=policy) is None:
            return None

    tail_archive = reconstruct_archive_from_journal_rows(
        tail_rows + tail_events,
        None,
        current_run_only=False,
        objective_policy=policy,
    )
    if tail_archive is None:
        return None
    return _fold_tail_archive_into_snapshot(snapshot_archive, tail_archive)


def _row_requires_prefix_raw_samples(row: dict[str, Any]) -> bool:
    bug = row.get("bug_corrupted_by") or ""
    eval_details = row.get("eval_details") or {}
    learning_exclusion = {}
    if isinstance(eval_details, dict):
        learning_exclusion = eval_details.get("learning_exclusion") or {}
    excluded_by = ""
    if isinstance(learning_exclusion, dict):
        excluded_by = str(learning_exclusion.get("by") or "")
    return bug == "mad_noise" or excluded_by in WITHIN_NOISE_EXCLUSIONS


def _reference_point_for_archive_policy(policy: str) -> tuple[float, ...]:
    if policy == TASK_RATE_OBJECTIVE_POLICY:
        return TASK_RATE_REFERENCE_POINT
    return spec_for(DEFAULT_FRONTIER_TIER).reference_point


def _max_trial_id(*values: Any) -> int | None:
    found: list[int] = []
    for value in values:
        try:
            found.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(found) if found else None


def _merge_exclusion_slot(
    prefix: dict[str, Any],
    tail: dict[str, Any],
) -> dict[str, Any]:
    return {
        "count": int(prefix.get("count") or 0) + int(tail.get("count") or 0),
        "max_trial_id": _max_trial_id(
            prefix.get("max_trial_id"),
            tail.get("max_trial_id"),
        ),
    }


def _merge_supersession_meta(
    prefix: dict[str, Any],
    tail: dict[str, Any],
) -> dict[str, Any]:
    target_ids = {
        int(value)
        for value in (prefix.get("target_trial_ids") or [])
        + (tail.get("target_trial_ids") or [])
        if isinstance(value, int) or str(value).isdigit()
    }
    return {
        "events_applied": int(prefix.get("events_applied") or 0)
        + int(tail.get("events_applied") or 0),
        "target_trial_ids": sorted(target_ids),
        "field_names": sorted({
            str(value)
            for value in (prefix.get("field_names") or [])
            + (tail.get("field_names") or [])
        }),
    }


def _fold_tail_archive_into_snapshot(
    snapshot_archive: dict[str, Any],
    tail_archive: dict[str, Any],
) -> dict[str, Any] | None:
    policy = str(snapshot_archive.get("objective_policy") or LEGACY_OBJECTIVE_POLICY)
    ref = _reference_point_for_archive_policy(policy)
    merged = copy.deepcopy(snapshot_archive)
    tail_entries = copy.deepcopy(tail_archive.get("all_entries") or [])
    if not all(isinstance(entry, dict) for entry in tail_entries):
        return None

    frontiers_by_tier = {
        str(tier): copy.deepcopy(frontier)
        for tier, frontier in (merged.get("frontiers_by_tier") or {}).items()
        if isinstance(frontier, list)
    }
    hv_history_by_tier = {
        str(tier): copy.deepcopy(history)
        for tier, history in (merged.get("hv_history_by_tier") or {}).items()
        if isinstance(history, list)
    }

    all_entries = copy.deepcopy(merged.get("all_entries") or [])
    all_entries.extend(tail_entries)
    for entry in tail_entries:
        try:
            tier = int(entry.get("eval_tier", DEFAULT_FRONTIER_TIER))
            trial_id = int(entry.get("trial_id"))
        except (TypeError, ValueError):
            return None
        objectives = entry.get("objectives")
        if not isinstance(objectives, list):
            return None
        key = str(tier)
        frontier = frontiers_by_tier.setdefault(key, [])
        if not any(dominates(front["objectives"], objectives) for front in frontier):
            frontiers_by_tier[key] = [
                front
                for front in frontier
                if not dominates(objectives, front["objectives"])
            ]
            frontiers_by_tier[key].append(entry)
        history = hv_history_by_tier.setdefault(key, [])
        hv = round(
            hypervolume(
                [front["objectives"] for front in frontiers_by_tier[key]],
                ref=ref,
            ),
            4,
        )
        if history:
            hv = max(history[-1][1], hv)
        history.append([trial_id, hv])

    canonical_key = str(merged.get("canonical_tier") or DEFAULT_FRONTIER_TIER)
    merged["frontiers_by_tier"] = {
        key: frontiers_by_tier[key] for key in sorted(frontiers_by_tier, key=int)
    }
    merged["hv_history_by_tier"] = {
        key: hv_history_by_tier[key] for key in sorted(hv_history_by_tier, key=int)
    }
    merged["frontier"] = copy.deepcopy(frontiers_by_tier.get(canonical_key, []))
    merged["hypervolume_history"] = copy.deepcopy(
        hv_history_by_tier.get(canonical_key, [])
    )
    merged["all_entries"] = all_entries
    merged["t0_audit"] = copy.deepcopy(merged.get("t0_audit") or [])
    merged["t0_audit"].extend(copy.deepcopy(tail_archive.get("t0_audit") or []))
    merged["journal_max_trial_id"] = _max_trial_id(
        merged.get("journal_max_trial_id"),
        tail_archive.get("journal_max_trial_id"),
    )

    prefix_exclusions = merged.get("exclusions") or {}
    tail_exclusions = tail_archive.get("exclusions") or {}
    merged["exclusions"] = {
        "bug_corrupted": _merge_exclusion_slot(
            prefix_exclusions.get("bug_corrupted") or {},
            tail_exclusions.get("bug_corrupted") or {},
        ),
        "truncated_above_cap": _merge_exclusion_slot(
            prefix_exclusions.get("truncated_above_cap") or {},
            tail_exclusions.get("truncated_above_cap") or {},
        ),
        "max_trial_id_cap": tail_exclusions.get(
            "max_trial_id_cap",
            prefix_exclusions.get("max_trial_id_cap"),
        ),
    }
    merged["supersessions"] = _merge_supersession_meta(
        merged.get("supersessions") or {},
        tail_archive.get("supersessions") or {},
    )
    return merged


def _journal_trial_stats(
    rows: list[dict[str, Any]], through_trial_id: int,
) -> tuple[int, int | None, int | None]:
    tail_trial_count = 0
    tail_max_trial_id: int | None = None
    journal_max_trial_id: int | None = None
    for row in rows:
        trial_id = _trial_id_from_row(row)
        if trial_id is None:
            continue
        if journal_max_trial_id is None or trial_id > journal_max_trial_id:
            journal_max_trial_id = trial_id
        if trial_id > through_trial_id:
            tail_trial_count += 1
            if tail_max_trial_id is None or trial_id > tail_max_trial_id:
                tail_max_trial_id = trial_id
    return tail_trial_count, tail_max_trial_id, journal_max_trial_id


def _rows_for_prefix_replay(
    rows: list[dict[str, Any]], through_trial_id: int,
) -> list[dict[str, Any]]:
    prefix_rows: list[dict[str, Any]] = []
    for row in rows:
        trial_id = _trial_id_from_row(row)
        if trial_id is None:
            prefix_rows.append(row)
        elif trial_id <= through_trial_id:
            prefix_rows.append(row)
    return prefix_rows


def build_snapshot_replay_diagnostic(
    rows: Iterable[dict[str, Any]],
    ledger_events: Iterable[dict[str, Any]],
) -> JournalSnapshotReplayDiagnostic:
    """Check whether the latest snapshot can act as a bounded replay prefix.

    This is diagnostics-only. It does not use a snapshot as replay authority and
    does not mutate archive, baseline, journal, or planner state.
    """
    all_ledger_events = [
        event
        for event in ledger_events
        if event.get("type") and "trial_id" not in event
    ]
    snapshot_indices = _snapshot_event_indices(all_ledger_events)
    if not snapshot_indices:
        return JournalSnapshotReplayDiagnostic(status="no_events")

    latest_snapshot_index, latest_event_raw = snapshot_indices[-1]
    latest_event = copy.deepcopy(latest_event_raw)
    try:
        through_trial_id = int(latest_event["through_trial_id"])
    except (KeyError, TypeError, ValueError):
        return JournalSnapshotReplayDiagnostic(
            status="invalid_latest_event",
            event_count=len(snapshot_indices),
            latest_event=latest_event,
            warnings=["latest snapshot event has no usable through_trial_id"],
        )
    snapshot = latest_event.get("snapshot")
    if not isinstance(snapshot, dict):
        return JournalSnapshotReplayDiagnostic(
            status="invalid_latest_event",
            event_count=len(snapshot_indices),
            latest_event=latest_event,
            through_trial_id=through_trial_id,
            warnings=["latest snapshot event has no usable snapshot payload"],
        )

    rows_list = list(rows)
    tail_count, tail_max, journal_max = _journal_trial_stats(rows_list, through_trial_id)
    warnings: list[str] = []
    if journal_max is not None and through_trial_id > journal_max:
        warnings.append(
            f"snapshot through_trial_id {through_trial_id} exceeds journal max "
            f"trial id {journal_max}"
        )
    post_snapshot_prefix_events = [
        event
        for event in all_ledger_events[latest_snapshot_index + 1:]
        if event.get("type") == "supersession"
        and _event_targets_prefix(event, through_trial_id)
    ]
    if post_snapshot_prefix_events:
        warnings.append(
            "post-snapshot supersession targets the snapshot prefix; bounded replay "
            "must fold tail ledger events or rebuild from journal"
        )

    expected_hash = _expected_snapshot_hash(latest_event)
    recorded_hash = str(latest_event.get("snapshot_hash") or "")
    if not recorded_hash:
        hash_status = "missing"
        warnings.append("latest snapshot event has no snapshot_hash")
    elif expected_hash is None:
        hash_status = "uncheckable"
        warnings.append("latest snapshot event hash cannot be recomputed")
    elif recorded_hash == expected_hash:
        hash_status = "match"
    else:
        hash_status = "mismatch"
        warnings.append("latest snapshot_hash does not match event payload")

    status = "hash_verified" if hash_status == "match" else "hash_unverified"
    snapshot_archive = _snapshot_archive_payload(snapshot)
    if snapshot_archive is None:
        warnings.append("latest snapshot payload has no archive view to verify")
    else:
        prefix_rows = _rows_for_prefix_replay(rows_list, through_trial_id)
        prefix_archive = reconstruct_archive_from_journal_rows(
            prefix_rows,
            None,
            current_run_only=False,
        )
        if prefix_archive is None:
            status = "archive_prefix_empty"
            warnings.append("journal prefix replay produced no archive payload")
        elif canonical_jsonable(snapshot_archive) == canonical_jsonable(prefix_archive):
            status = (
                "archive_prefix_match"
                if hash_status == "match"
                else "archive_prefix_hash_unverified"
            )
        else:
            status = "archive_prefix_drift"
            warnings.append(
                "latest snapshot archive payload differs from current policy prefix replay"
            )

    prefix_invalidated = (
        bool(post_snapshot_prefix_events) or status == "archive_prefix_drift"
    )
    if prefix_invalidated:
        bounded_replay_readiness = "prefix_invalidated"
    elif status == "archive_prefix_match" and hash_status == "match":
        bounded_replay_readiness = (
            "current" if tail_count == 0 else "tail_unverified"
        )
        if tail_count:
            warnings.append(
                "journal has post-snapshot trials; bounded replay must verify tail "
                "folding before using the snapshot as current"
            )
    else:
        bounded_replay_readiness = "not_ready"

    return JournalSnapshotReplayDiagnostic(
        status=status,
        bounded_replay_readiness=bounded_replay_readiness,
        event_count=len(snapshot_indices),
        hash_status=hash_status,
        latest_event=latest_event,
        through_trial_id=through_trial_id,
        policy_version=str(latest_event.get("policy_version") or ""),
        snapshot_hash=recorded_hash,
        parent_snapshot_hash=str(latest_event.get("parent_snapshot_hash") or ""),
        tail_trial_count=tail_count,
        tail_max_trial_id=tail_max,
        journal_max_trial_id=journal_max,
        post_snapshot_prefix_event_count=len(post_snapshot_prefix_events),
        warnings=warnings,
    )


def _short_hash(value: str) -> str:
    return value[:12] if value else "n/a"


def format_snapshot_replay_summary(
    diagnostic: JournalSnapshotReplayDiagnostic,
) -> list[str]:
    """Human-readable snapshot replay diagnostics for status/report output."""
    lines = [f"Journal snapshot events: {diagnostic.event_count}"]
    if diagnostic.status == "no_events":
        lines.append("Journal snapshot replay: no snapshot events")
        return lines

    through = diagnostic.through_trial_id
    tail_max = diagnostic.tail_max_trial_id
    lines.append(
        "Latest journal snapshot: "
        f"trial #{through if through is not None else 'n/a'} "
        f"policy={diagnostic.policy_version or 'n/a'} "
        f"hash={_short_hash(diagnostic.snapshot_hash)} "
        f"parent={_short_hash(diagnostic.parent_snapshot_hash)} "
        f"tail_trials={diagnostic.tail_trial_count} "
        f"tail_max={tail_max if tail_max is not None else 'n/a'}"
    )
    lines.append(f"Journal snapshot hash status: {diagnostic.hash_status}")
    lines.append(f"Journal snapshot replay status: {diagnostic.status}")
    lines.append(
        "Journal snapshot bounded replay readiness: "
        f"{diagnostic.bounded_replay_readiness}"
    )
    for warning in diagnostic.warnings:
        lines.append(f"Journal snapshot warning: {warning}")
    return lines
