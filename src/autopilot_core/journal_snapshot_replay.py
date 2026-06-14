"""Read-only diagnostics for journal snapshot replay readiness."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.autopilot_core.baseline_ledger import canonical_jsonable
from src.autopilot_core.journal_reconstruction import (
    reconstruct_archive_from_journal_rows,
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
