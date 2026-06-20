"""Read-only journal snapshot replay diagnostic tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.autopilot_core.journal_reconstruction import reconstruct_archive_from_journal_rows
from src.autopilot_core.journal_snapshot_replay import (
    JOURNAL_SNAPSHOT_EVENT_TYPE,
    archive_payload_from_current_snapshot,
    build_snapshot_replay_diagnostic,
    format_snapshot_replay_summary,
)


def _row(trial_id: int, *, quality: float = 1.0, speed: float = 40.0) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "timestamp": f"2026-06-14T00:00:0{trial_id}Z",
        "species": "unit",
        "action_type": "seed_batch",
        "tier": 1,
        "quality": quality,
        "speed": speed,
        "cost": 0.2,
        "reliability": 0.9,
        "pareto_status": "frontier",
    }


def _snapshot_event(
    *,
    through_trial_id: int,
    snapshot: dict[str, Any],
    policy_version: str = "unit-policy-v1",
    parent_snapshot_hash: str = "",
    snapshot_hash: str | None = None,
) -> dict[str, Any]:
    event = {
        "type": JOURNAL_SNAPSHOT_EVENT_TYPE,
        "through_trial_id": through_trial_id,
        "snapshot": snapshot,
        "policy_version": policy_version,
        "actor": "unit-test",
        "parent_snapshot_hash": parent_snapshot_hash,
        "timestamp": "2026-06-14T00:00:03Z",
    }
    if snapshot_hash is None:
        payload = {
            "through_trial_id": through_trial_id,
            "snapshot": snapshot,
            "policy_version": policy_version,
            "parent_snapshot_hash": parent_snapshot_hash,
        }
        encoded = json.dumps(
            payload, sort_keys=True, default=str, separators=(",", ":")
        )
        snapshot_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    event["snapshot_hash"] = snapshot_hash
    return event


def _archive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    archive = reconstruct_archive_from_journal_rows(rows, None)
    assert archive is not None
    return archive


def test_snapshot_replay_reports_no_events() -> None:
    diagnostic = build_snapshot_replay_diagnostic([_row(1)], [])

    assert diagnostic.status == "no_events"
    assert diagnostic.bounded_replay_readiness == "not_ready"
    assert format_snapshot_replay_summary(diagnostic) == [
        "Journal snapshot events: 0",
        "Journal snapshot replay: no snapshot events",
    ]


def test_snapshot_replay_verifies_hash_without_archive_payload() -> None:
    rows = [_row(1), _row(2)]
    event = _snapshot_event(
        through_trial_id=1,
        snapshot={"metadata": {"note": "substrate only"}},
    )

    diagnostic = build_snapshot_replay_diagnostic(rows + [event], [event])

    assert diagnostic.status == "hash_verified"
    assert diagnostic.bounded_replay_readiness == "not_ready"
    assert diagnostic.hash_status == "match"
    assert diagnostic.tail_trial_count == 1
    assert diagnostic.tail_max_trial_id == 2
    assert diagnostic.warnings == ["latest snapshot payload has no archive view to verify"]


def test_snapshot_replay_reports_current_archive_prefix() -> None:
    rows = [_row(1, quality=1.2)]
    event = _snapshot_event(
        through_trial_id=1,
        snapshot={"archive": _archive(rows)},
    )

    diagnostic = build_snapshot_replay_diagnostic(rows + [event], [event])

    assert diagnostic.status == "archive_prefix_match"
    assert diagnostic.bounded_replay_readiness == "current"
    assert diagnostic.hash_status == "match"
    assert diagnostic.tail_trial_count == 0
    assert diagnostic.warnings == []
    assert (
        "Journal snapshot bounded replay readiness: current"
        in format_snapshot_replay_summary(diagnostic)
    )


def test_current_snapshot_payload_helper_returns_verified_archive() -> None:
    rows = [_row(1, quality=1.2)]
    archive = _archive(rows)
    event = _snapshot_event(
        through_trial_id=1,
        snapshot={"archive": archive},
    )

    payload = archive_payload_from_current_snapshot(rows + [event], [event])

    assert payload == archive
    assert payload is not archive


def test_snapshot_replay_flags_unverified_tail_after_matching_prefix() -> None:
    rows = [_row(1, quality=1.2), _row(2, quality=1.1)]
    event = _snapshot_event(
        through_trial_id=1,
        snapshot={"archive": _archive([rows[0]])},
    )

    diagnostic = build_snapshot_replay_diagnostic(rows + [event], [event])

    assert diagnostic.status == "archive_prefix_match"
    assert diagnostic.bounded_replay_readiness == "tail_unverified"
    assert diagnostic.hash_status == "match"
    assert diagnostic.event_count == 1
    assert diagnostic.tail_trial_count == 1
    assert any("post-snapshot trials" in item for item in diagnostic.warnings)
    assert archive_payload_from_current_snapshot(rows + [event], [event]) is None


def test_snapshot_replay_requires_matching_hash_for_ready_status() -> None:
    rows = [_row(1)]
    event = _snapshot_event(
        through_trial_id=1,
        snapshot={"archive": _archive(rows)},
        snapshot_hash="not-the-real-hash",
    )

    diagnostic = build_snapshot_replay_diagnostic(rows + [event], [event])

    assert diagnostic.status == "archive_prefix_hash_unverified"
    assert diagnostic.bounded_replay_readiness == "not_ready"
    assert diagnostic.hash_status == "mismatch"
    assert "latest snapshot_hash does not match event payload" in diagnostic.warnings


def test_snapshot_replay_flags_post_snapshot_supersession_prefix_drift() -> None:
    rows = [_row(1, quality=2.0), _row(2, quality=1.0)]
    event = _snapshot_event(
        through_trial_id=2,
        snapshot={"archive": _archive(rows)},
    )
    supersession = {
        "type": "supersession",
        "target_trial_ids": [1],
        "fields": {
            "bug_corrupted_by": "resource_contention",
            "bug_corrupted_reason": "synthetic contamination window",
        },
        "reason": "synthetic contamination window",
        "policy_version": "supersession-v1",
        "actor": "unit-test",
        "timestamp": "2026-06-14T00:00:04Z",
    }

    diagnostic = build_snapshot_replay_diagnostic(
        rows + [event, supersession],
        [event, supersession],
    )

    assert diagnostic.status == "archive_prefix_drift"
    assert diagnostic.bounded_replay_readiness == "prefix_invalidated"
    assert diagnostic.post_snapshot_prefix_event_count == 1
    assert any("post-snapshot supersession" in item for item in diagnostic.warnings)
    assert any("archive payload differs" in item for item in diagnostic.warnings)


def test_snapshot_replay_latest_invalid_event_blocks_older_ready_snapshot() -> None:
    rows = [_row(1, quality=1.2)]
    valid_event = _snapshot_event(
        through_trial_id=1,
        snapshot={"archive": _archive(rows)},
    )
    invalid_latest = {
        "type": JOURNAL_SNAPSHOT_EVENT_TYPE,
        "snapshot": {"archive": _archive(rows)},
        "policy_version": "unit-policy-v2",
        "actor": "unit-test",
        "timestamp": "2026-06-14T00:00:04Z",
    }

    diagnostic = build_snapshot_replay_diagnostic(
        rows + [valid_event, invalid_latest],
        [valid_event, invalid_latest],
    )

    assert diagnostic.status == "invalid_latest_event"
    assert diagnostic.bounded_replay_readiness == "not_ready"
    assert diagnostic.event_count == 2
    assert diagnostic.warnings == [
        "latest snapshot event has no usable through_trial_id"
    ]
