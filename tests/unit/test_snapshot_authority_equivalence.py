"""B5-M1: the snapshot authority path returns exactly what a full replay returns, cheaply.

A hash-valid segment snapshot is cut from the real replay fixture at several points; the
snapshot (+ tail fold) payload, or the full replay it falls back to, must be canonical-
equal to a full replay of every row. The spy pins the cost: at most one prefix replay per
call, none on a repeat call that shares the journal-scoped verification cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.autopilot.experiment_journal import (
    SEGMENT_SNAPSHOT_POLICY_VERSION,
    _snapshot_hash,
)
from src.autopilot_core import journal_snapshot_replay as jsr
from src.autopilot_core.baseline_ledger import canonical_jsonable
from src.autopilot_core.journal_reconstruction import reconstruct_archive_from_journal_rows

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gate_frontier_replay.json"
POLICIES = ("legacy_4d_v1", "task_rate_4d_v7_physical_cohort_exclusion")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return json.loads(FIXTURE.read_text())["rows"]


def _snapshot_event(rows: list[dict], through: int, policy: str) -> dict:
    prefix = [row for row in rows if int(row["trial_id"]) <= through]
    snapshot = {
        "archive": reconstruct_archive_from_journal_rows(prefix, None, objective_policy=policy),
        "replay_state": jsr.representative_replay_state_from_rows(prefix, objective_policy=policy),
        "source": {"kind": "full_journal_replay", "trial_count": len(prefix)},
    }
    return {
        "type": "journal_snapshot",
        "through_trial_id": through,
        "snapshot": snapshot,
        "policy_version": SEGMENT_SNAPSHOT_POLICY_VERSION,
        "actor": "unit-test",
        "parent_snapshot_hash": "",
        "snapshot_hash": _snapshot_hash(
            through_trial_id=through,
            snapshot=snapshot,
            policy_version=SEGMENT_SNAPSHOT_POLICY_VERSION,
        ),
    }


def _tail_supersession(target: int) -> dict:
    return {
        "type": "supersession",
        "target_trial_ids": [target],
        "fields": {"bug_corrupted_by": "resource_contention"},
        "reason": "synthetic tail contamination",
        "policy_version": "supersession-v1",
        "actor": "unit-test",
        "timestamp": "2026-10-01T00:00:00Z",
    }


# 1501: bug-tagged tail 1502-1505; 1453 / 1398: tails holding representative members;
# 1505: no tail (current).
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("through", [1501, 1453, 1398, 1505])
@pytest.mark.parametrize("with_tail_event", [False, True])
def test_snapshot_payload_equals_full_replay(rows, policy, through, with_tail_event):
    event = _snapshot_event(rows, through, policy)
    ledger = [event] + ([_tail_supersession(through + 2)] if with_tail_event and through < 1505 else [])
    all_rows = rows + ledger[1:]
    full = reconstruct_archive_from_journal_rows(all_rows, None, objective_policy=policy)
    payload = jsr.archive_payload_from_verified_snapshot(all_rows, ledger) or full
    assert canonical_jsonable(payload) == canonical_jsonable(full)


def test_verified_snapshot_replays_the_prefix_at_most_once(rows, monkeypatch):
    calls: list[int] = []
    real = jsr.reconstruct_archive_from_journal_rows

    def spy(replay_rows, *args, **kwargs):
        calls.append(len(list(replay_rows)))
        return real(replay_rows, *args, **kwargs)

    monkeypatch.setattr(jsr, "reconstruct_archive_from_journal_rows", spy)
    policy = POLICIES[0]
    event = _snapshot_event(rows, 1501, policy)
    cache: dict = {}
    first = jsr.archive_payload_from_verified_snapshot(
        rows, [event], prefix_verification_cache=cache
    )
    assert first is not None
    prefix_replays = [n for n in calls if n > 100]
    assert len(prefix_replays) == 1, calls
    calls.clear()
    again = jsr.archive_payload_from_verified_snapshot(
        rows, [event], prefix_verification_cache=cache
    )
    assert canonical_jsonable(again) == canonical_jsonable(first)
    assert [n for n in calls if n > 100] == [], calls


def test_pre_p2_snapshot_falls_back_until_current_format_rebuild(rows):
    """A pre-P2 snapshot stays immutable and uses full replay until it is recut.

    The real fixture's pre-P2 archive is pinned by its historical golden digest. The
    current-format snapshot produced at the same boundary must then verify and equal
    full replay exactly.
    """
    policy = POLICIES[0]
    through = max(int(row["trial_id"]) for row in rows)
    current = reconstruct_archive_from_journal_rows(
        rows, None, objective_policy=policy
    )
    assert current is not None

    # Recreate the old archive payload from the pinned pre-P2 result. The only archive
    # field added by B5-12 is an additive exclusion counter; this fixture's
    # representative clusters are unchanged by B5-M2.
    pre_p2_archive = json.loads(json.dumps(current))
    pre_p2_archive["exclusions"].pop("unmeasured_objectives")
    digest = hashlib.sha256(
        json.dumps(pre_p2_archive, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert digest == "3d04f5798fa2ae91bb2ae96e6c3504f6c55fd1bbf9fc33aee55f5ecd6fc9f29e"

    def snapshot_event(archive: dict, policy_version: str) -> dict:
        snapshot = {
            "archive": archive,
            "replay_state": jsr.representative_replay_state_from_rows(
                rows, objective_policy=policy
            ),
            "source": {"kind": "full_journal_replay", "trial_count": len(rows)},
        }
        event = {
            "type": "journal_snapshot",
            "through_trial_id": through,
            "snapshot": snapshot,
            "policy_version": policy_version,
            "actor": "migration-test",
            "parent_snapshot_hash": "",
        }
        event["snapshot_hash"] = _snapshot_hash(
            through_trial_id=through,
            snapshot=snapshot,
            policy_version=policy_version,
        )
        return event

    pre_p2_event = snapshot_event(pre_p2_archive, "journal-segment-snapshot-v1")
    pre_p2_cache: dict = {}
    diagnostic = jsr.build_snapshot_replay_diagnostic(
        rows, [pre_p2_event], prefix_verification_cache=pre_p2_cache
    )
    assert diagnostic.status == "archive_prefix_drift"
    assert diagnostic.bounded_replay_readiness == "prefix_invalidated"
    assert diagnostic.hash_status == "match"

    bounded = jsr.archive_payload_from_verified_snapshot(
        rows, [pre_p2_event], prefix_verification_cache=pre_p2_cache
    )
    assert bounded is None
    # This is the consumer fallback when bounded authority declines the old snapshot.
    fallback = reconstruct_archive_from_journal_rows(rows, None, objective_policy=policy)
    assert canonical_jsonable(current) == canonical_jsonable(fallback)

    rebuilt_event = snapshot_event(current, SEGMENT_SNAPSHOT_POLICY_VERSION)
    rebuilt_cache: dict = {}
    rebuilt = jsr.build_snapshot_replay_diagnostic(
        rows, [rebuilt_event], prefix_verification_cache=rebuilt_cache
    )
    rebuilt_payload = jsr.archive_payload_from_verified_snapshot(
        rows, [rebuilt_event], prefix_verification_cache=rebuilt_cache
    )
    assert rebuilt.status == "archive_prefix_match"
    assert rebuilt.bounded_replay_readiness == "current"
    assert canonical_jsonable(rebuilt_payload) == canonical_jsonable(current)
