"""Tests for append-only AutoPilot journal supersession events."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(AUTOPILOT_DIR))

from experiment_journal import ExperimentJournal, JournalEntry


def _entry(trial_id: int) -> JournalEntry:
    return JournalEntry(
        trial_id=trial_id,
        timestamp=f"2026-06-13T00:00:{trial_id:02d}+00:00",
        species="seeder",
        action_type="seed_batch",
        tier=1,
        quality=1.0,
        speed=10.0,
        cost=0.5,
        reliability=1.0,
        pareto_status="dominated",
    )


def test_supersession_event_round_trips_without_counting_as_trial(tmp_path: Path) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(1))

    event = journal.append_supersession_event(
        target_trial_ids=[1],
        fields={"bug_corrupted_by": "abc123"},
        reason="test event",
        policy_version="supersession-v1",
        actor="unit-test",
    )

    assert event["type"] == "supersession"
    assert journal.count() == 1
    assert journal.next_trial_id() == 2

    reloaded = ExperimentJournal(journal_dir=tmp_path)
    assert reloaded.count() == 1
    assert reloaded.next_trial_id() == 2
    assert reloaded.supersession_events() == [event]


def test_matching_trial_ids_does_not_mutate_entries(tmp_path: Path) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(1))
    journal.record(_entry(2))

    matched = journal.matching_trial_ids(trial_id_min=2)

    assert matched == [2]
    assert [entry.bug_corrupted_by for entry in journal.all_entries()] == ["", ""]
