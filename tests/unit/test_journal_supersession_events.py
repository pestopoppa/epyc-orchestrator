"""Tests for append-only AutoPilot journal supersession events."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(AUTOPILOT_DIR))

from experiment_journal import ExperimentJournal, JournalEntry
from src.autopilot_core.journal_reconstruction import reconstruct_archive_from_journal_rows


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


def test_reconstruction_folds_supersession_events_without_mutating_trials(
    tmp_path: Path,
) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(1))
    dominant = _entry(2)
    dominant.quality = 2.0
    dominant.speed = 99.0
    journal.record(dominant)
    journal.append_supersession_event(
        target_trial_ids=[2],
        fields={
            "bug_corrupted_by": "resource_contention",
            "bug_corrupted_reason": "synthetic contention window",
        },
        reason="synthetic contention window",
        policy_version="supersession-v1",
        actor="unit-test",
    )
    rows = [asdict(entry) for entry in journal.all_entries()]
    rows.extend(journal.supersession_events())

    archive = reconstruct_archive_from_journal_rows(rows, None, current_run_only=False)

    assert archive is not None
    assert [entry["trial_id"] for entry in archive["frontier"]] == [1]
    assert archive["exclusions"]["bug_corrupted"] == {"count": 1, "max_trial_id": 2}
    assert archive["supersessions"] == {
        "events_applied": 1,
        "target_trial_ids": [2],
        "field_names": ["bug_corrupted_by", "bug_corrupted_reason"],
    }
    assert journal.all_entries()[1].bug_corrupted_by == ""


def test_runtime_prompt_views_fold_supersession_events_without_mutating_trials(
    tmp_path: Path,
) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    trusted = _entry(1)
    trusted.pareto_status = "frontier"
    trusted.hypothesis = "trusted signal"
    journal.record(trusted)
    contaminated = _entry(2)
    contaminated.pareto_status = "frontier"
    contaminated.quality = 2.0
    contaminated.hypothesis = "contaminated signal"
    journal.record(contaminated)
    journal.append_supersession_event(
        target_trial_ids=[2],
        fields={
            "bug_corrupted_by": "resource_contention",
            "bug_corrupted_reason": "synthetic contention window",
        },
        reason="synthetic contention window",
        policy_version="supersession-v1",
        actor="unit-test",
    )

    folded = journal.entries_with_supersessions()

    assert journal.all_entries()[1].bug_corrupted_by == ""
    assert folded[1].bug_corrupted_by == "resource_contention"
    assert [entry.trial_id for entry in journal.trustworthy_entries()] == [1]
    assert journal.trustworthiness_score()["corrupted_by"] == {
        "resource_contention": 1
    }
    summary = journal.summary_text()
    assert "#2 [seeder/seed_batch] CORRUPTED_BY=resource_contention" in summary
    assert "q=2.000" not in summary
    insights = journal.insights_text()
    assert "#1" in insights
    assert "#2" not in insights
