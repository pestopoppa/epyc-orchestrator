"""Phase 6b — In-flight trial marker + crash recovery tests.

Per handoffs/active/autopilot-exogenous-restart-resilience.md Phase 6b
verification gate. Covers:

  1. No marker present → no-op (trial_counter unchanged, state untouched
     besides the marker field itself).
  2. Case (a): in_flight_trial.trial_id == journal_max
     → trial DID land in the journal before the crash. The recovery
     code must:
       - bump trial_counter to max(current, journal_max + 1)
       - clear the marker
       - NOT write a placeholder
  3. Case (b): in_flight_trial.trial_id > journal_max
     → trial died before journal.record. The recovery code must:
       - write an AUTOPILOT_KILLED placeholder JournalEntry tagged
         bug_corrupted_by=autopilot_killed_mid_trial
       - bump trial_counter to prior_tid + 1
       - clear the marker
  4. `_maybe_reimport_pareto_from_journal` helper:
       - skips bug_corrupted entries
       - skips entries already in the archive (no double-count)
       - re-imports a valid entry when missing
       - skips silently when no JournalEntry matches the trial_id
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import autopilot  # noqa: E402
from experiment_journal import ExperimentJournal, JournalEntry  # noqa: E402
from pareto_archive import ParetoArchive, ParetoEntry  # noqa: E402


# ───────── fixtures ──────────


@pytest.fixture
def journal(tmp_path: Path) -> ExperimentJournal:
    return ExperimentJournal(journal_dir=tmp_path / "journal")


@pytest.fixture
def archive(tmp_path: Path) -> ParetoArchive:
    return ParetoArchive(state_path=tmp_path / "archive.json")


def _make_entry(
    trial_id: int,
    *,
    quality: float = 0.7,
    speed: float = 40.0,
    cost: float = 0.3,
    reliability: float = 0.9,
    bug_corrupted_by: str = "",
    species: str = "test_species",
) -> JournalEntry:
    return JournalEntry(
        trial_id=trial_id,
        timestamp="2026-05-24T00:00:00Z",
        species=species,
        action_type="seed_batch",
        tier=2,
        quality=quality,
        speed=speed,
        cost=cost,
        reliability=reliability,
        pareto_status="frontier",
        bug_corrupted_by=bug_corrupted_by,
    )


# ───────── _recover_from_in_flight_trial ──────────


def test_recovery_no_marker_is_noop(
    journal: ExperimentJournal, archive: ParetoArchive
) -> None:
    state = {"in_flight_trial": None, "trial_counter": 5}
    new_counter = autopilot._recover_from_in_flight_trial(
        state, journal, archive, trial_counter=5,
    )
    assert new_counter == 5
    assert state["in_flight_trial"] is None
    assert state["trial_counter"] == 5
    # No placeholder created
    assert journal.next_trial_id() == 0


def test_recovery_case_a_journal_advanced_bumps_counter(
    journal: ExperimentJournal, archive: ParetoArchive
) -> None:
    """journal_max >= prior_tid → trial recorded; bump counter, no placeholder."""
    # Set up: journal has trial 7 already recorded
    journal.record(_make_entry(trial_id=7, quality=0.8))
    state = {
        "in_flight_trial": {
            "trial_id": 7,
            "action": {"type": "seed_batch"},
            "host_pid": 999,
            "host_started_at": 1000.0,
        },
        "trial_counter": 7,
    }
    new_counter = autopilot._recover_from_in_flight_trial(
        state, journal, archive, trial_counter=7,
    )
    # Counter bumped past the journaled trial
    assert new_counter == 8
    assert state["trial_counter"] == 8
    # Marker cleared
    assert state["in_flight_trial"] is None
    # No placeholder added (still just trial 7)
    assert journal.next_trial_id() == 8
    entries = journal.all_entries()
    assert len(entries) == 1
    assert entries[0].trial_id == 7
    assert entries[0].bug_corrupted_by == ""  # original entry untouched


def test_recovery_case_b_no_journal_writes_placeholder(
    journal: ExperimentJournal, archive: ParetoArchive
) -> None:
    """journal_max < prior_tid → died before record; write placeholder."""
    # Journal has trial 5 but in_flight claims trial 6 was running
    journal.record(_make_entry(trial_id=5))
    state = {
        "in_flight_trial": {
            "trial_id": 6,
            "action": {"type": "prompt_mutation"},
            "host_pid": 111,
            "host_started_at": 2000.0,
        },
        "trial_counter": 6,
    }
    new_counter = autopilot._recover_from_in_flight_trial(
        state, journal, archive, trial_counter=6,
    )
    assert new_counter == 7
    assert state["trial_counter"] == 7
    assert state["in_flight_trial"] is None
    # Placeholder must be present
    entries = journal.all_entries()
    placeholder = next((e for e in entries if e.trial_id == 6), None)
    assert placeholder is not None
    assert placeholder.bug_corrupted_by == "autopilot_killed_mid_trial"
    assert placeholder.deficiency_category == "autopilot_killed_mid_trial"
    assert placeholder.species == "(killed)"
    assert placeholder.action_type == "prompt_mutation"
    assert placeholder.pareto_status == "dominated"
    # Quality/speed/cost/reliability are all zero (no eval evidence)
    assert placeholder.quality == 0.0
    assert placeholder.speed == 0.0


def test_recovery_case_a_counter_never_decreases(
    journal: ExperimentJournal, archive: ParetoArchive
) -> None:
    """If trial_counter is already past journal_max, leave it alone."""
    journal.record(_make_entry(trial_id=3))
    state = {
        "in_flight_trial": {
            "trial_id": 3,
            "action": {"type": "seed_batch"},
            "host_pid": 1,
            "host_started_at": 0.0,
        },
        "trial_counter": 20,
    }
    new_counter = autopilot._recover_from_in_flight_trial(
        state, journal, archive, trial_counter=20,
    )
    assert new_counter == 20
    assert state["trial_counter"] == 20


def test_recovery_idempotent_after_first_call(
    journal: ExperimentJournal, archive: ParetoArchive
) -> None:
    """Calling recovery a second time with cleared marker is a no-op."""
    journal.record(_make_entry(trial_id=4))
    state = {
        "in_flight_trial": {
            "trial_id": 4,
            "action": {"type": "seed_batch"},
            "host_pid": 1,
            "host_started_at": 0.0,
        },
        "trial_counter": 4,
    }
    counter1 = autopilot._recover_from_in_flight_trial(
        state, journal, archive, trial_counter=4,
    )
    # Second call: marker is now None, must be a no-op
    counter2 = autopilot._recover_from_in_flight_trial(
        state, journal, archive, trial_counter=counter1,
    )
    assert counter1 == counter2 == 5


# ───────── _maybe_reimport_pareto_from_journal ──────────


def test_reimport_skips_bug_corrupted_entry(
    journal: ExperimentJournal, archive: ParetoArchive
) -> None:
    journal.record(_make_entry(trial_id=2, bug_corrupted_by="autopilot_killed_mid_trial"))
    result = autopilot._maybe_reimport_pareto_from_journal(archive, journal, 2)
    assert result is False
    assert len(archive._all_entries) == 0


def test_reimport_skips_when_already_in_archive(
    journal: ExperimentJournal, archive: ParetoArchive
) -> None:
    journal.record(_make_entry(trial_id=3))
    # Pre-populate the archive with this trial id
    archive.update(ParetoEntry(trial_id=3, objectives=(0.7, 40.0, -0.3, 0.9)))
    before = len(archive._all_entries)
    result = autopilot._maybe_reimport_pareto_from_journal(archive, journal, 3)
    assert result is False
    assert len(archive._all_entries) == before  # no duplicate


def test_reimport_returns_false_when_no_journal_entry(
    journal: ExperimentJournal, archive: ParetoArchive
) -> None:
    # Journal is empty
    result = autopilot._maybe_reimport_pareto_from_journal(archive, journal, 99)
    assert result is False
    assert len(archive._all_entries) == 0


def test_reimport_adds_missing_valid_entry(
    journal: ExperimentJournal, archive: ParetoArchive
) -> None:
    """Reproduces handoff Section 5.7 corruption window: journal advanced,
    archive missed the save. Reimport should add the entry to the archive."""
    journal.record(_make_entry(
        trial_id=42, quality=0.85, speed=55.0, cost=0.25, reliability=0.95,
    ))
    assert len(archive._all_entries) == 0
    result = autopilot._maybe_reimport_pareto_from_journal(archive, journal, 42)
    assert result is True
    assert len(archive._all_entries) == 1
    re_imported = archive._all_entries[0]
    assert re_imported.trial_id == 42
    # Convention check: ParetoEntry.objectives uses (quality, speed, -cost, reliability)
    assert re_imported.objectives == (0.85, 55.0, -0.25, 0.95)
