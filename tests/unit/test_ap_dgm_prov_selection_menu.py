"""AP-DGM-PROV option A: observe-only selector-menu capture on the lineage record.

Pins: selection output is byte-identical with capture on or off (only the extra
``selection_menu`` key is added); the menu is the derived same-species committed-promotion
set the rule walks, newest first, with exclusion counts; decoding_id is a digest of the
DECLARED settings and is None (never invented) when nothing is declared; the holdout marker
is the declared id or ``declared: False``; a capture failure is fail-open; capture is
default-off; the capture does not mutate the journal.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import experiment_journal as ej  # noqa: E402
from experiment_journal import (  # noqa: E402
    ExperimentJournal,
    JournalEntry,
    SupersessionEvent,
    capture_selection_menu,
    declared_decoding_identity,
    declared_holdout_marker,
    select_lineage_parent,
    selection_menu_capture_enabled,
)


def _entry(trial_id: int, species: str = "seeder", **kw) -> JournalEntry:
    base = dict(
        trial_id=trial_id, timestamp="2026-11-09T00:00:00+00:00", species=species,
        action_type="seed_batch", tier=1, quality=2.0, speed=10.0, cost=0.1,
        reliability=1.0, pareto_status="frontier",
    )
    base.update(kw)
    return JournalEntry(**base)


def _promote(journal: ExperimentJournal, trial_id: int) -> None:
    journal.append_baseline_promotion_event(
        source_trial_id=trial_id, tier=1, previous_quality=1.0, new_quality=2.0,
        reason="test", proof={}, result_metrics={}, baseline_state={},
    )


@pytest.fixture
def journal(tmp_path: Path) -> ExperimentJournal:
    return ExperimentJournal(journal_dir=tmp_path / "journal", segment_snapshots=False)


@pytest.fixture
def busy_journal(journal):
    journal.record(_entry(0))
    _promote(journal, 0)
    journal.record(_entry(1, bug_corrupted_by="abc"))
    _promote(journal, 1)
    journal.record(_entry(2))
    _promote(journal, 2)
    journal.record(_entry(3, species="numeric_swarm"))
    _promote(journal, 3)
    journal.record(_entry(4))
    _promote(journal, 4)
    journal.record(_entry(5))  # never promoted
    _promote(journal, 99)  # no such row
    _promote(journal, 2)  # duplicate promotion of 2
    return journal


def _strip(lineage: dict) -> dict:
    return {k: v for k, v in lineage.items() if k != "selection_menu"}


@pytest.mark.parametrize("before", [0, 1, 3, 5, 6, 8, 100])
@pytest.mark.parametrize("species", ["seeder", "numeric_swarm", "absent"])
def test_selection_identical_with_capture_on(busy_journal, species, before):
    off = select_lineage_parent(busy_journal, species, before_trial_id=before)
    on = select_lineage_parent(busy_journal, species, before_trial_id=before, capture_menu=True)
    assert "selection_menu" not in off
    assert "selection_menu" in on
    assert _strip(on) == off  # every pre-existing key, including parent_trial_id, unchanged
    menu = on["selection_menu"]
    ids = [c["trial_id"] for c in menu["eligible_candidates"]]
    if off["parent_trial_id"] is None:
        assert ids == []
    else:
        assert ids[0] == off["parent_trial_id"]  # the rule picks the head of the menu


def test_menu_contents_order_and_exclusions(busy_journal):
    menu = capture_selection_menu(busy_journal, "seeder", before_trial_id=6)
    assert [c["trial_id"] for c in menu["eligible_candidates"]] == [2, 4, 0]  # newest PROMOTION EVENT first (2 was promoted again last)
    assert all(c["promotion_timestamp"] for c in menu["eligible_candidates"])
    assert menu["excluded_counts"] == {
        "malformed_source": 0, "unknown_or_later_row": 1, "other_species": 1,
        "bug_corrupted": 1, "duplicate_promotion": 1,
    }
    assert menu["menu_kind"] == "derived_same_species_committed_promotions"
    assert menu["observe_only"] is True
    assert menu["selector_revision"] == "latest_committed_baseline_promotion_same_species/v1"
    assert menu["eligibility_tie_break"] == "newest_baseline_promotion_event_first"
    assert menu["before_trial_id"] == 6


def test_later_and_superseded_rows_are_not_in_the_menu(busy_journal):
    menu = capture_selection_menu(busy_journal, "seeder", before_trial_id=3)
    assert [c["trial_id"] for c in menu["eligible_candidates"]] == [2, 0]
    busy_journal.append_ledger_event({**SupersessionEvent(
        target_trial_ids=[2], fields={"bug_corrupted_by": "x"}, reason="t",
        policy_version="t", actor="t").__dict__})
    on = select_lineage_parent(busy_journal, "seeder", before_trial_id=3, capture_menu=True)
    assert [c["trial_id"] for c in on["selection_menu"]["eligible_candidates"]] == [0]
    assert on["parent_trial_id"] == 0


def test_decoding_id_is_declared_not_invented():
    assert declared_decoding_identity({}) == {
        "decoding_id": None, "status": "unresolved", "settings": {}}
    env = {"AUTOPILOT_CODEX_MODEL": "m", "AUTOPILOT_CODEX_EFFORT": "high", "UNRELATED": "x"}
    a = declared_decoding_identity(env)
    assert a["status"] == "env_declared"
    assert a["decoding_id"].startswith("dec-") and len(a["decoding_id"]) == 20
    assert a["settings"] == {"AUTOPILOT_CODEX_MODEL": "m", "AUTOPILOT_CODEX_EFFORT": "high"}
    assert declared_decoding_identity({**env, "AUTOPILOT_CODEX_EFFORT": "low"})["decoding_id"] \
        != a["decoding_id"]
    assert declared_decoding_identity(dict(reversed(list(env.items()))))["decoding_id"] \
        == a["decoding_id"]


def test_holdout_marker_is_declared_or_absent():
    assert declared_holdout_marker({}) == {"declared": False, "holdout_id": None, "declared_at": None}
    got = declared_holdout_marker({
        "AUTOPILOT_SELECTOR_HOLDOUT_ID": "h1",
        "AUTOPILOT_SELECTOR_HOLDOUT_DECLARED_AT": "2026-12-01T00:00:00Z"})
    assert got == {"declared": True, "holdout_id": "h1", "declared_at": "2026-12-01T00:00:00Z"}


def test_menu_carries_decoding_and_holdout(journal):
    journal.record(_entry(0))
    _promote(journal, 0)
    env = {"AUTOPILOT_LOCAL_PLANNER_TEMPERATURE": "0.0", "AUTOPILOT_SELECTOR_HOLDOUT_ID": "h1"}
    menu = capture_selection_menu(journal, "seeder", before_trial_id=1, environ=env)
    assert menu["decoding_id"] == declared_decoding_identity(env)["decoding_id"]
    assert menu["holdout"]["holdout_id"] == "h1"


def test_capture_is_default_off_and_env_gated():
    assert selection_menu_capture_enabled({}) is False
    assert selection_menu_capture_enabled({"AUTOPILOT_SELECTOR_MENU_CAPTURE": "0"}) is False
    assert selection_menu_capture_enabled({"AUTOPILOT_SELECTOR_MENU_CAPTURE": "1"}) is True


def test_capture_failure_is_fail_open_and_selection_unchanged(busy_journal, monkeypatch):
    off = select_lineage_parent(busy_journal, "seeder", before_trial_id=6)

    def boom(*a, **k):
        raise RuntimeError("menu exploded")

    monkeypatch.setattr(ej, "capture_selection_menu", boom)
    on = select_lineage_parent(busy_journal, "seeder", before_trial_id=6, capture_menu=True)
    assert on["selection_menu_error"].startswith("RuntimeError")
    assert {k: v for k, v in on.items() if k != "selection_menu_error"} == off


def test_capture_does_not_mutate_journal(busy_journal):
    before_entries = copy.deepcopy([e.__dict__ for e in busy_journal.all_entries()])
    before_events = copy.deepcopy(busy_journal.ledger_events())
    select_lineage_parent(busy_journal, "seeder", before_trial_id=6, capture_menu=True)
    assert [e.__dict__ for e in busy_journal.all_entries()] == before_entries
    assert busy_journal.ledger_events() == before_events


def test_menu_round_trips_on_the_journal_row(journal, tmp_path):
    journal.record(_entry(0))
    _promote(journal, 0)
    lin = select_lineage_parent(journal, "seeder", before_trial_id=1, capture_menu=True)
    journal.record(_entry(1, lineage=lin))
    row = ExperimentJournal(
        journal_dir=tmp_path / "journal", segment_snapshots=False
    ).all_entries()[1]
    assert row.lineage["selection_menu"]["eligible_candidates"][0]["trial_id"] == 0
    assert row.lineage["parent_trial_id"] == 0
