"""AP-DGM-PROV option C: planner proposals as the selector menu (observe-only, default off).

Pins: with the flag unset nothing changes (PlannerDecision.proposal_menu is None and the
lineage record has no ``planner_proposal_menu`` key); with it set the lineage selection is
identical (only the extra key is added); the menu lists the primary draft, the failover draft
and the critic-revised action and links the final and dispatched actions by digest without
claiming a choice that no proposal produced; failures are fail-open; the menu is a detached
copy and the journal is not mutated.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import experiment_journal as ej  # noqa: E402
from experiment_journal import (  # noqa: E402
    ExperimentJournal,
    JournalEntry,
    capture_planner_proposal_menu,
    planner_proposal_menu_enabled,
    select_lineage_parent,
)

pc = importlib.import_module("planner_coordinator")

from tests.unit import test_autopilot_planner_coordinator as tpc  # noqa: E402

FLAG = "AUTOPILOT_SELECTOR_PLANNER_PROPOSAL_MENU"


@pytest.fixture(autouse=True)
def _isolate_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same isolation as test_autopilot_planner_coordinator: no archive writes, no live repair."""
    monkeypatch.setattr(pc, "_append_planner_archive", lambda _record: None)
    monkeypatch.setenv("AUTOPILOT_PLANNER_SPEND_BREAKER", "0")
    monkeypatch.setattr(pc, "action_repair_completer", lambda: tpc._unreachable_repair_complete)
ORIGINAL = {"type": "structural_experiment", "flags": {"a": True}}
REVISED = {"type": "seed_batch", "n_questions": 10}


def _entry(trial_id: int, species: str = "seeder") -> JournalEntry:
    return JournalEntry(
        trial_id=trial_id, timestamp="2026-11-09T00:00:00+00:00", species=species,
        action_type="seed_batch", tier=1, quality=2.0, speed=10.0, cost=0.1,
        reliability=1.0, pareto_status="frontier",
    )


@pytest.fixture
def journal(tmp_path: Path) -> ExperimentJournal:
    j = ExperimentJournal(journal_dir=tmp_path / "journal", segment_snapshots=False)
    for tid in (0, 1, 2):
        j.record(_entry(tid))
        j.append_baseline_promotion_event(
            source_trial_id=tid, tier=1, previous_quality=1.0, new_quality=2.0,
            reason="test", proof={}, result_metrics={}, baseline_state={},
        )
    return j


def _revise_plan(monkeypatch=None) -> Any:
    claude = tpc.FakeProvider(
        "claude",
        [tpc.PlannerProviderResult(provider="claude", role="draft", ok=True,
                                   text=tpc._action_text(ORIGINAL))],
        supports_resume=True,
    )
    codex = tpc.FakeProvider(
        "codex",
        [tpc.PlannerProviderResult(
            provider="codex", role="critique", ok=True,
            text=tpc._critique_text({"decision": "revise", "confidence": 0.8,
                                     "issues": ["x"], "revised_action": REVISED}))],
    )
    return tpc._plan(
        "prompt", session_id=None, planner_state={},
        settings=tpc.PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=tpc._factory({"claude": claude, "codex": codex}),
    )


def test_flag_is_default_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert planner_proposal_menu_enabled() is False
    assert pc.proposal_menu_capture_enabled() is False
    for v in ("", "0", "no", "off", "false"):
        assert planner_proposal_menu_enabled({FLAG: v}) is False
        assert pc.proposal_menu_capture_enabled({FLAG: v}) is False
    assert planner_proposal_menu_enabled({FLAG: "1"}) is True
    assert pc.proposal_menu_capture_enabled({FLAG: "true"}) is True


def test_planner_decision_unchanged_when_flag_unset(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    off = _revise_plan()
    assert off.proposal_menu is None
    monkeypatch.setenv(FLAG, "1")
    on = _revise_plan()
    assert on.proposal_menu is not None
    # Every dispatch-relevant field is identical with the flag on or off.
    for field in ("action", "rationale", "draft_action", "mode", "degraded", "fallback_reason",
                  "draft_provider", "critic_provider", "canonical_text"):
        assert getattr(on, field) == getattr(off, field), field
    assert on.critique.decision == off.critique.decision == "revise"


def test_menu_lists_draft_and_critic_revision_and_links_final(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    d = _revise_plan()
    m = d.proposal_menu
    assert m["menu_kind"] == "planner_proposals" and m["observe_only"] is True
    assert [p["stage"] for p in m["proposals"]] == ["draft_primary", "critic_revised"]
    assert m["proposals"][0]["action"] == ORIGINAL and m["proposals"][1]["action"] == REVISED
    assert m["proposals"][0]["provider"] == "claude" and m["proposals"][1]["provider"] == "codex"
    assert all(p["usable"] for p in m["proposals"])
    assert d.action == REVISED
    assert m["final_action_stage"] == "critic_revised"
    assert m["final_action_digest"] == m["proposals"][1]["action_digest"]
    assert m["critic_decision"] == "revise"


def test_failover_draft_is_a_menu_entry(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    claude = tpc.FakeProvider(
        "claude", [tpc.PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json")],
        supports_resume=True)
    codex = tpc.FakeProvider(
        "codex", [tpc.PlannerProviderResult(provider="codex", role="draft", ok=True,
                                            text=tpc._action_text(REVISED))])
    d = tpc._plan(
        "prompt", session_id="old", planner_state={},
        settings=tpc.PlannerSettings(mode="fallback"),
        provider_factory=tpc._factory({"claude": claude, "codex": codex}),
    )
    m = d.proposal_menu
    assert d.action == REVISED
    # The unparseable primary yields no action, so it is not an entry; the failover draft is.
    assert [p["stage"] for p in m["proposals"]] == ["draft_fallback"]
    assert m["final_action_stage"] == "draft_fallback"


def test_no_action_path_still_has_a_menu(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    claude = tpc.FakeProvider(
        "claude", [tpc.PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json")])
    codex = tpc.FakeProvider(
        "codex", [tpc.PlannerProviderResult(provider="codex", role="draft", ok=True, text="nope")])
    d = tpc._plan(
        "prompt", session_id=None, planner_state={},
        settings=tpc.PlannerSettings(mode="fallback"),
        provider_factory=tpc._factory({"claude": claude, "codex": codex}),
    )
    assert d.action is None
    assert d.proposal_menu["proposals"] == []
    assert d.proposal_menu["final_action_digest"] is None
    assert d.proposal_menu["final_action_stage"] is None


def test_substituted_final_action_claims_no_stage():
    entry = []
    pc._menu_add(entry, "draft_primary", "claude", ORIGINAL)
    m = pc.build_proposal_menu(entry, final_action={"type": "numeric_trial", "surface": "s"})
    assert m["final_action_stage"] is None and m["final_action_digest"].startswith("act-")


def test_menu_add_is_noop_when_off_and_ignores_non_dict_actions():
    pc._menu_add(None, "draft_primary", "claude", ORIGINAL)  # must not raise
    lst: list = []
    pc._menu_add(lst, "draft_primary", "claude", None)
    pc._menu_add(lst, "draft_primary", "claude", "not a dict")
    assert lst == []
    assert pc.build_proposal_menu(None, final_action=ORIGINAL) is None


@pytest.mark.parametrize("before", [0, 1, 3, 100])
@pytest.mark.parametrize("species", ["seeder", "absent"])
def test_lineage_selection_identical_with_proposal_menu(journal, species, before):
    off = select_lineage_parent(journal, species, before_trial_id=before)
    menu = {"schema_version": 1, "menu_kind": "planner_proposals", "observe_only": True,
            "proposals": [{"stage": "draft_primary", "action": ORIGINAL,
                           "action_digest": pc._proposal_action_digest(ORIGINAL)}],
            "final_action_digest": pc._proposal_action_digest(ORIGINAL),
            "final_action_stage": "draft_primary", "critic_decision": "approve"}
    on = select_lineage_parent(journal, species, before_trial_id=before,
                               proposal_menu=menu, dispatched_action=ORIGINAL)
    assert "planner_proposal_menu" not in off
    assert {k: v for k, v in on.items() if k != "planner_proposal_menu"} == off
    assert on["parent_trial_id"] == off["parent_trial_id"]
    got = on["planner_proposal_menu"]
    assert got["dispatched_stage"] == "draft_primary"
    assert got["selector_revision"].endswith("/v1")
    assert "holdout" in got and "decoding_id" in got


def test_default_call_has_no_new_keys(journal):
    r = select_lineage_parent(journal, "seeder", before_trial_id=3)
    assert set(r) == {"schema_version", "rule", "species", "parent_trial_id",
                      "parent_promotion_timestamp", "heuristic_rule",
                      "heuristic_parent_trial_id"}


def test_dispatched_substitute_is_not_linked_and_menu_is_detached():
    menu = {"proposals": [{"stage": "draft_primary", "action": ORIGINAL,
                           "action_digest": pc._proposal_action_digest(ORIGINAL)}]}
    before = copy.deepcopy(menu)
    out = capture_planner_proposal_menu(menu, dispatched_action={"type": "other"})
    assert out["dispatched_stage"] is None and out["dispatched_action_digest"].startswith("act-")
    out["proposals"][0]["action"]["flags"]["a"] = False
    assert menu == before  # caller's menu not aliased
    assert capture_planner_proposal_menu(None) is None
    assert capture_planner_proposal_menu("junk") is None
    out2 = capture_planner_proposal_menu(menu, dispatched_action=None)
    assert out2["dispatched_action_digest"] is None and out2["dispatched_stage"] is None


def test_capture_failure_is_fail_open_and_journal_unmutated(journal, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ej, "capture_planner_proposal_menu", boom)
    before_events = list(journal.baseline_promotion_events())
    ref = select_lineage_parent(journal, "seeder", before_trial_id=3)
    r = select_lineage_parent(journal, "seeder", before_trial_id=3, proposal_menu={"proposals": []})
    assert r["parent_trial_id"] == ref["parent_trial_id"]
    assert r["planner_proposal_menu_error"].startswith("RuntimeError")
    assert "planner_proposal_menu" not in r
    assert list(journal.baseline_promotion_events()) == before_events


def test_menu_is_json_serialisable(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    d = _revise_plan()
    json.dumps(d.proposal_menu)
