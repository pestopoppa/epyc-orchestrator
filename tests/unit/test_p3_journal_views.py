"""Read-view ownership and the separate legacy quality ceiling."""

from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/autopilot"))
import autopilot
import safety_gate as sg
from experiment_journal import ExperimentJournal, JournalEntry
from src.autopilot_core.action_identity import row_config_identity
from src.autopilot_core.journal_reconstruction import reconstruct_archive_from_journal_rows
from src.autopilot_core.live_reproductions import live_reproductions
from src.autopilot_core.tier_specs import LEGACY_OBJECTIVE_POLICY


def entry(tid, quality=1.0):
    return JournalEntry(
        trial_id=tid,
        timestamp=f"2026-06-13T00:00:{tid:02d}+00:00",
        species="unit",
        action_type="structural_experiment",
        tier=1,
        quality=quality,
        speed=10.0,
        cost=0.5,
        reliability=1.0,
        pareto_status="frontier",
        config_snapshot={"type": "structural_experiment", "flags": {"x": 1}},
        eval_details={"nested": {"value": 1}, "frontier_admission": "representative"},
    )


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setattr(ExperimentJournal, "_emit_trace_event", lambda *args: None)
    j = ExperimentJournal(tmp_path, segment_snapshots=False)
    j.record(entry(1, 2.0))
    yield j
    sg.configure_promotion_guard_archive(None)


def test_ceiling_uses_injected_rows_and_all_eras(journal, monkeypatch):
    import scripts.autopilot.experiment_journal as journal_module

    monkeypatch.setattr(journal_module, "ExperimentJournal", lambda: pytest.fail("fresh load"))
    # Live provider deliberately has no old-era evidence; the ceiling must retain it.
    sg.configure_promotion_guard_archive(
        lambda **kw: sg.PromotionGuardView(
            archive=None, objective_policy=LEGACY_OBJECTIVE_POLICY, scope="live"
        ),
        journal=journal,
    )
    archive = sg._pareto_archive_for_safety_guard()
    expected = reconstruct_archive_from_journal_rows(journal.rows(), None)
    assert archive is not None and expected is not None
    assert max(e.objectives[0] for e in archive.frontier(1)) == 2.0
    assert (
        reconstruct_archive_from_journal_rows(journal.rows(), None, exclude_before_ts=2_000_000_000)
        is None
    )
    sg.configure_promotion_guard_archive(None)
    assert sg._QUALITY_CEILING_JOURNAL is None
    assert sg._PROMOTION_GUARD_PROVIDER is None


@pytest.mark.parametrize("fails", [False, True])
def test_loop_releases_journal_on_return_and_error(journal, monkeypatch, fails):
    class Breadcrumb:
        def __getattr__(self, name):
            return lambda *a, **kw: None

    monkeypatch.setattr(autopilot, "ExitBreadcrumb", Breadcrumb)

    def inner(*args):
        sg.configure_promotion_guard_archive(lambda **kw: None, journal=journal)
        if fails:
            raise RuntimeError("controlled")

    monkeypatch.setattr(autopilot, "_run_loop_inner", inner)
    if fails:
        with pytest.raises(RuntimeError, match="controlled"):
            autopilot.run_loop(use_tui=False)
    else:
        autopilot.run_loop(use_tui=False)
    assert sg._QUALITY_CEILING_JOURNAL is None
    assert sg._PROMOTION_GUARD_PROVIDER is None


def test_consumers_preserve_nested_views_and_invalidate(journal):
    fields = {
        "eval_details": {"nested": {"value": 2}, "frontier_admission": "representative"},
        "config_snapshot": {"type": "structural_experiment", "flags": {"x": 2}},
    }
    original = copy.deepcopy(fields)
    journal.rows(folded=True)  # warm both generations
    journal.append_supersession_event(
        target_trial_ids=[1], fields=fields, reason="unit", policy_version="unit", actor="unit"
    )
    rows = journal.rows() + journal.supersession_events()
    before = copy.deepcopy(rows)
    folded_before = copy.deepcopy([asdict(e) for e in journal.folded_entries()])
    assert journal.rows(folded=True)[0]["eval_details"]["nested"]["value"] == 2
    assert journal.rows()[0]["eval_details"]["nested"]["value"] == 1
    for _ in range(2):
        assert reconstruct_archive_from_journal_rows(rows, None) is not None
        assert live_reproductions(
            rows,
            tier=1,
            identity=row_config_identity(journal.rows(folded=True)[0]) or "",
            objective_policy=LEGACY_OBJECTIVE_POLICY,
            exclude_before_ts=None,
        )
    assert rows == before
    assert [asdict(e) for e in journal.folded_entries()] == folded_before
    assert fields == original
    journal.record(entry(2))
    assert len(journal.rows()) == 2
    # Reload means a fresh constructor, never calling private _load_existing again.
    reloaded = ExperimentJournal(journal.journal_dir, segment_snapshots=False)
    assert reloaded.rows(folded=True) == journal.rows(folded=True)


def test_apply_params_forwards_successful_restart_boundary(monkeypatch):
    import config_applicator as app

    events = []

    class Journal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return kwargs

    def restart(**kwargs):
        kwargs["journal"].append_role_restart_boundary_event(
            boundary_trial_id=kwargs["trial_id"], status="ok", role=kwargs["role"]
        )
        return {"status": "ok"}

    monkeypatch.setattr(app, "_ap3_role_restart_gate_mode", lambda: "scoped")
    monkeypatch.setattr(app, "_ap3_scope_errors", lambda params: [])
    monkeypatch.setattr(app, "_ap3_restart_block", lambda: None)
    monkeypatch.setattr(app, "_ap3_prewarm_role_targets", lambda *a, **kw: {"status": "ok"})
    monkeypatch.setattr(app, "_ap3_strict_smoke_with_readiness", lambda **kw: None)
    monkeypatch.setattr(app, "restart_role", restart)
    result = app.apply_params(
        {"role_restart.frontdoor_draft_max": 4}, journal=Journal(), trial_id=7
    )
    assert result["status"] == "ok"
    assert events == [{"boundary_trial_id": 7, "status": "ok", "role": "frontdoor"}]


def test_failed_critique_trace_reaches_prior_decision_digest(tmp_path):
    import json

    archive = tmp_path / "planner.jsonl"
    archive.write_text(
        json.dumps(
            {
                "type": "planner_coordinator",
                "action_type": "seed_batch",
                "critic_provider": "codex",
                "provider_trace": [
                    {
                        "role": "critique",
                        "provider": "codex",
                        "ok": False,
                        "error": "synthetic critic disconnect",
                    }
                ],
            }
        )
        + "\n"
    )
    text = autopilot._build_prior_planner_decision_digest(archive)
    assert "synthetic critic disconnect" in text
    assert "codex" in text
    archive.write_text(
        json.dumps(
            {
                "type": "planner_coordinator",
                "action_type": "seed_batch",
                "provider_trace": [
                    None,
                    {"ok": True, "error": "successful event must be hidden"},
                    {"ok": False, "error": "failure\n\x1b" + "x" * 500},
                    {"ok": False, "error": "second failure"},
                    {"ok": False, "error": "third failure must be hidden"},
                ],
            }
        )
        + "\n"
    )
    text = autopilot._build_prior_planner_decision_digest(archive)
    assert "provider_failures=" in text and "second failure" in text
    assert "successful event" not in text and "third failure" not in text
    assert "\n" not in text and "\x1b" not in text and "x" * 161 not in text


def test_trace_emit_failure_warns_once_without_losing_rows(tmp_path, monkeypatch, caplog):
    import importlib

    trace_emit = importlib.import_module("src.trace.emit")

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic trace unavailable")

    monkeypatch.setattr(trace_emit, "emit", fail)
    j = ExperimentJournal(tmp_path, segment_snapshots=False)
    j.record(entry(1))
    j.record(entry(2))
    assert j.count() == 2
    assert sum("Live trace event emission failed" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("action_type", ["review_policy_trial", "screening_tier_driver"])
def test_formerly_omitted_actions_reach_loop_dispatch(monkeypatch, action_type):
    from tests.unit.autopilot_loop_harness import run_loop_inner, fresh_state

    dispatched = []
    run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        auto_actions=[{"type": action_type}],
        higher_tier_probe=False,
        dispatch_actions_out=dispatched,
    )
    assert [a["type"] for a in dispatched] == [action_type]


def test_legacy_baseline_readers_preserve_unknown_revision_and_quality():
    from types import SimpleNamespace
    import crossover_diagnostics
    import decision_cockpit
    from src.autopilot_core import ap55_promotion_gate

    row = {
        "trial_id": 1,
        "tier": 1,
        "quality": 1.5,
        "baseline_pin": {},
        "failure_analysis": "Quality regression: 1.444 vs baseline 1.900 (-24%)",
    }
    delta = decision_cockpit.delta_row(row, incumbent={"available": False}, era="", replication={})
    assert delta["vs_pinned_baseline"]["baseline_quality"] == 1.9
    assert delta["vs_pinned_baseline"]["era"] is None
    assert ap55_promotion_gate._baseline_revision(row) is None
    # Legacy prose must never invent the native scope/revision needed for crossover.
    assert crossover_diagnostics._current_native_scope(SimpleNamespace(**row)) is None
