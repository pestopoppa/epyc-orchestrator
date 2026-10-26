"""Every durable halt in ``_run_loop_inner``, driven through the real loop (B1-03).

A halt latches ``paused=True`` plus ``_dispatch_deficiency``, persists both, publishes its
phase and leaves the loop. Each case asserts all four; the pre-dispatch halts also assert
nothing was dispatched. Mutation-checked by deleting each site's ``state["paused"] = True``
line in turn: exactly the matching case goes red.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
from contextlib import contextmanager
from typing import Any

import pytest

from tests.unit.autopilot_loop_harness import (
    PlannerCritique,
    autopilot,
    fresh_state,
    measured_result,
    planner_decision,
    run_loop_inner,
)

SEED = {"type": "seed_batch", "n_questions": 3}
NUMERIC = {"type": "numeric_trial", "surface": "s", "params": {"x": 1}}


def test_pause_during_planner_turn_is_merged_before_wal_or_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    def pause_during_plan(_decision: Any, _handlers: dict[int, Any]) -> None:
        autopilot.cmd_pause(None)

    run = run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=True,
        higher_tier_probe=False,
        planner_decisions=[planner_decision(SEED)],
        planner_hook=pause_during_plan,
        real_state_io=True,
        stop_after_sleeps=1,
        tmp_path=tmp_path,
        dispatch_actions_out=[],
    )

    on_disk = json.loads((tmp_path / "autopilot_state.json").read_text())
    assert run.dispatched == []
    assert on_disk["paused"] is True
    assert on_disk.get("in_flight_trial") is None


def test_sigterm_during_planner_turn_does_not_dispatch_selected_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    breadcrumb = autopilot.ExitBreadcrumb(tmp_path / "exit.jsonl")

    def terminate_during_plan(_decision: Any, handlers: dict[int, Any]) -> None:
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    run = run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=True,
        higher_tier_probe=False,
        planner_decisions=[planner_decision(SEED)],
        planner_hook=terminate_during_plan,
        real_state_io=True,
        breadcrumb=breadcrumb,
        tmp_path=tmp_path,
        dispatch_actions_out=[],
    )

    on_disk = json.loads((tmp_path / "autopilot_state.json").read_text())
    assert run.dispatched == []
    assert on_disk.get("in_flight_trial") is None
    assert run.names.count("plan") == 1


@pytest.mark.parametrize(
    "use_controller", [False, True], ids=["no-controller", "forced-controller"]
)
def test_sigterm_during_final_selection_does_not_write_wal_or_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, use_controller: bool
) -> None:
    def terminate_after_selection(_action: dict[str, Any], handlers: dict[int, Any]) -> None:
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    run = run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=use_controller,
        higher_tier_probe=False,
        planner_decisions=[planner_decision(SEED)] if use_controller else None,
        auto_actions=[SEED],
        force_fresh_eval_context=(
            {"candidate": "candidate-a", "source_trial_id": 11} if use_controller else None
        ),
        selection_hook=terminate_after_selection,
        real_state_io=True,
        tmp_path=tmp_path,
        dispatch_actions_out=[],
    )

    on_disk = json.loads((tmp_path / "autopilot_state.json").read_text())
    assert run.dispatched == []
    assert on_disk.get("in_flight_trial") is None
    assert "dispatch" not in run.names


def test_late_pause_during_no_controller_selection_is_merged_before_wal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    def pause_after_selection(_action: dict[str, Any], _handlers: dict[int, Any]) -> None:
        autopilot.cmd_pause(None)

    run = run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=False,
        higher_tier_probe=False,
        auto_actions=[SEED],
        selection_hook=pause_after_selection,
        real_state_io=True,
        stop_after_sleeps=1,
        tmp_path=tmp_path,
        dispatch_actions_out=[],
    )

    on_disk = json.loads((tmp_path / "autopilot_state.json").read_text())
    assert run.dispatched == []
    assert on_disk["paused"] is True
    assert on_disk.get("in_flight_trial") is None


def test_pause_winning_admission_lock_prevents_wal_and_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    original_lock = autopilot.state_write_lock
    armed = {"value": False}
    injected = {"value": False}

    @contextmanager
    def pause_before_admission_lock(path: Any):
        if armed["value"] and not injected["value"]:
            injected["value"] = True
            # cmd_pause is the competing writer. It acquires and releases the
            # real lock before the loop gets its admission lock.
            autopilot.cmd_pause(None)
        with original_lock(path):
            yield

    monkeypatch.setattr(autopilot, "state_write_lock", pause_before_admission_lock)

    def arm_before_admission(_action: dict[str, Any], _handlers: dict[int, Any]) -> None:
        armed["value"] = True

    run = run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=False,
        higher_tier_probe=False,
        auto_actions=[SEED],
        selection_hook=arm_before_admission,
        real_state_io=True,
        stop_after_sleeps=1,
        tmp_path=tmp_path,
        dispatch_actions_out=[],
    )

    on_disk = json.loads((tmp_path / "autopilot_state.json").read_text())
    assert injected["value"] is True
    assert run.dispatched == []
    assert on_disk["paused"] is True
    assert on_disk.get("in_flight_trial") is None


def test_no_controller_keeps_preplanner_multitier_forced_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forced = {"type": "deep_eval", "tier": 2}
    context = {
        "stage": "tier_2",
        "candidate": "candidate-a",
        "candidate_action": {"type": "numeric_trial", "surface": "candidate-a"},
    }
    monkeypatch.setattr(
        autopilot,
        "_maybe_force_multitier_due_action",
        lambda **_kwargs: (forced, {"multitier_validation": True}, context),
    )

    run = run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=False,
        higher_tier_probe=False,
        auto_actions=[SEED],
    )

    assert run.dispatched == [forced]
    assert "auto_action" not in run.names


def test_skip_journal_failure_keeps_wal_for_startup_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    def fail_skip_record(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("journal unavailable")

    monkeypatch.setattr(autopilot, "_record_skip_trial", fail_skip_record)
    with caplog.at_level(logging.ERROR, logger="autopilot"):
        with pytest.raises(OSError, match="journal unavailable"):
            run_loop_inner(
                monkeypatch,
                state=fresh_state(),
                use_controller=True,
                higher_tier_probe=False,
                planner_decisions=[planner_decision(SEED)],
                dispatch_results=[
                    autopilot.SkipOutcome("skipped", "validator rejected", "seed_batch")
                ],
                real_state_io=True,
                tmp_path=tmp_path,
            )

    on_disk = json.loads((tmp_path / "autopilot_state.json").read_text())
    assert on_disk["in_flight_trial"]["trial_id"] == 0
    assert "skip-trial journal write failed" in caplog.text


def test_loop_persists_only_allowlisted_selection_rationale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rationale = {
        "falsifier": "not selection metadata",
        "p0_3_blacklist_reexploration": True,
        "p0_3_blacklist_reexploration_target": "target-a",
        "p0_3_blacklist_reexploration_source_trial": 42,
        "p0_3_blacklist_reexploration_reason": "new instrument era",
        "p0_3_blacklist_reexploration_scope": "audit-only",
        "private_planner_note": "do not persist",
    }
    monkeypatch.setattr(
        autopilot,
        "_p0_3_retryable_blacklist_match",
        lambda *_args, **_kwargs: {
            "target_key": "target-a",
            "source_trial": 42,
            "reason": "new instrument era",
            "retry_scope": "audit-only",
        },
    )
    entries: list[Any] = []

    run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=True,
        higher_tier_probe=False,
        planner_decisions=[planner_decision(SEED, rationale=rationale)],
        blacklisted="matching blacklist row",
        journal_entries_out=entries,
    )

    persisted = entries[-1].eval_details["selection_rationale"]
    assert persisted["p0_3_blacklist_reexploration_target"] == "target-a"
    assert persisted["p0_3_blacklist_reexploration_source_trial"] == 42
    assert "private_planner_note" not in persisted
    assert "falsifier" not in persisted


def test_skip_trial_persists_allowlisted_selection_rationale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rationale = {
        "frontier_rerun_forced": True,
        "frontier_rerun_reason": "new measurement era",
        "frontier_rerun_completed_numeric_trials": 2,
        "frontier_rerun_min_numeric_trials": 4,
        "planner_private": {"large": "not persisted"},
    }
    entries: list[Any] = []

    run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=True,
        higher_tier_probe=False,
        planner_decisions=[planner_decision(SEED, rationale=rationale)],
        dispatch_results=[autopilot.SkipOutcome("skipped", "blocked by validator", "seed_batch")],
        journal_entries_out=entries,
    )

    persisted = entries[-1].eval_details["selection_rationale"]
    assert persisted["frontier_rerun_reason"] == "new measurement era"
    assert "planner_private" not in persisted


def test_controller_iterations_do_not_duplicate_benchmark_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path = str(autopilot.ORCH_ROOT / "scripts" / "benchmark")
    before = sys.path.count(benchmark_path)

    run_loop_inner(
        monkeypatch,
        state=fresh_state(),
        use_controller=True,
        higher_tier_probe=False,
        planner_decisions=[planner_decision(SEED)],
        max_trials=3,
    )

    assert sys.path.count(benchmark_path) == before


def _reject(action: dict[str, Any], draft: dict[str, Any]) -> Any:
    return planner_decision(
        action, draft_action=draft, critique=PlannerCritique(decision="reject", issues=["no"])
    )


def _critic_unavailable(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    # Degraded with no critique object: the draft is uncritiqued and untrusted.
    return {"planner_decisions": [planner_decision(SEED, degraded=True, critique=None)]}


def _critic_reject_no_safe_fallback(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    skip = autopilot.SkipOutcome("skipped", "no metric fallback", "seed_batch")
    monkeypatch.setattr(
        autopilot,
        "_repair_critic_reject_fallback_for_w8",
        lambda action, blacklist, rationale, **_kw: (action, rationale, skip, False),
    )
    return {"planner_decisions": [_reject(SEED, NUMERIC)]}


def _critic_reject_loop(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {
        "planner_decisions": [_reject(SEED, NUMERIC)],
        "state": {"consecutive_rejected_drafts": autopilot.MAX_CONSECUTIVE_REJECTED_DRAFTS - 1},
    }


def _planners_offline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import species.numeric_swarm as numeric_swarm

    monkeypatch.setattr(numeric_swarm, "SURFACES", {})
    return {"planner_decisions": [planner_decision(None, providers_unavailable=True)]}


def _planner_deterministic_guard(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    blocks = autopilot.MAX_CONSECUTIVE_PLANNER_DETERMINISTIC_BLOCKS - 1
    # The 2026-08-04 trial-1472 shape: a binding 'revise' whose action equals the draft.
    revise = PlannerCritique(decision="revise", issues=["same action"])
    return {
        "planner_decisions": [
            planner_decision(
                None, draft_action=NUMERIC, critique=revise, deterministic_block_reason="same"
            )
        ],
        "state": {autopilot.PLANNER_DETERMINISTIC_BLOCK_STATE_KEY: blocks},
    }


def _seq_gate_preflight(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {
        "use_controller": False,
        "seq_gate_preflight_payload": {
            "status": "blocked_unreachable",
            "reason": "alpha_wealth_exhausted",
            "original_action": NUMERIC,
            "replacement_action": None,
        },
    }


def _meta_loop(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    # Without the halt the loop would spin on metric-free meta actions, so the second
    # action is a metric trial that spends the one-trial budget instead.
    return {
        "use_controller": False,
        "auto_actions": [{"type": "distill_knowledge"}, NUMERIC],
        "dispatch_results": [None, measured_result()],
        "state": {"consecutive_meta_actions": autopilot.MAX_CONSECUTIVE_META - 1},
    }


def _skip_loop(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {
        "use_controller": False,
        "auto_actions": [NUMERIC],
        "dispatch_results": [autopilot.SkipOutcome("skipped", "handler no-op", "numeric_trial")],
        "state": {"consecutive_skip_actions": autopilot.MAX_CONSECUTIVE_SKIP - 1},
    }


# (deficiency, halt phase, scenario, dispatches before halting)
HALTS = [
    ("critic_unavailable", "critic_unavailable_halt", _critic_unavailable, 0),
    (
        "critic_reject_no_safe_fallback",
        "critic_reject_no_safe_fallback_halt",
        _critic_reject_no_safe_fallback,
        0,
    ),
    ("critic_reject_loop", "critic_reject_loop_halt", _critic_reject_loop, 0),
    (
        "planners_offline_no_deterministic_fallback",
        "planners_offline_halt",
        _planners_offline,
        0,
    ),
    (
        "planner_deterministic_guard",
        "planner_deterministic_guard_halt",
        _planner_deterministic_guard,
        0,
    ),
    ("seq_gate_preflight_blocked", "seq_gate_preflight_halt", _seq_gate_preflight, 0),
    ("meta_action_loop", "meta_loop_halt", _meta_loop, 1),
    ("skip_action_loop", "skip_loop_halt", _skip_loop, 1),
]


@pytest.mark.parametrize(
    "deficiency, halt_phase, scenario, dispatches", HALTS, ids=[h[1] for h in HALTS]
)
def test_halt_latches_pause_persists_it_and_exits(
    monkeypatch: pytest.MonkeyPatch,
    deficiency: str,
    halt_phase: str,
    scenario: Any,
    dispatches: int,
) -> None:
    kwargs = {"use_controller": True, "higher_tier_probe": False, **scenario(monkeypatch)}
    state = fresh_state(**kwargs.pop("state", {}))

    run = run_loop_inner(monkeypatch, state=state, **kwargs)

    assert state["paused"] is True
    assert state["_dispatch_deficiency"] == deficiency
    halted_at = next(
        i for i, (name, p) in enumerate(run.events) if name == "phase" and p["name"] == halt_phase
    )
    # The latch reached disk before the halt was announced, and the loop then shut down.
    assert any(p["paused"] for i, p in run.saves() if i < halted_at)
    assert run.phases[-2:] == [halt_phase, "shutting_down"]
    assert len(run.dispatched) == dispatches
