"""Multi-trial and loop-top behaviour of the real ``_run_loop_inner`` (B5-02).

Characterization tests for the paths a decomposition of the loop must keep: state carried
across trials, the plot/checkpoint cadence, the pause poll at the loop top, the
consecutive-failure rollback and the pre-dispatch blacklist skip. Mutation-checked: dropping
each carried key, cadence, poll, restore or skip in autopilot.py turns its case red.
"""

from __future__ import annotations

import signal

import pytest

from src.autopilot_core import mutation_guards
from tests.unit.autopilot_loop_harness import (
    autopilot,
    fresh_state,
    measured_result,
    run_loop_inner,
    safety_gate,
)

NUMERIC = {"type": "numeric_trial", "surface": "s", "params": {"x": 1}}


def _sync_plot_renders(run) -> int:
    return sum(p["cmd"][-1] == "plot" for p in run.of("subprocess_run"))


def _autonomous(monkeypatch: pytest.MonkeyPatch, **kwargs):
    kwargs.setdefault("state", fresh_state())
    return run_loop_inner(monkeypatch, auto_actions=[NUMERIC], higher_tier_probe=False, **kwargs)


def test_three_trials_carry_state_and_fire_the_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autopilot, "PLOT_INTERVAL", 2)
    results = [measured_result(quality=q) for q in (2.1, 2.2, 2.3)]

    run = _autonomous(
        monkeypatch,
        state=fresh_state(trial_counter=24),
        max_trials=3,
        dispatch_results=results,
        verdict_passed=False,
    )

    assert [e.trial_id for e in run.journal_entries] == [24, 25, 26]
    assert run.state["trial_counter"] == 27
    assert run.state["consecutive_failures"] == 3
    assert run.state["quality_history_by_tier"] == {"2": [2.1, 2.2, 2.3]}
    # Plots on the first trial (stale clock) and every PLOT_INTERVAL; the auto-checkpoint
    # every 25 trials; one synchronous render and one checkpoint at shutdown.
    assert [p["task"] for p in run.of("subprocess")] == ["plots-trial-24", "plots-trial-26"]
    assert [p["trial_id"] for p in run.of("checkpoint")] == [25, 27]
    assert _sync_plot_renders(run) == 1
    assert run.phases[-2:] == ["max_trials_reached", "shutting_down"]


def test_paused_loop_polls_without_dispatching(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _autonomous(monkeypatch, state=fresh_state(paused=True), stop_after_sleeps=2)

    assert [p["seconds"] for p in run.of("sleep")] == [autopilot.PAUSE_POLL_S] * 2
    assert run.phases.count("paused") == 2 and "health_check" not in run.phases
    assert run.dispatched == [] and run.journal_entries == []
    # One synchronous plot render on entering the episode, one at shutdown.
    assert _sync_plot_renders(run) == 2
    assert run.state["paused"] is True


def test_consecutive_failure_limit_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _autonomous(
        monkeypatch,
        dispatch_results=[measured_result()],
        verdict_passed=False,
        should_rollback=True,
    )

    [blacklisted] = run.of("append_blacklist")
    assert blacklisted["reason_class"] == "safety_failure"
    assert run.index("append_blacklist") < run.index("restore_checkpoint") < run.index("record")
    [entry] = run.journal_entries
    assert (
        entry.deficiency_category == "consecutive_failures"
    )  # AP-14, via the dispatch side channel
    assert run.state["consecutive_failures"] == 0  # gate.reset_failures() after the restore
    assert run.state["paused"] is False


def test_blacklisted_action_is_journaled_invalid_and_never_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _autonomous(monkeypatch, blacklisted="known bad config")

    assert run.dispatched == []
    [entry] = run.journal_entries
    assert (entry.outcome_status, entry.pareto_status) == ("invalid", "skipped")
    assert entry.failure_analysis == "action blacklisted: known bad config"
    assert run.state["last_invalid_status"] == "invalid"


def test_loop_hooks_do_not_leak_into_the_test_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1-15: the loop installs signal handlers, the promotion-guard provider and the
    eval-leakage observer and never removes them; the harness must contain all three."""
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    run = _autonomous(monkeypatch)

    assert set(run.handlers) == {signal.SIGINT, signal.SIGTERM}  # the loop did install them
    assert {"guard_scope", "leakage_install"} <= set(run.names)
    assert {sig: signal.getsignal(sig) for sig in before} == before
    assert safety_gate._PROMOTION_GUARD_PROVIDER is None
    assert mutation_guards._EVAL_LEAKAGE_OBSERVER is None
