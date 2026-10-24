"""Every durable halt in ``_run_loop_inner``, driven through the real loop (B1-03).

A halt latches ``paused=True`` plus ``_dispatch_deficiency``, persists both, publishes its
phase and leaves the loop. Each case asserts all four; the pre-dispatch halts also assert
nothing was dispatched. Mutation-checked by deleting each site's ``state["paused"] = True``
line in turn: exactly the matching case goes red.
"""

from __future__ import annotations

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
    ("planner_deterministic_guard", "planner_deterministic_guard_halt", _planner_deterministic_guard, 0),
    ("seq_gate_preflight_blocked", "seq_gate_preflight_halt", _seq_gate_preflight, 0),
    ("meta_action_loop", "meta_loop_halt", _meta_loop, 1),
    ("skip_action_loop", "skip_loop_halt", _skip_loop, 1),
]


@pytest.mark.parametrize(
    "deficiency, halt_phase, scenario, dispatches", HALTS, ids=[h[1] for h in HALTS]
)
def test_halt_latches_pause_persists_it_and_exits(
    monkeypatch: pytest.MonkeyPatch, deficiency: str, halt_phase: str, scenario: Any, dispatches: int
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
