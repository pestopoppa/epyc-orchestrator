"""AP-67 (S3-AP-05): planner-authored numeric trials must reach the sampler (default off).

The explicit-params branch applies and evaluates without ``study.ask``/``tell``, so NSGA-II is
blind to every LLM trial. With AUTOPILOT_PLANNER_NUMERIC_TELL=1 a COMPLETED planner trial is added
to the surface study (``source=planner``, native-qualified provenance). Only full-vector, in-bounds
trials are added; failures are not injected.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

autopilot = importlib.import_module("scripts.autopilot.autopilot")
numeric_swarm = importlib.import_module("scripts.autopilot.species.numeric_swarm")

ENV = "AUTOPILOT_PLANNER_NUMERIC_TELL"
SURFACE = "think_harder"
FULL = {
    "think_harder.min_expected_roi": 0.02,
    "think_harder.token_budget_min": 1024,
    "think_harder.token_budget_max": 4096,
    "think_harder.cot_roi_threshold": 0.3,
}
KW = dict(cost_tier_observed_count=1, reliability_observed_count=1)
# AP-9 limits an explicit planner numeric trial to ONE param, so the end-to-end path is only
# representable (full vector) on a single-param surface.
ONE_SURFACE = "chat_pipeline"
ONE = {"chat.try_cheap_first_q_threshold": 0.6}


def _swarm(tmp_path):
    return numeric_swarm.NumericSwarm(db_path=tmp_path / "numeric.db")


def test_record_planner_trial_adds_a_complete_tagged_trial_the_sampler_can_use(tmp_path):
    swarm = _swarm(tmp_path)
    out = swarm.record_planner_trial(SURFACE, FULL, (2.0, 10.0, -0.1, 1.0), **KW)
    assert out["status"] == "recorded"
    study = swarm._get_study(SURFACE)
    t = study.trials[0]
    assert t.state.name == "COMPLETE" and t.params == FULL
    assert t.user_attrs["source"] == "planner"
    assert t.user_attrs["objective_source"] == "native_qualified"
    assert t.user_attrs["constraint_violation"] == 0.0
    # the NSGA-II sampler can still ask after seeing the planner trial (constraints readable)
    for _ in range(3):
        sug = swarm.suggest_trial(SURFACE)
        swarm.report_result(SURFACE, sug["trial_number"], (2.1, 11.0, -0.1, 1.0), **KW)
    assert len([x for x in study.trials if x.state.name == "COMPLETE"]) == 4


def test_slo_miss_planner_trial_carries_the_constraint_violation(tmp_path):
    swarm = _swarm(tmp_path)
    swarm.record_planner_trial(SURFACE, FULL, (0.5, 10.0, -0.1, 1.0),
                               normalized_overages=[0.5], overage_cap_pairs=[("quality_floor", 0.5, 1.0)],
                               outcome_class="slo_miss", **KW)
    t = swarm._get_study(SURFACE).trials[0]
    assert 0 < t.user_attrs["constraint_violation"] < 1
    assert t.user_attrs["outcome_class"] == "slo_miss"


@pytest.mark.parametrize("params,why", [
    ({k: v for k, v in FULL.items() if k != "think_harder.cot_roi_threshold"}, "partial_or_foreign"),
    ({**FULL, "think_harder.bogus": 1.0}, "partial_or_foreign"),
    ({**FULL, "think_harder.min_expected_roi": 5.0}, "outside"),
    ({**FULL, "think_harder.token_budget_min": 1024.5}, "non-integer"),
    ({**FULL, "think_harder.cot_roi_threshold": True}, "non-numeric"),
    ({**FULL, "think_harder.cot_roi_threshold": float("nan")}, "non-numeric"),
])
def test_unrepresentable_planner_trials_are_skipped_and_leave_the_study_untouched(tmp_path, params, why):
    swarm = _swarm(tmp_path)
    out = swarm.record_planner_trial(SURFACE, params, (2.0, 10.0, -0.1, 1.0), **KW)
    assert out["status"] == "skipped" and why in out["reason"]
    assert len(swarm._get_study(SURFACE).trials) == 0


def test_bad_objectives_or_counts_are_skipped(tmp_path):
    swarm = _swarm(tmp_path)
    assert swarm.record_planner_trial(SURFACE, FULL, (2.0, float("inf"), -0.1, 1.0), **KW)["status"] == "skipped"
    assert swarm.record_planner_trial(
        SURFACE, FULL, (2.0, 10.0, -0.1, 1.0),
        cost_tier_observed_count=0, reliability_observed_count=1)["status"] == "skipped"
    assert swarm.record_planner_trial("nope", FULL, (2.0, 10.0, -0.1, 1.0), **KW)["status"] == "skipped"
    assert len(swarm._get_study(SURFACE).trials) == 0


def _dispatch(monkeypatch, tmp_path, flag, params=None, surface=ONE_SURFACE):
    from safety_gate import EvalResult, SafetyGate

    if flag:
        monkeypatch.setenv(ENV, "1")
    else:
        monkeypatch.delenv(ENV, raising=False)
    swarm = _swarm(tmp_path)
    gate = SafetyGate(baseline_path=tmp_path / "absent.yaml")
    result = EvalResult(tier=1, quality=2.0, speed=20.0, cost=0.2, reliability=1.0, n_questions=1,
                        eval_wall_s=2.0, question_results=[{"qid": "q1"}],
                        details={"cost_tier_observed_count": 1, "reliability_observed_count": 1})

    class Tower:
        def hybrid_eval(self):
            return result

    monkeypatch.setattr(autopilot, "apply_params", lambda _: {"status": "ok"})
    state: dict = {}
    autopilot.dispatch_action(
        {"type": "numeric_trial", "surface": surface, "params": dict(params or ONE)},
        seeder=None, swarm=swarm, forge=None, lab=None, tower=Tower(), gate=gate,
        archive=None, journal=None, state=state)
    return swarm, gate, result, state


def test_flag_off_planner_trial_never_reaches_the_study(monkeypatch, tmp_path):
    from actions import finalize_numeric_trial, run_numeric_gate_check

    swarm, gate, result, state = _dispatch(monkeypatch, tmp_path, flag=False)
    assert "_current_planner_numeric_trial" not in state
    verdict, obs = run_numeric_gate_check(state, gate, result)
    finalize_numeric_trial(state, swarm, result, verdict, gate, observation=obs)
    assert len(swarm._get_study(ONE_SURFACE).trials) == 0


def test_flag_on_completed_planner_trial_is_told_to_the_study(monkeypatch, tmp_path):
    from actions import finalize_numeric_trial, run_numeric_gate_check

    swarm, gate, result, state = _dispatch(monkeypatch, tmp_path, flag=True)
    assert state["_current_planner_numeric_trial"]["surface"] == ONE_SURFACE
    verdict, obs = run_numeric_gate_check(state, gate, result)
    finalize_numeric_trial(state, swarm, result, verdict, gate, observation=obs)
    trials = swarm._get_study(ONE_SURFACE).trials
    assert len(trials) == 1 and trials[0].user_attrs["source"] == "planner"
    assert trials[0].params == ONE
    assert "_current_planner_numeric_trial" not in state


def test_stale_marker_is_never_told_against_another_trials_result(monkeypatch, tmp_path):
    from actions import finalize_numeric_trial
    from safety_gate import EvalResult, SafetyVerdict

    swarm = _swarm(tmp_path)
    monkeypatch.setenv(ENV, "1")
    state = {"_current_planner_numeric_trial": {"surface": SURFACE, "params": dict(FULL)}}
    other = EvalResult(tier=1, quality=2.0, speed=20.0, cost=0.2, reliability=1.0,
                       details={"cost_tier_observed_count": 1, "reliability_observed_count": 1,
                                "numeric_trial_applied_params": {"think_harder.cot_roi_threshold": 0.5}})
    finalize_numeric_trial(state, swarm, other, SafetyVerdict(passed=True), None)
    assert len(swarm._get_study(SURFACE).trials) == 0 and not state


def test_excluded_or_unqualified_planner_trials_are_dropped_not_injected(monkeypatch, tmp_path):
    from actions import finalize_numeric_trial
    from safety_gate import EvalResult, SafetyVerdict

    swarm = _swarm(tmp_path)
    for kw, details in (
        ({"excluded": True}, {"cost_tier_observed_count": 1, "reliability_observed_count": 1}),
        ({}, {}),  # no qualified cost/reliability provenance
    ):
        r = EvalResult(tier=1, quality=2.0, speed=20.0, cost=0.2, reliability=1.0,
                       details={**details, "numeric_trial_applied_params": dict(FULL)})
        state = {"_current_planner_numeric_trial": {"surface": SURFACE, "params": dict(FULL)}}
        finalize_numeric_trial(state, swarm, r, SafetyVerdict(passed=True), None, **kw)
        assert not state
    assert len(swarm._get_study(SURFACE).trials) == 0


def test_multi_param_planner_action_is_skipped_by_ap9_so_never_reaches_the_study(monkeypatch, tmp_path):
    """Documents the limit: a multi-param explicit trial never evaluates (AP-9), so there is
    nothing to tell; partial vectors on multi-param surfaces need a live-value fill (open)."""
    swarm, gate, result, state = _dispatch(monkeypatch, tmp_path, flag=True, params=FULL,
                                           surface=SURFACE)
    assert "_current_planner_numeric_trial" not in state
    assert len(swarm._get_study(SURFACE).trials) == 0
