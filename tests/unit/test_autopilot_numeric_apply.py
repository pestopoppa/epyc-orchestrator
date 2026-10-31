"""Tests for NumericSwarm application/evaluation ordering."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

autopilot = importlib.import_module("scripts.autopilot.autopilot")
experiment_journal = importlib.import_module("scripts.autopilot.experiment_journal")
numeric_swarm = importlib.import_module("scripts.autopilot.species.numeric_swarm")


class FailingTower:
    def hybrid_eval(self):  # pragma: no cover - must not be called
        raise AssertionError("eval should be skipped when params fail to apply")


class FakeSwarm:
    def __init__(self) -> None:
        self.failed: tuple[str, int, str] | None = None
        self.infeasible = None

    def suggest_trial(self, surface: str) -> dict[str, object]:
        return {
            "trial_number": 7,
            "surface": surface,
            "params": {"think_harder.min_expected_roi": 0.05},
        }

    def mark_failed(self, surface: str, trial_number: int, reason: str = "") -> None:
        self.failed = (surface, trial_number, reason)

    def report_infeasible(self, surface, trial_number, *, outcome_class, reason, **kwargs):
        self.infeasible = (surface, trial_number, outcome_class, reason)


def test_chat_pipeline_numeric_surface_uses_runtime_q_threshold() -> None:
    names = [spec.name for spec in numeric_swarm.SURFACES["chat_pipeline"]]

    assert names == ["chat.try_cheap_first_q_threshold"]
    assert "chat.try_cheap_first_quality_threshold" not in names


def test_approved_chat_threshold_surfaces_have_operator_ranges() -> None:
    surfaces = numeric_swarm.SURFACES

    long_context = surfaces["chat_long_context"][0]
    assert long_context.name == "chat.long_context_threshold_chars"
    assert long_context.param_type == "int"
    assert (long_context.low, long_context.high) == (12000, 80000)

    summarization = surfaces["chat_summarization"][0]
    assert summarization.name == "chat.summarization_threshold_tokens"
    assert summarization.param_type == "int"
    assert (summarization.low, summarization.high) == (3000, 30000)

    # RI-18c removed the answer review gate and its chat.review_low_q_threshold surface.
    assert "chat_review_low" not in surfaces

    review_skip = surfaces["chat_review_skip"][0]
    assert review_skip.name == "chat.review_skip_q_threshold"
    assert (review_skip.low, review_skip.high) == (0.45, 0.80)


def test_gpu_placement_policy_surface_exposes_thresholds_not_enable_flag() -> None:
    specs = numeric_swarm.SURFACES["gpu_placement_policy"]
    by_name = {spec.name: spec for spec in specs}

    assert "placement_policy.teleport_enabled" not in by_name
    assert by_name["placement_policy.long_running_trigger_tokens"].param_type == "int"
    assert by_name["placement_policy.rate_window_tokens"].param_type == "int"
    assert by_name["placement_policy.min_resident_remaining_tokens"].param_type == "int"
    assert by_name["placement_policy.min_speedup"].param_type == "float"
    assert by_name["placement_policy.lease_interactive_weight"].param_type == "float"
    assert by_name["placement_policy.lease_batch_weight"].param_type == "float"
    assert by_name["placement_policy.lease_eval_weight"].param_type == "float"


def test_spec_decode_role_restart_surface_exposes_numeric_only() -> None:
    specs = numeric_swarm.SURFACES["spec_decode_role_restart"]
    by_name = {spec.name: spec for spec in specs}

    assert by_name["role_restart.frontdoor_draft_max"].param_type == "int"
    assert by_name["role_restart.frontdoor_draft_min"].param_type == "int"
    assert by_name["role_restart.frontdoor_draft_p_min"].param_type == "float"
    assert by_name["role_restart.frontdoor_draft_p_split"].param_type == "float"
    assert by_name["role_restart.frontdoor_ngram_mod_n_min"].param_type == "int"
    assert by_name["role_restart.frontdoor_ngram_mod_n_max"].param_type == "int"
    assert by_name["role_restart.frontdoor_ngram_mod_n_match"].param_type == "int"
    assert by_name["role_restart.worker_draft_max"].param_type == "int"
    assert by_name["role_restart.worker_draft_min"].param_type == "int"
    assert by_name["role_restart.worker_draft_p_min"].param_type == "float"
    assert by_name["role_restart.worker_draft_p_split"].param_type == "float"
    assert by_name["role_restart.worker_threads_draft"].param_type == "int"
    assert by_name["role_restart.worker_ngram_mod_n_min"].param_type == "int"
    assert by_name["role_restart.worker_ngram_mod_n_max"].param_type == "int"
    assert by_name["role_restart.worker_ngram_mod_n_match"].param_type == "int"
    assert by_name["role_restart.architect_draft_max"].param_type == "int"
    assert by_name["role_restart.architect_draft_min"].param_type == "int"
    assert by_name["role_restart.architect_draft_p_min"].param_type == "float"
    assert by_name["role_restart.architect_draft_p_split"].param_type == "float"
    assert by_name["role_restart.architect_ngram_mod_n_min"].param_type == "int"
    assert by_name["role_restart.architect_ngram_mod_n_max"].param_type == "int"
    assert by_name["role_restart.architect_ngram_mod_n_match"].param_type == "int"
    assert "role_restart.worker_spec_type" not in by_name
    assert "role_restart.worker_kv_profile" not in by_name


def test_suggested_numeric_trial_env_restart_skips_without_marking_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        autopilot,
        "apply_params",
        lambda _params: {"status": "error", "errors": ["env_restart: reload failed"]},
    )
    swarm = FakeSwarm()

    result, species = autopilot.dispatch_action(
        {"type": "numeric_trial", "surface": "think_harder"},
        seeder=None,
        swarm=swarm,
        forge=None,
        lab=None,
        tower=FailingTower(),
        gate=None,
        archive=None,
        journal=None,
        state={},
    )

    assert isinstance(result, autopilot.SkipOutcome)
    assert result.status == "skipped"
    assert "env_restart: reload failed" in result.reason
    assert result.bug_corrupted_by == "env_restart_apply_failure"
    assert species == "numeric_swarm"
    assert swarm.failed == ("think_harder", 7, "env_restart: reload failed")
    assert swarm.infeasible is None


def test_suggested_numeric_trial_non_infra_apply_error_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        autopilot,
        "apply_params",
        lambda _params: {"status": "error", "errors": ["hot_swap: rejected"]},
    )
    swarm = FakeSwarm()

    result, species = autopilot.dispatch_action(
        {"type": "numeric_trial", "surface": "think_harder"},
        seeder=None,
        swarm=swarm,
        forge=None,
        lab=None,
        tower=FailingTower(),
        gate=None,
        archive=None,
        journal=None,
        state={},
    )

    assert isinstance(result, autopilot.SkipOutcome)
    assert result.status == "skipped"
    assert result.bug_corrupted_by == ""
    assert "hot_swap: rejected" in result.reason
    assert species == "numeric_swarm"
    assert swarm.failed == ("think_harder", 7, "hot_swap: rejected")
    assert swarm.infeasible is None


def test_explicit_numeric_trial_failure_skips_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        autopilot,
        "apply_params",
        lambda _params: {"status": "error", "errors": ["unknown_params: x"]},
    )

    result, species = autopilot.dispatch_action(
        {
            "type": "numeric_trial",
            "surface": "think_harder",
            "params": {"think_harder.min_expected_roi": 0.05},
        },
        seeder=None,
        swarm=FakeSwarm(),
        forge=None,
        lab=None,
        tower=FailingTower(),
        gate=None,
        archive=None,
        journal=None,
        state={},
    )

    assert isinstance(result, autopilot.SkipOutcome)
    assert result.status == "invalid"
    assert "unknown_params: x" in result.reason
    assert species == "numeric_swarm"


def test_bug_corrupted_skip_trial_journals_contamination_tag(tmp_path: Path) -> None:
    journal = experiment_journal.ExperimentJournal(journal_dir=tmp_path)

    autopilot._record_skip_trial(
        journal,
        42,
        {"type": "numeric_trial", "surface": "memrl_retrieval"},
        "numeric_swarm",
        "skipped",
        "numeric_trial params failed to apply: env_restart: reload failed",
        123,
        bug_corrupted_by="env_restart_apply_failure",
        bug_corrupted_reason="reload failed before params applied",
    )

    entry = experiment_journal.ExperimentJournal(journal_dir=tmp_path).all_entries()[0]
    assert entry.outcome_status == "skipped"
    assert entry.bug_corrupted_by == "env_restart_apply_failure"
    assert entry.bug_corrupted_reason == "reload failed before params applied"


def test_native_dispatch_and_gate_tell_real_study(monkeypatch, tmp_path):
    from actions import finalize_numeric_trial, run_numeric_gate_check
    from safety_gate import EvalResult, SafetyGate

    swarm = numeric_swarm.NumericSwarm(db_path=tmp_path / 'numeric.db')
    gate = SafetyGate(baseline_path=tmp_path / 'absent.yaml')
    state = {}
    result = EvalResult(tier=1, quality=0.5, speed=20., cost=0.2, reliability=1.,
                        n_questions=1, eval_wall_s=2., question_results=[{'qid': 'q1'}],
                        details={'cost_tier_observed_count': 1, 'reliability_observed_count': 1})
    class Tower:
        def hybrid_eval(self):
            return result

    monkeypatch.setattr(autopilot, 'apply_params', lambda _: {'status': 'ok'})
    observed, _ = autopilot.dispatch_action(
        {'type': 'numeric_trial', 'surface': 'think_harder'}, seeder=None, swarm=swarm,
        forge=None, lab=None, tower=Tower(), gate=gate, archive=None, journal=None, state=state)
    assert observed is result
    study = swarm._get_study('think_harder')
    assert study.trials[0].state.name == 'RUNNING'
    verdict, observation = run_numeric_gate_check(state, gate, result)
    assert not verdict and observation
    finalize_numeric_trial(state, swarm, result, verdict, gate, observation=observation)
    trial = study.trials[0]
    assert trial.state.name == 'COMPLETE'
    assert trial.user_attrs['outcome_class'] == 'slo_miss'
    assert trial.user_attrs['objective_source'] == 'native_qualified'
    assert 0 < trial.user_attrs['constraint_violation'] < 1
    assert trial.user_attrs['overage_cap_pairs']
    assert trial.values[1] == pytest.approx(1800.)
    assert not state


def test_native_dispatch_infra_apply_is_terminal_pruned(monkeypatch, tmp_path):
    swarm = numeric_swarm.NumericSwarm(db_path=tmp_path / 'numeric.db')
    monkeypatch.setattr(autopilot, 'apply_params', lambda _: {
        'status': 'error', 'errors': ['env_restart: reload failed']})
    autopilot.dispatch_action({'type': 'numeric_trial', 'surface': 'think_harder'},
        seeder=None, swarm=swarm, forge=None, lab=None, tower=FailingTower(),
        gate=None, archive=None, journal=None, state={})
    trial = swarm._get_study('think_harder').trials[0]
    assert trial.state.name == 'PRUNED'
    assert 'env_restart' in trial.user_attrs['outcome_reason']


def test_native_dispatch_invalid_config_waits_then_flushes(monkeypatch, tmp_path):
    swarm = numeric_swarm.NumericSwarm(db_path=tmp_path / 'numeric.db')
    monkeypatch.setattr(autopilot, 'apply_params', lambda _: {
        'status': 'error', 'unknown_params': ['think_harder.min_expected_roi'],
        'hot_swap_result': {'status': 'no_changes'},
        'errors': ['unknown_params: think_harder.min_expected_roi']})
    autopilot.dispatch_action({'type': 'numeric_trial', 'surface': 'think_harder'},
        seeder=None, swarm=swarm, forge=None, lab=None, tower=FailingTower(),
        gate=None, archive=None, journal=None, state={})
    study = swarm._get_study('think_harder')
    assert study.trials[0].state.name == 'RUNNING'
    assert study.trials[0].user_attrs['outcome_class'] == 'preflight'
    t = swarm.suggest_trial('think_harder')
    swarm.report_result('think_harder', t['trial_number'], (2., 10., -0.1, 1.),
                        cost_tier_observed_count=1, reliability_observed_count=1)
    assert study.trials[0].state.name == 'COMPLETE'
    assert study.trials[0].user_attrs['objective_source'] == 'synthetic_worst_observed'


def test_unmeasured_axis_without_causal_failure_is_pruned(tmp_path):
    from actions import finalize_numeric_trial
    from safety_gate import EvalResult, SafetyVerdict
    swarm = numeric_swarm.NumericSwarm(db_path=tmp_path / 'numeric.db')
    t = swarm.suggest_trial('monitor')
    state = {'_current_optuna_trial': t}
    r = EvalResult(tier=1, quality=2., speed=10., cost=0.1, reliability=1.)
    finalize_numeric_trial(state, swarm, r, SafetyVerdict(passed=True), None)
    assert swarm._get_study('monitor').trials[0].state.name == 'PRUNED'
    assert not state


def test_constraint_observation_does_not_change_verdict_or_schema(tmp_path):
    import copy
    from dataclasses import asdict
    from safety_gate import EvalResult, SafetyGate
    def result():
        return EvalResult(tier=1, quality=.5, speed=20., cost=.2, reliability=1.)
    plain = SafetyGate(baseline_path=tmp_path/'absent.yaml').check(result())
    observed, pairs = SafetyGate(baseline_path=tmp_path/'absent.yaml').check_with_constraint_observation(result())
    assert asdict(plain) == asdict(observed) == asdict(copy.deepcopy(observed))
    assert type(observed.violations) is list
    assert pairs == [('quality_floor', .5, 1.)]


def test_constraint_observation_unknown_for_infra_and_failed_capture(tmp_path):
    from safety_gate import EvalResult, SafetyGate, quality_floor_leg
    result = EvalResult(tier=1, quality=.1, speed=20., cost=.2, reliability=.1)
    verdict, pairs = SafetyGate(baseline_path=tmp_path/'absent.yaml').check_with_constraint_observation(result)
    assert verdict.retry_not_revert and pairs is None
    def broken(*args):
        raise RuntimeError('observer failure')
    assert quality_floor_leg(.5, 1, observer=broken) == quality_floor_leg(.5, 1)


def test_low_support_uses_actual_binding_cap_for_observation():
    from safety_gate import per_suite_regression_leg
    args = ({'math': .2}, {'math': 3.}, {'math': 4}, {'math': 5})
    seen = []
    instrumented = per_suite_regression_leg(*args, observer=lambda *record: seen.append(record))
    plain = per_suite_regression_leg(*args)
    assert instrumented == plain and len(plain.violations) == 1
    assert seen[0][0] == 'per_suite:math'
    assert seen[0][1] == pytest.approx(.3)
    assert seen[0][2] == 2.5


def test_native_apply_exception_does_not_leave_running_trial(monkeypatch, tmp_path):
    swarm = numeric_swarm.NumericSwarm(db_path=tmp_path / 'numeric.db')
    def broken(_):
        raise OSError('endpoint unavailable')
    monkeypatch.setattr(autopilot, 'apply_params', broken)
    with pytest.raises(OSError):
        autopilot.dispatch_action({'type': 'numeric_trial', 'surface': 'think_harder'},
            seeder=None, swarm=swarm, forge=None, lab=None, tower=FailingTower(),
            gate=None, archive=None, journal=None, state={})
    assert swarm._get_study('think_harder').trials[0].state.name == 'PRUNED'


def test_objective_builder_exception_does_not_strand_trial(monkeypatch, tmp_path):
    import actions
    from safety_gate import EvalResult, SafetyVerdict
    swarm = numeric_swarm.NumericSwarm(db_path=tmp_path / 'numeric.db')
    t = swarm.suggest_trial('monitor')
    state = {'_current_optuna_trial': t}
    def broken(_):
        raise ValueError('broken evidence')
    monkeypatch.setattr(actions, 'objectives_from', broken)
    with pytest.raises(ValueError):
        actions.finalize_numeric_trial(state, swarm,
            EvalResult(tier=1, quality=2., speed=10., cost=.1, reliability=1.,
                       details={'cost_tier_observed_count': 1,
                                'reliability_observed_count': 1}),
            SafetyVerdict(passed=True), None)
    assert swarm._get_study('monitor').trials[0].state.name == 'PRUNED'
    assert not state


def test_task_failure_and_unmeasured_rate_are_nonanswer(monkeypatch, tmp_path):
    from actions import finalize_numeric_trial, run_numeric_gate_check
    from eval_tower import EvalTower, QuestionResult
    from safety_gate import SafetyGate
    swarm = numeric_swarm.NumericSwarm(db_path=tmp_path / 'numeric.db')
    state = {}
    row = QuestionResult(question_id='q', suite='math', prompt='p', expected='a',
                         correct=False, error='task failed', disposition='task_failed',
                         cost_tier=2, tokens_generated=0, elapsed_s=.1, eval_wall_s=.1)
    result = EvalTower()._aggregate([row], tier=1)
    assert result.task_failed_count == 1 and result.quality_measured is True
    class Tower:
        def hybrid_eval(self):
            return result
    monkeypatch.setattr(autopilot, 'apply_params', lambda _: {'status': 'ok'})
    gate = SafetyGate(baseline_path=tmp_path / 'absent.yaml')
    actual, _ = autopilot.dispatch_action({'type': 'numeric_trial', 'surface': 'monitor'},
        seeder=None, swarm=swarm, forge=None, lab=None, tower=Tower(), gate=gate,
        archive=None, journal=None, state=state)
    verdict, observation = run_numeric_gate_check(state, gate, actual)
    finalize_numeric_trial(state, swarm, actual, verdict, gate, observation=observation)
    trial = swarm._get_study('monitor').trials[0]
    assert trial.state.name == 'PRUNED'
    assert trial.user_attrs['outcome_class'] == 'unknown'
    assert trial.values is None


def test_native_eval_memory_error_records_infeasible(monkeypatch, tmp_path):
    class Tower:
        def hybrid_eval(self):
            raise MemoryError('evaluation OOM')
    swarm = numeric_swarm.NumericSwarm(db_path=tmp_path / 'numeric.db')
    monkeypatch.setattr(autopilot, 'apply_params', lambda _: {'status': 'ok'})
    with pytest.raises(MemoryError):
        autopilot.dispatch_action({'type': 'numeric_trial', 'surface': 'monitor'},
            seeder=None, swarm=swarm, forge=None, lab=None, tower=Tower(),
            gate=None, archive=None, journal=None, state={})
    trial = swarm._get_study('monitor').trials[0]
    assert trial.state.name == 'RUNNING'
    assert trial.user_attrs['anchor_queued'] and trial.user_attrs['outcome_class'] == 'runtime'


def test_missing_quality_qualification_and_unqualified_cost_are_nonanswers(tmp_path):
    from types import SimpleNamespace
    from actions import finalize_numeric_trial
    from safety_gate import EvalResult, SafetyVerdict
    swarm = numeric_swarm.NumericSwarm(db_path=tmp_path / 'numeric.db')
    legacy = SimpleNamespace(tier=1, quality=2., speed=10., cost=.5, reliability=1.,
        n_questions=1, eval_wall_s=2., question_results=[{'qid': 'q'}],
        details={'cost_tier_observed_count': 1, 'reliability_observed_count': 1})
    fallback = EvalResult(tier=1, quality=2., speed=10., cost=.5, reliability=1.,
        n_questions=1, eval_wall_s=2., question_results=[{'qid': 'q'}],
        details={'cost_tier_observed_count': 0, 'reliability_observed_count': 1})
    legacy_none = SimpleNamespace(**vars(legacy), quality_measured=None)
    unscored = EvalResult(tier=1, quality=0., speed=10., cost=.5, reliability=1.,
        quality_measured=False, n_questions=1, eval_wall_s=2.,
        question_results=[{'qid': 'q'}],
        details={'cost_tier_observed_count': 1, 'reliability_observed_count': 1})
    for result in (legacy, legacy_none, unscored, fallback):
        t = swarm.suggest_trial('monitor')
        state = {'_current_optuna_trial': t}
        finalize_numeric_trial(state, swarm, result, SafetyVerdict(passed=True), None)
        assert swarm._get_study('monitor').trials[t['trial_number']].state.name == 'PRUNED'
    assert swarm._anchor(swarm._get_study('monitor')) is None


def test_eval_tower_emits_actual_cost_and_reliability_provenance():
    from eval_tower import EvalTower, QuestionResult
    def row(tier):
        return QuestionResult(question_id='q', suite='math', prompt='p', expected='a',
                              correct=True, cost_tier=tier, tokens_generated=10,
                              elapsed_s=2., eval_wall_s=2.)
    absent = EvalTower()._aggregate([row(0)], tier=1)
    present = EvalTower()._aggregate([row(2)], tier=1)
    assert absent.cost == .5 and absent.details['cost_tier_observed_count'] == 0
    assert present.cost == .5 and present.details['cost_tier_observed_count'] == 1
    assert absent.details['reliability_observed_count'] == 1
    assert present.details['reliability_observed_count'] == 1
