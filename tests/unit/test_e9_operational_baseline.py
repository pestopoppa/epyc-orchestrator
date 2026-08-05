from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.autopilot import collect_e9_operational_baseline as collector


def _result(**overrides):
    values = {
        "tier": 1,
        "quality": 1.8,
        "speed": 55.0,
        "cost": 0.25,
        "reliability": 0.99,
        "per_suite_quality": {"general": 2.0},
        "per_suite_counts": {"general": 4},
        "n_questions": collector.EVAL_T1_SPEC_N,
        "eval_concurrency": 6,
        "speed_metric_mode": "aggregate_batch_tps",
        "details": {
            "eval_execution_instrument_id": collector.EVAL_EXECUTION_INSTRUMENT_ID,
            "eval_scoring_schedule_id": collector.EVAL_SCORING_SCHEDULE_ID,
            "task_rate_qph": 120.0,
            "eval_contaminated_by_abandoned_requests": False,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_candidate_is_fresh_t1_only_and_stamps_both_eras():
    candidate = collector.candidate_baseline_state(_result())

    assert candidate["baselines_by_tier"] == {"1": 1.8}
    assert candidate["per_suite_quality_by_tier"] == {"1": {"general": 2.0}}
    assert candidate["per_suite_counts_by_tier"] == {"1": {"general": 4}}
    assert candidate["frontdoor_speed"] == 55.0
    assert candidate["eval_quality_era"] == collector.QUALITY_ERA
    assert candidate["autopilot_speed_era"] == collector.SPEED_ERA


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eval_concurrency": 1}, "resource lanes inactive"),
        ({"speed_metric_mode": "median_request_tps"}, "speed_metric_mode"),
        ({"reliability": 0.79}, "reliability"),
        ({"n_questions": 50}, "n_questions"),
    ],
)
def test_result_gate_fails_closed(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        collector._validate_result(_result(**overrides))


def test_instrument_state_requires_ratified_e9_ids():
    state = {
        "pareto_objective_policy": collector.POLICY,
        "eval_execution_instrument_id": collector.EVAL_EXECUTION_INSTRUMENT_ID,
        "eval_scoring_schedule_id": collector.EVAL_SCORING_SCHEDULE_ID,
        "active_instrument_eras": {
            "eval_quality": collector.QUALITY_ERA,
            "autopilot_speed": collector.SPEED_ERA,
        },
    }
    collector._validate_instrument_state(state)

    state["eval_execution_instrument_id"] = "old"
    with pytest.raises(RuntimeError, match="execution_instrument"):
        collector._validate_instrument_state(state)
