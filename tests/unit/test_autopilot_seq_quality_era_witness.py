"""Native sequential quality provenance and cache fencing."""
from types import SimpleNamespace

import pytest

from tests.unit.test_autopilot_seq_rate_era_fence import (
    ACTION, BOUNDARY, CANDIDATE, EvalResult, ExperimentJournal, JournalEntry,
    SafetyGate, autopilot, entry, inputs,
)


def witness(cutoff=BOUNDARY):
    return {"version": 1, "quality_exclude_before_ts": cutoff}


@pytest.mark.parametrize("bad", [None, {}, {"version": True, "quality_exclude_before_ts": BOUNDARY},
    {"version": 1, "quality_exclude_before_ts": True},
    {"version": 1, "quality_exclude_before_ts": float("nan")},
    {"version": 1, "quality_exclude_before_ts": float("inf")},
    {"version": 1, "quality_exclude_before_ts": 10**1000}, witness(BOUNDARY + 1)])
def test_historical_quality_requires_native_matching_witness(bad):
    rows = [entry(i, "2026-08-11T00:00:00Z") for i in range(1, 5)]
    rows[0].seq["quality_era_witness"] = witness()
    rows[1].seq["quality_era_witness"] = bad
    fenced = inputs(rows, quality_exclude_before_ts=BOUNDARY)
    assert fenced["prior_quality_obs"] == [(1, 0.2)]
    assert len(inputs(rows)["prior_quality_obs"]) == 4
    assert fenced["baseline_profile"] == inputs(rows)["baseline_profile"]
    assert fenced["alpha_wealth"] == inputs(rows)["alpha_wealth"]


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 10**1000, "0.2"])
def test_malformed_quality_value_cannot_qualify_under_witness(bad):
    row = entry(1, "2026-08-11T00:00:00Z")
    row.seq.update(z=bad, quality_era_witness=witness())
    assert inputs([row], quality_exclude_before_ts=BOUNDARY)["prior_quality_obs"] == []


@pytest.mark.parametrize("seq", [{}, {"z": True}, {"z": float("nan")}, {"z": 10**1000}])
def test_no_quality_witness_without_computed_finite_axis(seq):
    autopilot._seq_attach_rate_era_witness(SimpleNamespace(seq=seq), inputs([]))
    assert "quality_era_witness" not in seq


def test_action_to_main_quality_cache_and_native_journal(tmp_path):
    import actions
    rows = [entry(i, "2026-08-11T00:00:00Z") for i in range(1, 4)]
    journal = SimpleNamespace(entries_with_supersessions=lambda: rows)
    state = {"active_instrument_eras": {"eval_quality": "test-era"},
             "quality_exclude_before_ts": BOUNDARY}
    gate = SafetyGate(baseline_path=tmp_path / "absent.yaml", use_sequential=True)
    result = EvalResult(tier=1, quality=3, speed=10, cost=0.2, reliability=1,
        n_questions=1, question_results=[{"qid": "q1", "correct": True}], eval_wall_s=1800)
    verdict = actions._action_gate_check(ACTION, SimpleNamespace(gate=gate, state=state,
                                                               journal=journal), result)
    assert verdict.seq["quality_era_witness"] == witness()
    assert verdict.seq["k"] == 1
    path = tmp_path / "journal"
    written = ExperimentJournal(journal_dir=path)
    written.record(JournalEntry(trial_id=4, timestamp="2026-08-12T00:00:00Z", species="test",
        action_type="seed_batch", tier=1, quality=3, speed=10, cost=0.2, reliability=1,
        pareto_status="frontier", config_snapshot=ACTION, eval_details=rows[0].eval_details,
        seq=verdict.seq))
    reread = ExperimentJournal(journal_dir=path)
    rebuilt = autopilot._seq_inputs_for_trial(journal=reread, action=ACTION, tier=1,
                                              quality_exclude_before_ts=BOUNDARY)
    assert rebuilt["prior_quality_obs"] == [(4, verdict.seq["z"])]
    profile = inputs(rows, quality_exclude_before_ts=BOUNDARY + 86400)
    history_before, failures_before = len(gate._history_for_tier(1)), gate._consecutive_failures
    assert autopilot._seq_prepare_cached_verdict(result, profile)
    fresh, _ = autopilot.run_numeric_gate_check(state, gate, result,
        question_results=result.question_results, task_rate=2,
        baseline_profile=profile["baseline_profile"], baseline_task_rate=profile["baseline_task_rate"],
        prior_quality_obs=profile["prior_quality_obs"], prior_rate_obs=profile["prior_rate_obs"],
        candidate=CANDIDATE, core_id=profile["core_id"])
    autopilot._seq_attach_rate_era_witness(fresh, profile)
    assert fresh is not verdict and fresh.seq["quality_era_witness"] == witness(BOUNDARY + 86400)
    assert (len(gate._history_for_tier(1)), gate._consecutive_failures) == (history_before, failures_before)


def test_failed_cache_is_retained_without_fabricated_witness(tmp_path):
    from safety_gate import SafetyVerdict
    profile = inputs([entry(i, "2026-08-11T00:00:00Z") for i in range(1, 4)],
                     quality_exclude_before_ts=BOUNDARY)
    result = EvalResult(tier=1, quality=3, speed=10, cost=0.2, reliability=1,
        n_questions=1, question_results=[{"qid": "q1", "correct": True}],
        gate_verdict=SafetyVerdict(passed=False, seq={"z": 0.2}))
    assert autopilot._seq_prepare_cached_verdict(result, profile)
    gate = SafetyGate(baseline_path=tmp_path / "absent.yaml", use_sequential=True)
    verdict = gate.check(result, question_results=result.question_results,
                         baseline_profile=profile["baseline_profile"])
    autopilot._seq_attach_rate_era_witness(verdict, profile)
    assert not verdict.passed and verdict.seq is None
