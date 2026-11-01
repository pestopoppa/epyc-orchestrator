"""The sequential QPH evidence observes the ratified speed-era boundary."""
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))
import autopilot  # noqa: E402
from safety_gate import EvalResult, SafetyGate  # noqa: E402
from experiment_journal import ExperimentJournal, JournalEntry  # noqa: E402

ACTION = {"type": "seed_batch"}
CANDIDATE = autopilot._config_fingerprint(ACTION)
BOUNDARY = datetime(2026, 8, 10, tzinfo=timezone.utc).timestamp()


def entry(tid, timestamp, wall=1800, witness=True):
    row = SimpleNamespace(
        trial_id=tid, timestamp=timestamp, tier=1, bug_corrupted_by="",
        outcome_status="ok", pareto_status="frontier", keep_revert_decision="",
        quality=3.0, eval_details={"eval_wall_s": wall,
            "question_results": [{"qid": "q1", "correct": True}]},
        seq={"candidate": CANDIDATE, "core_id": "core_v1", "z": 0.2,
             "z_rate": 0.1, "state": "accumulating"},
    )
    if witness:
        row.seq["rate_era_witness"] = {"version": 1, "speed_exclude_before_ts": BOUNDARY,
                                       "quality_exclude_before_ts": None}
    return row


def inputs(rows, **kwargs):
    # The reference helper also reads the journal; leave that unrelated state empty.
    class Journal:
        def entries_with_supersessions(self):
            return rows
        def entries(self):
            return rows
    return autopilot._seq_inputs_for_trial(journal=Journal(), action=ACTION, tier=1, **kwargs)


def test_speed_fence_independent_of_quality_and_includes_boundary():
    rows = [entry(1, "2026-08-09T00:00:00Z", 3600),
            entry(2, "bad timestamp", 3600),
            entry(3, "2026-08-08T00:00:00Z", 3600),
            entry(4, "2026-08-07T00:00:00Z", 3600),
            entry(9, "2026-08-06T00:00:00Z", 3600),
            entry(5, "2026-08-10T00:00:00Z"),
            entry(6, "2026-08-11T00:00:00Z"),
            entry(7, "2026-08-12T00:00:00Z"),
            entry(8, "2026-08-13T00:00:00Z")]
    old = inputs(rows)
    fenced = inputs(rows, rate_exclude_before_ts=BOUNDARY)
    assert fenced["baseline_task_rate"] == pytest.approx(2)
    assert old["baseline_task_rate"] == pytest.approx(1)
    assert [x[0] for x in fenced["prior_rate_obs"]] == [5, 6, 7, 8]
    assert fenced["prior_quality_obs"] == old["prior_quality_obs"]
    assert fenced["baseline_profile"] == old["baseline_profile"]
    assert fenced["alpha_wealth"] == old["alpha_wealth"]
    for row in rows:
        row.seq["rate_era_witness"]["quality_exclude_before_ts"] = BOUNDARY - 86400
        row.seq["quality_era_witness"] = {"version": 1, "quality_exclude_before_ts": BOUNDARY - 86400}
    speed_later = inputs(rows, quality_exclude_before_ts=BOUNDARY - 86400,
                         rate_exclude_before_ts=BOUNDARY)
    assert [x[0] for x in speed_later["prior_rate_obs"]] == [5, 6, 7, 8]
    assert speed_later["baseline_task_rate"] == pytest.approx(2)
    assert len(speed_later["prior_quality_obs"]) == 5
    for row in rows:
        row.seq["rate_era_witness"]["quality_exclude_before_ts"] = BOUNDARY + 86400
        row.seq["quality_era_witness"]["quality_exclude_before_ts"] = BOUNDARY + 86400
    quality_later = inputs(rows, quality_exclude_before_ts=BOUNDARY + 86400,
                           rate_exclude_before_ts=BOUNDARY)
    assert [x[0] for x in quality_later["prior_rate_obs"]] == [6, 7, 8]
    assert quality_later["baseline_task_rate"] == pytest.approx(2)
    assert len(quality_later["prior_quality_obs"]) == 3


def test_fenced_prior_rate_rebuilds_gate_count_and_thin_pool_skips(tmp_path):
    rows = [entry(1, "2026-08-09T00:00:00Z"),
            entry(2, "2026-08-10T00:00:00Z"),
            entry(3, "2026-08-11T00:00:00Z"),
            entry(4, "2026-08-12T00:00:00Z")]
    fenced = inputs(rows, rate_exclude_before_ts=BOUNDARY)
    result = EvalResult(tier=1, quality=3, speed=10, cost=0.2, reliability=1,
                        n_questions=1, question_results=[{"qid": "q1", "correct": True}],
                        eval_wall_s=1800)
    def check(value):
        result.gate_verdict = None
        gate = SafetyGate(baseline_path=tmp_path / "absent.yaml", use_sequential=True)
        return gate.check(result, question_results=result.question_results,
            baseline_profile=value["baseline_profile"], task_rate=2.0,
            baseline_task_rate=value["baseline_task_rate"],
            prior_quality_obs=value["prior_quality_obs"], prior_rate_obs=value["prior_rate_obs"],
            candidate=CANDIDATE, core_id="core_v1").seq
    seq = check(fenced)
    assert seq["k_rate"] == 4
    autopilot._seq_attach_rate_era_witness(SimpleNamespace(seq=seq), fenced)
    assert seq["rate_era_witness"] == fenced["rate_era_witness"]
    journal = ExperimentJournal(journal_dir=tmp_path / "journal")
    journal.record(JournalEntry(trial_id=5, timestamp="2026-08-13T00:00:00Z", species="test",
        action_type="seed_batch", tier=1, quality=3, speed=10, cost=0.2, reliability=1,
        pareto_status="frontier", config_snapshot=ACTION, eval_details=rows[0].eval_details, seq=seq))
    reread = ExperimentJournal(journal_dir=tmp_path / "journal")
    rebuilt = autopilot._seq_inputs_for_trial(journal=reread, action=ACTION, tier=1,
                                               rate_exclude_before_ts=BOUNDARY)
    assert len(rebuilt["prior_rate_obs"]) == 1
    thin = inputs(rows, rate_exclude_before_ts=BOUNDARY + 86400)
    seq = check(thin)
    assert seq["rate_axis_available"] is False
    autopilot._seq_attach_rate_era_witness(SimpleNamespace(seq=seq), thin)
    assert "rate_era_witness" not in seq
    assert seq["rate_axis_skip_reason"] == "incumbent_task_rate_comparator_unavailable"


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -1])
def test_invalid_cutoffs_fail_closed(bad):
    with pytest.raises(ValueError, match="invalid sequential era cutoff"):
        inputs([], rate_exclude_before_ts=bad)
    with pytest.raises(ValueError):
        autopilot._seq_speed_cutoff_from_state({"active_instrument_eras": {"autopilot_speed": "x"},
            "pareto_epoch_ts": 1, "pareto_exclude_before_ts": bad})


def test_legacy_and_mismatched_witnesses_do_not_rebuild_rate():
    rows = [entry(1, "2026-08-10T00:00:00Z", witness=False),
            entry(2, "2026-08-11T00:00:00Z"), entry(3, "2026-08-12T00:00:00Z")]
    rows[1].seq["rate_era_witness"]["quality_exclude_before_ts"] = 1.0
    rows[2].seq["rate_era_witness"]["version"] = True
    fenced = inputs(rows, rate_exclude_before_ts=BOUNDARY)
    assert fenced["prior_rate_obs"] == []
    assert len(fenced["prior_quality_obs"]) == 3
    assert len(inputs(rows)["prior_rate_obs"]) == 3


def test_native_journal_roundtrip_retains_rate_witness(tmp_path):
    row = entry(1, "2026-08-10T00:00:00Z")
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(JournalEntry(trial_id=1, timestamp=row.timestamp, species="test",
        action_type="seed_batch", tier=1, quality=3.0, speed=10.0, cost=0.2,
        reliability=1.0, pareto_status="frontier", config_snapshot=ACTION,
        eval_details=row.eval_details, seq=row.seq))
    read = ExperimentJournal(journal_dir=tmp_path)
    got = autopilot._seq_inputs_for_trial(journal=read, action=ACTION, tier=1,
                                         rate_exclude_before_ts=BOUNDARY)
    assert got["prior_rate_obs"] == [(1, 0.1)]


@pytest.mark.parametrize("seq", [
    {"rate_axis_available": False, "z_rate": 0.1},
    {"rate_axis_available": True, "z_rate": float("nan")},
    {"rate_axis_available": True, "z_rate": True},
    {"rate_axis_available": True},
])
def test_unmeasured_rate_is_not_stamped(seq):
    autopilot._seq_attach_rate_era_witness(SimpleNamespace(seq=seq), inputs([]))
    assert "rate_era_witness" not in seq


def test_action_to_main_cache_uses_current_era_and_keeps_witness(tmp_path):
    import actions
    rows = [entry(i, "2026-08-09T00:00:00Z", 3600, witness=False) for i in range(1, 5)]
    rows += [entry(i, "2026-08-10T00:00:00Z") for i in range(5, 8)]
    journal = SimpleNamespace(entries_with_supersessions=lambda: rows)
    state = {"active_instrument_eras": {"autopilot_speed": "existing-era"},
             "pareto_epoch_ts": BOUNDARY, "pareto_exclude_before_ts": BOUNDARY}
    gate = SafetyGate(baseline_path=tmp_path / "absent.yaml", use_sequential=True)
    result = EvalResult(tier=1, quality=3, speed=10, cost=0.2, reliability=1,
        n_questions=1, question_results=[{"qid": "q1", "correct": True}], eval_wall_s=1800)
    verdict = actions._action_gate_check(ACTION, SimpleNamespace(gate=gate, state=state,
                                                               journal=journal), result)
    assert verdict.seq["baseline_task_rate_qph"] == pytest.approx(2)
    assert verdict.seq["k_rate"] == 4
    witness = verdict.seq["rate_era_witness"].copy()
    profile = autopilot._seq_inputs_for_trial(journal=journal, action=ACTION, tier=1,
                                               rate_exclude_before_ts=BOUNDARY)
    assert autopilot._seq_prepare_cached_verdict(result, profile) is False
    again, _ = autopilot.run_numeric_gate_check(state, gate, result,
        question_results=result.question_results, task_rate=2,
        baseline_profile=profile["baseline_profile"], baseline_task_rate=profile["baseline_task_rate"],
        prior_quality_obs=profile["prior_quality_obs"], prior_rate_obs=profile["prior_rate_obs"],
        candidate=profile["candidate"], core_id=profile["core_id"])
    assert again is verdict
    assert again.seq["rate_era_witness"] == witness


def test_stale_cache_recomputes_or_retains_failed_verdict_without_seq(tmp_path):
    from safety_gate import SafetyVerdict
    rows = [entry(i, "2026-08-10T00:00:00Z") for i in range(1, 4)]
    profile = inputs(rows, rate_exclude_before_ts=BOUNDARY)
    gate = SafetyGate(baseline_path=tmp_path / "absent.yaml", use_sequential=True)
    result = EvalResult(tier=1, quality=3, speed=10, cost=0.2, reliability=1,
        n_questions=1, question_results=[{"qid": "q1", "correct": True}], eval_wall_s=1800)
    legacy = {"confirmed": True, "z_rate": 0.7}
    result.gate_verdict = SafetyVerdict(passed=True, seq=legacy)
    assert autopilot._seq_prepare_cached_verdict(result, profile) is True
    verdict, _ = autopilot.run_numeric_gate_check({}, gate, result,
        question_results=result.question_results, task_rate=2,
        baseline_profile=profile["baseline_profile"], baseline_task_rate=profile["baseline_task_rate"],
        prior_quality_obs=profile["prior_quality_obs"], prior_rate_obs=profile["prior_rate_obs"],
        candidate=profile["candidate"], core_id=profile["core_id"])
    autopilot._seq_attach_rate_era_witness(verdict, profile)
    assert verdict.seq is not legacy
    assert verdict.seq["rate_era_witness"] == profile["rate_era_witness"]
    assert len(gate._history_for_tier(1)) == 0
    failed = SafetyVerdict(passed=False, seq={"confirmed": True, "z_rate": 0.7})
    result.gate_verdict = failed
    assert autopilot._seq_prepare_cached_verdict(result, profile) is True
    final = gate.check(result, question_results=result.question_results,
                       baseline_profile=profile["baseline_profile"])
    assert final.passed is False and final.seq is None


def test_action_cache_new_speed_boundary_recomputes_without_double_count(tmp_path):
    import actions
    from safety_gate import SafetyVerdict
    rows = [entry(i, "2026-08-10T00:00:00Z") for i in range(1, 4)]
    rows += [entry(i, "2026-08-11T00:00:00Z", wall=900) for i in range(4, 7)]
    journal = SimpleNamespace(entries_with_supersessions=lambda: rows)
    state = {"active_instrument_eras": {"autopilot_speed": "existing-era"},
             "pareto_epoch_ts": BOUNDARY, "pareto_exclude_before_ts": BOUNDARY}
    gate = SafetyGate(baseline_path=tmp_path / "absent.yaml", use_sequential=True)
    result = EvalResult(tier=1, quality=3, speed=10, cost=0.2, reliability=1,
        n_questions=1, question_results=[{"qid": "q1", "correct": True}], eval_wall_s=900)
    prior = actions._action_gate_check(ACTION, SimpleNamespace(gate=gate, state=state,
                                                              journal=journal), result)
    assert isinstance(prior, SafetyVerdict) and prior.seq is not None
    old_seq = prior.seq.copy()
    history_before = len(gate._history_for_tier(1))
    failures_before = gate._consecutive_failures
    new_boundary = BOUNDARY + 86400
    profile = autopilot._seq_inputs_for_trial(journal=journal, action=ACTION, tier=1,
                                               rate_exclude_before_ts=new_boundary)
    assert autopilot._seq_prepare_cached_verdict(result, profile) is True
    verdict, _ = autopilot.run_numeric_gate_check({}, gate, result,
        question_results=result.question_results, task_rate=4,
        baseline_profile=profile["baseline_profile"], baseline_task_rate=profile["baseline_task_rate"],
        prior_quality_obs=profile["prior_quality_obs"], prior_rate_obs=profile["prior_rate_obs"],
        candidate=profile["candidate"], core_id=profile["core_id"])
    autopilot._seq_attach_rate_era_witness(verdict, profile)
    assert prior.seq == old_seq
    assert verdict is not prior and verdict.seq["baseline_task_rate_qph"] == pytest.approx(4)
    assert verdict.seq["k_rate"] == 1
    assert verdict.seq["rate_era_witness"]["speed_exclude_before_ts"] == new_boundary
    assert len(gate._history_for_tier(1)) == history_before
    assert gate._consecutive_failures == failures_before


def test_thin_new_rate_era_cannot_confirm_cached_old_verdict(tmp_path):
    from safety_gate import SafetyVerdict
    rows = [entry(i, "2026-08-10T00:00:00Z") for i in range(1, 4)]
    rows += [entry(4, "2026-08-11T00:00:00Z")]
    profile = inputs(rows, rate_exclude_before_ts=BOUNDARY + 86400)
    assert profile["baseline_profile"] and profile["baseline_task_rate"] is None
    gate = SafetyGate(baseline_path=tmp_path / "absent.yaml", use_sequential=True)
    result = EvalResult(tier=1, quality=3, speed=10, cost=0.2, reliability=1,
        n_questions=1, question_results=[{"qid": "q1", "correct": True}], eval_wall_s=1800,
        gate_verdict=SafetyVerdict(passed=True, seq={"confirmed": True, "z_rate": 0.7}))
    assert autopilot._seq_prepare_cached_verdict(result, profile)
    verdict, _ = autopilot.run_numeric_gate_check({}, gate, result,
        question_results=result.question_results, task_rate=2,
        baseline_profile=profile["baseline_profile"], baseline_task_rate=profile["baseline_task_rate"],
        prior_quality_obs=profile["prior_quality_obs"], prior_rate_obs=profile["prior_rate_obs"],
        candidate=profile["candidate"], core_id=profile["core_id"])
    assert verdict.seq["rate_axis_available"] is False
    assert verdict.seq["confirmed"] is False
    assert "rate_era_witness" not in verdict.seq


def test_oversized_integer_witness_and_cutoff_fail_closed():
    row = entry(1, "2026-08-10T00:00:00Z")
    row.seq["rate_era_witness"]["speed_exclude_before_ts"] = 10 ** 1000
    result = inputs([row], rate_exclude_before_ts=BOUNDARY)
    assert result["prior_rate_obs"] == []
    assert len(result["prior_quality_obs"]) == 1
    with pytest.raises(ValueError, match="invalid sequential era cutoff"):
        inputs([], rate_exclude_before_ts=10 ** 1000)
    with pytest.raises(ValueError, match="invalid sequential era cutoff"):
        autopilot._seq_speed_cutoff_from_state({"active_instrument_eras": {"autopilot_speed": "x"},
            "pareto_epoch_ts": 1, "pareto_exclude_before_ts": 10 ** 1000})
