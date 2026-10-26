"""SEQ-5/6/7, MHS-16 pure replays and EV-18 ITT helpers (read-only, synthetic journals)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.autopilot_core.itt_quality import itt_counts, itt_delta_flag  # noqa: E402
from src.autopilot_core.sequential_replay import (  # noqa: E402
    cumulative_drift_report,
    frozen_null_replay,
    paired_keep_replay,
    promotion_attempt_ledger,
)


def _qr(bits):
    return [{"qid": f"q{i}", "correct": bool(b)} for i, b in enumerate(bits)]


def _row(tid, bits, *, keep=True, tier=1, seq=None, action="numeric_trial", ts="2026-06-01T00:00:00+00:00"):
    return {
        "trial_id": tid, "tier": tier, "timestamp": ts, "action_type": action,
        "keep_revert_decision": "keep" if keep else "revert",
        "config_snapshot": {"type": action},
        "eval_details": {"question_results": _qr(bits), "per_suite_quality": {"s": float(sum(bits))}},
        **({"seq": seq} if seq else {}),
    }


def test_seq5_empty_journal_is_zero_and_lower_bound():
    out = promotion_attempt_ledger([])
    assert out["promotion_attempts_opened"] == 0
    assert out["promotion_attempts_opened_is_lower_bound"] is True
    assert out["promotion_alpha_per_attempt"] == 0.01
    assert out["rate_reference_is_latest_only"] is True


def test_seq5_counts_opened_and_finalized():
    rows = [
        {"trial_id": 1, "seq": {"candidate": "a", "confirmed": True, "state": "confirmed"}},
        {"trial_id": 2, "seq": {"candidate": "a", "baseline_promotion_fresh_eval": True,
                                "baseline_promotion_finalized": True,
                                "baseline_promotion_fresh_eval_for": {"candidate": "a", "source_trial_id": 1}}},
        {"trial_id": 3, "seq": {"candidate": "b", "confirmed": True}},
    ]
    out = promotion_attempt_ledger(rows)
    assert (out["promotion_attempts_opened"], out["promotion_attempts_finalized"]) == (2, 1)
    assert out["promotion_attempts_failed_or_interrupted"] == 1
    assert out["lifetime_false_promotion_bound"] == 0.02
    assert out["fingerprints_confirmed"] == 2


def test_seq7_flags_insignificant_keep_and_counts_missing_vectors():
    rows = [_row(1, [1, 0, 1, 0, 1, 0, 1, 0]), _row(2, [1, 0, 1, 0, 1, 0, 1, 1]),
            {"trial_id": 3, "tier": 1, "keep_revert_decision": "keep", "eval_details": {}}]
    out = paired_keep_replay(rows)
    assert out["keep_rows_total"] == 3
    assert out["keep_rows_without_question_results"] == 1
    assert out["keep_rows_without_predecessor_vector"] == 1
    assert out["false_elite_count"] == 1  # one gained pair: z = 1.0 < 1.96


def test_mhs16_flips_against_seed_and_bucket_runs():
    seed = [1, 1, 1, 1]
    rows = [_row(i, seed) for i in range(1, 4)] + [_row(4, [0, 1, 1, 1]), _row(5, [0, 0, 1, 1])]
    out = cumulative_drift_report(rows)
    t = out["tiers"]["1"]
    assert t["rows_compared"] == 2 and t["distinct_questions_ever_flipped"] == 2
    assert t["max_flips_in_one_row"] == 2
    assert t["same_bucket_concentration_warn"] is True


def test_seq6_replays_and_counts_not_replayable():
    base = [_row(i, [1, 0, 1, 1, 0, 1]) for i in range(1, 5)]
    cand = [
        _row(10 + k, [1, 1, 1, 1, 0, 1], keep=False,
             seq={"candidate": "c1", "core_id": "core_v1", "z": 0.1, "k": k + 1})
        for k in range(3)
    ]
    out = frozen_null_replay(base + cand)
    assert out["denominator_groups_with_seq_vector_rows"] == 1
    assert out["replayed_groups"] == 1
    assert "harness_validation" in out
    thin = frozen_null_replay([_row(1, [1, 0]), *cand])
    assert thin["replayed_groups"] == 0
    assert thin["not_replayable"]["fewer_than_3_baseline_rows_at_first_trial"] == 1


def test_itt_counts_and_delta_flag():
    c = itt_counts(correct=6, n_scored=8, parse_failure_count=2, scoring_failed_count=2)
    assert c["itt_denominator"] == 10 and abs(c["quality_itt"] - 1.8) < 1e-9
    capped = itt_counts(correct=6, n_scored=8, parse_failure_count=5, scoring_failed_count=1)
    assert capped["itt_denominator"] == 9
    base = {"correct": 50, "n_scored": 100, "itt_denominator": 100}
    cand = {"correct": 40, "n_scored": 70, "itt_denominator": 100}
    flag = itt_delta_flag(base, cand)
    assert flag["sign_differs"] is True and flag["requires_second_scorer_rescore"] is True
    assert itt_delta_flag({}, {})["comparable"] is False


def test_self_criticism_appends_itt_text_without_changing_decision():
    from safety_gate import EvalResult, SafetyVerdict
    from self_criticism import generate_self_criticism

    ev = EvalResult(
        tier=1, quality=1.5, speed=20.0, cost=0.5, reliability=1.0, n_questions=40,
        details={"correct": 40, "n_scored": 70, "itt_denominator": 100},
    )
    verdict = SafetyVerdict(passed=True)
    ref = {"correct": 50, "n_scored": 100, "itt_denominator": 100}
    with_ref = generate_self_criticism({"type": "x"}, ev, verdict, "", baseline_quality=1.0,
                                       reference_details=ref)
    without = generate_self_criticism({"type": "x"}, ev, verdict, "", baseline_quality=1.0)
    assert with_ref.keep_or_revert == without.keep_or_revert
    assert "SCORER-DEPENDENT" in with_ref.keep_revert_reasoning
    assert "ITT" not in without.keep_revert_reasoning
