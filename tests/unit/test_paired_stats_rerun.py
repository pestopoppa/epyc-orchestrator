"""Rerun-arm flip rate + no-claim rule (AEI-RERUN-FLIP-CODE; P-AB-1 / P-PAIRED)."""

from __future__ import annotations

import json
import math

import pytest

from scripts.autopilot import paired_stats as ps

N = 100


def _vec(correct: list[bool]) -> dict[str, ps.QuestionOutcome]:
    return {
        f"q{i:03d}": ps.QuestionOutcome(qid=f"q{i:03d}", suite="s", correct=ok, trial_id=0)
        for i, ok in enumerate(correct)
    }


def _flipped(base: list[bool], idxs: range) -> list[bool]:
    return [(not v) if i in idxs else v for i, v in enumerate(base)]


# A: first 60 correct. B: A with 2 correct->wrong (b=2) and 12 wrong->correct (c=12).
A = [i < 60 for i in range(N)]
B_list = list(A)
for i in (0, 1):
    B_list[i] = False
for i in range(60, 72):
    B_list[i] = True
VA, VB = _vec(A), _vec(B_list)


def test_no_rerun_is_observation_only_refused_for_decision() -> None:
    """Absent R: a b_better/a_better is never returned as a usable verdict."""
    result = ps.mcnemar_from_vectors(VA, VB, "a", "b")
    assert ps.mcnemar_verdict(2, 12)["verdict"] == "b_better"  # raw stats intact
    with pytest.raises(ps.RerunArmMissingError, match="observation only"):
        ps.verdict_from_result(result)
    with pytest.raises(ps.RerunArmMissingError):
        ps.verdict_from_result(result, rerun=None)
    swapped = ps.mcnemar_from_vectors(VB, VA, "b", "a")
    assert ps.mcnemar_verdict(12, 2)["verdict"] == "a_better"
    with pytest.raises(ps.RerunArmMissingError):
        ps.verdict_from_result(swapped)
    # same family as the mismatched-pair refusal
    assert issubclass(ps.RerunArmMissingError, ValueError)
    assert issubclass(ps.PairedComparisonMismatchError, ValueError)


def test_wilson_interval_known_values() -> None:
    lo, hi = ps.wilson_interval(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-3)
    assert hi == pytest.approx(0.1744, abs=1e-3)
    assert ps.wilson_interval(0, 0) == (0.0, 1.0)
    lo0, hi0 = ps.wilson_interval(0, 50)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.1


def test_flip_rate_counts_and_exclusions() -> None:
    r = _flipped(A, range(0, 3))  # 3 A-correct -> R-wrong
    r2 = _flipped(r, range(60, 62))  # 2 A-wrong -> R-correct
    vr = _vec(r2)
    del vr["q099"]  # infra-failed in R: excluded and counted
    flip = ps.rerun_flip_rate(VA, vr)
    assert (flip["f_ar"], flip["f_ra"], flip["n_ar"]) == (3, 2, 99)
    assert flip["flip_rate"] == pytest.approx(5 / 99)
    assert flip["excluded_a_only"] == 1 and flip["excluded_r_only"] == 0
    assert flip["wilson95_low"] < flip["flip_rate"] < flip["wilson95_high"]


def test_downgrade_when_delta_within_flip_rate() -> None:
    # 12 flips of 100 -> flip_rate 0.12 >= |delta| 0.10
    vr = _vec(_flipped(A, range(0, 12)))
    flip = ps.rerun_flip_rate(VA, vr)
    assert flip["flip_rate"] == pytest.approx(0.12)
    result = ps.mcnemar_from_vectors(VA, VB, "a", "b")
    out = ps.verdict_from_result(result, rerun=flip)
    assert out["verdict"] == "indistinguishable"
    assert out["reason"] == "within_rerun_noise"
    assert out["verdict_before_rerun"] == "b_better"
    assert out["p_value"] == pytest.approx(0.01293945)  # p stays on record
    assert out["abs_delta"] == pytest.approx(0.10)
    assert out["discordant_rate"] == pytest.approx(0.14)
    assert out["rerun"]["n_ar"] == N


def test_tie_is_no_claim() -> None:
    vr = _vec(_flipped(A, range(0, 10)))  # flip_rate exactly 0.10 == |delta|
    flip = ps.rerun_flip_rate(VA, vr)
    out = ps.verdict_from_result(ps.mcnemar_from_vectors(VA, VB, "a", "b"), rerun=flip)
    assert out["verdict"] == "indistinguishable" and out["reason"] == "within_rerun_noise"


def test_claim_survives_when_delta_above_flip_rate() -> None:
    vr = _vec(_flipped(A, range(0, 3)))  # flip_rate 0.03 < 0.10
    flip = ps.rerun_flip_rate(VA, vr)
    out = ps.verdict_from_result(ps.mcnemar_from_vectors(VA, VB, "a", "b"), rerun=flip)
    assert out["verdict"] == "b_better"
    assert "reason" not in out and "verdict_before_rerun" not in out
    assert out["rerun"]["flip_rate"] == pytest.approx(0.03)
    assert math.isclose(out["abs_delta"], 0.10)


def test_a_better_direction_also_downgrades() -> None:
    result = ps.mcnemar_from_vectors(VB, VA, "b", "a")  # arms swapped: a_better
    assert ps.mcnemar_verdict(12, 2)["verdict"] == "a_better"
    flip = ps.rerun_flip_rate(VB, _vec(_flipped(B_list, range(0, 15))))
    out = ps.verdict_from_result(result, rerun=flip)
    assert out["verdict"] == "indistinguishable" and out["reason"] == "within_rerun_noise"


def test_indistinguishable_stays_with_no_reason() -> None:
    result = ps.mcnemar_from_vectors(VA, VA, "a", "a2")
    flip = ps.rerun_flip_rate(VA, VA)
    out = ps.verdict_from_result(result, rerun=flip)
    assert out["verdict"] == "indistinguishable"
    assert "reason" not in out
    assert out["rerun"]["flip_rate"] == 0.0


def test_cli_rerun_trial(tmp_path, capsys) -> None:
    def row(tid: int, vec: list[bool]) -> dict:
        return {
            "trial_id": tid,
            "eval_details": {
                "question_results": [
                    {"qid": f"q{i:03d}", "suite": "s", "correct": ok} for i, ok in enumerate(vec)
                ]
            },
        }

    j = tmp_path / "autopilot_journal.jsonl"
    rows = [row(1, A), row(2, B_list), row(3, _flipped(A, range(0, 12)))]
    j.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert ps.main(["--journal", str(j), "mcnemar", "1", "2", "--rerun-trial", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"]["reason"] == "within_rerun_noise"
    # without --rerun-trial: raw stats, no verdict key, refusal on stderr, exit 2
    assert ps.main(["--journal", str(j), "mcnemar", "1", "2"]) == 2
    cap = capsys.readouterr()
    payload = json.loads(cap.out)
    assert "verdict" not in payload and payload["a_correct_b_wrong"] == 2
    assert "no rerun arm R" in cap.err
