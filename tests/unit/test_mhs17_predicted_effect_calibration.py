"""MHS-17: predicted-effect calibration scorer on a synthetic ledger."""

import sys
from pathlib import Path

import pytest


from predicted_effect_calibration import (  # noqa: E402
    calibration_report,
    extract_row,
    score_mutation,
)


def _row(mid, rnd, n, pf, pr, f2p, p2f, nest=False):
    body = {
        "predicted_fixed": pf,
        "predicted_at_risk": pr,
        "observed_fail_to_pass": f2p,
        "observed_pass_to_fail": p2f,
        "panel_size": n,
    }
    base = {"mutation_id": mid, "round": rnd}
    return {**base, "eval_details": {"predicted_effect": body}} if nest else {**base, **body}


LEDGER = [
    # exact fix, regression covered by at-risk
    _row("m1", 1, 10, ["a", "b"], ["x"], ["a", "b"], ["x"]),
    # fix half right; one regression NOT predicted (unattributed)
    _row("m2", 1, 10, ["c", "d"], ["y"], ["c"], ["z"], nest=True),
    # predicted a fix that did not happen, no regression
    _row("m3", 2, 20, ["e"], [], [], []),
    # not annotated: counted, not dropped
    {"mutation_id": "m4", "round": 2},
    # annotated, observed missing: counted separately
    {"mutation_id": "m5", "round": 2, "predicted_fixed": ["q"]},
]


def test_per_mutation_verdicts_by_set_intersection():
    v = {s["mutation_id"]: s for s in calibration_report(LEDGER)["verdicts"]}
    assert v["m1"]["fix_verdict"] == "exact" and v["m1"]["regression_verdict"] == "attributed"
    assert v["m2"]["fix_verdict"] == "partial"
    assert v["m2"]["regression_verdict"] == "unattributed"
    assert v["m2"]["unattributed_regressions"] == ["z"]
    assert v["m3"]["fix_verdict"] == "predicted_not_observed"
    assert v["m3"]["regression_verdict"] == "no_regression"


def test_counts_do_not_drop_rows():
    r = calibration_report(LEDGER)
    assert (r["rows_total"], r["rows_scored"]) == (5, 3)
    assert r["rows_unannotated"] == 1 and r["rows_annotated_without_observed"] == 1


def test_round_precision_recall_and_random_baseline():
    r1 = calibration_report(LEDGER)["rounds"]["1"]["fixes"]
    # round 1: predicted 4 (a,b,c,d), observed 3 (a,b,c), hits 3
    assert (r1["predicted"], r1["observed"], r1["hits"]) == (4, 3, 3)
    assert r1["precision"] == pytest.approx(0.75) and r1["recall"] == pytest.approx(1.0)
    # random expected hits = sum k*F/n = 2*2/10 + 2*1/10
    assert r1["random_expected_hits"] == pytest.approx(0.6)
    assert r1["random_precision"] == pytest.approx(0.6 / 4)
    assert r1["random_recall"] == pytest.approx(0.6 / 3)
    reg = calibration_report(LEDGER)["rounds"]["1"]["regressions"]
    assert (reg["predicted"], reg["observed"], reg["hits"]) == (2, 2, 1)


def test_unattributed_regression_list_is_explicit():
    u = calibration_report(LEDGER)["unattributed regressions"]
    assert u == [{"mutation_id": "m2", "round": 1, "tasks": ["z"]}]


def test_empty_denominators_are_none_not_zero():
    r = calibration_report([_row("m", 1, 5, [], [], [], [])])["rounds"]["1"]["fixes"]
    assert r["precision"] is None and r["recall"] is None


def test_missing_panel_size_marks_baseline_incomplete():
    r = calibration_report([_row("m", 1, None, ["a"], [], ["a"], [])])["rounds"]["1"]["fixes"]
    assert r["baseline_complete"] is False and r["random_precision"] is None


def test_panel_list_gives_size():
    row = {"mutation_id": "p", "predicted_fixed": ["a"], "observed_fail_to_pass": ["a"],
           "observed_pass_to_fail": [], "predicted_at_risk": [], "panel": ["a", "b", "c", "d"]}
    assert extract_row(row)["panel_size"] == 4
    assert score_mutation(extract_row(row))["fix_verdict"] == "exact"
