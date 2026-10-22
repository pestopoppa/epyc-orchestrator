"""File-scoped tests for scripts/autopilot/degradation_stats.py (EV-DEGR1)."""

import math

import pytest

from scripts.autopilot import degradation_stats as ds
from scripts.autopilot.paired_stats import QuestionOutcome, mcnemar_verdict


def test_e7c_regression_one_sided():
    assert round(ds.one_sided_degradation_p(61, 58), 4) == 0.4273


def test_zero_discordant_is_one_not_zero():
    assert ds.one_sided_degradation_p(0, 0) == 1.0
    r = ds.aggregate_degradation({"a": (0, 0), "b": (0, 0)})
    assert r["p_pooled"] == 1.0 and r["p_fisher"] == pytest.approx(1.0)
    assert r["p_max_drop"] == 1.0 and not r["any_degraded"]
    assert all(v["p_one_sided"] == 1.0 for v in r["per_suite"].values())


def test_empty_and_tiny_n():
    assert ds.aggregate_degradation({})["p_pooled"] == 1.0
    assert ds.one_sided_degradation_p(1, 0) == 0.5
    assert ds.one_sided_degradation_p(0, 1) == 1.0
    v = ds.degradation_verdict(0, 0)
    assert v["direction"] == "tie" and v["degraded"] is False


def test_single_suite_fisher_equals_pooled():
    r = ds.aggregate_degradation({"s": (30, 10)})
    assert r["p_fisher"] == pytest.approx(r["p_pooled"], rel=1e-9)
    assert r["any_degraded"] is True


def test_seeded_reproducible_and_zero_suite_does_not_break_maxdrop():
    s = {"a": (12, 3), "b": (0, 0), "c": (4, 5)}
    assert ds.aggregate_degradation(s) == ds.aggregate_degradation(s)
    assert ds.aggregate_degradation(s)["p_max_drop"] < 1.0


def test_improvement_not_flagged():
    r = ds.aggregate_degradation({"a": (2, 20), "b": (1, 9)})
    assert not r["any_degraded"]
    assert ds.degradation_verdict(2, 20)["direction"] == "candidate_better"


def test_power_arithmetic_matches_handoff_claims():
    assert math.ceil(ds.power_required_n(0.10, 0.05)) == 248
    assert ds.power_at_n(250, 0.10, 0.02) == pytest.approx(0.26, abs=0.01)
    assert ds.min_detectable_delta(250, 0.10) == pytest.approx(0.0497, abs=1e-3)
    m = ds.margin_sizing(None, None)
    assert m["required_n"] is None and m["margin"] == 0.02


def test_vectors_and_additive_with_mcnemar():
    def mk(q, s, c):
        return QuestionOutcome(qid=q, suite=s, correct=c, trial_id=0)

    base = {f"q{i}": mk(f"q{i}", "s1" if i < 5 else "s2", True) for i in range(10)}
    cand = {k: mk(k, v.suite, i % 2 == 0) for i, (k, v) in enumerate(base.items())}
    by = ds.count_discordant_by_suite(base, cand)
    assert by == {"s1": (2, 0), "s2": (3, 0)}
    before = mcnemar_verdict(5, 0)
    ds.degradation_from_vectors(base, cand)
    assert mcnemar_verdict(5, 0) == before and "p_one_sided" not in before


def test_cli_smoke(capsys):
    assert ds.main(["verdict", "--b", "61", "--c", "58"]) == 0
    assert '"n_discordant": 119' in capsys.readouterr().out
    assert ds.main(["size", "--p-flip", "0.1", "--n", "250"]) == 0
    assert '"required_n": 1546' in capsys.readouterr().out
