"""REPL-EMB-2.2 statistics: overlap/recall accounting, paired bootstrap, kill + selection rules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
import repl_emb_stats as st  # noqa: E402


def vec(ones: int, n: int, offset: int = 0) -> np.ndarray:
    """0/1 vector of length n with ``ones`` ones starting at ``offset``."""
    v = np.zeros(n)
    v[offset : offset + ones] = 1.0
    return v


# ───────────────────────────────────────────────────────────────────────────── overlaps / recall


def test_overlaps_is_inclusive_on_both_ends():
    assert st.overlaps((10, 20), [(20, 30)])
    assert st.overlaps((10, 20), [(1, 10)])
    assert st.overlaps((10, 20), [(12, 14)])  # contained
    assert st.overlaps((12, 14), [(10, 20)])  # containing
    assert not st.overlaps((10, 20), [(21, 30)])
    assert not st.overlaps((10, 20), [(1, 9)])
    assert st.overlaps((10, 20), [(1, 5), (19, 40)])  # any range
    assert not st.overlaps((10, 20), [])
    with pytest.raises(ValueError):
        st.overlaps((20, 10), [(1, 2)])
    with pytest.raises(ValueError):
        st.overlaps((1, 2), [(9, 3)])


def test_recall_at_k_same_section_and_rank_cutoff():
    gold = [[("a.py", 10, 20)], [("b.py", 1, 5)], [("c.py", 1, 5)]]
    ranked = [
        [("zzz.py", 10, 20), ("a.py", 1, 9), ("a.py", 20, 25)],  # hit only at rank 3
        [("a.py", 1, 5)],  # right lines, wrong section -> miss
        [],  # returned nothing -> miss, NOT a failure
    ]
    assert st.recall_at_k(ranked, gold, 1).tolist() == [0, 0, 0]
    assert st.recall_at_k(ranked, gold, 2).tolist() == [0, 0, 0]
    assert st.recall_at_k(ranked, gold, 3).tolist() == [1, 0, 0]
    assert st.recall_at_k(ranked, gold, 10).tolist() == [1, 0, 0]
    assert st.count_failed(ranked) == 0


def test_failed_query_is_a_miss_and_counted_separately():
    gold = [[("a.py", 1, 5)], [("a.py", 1, 5)], [("a.py", 1, 5)]]
    ranked = [[("a.py", 1, 5)], None, []]
    assert st.recall_at_k(ranked, gold, 5).tolist() == [1, 0, 0]
    assert st.count_failed(ranked) == 1  # only the None; the empty list is a plain miss


def test_recall_multi_range_gold_and_all_k():
    gold = [[("a.py", 1, 5), ("a.py", 100, 110)]]
    ranked = [[("a.py", 50, 60), ("a.py", 108, 120)]]
    out = st.recall_all_k(ranked, gold)
    assert {k: v.tolist() for k, v in out.items()} == {1: [0], 3: [1], 5: [1], 10: [1]}


def test_recall_length_mismatch_and_bad_k():
    with pytest.raises(ValueError):
        st.recall_at_k([[]], [], 5)
    with pytest.raises(ValueError):
        st.recall_at_k([], [], 0)


# ───────────────────────────────────────────────────────────────────────────── bootstrap


def test_paired_bootstrap_known_answer_constant_difference():
    a, b = np.ones(30), np.zeros(30)
    r = st.paired_bootstrap(a, b)
    assert (r["mean_diff"], r["ci_lo"], r["ci_hi"]) == (1.0, 1.0, 1.0)
    assert r["resamples"] == 10000 and r["seed"] == 20261020 and r["n"] == 30


def test_paired_bootstrap_known_answer_two_queries():
    # d = [1, 0]: resampled means are 0, .5, 1 with probability 1/4, 1/2, 1/4 -> CI [0, 1]
    r = st.paired_bootstrap([1, 0], [0, 0])
    assert r["mean_diff"] == 0.5
    assert (r["ci_lo"], r["ci_hi"]) == (0.0, 1.0)
    # d = [1, -1]: means -1, 0, 1 -> CI [-1, 1]
    r = st.paired_bootstrap([1, 0], [0, 1])
    assert r["mean_diff"] == 0.0
    assert (r["ci_lo"], r["ci_hi"]) == (-1.0, 1.0)


def test_paired_bootstrap_interval_is_near_the_normal_approximation():
    n = 400
    # discordant pairs: a-only 70, b-only 30, both 150 -> mean diff 0.10
    a = np.concatenate([np.ones(70), np.zeros(30), np.ones(150), np.zeros(150)])
    b = np.concatenate([np.zeros(70), np.ones(30), np.ones(150), np.zeros(150)])
    r = st.paired_bootstrap(a, b)
    d = a - b
    se = d.std() / np.sqrt(n)
    assert r["mean_diff"] == pytest.approx(0.10)
    assert r["ci_lo"] == pytest.approx(0.10 - 1.96 * se, abs=0.01)
    assert r["ci_hi"] == pytest.approx(0.10 + 1.96 * se, abs=0.01)


def test_paired_bootstrap_is_deterministic_and_seed_sensitive():
    rng = np.random.default_rng(3)
    a, b = (rng.random(60) < 0.6).astype(float), (rng.random(60) < 0.4).astype(float)
    r1, r2 = st.paired_bootstrap(a, b), st.paired_bootstrap(a, b)
    assert r1 == r2
    r3 = st.paired_bootstrap(a, b, seed=1)
    assert r3["seed"] == 1
    # the discrete difference makes two seeds' intervals coincide often; across several seeds
    # (few resamples, so the percentiles are noisy) they must not all coincide
    cis = {
        (r["ci_lo"], r["ci_hi"])
        for r in (st.paired_bootstrap(a, b, resamples=200, seed=s) for s in range(8))
    }
    assert len(cis) > 1
    # the resample count is part of the record and of the result
    assert st.paired_bootstrap(a, b, resamples=2500)["resamples"] == 2500
    # the draw is paired: swapping a and b mirrors the interval exactly
    rs = st.paired_bootstrap(b, a)
    assert rs["ci_lo"] == pytest.approx(-r1["ci_hi"])
    assert rs["ci_hi"] == pytest.approx(-r1["ci_lo"])


def test_paired_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        st.paired_bootstrap([1, 0], [1])
    with pytest.raises(ValueError):
        st.paired_bootstrap([], [])
    with pytest.raises(ValueError):
        st.paired_bootstrap([1, np.nan], [0, 0])
    with pytest.raises(ValueError):
        st.paired_bootstrap([[1, 0]], [[0, 0]])


# ───────────────────────────────────────────────────────────────────────────── kill rule


def test_kill_rule_passes_with_margin_and_positive_lower_bound():
    n = 120
    # hybrid-only 25, lexical-only 5, both 50 -> diff 20/120 = 0.167, SE ~ 0.043, lo ~ 0.08
    h = np.concatenate([np.ones(25), np.zeros(5), np.ones(50), np.zeros(40)])
    lx = np.concatenate([np.zeros(25), np.ones(5), np.ones(50), np.zeros(40)])
    r = st.kill_rule(h, lx)
    assert r["n"] == n and r["pass"] is True
    assert r["margin_ok"] and r["ci_ok"] and r["ci_lo"] > 0
    assert r["M"] == 0.10 and r["mean_diff"] == pytest.approx(20 / 120)


def test_kill_rule_fails_when_mean_clears_M_but_lower_bound_does_not():
    # n = 20, 3 hybrid-only wins: diff 0.15 >= 0.10 but P(no win drawn) = (17/20)^20 = 3.9% > 2.5%
    h = np.concatenate([np.ones(3), np.zeros(17)])
    lx = np.zeros(20)
    r = st.kill_rule(h, lx)
    assert r["mean_diff"] == pytest.approx(0.15)
    assert r["margin_ok"] is True
    assert r["ci_lo"] <= 0.0
    assert r["ci_ok"] is False
    assert r["pass"] is False


def test_kill_rule_margin_boundary_is_inclusive_and_just_below_fails():
    n = 120
    zeros = np.zeros(n)
    at_m = st.kill_rule(vec(12, n), zeros)  # exactly 0.10
    assert at_m["mean_diff"] == pytest.approx(0.10)
    assert at_m["margin_ok"] is True and at_m["ci_ok"] is True and at_m["pass"] is True
    below = st.kill_rule(vec(11, n), zeros)  # 0.0917
    assert below["ci_ok"] is True  # the CI alone would pass ...
    assert below["margin_ok"] is False and below["pass"] is False  # ... the margin does not


def test_kill_rule_negative_difference_fails():
    r = st.kill_rule(np.zeros(120), vec(40, 120))
    assert r["mean_diff"] < 0 and r["pass"] is False


def test_kill_rule_is_deterministic():
    h, lx = vec(40, 120), vec(20, 120, 10)
    assert st.kill_rule(h, lx) == st.kill_rule(h, lx)


def test_lexical_baseline_picks_better_arm_and_ties_go_to_bm25():
    g, b = vec(30, 100), vec(20, 100)
    assert st.lexical_baseline(g, b)["L"] == st.lexical_baseline(g, b)["which"] == "grep"
    assert st.lexical_baseline(b, g)["L"] == "bm25"
    tie = st.lexical_baseline(vec(25, 100), vec(25, 100, 40))
    assert tie["L"] == "bm25" and tie["tie"] is True
    one = st.lexical_baseline(vec(26, 100), vec(25, 100))
    assert one["L"] == "grep" and one["tie"] is False


def _passing_pair(n=120):
    h = np.concatenate([np.ones(25), np.zeros(5), np.ones(50), np.zeros(n - 80)])
    lx = np.concatenate([np.zeros(25), np.ones(5), np.ones(50), np.zeros(n - 80)])
    return h, lx


def test_kill_decision_passes_iff_one_registered_hybrid_passes():
    h, lx = _passing_pair()
    weak = vec(10, 120)
    d = st.kill_decision({"bge.hybrid": weak, "granite.hybrid": h}, {}, lx, "bm25")
    assert d["verdict"] == "PASS" and d["registered_passers"] == ["granite.hybrid"]
    assert d["registered"]["bge.hybrid"]["pass"] is False  # every individual result is reported
    assert d["registered"]["granite.hybrid"]["pass"] is True
    assert d["changed_by_extension"] is False and d["extension_only"] == []


def test_kill_decision_fail_when_no_registered_hybrid_passes():
    lx = vec(40, 120)
    d = st.kill_decision(
        {"bge.hybrid": vec(41, 120), "granite.hybrid": vec(45, 120)}, {}, lx, "grep"
    )
    assert d["verdict"] == "FAIL" and d["underpowered"] is False


def test_kill_decision_small_n_kills_by_design():
    h, lx = _passing_pair(120)
    # the same evidence on 60 queries: would pass on its own, but n < 120 -> UNDERPOWERED/KILL
    d = st.kill_decision({"bge.hybrid": h[:60]}, {}, lx[:60], "bm25")
    assert d["n"] == 60 and d["underpowered"] is True
    assert d["verdict"] == "UNDERPOWERED/KILL"
    assert d["registered"]["bge.hybrid"]["mean_diff"] > 0.10  # the evidence is still reported
    # exactly n_min is powered
    assert st.kill_decision({"bge.hybrid": h}, {}, lx, "bm25")["verdict"] == "PASS"


def test_kill_decision_flags_extension_only_pass():
    h, lx = _passing_pair()
    d = st.kill_decision({"bge.hybrid": vec(10, 120)}, {"eg2.hybrid": h}, lx, "bm25")
    assert d["verdict"] == "FAIL"  # the registered verdict stands
    assert d["verdict_with_extension"] == "PASS"
    assert d["changed_by_extension"] is True
    assert d["extension_only"] == ["eg2.hybrid"]


def test_kill_decision_extension_does_not_flag_when_registered_already_pass():
    h, lx = _passing_pair()
    d = st.kill_decision({"bge.hybrid": h}, {"eg2.hybrid": h}, lx, "bm25")
    assert d["verdict"] == d["verdict_with_extension"] == "PASS"
    assert d["extension_only"] == [] and d["changed_by_extension"] is False


def test_kill_decision_incomplete_not_fail_when_a_registered_arm_is_missing():
    lx = vec(40, 120)
    d = st.kill_decision(
        {"bge.hybrid": vec(41, 120)}, {}, lx, "bm25", missing_registered=["granite.hybrid"]
    )
    assert d["verdict"] == "INCOMPLETE"
    # a pass from the arm that is present is still a pass
    h, lx2 = _passing_pair()
    ok = st.kill_decision({"bge.hybrid": h}, {}, lx2, "bm25", missing_registered=["granite.hybrid"])
    assert ok["verdict"] == "PASS"
    # no registered arm scored at all is INCOMPLETE too, never FAIL
    assert st.kill_decision({}, {}, lx, "bm25")["verdict"] == "INCOMPLETE"


# ───────────────────────────────────────────────────────────────────────────── selection rule


def test_selection_picks_cheapest_within_x_of_best():
    arms = {
        "bge.hybrid": {"recall5": 0.70, "cpu_s_per_query": 0.50, "p95_ms": 400},
        "granite.hybrid": {"recall5": 0.66, "cpu_s_per_query": 0.10, "p95_ms": 90},  # within X
        "potion.hybrid": {
            "recall5": 0.60,
            "cpu_s_per_query": 0.01,
            "p95_ms": 5,
        },  # cheaper, too low
        "eg2.hybrid": {"recall5": 0.69, "cpu_s_per_query": 0.30, "p95_ms": 200},
    }
    r = st.selection_rule(arms)
    assert r["status"] == "selected" and r["selected"] == "granite.hybrid"
    assert r["best_arm"] == "bge.hybrid" and r["best_recall5"] == 0.70
    assert r["threshold"] == pytest.approx(0.65)
    assert r["candidates"] == ["bge.hybrid", "eg2.hybrid", "granite.hybrid"]
    by = {row["arm"]: row for row in r["table"]}
    assert by["potion.hybrid"]["candidate"] is False
    assert by["granite.hybrid"]["selected"] is True


def test_selection_x_boundary_is_inclusive():
    arms = {
        "best": {"recall5": 0.80, "cpu_s_per_query": 1.0, "p95_ms": 10},
        "edge": {"recall5": 0.75, "cpu_s_per_query": 0.1, "p95_ms": 10},  # exactly best - X
        "out": {"recall5": 0.7499, "cpu_s_per_query": 0.01, "p95_ms": 10},
    }
    r = st.selection_rule(arms)
    assert r["selected"] == "edge"


def test_selection_ties_on_cpu_break_on_p95_then_name():
    arms = {
        "a": {"recall5": 0.7, "cpu_s_per_query": 0.2, "p95_ms": 50},
        "b": {"recall5": 0.7, "cpu_s_per_query": 0.2, "p95_ms": 30},
        "c": {"recall5": 0.7, "cpu_s_per_query": 0.2, "p95_ms": 30},
    }
    assert st.selection_rule(arms)["selected"] == "b"


def test_selection_is_pending_without_cost_and_never_guesses():
    arms = {
        "bge.hybrid": {"recall5": 0.70, "cpu_s_per_query": None, "p95_ms": None},
        "granite.hybrid": {"recall5": 0.69, "cpu_s_per_query": 0.1, "p95_ms": 9},
    }
    r = st.selection_rule(arms)
    assert r["status"] == "pending speed block" and r["selected"] is None
    assert r["missing_cost"] == ["bge.hybrid"]
    # a cost-less arm OUTSIDE the candidate band does not block the selection
    arms2 = dict(arms)
    arms2["bge.hybrid"] = {"recall5": 0.70, "cpu_s_per_query": 0.5, "p95_ms": 1}
    arms2["far.hybrid"] = {"recall5": 0.30}
    assert st.selection_rule(arms2)["selected"] == "granite.hybrid"


# ───────────────────────────────────────────────────────────────────────────── strata / Wilson


def test_wilson_interval_known_values():
    lo, hi = st.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = st.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    assert st.wilson_interval(0, 0) == (None, None)


def test_stratum_report_per_stratum_and_pooled():
    strata = ["probe45"] * 4 + ["spill_follow"] * 6
    arm = {
        1: np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0.0]),
        3: np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0.0]),
        5: np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0.0]),
        10: np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0.0]),
    }
    rep = st.stratum_report({"bm25": arm}, strata)
    b = rep["bm25"]
    assert set(b) == {"probe45", "spill_follow", "pooled"}
    assert b["probe45"][5]["recall"] == 0.75 and b["probe45"][5]["n"] == 4
    assert b["spill_follow"][5]["recall"] == pytest.approx(3 / 6)
    assert b["pooled"][10]["hits"] == 8 and b["pooled"][10]["n"] == 10
    assert b["pooled"][1]["recall"] == pytest.approx(0.2)
    lo, hi = st.wilson_interval(6, 10)
    assert (b["pooled"][5]["ci_lo"], b["pooled"][5]["ci_hi"]) == (lo, hi)
    assert b["pooled"][5]["ci"] == "wilson-95"
    with pytest.raises(ValueError):
        st.stratum_report(
            {"x": {1: np.zeros(3), 3: np.zeros(3), 5: np.zeros(3), 10: np.zeros(3)}}, strata
        )


def test_latency_summary():
    s = st.latency_summary([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s["p50_ms"] == 30.0 and s["n"] == 5 and s["mean_ms"] == 30.0
    assert s["p95_ms"] == pytest.approx(48.0)
    assert st.latency_summary([])["p95_ms"] is None


def test_predeclared_interpretations_are_in_the_docstring():
    assert set(st.PREDECLARED_INTERPRETATIONS) >= {
        "a_decision_set",
        "b_registered_hybrids",
        "c_selection",
    }
    for text in st.PREDECLARED_INTERPRETATIONS.values():
        assert text in (st.__doc__ or "")


def test_kill_decision_floor_failures_make_it_underpowered_whatever_the_evidence():
    h, lx = _passing_pair()  # n = 120 and a clear pass on its own
    assert st.kill_decision({"bge.hybrid": h}, {}, lx, "bm25")["verdict"] == "PASS"
    d = st.kill_decision({"bge.hybrid": h}, {}, lx, "bm25", floor_failures=["trace rows=60 < 75"])
    assert d["underpowered"] is True and d["verdict"] == "UNDERPOWERED/KILL"
    assert d["verdict_with_extension"] == "UNDERPOWERED/KILL"
    assert d["floor_failures"] == ["trace rows=60 < 75"]
    assert d["registered"]["bge.hybrid"]["pass"] is True  # the evidence is still reported
    assert d["extension_only"] == []
    assert st.TRACE_N_MIN == 75
