"""EV-20: per-task paired path of evaluate_skill_efficacy (pure functions, zero inference)."""

import sys
from pathlib import Path

import pytest


from skill_efficacy import (  # noqa: E402
    evaluate_skill_efficacy,
    mcnemar_exact_p,
    paired_task_readout,
)

# intake-1321#record Table 2: 18 library conditions as (gains c, regressions b).
TABLE2 = {
    ("oqa", "minimax", "anthropic"): (14, 13), ("oqa", "minimax", "openai"): (16, 12),
    ("oqa", "minimax", "ours"): (15, 6), ("oqa", "codex", "anthropic"): (17, 15),
    ("oqa", "codex", "openai"): (11, 9), ("oqa", "codex", "ours"): (16, 13),
    ("oqa", "claude", "anthropic"): (10, 2), ("oqa", "claude", "openai"): (12, 7),
    ("oqa", "claude", "ours"): (11, 4),
    ("sb", "minimax", "anthropic"): (62, 35), ("sb", "minimax", "openai"): (45, 41),
    ("sb", "minimax", "ours"): (48, 39), ("sb", "codex", "anthropic"): (30, 23),
    ("sb", "codex", "openai"): (30, 27), ("sb", "codex", "ours"): (28, 23),
    ("sb", "claude", "anthropic"): (59, 16), ("sb", "claude", "openai"): (63, 20),
    ("sb", "claude", "ours"): (66, 19),
}
# reported two-sided p where the record prints one (4 decimals as printed).
TABLE2_P = {
    ("oqa", "minimax", "anthropic"): 1.000, ("oqa", "minimax", "openai"): 0.572,
    ("oqa", "minimax", "ours"): 0.078, ("oqa", "claude", "anthropic"): 0.039,
    ("sb", "minimax", "anthropic"): 0.008,
}


def _arms(c, b, extra_pass=3, extra_fail=2):
    """Build baseline/candidate task dicts with exactly c rescued and b regressed."""
    base, cand = {}, {}
    for i in range(c):
        base[f"r{i}"], cand[f"r{i}"] = False, True
    for i in range(b):
        base[f"g{i}"], cand[f"g{i}"] = True, False
    for i in range(extra_pass):
        base[f"p{i}"] = cand[f"p{i}"] = True
    for i in range(extra_fail):
        base[f"f{i}"] = cand[f"f{i}"] = False
    return base, cand


def test_table2_totals_match_record():
    oqa = [v for k, v in TABLE2.items() if k[0] == "oqa"]
    sb = [v for k, v in TABLE2.items() if k[0] == "sb"]
    assert (sum(c for c, _ in oqa), sum(b for _, b in oqa)) == (122, 81)
    assert (sum(c for c, _ in sb), sum(b for _, b in sb)) == (431, 243)
    assert len(TABLE2) == 18


@pytest.mark.parametrize("cell,p", sorted(TABLE2_P.items()))
def test_exact_mcnemar_reproduces_reported_p(cell, p):
    c, b = TABLE2[cell]
    assert round(mcnemar_exact_p(b, c), 3) == p


def test_claude_sb_cells_are_below_001():
    for lib in ("anthropic", "openai", "ours"):
        c, b = TABLE2[("sb", "claude", lib)]
        assert mcnemar_exact_p(b, c) < 0.001


@pytest.mark.parametrize("cell", sorted(TABLE2))
def test_per_task_path_recovers_c_b_p_for_every_cell(cell):
    c, b = TABLE2[cell]
    base, cand = _arms(c, b)
    r = paired_task_readout(base, cand)
    assert (r.c, r.b) == (c, b)
    assert r.mcnemar_p == mcnemar_exact_p(b, c)
    assert len(r.rescued) == c and len(r.regressed) == b


# intake-1354#record page matrix (DIVE-ADDED [8]): (gained c, lost b) -> exact p.
@pytest.mark.parametrize("c,b,p,tol", [(22, 2, 3.6e-5, 0.05e-5), (9, 5, 0.42, 0.005), (1, 26, 4.2e-7, 0.05e-7)])
def test_intake_1354_matrix_p_values(c, b, p, tol):
    assert mcnemar_exact_p(b, c) == pytest.approx(p, abs=tol)


def test_degenerate_pairs():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 5) == 1.0


def test_rescued_and_regressed_lists_are_sorted_ids():
    r = paired_task_readout({"b": False, "a": False, "z": True}, {"b": True, "a": True, "z": False})
    assert r.rescued == ["a", "b"] and r.regressed == ["z"]


def test_pairwise_drop_removes_task_from_every_arm():
    base = {"t1": True, "t2": False, "t3": True}
    cand = {"t1": False, "t2": True, "t3": None}  # t3 errored in candidate only
    inc = [{"t1": True, "t2": False, "t3": True}]
    r = paired_task_readout(base, cand, incumbent_replicates=inc)
    assert r.n_dropped == 1 and r.n_tasks == 2
    assert (r.c, r.b) == (1, 1)


def test_pairwise_drop_also_drops_when_only_an_incumbent_replicate_errored():
    base = {"t1": True, "t2": True}
    cand = {"t1": False, "t2": True}
    inc = [{"t1": True, "t2": None}, {"t1": True, "t2": True}]
    r = paired_task_readout(base, cand, incumbent_replicates=inc)
    assert r.n_dropped == 1 and r.b == 1


def test_count_as_fail_in_both_arms_never_one():
    base = {"t1": None, "t2": True}
    cand = {"t1": None, "t2": True}
    r = paired_task_readout(base, cand, errored="count_as_fail_both")
    assert r.n_dropped == 0 and r.n_tasks == 2 and (r.c, r.b) == (0, 0)
    # an errored candidate sample counts as a fail; the baseline pass makes it a regression
    r2 = paired_task_readout({"t": True}, {"t": None}, errored="count_as_fail_both")
    assert r2.b == 1
    # and the same rule on the baseline side: errored baseline + passing candidate = rescued
    r3 = paired_task_readout({"t": None}, {"t": True}, errored="count_as_fail_both")
    assert r3.c == 1


def test_bad_errored_mode_rejected():
    with pytest.raises(ValueError):
        paired_task_readout({}, {}, errored="drop_candidate_only")


def _inc(b_values, n_extra=20):
    """Incumbent replicates whose regressions vs an all-pass baseline are b_values."""
    base = {f"t{i}": True for i in range(n_extra)}
    reps = []
    for bv in b_values:
        rep = dict(base)
        for i in range(bv):
            rep[f"t{i}"] = False
        reps.append(rep)
    return base, reps


def test_zero_growth_budget_pass_and_fail():
    base, reps = _inc([3, 5])  # b_incumbent mean 4, SD sqrt(2)
    base = {**base, **{f"u{i}": False for i in range(10)}}
    reps = [{**r, **{f"u{i}": False for i in range(10)}} for r in reps]
    cand_ok = {k: (k.startswith("t")) for k in base}
    cand_ok.update({"t0": False, "t1": False, "t2": False})
    for i in range(6):
        cand_ok[f"u{i}"] = True  # c=6 > b=3, b=3 <= 4
    r = paired_task_readout(base, cand_ok, incumbent_replicates=reps)
    assert r.b_incumbent_mean == 4.0 and r.b_incumbent_sd == pytest.approx(2 ** 0.5)
    assert (r.b_candidate, r.c_candidate) == (3, 6)
    assert r.would_pass_zero_growth_budget is True
    cand_bad = dict(cand_ok)
    for i in range(3, 8):
        cand_bad[f"t{i}"] = False  # b=8 > 4
    assert paired_task_readout(base, cand_bad, incumbent_replicates=reps).would_pass_zero_growth_budget is False


def test_zero_growth_requires_c_above_b():
    base, reps = _inc([4, 4])
    cand = {k: True for k in base}
    cand["t0"] = False
    r = paired_task_readout(base, cand, incumbent_replicates=reps)
    assert r.b == 1 and r.c == 0 and r.would_pass_zero_growth_budget is False


def test_too_few_incumbent_replicates_is_none_not_false():
    base, reps = _inc([2])
    r = paired_task_readout(base, {k: True for k in base}, incumbent_replicates=reps)
    assert r.would_pass_zero_growth_budget is None and r.b_incumbent_sd is None
    r3 = paired_task_readout(base, {k: True for k in base}, incumbent_replicates=_inc([2, 2])[1],
                             min_incumbent_replicates=3)
    assert r3.would_pass_zero_growth_budget is None
    r0 = paired_task_readout(base, {k: True for k in base})
    assert r0.would_pass_zero_growth_budget is None and r0.b_incumbent_mean is None


def test_default_path_is_unchanged_and_paired_is_report_only():
    wo, wi = {"math": 0.5, "code": 0.5}, {"math": 0.6, "code": 0.55}
    plain = evaluate_skill_efficacy(wo, wi)
    assert plain.paired is None
    base, cand = _arms(3, 10)  # per-task regressions dominate, suite scores still accept
    with_pt = evaluate_skill_efficacy(wo, wi, per_task_without=base, per_task_with=cand)
    assert with_pt.paired is not None and with_pt.paired.b == 10
    assert (with_pt.accept, with_pt.reason, with_pt.aggregate_delta, with_pt.per_suite_delta) == (
        plain.accept, plain.reason, plain.aggregate_delta, plain.per_suite_delta)
    assert with_pt == plain and repr(with_pt) == repr(plain)
    # only one side supplied: no readout, no error
    assert evaluate_skill_efficacy(wo, wi, per_task_without=base).paired is None
