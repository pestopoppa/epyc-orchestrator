"""DAR-SPLIT-1: split-half decisive-subset selection with a binomial null."""
from __future__ import annotations

import random

from scripts.analysis.dar_decisive_subset import (
    SEED,
    binom_sf,
    select_on_fold,
    selection_null,
    split_folds,
    split_policy_lift,
    summarize,
)


def _fixture(n_obj, n_obs, planted_frac, planted_rate, base_rate, seed=7):
    rng = random.Random(seed)
    data, planted = {}, set()
    for i in range(n_obj):
        obj = f"o{i}"
        is_p = i < int(n_obj * planted_frac)
        if is_p:
            planted.add(obj)
        data[obj] = {}
        for role in ("r1", "r2", "r3"):
            p = planted_rate if (is_p and role == "r1") else base_rate
            data[obj][role] = [1 if rng.random() < p else 0 for _ in range(n_obs)]
    return data, planted


def test_folds_are_disjoint_and_seeded():
    data, _ = _fixture(5, 20, 0, 0.5, 0.5)
    f1, f2 = split_folds(data, SEED), split_folds(data, SEED)
    assert f1 == f2
    for obj, roles in f1.items():
        for role, (A, B) in roles.items():
            assert len(A) + len(B) == 20
            assert sorted(A + B) == sorted(data[obj][role])


def test_binom_sf_exact():
    assert abs(binom_sf(3, 3, 0.5) - 0.125) < 1e-12
    assert binom_sf(0, 5, 0.3) == 1.0


def test_pure_noise_yields_no_significant_decisive_subset():
    data, _ = _fixture(300, 40, 0.0, 0.5, 0.5)
    for sel in (0, 1):
        res = split_policy_lift(data, 0.05, sel)
        s = summarize(res)
        null = selection_null(res)
        assert null["n_selected"] <= 0.12 * null["n_tested"]
        assert null["p_selected_exceeds_null"] > 0.01
        lo, hi = s["decisive_ci95_pp"] if s["decisive_n_objectives"] else (0, 0)
        assert lo <= 0.0 <= hi
        assert abs(s["decisive_lift_pp"]) < 5.0


def test_planted_signal_survives_the_split():
    data, planted = _fixture(300, 40, 0.3, 0.9, 0.3)
    folds = split_folds(data, SEED)
    hit = sum(1 for o in planted if select_on_fold(folds[o], 0, 0.05) == (True, "r1"))
    assert hit >= 0.9 * len(planted)
    s = summarize(split_policy_lift(data, 0.05, 0, folds=folds))
    assert s["decisive_ci95_pp"][0] > 20.0
    assert s["selection_null"]["p_selected_exceeds_null"] < 1e-6
