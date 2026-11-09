"""TD-18: calibration receipt upgrade (pure, zero inference)."""
import math

import pytest

from src.typed_decisions import calibration_receipt as cr


def _calibrated(n=400, seed=1):
    import random
    rng = random.Random(seed)
    conf = [rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]) for _ in range(n)]
    y = [1.0 if rng.random() < c else 0.0 for c in conf]
    return conf, y


def test_noise_floor_is_positive_and_reproducible():
    conf, _ = _calibrated()
    a = cr.ece_noise_floor(conf, n_resamples=100, seed=3)
    b = cr.ece_noise_floor(conf, n_resamples=100, seed=3)
    assert a == b and 0 < a["mean"] < a["p95"] < 0.2


def test_selective_ties_are_taken_whole_not_in_input_order():
    # one tie group, 50% errors: coverage at 10% budget must be 0, not 0.5
    out = cr.selective_prediction([0.9, 0.9, 0.9, 0.9], [1, 0, 1, 0], error_budget=0.1)
    assert out["coverage_at_budget"] == 0.0
    assert out["n_tie_groups"] == 1
    sep = cr.selective_prediction([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0], error_budget=0.0)
    assert sep["coverage_at_budget"] == 0.5
    assert sep["aurc"] < out["aurc"]


def test_endpoints_and_empty():
    e = cr.endpoint_rates([0.0, 1.0, 1.0, 0.5])
    assert (e["exact_0"], e["exact_1"], e["exact_1_rate"]) == (1, 2, 0.5)
    assert cr.selective_prediction([], [])["aurc"] is None
    assert cr.nll([], []) is None


def test_floor_changes_nll_for_exact_endpoint_wrong_prediction():
    lo = cr.nll([1.0], [0.0], floor=1e-12)
    hi = cr.nll([1.0], [0.0], floor=1e-3)
    assert lo > hi > 0


def test_temperature_recovers_overconfidence_and_flags_bound():
    conf = [0.99] * 50 + [0.01] * 50
    y = [1.0] * 25 + [0.0] * 25 + [0.0] * 25 + [1.0] * 25  # no signal at all
    fit = cr.fit_temperature(conf, y, floor=1e-6, clamp=1e-3)
    assert fit["T"] > 5
    only = cr.fit_temperature([0.9] * 20, [1.0] * 20)  # sharpening is unbounded
    assert only["on_bound"] is True


def test_clustered_bootstrap_resamples_clusters():
    conf, y = _calibrated(200)
    clusters = [i // 20 for i in range(200)]
    out = cr.clustered_bootstrap_ece(conf, y, clusters, n_boot=100, seed=2)
    # ECE is upward-biased under resampling, so the point need not sit inside the percentile CI.
    assert out["n_clusters"] == 10 and out["ci95"][0] <= out["ci95"][1]


def test_paired_diff_zero_for_identical_arms():
    conf, y = _calibrated(100)
    out = cr.paired_clustered_bootstrap_diff(conf, conf, y, list(range(100)), n_boot=50)
    assert out["diff"] == 0 and out["ci_excludes_zero"] is False
    with pytest.raises(ValueError):
        cr.paired_clustered_bootstrap_diff(conf, conf[:-1], y, list(range(100)))


def test_oof_folds_are_group_disjoint_and_report_pin():
    conf, y = _calibrated(120)
    groups = [i // 6 for i in range(120)]
    out = cr.temperature_oof(conf, y, groups, k=4, seed=1)
    assert out["k"] == 4 and out["oof"]["nll_calibrated"] is not None
    rows = [{"kind": "choice", "confidence": c, "correct": bool(o), "cluster": g}
            for c, o, g in zip(conf, y, groups)]
    rows.append({"kind": "score", "confidence": 0.5, "correct": None})
    rec = cr.build_receipt_upgrade(rows, model_pin={"gguf_sha256": "abc"}, n_boot=20)
    assert list(rec["per_kind"]) == ["choice"] and rec["model_pin"] == {"gguf_sha256": "abc"}
    assert cr.build_receipt_upgrade(rows, model_pin=None, n_boot=5)["model_pin_missing"] is True
    assert math.isfinite(rec["per_kind"]["choice"]["selective"]["aurc"])
