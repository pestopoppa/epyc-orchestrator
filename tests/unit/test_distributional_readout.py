"""New, hand-computable fixtures for the Table-3 clean-room math port.

These are independently derived from arXiv:2503.03064v2, not the historical
intake-1169 vectors. EV-15a remains open until those original vectors are found.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from src.typed_decisions.distributional_readout import (
    first_percentile,
    mean,
    median,
    mode,
    probability_of_superiority,
    quantile_integral,
    risk_averse_mean,
    rounded_mean,
)

READOUTS = (
    mode,
    mean,
    rounded_mean,
    median,
    first_percentile,
    risk_averse_mean,
    quantile_integral,
    probability_of_superiority,
)


def _f64(values):
    return np.asarray(values, dtype=np.float64)


def test_basic_formulas_match_hand_calculation():
    # New K=2 fixture. Mode/rounded mean/median favor p1; the 1st-percentile
    # result ties because both 1% quantiles are score 1.
    p1 = _f64([0.25, 0.75])
    p2 = _f64([0.75, 0.25])
    assert mode(p1, p2).item() == 1.0
    assert mean(p1, p2).item() == pytest.approx(0.5 / (0.625 + sqrt(0.375)))
    assert rounded_mean(p1, p2).item() == 1.0
    assert median(p1, p2).item() == 1.0
    assert first_percentile(p1, p2).item() == 0.0
    # Independent win mass is 9/16, loss mass is 1/16.
    assert probability_of_superiority(p1, p2).item() == pytest.approx(0.5)
    # For this two-point pair, comonotone quantile agreement also has net +1/2.
    assert quantile_integral(p1, p2).item() == pytest.approx(0.5)


def test_new_k5_ram_fixture_flips_sign_against_mean():
    # New hand-derived fixture, not the K=5 intake-1169 vector:
    # X1 is half at 1 and half at 5; X2 is deterministic at 2.
    p1 = _f64([0.5, 0.0, 0.0, 0.0, 0.5])
    p2 = _f64([0.0, 1.0, 0.0, 0.0, 0.0])
    # Raw D has E[D]=1, E|D|=2, sigma(D)=2.
    assert mean(p1, p2).item() == pytest.approx(0.25)
    # lower-semideviation(X1)=sqrt(2), lower-semideviation(X2)=0.
    # The shifted mean comparator has E[D]=1-sqrt(2), E|D|=2, sigma(D)=2.
    assert risk_averse_mean(p1, p2).item() == pytest.approx((1.0 - sqrt(2.0)) / 4.0)
    assert mean(p1, p2).item() > 0.0 > risk_averse_mean(p1, p2).item()


def test_new_k9_quantile_and_independent_superiority_can_disagree_in_sign():
    # New hand-derived fixture, not the K=9 intake-1169 vector.
    # p1: 0.3 at 2, 0.7 at 9. p2: 0.2 at 1, 0.8 at 9.
    p1 = _f64([0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    p2 = _f64([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
    # Quantile coupling: +0.2 on [0,.2), -0.1 on [.2,.3), tie thereafter.
    assert quantile_integral(p1, p2).item() == pytest.approx(0.1)
    # Independent coupling: wins .06 + .14, losses .24, net -.04.
    assert probability_of_superiority(p1, p2).item() == pytest.approx(-0.04)
    assert quantile_integral(p1, p2).item() > 0 > probability_of_superiority(p1, p2).item()


def test_superiority_uses_only_actual_upper_tail_and_is_antisymmetric():
    # The accepted float64 simplex tolerance can admit a tiny residual. For
    # K=1 every pair ties, so that residual must not become invented upper mass.
    tied_p1 = _f64([1.0])
    tied_p2 = _f64([1.0 - 5e-13])
    assert probability_of_superiority(tied_p1, tied_p2).item() == 0.0

    p1 = _f64([0.25, 0.75])
    p2 = _f64([0.75, 0.25])
    before1 = p1.copy()
    before2 = p2.copy()
    forward = probability_of_superiority(p1, p2)
    reverse = probability_of_superiority(p2, p1)
    assert forward.item() == pytest.approx(0.5)
    assert reverse.item() == pytest.approx(-0.5)
    assert forward.item() == pytest.approx(-reverse.item())
    np.testing.assert_array_equal(p1, before1)
    np.testing.assert_array_equal(p2, before2)


def test_arbitrary_leading_batch_dimensions_and_float64_are_preserved():
    p1_row = _f64([0.25, 0.75])
    p2_row = _f64([0.75, 0.25])
    p1 = np.broadcast_to(p1_row, (2, 3, 2)).copy()
    p2 = np.broadcast_to(p2_row, (2, 3, 2)).copy()
    before1 = p1.copy()
    before2 = p2.copy()
    for readout in READOUTS:
        result = readout(p1, p2)
        assert result.shape == (2, 3)
        assert result.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(p1, before1)
        np.testing.assert_array_equal(p2, before2)
        np.testing.assert_allclose(result, np.broadcast_to(readout(p1_row, p2_row), (2, 3)))


def test_float32_and_float64_are_not_silently_cast():
    p1 = np.asarray([0.25, 0.75], dtype=np.float32)
    p2 = np.asarray([0.75, 0.25], dtype=np.float32)
    for readout in READOUTS:
        assert readout(p1, p2).dtype == np.dtype(np.float32)
    assert mean(_f64([0.5, 0.5]), _f64([0.0, 1.0])).dtype == np.dtype(np.float64)


def test_exact_mode_and_rounded_mean_ties_refuse_underspecified_cases():
    with pytest.raises(ValueError, match="tied maxima"):
        mode(_f64([0.5, 0.5]), _f64([0.0, 1.0]))
    # Mean 1.5 is exactly equidistant from scores 1 and 2.
    with pytest.raises(ValueError, match="nearest-score tie"):
        rounded_mean(_f64([0.5, 0.5]), _f64([0.0, 1.0]))


def test_degenerate_and_single_score_distributions_are_defined():
    p1 = _f64([1.0])
    p2 = _f64([1.0])
    assert mode(p1, p2).item() == 0.0
    assert mean(p1, p2).item() == 0.0  # paper explicitly defines 0/0 as zero
    assert rounded_mean(p1, p2).item() == 0.0
    assert median(p1, p2).item() == 0.0
    assert first_percentile(p1, p2).item() == 0.0
    assert risk_averse_mean(p1, p2).item() == 0.0
    assert quantile_integral(p1, p2).item() == 0.0
    assert probability_of_superiority(p1, p2).item() == 0.0


def test_invalid_probabilities_and_shape_dtype_mismatches_refuse():
    valid = _f64([0.5, 0.5])
    invalid_rows = (
        _f64([0.4, 0.4]),
        _f64([0.0, 0.0]),
        _f64([float("nan"), 1.0]),
        _f64([-0.1, 1.1]),
    )
    for invalid in invalid_rows:
        with pytest.raises(ValueError):
            mean(invalid, valid)
    with pytest.raises(ValueError, match="identical shapes"):
        mean(valid, _f64([0.2, 0.3, 0.5]))
    with pytest.raises(ValueError, match="same dtype"):
        mean(valid, np.asarray([0.5, 0.5], dtype=np.float32))
    with pytest.raises(TypeError, match="float32 and float64"):
        mean(np.asarray([0.5, 0.5], dtype=np.float16), np.asarray([0.5, 0.5], dtype=np.float16))
    with pytest.raises(TypeError, match="NumPy arrays"):
        mean([0.5, 0.5], [0.5, 0.5])


def test_sum_tolerance_is_bounded_for_large_k_and_rejects_bad_mass():
    k = 100_000
    zeros = np.zeros(k, dtype=np.float32)
    valid = np.zeros(k, dtype=np.float32)
    valid[0] = 1.0
    with pytest.raises(ValueError, match="sum to one"):
        mean(zeros, valid)

    materially_unnormalized = np.full(k, 1.1 / k, dtype=np.float32)
    with pytest.raises(ValueError, match="sum to one"):
        mean(materially_unnormalized, valid)

    # The cap permits ordinary float32 representation error without widening
    # acceptance as K grows; this vector's represented sum is near one.
    near_simplex = np.full(k, 1.0 / k, dtype=np.float32)
    assert abs(float(near_simplex.sum()) - 1.0) <= 1e-6
    # Use a linear-time readout here; mean's exact independent-pair formula
    # needs O(K²) pair support and the fixture is deliberately large.
    assert first_percentile(near_simplex, valid).dtype == np.dtype(np.float32)
    assert first_percentile(near_simplex, valid).item() == 1.0
