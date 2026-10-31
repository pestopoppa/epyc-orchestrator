"""Pure distributional readouts from the paper's Table 3.

Formulas are a clean-room transcription of Wang, Zhang, and Choi,
"Improving LLM-as-a-Judge Inference with the Judgment Distribution," arXiv:
2503.03064v2, Table 3 and Appendix B.1 (CC BY 4.0):
https://arxiv.org/html/2503.03064v2

Inputs are NumPy probability arrays over the ordered score support 1..K,
along the last dimension. This module is a math instrument only; it performs
no model readout, prompt handling, aggregation, or judge integration.

The paper leaves exact ties in mode and rounded-mean nearest-score selection
unspecified. These functions refuse such rows instead of silently selecting
one of the tied scores. Quantiles use the conventional left inverse
inf{x: F(x) >= q}; within the bounded floating-point simplex tolerance, the
inverse CDF closes at the final supported score without normalizing or
mutating the supplied probabilities.
"""

from __future__ import annotations

import numpy as np

_SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_MAX_SUM_ABS_TOLERANCE = 1e-6
_MIN_SUM_ABS_TOLERANCE = 1e-12


def _validate_probabilities(p1: np.ndarray, p2: np.ndarray) -> int:
    if not isinstance(p1, np.ndarray) or not isinstance(p2, np.ndarray):
        raise TypeError("p1 and p2 must be NumPy arrays")
    if p1.ndim < 1 or p2.ndim < 1:
        raise ValueError("probability arrays must have a score axis")
    if p1.shape != p2.shape:
        raise ValueError("p1 and p2 must have identical shapes")
    if p1.dtype != p2.dtype:
        raise ValueError("p1 and p2 must have the same dtype")
    if p1.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("only float32 and float64 probabilities are supported")

    k = p1.shape[-1]
    if k < 1:
        raise ValueError("the score axis must contain at least one option")

    for name, probabilities in (("p1", p1), ("p2", p2)):
        if not bool(np.isfinite(probabilities).all()):
            raise ValueError(f"{name} must contain only finite probabilities")
        if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
            raise ValueError(f"{name} probabilities must be in [0, 1]")
        sums = probabilities.sum(axis=-1)
        eps = np.finfo(probabilities.dtype).eps
        tolerance = min(_MAX_SUM_ABS_TOLERANCE, max(_MIN_SUM_ABS_TOLERANCE, 10.0 * eps * k))
        if bool((np.abs(sums - 1.0) > tolerance).any()):
            raise ValueError(f"each {name} row must sum to one (tolerance={tolerance:g})")
    return k


def _scores(probabilities: np.ndarray) -> np.ndarray:
    return np.arange(1, probabilities.shape[-1] + 1, dtype=probabilities.dtype)


def _mean_on_supports(
    p1: np.ndarray,
    p2: np.ndarray,
    scores1: np.ndarray,
    scores2: np.ndarray,
) -> np.ndarray:
    """Mean comparison with independent distributions on possibly shifted supports."""
    mean1 = (p1 * scores1).sum(axis=-1)
    mean2 = (p2 * scores2).sum(axis=-1)
    expected_difference = mean1 - mean2

    difference = scores1[..., :, None] - scores2[..., None, :]
    joint = p1[..., :, None] * p2[..., None, :]
    expected_absolute_difference = (joint * np.abs(difference)).sum(axis=(-1, -2))

    variance1 = (p1 * np.square(scores1 - mean1[..., None])).sum(axis=-1)
    variance2 = (p2 * np.square(scores2 - mean2[..., None])).sum(axis=-1)
    standard_deviation_difference = np.sqrt(np.maximum(variance1 + variance2, 0.0))
    denominator = expected_absolute_difference + standard_deviation_difference
    nonzero = denominator != 0
    safe_denominator = np.where(nonzero, denominator, np.ones_like(denominator))
    quotient = expected_difference / safe_denominator
    return np.where(nonzero, quotient, np.zeros_like(quotient))


def mode(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return sign(argmax score of p1 − argmax score of p2); refuse tied modes."""
    k = _validate_probabilities(p1, p2)
    maxima1 = p1 == p1.max(axis=-1, keepdims=True)
    maxima2 = p2 == p2.max(axis=-1, keepdims=True)
    if bool((maxima1.sum(axis=-1) != 1).any()) or bool((maxima2.sum(axis=-1) != 1).any()):
        raise ValueError("mode is undefined by the paper for a distribution with tied maxima")
    indices = np.arange(k)
    index1 = (maxima1 * indices).sum(axis=-1)
    index2 = (maxima2 * indices).sum(axis=-1)
    return np.sign((index1 - index2).astype(dtype=p1.dtype, copy=False))


def mean(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Table-3 mean comparator, with independent score draws."""
    _validate_probabilities(p1, p2)
    scores = _scores(p1)
    return _mean_on_supports(p1, p2, scores, scores)


def rounded_mean(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compare each expected score's nearest support value; refuse half-way ties."""
    _validate_probabilities(p1, p2)
    scores = _scores(p1)
    means = ((p1 * scores).sum(axis=-1), (p2 * scores).sum(axis=-1))
    rounded = []
    for distribution_mean in means:
        distances = np.abs(distribution_mean[..., None] - scores)
        minimum = distances.min(axis=-1, keepdims=True)
        nearest = distances == minimum
        if bool((nearest.sum(axis=-1) != 1).any()):
            raise ValueError(
                "rounded mean is undefined by the paper for an exact nearest-score tie"
            )
        rounded.append((nearest * np.arange(1, scores.size + 1)).sum(axis=-1))
    difference = (rounded[0] - rounded[1]).astype(dtype=p1.dtype, copy=False)
    return np.sign(difference)


def _quantile_indices(probabilities: np.ndarray, q: float) -> np.ndarray:
    cumulative = probabilities.cumsum(axis=-1)
    # Count cumulative values strictly below q: this is the left inverse
    # inf{x : F(x) >= q}. Clamp only the tolerated floating-point simplex tail.
    indices = (cumulative < q).sum(axis=-1)
    return np.minimum(indices, probabilities.shape[-1] - 1)


def median(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compare left-inverse 0.5 quantiles."""
    _validate_probabilities(p1, p2)
    q1 = _quantile_indices(p1, 0.5)
    q2 = _quantile_indices(p2, 0.5)
    return np.sign((q1 - q2).astype(dtype=p1.dtype, copy=False))


def first_percentile(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compare left-inverse 0.01 quantiles."""
    _validate_probabilities(p1, p2)
    q1 = _quantile_indices(p1, 0.01)
    q2 = _quantile_indices(p2, 0.01)
    return np.sign((q1 - q2).astype(dtype=p1.dtype, copy=False))


def risk_averse_mean(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Shift each score support by lower semideviation, then recompute mean."""
    _validate_probabilities(p1, p2)
    scores = _scores(p1)
    means1 = (p1 * scores).sum(axis=-1)
    means2 = (p2 * scores).sum(axis=-1)
    lower1 = np.sqrt((p1 * np.square(np.maximum(means1[..., None] - scores, 0.0))).sum(axis=-1))
    lower2 = np.sqrt((p2 * np.square(np.maximum(means2[..., None] - scores, 0.0))).sum(axis=-1))
    shifted1 = scores - lower1[..., None]
    shifted2 = scores - lower2[..., None]
    # RAM is Table-3 mean applied to the shifted supports. Thus its E|D|
    # denominator is recomputed after shifting, not reused from raw scores.
    return _mean_on_supports(p1, p2, shifted1, shifted2)


def quantile_integral(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Integrate sign differences of left-inverse quantiles over [0, 1]."""
    k = _validate_probabilities(p1, p2)
    original_shape = p1.shape[:-1]
    flat1 = p1.reshape(-1, k)
    flat2 = p2.reshape(-1, k)
    if flat1.shape[0] == 0:
        return np.empty(original_shape, dtype=p1.dtype)

    cdf1 = np.clip(flat1.cumsum(axis=-1), 0.0, 1.0)
    cdf2 = np.clip(flat2.cumsum(axis=-1), 0.0, 1.0)
    results = np.empty((flat1.shape[0],), dtype=p1.dtype)
    for row in range(flat1.shape[0]):
        breakpoints = np.sort(np.concatenate(([0.0], cdf1[row], cdf2[row], [1.0])))
        lower = breakpoints[:-1]
        upper = breakpoints[1:]
        widths = upper - lower
        midpoints = (lower + upper) * 0.5
        index1 = np.minimum(np.searchsorted(cdf1[row], midpoints, side="left"), k - 1)
        index2 = np.minimum(np.searchsorted(cdf2[row], midpoints, side="left"), k - 1)
        signed = np.sign((index1 - index2).astype(dtype=p1.dtype, copy=False))
        results[row] = (widths * signed).sum(dtype=p1.dtype)
    return results.reshape(original_shape)


def probability_of_superiority(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return P(X1 > X2) − P(X1 < X2) for independent score draws."""
    _validate_probabilities(p1, p2)
    below = np.concatenate((np.zeros_like(p2[..., :1]), p2.cumsum(axis=-1)[..., :-1]), axis=-1)
    reverse_cumulative = p2[..., ::-1].cumsum(axis=-1)[..., ::-1]
    above = np.concatenate((reverse_cumulative[..., 1:], np.zeros_like(p2[..., :1])), axis=-1)
    wins = (p1 * below).sum(axis=-1)
    losses = (p1 * above).sum(axis=-1)
    return wins - losses
