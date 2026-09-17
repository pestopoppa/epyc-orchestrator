"""Local confidence statistics for the typed decision plane (intake-1473).

Three pure, dependency-free helpers:

  * ``normalize_probabilities`` — clamp/rescale a raw candidate->probability
    mapping into a valid distribution (uniform fallback for a zero total).
  * ``choice_confidence`` — how far a categorical distribution has moved
    from uniform toward one-hot, rescaled to ``[0, 1]``.
  * ``score_confidence`` — how concentrated an ordinal distribution is
    around its mode, relative to the spread of the uniform distribution
    over the same levels, floored at ``0``.

THESE ARE LOCAL STATISTICS, NOT CALIBRATED CONFIDENCE. A value of ``0.64``
means "this distribution is 0.64 of the way from uniform to one-hot under
the adapter's normalization" — it is NOT a statement that the answer is
correct with probability 0.64. Using these numbers as probabilities in
downstream arithmetic requires an explicit calibration step against labeled
outcomes (ECE / reliability curves); nothing in this module claims that
step has been taken.

The formulas are adapter-verified from intake-1473 (the typed-decision
adapter spec), and are reproduced here exactly:

    choice_confidence(p) = (max(p) - 1/n) / (1 - 1/n)
        uniform -> 0.0; [0.82, 0.18] -> 0.64; single outcome -> 1.0

    score_confidence(p)  = max(0, 1 - sum_i p_i*|i - mode|
                                   / uniform_mean_absolute_deviation)
        one-hot -> 1.0; uniform -> 0.0; the denominator is the mean
        absolute deviation of the uniform distribution over the candidate
        levels around its mean.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def normalize_probabilities(
    probabilities: Mapping[str | int, float],
) -> dict[str | int, float]:
    """Rescale a raw candidate->probability mapping into a valid distribution.

    Non-finite and negative values are treated as ``0.0``. The surviving
    values are divided by their total, so the result sums to ``1.0``
    (within float rounding). If the total is zero — every value missing,
    zero, negative or non-finite — the result falls back to the uniform
    distribution over the same keys. An empty mapping stays empty (there is
    no candidate set to be uniform over); callers that require candidates
    treat that as ``invalid_value``, never as a default answer.
    """
    cleaned: dict[str | int, float] = {}
    total = 0.0
    for key, value in probabilities.items():
        try:
            p = float(value)
        except (TypeError, ValueError):
            p = 0.0
        if not math.isfinite(p) or p < 0.0:
            p = 0.0
        cleaned[key] = p
        total += p
    count = len(cleaned)
    if count == 0:
        return {}
    if total <= 0.0:
        uniform = 1.0 / count
        return {key: uniform for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def choice_confidence(probabilities: Mapping[str | int, float] | Sequence[float]) -> float:
    """Rescaled margin above uniform for a categorical distribution.

    ``(max(p) - 1/n) / (1 - 1/n)``. A uniform distribution scores ``0.0``,
    a one-hot distribution ``1.0``, and ``[0.82, 0.18]`` scores ``0.64``.
    A single-outcome distribution is degenerate and scores ``1.0``; an
    empty input scores ``0.0``. The result is clamped to ``[0, 1]`` so
    unnormalized input cannot escape the range.
    """
    values = (
        [float(value) for value in probabilities.values()]
        if isinstance(probabilities, Mapping)
        else [float(value) for value in probabilities]
    )
    count = len(values)
    if count == 0:
        return 0.0
    if count == 1:
        return 1.0
    uniform = 1.0 / count
    scaled = (max(values) - uniform) / (1.0 - uniform)
    return min(1.0, max(0.0, scaled))


def score_confidence(probabilities: Mapping[int, float] | Sequence[float]) -> float:
    """Concentration of an ordinal distribution around its mode.

    ``max(0, 1 - expected_absolute_deviation_from_mode /
    uniform_mean_absolute_deviation)`` where the uniform mean absolute
    deviation is measured over the same candidate levels around their mean.
    A one-hot distribution scores ``1.0``; a uniform distribution clamps to
    ``0.0``. Mapping input keys are the levels; sequence input is indexed
    ``0..n-1``. Non-finite entries are dropped, negative totals score
    ``0.0``, and a degenerate (<= 1 level) candidate set scores ``1.0``.
    """
    if isinstance(probabilities, Mapping):
        if not probabilities:
            return 0.0
        levels = [int(key) for key in probabilities]
        values = [float(value) for value in probabilities.values()]
    else:
        values = [float(value) for value in probabilities]
        levels = list(range(len(values)))

    finite = [
        (level, value)
        for level, value in zip(levels, values)
        if math.isfinite(value) and value >= 0.0
    ]
    if len(finite) <= 1:
        return 1.0
    levels = [level for level, _ in finite]
    values = [value for _, value in finite]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    values = [value / total for value in values]

    mode_index = max(range(len(values)), key=lambda index: values[index])
    mode_level = levels[mode_index]
    expected_deviation = sum(
        value * abs(level - mode_level) for level, value in zip(levels, values)
    )
    uniform_deviation = _uniform_mean_absolute_deviation(levels)
    if uniform_deviation <= 0.0:
        return 1.0
    scaled = 1.0 - expected_deviation / uniform_deviation
    return min(1.0, max(0.0, scaled))


def _uniform_mean_absolute_deviation(levels: Sequence[int]) -> float:
    """Mean absolute deviation of the uniform distribution over ``levels``.

    ``(1/n) * sum_i |level_i - mean(levels)|``. Returns ``0.0`` for a
    single level, which callers map to full confidence (nothing to be
    uncertain about).
    """
    count = len(levels)
    if count <= 1:
        return 0.0
    mean = sum(levels) / count
    return sum(abs(level - mean) for level in levels) / count
