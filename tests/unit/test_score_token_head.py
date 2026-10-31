"""EV-15d keeps each call's raw score-token mass before readout normalization."""

import math

import pytest

from src.typed_decisions.score_token_head import (
    ROUNDING_TOLERANCE,
    score_token_head_from_top_logprobs,
)


SCORE_IDS = tuple(range(100, 109))


def entries(probabilities, *, omit=()):
    return [
        {"id": token_id, "prob": probability}
        for token_id, probability in zip(SCORE_IDS, probabilities, strict=True)
        if token_id not in omit
    ]


def test_complete_head_records_mass_before_post_gate_normalization():
    result = score_token_head_from_top_logprobs(
        entries([0.1] * 8 + [0.2]),
        SCORE_IDS,
        minimum_retained_mass=0.95,
    )

    assert result.status == "accepted"
    assert result.retained_mass == pytest.approx(1.0)
    assert result.missing_token_ids == ()
    assert result.probabilities == pytest.approx((0.1,) * 8 + (0.2,))
    assert result.to_dict()["schema"] == "ev15-score-token-head.v1"


def test_underfloor_refusal_keeps_mass_and_does_not_create_judgment():
    result = score_token_head_from_top_logprobs(
        entries([0.1] * 8 + [0.0]),
        SCORE_IDS,
        minimum_retained_mass=0.9,
    )

    assert result.status == "refused"
    assert result.retained_mass == pytest.approx(0.8)
    assert result.probabilities is None
    assert result.refusal_reason == "below_retained_mass_floor"


def test_missing_score_token_is_zero_and_mass_is_checked_before_normalization():
    values = [0.2] * 5 + [0.0] * 4
    result = score_token_head_from_top_logprobs(
        entries(values, omit=(SCORE_IDS[8],)),
        SCORE_IDS,
        minimum_retained_mass=0.9,
    )

    assert result.status == "accepted"
    assert result.missing_token_ids == (SCORE_IDS[8],)
    assert result.unnormalized_probabilities[-1] == 0.0
    assert result.retained_mass == pytest.approx(1.0)
    assert result.probabilities == pytest.approx((0.2,) * 5 + (0.0,) * 4)


def test_zero_mass_refuses_even_when_floor_is_zero():
    result = score_token_head_from_top_logprobs(
        entries([0.0] * 9),
        SCORE_IDS,
        minimum_retained_mass=0.0,
    )
    assert result.status == "refused"
    assert result.retained_mass == 0.0
    assert result.probabilities is None
    assert result.refusal_reason == "zero_retained_mass"


def test_all_score_tokens_missing_refuses_with_zero_mass_and_full_missing_list():
    result = score_token_head_from_top_logprobs(
        [],
        SCORE_IDS,
        minimum_retained_mass=0.0,
    )
    assert result.status == "refused"
    assert result.retained_mass == 0.0
    assert result.missing_token_ids == SCORE_IDS
    assert result.probabilities is None
    assert result.refusal_reason == "zero_retained_mass"


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"])
def test_invalid_floors_refuse(bad):
    with pytest.raises((TypeError, ValueError), match="minimum_retained_mass"):
        score_token_head_from_top_logprobs(
            entries([1.0] + [0.0] * 8), SCORE_IDS, minimum_retained_mass=bad
        )


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"])
def test_invalid_probabilities_refuse(bad):
    with pytest.raises((TypeError, ValueError)):
        score_token_head_from_top_logprobs(
            [{"id": SCORE_IDS[0], "prob": bad}],
            SCORE_IDS,
            minimum_retained_mass=0.0,
        )


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 0.1, "-0.1"])
def test_invalid_logprobs_refuse(bad):
    with pytest.raises((TypeError, ValueError)):
        score_token_head_from_top_logprobs(
            [{"id": SCORE_IDS[0], "logprob": bad}],
            SCORE_IDS,
            minimum_retained_mass=0.0,
        )


def test_duplicate_head_ids_are_rejected_even_outside_score_support():
    rows = entries([1.0] + [0.0] * 8)
    rows.extend([{"id": 999, "prob": 0.0}, {"id": 999, "prob": 0.0}])
    with pytest.raises(ValueError, match="duplicate token ID"):
        score_token_head_from_top_logprobs(rows, SCORE_IDS, minimum_retained_mass=0.9)


def test_duplicate_or_boolean_score_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        score_token_head_from_top_logprobs(
            entries([1.0] + [0.0] * 8), SCORE_IDS[:-1] + (SCORE_IDS[0],), minimum_retained_mass=0.9
        )
    with pytest.raises(TypeError, match="integer"):
        score_token_head_from_top_logprobs(
            entries([1.0] + [0.0] * 8), (True, *SCORE_IDS[1:]), minimum_retained_mass=0.9
        )


def test_head_mass_above_one_only_tolerates_small_rounding_error():
    within = score_token_head_from_top_logprobs(
        entries([0.5, 0.5 + ROUNDING_TOLERANCE / 2] + [0.0] * 7),
        SCORE_IDS,
        minimum_retained_mass=0.9,
    )
    assert within.status == "accepted"
    assert within.retained_mass > 1.0

    with pytest.raises(ValueError, match="head probability mass"):
        score_token_head_from_top_logprobs(
            entries([0.5, 0.5 + ROUNDING_TOLERANCE * 2] + [0.0] * 7),
            SCORE_IDS,
            minimum_retained_mass=0.9,
        )


def test_logprob_head_entries_are_converted_without_pre_mass_normalization():
    rows = [{"id": token_id, "logprob": -3.0} for token_id in SCORE_IDS]
    result = score_token_head_from_top_logprobs(
        rows,
        SCORE_IDS,
        minimum_retained_mass=0.4,
    )
    expected_mass = 9 * math.exp(-3.0)
    assert result.status == "accepted"
    assert result.retained_mass == pytest.approx(expected_mass)
