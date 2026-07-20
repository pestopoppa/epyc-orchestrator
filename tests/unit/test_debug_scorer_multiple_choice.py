"""Regression coverage for multiple-choice textual labels."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from debug_scorer import score_answer  # noqa: E402


def test_multiple_choice_accepts_configured_textual_label() -> None:
    assert score_answer(
        answer="**incorrect**",
        expected="incorrect",
        scoring_method="multiple_choice",
        scoring_config={"choices": ["correct", "incorrect"]},
    )


def test_multiple_choice_maps_letter_to_configured_textual_expected() -> None:
    assert score_answer(
        answer="Answer: B",
        expected="incorrect",
        scoring_method="multiple_choice",
        scoring_config={"choices": ["correct", "incorrect"]},
    )
    assert not score_answer(
        answer="Answer: A",
        expected="incorrect",
        scoring_method="multiple_choice",
        scoring_config={"choices": ["correct", "incorrect"]},
    )


def test_multiple_choice_accepts_parenthesized_expected_letter() -> None:
    assert score_answer(
        answer="Answer: B",
        expected="(B)",
        scoring_method="multiple_choice",
        scoring_config={},
    )


def test_multiple_choice_prefers_longer_overlapping_text_choice() -> None:
    choices = ["cat", "black cat"]

    assert score_answer(
        answer="black cat",
        expected="black cat",
        scoring_method="multiple_choice",
        scoring_config={"choices": choices},
    )
    assert not score_answer(
        answer="black cat",
        expected="cat",
        scoring_method="multiple_choice",
        scoring_config={"choices": choices},
    )


def test_multiple_choice_prefers_containing_choice_with_same_end() -> None:
    choices = ["None", "None of the above"]

    assert score_answer(
        answer="None of the above",
        expected="None of the above",
        scoring_method="multiple_choice",
        scoring_config={"choices": choices},
    )
    assert not score_answer(
        answer="None of the above",
        expected="None",
        scoring_method="multiple_choice",
        scoring_config={"choices": choices},
    )


def test_multiple_choice_requires_choices_for_textual_expected() -> None:
    assert not score_answer(
        answer="**incorrect**",
        expected="incorrect",
        scoring_method="multiple_choice",
        scoring_config={},
    )
