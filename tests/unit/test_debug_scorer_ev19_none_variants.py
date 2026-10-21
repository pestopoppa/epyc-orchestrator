"""EV-19.5: tulving_episodic empty-gold controls accept normalised 'None' variants.

Operator decision 2026-10-21 ("Accept variants"): a reply that is, as a WHOLE, a bare
abstention is correct on an empty-gold control, modulo a leading bullet, an ``Answer:``
prefix, trailing punctuation and case/whitespace. ``None`` buried in a longer answer, or any
reply that names events, must still fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from debug_scorer import score_answer  # noqa: E402

CFG = {"normalize": True, "threshold": 0.5}


def _control(answer: str) -> bool:
    return score_answer(answer, "[]", "f1_list", CFG)


ACCEPTED = [
    "None",
    "none",
    "NONE",
    "  None  ",
    "\nNone\n",
    "None.",
    "None!",
    "none;",
    "- None",
    "* None",
    "• None",
    "-None",
    "  - none.  ",
    "Answer: None",
    "answer:none",
    "ANSWER: NONE.",
    "Answer:\nNone",
    "- Answer: None",
    "Answer: - None.",
    "N/A",
    "- n/a.",
]

REJECTED = [
    "None of the above events matched.",
    "- None of the above",
    "Answer: None of the events",
    "None, except the dinner at Mario's.",
    "None\n- Dinner at Mario's",
    "- None\n- Board meeting",
    "- Dinner at Mario's",
    "Answer: Dinner at Mario's",
    "Answer: None. The events were: Dinner at Mario's",
    "There was none.",
    "Not none",
    "",
]


@pytest.mark.parametrize("answer", ACCEPTED)
def test_empty_gold_accepts_normalised_none(answer: str) -> None:
    assert _control(answer) is True


@pytest.mark.parametrize("answer", [a for a in REJECTED if a.strip()])
def test_empty_gold_rejects_none_inside_longer_answer_or_named_events(answer: str) -> None:
    assert _control(answer) is False


def test_empty_gold_blank_reply_is_still_parse_failure_not_pass() -> None:
    # Unchanged behaviour: nothing to compare is a model-side parse failure (excluded),
    # never a silent pass. The normaliser must not turn "" into an abstention.
    from debug_scorer import ScoringUnavailableError

    try:
        out = _control("")
    except ScoringUnavailableError:
        return
    assert out is False


def test_variants_do_not_change_non_empty_gold() -> None:
    gold = '["High Line"]'
    assert score_answer("- None", gold, "f1_list", CFG) is False
    assert score_answer("Answer: None.", gold, "f1_list", CFG) is False
    assert score_answer("- High Line", gold, "f1_list", CFG) is True
    assert score_answer("Answer: High Line", gold, "f1_list", CFG) is True
