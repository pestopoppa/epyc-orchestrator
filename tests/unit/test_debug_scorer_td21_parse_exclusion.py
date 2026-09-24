"""TD-21.11..21.14: model-side parse failures vs. parsed-and-wrong.

Proves, for each converted site (multiple_choice, f1_list,
structural_exact_match, exact_match):

  1. An unparseable model answer is DETECTED (counted via
     ``parse_failure_stats()``) regardless of the exclusion flag.
  2. With ``EXCLUDE_UNPARSEABLE_ANSWERS`` False (the shipped default), the
     row scores exactly as before: ``False``, no exception — live scores are
     byte-identical to pre-TD-21.11..21.14 (see also the SCORE-25/26 golden
     fixtures in ``test_debug_scorer_score25_26.py`` and the B7 golden-corpus
     pin, both still green).
  3. With the flag True, the SAME unparseable input raises
     ``AnswerParseError`` (a ``ScoringUnavailableError`` subclass), routing
     through ``seeding_scoring.score_answer_or_error``'s existing
     EXCLUDED/``scoring_failed`` path.
  4. A cleanly PARSED-AND-WRONG answer (a real letter/list/marker/tag that
     just doesn't match gold) stays a plain wrong answer under BOTH flag
     settings — the flag only ever changes the unparseable case.
  5. The happy path (correct answer) is unaffected by the flag either way.

``handoffs/active/typed-decision-plane.md`` TD-21.11..21.14;
``artifacts/audits/td-json-consumer-audit-20260924.md`` J-03/J-05/J-06/J-07.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

import debug_scorer  # noqa: E402
from debug_scorer import (  # noqa: E402
    AnswerParseError,
    ScoringUnavailableError,
    score_answer,
)

# ── seeding_scoring, loaded the way test_seeding_scoring.py loads it: a
# private module identity so its own ScoringUnavailableError/AnswerParseError
# classes are self-consistent with the debug_scorer copy IT dynamically
# loads (mirrors seeding_scoring._load_orchestrator_debug_scorer). ──────────
_SS_ROOT = REPO_ROOT / "scripts" / "benchmark"
_SS_SPEC = importlib.util.spec_from_file_location(
    "seeding_scoring_td21_test", _SS_ROOT / "seeding_scoring.py"
)
_SS: ModuleType = importlib.util.module_from_spec(_SS_SPEC)
sys.modules["seeding_scoring_td21_test"] = _SS
_SS_SPEC.loader.exec_module(_SS)


@pytest.fixture(autouse=True)
def _reset_flag_and_stats():
    """Every test starts from the shipped default: flag OFF, counters zero."""
    debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = False
    debug_scorer.reset_parse_failure_stats()
    yield
    debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = False
    debug_scorer.reset_parse_failure_stats()


def test_flag_default_is_off():
    # The one-line flip a ratified EQ-1 era would make; must ship False.
    assert debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS is False


# ── multiple_choice (TD-21.11) ─────────────────────────────────────────────


def test_multiple_choice_unparseable_counts_and_scores_false_by_default():
    result = score_answer(
        answer="I refuse to pick a letter, this question is ambiguous.",
        expected="B",
        scoring_method="multiple_choice",
        scoring_config={},
    )
    assert result is False
    assert debug_scorer.parse_failure_stats().get("multiple_choice") == 1


def test_multiple_choice_unparseable_raises_when_flag_on():
    debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = True
    with pytest.raises(AnswerParseError):
        score_answer(
            answer="I refuse to pick a letter, this question is ambiguous.",
            expected="B",
            scoring_method="multiple_choice",
            scoring_config={},
        )
    assert debug_scorer.parse_failure_stats().get("multiple_choice") == 1


def test_multiple_choice_parsed_and_wrong_stays_wrong_under_both_flags():
    for flag in (False, True):
        debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = flag
        assert (
            score_answer("Answer: A", "B", "multiple_choice", {}) is False
        ), f"flag={flag}"
    assert "multiple_choice" not in debug_scorer.parse_failure_stats()


def test_multiple_choice_happy_path_unaffected_by_flag():
    for flag in (False, True):
        debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = flag
        assert score_answer("Answer: B", "B", "multiple_choice", {}) is True, f"flag={flag}"


# ── f1_list (TD-21.12) ──────────────────────────────────────────────────────

_F1_CFG = {"normalize": True, "threshold": 0.5}


def test_f1_list_unparseable_counts_and_scores_false_by_default():
    # A prose paragraph with no bullet/numbered/comma structure; the raw
    # line-split fallback produces junk that misses threshold.
    result = score_answer(
        answer="I am not sure which locations were mentioned in this story.",
        expected='["High Line", "Central Park"]',
        scoring_method="f1_list",
        scoring_config=_F1_CFG,
    )
    assert result is False
    assert debug_scorer.parse_failure_stats().get("f1_list") == 1


def test_f1_list_unparseable_raises_when_flag_on():
    debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = True
    with pytest.raises(AnswerParseError):
        score_answer(
            answer="I am not sure which locations were mentioned in this story.",
            expected='["High Line", "Central Park"]',
            scoring_method="f1_list",
            scoring_config=_F1_CFG,
        )


def test_f1_list_parsed_and_wrong_stays_wrong_under_both_flags():
    for flag in (False, True):
        debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = flag
        assert (
            score_answer("- totally unrelated nonsense", '["High Line"]', "f1_list", _F1_CFG)
            is False
        ), f"flag={flag}"
    assert "f1_list" not in debug_scorer.parse_failure_stats()


def test_f1_list_happy_path_unaffected_by_flag():
    for flag in (False, True):
        debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = flag
        assert (
            score_answer("- High Line", '["High Line"]', "f1_list", _F1_CFG) is True
        ), f"flag={flag}"


# ── structural_exact_match (TD-21.13) ───────────────────────────────────────

_ST_CFG = {"is_scorable": True, "extract_pattern": r"solution\s*=\s*(.+)"}


def test_structural_unparseable_counts_and_scores_false_by_default():
    result = score_answer(
        answer="the answer is 391365",  # no 'solution = ' marker at all
        expected="391365",
        scoring_method="structural_exact_match",
        scoring_config=_ST_CFG,
    )
    assert result is False
    assert debug_scorer.parse_failure_stats().get("structural_exact_match") == 1


def test_structural_unparseable_raises_when_flag_on():
    debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = True
    with pytest.raises(AnswerParseError):
        score_answer(
            answer="the answer is 391365",
            expected="391365",
            scoring_method="structural_exact_match",
            scoring_config=_ST_CFG,
        )


def test_structural_parsed_and_wrong_stays_wrong_under_both_flags():
    for flag in (False, True):
        debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = flag
        assert (
            score_answer("solution = 999999", "391365", "structural_exact_match", _ST_CFG)
            is False
        ), f"flag={flag}"
    assert "structural_exact_match" not in debug_scorer.parse_failure_stats()


def test_structural_happy_path_unaffected_by_flag():
    for flag in (False, True):
        debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = flag
        assert (
            score_answer("solution = 391365", "391365", "structural_exact_match", _ST_CFG)
            is True
        ), f"flag={flag}"


# ── exact_match's last-resort fallback (TD-21.14) ───────────────────────────


def test_exact_match_unparseable_counts_and_scores_false_by_default():
    # No <answer>/####/\boxed{} tag anywhere, and the raw final line (a
    # sentence, not a value) does not match `expected` under any of the
    # OCR-prose fallbacks either.
    result = score_answer(
        answer="I could not determine a final numeric answer from this problem.",
        expected="42",
        scoring_method="exact_match",
        scoring_config={},
    )
    assert result is False
    assert debug_scorer.parse_failure_stats().get("exact_match") == 1


def test_exact_match_unparseable_raises_when_flag_on():
    debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = True
    with pytest.raises(AnswerParseError):
        score_answer(
            answer="I could not determine a final numeric answer from this problem.",
            expected="42",
            scoring_method="exact_match",
            scoring_config={},
        )


def test_exact_match_structured_mismatch_stays_wrong_under_both_flags():
    # A real <answer> tag that simply disagrees with gold: never a parse
    # failure, regardless of the flag.
    for flag in (False, True):
        debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = flag
        assert (
            score_answer("<answer>7</answer>", "42", "exact_match", {}) is False
        ), f"flag={flag}"
    assert "exact_match" not in debug_scorer.parse_failure_stats()


def test_exact_match_happy_path_unaffected_by_flag():
    for flag in (False, True):
        debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = flag
        assert score_answer("<answer>42</answer>", "42", "exact_match", {}) is True, f"flag={flag}"


def test_exact_match_fallback_that_happens_to_match_is_not_a_parse_failure():
    # Existing golden behavior (test_debug_scorer_semantics.py): the raw
    # last-line fallback CAN legitimately match gold (e.g. via the quoted-
    # text OCR fallback). That must stay a clean True, not a parse failure,
    # under either flag setting.
    answer = 'Earlier evidence mentions "Paris".\nFinal answer: "London"'
    for flag in (False, True):
        debug_scorer.EXCLUDE_UNPARSEABLE_ANSWERS = flag
        assert score_answer(answer, "London", "exact_match", {}) is True, f"flag={flag}"
    assert "exact_match" not in debug_scorer.parse_failure_stats()


# ── AnswerParseError subclasses ScoringUnavailableError ────────────────────


def test_answer_parse_error_is_a_scoring_unavailable_error():
    assert issubclass(AnswerParseError, ScoringUnavailableError)


# ── end-to-end: seeding_scoring.score_answer_or_error already excludes it ──


def test_score_answer_or_error_excludes_parse_failure_when_flag_on():
    """The literal "route through the existing exclusion path" claim.

    With the flag on, seeding_scoring.score_answer_or_error (the exclusion
    path named in TD-21.11..21.14, seeding_scoring.py:83-113) needs NO code
    change of its own: it already catches ScoringUnavailableError generically,
    and AnswerParseError is one.

    The flag lives on the debug_scorer module INSTANCE that
    ``seeding_scoring._load_orchestrator_debug_scorer`` dynamically loads
    under its own private ``sys.modules`` key — a separate object from this
    test file's plain ``import debug_scorer`` above — so it must be set
    there, not on ``_SS`` (seeding_scoring itself has no such attribute).
    """
    scorer_mod = _SS._load_orchestrator_debug_scorer()
    scorer_mod.EXCLUDE_UNPARSEABLE_ANSWERS = True
    try:
        verdict, reason = _SS.score_answer_or_error(
            answer="I refuse to pick a letter.",
            expected="B",
            scoring_method="multiple_choice",
            scoring_config={},
        )
    finally:
        scorer_mod.EXCLUDE_UNPARSEABLE_ANSWERS = False
        scorer_mod.reset_parse_failure_stats()
    assert verdict is None
    assert reason is not None
    assert "scoring_unavailable" in reason
    assert "answer_parse_failed[multiple_choice]" in reason


# ── _is_valid_json: fish_json extraction fix (TD-21.14, not exclusion) ─────


def test_is_valid_json_still_accepts_a_single_clean_object():
    assert debug_scorer._is_valid_json('{"a": 1}') is True


def test_is_valid_json_rejects_pure_prose():
    assert debug_scorer._is_valid_json("there is no json here at all") is False


def test_is_valid_json_fish_json_finds_last_balanced_object_legacy_slice_missed():
    # The pre-TD-21.14 find("{")/rfind("}") slice spans from the FIRST "{" to
    # the LAST "}", producing '{"a": 1} some text {"b": 2}' here — not valid
    # JSON, so the legacy implementation returned False. fish_json's
    # balanced-bracket scan instead finds the LAST well-formed top-level
    # object ({"b": 2}) and this is now True. This does not change
    # json_valid's classification as a real oracle (see docstring) — it only
    # fixes a false negative in how the JSON is located.
    text = '{"a": 1} some text {"b": 2}'
    assert debug_scorer._is_valid_json(text) is True


def test_json_valid_verifier_end_to_end_via_score_answer():
    result = score_answer(
        answer='noisy prefix {"a": 1} middle {"ok": true} trailing',
        expected="",
        scoring_method="programmatic",
        scoring_config={"verifier": "json_valid"},
    )
    assert result is True


def test_score_answer_or_error_unaffected_when_flag_off():
    """Default OFF: the seeding exclusion path is not exercised by a parse
    failure at all — the row keeps scoring `False`, exactly as before."""
    verdict, reason = _SS.score_answer_or_error(
        answer="I refuse to pick a letter.",
        expected="B",
        scoring_method="multiple_choice",
        scoring_config={},
    )
    assert verdict is False
    assert reason is None
