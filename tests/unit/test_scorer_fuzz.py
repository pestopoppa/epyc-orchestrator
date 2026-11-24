"""EV-SCORER-FUZZ-1: the scorer fuzz harness (offline, no model calls).

The exact_match tests need nothing but the repo; the math_verify tests skip when the
``math-verify`` package is absent (the harness itself refuses to run without it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

import scorer_fuzz as sf  # noqa: E402

# Pinned fixture identity: a change to the gold fixture or to the generators must be a
# deliberate act (re-file the fuzz report under EV-SCORER-FUZZ-1 with the new digest).
GOLD_SHA256 = "2cb68261937d35d04a74bbe1a4a8c0f066be3a178ec4dc1a2572aab4cb4ff4cb"
# Re-pinned 2026-11-24 (schema v2): the 2 unit/base-suffix golds (106^\circ, 2516_8) moved
# from gold_prefix_collision (strict reject) to format_variants/suffix_dropped (strict
# accept). v1 digest: a651969fa5f0c67897b6578352b7303cd33ac3926bfca1ed7cfb2de2097a1ac3.
CASES_DIGEST = "502c54e2869e8d618397ce3e575ba8a5c883997e41e1e62d16e69668e2bcb533"


@pytest.fixture(scope="module")
def golds():
    return sf.read_gold(sf.DEFAULT_FIXTURE)


@pytest.fixture(scope="module")
def cases(golds):
    return sf.generate_cases(golds)


def test_gold_fixture_is_pinned(golds):
    assert sf.file_sha256(sf.DEFAULT_FIXTURE) == GOLD_SHA256
    assert len(golds) == 200
    assert {g.source for g in golds} == {"gsm8k", "math500"}


def test_case_digest_is_pinned_and_deterministic(golds, cases):
    assert sf.cases_digest(cases) == CASES_DIGEST
    assert sf.cases_digest(sf.generate_cases(golds)) == CASES_DIGEST


def test_every_intake_2889_category_is_generated(cases):
    present = {c.category for c in cases}
    for cat in ("partial_answer", "missing_marker", "contradictory_finals", "loose_float", "wrong_then_correct"):
        assert cat in present
    assert present <= set(sf.CATEGORIES)


def test_strict_labels_never_accept_a_wrong_or_uncommitted_completion(cases):
    for c in cases:
        if c.category in ("partial_answer", "missing_marker", "contradictory_finals", "clean_wrong"):
            assert c.strict_accept is False
        if c.category in ("clean_correct", "format_variants", "wrong_then_correct"):
            assert c.strict_accept is True
        assert c.expected  # every case carries its gold


def test_loose_float_only_for_numeric_gold_and_drift_is_exact(cases):
    lf = [c for c in cases if c.category == "loose_float"]
    assert lf
    for c in lf:
        assert sf._NUMERIC.match(c.expected)
    sample = next(c for c in lf if c.template == "drift_1e-3" and c.expected == "70000")
    assert "70000.001" in sample.answer


def test_wrong_value_never_equals_gold(golds):
    pool = [g.gold for g in golds if g.source == "math500"]
    for g in golds:
        assert sf._wrong_for(g.gold, pool) != g.gold


def test_wilson_interval_bounds():
    lo, hi = sf.wilson(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    assert sf.wilson(0, 0) == (0.0, 1.0)
    lo, hi = sf.wilson(100, 100)
    assert hi == pytest.approx(1.0) and lo > 0.95


def test_aggregate_fpr_fnr_and_coverage():
    rows = [
        {"scorer": "s", "source": "x", "category": "c", "template": "t", "strict_accept": False, "outcome": "accept"},
        {"scorer": "s", "source": "x", "category": "c", "template": "t", "strict_accept": False, "outcome": "reject"},
        {"scorer": "s", "source": "x", "category": "c", "template": "t", "strict_accept": True, "outcome": "reject"},
        {"scorer": "s", "source": "x", "category": "c", "template": "t", "strict_accept": True, "outcome": "excluded"},
    ]
    (row,) = sf.aggregate(rows)
    assert row["fpr"] == 0.5 and row["fp"] == 1 and row["n_strict_reject"] == 2
    assert row["fnr"] == 1.0 and row["n_strict_accept"] == 1  # the excluded case is not scored
    assert row["coverage"] == 0.75 and row["excluded"] == 1


def test_markdown_table_renders_every_group():
    rows = [
        {"scorer": "s", "source": "x", "category": "c", "template": "t", "strict_accept": False, "outcome": "reject"},
    ]
    md = sf.render_markdown(sf.aggregate(rows))
    assert "| s | x | c |" in md and "0.000 (0/1)" in md


# ── exact_match: characterisation of the documented findings ─────────────


def _rows(scorer, cases):
    return sf.run_fuzz([c for c in cases], [scorer])


def test_exact_match_scorer_semantics_v2(cases):
    """SCORER-EM-CONTRADICTORY + SCORER-EM-FLOAT-TOL landed (scorer semantics v2)."""
    rows = sf.aggregate(_rows("exact_match", cases), by_template=True)
    table = {(r["source"], r["category"], r["template"]): r for r in rows}
    for src in ("gsm8k", "math500"):
        # controls behave
        assert table[(src, "clean_correct", "boxed")]["fnr"] == 0.0
        assert table[(src, "clean_wrong", "boxed")]["fpr"] == 0.0
        # no FP on prose-only partial answers
        for tpl in ("considered", "intermediate"):
            assert table[(src, "partial_answer", tpl)]["fpr"] == 0.0
        # BY DESIGN (unchanged at v2): the last-line fallback accepts a bare correct line
        assert table[(src, "missing_marker", "bare_last_line")]["fpr"] == 1.0
        # FIXED: conflicting finals are rejected on every template, either order
        for tpl in ("wrong_then_gold_boxed", "gold_then_wrong_boxed", "gold_then_wrong_answer_tags", "wrong_then_gold_hashes"):
            assert table[(src, "contradictory_finals", tpl)]["fpr"] == 0.0, tpl
        # FIXED: no drift is accepted, and an equal value with a trailing zero still is
        for tpl in ("drift_1e-3", "drift_1e-5", "drift_1e-7"):
            assert table[(src, "loose_float", tpl)]["fpr"] == 0.0, tpl
        assert table[(src, "loose_float", "trailing_zero_equal")]["fnr"] == 0.0
        # retraction + one committed final is accepted
        assert table[(src, "wrong_then_correct", "retracted")]["fnr"] == 0.0
        assert table[(src, "wrong_then_correct", "retracted_tags")]["fnr"] == 0.0
    for r in rows:
        # no new FNR on the gold fixture. suffix_dropped (106 for 106^\circ) is a
        # math-equivalence convention a string scorer cannot know: exact_match rejects it
        # at v1 and v2 alike, and math_verify (the math-suite scorer) accepts it.
        if r["template"] != "suffix_dropped":
            assert r["fn"] == 0, r
        assert r["coverage"] == 1.0, r


# ── math_verify: needs the package ───────────────────────────────────────


def test_math_verify_scorer_semantics_v2(cases):
    """SCORER-MV-GOLDWRAP + SCORER-MV-STRICT-FINAL landed (scorer semantics v2)."""
    pytest.importorskip("math_verify")
    rows = sf.aggregate(_rows("math_verify", cases), by_template=True)
    for r in rows:
        if r["category"] in (
            "partial_answer",
            "missing_marker",
            "contradictory_finals",
            "loose_float",
            "clean_wrong",
            "gold_prefix_collision",
        ):
            assert r["fp"] == 0, r
        assert r["fn"] == 0, r
        assert r["coverage"] == 1.0, r


def test_exact_match_v2_edge_cases():
    ds = sf.load_debug_scorer()

    def em(answer, expected):
        return ds.score_answer(answer, expected, "exact_match", {})

    assert em("<answer>0.30000000000000004</answer>", "0.3") is True  # float64 rendering noise
    assert em("<answer>70000.0000001</answer>", "70000") is False  # a stated drift
    assert em("<answer>18.00</answer>", "18") is True
    assert em("<answer>1,000</answer>", "1000") is True
    assert em("<answer>\\boxed{42}</answer>", "42") is True  # inner box is the tag's own wrapper
    assert em("<answer>42</answer> and so \\boxed{42}", "42") is True  # agreeing finals
    assert em("#### 42\n#### 41", "42") is False  # conflicting finals
    assert em("work...\n42", "42") is True  # documented bare-last-line fallback, unchanged


def test_math_verify_v2_edge_cases():
    pytest.importorskip("math_verify")
    ds = sf.load_debug_scorer()

    def mv(answer, expected):
        return ds.score_answer(answer, expected, "math_verify", {"extraction_mode": "latex"})

    assert mv("so <answer>3</answer>", "3") is True  # an <answer> tag is a committed final
    assert mv("so the answer is 3", "3") is False  # prose only
    assert mv("\\boxed{3\\sqrt{13}}", "3\\sqrt{13}") is True  # bare-LaTeX gold
    assert mv("\\boxed{3}", "3\\sqrt{13}") is False
    assert mv("\\boxed{32,348}", "\\$32,\\!348") is True  # escaped dollar is not a delimiter
    assert mv("\\boxed{\\frac{1}{2}} ... \\boxed{0.5}", "\\frac12") is True  # equivalent finals agree


def test_scorer_semantics_version_is_v2():
    assert sf.load_debug_scorer().SCORER_SEMANTICS_VERSION == "strict_finals_v2_20261124"


def test_unknown_scorer_is_refused(cases):
    with pytest.raises(ValueError):
        sf.score_case("nope", cases[0])
