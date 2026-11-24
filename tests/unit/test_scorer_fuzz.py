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
CASES_DIGEST = "a651969fa5f0c67897b6578352b7303cd33ac3926bfca1ed7cfb2de2097a1ac3"


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


def test_exact_match_findings_are_characterised(cases):
    table = {(r["source"], r["category"], r["template"]): r for r in sf.aggregate(_rows("exact_match", cases), by_template=True)}
    # controls behave
    assert table[("gsm8k", "clean_correct", "boxed")]["fnr"] == 0.0
    assert table[("gsm8k", "clean_wrong", "boxed")]["fpr"] == 0.0
    # no FP on prose-only partial answers
    for tpl in ("considered", "intermediate"):
        assert table[("gsm8k", "partial_answer", tpl)]["fpr"] == 0.0
    # BY DESIGN last-line fallback accepts a bare correct line
    assert table[("gsm8k", "missing_marker", "bare_last_line")]["fpr"] == 1.0
    # DEFECT-vs-strict: the last boxed final wins, the FIRST <answer> tag wins
    assert table[("gsm8k", "contradictory_finals", "wrong_then_gold_boxed")]["fpr"] == 1.0
    assert table[("gsm8k", "contradictory_finals", "gold_then_wrong_answer_tags")]["fpr"] == 1.0
    # DEFECT-vs-strict: absolute 1e-6 numeric tolerance accepts a 1e-7 drift, not 1e-5
    assert table[("gsm8k", "loose_float", "drift_1e-7")]["fpr"] == 1.0
    assert table[("gsm8k", "loose_float", "drift_1e-5")]["fpr"] == 0.0
    assert table[("gsm8k", "loose_float", "drift_1e-3")]["fpr"] == 0.0
    # retraction + one committed final is accepted
    assert table[("gsm8k", "wrong_then_correct", "retracted")]["fnr"] == 0.0


# ── math_verify: needs the package ───────────────────────────────────────


def test_math_verify_findings_are_characterised(cases):
    pytest.importorskip("math_verify")
    table = {(r["source"], r["category"], r["template"]): r for r in sf.aggregate(_rows("math_verify", cases), by_template=True)}
    # partial / unmarked prose is accepted whenever the gold value is the last number (intake-2889 replay)
    assert table[("gsm8k", "partial_answer", "considered")]["fpr"] == 1.0
    assert table[("gsm8k", "missing_marker", "sentence")]["fpr"] == 1.0
    # loose float is NOT accepted
    assert table[("gsm8k", "loose_float", "drift_1e-3")]["fpr"] == 0.0
    # CHARACTERISATION of the bare-LaTeX-gold defect (flip these when the fix box lands):
    # a wrong leading-number final is accepted and correct non-trivial finals are rejected
    assert table[("math500", "gold_prefix_collision", "leading_number")]["fpr"] == 1.0
    assert table[("math500", "clean_correct", "boxed")]["fnr"] > 0.0
    assert table[("math500", "clean_correct", "boxed")]["coverage"] < 1.0


def test_goldwrap_candidate_removes_the_gold_parse_defect(cases):
    pytest.importorskip("math_verify")
    table = {
        (r["source"], r["category"], r["template"]): r
        for r in sf.aggregate(_rows("math_verify_goldwrap", cases), by_template=True)
    }
    assert table[("math500", "clean_correct", "boxed")]["fnr"] == 0.0
    assert table[("math500", "clean_correct", "boxed")]["coverage"] > 0.95


def test_strict_candidate_zeroes_partial_missing_and_contradictory(cases):
    pytest.importorskip("math_verify")
    for r in sf.aggregate(_rows("math_verify_strict_candidate", cases)):
        if r["category"] in ("partial_answer", "missing_marker", "contradictory_finals", "loose_float", "clean_wrong"):
            assert r["fp"] == 0, r


def test_unknown_scorer_is_refused(cases):
    with pytest.raises(ValueError):
        sf.score_case("nope", cases[0])
