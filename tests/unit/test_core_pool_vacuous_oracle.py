"""The pool-build guard against oracles satisfied by echoing the input.

Origin: epyc-root `artifacts/audit/debugbench-oracle-vacuity-20260812.md` (`mainC`,
2026-08-12). Every debugbench row in both core pools carried an `expected` that was a
byte-exact 100-character PREFIX of the upstream reference solution, scored `substring`.
That prefix was already present in the buggy code the model was handed, on 4 of 4 rows,
so a model that changed nothing and echoed its input scored a PASS. Corpus-wide the
construction is vacuous on 3,233 of 4,250 upstream rows (76.1%).

The test that matters is NOT "is `expected` short or uninformative" — a longer prefix is
just as vacuous. It is "is the oracle already satisfied by the input?"
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "autopilot"), str(REPO / "src"), str(REPO)]
_spec = importlib.util.spec_from_file_location(
    "cvs", REPO / "scripts" / "autopilot" / "core_v2_select.py")
cvs = importlib.util.module_from_spec(_spec)
sys.modules["cvs"] = cvs
_spec.loader.exec_module(cvs)


def _row(**kw):
    base = {"id": "r1", "suite": "s", "scoring_method": "substring"}
    base.update(kw)
    return base


def test_the_debugbench_shape_is_caught() -> None:
    """The real defect: expected is a long prefix of the code the model was handed."""
    solution = "class Solution { public int atoms(String f) { Map<String,Integer> m = new HashMap<>();"
    hits = cvs.vacuous_rows([_row(expected=solution[:60], buggy_code=solution + " /*bug*/ }")])
    assert len(hits) == 1
    assert hits[0]["severity"] == "structural"


def test_a_genuine_oracle_is_not_flagged() -> None:
    """The compliant path — this must not forbid its own idiom."""
    assert cvs.vacuous_rows([_row(expected="42", prompt="What is six times seven?")]) == []


def test_short_incidental_containment_is_reported_but_not_structural() -> None:
    """A 1-char answer appears in almost any prompt; that is chance, not a broken oracle."""
    hits = cvs.vacuous_rows([_row(expected="4", prompt="Compute 2+2 given 4 options")])
    assert [h["severity"] for h in hits] == ["incidental"]


def test_non_substring_scorers_are_left_alone() -> None:
    """An exact or programmatic scorer is not made vacuous by containment."""
    assert cvs.vacuous_rows(
        [_row(expected="x" * 80, prompt="x" * 200, scoring_method="exact")]) == []


def test_a_longer_prefix_is_not_a_fix() -> None:
    """Guards the tempting wrong repair: widening 100 -> 500 keeps it vacuous."""
    body = "y" * 900
    hits = cvs.vacuous_rows([_row(expected=body[:500], buggy_code=body)])
    assert hits and hits[0]["severity"] == "structural"


def test_missing_or_empty_expected_is_ignored_not_crashed() -> None:
    assert cvs.vacuous_rows([_row(prompt="p"), _row(expected="", prompt="p"),
                             _row(expected=None, prompt="p")]) == []
