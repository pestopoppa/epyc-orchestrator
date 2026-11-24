"""AEI-DROPS-MANIFEST: drops manifest and graded-fraction imbalance flag."""

from __future__ import annotations

import sys
from pathlib import Path

from src.autopilot_core.drops_manifest import drops_manifest, graded_fraction_flag

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))


def test_aggregate_carries_manifest_beside_itt_fields():
    from eval_tower import DISPOSITION_INFRA_FAILED, DISPOSITION_SCORED, EvalTower, QuestionResult

    def row(qid, disposition=DISPOSITION_SCORED, error="", infra_reason=""):
        return QuestionResult(
            question_id=qid, suite="unit", prompt="p", expected="e", qid=qid, answer="a",
            correct=True, disposition=disposition, error=error, infra_reason=infra_reason,
            tokens_generated=1, elapsed_s=1.0, eval_wall_s=1.0,
        )

    rows = [row("q1"), row("q2"), row("q3", DISPOSITION_INFRA_FAILED, "infra_failed: timeout", "timeout")]
    d = EvalTower()._aggregate(rows, tier=1).details
    m = d["drops_manifest"]
    assert {"quality_itt", "itt_denominator", "graded_fraction"} <= set(d)
    assert (m["n_attempted"], m["n_scored"], m["n_dropped"]) == (3, 2, 1)
    assert m["n_dropped_by_reason"] == {"infra_failed:timeout": 1}
    assert d["graded_fraction"] == m["graded_fraction"] == 2 / 3


def test_manifest_counts_and_reasons():
    m = drops_manifest(
        n_scheduled=12,
        n_attempted=10,
        n_scored=7,
        infra_failed_reasons={"timeout": 2},
        scoring_failed=1,
    )
    assert m["n_scheduled"] == 12 and m["n_attempted"] == 10 and m["n_scored"] == 7
    assert m["n_dropped_by_reason"] == {"infra_failed:timeout": 2, "scoring_failed:unclassified": 1}
    assert m["n_dropped"] == 3
    assert m["graded_fraction"] == 0.7


def test_empty_attempts_has_no_fraction():
    assert drops_manifest(n_scheduled=0, n_attempted=0, n_scored=0)["graded_fraction"] is None


def test_unequal_graded_fractions_raise_flag():
    base = drops_manifest(n_scheduled=100, n_attempted=100, n_scored=98)
    cand = drops_manifest(n_scheduled=100, n_attempted=100, n_scored=51, scoring_failed=49)
    flag = graded_fraction_flag(base, cand)
    assert flag["comparable"] and flag["imbalanced"] is True


def test_equal_or_close_fractions_do_not_flag():
    base = drops_manifest(n_scheduled=100, n_attempted=100, n_scored=95)
    same = drops_manifest(n_scheduled=100, n_attempted=100, n_scored=95)
    edge = drops_manifest(n_scheduled=100, n_attempted=100, n_scored=90)  # exactly 5 points
    assert graded_fraction_flag(base, same)["imbalanced"] is False
    assert graded_fraction_flag(base, edge)["imbalanced"] is False


def test_missing_fraction_is_not_comparable():
    assert graded_fraction_flag({}, {"graded_fraction": 0.9})["comparable"] is False
    assert graded_fraction_flag({"graded_fraction": None}, {"graded_fraction": 0.9})["comparable"] is False
