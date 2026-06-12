"""Resolution-aware per-suite regression gate + failed-verdict learning exclusion.

Root cause of the 2026-06-06 planner/critic deadlock (trial 707): the per-suite
regression gate compared per-suite scores quantized to {0.0, 1.5, 3.0} (≈2
questions/suite on a hybrid eval) against a fixed -0.1 floor. A single
correct→incorrect flip is a -1.5 swing — 15× the floor — so the gate fired on
essentially every seeder trial, every such trial was excluded via mad_noise, and
the planner looped until the critic guard halted it.

Two coupled fixes:
  1. safety_gate.per_suite_regression_threshold() widens the floor to the coarser
     of the result/baseline single-flip quantum (3/n) when counts are known.
  2. classify_learning_exclusion() treats mad_noise as benign ONLY when the
     verdict otherwise passed, so a failed verdict can't be laundered into a
     trusted Pareto representative.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

from safety_gate import (  # type: ignore[import-not-found]
    EvalResult,
    SafetyGate,
    PER_SUITE_REGRESSION,
    per_suite_regression_threshold,
)
from autopilot import classify_learning_exclusion  # type: ignore[import-not-found]


# ── per_suite_regression_threshold ────────────────────────────────────────────

def test_threshold_falls_back_to_fixed_floor_without_counts():
    """No counts (legacy baseline) ⇒ unchanged -0.1 floor."""
    assert per_suite_regression_threshold(None, None) == PER_SUITE_REGRESSION
    assert per_suite_regression_threshold(0, 0) == PER_SUITE_REGRESSION


def test_threshold_widens_to_single_flip_quantum_at_low_n():
    """At 2 questions/suite a single flip is 1.5 → threshold must be -1.5 so the
    flip is NOT counted as a regression."""
    assert per_suite_regression_threshold(2, 2) == -1.5
    # A one-flip drop (3.0 -> 1.5) is exactly -1.5, which is NOT < -1.5 → no fire.
    assert not ((1.5 - 3.0) < per_suite_regression_threshold(2, 2))
    # A two-flip collapse (3.0 -> 0.0) IS < -1.5 → genuine regression still fires.
    assert (0.0 - 3.0) < per_suite_regression_threshold(2, 2)


def test_threshold_uses_coarser_of_the_two_samples():
    """Mismatched n ⇒ the coarser (smaller-n) quantum governs."""
    # result n=2 (quantum 1.5) vs baseline n=10 (quantum 0.3) → 1.5 wins.
    assert per_suite_regression_threshold(2, 10) == -1.5
    assert per_suite_regression_threshold(10, 2) == -1.5


def test_threshold_never_tighter_than_fixed_floor():
    """At large n the quantum is tiny; the fixed -0.1 floor still applies."""
    assert per_suite_regression_threshold(100, 100) == PER_SUITE_REGRESSION


# ── gate.check integration ────────────────────────────────────────────────────

def _gate(tmp_path):
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml")
    g.baseline.frontdoor_speed = 1.0
    return g


def _trial(per_suite, counts, tier=1, quality=1.7):
    return EvalResult(
        tier=tier,
        quality=quality,
        speed=99.0,
        cost=0.1,
        reliability=0.99,
        per_suite_quality=per_suite,
        per_suite_counts=counts,
        routing_distribution={"worker": 1.0},
    )


def test_single_question_flip_at_n2_is_not_a_regression(tmp_path):
    """The trial-707 case: baseline 3.0, result 1.5, n=2 each → noise, not gate."""
    g = _gate(tmp_path)
    g.baseline.per_suite_quality_by_tier = {1: {"hotpotqa": 3.0}}
    g.baseline.per_suite_counts_by_tier = {1: {"hotpotqa": 2}}
    verdict = g.check(_trial({"hotpotqa": 1.5}, {"hotpotqa": 2}))
    assert "per_suite_regression" not in verdict.categories


def test_total_collapse_at_n2_still_regresses(tmp_path):
    """A real signal (every question in the suite now wrong) must still fire."""
    g = _gate(tmp_path)
    g.baseline.per_suite_quality_by_tier = {1: {"hotpotqa": 3.0}}
    g.baseline.per_suite_counts_by_tier = {1: {"hotpotqa": 2}}
    verdict = g.check(_trial({"hotpotqa": 0.0}, {"hotpotqa": 2}))
    assert "per_suite_regression" in verdict.categories


def test_small_drop_at_high_n_still_regresses(tmp_path):
    """At adequate n the gate keeps its teeth: a 0.3 drop trips the -0.1 floor."""
    g = _gate(tmp_path)
    g.baseline.per_suite_quality_by_tier = {1: {"coder": 2.4}}
    g.baseline.per_suite_counts_by_tier = {1: {"coder": 50}}
    verdict = g.check(_trial({"coder": 2.1}, {"coder": 50}))
    assert "per_suite_regression" in verdict.categories


def test_gate_check_is_idempotent_for_same_eval_result(tmp_path):
    """Action handlers and the main loop may both ask for a verdict; state mutates once."""
    g = _gate(tmp_path)
    result = _trial({"coder": 0.0}, {"coder": 50}, quality=0.0)
    first = g.check(result)
    second = g.check(result)
    assert second is first
    assert g.consecutive_failures == 1
    assert g.quality_history_for_tier(result.tier) == []


def test_counts_roundtrip_through_baseline_persistence(tmp_path):
    """per_suite_counts_by_tier must survive save()/load() so a refreshed baseline
    keeps its resolution after restart."""
    from safety_gate import Baseline
    path = tmp_path / "baseline.yaml"
    b = Baseline(source_path=path)
    b.per_suite_counts_by_tier = {1: {"hotpotqa": 2, "coder": 50}}
    b.per_suite_quality_by_tier = {1: {"hotpotqa": 3.0, "coder": 2.4}}
    b.save()
    restored = Baseline.load(path)
    assert restored.per_suite_counts_for_tier(1) == {"hotpotqa": 2, "coder": 50}


# ── classify_learning_exclusion: mad_noise must not mask a failed verdict ──────

@dataclass
class _FakeVerdict:
    categories: list[str] = field(default_factory=list)
    passed: bool = True


@dataclass
class _FakeEvalResult:
    n_exogenous_unrecovered: int = 0
    exogenous_question_ids: list[str] = field(default_factory=list)
    n_questions: int = 0


def test_mad_noise_benign_only_when_verdict_passed():
    v = _FakeVerdict(categories=["mad_noise"], passed=True)
    by, _, _ = classify_learning_exclusion(v, _FakeEvalResult())
    assert by == "mad_noise"


def test_mad_noise_with_failed_verdict_is_not_benign():
    """A failed verdict (e.g. genuine per-suite regression) co-tagged mad_noise must
    NOT be classified benign — else it is admitted as a trusted Pareto rep."""
    v = _FakeVerdict(categories=["per_suite_regression", "mad_noise"], passed=False)
    by, reason, def_cat = classify_learning_exclusion(v, _FakeEvalResult())
    assert by == "", "failed verdict must not be laundered as within-noise"
    assert reason == "" and def_cat == ""


def test_reproduction_confirmed_with_failed_verdict_is_not_benign():
    v = _FakeVerdict(
        categories=["per_suite_regression", "mad_noise", "reproduction_confirmed"],
        passed=False,
    )
    by, _, _ = classify_learning_exclusion(v, _FakeEvalResult())
    assert by == ""


# ── calibration path must propagate + persist per-suite counts ─────────────────

def test_calibration_applies_per_suite_counts():
    """_apply_calibrated_baseline_result must store counts, not just quality —
    else a live baseline refresh leaves the baseline-side 3/n term inactive."""
    from autopilot import _apply_calibrated_baseline_result  # type: ignore[import-not-found]
    from safety_gate import Baseline
    b = Baseline()
    r = EvalResult(
        tier=1, quality=1.7, speed=50.0, cost=0.5, reliability=0.98, n_questions=43,
        per_suite_quality={"hotpotqa": 1.5, "coder": 3.0},
        per_suite_counts={"hotpotqa": 2, "coder": 2},
    )
    _apply_calibrated_baseline_result(b, r)
    assert b.per_suite_counts_for_tier(1) == {"hotpotqa": 2, "coder": 2}


def test_write_baseline_yaml_tiers_roundtrips_counts(tmp_path):
    """Fresh-file and existing-file (drop+reappend) write paths both persist counts,
    and a second write replaces the block rather than duplicating it."""
    from autopilot import _write_baseline_yaml_tiers  # type: ignore[import-not-found]
    from safety_gate import Baseline
    path = tmp_path / "autopilot_baseline.yaml"

    b = Baseline(source_path=path)
    b.baselines_by_tier = {1: 1.7}
    b.per_suite_quality_by_tier = {1: {"hotpotqa": 3.0}}
    b.per_suite_counts_by_tier = {1: {"hotpotqa": 2}}
    _write_baseline_yaml_tiers(path, b)  # fresh-file branch
    assert Baseline.load(path).per_suite_counts_for_tier(1) == {"hotpotqa": 2}

    b.per_suite_counts_by_tier = {1: {"hotpotqa": 50}}  # re-measured at higher n
    _write_baseline_yaml_tiers(path, b)  # existing-file drop+reappend branch
    assert Baseline.load(path).per_suite_counts_for_tier(1) == {"hotpotqa": 50}
    assert path.read_text().count("per_suite_counts_by_tier:") == 1, "no stale duplicate block"
