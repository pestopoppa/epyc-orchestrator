"""SafetyGate MAD-based noise filter (intake-421 pi-autoresearch).

Median Absolute Deviation as a robust significance test on improvement-direction
quality deltas. Purely additive — emits warnings ("mad_noise" category) when an
"improvement" is within the noise band of recent history. Never blocks.

Tests the AUTOPILOT's safety_gate (scripts/autopilot/safety_gate.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import pytest
from safety_gate import (  # type: ignore[import-not-found]
    EvalResult,
    SafetyGate,
    MAD_MIN_SAMPLES,
    MAD_Z_THRESHOLD,
)


def _result(quality: float, speed: float = 99.0) -> EvalResult:
    """Build an otherwise-clean trial result at a given quality."""
    return EvalResult(
        tier=2,
        quality=quality,
        speed=speed,
        cost=0.1,
        reliability=0.99,
        per_suite_quality={"coder": quality},
        routing_distribution={"worker": 1.0},
    )


def _gate(tmp_path, history: list[float] | None = None) -> SafetyGate:
    """Build a gate with a high frontdoor_speed baseline so the throughput
    floor doesn't trip during MAD tests (those run far below the floor)."""
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml", quality_history=history)
    g.baseline.frontdoor_speed = 1.0  # any positive value low enough to ignore
    return g


def test_mad_skipped_when_history_below_min(tmp_path):
    """With fewer than MAD_MIN_SAMPLES, the filter must NOT fire."""
    g = _gate(tmp_path, history=[2.0] * (MAD_MIN_SAMPLES - 1))
    verdict = g.check(_result(2.01))
    assert "mad_noise" not in verdict.categories


def test_mad_fires_on_noise_level_improvement(tmp_path):
    """Improvement within ~1 MAD of history median → warning, no violation."""
    history = [2.00, 2.02, 1.98, 2.01, 1.99]
    g = _gate(tmp_path, history=history)
    verdict = g.check(_result(2.01))  # well inside the noise band
    assert verdict.passed, "MAD filter must never block; it only warns"
    assert "mad_noise" in verdict.categories
    assert any("within noise" in w.lower() for w in verdict.warnings)
    # Warning text must reflect autopilot-side semantics: trial is still
    # journaled, but archive.update + AP-22 memory are skipped by the
    # classify_learning_exclusion() helper. Must NOT say "do not learn"
    # (over-claim) or "still recorded and learned from" (stale wording from
    # the diagnostic-only era).
    assert not any("do not learn" in w.lower() for w in verdict.warnings), \
        "warning text must reflect new exclusion semantics, not the old over-claim"
    assert not any("still recorded and learned from" in w.lower() for w in verdict.warnings), \
        "stale diagnostic-only wording; should say 'excluded from archive/learning'"
    assert any("excluded from archive/learning" in w.lower() for w in verdict.warnings)


def test_mad_passes_significant_improvement(tmp_path):
    """A real improvement (> Z_THRESHOLD MADs from median) must NOT warn."""
    history = [2.00, 2.02, 1.98, 2.01, 1.99]
    g = _gate(tmp_path, history=history)
    verdict = g.check(_result(2.50))
    assert verdict.passed
    assert "mad_noise" not in verdict.categories


def test_mad_only_fires_on_improvement_branch(tmp_path):
    """A measurement below baseline takes the regression/warning branch,
    not the MAD branch."""
    history = [2.00, 2.02, 1.98, 2.01, 1.99]
    g = _gate(tmp_path, history=history)
    g.baseline.quality = 3.0  # so result.quality < baseline → not the "improvement" branch
    verdict = g.check(_result(2.01))
    assert "mad_noise" not in verdict.categories


def test_mad_zero_mad_flags_any_change_as_significant(tmp_path):
    """Pathological zero-MAD case (history is constant) — any new value differs."""
    g = _gate(tmp_path, history=[2.0, 2.0, 2.0])
    verdict = g.check(_result(2.0001))
    # equal to median would be "not significant"; differing → significant → no warning
    assert "mad_noise" not in verdict.categories


def test_quality_history_persists_across_check_calls(tmp_path):
    """Each completed trial extends the rolling window, bounded by maxlen."""
    g = _gate(tmp_path)
    for q in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]:
        g.check(_result(q))
    history = g.quality_history
    assert len(history) == 10, "rolling window must respect MAD_HISTORY_DEPTH"
    assert history[-1] == pytest.approx(2.6)
    assert history[0] == pytest.approx(1.7), "oldest entries evicted FIFO"


def test_negative_quality_not_recorded(tmp_path):
    """Junk trials (negative quality) shouldn't poison the noise estimate."""
    g = _gate(tmp_path, history=[2.0, 2.0, 2.0])
    g.check(_result(-1.0))
    assert g.quality_history == [2.0, 2.0, 2.0]


def test_mad_constants_sane():
    """Guard against silent threshold changes."""
    assert MAD_MIN_SAMPLES >= 3, "below 3 samples the median is meaningless"
    assert MAD_Z_THRESHOLD >= 1.5, "looser than ~1.5σ would let real noise through"
