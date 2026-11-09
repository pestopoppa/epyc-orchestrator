"""AP-72 (S3-AP-10): metric-event physicality bound in SafetyGate (default off).

A trial whose reported speed exceeds the operator-declared physical bound is invalid, not a
keep. Inert unless AUTOPILOT_SPEED_PHYSICAL_CEILING_TPS is set. The leg is identical in
``SafetyGate.check`` and the pure ``verdict_from_evidence``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts" / "autopilot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.autopilot import host_health as hh  # noqa: E402
from scripts.autopilot import safety_gate as sg  # noqa: E402

ENV = sg.SPEED_PHYSICAL_CEILING_ENV


def _state():
    return {
        "quality": 1.6, "speed": 12.0, "cost": 0.5, "reliability": 0.9, "frontdoor_speed": 12.0,
        "per_suite_quality": {"math": 2.0},
        "baselines_by_tier": {"1": 1.6},
        "per_suite_quality_by_tier": {"1": {"math": 2.0}},
        "per_suite_counts_by_tier": {"1": {"math": 6}},
    }


def _result(speed: float, reliability: float = 1.0) -> "sg.EvalResult":
    return sg.EvalResult(
        tier=1, quality=1.7, speed=speed, cost=0.5, reliability=reliability,
        per_suite_quality={"math": 2.0}, per_suite_counts={"math": 6},
        routing_distribution={"architect": 0.5, "worker": 0.5},
    )


@pytest.fixture(autouse=True)
def _no_host(monkeypatch):
    class _S:
        def is_throttled(self):
            return False, []

    monkeypatch.setattr(hh.HostHealthState, "snapshot", staticmethod(lambda: _S()))
    monkeypatch.delenv(ENV, raising=False)


def test_leg_semantics():
    assert sg.speed_physicality_leg(1e9, None) == sg.GateLeg()  # off
    assert sg.speed_physicality_leg(50.0, 50.0) == sg.GateLeg()  # at the bound is physical
    assert sg.speed_physicality_leg(math.nan, 50.0) == sg.GateLeg()  # not this leg's job
    leg = sg.speed_physicality_leg(50.1, 50.0)
    assert leg.categories == ("speed_above_physical_bound",)
    assert "not physical" in leg.violations[0]


@pytest.mark.parametrize("raw", ["", "abc", "0", "-3", "nan", "inf"])
def test_malformed_or_missing_ceiling_leaves_the_bound_off(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    assert sg._speed_physical_ceiling() is None


def test_default_off_check_is_unchanged_for_absurd_speed():
    gate = sg.SafetyGate(baseline_state=_state(), use_sequential=False)
    verdict = gate.check(_result(1e6))
    assert "speed_above_physical_bound" not in verdict.categories


def test_check_flags_above_bound_trial_and_passes_below(monkeypatch):
    monkeypatch.setenv(ENV, "200")
    gate = sg.SafetyGate(baseline_state=_state(), use_sequential=False)
    bad = gate.check(_result(5000.0))
    assert not bad.passed and "speed_above_physical_bound" in bad.categories
    ok = sg.SafetyGate(baseline_state=_state(), use_sequential=False).check(_result(12.0))
    assert "speed_above_physical_bound" not in ok.categories


def test_flagged_even_when_reliability_blocks_the_quality_legs(monkeypatch):
    monkeypatch.setenv(ENV, "200")
    gate = sg.SafetyGate(baseline_state=_state(), use_sequential=False)
    v = gate.check(_result(5000.0, reliability=0.1))
    assert v.reliability_blocked and "speed_above_physical_bound" in v.categories


@pytest.mark.parametrize("speed", [0.0, 12.0, 199.9, 200.0, 200.1, 5000.0])
@pytest.mark.parametrize("ceiling", [None, "200"])
def test_pure_verdict_equals_check(monkeypatch, speed, ceiling):
    if ceiling is not None:
        monkeypatch.setenv(ENV, ceiling)
    gate = sg.SafetyGate(baseline_state=_state(), use_sequential=False)
    result = _result(speed)
    ev = sg.gate_evidence_from(gate, result)
    pure = sg.verdict_from_evidence(result, ev)
    live = gate.check(result)
    assert (pure.passed, pure.violations, pure.categories) == (
        live.passed, live.violations, live.categories)


def test_explicit_evidence_ceiling_overrides_env(monkeypatch):
    gate = sg.SafetyGate(baseline_state=_state(), use_sequential=False)
    result = _result(500.0)
    ev = sg.gate_evidence_from(gate, result)
    from dataclasses import replace
    assert "speed_above_physical_bound" not in sg.verdict_from_evidence(result, ev).categories
    ev2 = replace(ev, speed_physical_ceiling=100.0)
    assert "speed_above_physical_bound" in sg.verdict_from_evidence(result, ev2).categories
