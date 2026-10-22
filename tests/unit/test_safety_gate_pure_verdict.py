"""HIL-6 seam S4: SafetyGate verdict legs as pure functions over explicit evidence.

``SafetyGate.check`` now calls the module-level legs in its original order and then applies its
side effects. ``verdict_from_evidence`` composes the same legs with no side effects. The proof
of byte-identity for AutoPilot is a replay of the AutoPilot journal through check() before and
after the refactor (digest over verdicts plus counters plus MAD windows; see the commit
message). These tests pin the pure contract and its equivalence to check().
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts" / "autopilot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.autopilot import safety_gate as sg  # noqa: E402
from scripts.autopilot import host_health as hh  # noqa: E402

SUITES = ("math", "coder", "general", "tool_use")


def _state():
    return {
        "quality": 1.6, "speed": 12.0, "cost": 0.5, "reliability": 0.9, "frontdoor_speed": 12.0,
        "per_suite_quality": {"math": 2.0, "coder": 1.5, "general": 1.8, "tool_use": 2.5},
        "baselines_by_tier": {"1": 1.6, "0": 2.2},
        "per_suite_quality_by_tier": {"1": {"math": 2.0, "coder": 1.5, "general": 1.8, "tool_use": 2.5}},
        "per_suite_counts_by_tier": {"1": {"math": 6, "coder": 2, "general": 8, "tool_use": 5}},
    }


def _results():
    out = []
    for i in range(60):
        q = [0.4, 1.2, 1.55, 1.6, 1.7, 2.4, float("nan")][i % 7]
        r = sg.EvalResult(
            tier=[1, 1, 0, 2][i % 4], quality=q, speed=[3.0, 9.5, 10.5, 12.0, 0.0][i % 5],
            cost=0.5, reliability=[1.0, 0.95, 0.5, float("nan")][i % 4],
            per_suite_quality={s: (0.3 * ((i + j) % 10)) for j, s in enumerate(SUITES)},
            per_suite_counts={s: 1 + (i + j) % 7 for j, s in enumerate(SUITES)},
            routing_distribution={"architect": [0.1, 0.9][i % 2]},
        )
        if i % 11 == 3:
            r.quality_measured = False
            r.quality_unmeasured_reason = "fixture"
        out.append(r)
    return out


@pytest.fixture
def no_host(monkeypatch):
    state = {"throttled": False}

    class _S:
        def is_throttled(self):
            return state["throttled"], (["fake"] if state["throttled"] else [])

    monkeypatch.setattr(hh.HostHealthState, "snapshot", staticmethod(lambda: _S()))
    return state


def _fields(v):
    return (v.passed, v.violations, v.warnings, v.categories, v.reliability_blocked,
            v.quality_unmeasured)


@pytest.mark.parametrize("kw", [
    {}, {"eval_quality_era": "E9"}, {"autopilot_speed_era": "S9"},
    {"quality_history_by_tier": {"1": [1.6, 1.61, 1.59, 1.6], "0": [2.2, 2.2, 2.2]}},
])
def test_pure_verdict_equals_check_on_the_non_sequential_path(no_host, kw):
    gate = sg.SafetyGate(baseline_state=_state(), use_sequential=False, **kw)
    for i, result in enumerate(_results()):
        no_host["throttled"] = i % 3 == 0
        ev = sg.gate_evidence_from(gate, result, host_throttled=no_host["throttled"],
                                   host_triggers=("fake",) if no_host["throttled"] else ())
        pure = sg.verdict_from_evidence(result, ev)
        assert _fields(pure) == _fields(gate.check(result)), i


def test_pure_verdict_changes_nothing_and_reads_no_live_state(monkeypatch):
    gate = sg.SafetyGate(baseline_state=_state(), use_sequential=False,
                         quality_history_by_tier={"1": [1.6, 1.61, 1.59]})
    evidences = [(r, sg.gate_evidence_from(gate, r)) for r in _results()]
    before = (gate.consecutive_failures, copy.deepcopy(gate._quality_history_by_tier),
              gate._rebaseline_hold_logged, gate._speed_rebaseline_hold_logged)

    def boom(*_a, **_k):
        raise AssertionError("pure verdict touched live state")

    # Any host read, archive read or promotion-guard view would raise.
    monkeypatch.setattr(hh.HostHealthState, "snapshot", staticmethod(boom))
    monkeypatch.setattr(sg, "_pareto_archive_for_safety_guard", boom)
    monkeypatch.setattr(sg, "_promotion_guard_view", boom)
    for result, ev in evidences:
        verdict = sg.verdict_from_evidence(result, ev)
        assert result.gate_verdict is None  # no cached verdict written
        assert verdict.seq is None
    after = (gate.consecutive_failures, copy.deepcopy(gate._quality_history_by_tier),
             gate._rebaseline_hold_logged, gate._speed_rebaseline_hold_logged)
    assert after == before


def test_gate_evidence_snapshot_does_not_grow_the_history_map():
    gate = sg.SafetyGate(baseline_state=_state(), use_sequential=False)
    sg.gate_evidence_from(gate, _results()[2])  # tier 0, never seen
    assert gate._quality_history_by_tier == {}


def test_legs_are_the_documented_rules():
    assert sg.quality_not_finite_leg(math.nan).categories == ("quality_not_finite",)
    assert sg.quality_not_finite_leg(1.0) == sg.GateLeg()
    assert sg.reliability_floor_leg(0.5, 0.8).categories == ("reliability_floor",)
    assert sg.reliability_floor_leg(math.nan, 0.8) == sg.GateLeg()  # NaN reliability: no block
    assert sg.quality_floor_leg(1.9, 0).categories == ("quality_floor",)
    assert sg.quality_floor_leg(1.9, 1) == sg.GateLeg()
    leg, improved = sg.regression_leg(1.4, 1.6)
    assert leg.categories == ("regression",) and not improved
    assert sg.regression_leg(1.58, 1.6)[0].warnings and not sg.regression_leg(1.58, 1.6)[1]
    assert sg.regression_leg(1.7, 1.6) == (sg.GateLeg(), True)
    assert sg.regression_leg(1.7, None) == (sg.GateLeg(), False)
    # Per-suite N-aware binding: low support is advisory, adequate support binds.
    sparse = sg.per_suite_regression_leg({"math": 0.0}, {"math": 3.0}, {"math": 2}, {"math": 2})
    assert sparse.categories == ("per_suite_regression_advisory",) and not sparse.violations
    dense = sg.per_suite_regression_leg({"math": 1.0}, {"math": 3.0}, {"math": 9}, {"math": 9})
    assert dense.categories == ("per_suite_regression",) and dense.violations
    short = sg.mad_significance([1.0, 1.0], 9.0)  # under MAD_MIN_SAMPLES: accept at face value
    assert short[0] is True and all(math.isnan(x) for x in short[1:])
    assert sg.mad_significance([1.0, 1.0, 1.0], 1.1)[0] is False  # zero-MAD window, small delta


def test_sibling_facade_uses_its_incumbent_as_the_reference():
    from src.harness_loop import substrate

    inc = sg.EvalResult(tier=1, quality=1.6, speed=12.0, cost=0.5, reliability=1.0,
                        per_suite_quality={"math": 2.0}, per_suite_counts={"math": 9})
    worse = sg.EvalResult(tier=1, quality=1.4, speed=12.0, cost=0.5, reliability=1.0,
                          per_suite_quality={"math": 1.0}, per_suite_counts={"math": 9})
    v = substrate.autopilot_gate_verdict(worse, inc)
    assert not v.passed and "regression" in v.categories and "per_suite_regression" in v.categories
    same = substrate.autopilot_gate_verdict(inc, inc)
    assert same.passed and same.categories == []
    assert worse.gate_verdict is None and inc.gate_verdict is None
