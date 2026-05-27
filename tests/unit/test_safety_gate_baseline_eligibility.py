"""SafetyGate.update_baseline HARD eligibility gate (operator audit #3, 2026-05-27).

The autopilot's production baseline must not be written on a stale/wrong topology or with unknown
concurrent-speed semantics — the poisoning guard. Tests the AUTOPILOT's safety_gate
(scripts/autopilot/safety_gate.py — the module the autopilot imports), NOT src/safety_gate.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import pytest
from safety_gate import EvalResult, SafetyGate  # type: ignore[import-not-found]


def _result(speed_metric_mode: str = "aggregate_batch_tps") -> EvalResult:
    return EvalResult(tier=2, quality=9.9, speed=99.0, cost=0.1, reliability=0.99,
                      per_suite_quality={"coder": 9.9}, speed_metric_mode=speed_metric_mode)


def test_baseline_refused_on_unrecognized_speed_mode(tmp_path):
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml")  # default Baseline (quality=2.0)
    before = g.baseline.quality
    g.update_baseline(_result(speed_metric_mode="bogus_mode"))
    assert g.baseline.quality == before, "baseline must NOT update with an unrecognized speed mode"
    elig, reason, proof = g._baseline_eligible(_result(speed_metric_mode="bogus_mode"))
    assert not elig and "speed_metric_mode" in reason


def test_baseline_written_when_eligible(tmp_path, monkeypatch):
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml")
    monkeypatch.setattr(g, "_baseline_eligible", lambda result: (True, "test-eligible", {"x": 1}))
    g.update_baseline(_result())
    assert g.baseline.quality == pytest.approx(9.9), "eligible result must write the baseline"


def test_baseline_eligibility_is_explicit_decision_with_proof(tmp_path):
    """Eligibility must be a deliberate bool decision carrying topology/matrix proof (or an error
    reason) — never eligible-by-default. Fail-closed when the live topology/matrix is unverifiable."""
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml")
    elig, reason, proof = g._baseline_eligible(_result(speed_metric_mode="aggregate_batch_tps"))
    assert isinstance(elig, bool)
    assert proof.get("speed_metric_mode") == "aggregate_batch_tps"
    assert ("matrix_status" in proof) or ("error" in proof), "must record topology/matrix proof or error"
