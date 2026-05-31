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
    # quality/per-suite must stay within the 0-3 scale; the loader + update_baseline
    # now reject out-of-scale values (see test_baseline_scale_guard.py).
    return EvalResult(tier=2, quality=2.9, speed=99.0, cost=0.1, reliability=0.99,
                      per_suite_quality={"coder": 2.9}, speed_metric_mode=speed_metric_mode)


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
    # Isolate from the live Pareto archive — bootstrap case where the archive-max guard is skipped.
    monkeypatch.setattr(SafetyGate, "_archive_best_quality", staticmethod(lambda: None))
    g.update_baseline(_result())
    assert g.baseline.quality == pytest.approx(2.9), "eligible result must write the baseline"


def test_baseline_refused_when_quality_exceeds_archive_max(tmp_path, monkeypatch):
    """A promotion whose quality exceeds the Pareto frontier max is a phantom/contaminated
    measurement (it was never archived) and must be refused — the 2026-05-31 gate-lock guard."""
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml")
    before = g.baseline.quality
    monkeypatch.setattr(g, "_baseline_eligible", lambda result: (True, "test-eligible", {"x": 1}))
    # Archive max 2.400; the result claims 2.9 — never achieved, must be refused.
    monkeypatch.setattr(SafetyGate, "_archive_best_quality", staticmethod(lambda: 2.4))
    g.update_baseline(_result())  # quality=2.9
    assert g.baseline.quality == before, "baseline must NOT update above the archive max"


def test_baseline_written_when_quality_within_archive_max(tmp_path, monkeypatch):
    """A promotion at/under the frontier max is a real achieved measurement — allowed."""
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml")
    monkeypatch.setattr(g, "_baseline_eligible", lambda result: (True, "test-eligible", {"x": 1}))
    monkeypatch.setattr(SafetyGate, "_archive_best_quality", staticmethod(lambda: 2.9))
    g.update_baseline(_result())  # quality=2.9, archive max 2.9 → within tolerance
    assert g.baseline.quality == pytest.approx(2.9), "result at the archive max must write"


def test_baseline_refused_above_max_when_source_not_on_frontier(tmp_path, monkeypatch):
    """Archive-first ordering: an above-max promotion whose source trial is NOT yet on the
    frontier is refused even with a source_trial_id — the caller must archive.update() first."""
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml")
    before = g.baseline.quality
    monkeypatch.setattr(g, "_baseline_eligible", lambda result: (True, "test-eligible", {"x": 1}))
    monkeypatch.setattr(SafetyGate, "_archive_best_quality", staticmethod(lambda: 2.4))
    monkeypatch.setattr(SafetyGate, "_archive_frontier_trial_ids", staticmethod(frozenset))
    g.update_baseline(_result(), source_trial_id=999)  # quality 2.9, trial 999 not archived
    assert g.baseline.quality == before, "above-max promotion with unarchived source must be refused"


def test_loader_rejects_above_archive_baseline(tmp_path, monkeypatch):
    """Baseline.load() falls back to the default floor when the persisted quality exceeds the
    Pareto archive max — defense-in-depth, since the scale guard alone passes a 2.9 stub."""
    import safety_gate as sg  # type: ignore[import-not-found]
    p = tmp_path / "baseline.yaml"
    p.write_text("quality: 2.9\nspeed: 99.0\ncost: 0.1\nreliability: 0.99\n")
    monkeypatch.setattr(sg, "_pareto_frontier_best_quality", lambda: 2.4)
    b = sg.Baseline.load(p)
    assert b.quality == pytest.approx(sg.Baseline().quality), "above-archive baseline must fall back"


def test_loader_accepts_within_archive_baseline(tmp_path, monkeypatch):
    """An honest baseline below the archive max loads unchanged."""
    import safety_gate as sg  # type: ignore[import-not-found]
    p = tmp_path / "baseline.yaml"
    p.write_text("quality: 1.16\nspeed: 18.0\ncost: 0.5\nreliability: 0.86\n")
    monkeypatch.setattr(sg, "_pareto_frontier_best_quality", lambda: 2.4)
    b = sg.Baseline.load(p)
    assert b.quality == pytest.approx(1.16), "within-archive baseline must load unchanged"


def test_baseline_eligibility_is_explicit_decision_with_proof(tmp_path):
    """Eligibility must be a deliberate bool decision carrying topology/matrix proof (or an error
    reason) — never eligible-by-default. Fail-closed when the live topology/matrix is unverifiable."""
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml")
    elig, reason, proof = g._baseline_eligible(_result(speed_metric_mode="aggregate_batch_tps"))
    assert isinstance(elig, bool)
    assert proof.get("speed_metric_mode") == "aggregate_batch_tps"
    assert ("matrix_status" in proof) or ("error" in proof), "must record topology/matrix proof or error"
