"""SafetyGate cost-instrument era plumbing (INF-45 W2); mirrors the eval_quality era tests."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import pytest
from safety_gate import EvalResult, SafetyGate  # type: ignore[import-not-found]  # noqa: E402

_ERA = "E1-cost-instrument"


@pytest.fixture(autouse=True)
def _hermetic_guard_archive(monkeypatch):
    import safety_gate as sg  # type: ignore[import-not-found]
    from scripts.autopilot.pareto_archive import ParetoArchive

    monkeypatch.setattr(sg, "_pareto_archive_for_safety_guard", lambda: ParetoArchive())


def _gate(tmp_path, **kwargs) -> SafetyGate:
    g = SafetyGate(baseline_path=tmp_path / "absent.yaml", **kwargs)
    g.baseline.frontdoor_speed = 1.0
    return g


def _result(quality: float, *, tier: int = 2) -> EvalResult:
    return EvalResult(
        tier=tier, quality=quality, speed=99.0, cost=0.1, reliability=0.99,
        per_suite_quality={"coder": quality}, routing_distribution={"worker": 1.0},
        n_questions=50, speed_metric_mode="aggregate_batch_tps",
    )


def test_legacy_baseline_vs_active_cost_era_trips_hold(tmp_path) -> None:
    assert _gate(tmp_path, cost_era=_ERA).cost_rebaseline_required is True


def test_same_era_baseline_does_not_trip_hold(tmp_path) -> None:
    g = _gate(tmp_path, cost_era=_ERA, baseline_state={"cost_era": _ERA})
    assert g.cost_rebaseline_required is False


def test_no_active_cost_era_is_inert(tmp_path) -> None:
    g = _gate(tmp_path)
    assert g.cost_rebaseline_required is False
    assert "cost_era" not in g.baseline.to_state_dict()


def test_cost_era_round_trips_through_state(tmp_path) -> None:
    g = _gate(tmp_path, baseline_state={"cost_era": f" {_ERA} "})
    assert g.baseline.cost_era == _ERA
    assert g.baseline.to_state_dict()["cost_era"] == _ERA


def test_cost_era_independent_of_quality_and_speed_fences(tmp_path) -> None:
    g = _gate(tmp_path, cost_era=_ERA)
    assert g.quality_rebaseline_required is False
    assert g.speed_rebaseline_required is False


def test_update_baseline_stamps_cost_era(tmp_path, monkeypatch) -> None:
    g = _gate(tmp_path, cost_era=_ERA, baseline_state={"cost_era": _ERA,
                                                      "baselines_by_tier": {"2": 1.0}})
    monkeypatch.setattr(g, "_baseline_eligible", lambda result: (True, "test-eligible", {}))
    monkeypatch.setattr(g, "_archive_best_quality", lambda tier: None)
    res = g.update_baseline(_result(2.9), source_trial_id=5)
    assert res.updated is True
    assert g.baseline.cost_era == _ERA
