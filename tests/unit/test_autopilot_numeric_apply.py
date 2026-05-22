"""Tests for NumericSwarm application/evaluation ordering."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

autopilot = importlib.import_module("scripts.autopilot.autopilot")


class FailingTower:
    def hybrid_eval(self):  # pragma: no cover - must not be called
        raise AssertionError("eval should be skipped when params fail to apply")


class FakeSwarm:
    def __init__(self) -> None:
        self.failed: tuple[str, int, str] | None = None

    def suggest_trial(self, surface: str) -> dict[str, object]:
        return {
            "trial_number": 7,
            "surface": surface,
            "params": {"think_harder.min_expected_roi": 0.05},
        }

    def mark_failed(self, surface: str, trial_number: int, reason: str = "") -> None:
        self.failed = (surface, trial_number, reason)


def test_suggested_numeric_trial_marks_failed_and_skips_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        autopilot,
        "apply_params",
        lambda _params: {"status": "error", "errors": ["env_restart: reload failed"]},
    )
    swarm = FakeSwarm()

    result, species = autopilot.dispatch_action(
        {"type": "numeric_trial", "surface": "think_harder"},
        seeder=None,
        swarm=swarm,
        forge=None,
        lab=None,
        tower=FailingTower(),
        gate=None,
        archive=None,
        journal=None,
        state={},
    )

    assert result is None
    assert species == "numeric_swarm"
    assert swarm.failed == ("think_harder", 7, "env_restart: reload failed")


def test_explicit_numeric_trial_failure_skips_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        autopilot,
        "apply_params",
        lambda _params: {"status": "error", "errors": ["unknown_params: x"]},
    )

    result, species = autopilot.dispatch_action(
        {
            "type": "numeric_trial",
            "surface": "think_harder",
            "params": {"think_harder.min_expected_roi": 0.05},
        },
        seeder=None,
        swarm=FakeSwarm(),
        forge=None,
        lab=None,
        tower=FailingTower(),
        gate=None,
        archive=None,
        journal=None,
        state={},
    )

    assert result is None
    assert species == "numeric_swarm"
