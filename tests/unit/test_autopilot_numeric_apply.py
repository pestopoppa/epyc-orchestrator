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
experiment_journal = importlib.import_module("scripts.autopilot.experiment_journal")
numeric_swarm = importlib.import_module("scripts.autopilot.species.numeric_swarm")


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


def test_chat_pipeline_numeric_surface_uses_runtime_q_threshold() -> None:
    names = [spec.name for spec in numeric_swarm.SURFACES["chat_pipeline"]]

    assert names == ["chat.try_cheap_first_q_threshold"]
    assert "chat.try_cheap_first_quality_threshold" not in names


def test_suggested_numeric_trial_env_restart_skips_without_marking_failed(
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

    assert isinstance(result, autopilot.SkipOutcome)
    assert result.status == "skipped"
    assert "env_restart: reload failed" in result.reason
    assert result.bug_corrupted_by == "env_restart_apply_failure"
    assert species == "numeric_swarm"
    assert swarm.failed is None


def test_suggested_numeric_trial_non_infra_apply_error_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        autopilot,
        "apply_params",
        lambda _params: {"status": "error", "errors": ["hot_swap: rejected"]},
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

    assert isinstance(result, autopilot.SkipOutcome)
    assert result.status == "skipped"
    assert result.bug_corrupted_by == ""
    assert "hot_swap: rejected" in result.reason
    assert species == "numeric_swarm"
    assert swarm.failed == ("think_harder", 7, "hot_swap: rejected")


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

    assert isinstance(result, autopilot.SkipOutcome)
    assert result.status == "invalid"
    assert "unknown_params: x" in result.reason
    assert species == "numeric_swarm"


def test_bug_corrupted_skip_trial_journals_contamination_tag(tmp_path: Path) -> None:
    journal = experiment_journal.ExperimentJournal(journal_dir=tmp_path)

    autopilot._record_skip_trial(
        journal,
        42,
        {"type": "numeric_trial", "surface": "memrl_retrieval"},
        "numeric_swarm",
        "skipped",
        "numeric_trial params failed to apply: env_restart: reload failed",
        123,
        bug_corrupted_by="env_restart_apply_failure",
        bug_corrupted_reason="reload failed before params applied",
    )

    entry = experiment_journal.ExperimentJournal(journal_dir=tmp_path).all_entries()[0]
    assert entry.outcome_status == "skipped"
    assert entry.bug_corrupted_by == "env_restart_apply_failure"
    assert entry.bug_corrupted_reason == "reload failed before params applied"
