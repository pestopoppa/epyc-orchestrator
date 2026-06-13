"""Tests for AutoPilot W8 controller guidance split."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

gen_system_card = importlib.import_module("gen_system_card")
autopilot = importlib.import_module("autopilot")


def _write_minimal_root(tmp_path: Path) -> None:
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir()
    (orchestration / "model_registry.yaml").write_text(
        """
server_mode:
  frontdoor:
    port: 8070
    model: frontdoor.gguf
    tier: hot
    acceleration: {type: none, lookup: false}
    throughput: 24.3
    description: Root model
  worker:
    port: 8072
    model: worker.gguf
    tier: hot
    acceleration: {type: speculative_decoding, draft_max: 2, lookup: false}
    throughput: 60.7
    description: Worker alias without a matching roles entry
roles:
  frontdoor:
    backend: {type: local}
  architect_coding:
    backend: {type: local}
    model: {name: removed-role.gguf}
""".lstrip()
    )
    (orchestration / "autopilot_baseline.yaml").write_text(
        """
baselines_by_tier:
  1: 0.5
per_suite_quality_by_tier:
  1:
    math: 3.0
""".lstrip()
    )


def test_system_card_uses_server_mode_not_removed_role(tmp_path: Path) -> None:
    _write_minimal_root(tmp_path)
    card = gen_system_card.generate_system_card(
        tmp_path,
        state_override={
            "paused": True,
            "trial_counter": 12,
            "baseline_state": {
                "baselines_by_tier": {"1": 1.9},
                "per_suite_quality_by_tier": {"1": {"math": 3.0}},
            },
        },
    )

    assert "| frontdoor | 8070 | frontdoor.gguf |" in card
    assert "| worker | 8072 | worker.gguf |" in card
    assert "| architect_coding |" not in card
    assert "architect_coding is not an active server role" in card


def test_system_card_prefers_state_baseline_over_yaml(tmp_path: Path) -> None:
    _write_minimal_root(tmp_path)
    card = gen_system_card.generate_system_card(
        tmp_path,
        state_override={
            "baseline_state": {
                "baselines_by_tier": {"1": 2.25},
                "per_suite_quality_by_tier": {"1": {"math": 3.0, "coder": 1.5}},
            }
        },
    )

    assert "Source: orchestration/autopilot_state.json:baseline_state" in card
    assert "T1: quality baseline 2.25" in card
    assert "quality baseline 0.5" not in card


def test_system_card_hides_stale_pause_reason_when_unpaused(tmp_path: Path) -> None:
    _write_minimal_root(tmp_path)
    card = gen_system_card.generate_system_card(
        tmp_path,
        state_override={
            "paused": False,
            "pause_reason": "old contested-window reason",
            "trial_counter": 99,
        },
    )

    assert "- paused: false" in card
    assert "old contested-window reason" not in card


def test_controller_template_uses_constitution_and_system_card() -> None:
    assert "{program}" not in autopilot.CONTROLLER_PROMPT_TEMPLATE
    assert "{constitution}" in autopilot.CONTROLLER_PROMPT_TEMPLATE
    assert "{system_card}" in autopilot.CONTROLLER_PROMPT_TEMPLATE
