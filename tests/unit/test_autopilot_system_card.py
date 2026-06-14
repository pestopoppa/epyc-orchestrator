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

LEGACY_ARCHITECT_ROLE = "architect" "_coding"


def _write_minimal_root(tmp_path: Path) -> None:
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir()
    (orchestration / "derived").mkdir()
    registry_yaml = """
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
  __LEGACY_ARCHITECT_ROLE__:  # legacy fixture: stack priors must exclude this from live roles
    backend: {type: local}
    model: {name: removed-role.gguf}
""".lstrip()
    (orchestration / "model_registry.yaml").write_text(
        registry_yaml.replace("__LEGACY_ARCHITECT_ROLE__", LEGACY_ARCHITECT_ROLE)
    )
    (orchestration / "derived" / "stack_priors.yaml").write_text(
        """
roles:
  frontdoor:
    role: frontdoor
    deployment_status: live_stack
    status: compiled
    display_name: frontdoor-prior.gguf
    serving:
      ports: [8070, 8080]
      endpoint: http://localhost:8070
      tier: hot
      binding: server_mode.direct
    priors:
      throughput_tps: 24.3
    acceleration:
      spec_type: none
  worker_general:
    role: worker_general
    deployment_status: live_stack
    status: compiled
    display_name: worker-prior.gguf
    serving:
      ports: [8072]
      endpoint: http://localhost:8072
      tier: hot
      binding: stack_manifest.role
    priors:
      throughput_tps: 60.7
    acceleration:
      spec_type: draft
      draft_max: 2
  worker_vision:
    role: worker_vision
    deployment_status: live_stack
    status: compiled
    display_name: vision-prior.gguf
    serving:
      ports: [8086]
      endpoint: http://localhost:8086
      tier: hot
      binding: stack_manifest.role
      launch:
        requirements:
          model_path: /models/vision-prior.gguf
          mmproj_path: /models/mmproj-model-f16.gguf
    priors:
      throughput_tps: 20.0
    acceleration:
      spec_type: baseline
  candidate_only:
    role: candidate_only
    deployment_status: benchmark_or_candidate
    status: compiled
    display_name: candidate.gguf
    serving:
      ports: [9999]
    priors:
      throughput_tps: 1.0
    acceleration:
      spec_type: none
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


def test_system_card_uses_stack_priors_not_registry_or_removed_role(tmp_path: Path) -> None:
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

    assert "Source: orchestration/derived/stack_priors.yaml" in card
    assert "| frontdoor | 8070, 8080 | frontdoor-prior.gguf |" in card
    assert "| worker_general | 8072 | worker-prior.gguf |" in card
    assert "| worker_vision | 8086 | vision-prior.gguf |" in card
    assert "mmproj=mmproj-model-f16.gguf" in card
    assert "worker.gguf" not in card
    assert "candidate.gguf" not in card
    assert f"| {LEGACY_ARCHITECT_ROLE} |" not in card
    assert f"{LEGACY_ARCHITECT_ROLE} is not an active server role" in card


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
    assert "{planner_evidence}" in autopilot.CONTROLLER_PROMPT_TEMPLATE
    assert (
        autopilot.CONTROLLER_PROMPT_TEMPLATE.index("### Pareto Frontier Geometry")
        < autopilot.CONTROLLER_PROMPT_TEMPLATE.index(
            "### Evidence Power and Sequential Candidate Status"
        )
        < autopilot.CONTROLLER_PROMPT_TEMPLATE.index("### Journal Trustworthiness")
    )
