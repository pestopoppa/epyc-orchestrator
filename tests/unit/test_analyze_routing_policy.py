"""Unit tests for the routing policy analysis helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
_SPEC = importlib.util.spec_from_file_location(
    "analyze_routing_policy_test",
    _BENCH / "analyze_routing_policy.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["analyze_routing_policy_test"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_live_specialist_roles_derive_from_stack_priors(tmp_path: Path) -> None:
    stack_priors = tmp_path / "stack_priors.yaml"
    stack_priors.write_text(
        """
roles:
  frontdoor:
    deployment_status: live_stack
  worker_general:
    deployment_status: live_stack
  toolrunner:
    deployment_status: live_stack
  architect_general:
    deployment_status: live_stack
  coder_escalation:
    deployment_status: live_stack
  ingest_long_context:
    deployment_status: live_stack
  architect_coding:
    deployment_status: retired
  reap_25b_frontdoor:
    deployment_status: benchmark_or_candidate
""",
        encoding="utf-8",
    )

    roles = _MOD._live_specialist_roles(stack_priors)

    assert roles == {"architect_general", "coder_escalation", "ingest_long_context"}


def test_live_specialist_roles_fallback_excludes_retired_role(tmp_path: Path) -> None:
    roles = _MOD._live_specialist_roles(tmp_path / "missing.yaml")

    assert "architect_coding" not in roles
    assert {"architect_general", "coder_escalation"}.issubset(roles)
