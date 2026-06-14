"""Tests for the unified orch CLI helpers."""

from __future__ import annotations

from pathlib import Path

from src.cli_orch import _fallback_status_targets, _stack_status_targets


def test_stack_status_targets_group_live_roles_by_port(tmp_path: Path) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        """
roles:
  architect_general:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8083
  coder_escalation:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
  frontdoor:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
  reap_25b_frontdoor:
    deployment_status: benchmark_or_candidate
    serving:
      endpoint: http://localhost:8090
""",
        encoding="utf-8",
    )

    targets = _stack_status_targets(priors)

    assert targets == [
        ("coder_escalation/frontdoor", 8070),
        ("architect_general", 8083),
    ]


def test_stack_status_targets_fallback_excludes_retired_ports(tmp_path: Path) -> None:
    targets = _stack_status_targets(tmp_path / "missing.yaml")

    assert ("architect_coding", 8084) not in targets
    assert ("coder_escalation/frontdoor/worker_summarize", 8070) in targets
    assert all(port != 8090 for _, port in targets)


def test_fallback_status_targets_derive_alias_groups_from_manifest() -> None:
    targets = _fallback_status_targets()

    assert ("coder_escalation/frontdoor/worker_summarize", 8070) in targets
    assert ("toolrunner/worker_explore/worker_general/worker_math", 8072) in targets
    assert all(port != 8090 for _, port in targets)
