"""Tests for the unified orch CLI helpers."""

from __future__ import annotations

from pathlib import Path

from src import cli_orch
from src.cli_orch import _fallback_status_targets, _stack_status_targets

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


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

    assert (_RETIRED_ARCHITECT_ROLE, 8084) not in targets
    assert ("coder_escalation/frontdoor/worker_summarize", 8070) in targets
    assert all(port != 8090 for _, port in targets)


def test_fallback_status_targets_derive_alias_groups_from_manifest() -> None:
    targets = _fallback_status_targets()

    assert ("coder_escalation/frontdoor/worker_summarize", 8070) in targets
    assert ("toolrunner/worker_general/worker_math", 8072) in targets
    assert all("worker_explore" not in name for name, _ in targets)
    assert all(port != 8090 for _, port in targets)


def test_fallback_status_targets_follow_manifest_without_literal_port_list(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli_orch,
        "HOT_SERVERS",
        [{"port": 9001, "roles": ["frontdoor", "coder_escalation"]}],
    )
    monkeypatch.setattr(
        cli_orch,
        "WARM_SERVERS",
        [{"port": 9002, "roles": ["worker_general", "embedder"]}],
    )
    monkeypatch.setattr(
        cli_orch,
        "ROLE_LAUNCH_META",
        {
            "frontdoor": {"mode": "default"},
            "worker_general": {"mode": "worker_pool"},
            "embedder": {"mode": "embedding"},
        },
    )

    targets = _fallback_status_targets()

    assert targets == [
        ("coder_escalation/frontdoor", 9001),
        ("worker_general", 9002),
    ]


def test_fallback_status_targets_exclude_embedding_mode_roles_from_manifest(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli_orch,
        "HOT_SERVERS",
        [{"port": 9001, "roles": ["frontdoor"]}],
    )
    monkeypatch.setattr(
        cli_orch,
        "WARM_SERVERS",
        [{"port": 9002, "roles": ["worker_general", "embedder", "embedder_1"]}],
    )
    monkeypatch.setattr(
        cli_orch,
        "ROLE_LAUNCH_META",
        {
            "frontdoor": {"mode": "default"},
            "worker_general": {"mode": "worker_pool"},
            "embedder": {"mode": "embedding"},
            "embedder_1": {"mode": "embedding"},
        },
    )

    targets = _fallback_status_targets()

    assert targets == [
        ("frontdoor", 9001),
        ("worker_general", 9002),
    ]
