"""Tests for AutoPilot preflight diagnostics."""

from __future__ import annotations

from pathlib import Path

from scripts.autopilot import preflight_audit as _MOD


def test_model_server_targets_derive_from_stack_priors(tmp_path: Path) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
  coder_escalation:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
  worker_vision:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8086
  reap_25b_frontdoor:
    deployment_status: benchmark_or_candidate
    serving:
      endpoint: http://localhost:8196
""",
        encoding="utf-8",
    )

    targets = set(_MOD._model_server_targets(priors, "http://orchestrator.local:8001"))

    assert ("API", "http://orchestrator.local:8001/health") in targets
    assert ("coder_escalation/frontdoor", "http://localhost:8070/health") in targets
    assert ("worker_vision", "http://localhost:8086/health") in targets
    assert all("8196" not in health_url for _, health_url in targets)


def test_model_server_targets_fallback_is_current(tmp_path: Path) -> None:
    targets = _MOD._model_server_targets(tmp_path / "missing.yaml", "http://localhost:8002")
    health_urls = {health_url for _, health_url in targets}

    assert ("API", "http://localhost:8002/health") in targets
    assert "http://localhost:8071/health" not in health_urls
    assert "http://localhost:8086/health" in health_urls
    assert "http://localhost:8087/health" in health_urls
