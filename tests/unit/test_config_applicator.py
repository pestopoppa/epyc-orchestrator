"""Tests for autopilot config application contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.autopilot import config_applicator as applicator


def test_restart_api_with_env_uses_stack_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        applicator,
        "health_check",
        lambda _url: applicator.HealthCheckResult(ok=True),
    )

    result = applicator.restart_api(
        env_overrides={"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": "0.05"},
        url="http://testserver",
    )

    assert result["status"] == "ok"
    assert result["method"] == "stack_reload"
    assert calls
    assert calls[0]["cmd"][-2:] == ["reload", "orchestrator"]
    assert calls[0]["cwd"] == str(ROOT)
    assert calls[0]["env"]["ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI"] == "0.05"


def test_apply_params_marks_env_restart_failure() -> None:
    def fail_env_params(_params, restart=True, url=applicator.ORCHESTRATOR_URL):
        return {"status": "error", "error": "reload failed"}

    original = applicator.apply_env_params
    applicator.apply_env_params = fail_env_params
    try:
        result = applicator.apply_params({"think_harder.min_expected_roi": 0.05})
    finally:
        applicator.apply_env_params = original

    assert result["status"] == "error"
    assert result["errors"] == ["env_restart: reload failed"]


def test_apply_params_marks_unknown_params_as_error() -> None:
    result = applicator.apply_params({"not_a_surface.value": 1})

    assert result["status"] == "error"
    assert result["unknown_params"] == ["not_a_surface.value"]
    assert "unknown_params: not_a_surface.value" in result["errors"]


def test_apply_params_marks_partial_kv_failure() -> None:
    def partial_kv_failure(_params, roles=None):
        return {
            "per_role": {
                "frontdoor": {"success": True, "error": None},
                "worker_general": {"success": False, "error": "slot busy"},
            }
        }

    original = applicator.apply_kv_compact
    applicator.apply_kv_compact = partial_kv_failure
    try:
        result = applicator.apply_params({"kv.keep_ratio": 0.5})
    finally:
        applicator.apply_kv_compact = original

    assert result["status"] == "error"
    assert result["errors"] == ["kv_compact:worker_general: slot busy"]
