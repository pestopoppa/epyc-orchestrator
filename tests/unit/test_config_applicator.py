"""Tests for autopilot config application contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.autopilot import config_applicator as applicator
from scripts.autopilot import kv_compress


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


def test_restart_role_success_uses_stack_reload(monkeypatch: pytest.MonkeyPatch) -> None:
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

    result = applicator.restart_role(
        "frontdoor",
        env_overrides={"ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS": "768"},
    )

    assert result["status"] == "ok"
    assert result["method"] == "stack_reload"
    assert result["role"] == "frontdoor"
    assert calls[0]["cmd"][-2:] == ["reload", "frontdoor"]
    assert calls[0]["cwd"] == str(ROOT)
    assert calls[0]["timeout"] == 180
    assert calls[0]["env"]["ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS"] == "768"


def test_restart_role_success_journals_restart_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    monkeypatch.setattr(applicator.subprocess, "run", lambda *a, **kw: None)

    result = applicator.restart_role(
        "frontdoor",
        env_overrides={"ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS": "768"},
        journal=FakeJournal(),
        affected_roles=["frontdoor", "coder_escalation"],
        trial_id=27,
        boundary_reason="unit-test restart",
        actor="unit-test",
    )

    assert result["status"] == "ok"
    assert result["restart_boundary_event"]["type"] == "role_restart_boundary"
    assert events == [
        {
            "role": "frontdoor",
            "affected_roles": ["frontdoor", "coder_escalation"],
            "env_keys": ["ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS"],
            "registry_override_keys": [],
            "status": "ok",
            "rollback_status": "",
            "reason": "unit-test restart",
            "actor": "unit-test",
            "boundary_trial_id": 27,
            "command": "orchestrator_stack.py reload frontdoor",
        }
    ]


def test_restart_role_rolls_back_to_prior_env_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("ORCHESTRATOR_WORKER_CALL_BUDGET_CAP", "8")

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)

    result = applicator.restart_role(
        "worker_general",
        env_overrides={"ORCHESTRATOR_WORKER_CALL_BUDGET_CAP": "12"},
    )

    assert result["status"] == "error"
    assert result["role"] == "worker_general"
    assert result["rollback"] == {
        "attempted": True,
        "status": "ok",
        "env_keys": ["ORCHESTRATOR_WORKER_CALL_BUDGET_CAP"],
    }
    assert calls[0]["env"]["ORCHESTRATOR_WORKER_CALL_BUDGET_CAP"] == "12"
    assert calls[1]["env"]["ORCHESTRATOR_WORKER_CALL_BUDGET_CAP"] == "8"


def test_restart_role_rollback_journals_restart_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    monkeypatch.setenv("ORCHESTRATOR_WORKER_CALL_BUDGET_CAP", "8")

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    calls = 0

    def fake_run(cmd, *, cwd, env, timeout, check):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)

    result = applicator.restart_role(
        "worker_general",
        env_overrides={"ORCHESTRATOR_WORKER_CALL_BUDGET_CAP": "12"},
        journal=FakeJournal(),
        trial_id=28,
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert events == [
        {
            "role": "worker_general",
            "affected_roles": ["worker_general"],
            "env_keys": ["ORCHESTRATOR_WORKER_CALL_BUDGET_CAP"],
            "registry_override_keys": [],
            "status": "error",
            "rollback_status": "ok",
            "reason": "intentional role restart",
            "actor": "config_applicator.restart_role",
            "boundary_trial_id": 28,
            "command": "orchestrator_stack.py reload worker_general",
        }
    ]


def test_restart_role_rolls_back_by_unsetting_new_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.delenv("ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K", raising=False)

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)

    result = applicator.restart_role(
        "ingest_long_context",
        env_overrides={"ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K": "12"},
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert calls[0]["env"]["ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K"] == "12"
    assert "ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K" not in calls[1]["env"]


def test_restart_role_rejects_registry_overrides_until_rollback_record_exists() -> None:
    result = applicator.restart_role(
        "frontdoor",
        registry_overrides={"model_id": "candidate"},
    )

    assert result == {
        "status": "error",
        "method": "stack_reload",
        "role": "frontdoor",
        "error": "registry_overrides are not yet supported",
        "registry_override_keys": ["model_id"],
    }


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


def test_kv_compaction_applicator_defaults_to_physical_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, float]] = []

    def fake_ports(*, include_aliases: bool = False) -> dict[str, int]:
        if include_aliases:
            return {"frontdoor": 8070, "coder_escalation": 8070, "worker_general": 8072}
        return {"frontdoor": 8070, "worker_general": 8072}

    def fake_compress_slot(*, port: int, keep_ratio: float, **_kwargs):
        calls.append((port, keep_ratio))
        return kv_compress.CompressResult(success=True, port=port)

    monkeypatch.setattr(kv_compress, "production_ports", fake_ports)
    monkeypatch.setattr(kv_compress, "compress_slot", fake_compress_slot)

    result = applicator.KvCompactionApplicator().apply({"kv.keep_ratio": 0.4})

    assert result.status == "ok"
    assert set(result.payload["per_role"]) == {"frontdoor", "worker_general"}
    assert calls == [(8070, 0.4), (8072, 0.4)]


def test_kv_compaction_applicator_honors_explicit_alias_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_ports(*, include_aliases: bool = False) -> dict[str, int]:
        if include_aliases:
            return {"frontdoor": 8070, "coder_escalation": 8070}
        return {"frontdoor": 8070}

    def fake_compress_slot(*, port: int, **_kwargs):
        calls.append(port)
        return kv_compress.CompressResult(success=True, port=port)

    monkeypatch.setattr(kv_compress, "production_ports", fake_ports)
    monkeypatch.setattr(kv_compress, "compress_slot", fake_compress_slot)

    result = applicator.KvCompactionApplicator(roles=["coder_escalation"]).apply(
        {"kv.keep_ratio": 0.4}
    )

    assert result.status == "ok"
    assert set(result.payload["per_role"]) == {"coder_escalation"}
    assert calls == [8070]
