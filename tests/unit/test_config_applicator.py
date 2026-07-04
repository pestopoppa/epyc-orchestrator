"""Tests for autopilot config application contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


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
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

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


def test_restart_role_pause_dispatch_restores_after_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    state_path = tmp_path / "autopilot_state.json"
    state_path.write_text(json.dumps({"paused": False}) + "\n", encoding="utf-8")

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })
        assert json.loads(state_path.read_text(encoding="utf-8"))["paused"] is True

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "frontdoor",
        pause_dispatch=True,
        autopilot_state_path=state_path,
        dispatch_pause_grace_s=0,
    )

    assert result["status"] == "ok"
    assert len(calls) == 1
    assert result["dispatch_pause"]["paused_pre"] is False
    assert result["dispatch_pause"]["restore"] == {
        "status": "ok",
        "state_path": str(state_path),
        "restored": True,
    }
    assert json.loads(state_path.read_text(encoding="utf-8"))["paused"] is False


def test_restart_role_pause_dispatch_restores_after_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    state_path = tmp_path / "autopilot_state.json"
    state_path.write_text(json.dumps({"paused": False}) + "\n", encoding="utf-8")

    def fake_run(cmd, *, cwd, env, timeout, check):
        nonlocal calls
        calls += 1
        assert json.loads(state_path.read_text(encoding="utf-8"))["paused"] is True
        if calls == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "worker_general",
        env_overrides={"ORCHESTRATOR_WORKER_CALL_BUDGET_CAP": "12"},
        pause_dispatch=True,
        autopilot_state_path=state_path,
        dispatch_pause_grace_s=0,
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert calls == 2
    assert result["dispatch_pause"]["restore"]["status"] == "ok"
    assert json.loads(state_path.read_text(encoding="utf-8"))["paused"] is False


def test_restart_role_pause_dispatch_missing_state_fails_before_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        applicator.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = applicator.restart_role(
        "frontdoor",
        pause_dispatch=True,
        autopilot_state_path=tmp_path / "missing_state.json",
        dispatch_pause_grace_s=0,
    )

    assert result["status"] == "error"
    assert result["error"] == "failed to pause autopilot dispatch"
    assert result["dispatch_pause"]["status"] == "error"
    assert calls == []


def test_restart_role_strict_requires_explicit_affected_roles_before_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        applicator.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = applicator.restart_role(
        "frontdoor",
        require_explicit_affected_roles=True,
        smoke_check=lambda _role, _affected_roles: True,
    )

    assert result["status"] == "error"
    assert result["reason"] == "affected_roles_required"
    assert result["error"] == "affected_roles required for strict role restart"
    assert calls == []


def test_restart_role_strict_requires_smoke_check_before_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        applicator.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = applicator.restart_role(
        "frontdoor",
        affected_roles=["frontdoor", "coder_escalation"],
        require_smoke_check=True,
    )

    assert result["status"] == "error"
    assert result["reason"] == "smoke_check_required"
    assert result["error"] == "smoke_check required for strict role restart"
    assert result["affected_roles"] == ["frontdoor", "coder_escalation"]
    assert calls == []


def test_restart_role_strict_accepts_explicit_scope_and_smoke_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        affected_roles=["coder_escalation", "frontdoor"],
        require_explicit_affected_roles=True,
        require_smoke_check=True,
        smoke_check=lambda _role, _affected_roles: True,
    )

    assert result["status"] == "ok"
    assert result["affected_roles"] == ["frontdoor", "coder_escalation"]
    assert result["smoke_check"]["status"] == "ok"
    assert calls[0]["cmd"][-2:] == ["reload", "frontdoor"]


def test_restart_role_resolves_stack_affinity_from_priors(
    tmp_path: Path,
) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        """
stack_priors_version: 4
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      launch:
        entries:
          - port: 8070
            primary_role: frontdoor
  coder_escalation:
    deployment_status: live_stack
    serving:
      launch:
        entries:
          - port: 8070
            primary_role: frontdoor
  worker_general:
    deployment_status: live_stack
    serving:
      launch:
        entries:
          - port: 8072
            primary_role: worker_general
""",
        encoding="utf-8",
    )

    assert applicator.resolve_restart_affected_roles(
        "coder_escalation",
        stack_priors_path=priors,
    ) == ["coder_escalation", "frontdoor"]


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
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

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
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

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
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "ingest_long_context",
        env_overrides={"ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K": "12"},
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert calls[0]["env"]["ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K"] == "12"
    assert "ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K" not in calls[1]["env"]


def test_restart_role_smoke_failure_rolls_back_without_success_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(
        applicator,
        "resolve_restart_affected_roles",
        lambda _role: ["frontdoor", "coder_escalation"],
    )

    result = applicator.restart_role(
        "frontdoor",
        env_overrides={"ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS": "768"},
        journal=FakeJournal(),
        smoke_check=lambda _role, _affected_roles: {
            "status": "error",
            "error": "role probe failed",
        },
    )

    assert result["status"] == "error"
    assert result["error"] == "role probe failed"
    assert result["rollback"] == {
        "attempted": True,
        "status": "ok",
        "env_keys": ["ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS"],
    }
    assert len(calls) == 2
    assert events == [
        {
            "role": "frontdoor",
            "affected_roles": ["frontdoor", "coder_escalation"],
            "env_keys": ["ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS"],
            "registry_override_keys": [],
            "status": "error",
            "rollback_status": "ok",
            "reason": "intentional role restart",
            "actor": "config_applicator.restart_role",
            "boundary_trial_id": None,
            "command": "orchestrator_stack.py reload frontdoor",
        }
    ]


def test_restart_role_applies_registry_overrides_and_journals_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[dict[str, object]] = []
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "server_mode": {
                    "frontdoor": {
                        "draft_max": 16,
                        "p_split": 0.0,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    monkeypatch.setattr(applicator.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "frontdoor",
        registry_overrides={
            "server_mode.frontdoor.draft_max": 24,
            "server_mode.frontdoor.p_split": 0.1,
        },
        registry_path=registry_path,
        journal=FakeJournal(),
    )

    assert result["status"] == "ok"
    assert result["registry_overrides"] == {
        "status": "ok",
        "registry_path": str(registry_path),
        "override_keys": [
            "server_mode.frontdoor.draft_max",
            "server_mode.frontdoor.p_split",
        ],
    }
    reloaded = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert reloaded["server_mode"]["frontdoor"]["draft_max"] == 24
    assert reloaded["server_mode"]["frontdoor"]["p_split"] == 0.1
    assert events[0]["registry_override_keys"] == [
        "server_mode.frontdoor.draft_max",
        "server_mode.frontdoor.p_split",
    ]


def test_restart_role_restores_registry_overrides_on_reload_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    registry_path = tmp_path / "model_registry.yaml"
    original_registry = {
        "server_mode": {
            "worker_general": {
                "compaction_profile": "default",
            }
        }
    }
    registry_path.write_text(
        yaml.safe_dump(original_registry, sort_keys=False),
        encoding="utf-8",
    )

    def fake_run(cmd, *, cwd, env, timeout, check):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "worker_general",
        registry_overrides={
            "server_mode.worker_general.compaction_profile": "S8",
        },
        registry_path=registry_path,
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert result["rollback"]["registry"] == {
        "status": "ok",
        "registry_path": str(registry_path),
        "restored_keys": ["server_mode.worker_general.compaction_profile"],
    }
    assert calls == 2
    assert yaml.safe_load(registry_path.read_text(encoding="utf-8")) == original_registry


def test_restart_role_restores_registry_overrides_on_smoke_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    registry_path = tmp_path / "model_registry.yaml"
    original_registry = {
        "server_mode": {
            "frontdoor": {
                "draft_max": 16,
            }
        }
    }
    registry_path.write_text(
        yaml.safe_dump(original_registry, sort_keys=False),
        encoding="utf-8",
    )

    def fake_run(cmd, *, cwd, env, timeout, check):
        nonlocal calls
        calls += 1
        if calls == 2:
            restored = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            assert restored == original_registry

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "frontdoor",
        registry_overrides={"server_mode.frontdoor.draft_max": 24},
        registry_path=registry_path,
        smoke_check=lambda _role, _affected_roles: False,
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert result["rollback"]["registry"]["status"] == "ok"
    assert calls == 2
    assert yaml.safe_load(registry_path.read_text(encoding="utf-8")) == original_registry


def test_restart_role_rejects_registry_override_missing_parent_before_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text(yaml.safe_dump({"server_mode": {}}), encoding="utf-8")
    monkeypatch.setattr(
        applicator.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = applicator.restart_role(
        "frontdoor",
        registry_overrides={"server_mode.frontdoor.draft_max": 24},
        registry_path=registry_path,
    )

    assert result["status"] == "error"
    assert result["error"].startswith("failed to apply registry_overrides:")
    assert result["registry_override_keys"] == ["server_mode.frontdoor.draft_max"]
    assert calls == []


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
