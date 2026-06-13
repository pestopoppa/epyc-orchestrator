from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from scripts.autopilot.species.structural_lab import StructuralLab
from scripts.server import orchestrator_stack
from src import features as feature_module
from src.api.routes.config import attest_config, update_config
from src.features import (
    Features,
    feature_sources,
    features,
    get_features,
    reset_features,
    runtime_flag_overrides,
    write_runtime_flag_overrides,
)


def test_runtime_flag_file_overrides_env(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    monkeypatch.setenv("ORCHESTRATOR_SPECIALIST_ROUTING", "0")
    reset_features()

    write_runtime_flag_overrides(
        {"specialist_routing": True},
        set_by="unit-test",
    )

    assert get_features().specialist_routing is True
    assert runtime_flag_overrides() == {"specialist_routing": True}
    assert feature_sources()["specialist_routing"].startswith("runtime_file:")
    payload = json.loads(runtime_path.read_text())
    assert payload["flags"]["specialist_routing"]["set_by"] == "unit-test"


def test_feature_namespace_env_avoids_legacy_settings_collision(monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRATOR_FEATURE_REPL", "0")
    monkeypatch.delenv("ORCHESTRATOR_REPL", raising=False)
    reset_features()

    assert get_features().repl is False
    assert feature_sources()["repl"] == "ORCHESTRATOR_FEATURE_REPL"


def test_features_singleton_reloads_runtime_file_after_ttl(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    monkeypatch.setattr(feature_module, "RUNTIME_FLAGS_TTL_S", 0.0)
    reset_features()

    assert features().model_fallback is False
    write_runtime_flag_overrides({"model_fallback": True}, set_by="unit-test")

    assert features().model_fallback is True


def test_config_post_writes_runtime_file_and_attests(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    reset_features()

    class Request:
        client = SimpleNamespace(host="127.0.0.1")

        async def json(self):
            return {"model_fallback": True, "unknown_flag": True}

    response = asyncio.run(update_config(Request(), current=Features()))
    assert response["status"] == "ok"
    assert response["features"]["model_fallback"] is True

    attestation = asyncio.run(attest_config(current=features()))
    assert attestation["flags"]["model_fallback"] is True
    assert attestation["sources"]["model_fallback"].startswith("runtime_file:")


def test_stack_production_feature_env_is_complete_and_wave_gated() -> None:
    env = orchestrator_stack._production_feature_env()

    assert env["ORCHESTRATOR_FEATURE_SPECIALIST_ROUTING"] == "1"
    assert env["ORCHESTRATOR_FEATURE_MODEL_FALLBACK"] == "1"
    assert env["ORCHESTRATOR_FEATURE_PLAN_REVIEW"] == "0"
    assert env["ORCHESTRATOR_FEATURE_ARCHITECT_DELEGATION"] == "0"
    assert env["ORCHESTRATOR_FEATURE_PARALLEL_EXECUTION"] == "0"
    assert env["ORCHESTRATOR_FEATURE_UNIFIED_STREAMING"] == "0"
    assert env["ORCHESTRATOR_FEATURE_ROUTING_CLASSIFIER"] == "0"
    assert env["ORCHESTRATOR_FEATURE_LANGGRAPH_ARCHITECT_CODING"] == "0"
    assert "ORCHESTRATOR_REPL" not in env
    assert "ORCHESTRATOR_LANGGRAPH_ARCHITECT_CODING" not in env
    for spec in feature_module._FEATURE_REGISTRY:
        assert f"ORCHESTRATOR_FEATURE_{spec.env_var}" in env


def test_stack_live_langgraph_env_excludes_retired_architect_coding() -> None:
    assert "ORCHESTRATOR_LANGGRAPH_ARCHITECT" in (
        orchestrator_stack.LANGGRAPH_PHASE3_LIVE_ENV_VARS
    )
    assert "ORCHESTRATOR_LANGGRAPH_ARCHITECT_CODING" not in (
        orchestrator_stack.LANGGRAPH_PHASE3_LIVE_ENV_VARS
    )


def test_structural_lab_uses_attest_for_current_flags(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"pid": 123, "flags": {"model_fallback": True}}

    monkeypatch.setattr("httpx.get", lambda *a, **k: Response())

    assert StructuralLab().current_flags() == {"model_fallback": True}


def test_apply_flag_experiment_returns_attestation(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"status": "ok", "features": {"model_fallback": True}}

    lab = StructuralLab()
    monkeypatch.setattr("httpx.post", lambda *a, **k: Response())
    monkeypatch.setattr(
        lab,
        "attest_flags",
        lambda expected: {"status": "ok", "expected": expected},
    )

    result = lab.apply_flag_experiment({"model_fallback": True})
    assert result["attestation"] == {
        "status": "ok",
        "expected": {"model_fallback": True},
    }
