"""Behavioral coverage for the default-off binding-routing feature."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

import src.api as api
from src.api.models import ChatRequest
from src.api.routes.chat_pipeline import routing_decision
from src.api.routes.chat_routing import _classify_and_route
from src.api.state import AppState
from src.features import Features, features, get_features, set_features
from src.routing_bindings import BindingPriority, BindingRouter, RoutingBinding


@pytest.fixture(autouse=True)
def _reset_binding_feature():
    original = get_features()
    set_features(Features(binding_routing=False))
    yield
    set_features(original)


def _request(**kwargs) -> ChatRequest:
    return ChatRequest(prompt="Implement a sorting function", **kwargs)


@asynccontextmanager
async def _isolated_lifespan(monkeypatch, tmp_path, enabled: bool):
    """Run the real API lifespan while replacing unrelated startup services."""
    set_features(Features(binding_routing=enabled))
    state = AppState()
    monkeypatch.setattr(api, "get_state", lambda: state)
    monkeypatch.setattr(api, "features", features)
    monkeypatch.setattr(api, "load_optional_imports", lambda: None)
    monkeypatch.setattr(api, "get_progress_logger_class", lambda: None)
    monkeypatch.setattr(api, "get_tool_registry_class", lambda: None)
    monkeypatch.setattr(api, "get_script_registry_class", lambda: None)
    monkeypatch.setattr(api, "shutdown_scoring", lambda: None)
    monkeypatch.setattr(
        "src.runtime.config_attestation.publish_config_attestation", lambda *_: None
    )
    monkeypatch.setattr(
        "src.runtime.config_attestation.remove_config_attestation", lambda: None
    )
    monkeypatch.setattr("src.api.services.memrl.shutdown_scoring", lambda: None)
    monkeypatch.setattr(
        "src.registry_loader.RegistryLoader", lambda **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        "src.llm_primitives.LLMPrimitives",
        lambda **_kwargs: SimpleNamespace(_backends={}),
    )
    monkeypatch.setattr("src.gate_runner.GateRunner", lambda **_kwargs: object())
    monkeypatch.setattr(
        "src.registry.production.build_tool_registry", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "src.registry.production.build_script_registry", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "src.runtime.instance_topology.get_instance_regions", lambda: {}
    )
    monkeypatch.setattr(
        "src.runtime.cpu_region_lock.sweep_stale_region_lock_payloads", lambda: 0
    )
    monkeypatch.setattr(
        "src.config.get_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(log_dir=tmp_path)),
    )

    async with api.lifespan(FastAPI()):
        yield state


@pytest.mark.asyncio
async def test_flag_off_keeps_heuristic_route_and_leaves_router_unset(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(routing_decision, "_needs_long_context", lambda *_: (False, "test"))
    monkeypatch.setattr(
        "src.classifiers.classify_and_route",
        lambda *_args: SimpleNamespace(role="coder_escalation", strategy="keyword"),
    )

    async with _isolated_lifespan(monkeypatch, tmp_path, enabled=False) as state:
        route, strategy, _ = routing_decision.select_initial_route(
            _request(), state, {}, False, {}, _classify_and_route
        )

        assert state.binding_router is None
        assert route == ["coder_escalation"]
        assert strategy == "keyword"


@pytest.mark.asyncio
async def test_flag_on_lifespan_router_applies_declared_binding_through_real_route(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(routing_decision, "_needs_long_context", lambda *_: (False, "test"))
    monkeypatch.setattr(
        "src.classifiers.classify_and_route",
        lambda *_args: SimpleNamespace(role="coder_escalation", strategy="keyword"),
    )

    async with _isolated_lifespan(monkeypatch, tmp_path, enabled=True) as state:
        assert isinstance(state.binding_router, BindingRouter)
        state.binding_router.add(
            RoutingBinding(
                "code",
                "worker_general",
                BindingPriority.USER_PREF,
                source="test-declared-override",
            )
        )
        route, strategy, _ = routing_decision.select_initial_route(
            _request(), state, {}, False, {}, _classify_and_route
        )

        assert route == ["worker_general"]
        assert strategy == "binding:keyword"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_kwargs", "expected_strategy"),
    [
        ({"force_role": "architect_general"}, "forced"),
        ({"role": "architect_general"}, "explicit"),
    ],
)
async def test_flag_on_preserves_forced_and_explicit_role_targets(
    monkeypatch, tmp_path, request_kwargs, expected_strategy
):
    monkeypatch.setattr(routing_decision, "_needs_long_context", lambda *_: (False, "test"))

    async with _isolated_lifespan(monkeypatch, tmp_path, enabled=True) as state:
        state.binding_router.add(
            RoutingBinding("reasoning", "worker_general", BindingPriority.SESSION)
        )
        route, strategy, _ = routing_decision.select_initial_route(
            _request(**request_kwargs), state, {}, False, {}, _classify_and_route
        )

        assert route == ["architect_general"]
        assert strategy == expected_strategy
