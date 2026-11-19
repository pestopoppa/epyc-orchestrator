"""M-12k memory-off switch: episodic lookups on the chat routing path can be skipped, default unchanged.

Pins: (1) default = lookups run (byte-identical); (2) per-request ``episodic_lookups=False`` skips the
HybridRouter on initial-role routing, mode selection, the try-cheap-first Q gate and the plan-review Q
gate; (3) the ``episodic_routing_lookups`` flag does the same globally and toggles via POST /config with
no restart; (4) a request ``True`` never overrides a flag that is off; (5) it is on the autopilot surface.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.api.models import ChatRequest
from src.api.routes.chat_pipeline import routing_decision
from src.api.routes.chat_review import _needs_plan_review
from src.api.routes.chat_routing import _select_mode
from src.api.services.episodic_switch import episodic_lookups_enabled
from src.features import _REGISTRY_BY_NAME, Features, features, get_features, reset_features


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(tmp_path / "runtime_flags.json"))
    monkeypatch.delenv("ORCHESTRATOR_EPISODIC_ROUTING_LOOKUPS", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_FEATURE_EPISODIC_ROUTING_LOOKUPS", raising=False)
    reset_features()
    yield
    reset_features()


def _router():
    router = MagicMock()
    router.route.return_value = (["worker_general"], "learned")
    router.route_with_mode.return_value = (["frontdoor"], "learned", "direct")
    router.retriever.retrieve_for_routing.return_value = []
    return router


def _classify(prompt, context, **_kw):
    return "frontdoor", "rules"


def _route(request, router):
    state = SimpleNamespace(hybrid_router=router, binding_router=None)
    return routing_decision.select_initial_route(request, state, {}, False, {}, _classify)


@pytest.fixture
def no_long_context(monkeypatch):
    monkeypatch.setattr(routing_decision, "_needs_long_context", lambda *_: (False, "test"))


def test_registered_default_on_and_dataclass_default_agree():
    spec = _REGISTRY_BY_NAME["episodic_routing_lookups"]
    assert spec.default_test is True and spec.default_prod is True
    assert Features().episodic_routing_lookups is True
    assert features().episodic_routing_lookups is True


def test_default_and_unset_request_field_run_lookups():
    assert episodic_lookups_enabled(None) is True
    assert episodic_lookups_enabled(ChatRequest(prompt="x")) is True
    assert ChatRequest(prompt="x").episodic_lookups is None
    assert episodic_lookups_enabled(ChatRequest(prompt="x", episodic_lookups=True)) is True


def test_initial_route_default_uses_router(no_long_context):
    router = _router()
    route, strategy, _ = _route(ChatRequest(prompt="hello", real_mode=True, mock_mode=False), router)
    assert (route, strategy) == (["worker_general"], "learned")
    router.route.assert_called_once()


def test_initial_route_request_off_skips_router(no_long_context):
    router = _router()
    req = ChatRequest(prompt="hello", real_mode=True, mock_mode=False, episodic_lookups=False)
    route, strategy, _ = _route(req, router)
    assert strategy == "rules"
    router.route.assert_not_called()
    router.route_with_skills.assert_not_called()


def test_select_mode_default_vs_off():
    router = _router()
    state = SimpleNamespace(hybrid_router=router)
    long_prompt = "Refactor this module into two files and explain the tradeoffs in detail " * 6
    assert _select_mode(long_prompt, "", state) == "direct"  # learned mode wins by default
    assert router.route_with_mode.call_count == 1
    router.route_with_mode.reset_mock()
    assert _select_mode(long_prompt, "", state, episodic_lookups=False) == "repl"
    router.route_with_mode.assert_not_called()


def test_plan_review_q_gate_skipped_when_off(monkeypatch):
    from src import proactive_delegation

    monkeypatch.setattr(
        proactive_delegation, "classify_task_complexity",
        lambda _o: (proactive_delegation.TaskComplexity.MODERATE, []),
    )
    router = _router()
    state = SimpleNamespace(plan_review_phase="B", hybrid_router=router)
    _needs_plan_review({"objective": "task"}, ["coder"], state)
    assert router.retriever.retrieve_for_routing.call_count == 1
    router.retriever.retrieve_for_routing.reset_mock()
    assert _needs_plan_review({"objective": "task"}, ["coder"], state, episodic_lookups=False) is True
    router.retriever.retrieve_for_routing.assert_not_called()


def test_flag_off_is_global_and_request_true_cannot_override(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_EPISODIC_ROUTING_LOOKUPS", "0")
    reset_features()
    assert get_features().episodic_routing_lookups is False
    assert episodic_lookups_enabled(ChatRequest(prompt="x")) is False
    assert episodic_lookups_enabled(ChatRequest(prompt="x", episodic_lookups=True)) is False


def test_runtime_toggle_via_post_config_without_restart():
    from src.api.routes.config import update_config
    from src import flag_env as flag_env_module

    def post(**flags):
        class Request:
            client = SimpleNamespace(host="127.0.0.1")

            async def json(self):
                return dict(flags)

        flag_env_module.reset_cache()
        return asyncio.run(update_config(Request(), current=features()))

    assert episodic_lookups_enabled(None) is True
    post(episodic_routing_lookups=False)
    assert episodic_lookups_enabled(None) is False
    post(episodic_routing_lookups=True)
    assert episodic_lookups_enabled(None) is True


def test_flag_is_on_the_autopilot_surface():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert '"episodic_routing_lookups",' in (root / "scripts/autopilot/config_applicator.py").read_text()
    assert "| `episodic_routing_lookups` |" in (root / "scripts/autopilot/program.md").read_text()
    assert "episodic_routing_lookups" in Features().summary()


def test_flag_is_in_runtime_flags_spec():
    from pathlib import Path

    spec = Path(__file__).resolve().parents[2] / "orchestration" / "runtime_flags.spec.yaml"
    assert "  episodic_routing_lookups: baseline" in spec.read_text()
