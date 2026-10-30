"""Native proactive planning preserves its structured, suffix-free call contract."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from src.api.routes.chat_pipeline import proactive_stage as stage
from src.proactive_delegation import TaskComplexity


def request_and_routing():
    req = SimpleNamespace(
        prompt="complex task",
        context="context",
        real_mode=True,
        task_root=None,
        context_bundle=None,
    )
    routing = SimpleNamespace(
        task_id="fixture", task_ir={"task_type": "chat"}, routing_decision=["frontdoor"]
    )
    return req, routing


@pytest.mark.parametrize("inline", [True, False])
def test_native_call_schema_suffix_and_single_step(monkeypatch, inline):
    import src.proactive_delegation as delegation

    monkeypatch.setattr(stage, "features", lambda: SimpleNamespace(parallel_execution=True))
    monkeypatch.setattr(
        delegation, "classify_task_complexity", lambda _: (TaskComplexity.COMPLEX, None)
    )
    monkeypatch.setattr(stage, "_should_inline_plan_call_for_test", lambda _: inline)
    recorded = []
    monkeypatch.setattr(
        stage, "_log_plan_steps_parsed", lambda state, task_id, steps: recorded.extend(steps)
    )
    primitive = SimpleNamespace(
        llm_call=MagicMock(
            return_value=json.dumps([{"id": "S1", "action": "work", "parallel_group": "g"}])
        )
    )
    req, routing = request_and_routing()
    assert (
        asyncio.run(stage._execute_proactive(req, routing, primitive, SimpleNamespace(), 0)) is None
    )
    kwargs = primitive.llm_call.call_args.kwargs
    assert kwargs["n_tokens"] == 256 and kwargs["skip_suffix"] is True
    schema = kwargs["json_schema"]
    item = schema["items"]
    assert item["required"] == ["id", "action"]
    assert set(item["properties"]) == {
        "id",
        "action",
        "actor",
        "parallel_group",
        "depends_on",
        "outputs",
    }
    assert item["properties"]["actor"]["enum"] == ["worker", "coder", "architect"]
    assert recorded == [
        {
            "id": "S1",
            "action": "work",
            "parallel_group": "g",
            "actor": "worker",
            "depends_on": [],
            "outputs": [],
        }
    ]


def test_empty_output_never_calls_repair():
    primitive = SimpleNamespace(llm_call=MagicMock(return_value="invented"))
    assert (
        asyncio.run(stage._decompose_plan_steps("  ", primitives=primitive, task_id="empty")) == []
    )
    primitive.llm_call.assert_not_called()


def test_real_primitive_emits_schema_and_no_critic_suffix(monkeypatch):
    from src.llm_primitives import LLMPrimitives
    from src.registry.registry_loader import RegistryLoader
    import src.proactive_delegation as delegation

    monkeypatch.setattr(stage, "features", lambda: SimpleNamespace(parallel_execution=True))
    monkeypatch.setattr(
        delegation, "classify_task_complexity", lambda _: (TaskComplexity.COMPLEX, None)
    )
    monkeypatch.setattr(stage, "_log_plan_steps_parsed", lambda *args: None)
    sent = []

    def send(self, request, *args, **kwargs):
        if request.url.path == "/completion":
            payload = json.loads(request.read())
            sent.append(payload)
            body = {
                "content": '[{\\"id\\":\\"S1\\",\\"action\\":\\"work\\"}]',
                "stop": True,
                "tokens_predicted": 12,
                "tokens_evaluated": 20,
                "timings": {},
            }
            body["content"] = json.dumps([{"id": "S1", "action": "work"}])
            if payload.get("stream"):
                return httpx.Response(
                    200,
                    content=("data: " + json.dumps(body) + "\n\n").encode(),
                    headers={"content-type": "text/event-stream"},
                    request=request,
                )
            return httpx.Response(200, json=body, request=request)
        return httpx.Response(200, json={"status": "ok"}, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    registry = RegistryLoader(validate_paths=False)
    primitive = LLMPrimitives(
        mock_mode=False,
        server_urls={"architect_critic": "http://127.0.0.1:19997"},
        registry=registry,
        server_urls_source="request",
    )
    req, routing = request_and_routing()
    assert (
        asyncio.run(stage._execute_proactive(req, routing, primitive, SimpleNamespace(), 0)) is None
    )
    assert len(sent) == 1
    assert sent[0]["json_schema"] == stage._PLAN_STEPS_SCHEMA
    assert "Attack the plan" not in sent[0]["prompt"]
    assert "complex task" in sent[0]["prompt"]
