from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import httpx

from scripts.research.deploy_ec2_lease import (
    dispatch_lane_b,
    dry_run_plan,
    execute_sequence,
    lane_b_client_body,
    run_session_id,
)
from src.api import app
from src.api.routes import openai_compat
from src.api.state import get_state, reset_state
from src.features import reset_features


def test_dry_run_includes_generated_lane_b_session_id():
    plan = dry_run_plan("http://127.0.0.1:8000", 8083)

    assert plan["lane_a"]["url"].endswith("/v1/passthrough/architect_critic/chat/completions")
    assert plan["lane_a"]["body_keys"] == ["chat_template_kwargs", "max_tokens", "messages"]
    assert plan["lane_b"]["url"].endswith("/v1/chat/completions")
    assert plan["lane_b"]["mode"] == "client"
    assert "x_session_id" in plan["lane_b"]["body_keys"]
    assert run_session_id() != run_session_id()


def test_actual_lane_b_dispatch_is_refused_without_id_and_accepts_generated_id(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    import src.llm_primitives as llm_primitives

    reset_features()
    reset_state()
    state = get_state()
    state.registry = MagicMock()

    primitives = MagicMock()
    primitives.chat_completion_call.return_value = {
        "content": "ok", "tool_calls": [], "finish_reason": "stop"
    }
    primitives.llm_call.side_effect = AssertionError("client mode must not use llm_call")
    primitives.total_tokens_generated = 1
    primitives.get_request_usage.return_value = openai_compat._client_usage(None).model_dump()
    monkeypatch.setattr(llm_primitives, "LLMPrimitives", lambda **_kw: primitives)

    messages = [{"role": "user", "content": "fixture request"}]
    body = lane_b_client_body(session_id="ses_ec2_fixture_01", model="orchestrator",
                              messages=messages)
    body["stream"] = True
    missing = dict(body)
    missing.pop("x_session_id")

    async def exercise():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            refused = await client.post("/v1/chat/completions", json=missing)
            assert refused.status_code == 422
            assert "x_session_id is required" in refused.json()["detail"]

            session_id, result = await dispatch_lane_b(
                client, "http://test", model="orchestrator", messages=messages)
            assert session_id.startswith("ses_ec2_")
            assert result["status_code"] == 200
            assert result["response_bytes"] > 0

    asyncio.run(exercise())

    primitives.chat_completion_call.assert_called_once()
    reset_state()
    reset_features()


def test_execute_overlaps_lane_a_with_lane_b_before_credit_call():
    lane_a_started = asyncio.Event()
    lane_b_posted = asyncio.Event()
    lane_a_finished = asyncio.Event()
    credit_seen = asyncio.Event()
    observed_session_ids: list[str] = []
    lane_a_count = 0

    class HeldLaneA(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.wait_for(lane_b_posted.wait(), timeout=1)
            lane_a_finished.set()
            yield b"data: lane-a-complete\n\n"

        async def aclose(self):
            return None

    class MockOrchestrator(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal lane_a_count
            if request.url.path == "/v1/passthrough/architect_critic/chat/completions":
                lane_a_count += 1
                if lane_a_count == 1:
                    lane_a_started.set()
                    return httpx.Response(200, stream=HeldLaneA())
                assert lane_a_finished.is_set(), "credit request must follow both lease requests"
                credit_seen.set()
                return httpx.Response(200, content=b"credit-complete")
            if request.url.path == "/v1/chat/completions":
                assert lane_a_started.is_set()
                assert not lane_a_finished.is_set(), "lane A must remain open when lane B posts"
                request_body = json.loads(request.content)
                observed_session_ids.append(request_body["x_session_id"])
                lane_b_posted.set()
                return httpx.Response(200, content=b"lane-b-complete")
            raise AssertionError(f"unexpected route: {request.url.path}")

    config = {
        "lane_a_chat_template_kwargs": {"enable_thinking": False},
        "lane_a_first_messages": [{"role": "user", "content": "real lane A input"}],
        "lane_b_messages": [{"role": "user", "content": "real lane B input"}],
        "lane_a_credit_messages": [
            {"role": "user", "content": "real lane A input plus extension"}
        ],
        "lane_b_model": "orchestrator",
    }

    async def exercise():
        async with httpx.AsyncClient(transport=MockOrchestrator(), base_url="http://test") as client:
            return await execute_sequence("http://test", config, delay_s=0, client=client)

    result = asyncio.run(exercise())
    assert credit_seen.is_set()
    assert lane_a_count == 2
    assert len(observed_session_ids) == 1
    assert result["lane_b_session_id"] == observed_session_ids[0]
    assert observed_session_ids[0].startswith("ses_ec2_")


def test_cli_dry_run_prints_x_session_id_without_network(capsys):
    from scripts.research import deploy_ec2_lease

    rc = deploy_ec2_lease.main(["--dry-run"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run; no requests sent"
    assert "x_session_id" in plan["lane_b"]["body_keys"]
