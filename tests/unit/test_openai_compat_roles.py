from __future__ import annotations

from types import SimpleNamespace

from src.api.routes import openai_compat
from src.api.models import OpenAIMessage


def test_available_roles_falls_back_to_current_live_role_surface(monkeypatch):
    monkeypatch.setattr(openai_compat, "_live_stack_role_ids", lambda: [])
    monkeypatch.setattr(
        openai_compat,
        "HOT_SERVERS",
        [
            {"port": 8070, "roles": ["frontdoor", "coder", "worker_summarize"]},
            {"port": 8072, "roles": ["worker_fast", "worker_math", "toolrunner"]},
            {"port": 8083, "roles": ["architect_general"]},
            {"port": 8085, "roles": ["ingest_long_context"]},
            {"port": 8086, "roles": ["worker_vision"]},
            {"port": 8087, "roles": ["vision_escalation"]},
        ],
    )
    monkeypatch.setattr(
        openai_compat,
        "WARM_SERVERS",
        [{"port": 8070, "roles": ["frontdoor"]}],
    )

    roles = openai_compat.available_roles()

    assert roles[:3] == ["orchestrator", "architect", "worker"]
    assert "worker_fast" not in roles
    assert {
        "frontdoor",
        "coder_escalation",
        "architect_general",
        "worker_general",
        "worker_math",
        "toolrunner",
        "worker_vision",
        "ingest_long_context",
        "vision_escalation",
        "worker_summarize",
    } <= set(roles)
    assert openai_compat._degraded_available_roles() == [
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
        "worker_general",
        "worker_math",
        "toolrunner",
        "architect_general",
        "ingest_long_context",
        "worker_vision",
        "vision_escalation",
    ]


def test_degraded_available_roles_follow_manifest_order(monkeypatch):
    monkeypatch.setattr(
        openai_compat,
        "HOT_SERVERS",
        [
            {"port": 8070, "roles": ["frontdoor", "coder", "worker_summarize"]},
            {"port": 8072, "roles": ["worker_fast", "worker_math", "toolrunner"]},
            {"port": 8090, "roles": ["embedder"]},
        ],
    )
    monkeypatch.setattr(
        openai_compat,
        "WARM_SERVERS",
        [
            {"port": 8083, "roles": ["architect_general"]},
            {"port": 8085, "roles": ["ingest_long_context"]},
        ],
    )
    monkeypatch.setattr(
        openai_compat,
        "ROLE_LAUNCH_META",
        {
            "embedder": {"mode": "embedding"},
            "worker_fast": {"mode": "worker_pool"},
        },
    )

    assert openai_compat._degraded_available_roles() == [
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
        "worker_general",
        "worker_math",
        "toolrunner",
        "architect_general",
        "ingest_long_context",
    ]


def test_available_roles_uses_stack_prior_roles_when_present(monkeypatch):
    monkeypatch.setattr(
        openai_compat,
        "_live_stack_role_ids",
        lambda: ["frontdoor", "toolrunner", "worker_general"],
    )

    assert openai_compat.available_roles() == [
        "orchestrator",
        "architect",
        "worker",
        "frontdoor",
        "toolrunner",
        "worker_general",
    ]


def test_available_roles_canonicalizes_live_stack_role_ids(monkeypatch):
    monkeypatch.setattr(
        openai_compat,
        "_live_stack_role_ids",
        lambda: [
            "frontdoor",
            "worker_fast",
            "worker_general",
            "coder",
            "worker_general",
        ],
    )

    assert openai_compat.available_roles() == [
        "orchestrator",
        "architect",
        "worker",
        "frontdoor",
        "worker_general",
        "coder_escalation",
    ]


def test_ordered_live_role_ids_uses_stack_prior_topology():
    def record(*ports: int, endpoint: str | None = None) -> dict:
        serving = {"ports": list(ports)}
        if endpoint is not None:
            serving["endpoint"] = endpoint
        return {"serving": serving}

    assert openai_compat._ordered_live_role_ids(
        {
            "toolrunner": record(8072),
            "worker_general": record(8072, 8082),
            "frontdoor": record(8080, 8070, endpoint="http://localhost:8070"),
            "vision_escalation": record(8087),
            "worker_summarize": record(8070),
        }
    ) == [
        "frontdoor",
        "worker_summarize",
        "toolrunner",
        "worker_general",
        "vision_escalation",
    ]


def test_degraded_available_roles_follow_server_lists_without_literal_port_map(
    monkeypatch,
):
    monkeypatch.setattr(
        openai_compat,
        "HOT_SERVERS",
        [
            {"port": 8070, "roles": ["frontdoor", "coder", "worker_fast"]},
            {"port": 8072, "roles": ["worker_coder", "worker_math", "toolrunner"]},
        ],
    )
    monkeypatch.setattr(
        openai_compat,
        "WARM_SERVERS",
        [
            {"port": 8083, "roles": ["architect_general"]},
            {"port": 8086, "roles": ["worker_vision"]},
            {"port": 8085, "roles": ["ingest_long_context"]},
            {"port": 8087, "roles": ["vision_escalation"]},
            {"port": 8070, "roles": ["worker_summarize", "embedder"]},
        ],
    )

    assert openai_compat._degraded_available_roles() == [
        "frontdoor",
        "coder_escalation",
        "worker_general",
        "worker_math",
        "toolrunner",
        "architect_general",
        "worker_vision",
        "ingest_long_context",
        "vision_escalation",
        "worker_summarize",
    ]


def test_history_message_dict_preserves_native_tool_fields():
    message = OpenAIMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query":"x"}'},
            }
        ],
    )

    data = openai_compat._history_message_dict(message)

    assert data["role"] == "assistant"
    assert data["content"] == ""
    assert data["tool_calls"][0]["id"] == "call_1"


def test_context_parts_render_tool_history_and_native_repl_bridge():
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    parts = openai_compat._context_parts_from_history(
        history,
        tools,
        {"type": "function", "function": {"name": "web_search"}},
    )
    rendered = "\n".join(parts)

    assert 'Assistant tool_calls: call_1: web_search({"query":"x"})' in rendered
    assert "Tool result call_1: result text" in rendered
    assert 'result = CALL("tool_name", arg=value)' in rendered
    assert "Tool choice policy: web_search." in rendered
    assert "- web_search - Search the web" in rendered


def test_openai_metadata_reports_internal_repl_tool_contract():
    repl = SimpleNamespace(
        _tool_invocations=2,
        _invoked_tools=[
            SimpleNamespace(tool_name="web_search"),
            SimpleNamespace(tool_name="read_file"),
        ],
    )
    meta = {"role": "frontdoor"}

    result = openai_compat._apply_openai_tool_contract_metadata(
        meta,
        request_tools=[{"type": "function", "function": {"name": "web_search"}}],
        repl=repl,
    )

    assert result["native_tool_contract"] == "internal_repl_execution"
    assert result["response_tool_calls"] == "not_emitted"
    assert result["tools_used"] == 2
    assert result["tools_called"] == ["web_search", "read_file"]
