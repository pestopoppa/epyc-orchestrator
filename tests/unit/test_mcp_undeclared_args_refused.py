#!/usr/bin/env python3
"""HS-4 P0.3 support — the orchestrator MCP server must not drop arguments silently.

The OpenCode session plugin stamps ``session_id`` onto ``orchestrator_*`` and
``memory_*`` MCP calls. A tool that does not declare it must REFUSE the call
rather than run without the key. Under the pinned ``fastmcp>=3`` this holds
(``additionalProperties: false`` plus a pydantic ``unexpected_keyword_argument``
error); these tests pin it, so a dependency change or a server launched under a
different interpreter/version cannot make the drop silent again.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client

from src import mcp_server
from src.mcp_server import mcp


def _run(coro):
    return asyncio.run(coro)


def test_every_orchestrator_tool_schema_forbids_undeclared_arguments():
    async def _schemas():
        async with Client(mcp) as client:
            return {t.name: t.inputSchema for t in await client.list_tools()}

    schemas = _run(_schemas())
    assert schemas, "no MCP tools registered"
    permissive = sorted(
        name for name, schema in schemas.items() if schema.get("additionalProperties") is not False
    )
    assert permissive == []


def test_undeclared_argument_is_refused_not_dropped():
    async def _call():
        async with Client(mcp) as client:
            return await client.call_tool(
                "orchestrator_route_explain",
                {"prompt": "hi", "x_undeclared_probe": "v"},
                raise_on_error=False,
            )

    result = _run(_call())
    assert result.is_error
    assert "x_undeclared_probe" in result.content[0].text


# HS-4 P0-MCP-b. OpenCode exposes these tools as "orchestrator_<tool>", and the plugin
# stamps session_id onto every one of them. A tool that does not declare it refuses
# every stamped call.
def test_every_orchestrator_tool_declares_session_id():
    async def _schemas():
        async with Client(mcp) as client:
            return {t.name: t.inputSchema for t in await client.list_tools()}

    schemas = _run(_schemas())
    missing = sorted(
        name
        for name, schema in schemas.items()
        if schema.get("properties", {}).get("session_id", {}).get("type") != "string"
        or "session_id" in schema.get("required", [])
    )
    assert missing == []


@pytest.mark.parametrize("tool", ["orchestrator_chat", "orchestrator_route_explain"])
def test_stamped_session_id_reaches_the_chat_payload(tool, monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(mcp_server, "_is_mcp_chat_enabled", lambda: True)
    monkeypatch.setattr(mcp_server, "_post_chat", lambda payload: sent.append(payload) or {"answer": "ok"})

    async def _call():
        async with Client(mcp) as client:
            return await client.call_tool(
                tool, {"prompt": "hi", "session_id": "ses_1"}, raise_on_error=False
            )

    result = _run(_call())
    assert not result.is_error
    assert [p.get("session_id") for p in sent] == ["ses_1"]


def test_stamped_session_id_is_accepted_by_a_session_independent_tool():
    async def _call():
        async with Client(mcp) as client:
            return await client.call_tool(
                "list_canvases",
                {"directory": "/nonexistent-hs4-probe", "session_id": "ses_1"},
                raise_on_error=False,
            )

    assert not _run(_call()).is_error


@pytest.mark.parametrize("tool", ["orchestrator_chat", "orchestrator_route_explain"])
def test_empty_session_id_is_not_forwarded(tool, monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(mcp_server, "_is_mcp_chat_enabled", lambda: True)
    monkeypatch.setattr(mcp_server, "_post_chat", lambda payload: sent.append(payload) or {"answer": "ok"})
    getattr(mcp_server, tool)("hi")
    assert "session_id" not in sent[0]


# HS-4 P0-MCP-a. The refusal above holds only under the pinned fastmcp, so every
# checked-in launcher of this server must use the project venv interpreter.
_VENV_PYTHON = "/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python"
_REPO = Path(__file__).resolve().parents[2]


def test_script_registry_launcher_uses_the_project_venv():
    from src.mcp_client import load_server_configs

    config = load_server_configs(_REPO / "orchestration" / "mcp_servers.yaml")["orchestrator"]
    assert config.command == _VENV_PYTHON


def test_repo_mcp_json_launches_the_server_from_the_project_venv():
    servers = json.loads((_REPO / ".mcp.json").read_text())["mcpServers"]
    assert servers["orchestrator"]["command"] == _VENV_PYTHON
