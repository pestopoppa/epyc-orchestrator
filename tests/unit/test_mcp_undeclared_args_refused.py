#!/usr/bin/env python3
"""HS-4 P0.3 support — the orchestrator MCP server must not drop arguments silently.

The OpenCode session plugin stamps ``session_id`` onto ``orchestrator_*`` and
``memory_*`` MCP calls. A tool that does not declare it must REFUSE the call
rather than run without the key. Under the pinned ``fastmcp>=3`` this holds
(``additionalProperties: false`` plus a pydantic ``unexpected_keyword_argument``
error); this test pins it, so a dependency change or a server launched under a
different interpreter/version cannot make the drop silent again.
"""

from __future__ import annotations

import asyncio

from fastmcp import Client

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


def test_undeclared_session_id_is_refused_not_dropped():
    async def _call():
        async with Client(mcp) as client:
            return await client.call_tool(
                "orchestrator_route_explain",
                {"prompt": "hi", "session_id": "ses_1"},
                raise_on_error=False,
            )

    result = _run(_call())
    assert result.is_error
    assert "session_id" in result.content[0].text
