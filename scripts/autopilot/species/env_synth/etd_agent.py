"""Environment-Task Discovery (ETD) agent (NIB2-44 AW-1).

Wraps an LLM ReAct loop over ``web_search``, ``fetch_url``, and MCP
tool enumeration. Its job: given a seed theme (or an autopilot
gap-descriptor like "need more medium-difficulty math reasoning with
tool use"), discover candidate environments, enumerate their MCP-
accessible tools, and return structured ``EnvironmentDiscovery`` records.

The agent itself is dependency-light: it accepts three pluggable
callables (``llm``, ``web_search``, ``fetch_url``) and a tool-enumeration
probe. Tests use fakes for all four.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from scripts.autopilot.species.env_synth.mcp_tool_registry import (
    MCPToolEntry,
    MCPToolRegistry,
)

log = logging.getLogger("autopilot.env_synth.etd_agent")


# Pluggable I/O contracts.
LLMCall = Callable[[str, str], Awaitable[str]]
WebSearch = Callable[[str, int], Awaitable[list[dict[str, Any]]]]
FetchURL = Callable[[str], Awaitable[str]]
ToolEnum = Callable[[str], Awaitable[list[MCPToolEntry]]]  # endpoint → [entries]


@dataclass
class EnvironmentDiscovery:
    environment_id: str
    theme: str
    description: str
    sources: list[str] = field(default_factory=list)    # URLs consulted
    tools: list[MCPToolEntry] = field(default_factory=list)
    discovered_at: str = ""


@dataclass
class ETDAgent:
    """ReAct agent over web + MCP enumeration. Keeps a short trajectory
    log for later meta-optimization and autopilot rollup.
    """

    llm: LLMCall
    web_search: WebSearch
    fetch_url: FetchURL
    tool_enum: ToolEnum
    registry: MCPToolRegistry

    max_react_steps: int = 4
    max_tools_per_env: int = 32

    async def discover(
        self,
        theme: str,
        *,
        gap_descriptor: str = "",
        max_environments: int = 5,
    ) -> list[EnvironmentDiscovery]:
        """Produce up to ``max_environments`` discoveries for ``theme``.

        The ReAct trace is minimal by design: search → narrow → enumerate.
        A richer agent can replace this with an LLM-driven loop; for
        Phase 1 scaffolding the deterministic shape keeps the code
        reviewable.
        """
        system = (
            "You are an Environment-Task Discovery agent for an autonomous "
            "research pipeline. Given a theme, emit a JSON list of candidate "
            "environments (name, description, search_queries) that expose "
            "MCP tools or equivalent web APIs the pipeline can probe."
        )
        user = f"Theme: {theme}\nGap descriptor: {gap_descriptor}\n"

        raw = await self.llm(system, user)
        try:
            candidates = json.loads(raw)
            if not isinstance(candidates, list):
                raise ValueError("expected a JSON list of environments")
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("ETD LLM output not usable: %s", e)
            return []

        discoveries: list[EnvironmentDiscovery] = []
        for cand in candidates[:max_environments]:
            name = str(cand.get("name", "")).strip()
            if not name:
                continue
            environment_id = _make_env_id(name)
            description = str(cand.get("description", "")).strip()
            queries = cand.get("search_queries") or [name]

            sources: list[str] = []
            endpoints: set[str] = set()

            for q in queries[: self.max_react_steps]:
                try:
                    results = await self.web_search(q, 5)
                except Exception as e:
                    log.debug("web_search for %r failed: %s", q, e)
                    continue
                for r in results:
                    url = r.get("url") or r.get("link")
                    if not url:
                        continue
                    sources.append(url)
                    if _looks_like_mcp_endpoint(url):
                        endpoints.add(url)

            collected_tools: list[MCPToolEntry] = []
            for endpoint in list(endpoints)[: self.max_tools_per_env]:
                try:
                    tools = await self.tool_enum(endpoint)
                except Exception as e:
                    log.debug("tool_enum for %s failed: %s", endpoint, e)
                    continue
                for t in tools[: self.max_tools_per_env]:
                    t.environment_id = environment_id
                    t.discovered_via = "etd_web"
                    self.registry.register(t)
                    collected_tools.append(t)

            discoveries.append(EnvironmentDiscovery(
                environment_id=environment_id,
                theme=theme,
                description=description,
                sources=sources,
                tools=collected_tools,
                discovered_at=datetime.now(timezone.utc).isoformat(),
            ))

        return discoveries


# ── helpers ─────────────────────────────────────────────────────────


def _make_env_id(name: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    return f"env_{slug[:48]}" if slug else "env_unknown"


def _looks_like_mcp_endpoint(url: str) -> bool:
    """Conservative heuristic: MCP endpoints expose /mcp, /jsonrpc, or /tools."""
    url_l = url.lower()
    return any(
        needle in url_l
        for needle in ("/mcp", "/jsonrpc", "/tools", "mcp.json", "openapi.json", "/.well-known/mcp")
    )
