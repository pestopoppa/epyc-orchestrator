"""Single source of truth for which orchestrator roles route through ``/v1/chat/completions``
(server-side jinja templating + thinking-off) vs ``/completion``.

The backend router (`src/llm_primitives/backend.py`) and the chat route's orchestrator-side
template-SKIP logic (`src/api/routes/chat.py`) MUST agree on this set: if a role routes to
chat-completions (server applies the GGUF jinja template) but the chat route still applies an
orchestrator-side template, the role is DOUBLE-templated; if the inverse, it is un-templated.
The live default is derived from generated stack priors so stack swaps do not require code edits.
If the priors artifact is missing or malformed, we fall back to a narrow degraded set instead of
the full historical literal table.

Read live from ``ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES`` so an A/B can flip it across restarts.
"""
from __future__ import annotations

import os
from typing import Any

from src.registry.stack_priors import live_stack_role_records
from src.roles import Role

try:
    from scripts.server.stack_manifest import HOT_SERVERS, ROLE_LAUNCH_META, WARM_SERVERS
except Exception:  # pragma: no cover - catastrophic import fallback
    HOT_SERVERS = ()
    WARM_SERVERS = ()
    ROLE_LAUNCH_META = {}

ENV_VAR = "ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES"


def _live_chat_completions_roles() -> set[str]:
    """Return live roles that the generated priors mark as chat-completions users."""
    try:
        records = live_stack_role_records()
    except Exception:
        return set()

    roles: set[str] = set()
    for role, record in records.items():
        launch = record.get("serving", {}).get("launch", {})
        runtime = launch.get("runtime", {})
        flags = runtime.get("flags", {})
        acceleration = record.get("acceleration", {})
        if (
            isinstance(flags, dict)
            and flags.get("jinja") is True
            and isinstance(acceleration, dict)
            and acceleration.get("enable_thinking") is False
        ):
            roles.add(role)
    return roles


def _launch_primary_for_role(role: str) -> tuple[str | None, dict[str, Any]]:
    """Return the primary launch role and metadata for a raw or canonical role."""
    canonical = Role.from_string(role)
    candidates = [role]
    if canonical is not None and str(canonical) not in candidates:
        candidates.append(str(canonical))

    for candidate in candidates:
        meta = ROLE_LAUNCH_META.get(candidate)
        if isinstance(meta, dict):
            return candidate, meta

    for primary, meta in ROLE_LAUNCH_META.items():
        if not isinstance(meta, dict):
            continue
        shared = meta.get("shared_with_first_n")
        if not isinstance(shared, list):
            continue
        if any(candidate in shared for candidate in candidates):
            return str(primary), meta

    return None, {}


def _degraded_chat_completions_roles() -> set[str]:
    """Derive the narrow fallback set from launch-manifest roles and order."""
    roles: set[str] = set()
    for server in tuple(HOT_SERVERS) + tuple(WARM_SERVERS):
        if not isinstance(server, dict):
            continue
        for role in server.get("roles") or ():
            if not isinstance(role, str):
                continue

            primary, meta = _launch_primary_for_role(role)
            mode = meta.get("mode")
            canonical = Role.from_string(role)
            role_value = str(canonical) if canonical is not None else role

            if mode == "default" and primary == str(Role.FRONTDOOR):
                roles.add(role_value)
            elif mode == "worker_pool" and meta.get("worker_type") == "explore":
                roles.add(role_value)

    return roles


def chat_completions_roles() -> set[str]:
    """The set of roles that route through /v1/chat/completions (server-side jinja). Read live."""
    raw = os.environ.get(ENV_VAR)
    if raw is not None:
        return {r.strip() for r in raw.split(",") if r.strip()}

    live_roles = _live_chat_completions_roles()
    return live_roles or _degraded_chat_completions_roles()
