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

from src.registry.stack_priors import live_stack_role_records

ENV_VAR = "ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES"

_DEGRADED_CHAT_COMPLETIONS_ROLES = frozenset(
    {
        "frontdoor",
        "coder_escalation",
        "worker_general",
        "worker_math",
        "worker_summarize",
        "toolrunner",
    }
)


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


def chat_completions_roles() -> set[str]:
    """The set of roles that route through /v1/chat/completions (server-side jinja). Read live."""
    raw = os.environ.get(ENV_VAR)
    if raw is not None:
        return {r.strip() for r in raw.split(",") if r.strip()}

    live_roles = _live_chat_completions_roles()
    return live_roles or set(_DEGRADED_CHAT_COMPLETIONS_ROLES)
