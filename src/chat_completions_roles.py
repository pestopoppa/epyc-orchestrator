"""Single source of truth for which orchestrator roles route through ``/v1/chat/completions``
(server-side jinja templating + thinking-off) vs ``/completion``.

The backend router (`src/llm_primitives/backend.py`) and the chat route's orchestrator-side
template-SKIP logic (`src/api/routes/chat.py`) MUST agree on this set: if a role routes to
chat-completions (server applies the GGUF jinja template) but the chat route still applies an
orchestrator-side template, the role is DOUBLE-templated; if the inverse, it is un-templated.
Before 2026-05-27 the two carried divergent inline defaults (chat.py omitted
``frontdoor,coder_escalation,architect_general``) and the env var is unset in prod, so the
defaults were load-bearing — hence this shared definition.

Read live from ``ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES`` so an A/B can flip it across restarts.
"""
from __future__ import annotations

import os

ENV_VAR = "ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES"

# frontdoor/coder_escalation/architect_general use chat-completions for thinking-off (the J1
# degenerate-<think> fix, 2026-05-23); ingest_long_context is EXCLUDED — thinking-on is
# load-bearing for Qwen3-Next-80B. coder_escalation shares the :8070 server with frontdoor, which
# is launched with --jinja, so server-side templating applies to it too.
DEFAULT_CHAT_COMPLETIONS_ROLES = (
    "worker_general,worker_explore,worker_math,worker_summarize,worker_coder,"
    "frontdoor,coder_escalation,architect_general"
)


def chat_completions_roles() -> set[str]:
    """The set of roles that route through /v1/chat/completions (server-side jinja). Read live."""
    raw = os.environ.get(ENV_VAR, DEFAULT_CHAT_COMPLETIONS_ROLES)
    return {r.strip() for r in raw.split(",") if r.strip()}
