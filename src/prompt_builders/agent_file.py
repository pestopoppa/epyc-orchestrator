"""Agent-file prompt injection (gated by the ``agent_file_injection`` flag).

``agent_file_block(role)`` returns the delimited block to append to a role's
system/instruction prompt, or "" when the flag is off, the role is outside the
``ORCHESTRATOR_AGENT_FILE_ROLES`` allowlist, or no agent file resolves. With
the flag off every caller's output is byte-identical to the pre-F2 prompt.
"""

from __future__ import annotations

import os

AGENT_FILE_ROLES_ENV = "ORCHESTRATOR_AGENT_FILE_ROLES"
DEFAULT_AGENT_FILE_ROLES = "worker_general,architect_general"


def _allowed_roles() -> set[str]:
    raw = os.environ.get(AGENT_FILE_ROLES_ENV, DEFAULT_AGENT_FILE_ROLES)
    return {r.strip() for r in raw.split(",") if r.strip()}


def agent_file_block(role: object) -> str:
    """Delimited agent-file block for ``role`` or "" (never raises)."""
    try:
        from src.features import features as _get_features

        if not getattr(_get_features(), "agent_file_injection", False):
            return ""
        role_name = str(getattr(role, "value", role) or "")
        if role_name not in _allowed_roles():
            return ""
        from src.config.agent_files import (
            read_agent_file_text,
            resolve_agent_file,
            resolve_agent_file_level,
        )

        path = resolve_agent_file(role_name)
        if path is None:
            return ""
        text = read_agent_file_text(path)
        if not text:
            return ""
        level = resolve_agent_file_level(role_name)
        return f"## Engineering standards (agent file, level {level})\n\n{text}"
    except Exception:
        return ""


def with_agent_file(prompt: str, role: object) -> str:
    """Append the agent-file block to ``prompt`` (unchanged when empty)."""
    block = agent_file_block(role)
    if not block:
        return prompt
    return f"{prompt}\n\n{block}" if prompt else block
