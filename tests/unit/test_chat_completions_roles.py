"""Unit tests for the shared chat-completions-roles source of truth (review finding #5 2026-05-27).

The chat route's template-skip logic and the backend router must agree on this set; before the
shared module they carried divergent inline defaults (chat.py omitted frontdoor/coder_escalation/
architect_general) and the env var is unset in prod, so the defaults were load-bearing.
"""
from src.chat_completions_roles import chat_completions_roles


def test_default_includes_validated_chat_completion_roles(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES", raising=False)
    roles = chat_completions_roles()
    # the roles backend.py routes to /v1/chat/completions for thinking-off must ALL be present, so
    # the chat route skips its orchestrator-side template wrap for them (no double-templating).
    for r in ("coder_escalation", "frontdoor", "architect_general",
              "worker_general", "worker_coder"):
        assert r in roles
    # ingest_long_context is excluded — thinking-on is load-bearing for Qwen3-Next-80B.
    assert "ingest_long_context" not in roles


def test_env_override_and_whitespace(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES", " foo , bar ,")
    assert chat_completions_roles() == {"foo", "bar"}
