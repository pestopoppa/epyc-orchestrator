"""Unit tests for the shared chat-completions-roles source of truth (review finding #5 2026-05-27).

The chat route's template-skip logic and the backend router must agree on this set; before the
shared module they carried divergent inline defaults. The env var is unset in prod, so the
generated stack-prior default is load-bearing.
"""
from src.roles import Role
from src.chat_completions_roles import chat_completions_roles


def test_default_comes_from_live_stack_priors(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES", raising=False)
    monkeypatch.setattr(
        "src.chat_completions_roles._live_chat_completions_roles",
        lambda: {
            "coder_escalation",
            "frontdoor",
            "toolrunner",
            "worker_general",
            "worker_math",
            "worker_summarize",
        },
    )

    roles = chat_completions_roles()

    assert roles == {
        "coder_escalation",
        "frontdoor",
        "toolrunner",
        "worker_general",
        "worker_math",
        "worker_summarize",
    }


def test_degraded_fallback_is_narrow_when_priors_missing(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES", raising=False)
    monkeypatch.setattr("src.chat_completions_roles._live_chat_completions_roles", lambda: set())

    assert chat_completions_roles() == {
        str(Role.FRONTDOOR),
        str(Role.CODER_ESCALATION),
        str(Role.WORKER_GENERAL),
        str(Role.WORKER_MATH),
        str(Role.WORKER_SUMMARIZE),
        str(Role.TOOLRUNNER),
    }


def test_env_override_and_whitespace(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES", " foo , bar ,")
    assert chat_completions_roles() == {"foo", "bar"}
