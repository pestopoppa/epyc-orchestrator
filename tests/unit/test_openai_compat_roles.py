from __future__ import annotations

from src.api.routes import openai_compat


def test_available_roles_falls_back_to_current_live_role_surface(monkeypatch):
    monkeypatch.setattr(openai_compat, "_live_stack_role_ids", lambda: [])

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


def test_ordered_role_ids_prefers_current_operator_surface_order():
    assert openai_compat._ordered_role_ids(
        [
            "toolrunner",
            "worker_general",
            "frontdoor",
            "vision_escalation",
            "worker_summarize",
        ]
    ) == [
        "frontdoor",
        "worker_general",
        "vision_escalation",
        "worker_summarize",
        "toolrunner",
    ]
