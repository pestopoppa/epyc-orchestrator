from __future__ import annotations

from src.api.routes import openai_compat


def test_available_roles_falls_back_to_current_live_role_surface(monkeypatch):
    monkeypatch.setattr(openai_compat, "_live_stack_role_ids", lambda: [])
    monkeypatch.setattr(
        openai_compat,
        "HOT_SERVERS",
        [
            {"port": 8070, "roles": ["frontdoor", "coder_escalation", "worker_summarize"]},
            {"port": 8072, "roles": ["worker_general", "worker_math", "toolrunner"]},
            {"port": 8083, "roles": ["architect_general"]},
            {"port": 8085, "roles": ["ingest_long_context"]},
            {"port": 8086, "roles": ["worker_vision"]},
            {"port": 8087, "roles": ["vision_escalation"]},
        ],
    )
    monkeypatch.setattr(
        openai_compat,
        "WARM_SERVERS",
        [{"port": 8070, "roles": ["frontdoor"]}],
    )

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
    assert openai_compat._degraded_available_roles() == [
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
    ]


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


def test_ordered_live_role_ids_uses_stack_prior_topology():
    def record(*ports: int) -> dict:
        return {"serving": {"ports": list(ports)}}

    assert openai_compat._ordered_live_role_ids(
        {
            "toolrunner": record(8072),
            "worker_general": record(8072, 8082),
            "frontdoor": record(8070, 8080),
            "vision_escalation": record(8087),
            "worker_summarize": record(8070),
        }
    ) == [
        "frontdoor",
        "worker_summarize",
        "toolrunner",
        "worker_general",
        "vision_escalation",
    ]


def test_degraded_available_roles_follow_server_lists_without_literal_port_map(
    monkeypatch,
):
    monkeypatch.setattr(
        openai_compat,
        "HOT_SERVERS",
        [
            {"port": 8070, "roles": ["frontdoor", "coder_escalation"]},
            {"port": 8072, "roles": ["worker_general", "worker_math", "toolrunner"]},
        ],
    )
    monkeypatch.setattr(
        openai_compat,
        "WARM_SERVERS",
        [
            {"port": 8083, "roles": ["architect_general"]},
            {"port": 8086, "roles": ["worker_vision"]},
            {"port": 8085, "roles": ["ingest_long_context"]},
            {"port": 8087, "roles": ["vision_escalation"]},
            {"port": 8070, "roles": ["worker_summarize", "embedder"]},
        ],
    )

    assert openai_compat._degraded_available_roles() == [
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
    ]
