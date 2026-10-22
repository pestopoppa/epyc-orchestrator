"""registry_role_urls honours ORCHESTRATOR_SERVER_URLS_<ROLE> (KPF-LB1.1 capture finding, 2026-10-22).

An in-process harness points every role at ONE private server through the env vars ServerURLsConfig
reads. A context-limit lookup made with no URLs (compaction on primitives without ``server_urls``,
context-recovery reroute candidates) used to resolve the role from the compiled stack priors and GET
/props on the PRODUCTION port. The override must win; without it nothing changes.
"""

from __future__ import annotations

import pytest

from src.backends import context_limits as cl

PRIORS = {"coder_escalation": "http://localhost:8083", "frontdoor": "full:http://localhost:8070,http://localhost:8080"}


@pytest.fixture
def priors(monkeypatch):
    import src.registry.stack_priors as sp

    monkeypatch.setattr(sp, "live_stack_serving_url_values", lambda _path=None: dict(PRIORS))
    for k in [k for k in __import__("os").environ if k.startswith(cl.SERVER_URLS_ENV_PREFIX)]:
        monkeypatch.delenv(k)


def test_no_override_is_the_priors(priors):
    assert cl.registry_role_urls() == {
        "coder_escalation": ["http://localhost:8083"],
        "frontdoor": ["http://localhost:8070", "http://localhost:8080"],
    }


def test_override_wins_for_its_role_only(priors, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_SERVER_URLS_CODER_ESCALATION", "http://127.0.0.1:18283")
    urls = cl.registry_role_urls()
    assert urls["coder_escalation"] == ["http://127.0.0.1:18283"]
    assert urls["frontdoor"] == ["http://localhost:8070", "http://localhost:8080"]


def test_resolver_fallback_follows_the_override(priors, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_SERVER_URLS_CODER_ESCALATION", "http://127.0.0.1:18283")
    seen: list[str] = []

    def fetch(url, _timeout):
        seen.append(url)
        return None

    r = cl.ContextLimitResolver(fetch_props=fetch, registry_facts=dict, live=True)
    r.limit_for_role("coder_escalation")          # no URLs passed: the fallback path
    assert seen == ["http://127.0.0.1:18283"]


def test_empty_override_is_ignored(priors, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_SERVER_URLS_CODER_ESCALATION", "  ")
    assert cl.registry_role_urls()["coder_escalation"] == ["http://localhost:8083"]
