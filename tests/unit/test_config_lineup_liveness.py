"""ESC-8 Fix 5: producer-lineup liveness validation in src.config.models.

A producer-derived server lineup (env-filter branch OR runtime-facts branch)
must be validated against the live fleet before it is trusted. A lineup whose
quarterable host-role ports (frontdoor 8070 / worker_general 8072 /
ingest_long_context 8085) are all dead — the env=full / valid-full-manifest
poison signature on a quarters-only fleet — is rejected and resolution falls
through to the next producer (ultimately stack priors).

All liveness is injected/mocked: these tests never open a real socket.
"""

from __future__ import annotations

import sys
import types

import pytest

from src.config import models

# Dead-in-quarter full host ports; anything else is a live quarter/single-role port.
_FULL_HOST_PORTS = {8070, 8072, 8085}


def _quarters_live(port: int) -> bool:
    """Quarters-only fleet: the full host ports are dead, everything else live."""
    return port not in _FULL_HOST_PORTS


def _fulls_live(_port: int) -> bool:
    """Full fleet up: every probed port (incl. 8070/8072/8085) is listening."""
    return True


def _all_dead(_port: int) -> bool:
    return False


def _server(port: int, *roles: str) -> dict:
    return {"port": port, "roles": list(roles)}


# --------------------------------------------------------------------------- #
# _selected_servers_are_live                                                   #
# --------------------------------------------------------------------------- #


def test_are_live_rejects_dead_full_host_ports() -> None:
    """env=full-style lineup naming only the dead full host ports is rejected."""
    lineup = [
        _server(8070, "frontdoor"),
        _server(8072, "worker_general"),
        _server(8085, "ingest_long_context"),
    ]
    assert models._selected_servers_are_live(lineup, probe=_quarters_live) is False
    assert models._selected_servers_are_live(lineup, probe=_fulls_live) is True


def test_are_live_accepts_live_quarter_host_ports() -> None:
    lineup = [
        _server(8082, "worker_general"),
        _server(8182, "worker_general"),
        _server(8080, "frontdoor"),
    ]
    assert models._selected_servers_are_live(lineup, probe=_quarters_live) is True


def test_are_live_empty_lineup_is_never_trusted() -> None:
    assert models._selected_servers_are_live(None, probe=_fulls_live) is False
    assert models._selected_servers_are_live([], probe=_fulls_live) is False


def test_are_live_no_host_discriminator_requires_one_live_port() -> None:
    """Without any quarterable host role, require at least one live port anywhere
    so a lineup of only dead ports is still rejected."""
    only_dead = [_server(9999, "worker_fast")]
    assert models._selected_servers_are_live(only_dead, probe=_all_dead) is False
    one_live = [_server(8102, "worker_fast")]
    assert models._selected_servers_are_live(one_live, probe=lambda p: p == 8102) is True


# --------------------------------------------------------------------------- #
# _runtime_or_env_selected_servers — producer precedence + validation          #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolate_probe(monkeypatch):
    """Never touch a real socket; default to 'all dead' unless a test overrides."""
    monkeypatch.setattr(models, "_port_listening", _all_dead)
    models._PORT_LISTENING_CACHE.clear()
    yield
    models._PORT_LISTENING_CACHE.clear()


def _patch_runtime_facts(monkeypatch, value):
    import scripts.server.runtime_facts_manifest as rfm

    monkeypatch.setattr(rfm, "read_runtime_stack_selected_servers", lambda **_kw: value)


def test_env_full_with_only_quarters_live_falls_through(monkeypatch, caplog) -> None:
    """env=full + quarters-only fleet → env lineup rejected → None (priors)."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    monkeypatch.setattr(models, "_port_listening", _quarters_live)
    _patch_runtime_facts(monkeypatch, None)

    with caplog.at_level("WARNING"):
        result = models._runtime_or_env_selected_servers()

    assert result is None
    assert any("lineup rejected" in rec.message for rec in caplog.records)


def test_env_full_with_full_ports_live_resolves_full(monkeypatch) -> None:
    """env=full + full ports live → the full manifest lineup is returned."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    monkeypatch.setattr(models, "_port_listening", _fulls_live)

    result = models._runtime_or_env_selected_servers()

    assert result is not None
    ports = {s.get("port") for s in result}
    # The frontdoor full host port must be present in a resolved full lineup.
    assert 8070 in ports


def test_env_unset_manifest_dead_ports_rejected(monkeypatch, caplog) -> None:
    """env unset + runtime-facts naming dead host ports → rejected → None."""
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    monkeypatch.setattr(models, "_port_listening", _quarters_live)
    _patch_runtime_facts(
        monkeypatch,
        [
            _server(8070, "frontdoor"),
            _server(8072, "worker_general"),
            _server(8085, "ingest_long_context"),
        ],
    )

    with caplog.at_level("WARNING"):
        result = models._runtime_or_env_selected_servers()

    assert result is None
    assert any("runtime-facts lineup rejected" in rec.message for rec in caplog.records)


def test_env_unset_manifest_live_ports_returned(monkeypatch) -> None:
    """env unset + runtime-facts naming live quarter host ports → returned."""
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    monkeypatch.setattr(models, "_port_listening", _quarters_live)
    live_lineup = [
        _server(8082, "worker_general"),
        _server(8080, "frontdoor"),
        _server(8185, "ingest_long_context"),
    ]
    _patch_runtime_facts(monkeypatch, live_lineup)

    result = models._runtime_or_env_selected_servers()

    assert result == live_lineup


def test_env_branch_importerror_is_logged_not_swallowed(monkeypatch, caplog) -> None:
    """Fix 5(b): the env-filter ImportError is logged at WARNING, not swallowed."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    monkeypatch.setattr(models, "_port_listening", _quarters_live)
    _patch_runtime_facts(monkeypatch, None)
    # Force the env-branch `from scripts.server.stack_manifest import ...` to fail
    # exactly as it does under the uvicorn circular-import chain.
    empty = types.ModuleType("scripts.server.stack_manifest")
    monkeypatch.setitem(sys.modules, "scripts.server.stack_manifest", empty)

    with caplog.at_level("WARNING"):
        result = models._runtime_or_env_selected_servers()

    assert result is None
    assert any("env-filter branch failed" in rec.message for rec in caplog.records)
