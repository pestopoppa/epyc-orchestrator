"""Reader-agreement tests for stack NUMA mode projections."""

from __future__ import annotations

from collections import defaultdict

import pytest

from scripts.server.stack_numa_mode import (
    DASHBOARD_RUNTIME_FALLBACK_NUMA_MODE,
    DEFAULT_STACK_NUMA_MODE,
    env_stack_numa_mode,
    normalize_stack_numa_mode,
)


def _dashboard_ports(mode: str) -> dict[str, list[int]]:
    from src.api.routes import dashboard_topology

    out: dict[str, set[int]] = defaultdict(set)
    for service in dashboard_topology.expected_stack_services(mode):
        port = service.get("port")
        roles = service.get("roles") or []
        if not isinstance(port, int):
            continue
        for role in roles:
            out[str(role)].add(port)
    return {role: sorted(ports) for role, ports in out.items()}


def _stack_prior_ports(mode: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[int]]:
    from src.registry.stack_priors import _stack_manifest_info

    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", mode)
    _aliases, roles = _stack_manifest_info()
    return {
        role: sorted(record.get("ports") or [])
        for role, record in roles.items()
        if record.get("ports")
    }


def _stack_change_guard_ports(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[int]]:
    from scripts.validate.stack_change_guard import _launch_manifest_targets

    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", mode)
    return {
        role: sorted(target.get("ports") or [])
        for role, target in _launch_manifest_targets().items()
        if target.get("ports")
    }


def _manifest_ports(mode: str) -> dict[str, list[int]]:
    from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS, _filter_by_numa_mode

    out: dict[str, set[int]] = defaultdict(set)
    for server in _filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, mode):
        port = server.get("port")
        roles = server.get("roles") or []
        if not isinstance(port, int):
            continue
        for role in roles:
            out[str(role)].add(port)
    return {role: sorted(ports) for role, ports in out.items()}


def test_stack_numa_mode_defaults_are_named() -> None:
    assert DEFAULT_STACK_NUMA_MODE == "full"
    assert DASHBOARD_RUNTIME_FALLBACK_NUMA_MODE == "both"
    assert normalize_stack_numa_mode(None) == "full"
    assert normalize_stack_numa_mode("stale", default="both") == "both"
    assert env_stack_numa_mode(environ={}) == "full"
    assert (
        env_stack_numa_mode(
            default="both",
            environ={"ORCHESTRATOR_STACK_NUMA_MODE": "QUARTER"},
        )
        == "quarter"
    )


@pytest.mark.parametrize("mode", ["full", "quarter", "both"])
def test_stack_numa_readers_agree_on_host_role_ports(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host/primary roles: the serving view (stack priors) and every
    launch-topology view (raw manifest, dashboard, stack-change guard) agree
    exactly across NUMA modes."""
    stack_prior_ports = _stack_prior_ports(mode, monkeypatch)
    dashboard_ports = _dashboard_ports(mode)
    guard_ports = _stack_change_guard_ports(mode, monkeypatch)
    manifest_ports = _manifest_ports(mode)

    host_roles = {
        "frontdoor",
        "worker_general",
        "ingest_long_context",
        "vision_escalation",
    }
    for role in host_roles:
        assert manifest_ports.get(role) == stack_prior_ports.get(role)
        assert dashboard_ports.get(role) == stack_prior_ports.get(role)
        assert guard_ports.get(role) == stack_prior_ports.get(role)


@pytest.mark.parametrize("mode", ["full", "quarter", "both"])
def test_alias_serving_fleet_converges_on_host_launch_views_diverge(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP-13: alias roles (shared_with_first_n) diverge by design.

    The launch-topology views (raw manifest, dashboard, stack-change guard) still
    report only the instances the alias is tagged onto (shared_with_first_n_count),
    and continue to agree with each other. The serving/routing view (stack priors)
    now inherits the host's FULL fleet so it converges with the operative-layer
    Fix-A delegated URL (worker_math/coder_escalation -> host fleet). In
    single-instance ``full`` mode the two coincide; in quarter/both the serving
    fleet is a strict superset of the launch-tagged subset.
    """
    stack_prior_ports = _stack_prior_ports(mode, monkeypatch)
    dashboard_ports = _dashboard_ports(mode)
    guard_ports = _stack_change_guard_ports(mode, monkeypatch)
    manifest_ports = _manifest_ports(mode)

    alias_hosts = {
        "coder_escalation": "frontdoor",
        "worker_summarize": "frontdoor",
    }
    for alias, host in alias_hosts.items():
        # Launch-topology views still agree with each other on the tagged subset.
        assert manifest_ports.get(alias) == dashboard_ports.get(alias)
        assert manifest_ports.get(alias) == guard_ports.get(alias)
        # Fleet convergence: the alias serving fleet equals its host serving fleet.
        assert stack_prior_ports.get(alias) == stack_prior_ports.get(host)
        # And the serving fleet is a superset of the launch-tagged subset.
        tagged = set(manifest_ports.get(alias) or [])
        serving = set(stack_prior_ports.get(alias) or [])
        assert tagged <= serving
        if mode == "full":
            assert tagged == serving
        else:
            assert tagged < serving
