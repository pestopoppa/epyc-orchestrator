"""Tests for orchestrator stack NUMA topology + pinning helper."""

from __future__ import annotations


from scripts.server.stack_numa import (
    MLOCK_ROLES,
    NUMA_CONFIG,
    NUMA_NODE0,
    NUMA_Q0A,
    _numa_prefix,
)


def test_numa_prefix_taskset_only_when_no_policy() -> None:
    # frontdoor instance 1 is NUMA_Q0A with no numactl_policy
    assert _numa_prefix("frontdoor", instance_idx=1) == ["taskset", "-c", NUMA_Q0A[0]]


def test_numa_prefix_wraps_with_numactl_when_policy_present() -> None:
    # architect_general has numactl_policy="interleave=all"
    prefix = _numa_prefix("architect_general", instance_idx=0)
    assert prefix == ["numactl", "--interleave=all", "--", "taskset", "-c", "0-95"]


def test_numa_prefix_returns_empty_for_unknown_role() -> None:
    assert _numa_prefix("nonexistent_role") == []


def test_numa_prefix_returns_empty_for_out_of_range_instance() -> None:
    # frontdoor has 5 instances (indices 0-4); instance 10 is out of range
    assert _numa_prefix("frontdoor", instance_idx=10) == []


def test_numa_prefix_defaults_to_instance_zero() -> None:
    assert _numa_prefix("frontdoor") == _numa_prefix("frontdoor", instance_idx=0)


def test_mlock_roles_derived_from_numa_config() -> None:
    """Every NUMA_CONFIG entry with mlock=True must show up in MLOCK_ROLES."""
    expected = {role for role, cfg in NUMA_CONFIG.items() if cfg.get("mlock")}
    assert MLOCK_ROLES == expected
    # Sanity: frontdoor and worker_general both have mlock
    assert "frontdoor" in MLOCK_ROLES
    assert "worker_general" in MLOCK_ROLES


def test_numa_config_schema_all_instances_are_three_tuples() -> None:
    """Every instance entry must be (cpu_list:str, port:int, threads:int)."""
    for role, cfg in NUMA_CONFIG.items():
        assert "instances" in cfg, f"{role} missing instances"
        for inst in cfg["instances"]:
            assert len(inst) == 3, f"{role}: instance {inst} is not 3-tuple"
            cpu_list, port, threads = inst
            assert isinstance(cpu_list, str), f"{role}: cpu_list not str"
            assert isinstance(port, int), f"{role}: port not int"
            assert isinstance(threads, int), f"{role}: threads not int"
            assert 1 <= threads <= 192


def test_numa_config_ports_are_unique_within_role() -> None:
    for role, cfg in NUMA_CONFIG.items():
        ports = [p for _, p, _ in cfg["instances"]]
        assert len(ports) == len(set(ports)), f"{role}: duplicate ports {ports}"


def test_numa_node0_constant_matches_first_frontdoor_instance() -> None:
    """Regression guard: frontdoor[0] depends on NUMA_NODE0; if NUMA_NODE0 moves, frontdoor[0] must too."""
    cpu_list, _, threads = NUMA_CONFIG["frontdoor"]["instances"][0]
    assert (cpu_list, threads) == NUMA_NODE0
