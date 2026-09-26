"""NUMA_FULL_T48 and the split thread invariant (DAR-LAT-3h, lineup-change C3).

`_assert_instance_invariants` used to require `-t == physical cores` for every
CPU decode instance. That equality stood in for the real rule (no SMT
OVERsubscription) and also rejected a deliberate under-subscription such as
architect_critic's 48 threads on the 96-core full cpuset. The rule is now split:
over-subscription is fatal for every shape, and under-subscription is fatal
unless the shape is registered in `_UNDERSUBSCRIBED_SHAPES`.
"""

from __future__ import annotations

import pytest

from scripts.server.stack_numa import (
    _CPU_SHAPES,
    _SHAPE_CLASSES,
    _UNDERSUBSCRIBED_SHAPES,
    NUMA_FULL,
    NUMA_FULL_T48,
    _assert_instance_invariants,
)


def _one(cpus: str, threads: int, shape: str, policy: str = "interleave=all"):
    config = {"_fixture": {"instances": [(cpus, 9999, threads)], "numactl_policy": policy}}
    return config, {"_fixture": (shape,)}


def test_t48_shape_is_the_full_cpuset_at_48_threads() -> None:
    assert NUMA_FULL_T48 == (NUMA_FULL[0], 48)
    assert _CPU_SHAPES["NUMA_FULL_T48"] == NUMA_FULL_T48
    # Same cpuset => same region set => same class, so slots_by_shape {full: N} still resolves.
    assert _SHAPE_CLASSES["NUMA_FULL_T48"] == _SHAPE_CLASSES["NUMA_FULL"] == "full"


def test_registered_undersubscription_imports() -> None:
    config, shapes = _one("0-95", 48, "NUMA_FULL_T48")
    _assert_instance_invariants(config, shapes)


def test_unregistered_undersubscription_is_still_fatal() -> None:
    # -t 48 on 0-95 under the NUMA_FULL name is a typo, not a decision.
    config, shapes = _one("0-95", 48, "NUMA_FULL")
    with pytest.raises(AssertionError, match="not registered in _UNDERSUBSCRIBED_SHAPES"):
        _assert_instance_invariants(config, shapes)


def test_oversubscription_is_fatal_even_for_a_registered_shape() -> None:
    # The original defect (the deleted NUMA_NODE0: 48 physical cores, 96 threads).
    config, shapes = _one("0-47,96-143", 96, "NUMA_FULL_T48", policy="interleave=0,1")
    with pytest.raises(AssertionError, match="SMT oversubscription"):
        _assert_instance_invariants(config, shapes)


def test_only_the_intended_shape_is_registered() -> None:
    assert _UNDERSUBSCRIBED_SHAPES == frozenset({"NUMA_FULL_T48"})


def test_live_wiring_passes() -> None:
    _assert_instance_invariants()
