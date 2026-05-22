"""Per-role instance topology — which atomic CPU regions each instance occupies.

Derived from NUMA_CONFIG (scripts/server/stack_numa.py). The single source
of truth for "which (role, instance_idx) pairs can run concurrently" is
the CPU-region overlap: two instances may run concurrently iff their
region sets are disjoint.

The four atomic regions partition the 96 physical cores of the EPYC 9655:
    q0 = cores 0-23   (NUMA node 0, half A)
    q1 = cores 24-47  (NUMA node 0, half B)
    q2 = cores 48-71  (NUMA node 1, half A)
    q3 = cores 72-95  (NUMA node 1, half B)

For each (role, instance_idx) we record the set of quarters it occupies.
The "full" instance for frontdoor (NUMA_NODE0 = 0-47) covers {q0, q1};
the "full" instance for worker_general (0-95) covers {q0, q1, q2, q3}.
Quarter instances cover exactly one region.

This module is import-safe — it does not perform any I/O and has no side
effects. It can be imported from anywhere in the orchestrator process
tree, and from tests, without coupling to running infrastructure.

2026-05-22 — added to support cross-process per-region locking
(`src/runtime/cpu_region_lock.py`). See progress entry for design notes.
"""

from __future__ import annotations

from typing import Iterable


# The four atomic quarters of the EPYC 9655's 96 physical cores.
ATOMIC_REGIONS = ("q0", "q1", "q2", "q3")

# Physical core ranges for each atomic region. Inclusive on both ends.
REGION_CORE_RANGE: dict[str, tuple[int, int]] = {
    "q0": (0, 23),
    "q1": (24, 47),
    "q2": (48, 71),
    "q3": (72, 95),
}


def parse_cpu_list(cpu_list: str) -> set[int]:
    """Parse a taskset-style cpu_list (e.g. '0-23,96-119') into a set of
    physical core IDs. Hyper-thread siblings (96+) are stripped — overlap
    is determined by physical core, not logical CPU.

    Edge cases: empty string returns empty set; single ints work; whitespace
    is tolerated.
    """
    result: set[int] = set()
    if not cpu_list or not cpu_list.strip():
        return result
    for segment in cpu_list.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if "-" in segment:
            lo_s, hi_s = segment.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                continue
            for c in range(lo, hi + 1):
                # Physical cores only — drop HT siblings (96-191)
                if 0 <= c <= 95:
                    result.add(c)
        else:
            try:
                c = int(segment)
            except ValueError:
                continue
            if 0 <= c <= 95:
                result.add(c)
    return result


def cores_to_regions(cores: Iterable[int]) -> frozenset[str]:
    """Return the set of atomic regions touched by an iterable of core IDs.

    A region is "touched" if at least one of its cores appears in `cores`.
    """
    touched: set[str] = set()
    for c in cores:
        for region, (lo, hi) in REGION_CORE_RANGE.items():
            if lo <= c <= hi:
                touched.add(region)
                break
    return frozenset(touched)


def cpu_list_to_regions(cpu_list: str) -> frozenset[str]:
    """Combine `parse_cpu_list` + `cores_to_regions` — convenience for
    consumers that have a taskset-style cpu_list string in hand."""
    return cores_to_regions(parse_cpu_list(cpu_list))


def build_instance_regions(numa_config: dict) -> dict[tuple[str, int], frozenset[str]]:
    """Derive {(role, instance_idx): regions} from a NUMA_CONFIG dict.

    Pure function — caller passes in the config (typically from
    `scripts.server.stack_numa.NUMA_CONFIG`). Tests pass synthetic
    configs.

    Returns one entry per (role, instance_idx). Instances with no CPU
    region overlap with the 0-95 physical cores (e.g. embedders pinned
    to HT-only ranges) get an empty frozenset — treat as non-conflicting
    in the lock layer.
    """
    out: dict[tuple[str, int], frozenset[str]] = {}
    for role, cfg in (numa_config or {}).items():
        instances = cfg.get("instances", [])
        for idx, entry in enumerate(instances):
            if not entry:
                continue
            cpu_list = entry[0] if len(entry) > 0 else ""
            out[(role, idx)] = cpu_list_to_regions(cpu_list)
    return out


def instances_overlap(
    instance_regions: dict[tuple[str, int], frozenset[str]],
    a: tuple[str, int],
    b: tuple[str, int],
) -> bool:
    """True iff (role_a, idx_a) and (role_b, idx_b) share at least one
    atomic region — i.e. they cannot run concurrently without CPU
    contention.

    Useful for tests + diagnostics. The lock layer doesn't call this
    directly; it just acquires the union of region locks for each
    instance and lets fcntl handle the rest.
    """
    return bool(instance_regions.get(a, frozenset()) & instance_regions.get(b, frozenset()))


# ── Derived module-level table (lazy import to avoid circulars) ─────────

_INSTANCE_REGIONS_CACHE: dict[tuple[str, int], frozenset[str]] | None = None


def get_instance_regions() -> dict[tuple[str, int], frozenset[str]]:
    """Return the live mapping derived from the orchestrator's NUMA_CONFIG.

    Memoized for cheap repeated access. The NUMA_CONFIG is effectively
    immutable at runtime, so caching is safe. Tests should use
    `build_instance_regions` directly with a synthetic config.
    """
    global _INSTANCE_REGIONS_CACHE
    if _INSTANCE_REGIONS_CACHE is None:
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
            _INSTANCE_REGIONS_CACHE = build_instance_regions(NUMA_CONFIG)
        except Exception:
            # Defensive: if stack_numa import path differs in some
            # deployment, return an empty mapping (lock layer treats
            # missing entries as no-CPU-conflict → no-op blocking).
            _INSTANCE_REGIONS_CACHE = {}
    return _INSTANCE_REGIONS_CACHE
