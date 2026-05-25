"""WP-2: within-role placement state machine.

Replaces the dispatcher's "try full first, then any quarter" greedy with a
topology-aware filter: never place a request on a cpuset that overlaps any
currently-active instance for the same role. If no safe candidate exists,
return a Queue decision so the caller polls for a release rather than
serializing on full's lock (which silently allowed full+quarter overlap
when a quarter held an overlapping region).

Topology overlap is a HARD safety veto — `placement_overlap=true` blocks
the candidate regardless of the measured throughput matrix verdict. The
matrix's `same_role.instance_pairs` (WP-6) layers throughput gates on top
of disjoint pairs that still underperform serial. This separation matches
the 2026-05-25 audit refinement in
handoffs/active/within-role-placement-state-machine.md § Phase 2.

This module is consumed by `ConcurrencyAwareBackend._dispatch` (WP-2 wiring)
and exposes its decision shape so tests + telemetry consumers can introspect
the *reason* a request was queued.

Cross-ref: handoffs/active/within-role-placement-state-machine.md § Phase 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


class QueueReason(str, Enum):
    """Why a request was queued instead of placed immediately.

    Surfaced in dashboard / telemetry so operators can distinguish safe
    queuing (topology_overlap; expected under high concurrency) from
    capacity bugs (deadline_exceeded; should be rare).
    """

    TOPOLOGY_OVERLAP = "topology_overlap"          # every candidate overlaps an in-flight holder
    MATRIX_FLOOR = "matrix_floor"                  # disjoint but matrix says ratio < floor (WP-6)
    MIGRATION_IN_FLIGHT = "migration_in_flight"    # waiting for a KV migration to complete (WP-3)
    DEADLINE_EXCEEDED = "deadline_exceeded"        # caller's deadline elapsed before a slot opened


@dataclass(frozen=True)
class Place:
    """A safe placement candidate, in dispatch-priority order.

    `internal_idx` matches ConcurrencyAwareBackend's signed indexing
    (-1 = full, 0..N-1 = quarter i+1 in NUMA_CONFIG). `topology_idx` is
    the unsigned NUMA_CONFIG position (0 = full, 1..N = quarters).
    """

    internal_idx: int
    topology_idx: int


@dataclass(frozen=True)
class Queue:
    """Decision: no safe placement; caller should poll-and-retry until deadline."""

    reason: QueueReason
    blocking_instance_idxs: tuple[int, ...] = field(default_factory=tuple)
    detail: str = ""


@dataclass(frozen=True)
class PlacementResult:
    """Either an ordered list of safe `Place` candidates (try non-blocking
    in order, race-tolerant) or a single `Queue` decision."""

    places: tuple[Place, ...] = ()
    queue: Optional[Queue] = None

    @property
    def is_queue(self) -> bool:
        return self.queue is not None


def _holder_regions_union(
    role: str,
    holder_idxs: Iterable[int],
    instance_regions: dict[tuple[str, int], frozenset[str]],
) -> frozenset[str]:
    """Compute the union of CPU regions held by all in-flight instances for `role`."""
    accum: set[str] = set()
    for idx in holder_idxs:
        accum |= instance_regions.get((role, idx), frozenset())
    return frozenset(accum)


def evaluate_placement(
    role: str,
    candidates: list[tuple[int, int]],
    holder_idxs: Iterable[int],
    instance_regions: dict[tuple[str, int], frozenset[str]],
) -> PlacementResult:
    """Filter dispatcher-priority `candidates` to those whose cpuset is
    disjoint from the union of regions held by current `holder_idxs`.

    Args:
      role: role name (e.g. "frontdoor"). Used to key into `instance_regions`.
      candidates: ordered list of (internal_idx, topology_idx) tuples in
        dispatcher priority order — sticky_quarter first, full next,
        quarters in NUMA-disjoint preference order last. Matches the
        existing `ConcurrencyAwareBackend._dispatch` candidate construction.
      holder_idxs: instance indices currently holding region locks for `role`,
        from `cpu_region_lock.active_region_holders()`. Use the topology_idx
        convention (0 = full, 1..N = quarters).
      instance_regions: full {(role, topology_idx): regions} mapping from
        `instance_topology.get_instance_regions()` or a test fixture.

    Returns:
      PlacementResult with `places=[…]` listing the safe candidates in
      priority order, OR `queue=Queue(...)` if every candidate overlaps
      an in-flight holder.
    """
    holders_union = _holder_regions_union(role, holder_idxs, instance_regions)

    safe: list[Place] = []
    blocking: list[int] = []
    for internal_idx, topology_idx in candidates:
        cand_regions = instance_regions.get((role, topology_idx), frozenset())
        if not cand_regions:
            # No region info for candidate (e.g. embedder on HT-only cores) →
            # treat as overlap-free; caller falls through to lock acquisition.
            safe.append(Place(internal_idx=internal_idx, topology_idx=topology_idx))
            continue
        if cand_regions & holders_union:
            blocking.append(topology_idx)
            continue
        safe.append(Place(internal_idx=internal_idx, topology_idx=topology_idx))

    if safe:
        return PlacementResult(places=tuple(safe))

    return PlacementResult(
        queue=Queue(
            reason=QueueReason.TOPOLOGY_OVERLAP,
            blocking_instance_idxs=tuple(sorted(set(blocking))),
            detail=(
                f"role={role} all {len(candidates)} candidate(s) overlap "
                f"holders={sorted(set(holder_idxs))} "
                f"(holder regions: {sorted(holders_union)})"
            ),
        ),
    )
