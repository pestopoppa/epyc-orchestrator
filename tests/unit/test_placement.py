"""WP-2 tests: evaluate_placement filters candidates by topology overlap.

Pure-function tests against the placement-policy module. Dispatcher
integration is exercised separately via test_concurrency_aware_*.
"""

from __future__ import annotations

import pytest

from src.scheduling.placement import (
    Place,
    PlacementResult,
    Queue,
    QueueReason,
    evaluate_placement,
)


# Synthetic frontdoor topology (matches stack_numa.py shape):
#   instance 0 = full (NUMA_NODE0)   → {q0, q1}
#   instance 1 = q0_inst             → {q0}
#   instance 2 = q1_inst             → {q1}
#   instance 3 = q2_inst             → {q2}
#   instance 4 = q3_inst             → {q3}
FRONTDOOR_TOPOLOGY = {
    ("frontdoor", 0): frozenset({"q0", "q1"}),
    ("frontdoor", 1): frozenset({"q0"}),
    ("frontdoor", 2): frozenset({"q1"}),
    ("frontdoor", 3): frozenset({"q2"}),
    ("frontdoor", 4): frozenset({"q3"}),
}

# Dispatcher priority order (matches _dispatch candidate construction):
#   (-1, 0) full, then quarters in NUMA-disjoint preference order:
#   q_idx 2 (topo 3 → q2), q_idx 3 (topo 4 → q3), q_idx 0 (topo 1 → q0), q_idx 1 (topo 2 → q1)
# For these tests we use the simplest order: full, q3, q2, q1, q0 — matches
# the actual preference for frontdoor's NUMA_NODE0 full.
FRONTDOOR_CANDIDATES = [
    (-1, 0),  # full
    (3, 4),   # q3 (NUMA-disjoint)
    (2, 3),   # q2 (NUMA-disjoint)
    (1, 2),   # q1 (overlaps full)
    (0, 1),   # q0 (overlaps full)
]


# ── No holders → full is the first safe candidate ──────────────────────


def test_no_holders_full_first() -> None:
    r = evaluate_placement("frontdoor", FRONTDOOR_CANDIDATES, [], FRONTDOOR_TOPOLOGY)
    assert not r.is_queue
    assert r.places[0] == Place(internal_idx=-1, topology_idx=0)
    # All 5 candidates are safe when no holders exist.
    assert len(r.places) == 5


# ── Full held → q3 + q2 safe; q1 + q0 overlap → filtered ───────────────


def test_full_held_filters_overlapping_quarters() -> None:
    r = evaluate_placement("frontdoor", FRONTDOOR_CANDIDATES, [0], FRONTDOOR_TOPOLOGY)
    assert not r.is_queue
    # full(0) overlaps with itself → filtered. q3(4) and q2(3) disjoint → kept.
    # q1(2) and q0(1) overlap full's {q0,q1} → filtered.
    assert r.places == (Place(3, 4), Place(2, 3))


def test_full_plus_q3_held_only_q2_safe() -> None:
    r = evaluate_placement("frontdoor", FRONTDOOR_CANDIDATES, [0, 4], FRONTDOOR_TOPOLOGY)
    assert not r.is_queue
    assert r.places == (Place(2, 3),)  # only q2 left


def test_full_plus_q3_plus_q2_held_no_safe_candidate() -> None:
    r = evaluate_placement("frontdoor", FRONTDOOR_CANDIDATES, [0, 3, 4], FRONTDOOR_TOPOLOGY)
    assert r.is_queue
    assert r.queue.reason is QueueReason.TOPOLOGY_OVERLAP
    # Holders union covers all 4 regions → every candidate overlaps.
    assert set(r.queue.blocking_instance_idxs) == {0, 1, 2, 3, 4}
    assert "holders=[0, 3, 4]" in r.queue.detail


def test_all_four_quarters_held_no_safe() -> None:
    """Sanity: 4 quarters in use → full blocked (overlap) → queue."""
    r = evaluate_placement("frontdoor", FRONTDOOR_CANDIDATES, [1, 2, 3, 4], FRONTDOOR_TOPOLOGY)
    assert r.is_queue
    assert r.queue.reason is QueueReason.TOPOLOGY_OVERLAP


# ── Cross-role isolation: holders on a different role don't affect this role ─


def test_holders_on_different_role_ignored() -> None:
    # Even though "worker_general" instance 0 covers everything, it shouldn't
    # affect frontdoor placement — same_role-only filter.
    topology = {**FRONTDOOR_TOPOLOGY, ("worker_general", 0): frozenset({"q0", "q1", "q2", "q3"})}
    r = evaluate_placement(
        "frontdoor",
        FRONTDOOR_CANDIDATES,
        [],  # no frontdoor holders
        topology,
    )
    assert not r.is_queue
    assert len(r.places) == 5


# ── Sticky-quarter scenario: q3 listed first via sticky session ────────


def test_sticky_quarter_promoted_to_head() -> None:
    sticky_candidates = [
        (3, 4),     # sticky q3 first
        (-1, 0),    # full
        (2, 3),     # q2
        (1, 2),     # q1
        (0, 1),     # q0
    ]
    r = evaluate_placement("frontdoor", sticky_candidates, [], FRONTDOOR_TOPOLOGY)
    assert r.places[0] == Place(3, 4)


# ── Candidate without region info (e.g. embedder on HT-only) → safe fallthrough ─


def test_unknown_region_candidate_is_considered_safe() -> None:
    """If a candidate's (role, idx) has no region entry, treat as overlap-free
    and pass through. The lock layer is the final arbiter."""
    candidates = [(-1, 0), (99, 99)]  # second candidate has no topology entry
    topology = {("frontdoor", 0): frozenset({"q0", "q1"})}
    r = evaluate_placement("frontdoor", candidates, [0], topology)
    # full is filtered (overlaps itself), unknown candidate is kept.
    assert r.places == (Place(99, 99),)


# ── Empty candidate list returns queue ─────────────────────────────────


def test_empty_candidates_returns_queue() -> None:
    r = evaluate_placement("frontdoor", [], [], FRONTDOOR_TOPOLOGY)
    assert r.is_queue
    assert r.queue.reason is QueueReason.TOPOLOGY_OVERLAP


# ── Worker-general shape: full=0-95 → every quarter overlaps ────────────


WORKER_GENERAL_TOPOLOGY = {
    ("worker_general", 0): frozenset({"q0", "q1", "q2", "q3"}),  # full = 0-95
    ("worker_general", 1): frozenset({"q0"}),
    ("worker_general", 2): frozenset({"q1"}),
    ("worker_general", 3): frozenset({"q2"}),
    ("worker_general", 4): frozenset({"q3"}),
}
WORKER_GENERAL_CANDIDATES = [
    (-1, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
]


def test_worker_general_full_held_no_safe_quarter() -> None:
    r = evaluate_placement(
        "worker_general",
        WORKER_GENERAL_CANDIDATES,
        [0],
        WORKER_GENERAL_TOPOLOGY,
    )
    assert r.is_queue
    assert r.queue.reason is QueueReason.TOPOLOGY_OVERLAP
    # All 5 candidates overlap full's {q0,q1,q2,q3} or self.
    assert set(r.queue.blocking_instance_idxs) == {0, 1, 2, 3, 4}


def test_worker_general_quarter_held_full_blocked() -> None:
    """One quarter in use → full overlaps → queue (no other safe candidate)."""
    r = evaluate_placement(
        "worker_general",
        WORKER_GENERAL_CANDIDATES,
        [1],  # q0 in use
        WORKER_GENERAL_TOPOLOGY,
    )
    # full overlaps q0 (its {q0,q1,q2,q3} ⊃ {q0}); q0 overlaps itself; q2/q3 disjoint.
    assert not r.is_queue
    assert r.places == (Place(1, 2), Place(2, 3), Place(3, 4))
