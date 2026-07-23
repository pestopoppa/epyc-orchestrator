"""Phase D — topology-aware quarter preference in ConcurrencyAwareBackend.

Per the contention matrix (handoff line 88, frontdoor full + own q3 = 1.71×),
when the full instance is busy the scheduler should pick a quarter on the
opposite NUMA half BEFORE quarters that overlap full's cpu-set.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

ca_mod = importlib.import_module("src.backends.concurrency_aware")
stack_numa = importlib.import_module("stack_numa")


class _StubBackend:
    """Minimal backend stub for ConcurrencyAwareBackend construction."""

    def __init__(self, url: str = "http://localhost:0"):
        self.config = type("C", (), {"base_url": url})()
        self.url = url


def _make_concurrency_aware(role: str) -> "ca_mod.ConcurrencyAwareBackend":
    """Construct a ConcurrencyAwareBackend with stub backends matching the
    instance count in NUMA_CONFIG for the given role."""
    instances = stack_numa.NUMA_CONFIG[role]["instances"]
    full = _StubBackend(f"http://localhost:{instances[0][1]}")
    quarters = [_StubBackend(f"http://localhost:{inst[1]}") for inst in instances[1:]]
    return ca_mod.ConcurrencyAwareBackend(
        full_backend=full,
        quarter_backends=quarters,
        role=role,
        full_port=instances[0][1],
    )


def test_frontdoor_quarter_preference_prefers_disjoint_NUMA() -> None:
    """frontdoor full is on NUMA_NODE0 (0-47,96-143).
    Quarters: q0=Q0A (0-23), q1=Q0B (24-47), q2=Q1A (48-71), q3=Q1B (72-95).
    Disjoint from full: q2, q3. Overlapping: q0, q1.
    Preferred order: [q2, q3, q0, q1] (disjoint first, then numerical within bucket)."""
    cab = _make_concurrency_aware("frontdoor")
    # Indices are 0-based for the QUARTERS list (i.e., q_idx in [0..3])
    # Per the spec: disjoint quarters (indices 2, 3) come BEFORE overlapping (0, 1)
    pref = cab._quarter_preference_order
    assert pref.index(2) < pref.index(0)
    assert pref.index(2) < pref.index(1)
    assert pref.index(3) < pref.index(0)
    assert pref.index(3) < pref.index(1)


def test_ingest_quarter_preference_matches_frontdoor_pattern() -> None:
    """ingest_long_context full is also on NUMA_NODE0 — same preference shape."""
    cab = _make_concurrency_aware("ingest_long_context")
    pref = cab._quarter_preference_order
    assert pref.index(2) < pref.index(0)
    assert pref.index(3) < pref.index(0)


def test_quarter_preference_full_on_NODE1_synthetic(monkeypatch) -> None:
    """NODE1-full disjointness ordering, exercised via a SYNTHETIC role.

    Historically this used vision_escalation (full on NUMA_NODE1 + 4 quarters),
    but that role became single-instance in the v7 one-instance vision layout
    (2026-07-20 recert) and no real NODE1-full multi-instance role remains. The
    ORDERING LOGIC is topology-generic and stays covered here: full on NODE1
    (48-95); quarters q0 (0-23) and q1 (24-47) are disjoint from full and must
    order BEFORE overlapping q2 (48-71) and q3 (72-95)."""
    synthetic = {
        "instances": [
            ("NUMA_NODE1", 9070),
            ("NUMA_Q0A", 9071),
            ("NUMA_Q0B", 9072),
            ("NUMA_Q1A", 9073),
            ("NUMA_Q1B", 9074),
        ],
        "full_instance_idx": 0,
        "cpu_lists": None,
    }
    real = stack_numa.NUMA_CONFIG.get("vision_escalation")
    monkeypatch.setitem(stack_numa.NUMA_CONFIG, "synthetic_node1_full", {
        **synthetic,
        # mirror the key shape the preference computation reads
        **({} if real is None else {k: v for k, v in real.items() if k not in synthetic}),
    })
    cab = _make_concurrency_aware("synthetic_node1_full")
    pref = cab._quarter_preference_order
    assert pref.index(0) < pref.index(2)
    assert pref.index(0) < pref.index(3)
    assert pref.index(1) < pref.index(2)
    assert pref.index(1) < pref.index(3)


def test_worker_general_quarter_preference_full_on_FULL_SOCKET() -> None:
    """worker_general full is on NUMA_FULL (0-95) — ALL quarters overlap.
    No quarter is disjoint, so the preference is just numerical order."""
    cab = _make_concurrency_aware("worker_general")
    pref = cab._quarter_preference_order
    assert pref == [0, 1, 2, 3]


def test_quarter_preference_fallback_when_numa_config_missing() -> None:
    """If the role isn't in NUMA_CONFIG, fall back to numerical order."""
    full = _StubBackend()
    quarters = [_StubBackend() for _ in range(3)]
    cab = ca_mod.ConcurrencyAwareBackend(
        full_backend=full,
        quarter_backends=quarters,
        role="totally_made_up_role",
    )
    assert cab._quarter_preference_order == [0, 1, 2]


def test_alias_topology_role_drives_preference_and_tap_metadata() -> None:
    """Logical aliases share the parent role's physical locks/topology."""
    instances = stack_numa.NUMA_CONFIG["frontdoor"]["instances"]
    full = _StubBackend(f"http://localhost:{instances[0][1]}")
    quarters = [_StubBackend(f"http://localhost:{inst[1]}") for inst in instances[1:]]
    cab = ca_mod.ConcurrencyAwareBackend(
        full_backend=full,
        quarter_backends=quarters,
        role="coder_escalation",
        full_port=instances[0][1],
        topology_role="frontdoor",
    )

    assert cab._quarter_preference_order == _make_concurrency_aware("frontdoor")._quarter_preference_order

    from src.runtime.instance_topology import get_instance_regions

    expected_regions = sorted(get_instance_regions().get(("frontdoor", 0), frozenset()))
    meta = cab._tap_dispatch_metadata(-1, full)
    assert meta["role"] == "coder_escalation"
    assert meta["topology_role"] == "frontdoor"
    assert meta["lock_role"] == "frontdoor"
    assert meta["instance_regions"] == expected_regions
    assert meta["instance_shape"] != "unknown"


def test_quarter_preference_disjoint_quarters_in_numerical_order() -> None:
    """Within the disjoint bucket, original numerical order is preserved.
    For frontdoor: disjoint = [q2, q3] — they should appear in that order."""
    cab = _make_concurrency_aware("frontdoor")
    pref = cab._quarter_preference_order
    # q2 should come before q3 (numerical order within disjoint bucket)
    disjoint_positions = [pref.index(2), pref.index(3)]
    assert disjoint_positions == sorted(disjoint_positions)


def test_quarter_preference_handles_no_quarters() -> None:
    """ConcurrencyAwareBackend requires ≥1 quarter; tests for edge cases."""
    full = _StubBackend()
    # Single quarter
    cab = ca_mod.ConcurrencyAwareBackend(
        full_backend=full,
        quarter_backends=[_StubBackend()],
        role="totally_made_up_role",
    )
    assert cab._quarter_preference_order == [0]
