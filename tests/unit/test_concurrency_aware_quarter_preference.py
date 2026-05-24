"""Phase D — topology-aware quarter preference in ConcurrencyAwareBackend.

Per the contention matrix (handoff line 88, frontdoor full + own q3 = 1.71×),
when the full instance is busy the scheduler should pick a quarter on the
opposite NUMA half BEFORE quarters that overlap full's cpu-set.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock


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


def test_vision_escalation_quarter_preference_full_on_NODE1() -> None:
    """vision_escalation full is on NUMA_NODE1 (48-95,144-191).
    Quarters: q0=Q0A (0-23), q1=Q0B (24-47), q2=Q1A (48-71), q3=Q1B (72-95).
    Disjoint from full: q0, q1. Overlapping: q2, q3.
    Preferred order should put q0 and q1 BEFORE q2 and q3."""
    cab = _make_concurrency_aware("vision_escalation")
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
