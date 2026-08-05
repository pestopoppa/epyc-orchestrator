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


# ----------------------------------------------------------------------------
# Topology-derived buckets.
#
# The RULE under test is fixed: quarters whose cores are disjoint from the full
# instance's cores are preferred, and within a bucket numerical order is
# preserved. WHICH quarters land in which bucket is a property of the machine,
# and the machine changed (2026-07-30: quarters retired, every role is now one
# 0-95 full plus halves that necessarily overlap it). Restating quarter indices
# 0..3 pinned the old machine, not the rule — so the buckets are computed here
# from the same NUMA_CONFIG cpu-lists the scheduler reads.
# ----------------------------------------------------------------------------


def _quarter_buckets(role: str) -> tuple[list[int], list[int]]:
    """(disjoint, overlapping) quarter indices for ``role``, per its cpu-sets."""
    from src.runtime.instance_topology import parse_cpu_list

    instances = stack_numa.NUMA_CONFIG[role]["instances"]
    full_cores = parse_cpu_list(instances[0][0])
    disjoint: list[int] = []
    overlapping: list[int] = []
    for q_idx, inst in enumerate(instances[1:]):
        if full_cores & parse_cpu_list(inst[0]):
            overlapping.append(q_idx)
        else:
            disjoint.append(q_idx)
    return disjoint, overlapping


def _assert_preference_follows_topology(role: str, pref: list[int]) -> None:
    """Every disjoint quarter precedes every overlapping one; order kept inside."""
    disjoint, overlapping = _quarter_buckets(role)
    assert sorted(pref) == list(range(len(disjoint) + len(overlapping)))
    assert pref == disjoint + overlapping
    for d in disjoint:
        for o in overlapping:
            assert pref.index(d) < pref.index(o), (
                f"{role}: quarter {d} is cpu-disjoint from full but was ordered "
                f"after overlapping quarter {o}"
            )
    # Within-bucket numerical order is preserved.
    assert [i for i in pref if i in disjoint] == sorted(disjoint)
    assert [i for i in pref if i in overlapping] == sorted(overlapping)


_SYNTHETIC_NODE1_ROLE = "synthetic_node1_full"


def _register_synthetic_node1_role(monkeypatch) -> str:
    """A NODE1-full role with four quarters: the fixture that can still express a
    NON-EMPTY disjoint bucket now that every real role's full spans 0-95.

    Historically this shape was vision_escalation (full on NUMA_NODE1 + 4
    quarters), retired in the v7 one-instance vision layout (2026-07-20 recert).
    """
    synthetic = {
        "instances": [
            ("48-95", 9070),
            ("0-23", 9071),
            ("24-47", 9072),
            ("48-71", 9073),
            ("72-95", 9074),
        ],
        "full_instance_idx": 0,
        "cpu_lists": None,
    }
    monkeypatch.setitem(stack_numa.NUMA_CONFIG, _SYNTHETIC_NODE1_ROLE, synthetic)
    return _SYNTHETIC_NODE1_ROLE


def test_frontdoor_quarter_preference_prefers_disjoint_NUMA() -> None:
    """frontdoor's preference order is exactly what its cpu-sets imply.

    Contention matrix (handoff line 88, frontdoor full + own q3 = 1.71x): a
    quarter that shares no cores with the in-flight full request is preferred.
    Under the current topology frontdoor's full spans 0-95, so both halves
    overlap it and the disjoint bucket is empty — the ordering is numerical.
    The NON-empty-bucket case is enforced by
    ``test_quarter_preference_full_on_NODE1_synthetic`` and by
    ``test_quarter_preference_disjoint_quarters_in_numerical_order`` below.
    """
    cab = _make_concurrency_aware("frontdoor")
    _assert_preference_follows_topology("frontdoor", cab._quarter_preference_order)


def test_ingest_quarter_preference_matches_frontdoor_pattern() -> None:
    """ingest_long_context shares frontdoor's instance shape — same rule, same
    derivation, so the two roles must agree whenever their cpu-sets do."""
    cab = _make_concurrency_aware("ingest_long_context")
    _assert_preference_follows_topology(
        "ingest_long_context", cab._quarter_preference_order
    )
    assert _quarter_buckets("ingest_long_context") == _quarter_buckets("frontdoor")
    assert (
        cab._quarter_preference_order
        == _make_concurrency_aware("frontdoor")._quarter_preference_order
    )


def test_quarter_preference_full_on_NODE1_synthetic(monkeypatch) -> None:
    """NODE1-full disjointness ordering, exercised via a SYNTHETIC role.

    Historically this used vision_escalation (full on NUMA_NODE1 + 4 quarters),
    but that role became single-instance in the v7 one-instance vision layout
    (2026-07-20 recert) and no real NODE1-full multi-instance role remains. The
    ORDERING LOGIC is topology-generic and stays covered here: full on NODE1
    (48-95); quarters q0 (0-23) and q1 (24-47) are disjoint from full and must
    order BEFORE overlapping q2 (48-71) and q3 (72-95)."""
    role = _register_synthetic_node1_role(monkeypatch)
    disjoint, overlapping = _quarter_buckets(role)
    # This fixture exists to keep the rule under ACTIVE enforcement: both
    # buckets must be non-empty or the ordering assertion is vacuous.
    assert disjoint == [0, 1]
    assert overlapping == [2, 3]
    cab = _make_concurrency_aware(role)
    pref = cab._quarter_preference_order
    _assert_preference_follows_topology(role, pref)
    assert pref.index(0) < pref.index(2)
    assert pref.index(0) < pref.index(3)
    assert pref.index(1) < pref.index(2)
    assert pref.index(1) < pref.index(3)


def test_worker_general_quarter_preference_full_on_FULL_SOCKET() -> None:
    """worker_general's full spans the whole 0-95 socket, so EVERY sibling
    instance overlaps it: no quarter is disjoint and the preference is plain
    numerical order over the instances the topology actually declares."""
    disjoint, overlapping = _quarter_buckets("worker_general")
    assert disjoint == [], (
        "worker_general's full no longer covers every sibling — this test's "
        "premise (all-overlap) is gone; re-derive it from the new topology."
    )
    cab = _make_concurrency_aware("worker_general")
    pref = cab._quarter_preference_order
    assert pref == overlapping == list(range(len(overlapping)))
    _assert_preference_follows_topology("worker_general", pref)


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


def test_quarter_preference_disjoint_quarters_in_numerical_order(monkeypatch) -> None:
    """Within the disjoint bucket, original numerical order is preserved.

    Carried by the synthetic NODE1-full role, the only fixture whose topology
    can still express a non-empty disjoint bucket (every real role's full spans
    0-95 since the 2026-07-30 quarters retirement). The real roles are checked
    too, so the property re-arms automatically if a disjoint-full topology
    returns."""
    role = _register_synthetic_node1_role(monkeypatch)
    disjoint, _ = _quarter_buckets(role)
    assert len(disjoint) >= 2, "fixture can no longer exercise within-bucket order"
    pref = _make_concurrency_aware(role)._quarter_preference_order
    positions = [pref.index(q) for q in disjoint]
    assert positions == sorted(positions)
    assert [q for q in pref if q in disjoint] == sorted(disjoint)

    for real_role in ("frontdoor", "ingest_long_context", "worker_general"):
        real_disjoint, _ = _quarter_buckets(real_role)
        real_pref = _make_concurrency_aware(real_role)._quarter_preference_order
        assert [q for q in real_pref if q in real_disjoint] == sorted(real_disjoint)


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
