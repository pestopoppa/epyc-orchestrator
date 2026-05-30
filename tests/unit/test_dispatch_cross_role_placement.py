"""Part A dispatch-level integration: ConcurrencyAwareBackend._dispatch with
ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT.

The pure `evaluate_placement` cross-role tests live in test_placement.py. This
file proves the behavior the FLAG actually ships: that `_dispatch` passes the
FULL active_region_holders() map (not just self-role) into evaluate_placement
when the flag is on, so a role lands on a quarter disjoint from ANOTHER role's
in-flight node-half — and that with the flag off, the cross-role holder is
invisible (legacy behaviour: full chosen first).

Mock seams mirror test_dispatch_placement_state_machine.py: _dispatch imports
active_region_holders / get_instance_regions / cpu_region_lock_for_instance
function-locally from their SOURCE modules, so patch them there.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

ca_mod = importlib.import_module("src.backends.concurrency_aware")


class _StubBackend:
    def __init__(self, url: str = "http://localhost:0"):
        self.config = type("C", (), {"base_url": url})()
        self.url = url


class _FakeLockCtx:
    """Mimics cpu_region_lock_for_instance's context-manager return."""

    def __init__(self, role: str, topo_idx: int, succeed: bool = True):
        self.role = role
        self.topo_idx = topo_idx
        self.succeed = succeed

    def __enter__(self):
        if not self.succeed:
            from src.runtime.cpu_region_lock import CpuRegionLockTimeout

            raise CpuRegionLockTimeout(f"mock timeout role={self.role} idx={self.topo_idx}")
        return [f"/tmp/cpu_region.{self.role}.mock-{self.topo_idx}.lock"]

    def __exit__(self, *exc):
        return False


# frontdoor: full(0)={q0,q1} (node0-half), quarters topo 1..4 = q0..q3.
# ingest_long_context full(0) = {q0,q1} — the cross-role holder to avoid.
_REGIONS = {
    ("frontdoor", 0): frozenset({"q0", "q1"}),
    ("frontdoor", 1): frozenset({"q0"}),
    ("frontdoor", 2): frozenset({"q1"}),
    ("frontdoor", 3): frozenset({"q2"}),
    ("frontdoor", 4): frozenset({"q3"}),
    ("ingest_long_context", 0): frozenset({"q0", "q1"}),
}


def _make_frontdoor_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")
    full = _StubBackend("http://localhost:8070")
    quarters = [_StubBackend(f"http://localhost:80{80 + i * 100}") for i in range(4)]
    return ca_mod.ConcurrencyAwareBackend(
        full_backend=full,
        quarter_backends=quarters,
        role="frontdoor",
        full_port=8070,
    )


def _wire(monkeypatch: pytest.MonkeyPatch, holders: dict, acquired: list[int]) -> None:
    monkeypatch.setattr(
        "src.runtime.instance_topology.get_instance_regions",
        lambda: dict(_REGIONS),
    )
    monkeypatch.setattr(
        "src.runtime.cpu_region_lock.active_region_holders",
        lambda *a, **k: dict(holders),
    )

    def _mock_lock(role, instance_idx, timeout_s=None, deadline_s=None):
        # All disjoint candidates acquire cleanly; we record which topo_idx the
        # dispatcher actually attempted/acquired.
        acquired.append(instance_idx)
        return _FakeLockCtx(role, instance_idx, succeed=True)

    monkeypatch.setattr(
        "src.runtime.cpu_region_lock.cpu_region_lock_for_instance", _mock_lock
    )


def test_cross_role_flag_on_avoids_other_role_held_node_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ingest_long_context holds {q0,q1}; with the flag ON, frontdoor must land
    on a node1 quarter (q2/q3 → topo 3/4), proving _dispatch passed the full
    holder map and honored the cross-role union (and the smallest-disjoint
    ordering selects a quarter, not full)."""
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    acquired: list[int] = []
    backend = _make_frontdoor_backend(monkeypatch)
    _wire(monkeypatch, {"ingest_long_context": [0]}, acquired)
    with backend._dispatch(session_id="s1") as (chosen_backend, idx, is_full):
        assert is_full is False
        assert acquired[-1] in (3, 4)  # q2 or q3 — never q0/q1/full(0)


def test_cross_role_flag_off_ignores_other_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same holders, flag OFF → cross-role holder invisible; frontdoor takes
    full (topo 0) as the first candidate. Proves the flag gates the behavior
    and the default path is unchanged."""
    monkeypatch.delenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", raising=False)
    acquired: list[int] = []
    backend = _make_frontdoor_backend(monkeypatch)
    _wire(monkeypatch, {"ingest_long_context": [0]}, acquired)
    with backend._dispatch(session_id="s1") as (chosen_backend, idx, is_full):
        assert is_full is True
        assert acquired[-1] == 0  # full — cross-role holder ignored
