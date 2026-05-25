"""WP-2 integration: ConcurrencyAwareBackend._dispatch + placement state machine.

Validates that when `ORCHESTRATOR_PLACEMENT_STATE_MACHINE=1`:
  * Topology-safe candidates pass through and acquire region locks.
  * When every candidate overlaps in-flight holders, the dispatcher
    polls + re-evaluates instead of falling back to blocking-on-full.
  * On poll timeout, `ContentionDenied` is raised (not `CpuRegionLockTimeout`).

We mock `cpu_region_lock_for_instance` and `active_region_holders` to control
the placement scenario without launching real llama-servers.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from unittest import mock

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
        self.entered = False

    def __enter__(self):
        if not self.succeed:
            from src.runtime.cpu_region_lock import CpuRegionLockTimeout
            raise CpuRegionLockTimeout(f"mock timeout role={self.role} idx={self.topo_idx}")
        self.entered = True
        return [f"/tmp/cpu_region.{self.role}.mock-{self.topo_idx}.lock"]

    def __exit__(self, *exc):
        self.entered = False
        return False


def _make_backend(monkeypatch: pytest.MonkeyPatch, role: str = "frontdoor"):
    """Build a ConcurrencyAwareBackend with 4 quarters wired to stubs."""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")
    full = _StubBackend("http://localhost:8070")
    quarters = [_StubBackend(f"http://localhost:80{80 + i * 100}") for i in range(4)]
    return ca_mod.ConcurrencyAwareBackend(
        full_backend=full,
        quarter_backends=quarters,
        role=role,
        full_port=8070,
    )


# Synthetic frontdoor topology — matches the production NUMA_NODE0 shape.
_FRONTDOOR_REGIONS = {
    ("frontdoor", 0): frozenset({"q0", "q1"}),
    ("frontdoor", 1): frozenset({"q0"}),
    ("frontdoor", 2): frozenset({"q1"}),
    ("frontdoor", 3): frozenset({"q2"}),
    ("frontdoor", 4): frozenset({"q3"}),
}


def test_dispatcher_uses_topology_safe_candidate_when_full_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full is in-flight (holder=[0]); dispatcher should skip q0/q1 (overlap)
    and acquire one of q2/q3 (disjoint)."""
    backend = _make_backend(monkeypatch)
    acquired_idxs: list[int] = []

    def _mock_lock(role, instance_idx, timeout_s=None, deadline_s=None):
        # full(0) is "held" — its lock attempt times out. Quarters succeed.
        succeed = (instance_idx != 0)
        if succeed:
            acquired_idxs.append(instance_idx)
        return _FakeLockCtx(role, instance_idx, succeed=succeed)

    monkeypatch.setattr("src.runtime.cpu_region_lock.cpu_region_lock_for_instance", _mock_lock)
    monkeypatch.setattr(
        "src.runtime.cpu_region_lock.active_region_holders",
        lambda: {"frontdoor": [0]},
    )
    monkeypatch.setattr(
        "src.runtime.instance_topology.get_instance_regions",
        lambda: _FRONTDOOR_REGIONS,
    )

    with backend._dispatch(session_id="s1") as (chosen_backend, idx, is_full):
        assert is_full is False
        # The chosen quarter should be from the disjoint set (topo 3 or 4 — q2/q3).
        assert acquired_idxs[-1] in (3, 4)
        assert idx in (2, 3)  # internal idx for q2/q3


def test_dispatcher_queues_when_all_candidates_overlap_then_succeeds_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All regions in use → first eval queues → re-poll after holders shrink → succeeds."""
    backend = _make_backend(monkeypatch)

    # First call: all 4 atomic regions covered by holders ⇒ every candidate overlaps.
    # After 1 poll: holders drop (q3 released) so q3 becomes safe.
    holder_state = {"calls": 0, "current": {"frontdoor": [0, 3, 4]}}

    def _holders():
        holder_state["calls"] += 1
        if holder_state["calls"] >= 2:
            holder_state["current"] = {"frontdoor": [0, 3]}  # q2 released → q3 still has q2 free? no: drop q3 idx
            holder_state["current"] = {"frontdoor": [0, 3]}  # q2's topo idx is 3; q3's topo idx is 4 — release q3 → topo 4 gone
            return {"frontdoor": [0, 3]}  # q3 (topo 4) released → safe
        return holder_state["current"]

    def _mock_lock(role, instance_idx, timeout_s=None, deadline_s=None):
        # full(0) and q0(1),q1(2) always overlap holders; q2(3) overlaps holder q2 (topo 3);
        # q3(4) becomes acquirable on second eval.
        return _FakeLockCtx(role, instance_idx, succeed=(instance_idx == 4))

    monkeypatch.setattr("src.runtime.cpu_region_lock.cpu_region_lock_for_instance", _mock_lock)
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", _holders)
    monkeypatch.setattr(
        "src.runtime.instance_topology.get_instance_regions",
        lambda: _FRONTDOOR_REGIONS,
    )

    start = time.perf_counter()
    with backend._dispatch(session_id="s2") as (chosen_backend, idx, is_full):
        elapsed = time.perf_counter() - start
        assert is_full is False
        assert idx == 3  # q3 internal idx
        # Must have polled at least once (each poll = 150ms; we expect 1-2 polls).
        assert elapsed >= 0.150
        assert holder_state["calls"] >= 2


def test_dispatcher_raises_contention_denied_on_poll_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no safe candidate ever appears, the poll loop times out at 60s.
    To keep the test fast, we patch the deadline to elapse immediately."""
    backend = _make_backend(monkeypatch)

    monkeypatch.setattr(
        "src.runtime.cpu_region_lock.active_region_holders",
        lambda: {"frontdoor": [0, 1, 2, 3, 4]},  # all instances "held"
    )
    monkeypatch.setattr(
        "src.runtime.instance_topology.get_instance_regions",
        lambda: _FRONTDOOR_REGIONS,
    )

    # Make the poll deadline elapse immediately by patching time.perf_counter.
    times = iter([0.0, 0.0, 100.0])  # first two calls return 0, third returns 100s
    monkeypatch.setattr("src.backends.concurrency_aware.time.perf_counter", lambda: next(times))
    monkeypatch.setattr("src.backends.concurrency_aware.time.sleep", lambda _x: None)

    from src.scheduling.contention_gate import ContentionDenied
    with pytest.raises(ContentionDenied) as exc_info:
        with backend._dispatch(session_id="s3"):
            pass
    assert "placement timeout" in str(exc_info.value)
    assert "frontdoor" in str(exc_info.value)


def test_dispatcher_legacy_path_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ORCHESTRATOR_PLACEMENT_STATE_MACHINE is not set, the legacy
    greedy try-loop + blocking-on-full fallback runs unchanged."""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.delenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", raising=False)

    full = _StubBackend("http://localhost:8070")
    quarters = [_StubBackend(f"http://localhost:80{80 + i * 100}") for i in range(4)]
    backend = ca_mod.ConcurrencyAwareBackend(
        full_backend=full, quarter_backends=quarters, role="frontdoor", full_port=8070,
    )

    # In legacy path, full's try-acquire succeeds → no quarter attempt needed.
    monkeypatch.setattr(
        "src.runtime.cpu_region_lock.cpu_region_lock_for_instance",
        lambda role, idx, **kw: _FakeLockCtx(role, idx, succeed=(idx == 0)),
    )

    with backend._dispatch(session_id="s_legacy") as (chosen_backend, idx, is_full):
        assert is_full is True
        assert idx == -1
