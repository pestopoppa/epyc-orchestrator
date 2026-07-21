"""DISPATCH-A2: misaligned-`full:` demotion into the quarters pool.

When the endpoint wired as `full:` is NOT the topology idx-0 port (a quarter
impersonating the 96-core full — the live worker_general/frontdoor wiring), the
construction site demotes it into the quarters pool at its TRUE topology index
(port-resolved) instead of stranding it. This restores the N-way concurrency
ceiling AND makes every quarter's region lock match the server's physical cores
(killing the q0-lock-on-q1-cores cross-role collision hazard).

Uses the REAL worker_general topology (NUMA_CONFIG ports 8072/8082/8182/8282/8382)
so the idx→port→cpuset consistency is asserted against the live truth.
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
from src.backends.concurrency_aware import _get_base_url
from src.llm_primitives.backend import BackendMixin
from src.runtime.instance_topology import (
    cpu_list_to_regions,
    get_instance_regions,
    topology_idx_for_port,
)
from scripts.server.stack_numa import NUMA_CONFIG

WG = "worker_general"


class _Host(BackendMixin):
    """Minimal BackendMixin host to exercise _init_caching_backends directly."""

    def __init__(self) -> None:
        self._backends: dict = {}


def _build(server_urls: dict, num_slots: int = 1) -> dict:
    host = _Host()
    host._init_caching_backends(server_urls, num_slots)
    return host._backends


def _port(backend) -> int | None:
    url = _get_base_url(backend)
    return int(url.rsplit(":", 1)[-1]) if url and ":" in url else None


def _urls(full_port: int, quarter_ports: tuple[int, ...]) -> str:
    parts = [f"full:http://localhost:{full_port}"]
    parts += [f"http://localhost:{p}" for p in quarter_ports]
    return ",".join(parts)


class _LockCtx:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def __enter__(self):
        if not self.ok:
            from src.runtime.cpu_region_lock import CpuRegionLockTimeout

            raise CpuRegionLockTimeout("mock")
        return ["/tmp/cpu_region.mock.lock"]

    def __exit__(self, *exc):
        return False


# ── construction: misaligned full is DEMOTED, not stranded ───────────────────

def test_misaligned_full_demoted_into_quarters_pool() -> None:
    """Live shape: `full:` marks 8082 (a quarter) on a quarters-only stack. It is
    demoted → no full served, ALL four quarters dispatchable at true idxs."""
    backends = _build({WG: _urls(8082, (8182, 8282, 8382))})
    be = backends[WG]
    assert isinstance(be, ca_mod.ConcurrencyAwareBackend)
    assert be._full is None                       # misaligned full demoted → none served
    assert len(be._quarters) == 4                 # 4-way ceiling restored
    assert be._quarter_topology_idx == [1, 2, 3, 4]
    assert [_port(q) for q in be._quarters] == [8082, 8182, 8282, 8382]
    assert be._full_slot_aligned is True          # no full slot → vacuously aligned
    assert be.max_concurrency() >= 4 if hasattr(be, "max_concurrency") else True


def test_demoted_region_locks_match_physical_cores() -> None:
    """idx → port → cpuset consistency (the anti-shift invariant): the region the
    dispatcher LOCKS for each quarter equals the physical cpuset of the server at
    that port, per NUMA_CONFIG."""
    backends = _build({WG: _urls(8082, (8182, 8282, 8382))})
    be = backends[WG]
    ir = get_instance_regions()
    port_to_cpulist = {int(e[1]): e[0] for e in NUMA_CONFIG[WG]["instances"]}

    for i, q in enumerate(be._quarters):
        topo = be._quarter_topology_idx[i]
        port = _port(q)
        locked_regions = ir[(WG, topo)]                      # what the lock covers
        physical_regions = cpu_list_to_regions(port_to_cpulist[port])  # server's real cores
        assert locked_regions == physical_regions, (
            f"quarter {i} (port {port}) locks {sorted(locked_regions)} "
            f"but physically runs on {sorted(physical_regions)}"
        )
        assert topology_idx_for_port(WG, port) == topo


def test_demoted_endpoint_dispatchable_and_locks_true_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The demoted endpoint (8082 → topo idx 1 → region q0) is reachable, and the
    dispatcher locks the topology index matching the chosen backend's port."""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")
    backends = _build({WG: _urls(8082, (8182, 8282, 8382))})
    be = backends[WG]

    attempted: list[int] = []

    def _mock_lock(role, instance_idx, timeout_s=None, deadline_s=None):
        attempted.append(instance_idx)
        return _LockCtx(ok=True)

    monkeypatch.setattr("src.runtime.cpu_region_lock.cpu_region_lock_for_instance", _mock_lock)
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", lambda *a, **k: {})
    monkeypatch.setattr("src.runtime.cpu_region_lock.held_regions_by_role", lambda *a, **k: {})

    with be._dispatch(session_id="d0") as (_bk, idx, is_full):
        assert is_full is False
        assert 0 <= idx < 4
        # The locked topology idx is the chosen quarter's TRUE (port-resolved) idx,
        # never the all-region idx 0.
        assert attempted[-1] == be._quarter_topology_idx[idx]
        assert attempted[-1] in (1, 2, 3, 4)
        assert 0 not in attempted
    # The demoted 8082 endpoint occupies internal slot 0 at topology idx 1 (q0).
    assert be._quarter_topology_idx[0] == 1
    assert _port(be._quarters[0]) == 8082


# ── construction: a REAL full (port == idx-0) is preserved unchanged ──────────

def test_aligned_full_preserved() -> None:
    """When `full:` IS the topology idx-0 port (a real 96-core full deployed),
    the full slot is served exactly as before and quarters keep idxs 1..4."""
    backends = _build({WG: _urls(8072, (8082, 8182, 8282, 8382))})
    be = backends[WG]
    assert be._full is not None                    # real full served
    assert be._full_port == 8072
    assert be._full_slot_aligned is True
    assert len(be._quarters) == 4
    assert be._quarter_topology_idx == [1, 2, 3, 4]
    assert [_port(q) for q in be._quarters] == [8082, 8182, 8282, 8382]


def test_aligned_full_emits_full_candidate_on_solo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real full + a SOLO_PREFER_FULL role → solo dispatch routes to the full
    instance (idx 0), proving the explicit-topology-idx change did not disturb
    the aligned full path. (worker_general is FULL_DISABLED, so use frontdoor's
    aligned idx-0 port 8070 via direct construction.)"""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")

    class _Stub:
        def __init__(self, url):
            self.config = type("C", (), {"base_url": url})()

    full = _Stub("http://localhost:8070")
    quarters = [_Stub(f"http://localhost:80{80 + i * 100}") for i in range(4)]
    be = ca_mod.ConcurrencyAwareBackend(
        full_backend=full, quarter_backends=quarters,
        role="frontdoor", full_port=8070,  # aligned idx-0 for frontdoor
    )
    assert be._full_slot_aligned is True
    assert be._quarter_topology_idx == [1, 2, 3, 4]  # legacy positional default

    attempted: list[int] = []

    def _mock_lock(role, instance_idx, timeout_s=None, deadline_s=None):
        attempted.append(instance_idx)
        return _LockCtx(ok=True)

    _FRONTDOOR_REGIONS = {
        ("frontdoor", 0): frozenset({"q0", "q1"}),
        ("frontdoor", 1): frozenset({"q0"}),
        ("frontdoor", 2): frozenset({"q1"}),
        ("frontdoor", 3): frozenset({"q2"}),
        ("frontdoor", 4): frozenset({"q3"}),
    }
    monkeypatch.setattr("src.runtime.cpu_region_lock.cpu_region_lock_for_instance", _mock_lock)
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", lambda *a, **k: {})
    monkeypatch.setattr("src.runtime.cpu_region_lock.held_regions_by_role", lambda *a, **k: {})
    monkeypatch.setattr(
        "src.runtime.instance_topology.get_instance_regions",
        lambda: dict(_FRONTDOOR_REGIONS),
    )

    with be._dispatch(session_id="solo") as (_bk, idx, is_full):
        assert is_full is True     # full instance chosen on solo
        assert idx == -1
        assert attempted[-1] == 0  # topology idx 0 (the real full) locked
