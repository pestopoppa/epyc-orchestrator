"""W-38 (C backfill): `select_backfill_candidate` wired into
ConcurrencyAwareBackend._dispatch behind ORCHESTRATOR_CONTENTION_BACKFILL.

Flag off => the selector is never consulted (byte-identical to pre-W-38).
Flag on  => when the head is blocked a selected candidate may be admitted, but
the gate veto, the selector's own overlap/None verdict and the poll-budget
deadline (ContentionDenied) are all preserved.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

ca_mod = importlib.import_module("src.backends.concurrency_aware")
contention = importlib.import_module("src.scheduling.contention")
gate_mod = importlib.import_module("src.scheduling.contention_gate")
placement_mod = importlib.import_module("src.scheduling.placement")

_REGIONS = {
    ("frontdoor", 0): frozenset({"q0", "q1", "q2", "q3"}),
    ("frontdoor", 1): frozenset({"q0"}),
    ("frontdoor", 2): frozenset({"q1"}),
    ("frontdoor", 3): frozenset({"q2"}),
    ("frontdoor", 4): frozenset({"q3"}),
}


class _Stub:
    def __init__(self, url):
        self.config = type("C", (), {"base_url": url})()


class _FakeGate:
    def __init__(self, admit_topo):
        self.admit_topo = set(admit_topo)
        self.seen: list[int] = []

    def evaluate(self, role, traffic_class, candidate_topology_idx=None):
        self.seen.append(candidate_topology_idx)
        ok = candidate_topology_idx in self.admit_topo
        return SimpleNamespace(
            admitted=ok,
            decision=SimpleNamespace(value="allow" if ok else "queue"),
            waited_s=0.0,
            blocking_roles=[],
            reason="fake",
        )


def _setup(monkeypatch, *, backfill: bool, gate_admits, places_topo=(1,), held=None):
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    monkeypatch.setenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", "1")
    if backfill:
        monkeypatch.setenv("ORCHESTRATOR_CONTENTION_BACKFILL", "1")
    else:
        monkeypatch.delenv("ORCHESTRATOR_CONTENTION_BACKFILL", raising=False)
    backend = ca_mod.ConcurrencyAwareBackend(
        full_backend=_Stub("http://localhost:8070"),
        quarter_backends=[_Stub(f"http://localhost:80{80 + i * 100}") for i in range(4)],
        role="frontdoor",
        full_port=8070,
    )
    held_map = {} if held is None else held
    monkeypatch.setattr("src.runtime.instance_topology.get_instance_regions", lambda: dict(_REGIONS))
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", lambda *a, **k: {})
    monkeypatch.setattr("src.runtime.cpu_region_lock.held_regions_by_role", lambda *a, **k: held_map)
    acquired: list[int] = []

    @contextlib.contextmanager
    def fake_lock(role, instance_idx, timeout_s=None, deadline_s=None, **kw):
        acquired.append(instance_idx)
        yield [f"/tmp/mock.{instance_idx}"]

    monkeypatch.setattr("src.runtime.cpu_region_lock.cpu_region_lock_for_instance", fake_lock)
    # Head-of-queue: placement offers only `places_topo`; the gate decides.
    topo_to_internal = {t: (-1 if t == 0 else t - 1) for t in range(5)}

    def fake_eval(**kw):
        return placement_mod.PlacementResult(
            places=tuple(
                placement_mod.Place(internal_idx=topo_to_internal[t], topology_idx=t)
                for t in places_topo
            )
        )

    monkeypatch.setattr("src.scheduling.placement.evaluate_placement", fake_eval)
    gate = _FakeGate(gate_admits)
    monkeypatch.setattr(gate_mod, "get_gate", lambda: gate)
    return backend, gate, acquired


def _spy_selector(monkeypatch, result):
    calls: list = []

    def fake(candidates, active_holders, traffic_class=None, **kw):
        calls.append(list(candidates))
        return result

    monkeypatch.setattr(contention, "select_backfill_candidate", fake)
    return calls


_REQ = SimpleNamespace(request_priority="background", max_queue_wait_ms=300)


def test_flag_off_never_consults_selector_and_head_stays_blocked(monkeypatch):
    backend, gate, acquired = _setup(monkeypatch, backfill=False, gate_admits={3})
    calls = _spy_selector(monkeypatch, ("frontdoor", 3))
    with pytest.raises(gate_mod.ContentionDenied) as ei:
        with backend._dispatch(request=_REQ):
            pass
    assert calls == []
    assert acquired == []
    assert gate.seen and set(gate.seen) == {1}  # only the head candidate was ever tried
    assert "placement_gate_timeout" in str(ei.value)


def test_flag_default_is_off(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_CONTENTION_BACKFILL", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", "1")
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    assert ca_mod.ConcurrencyAwareBackend._contention_backfill_enabled() is False


def test_backfill_flag_requires_dual_shape_gate(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_CONTENTION_BACKFILL", "1")
    monkeypatch.delenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    assert ca_mod.ConcurrencyAwareBackend._contention_backfill_enabled() is False


def test_flag_on_admits_selected_backfill_when_head_gate_blocked(monkeypatch):
    backend, gate, acquired = _setup(monkeypatch, backfill=True, gate_admits={3})
    calls = _spy_selector(monkeypatch, ("frontdoor", 3))
    with backend._dispatch(request=_REQ) as (_b, idx, is_full):
        assert acquired == [3]
        assert idx == 2 and is_full is False  # topology 3 == internal quarter 2
    # head candidate (topo 1) is excluded from the backfill pool: gate veto not overridden
    assert calls and all(c[0] != "frontdoor" or c[1] != 1 for c in calls[0])
    assert (1 not in [t for _r, t in calls[0]])


def test_gate_veto_preserved_for_backfill_pick(monkeypatch):
    backend, gate, acquired = _setup(monkeypatch, backfill=True, gate_admits=set())
    _spy_selector(monkeypatch, ("frontdoor", 3))
    with pytest.raises(gate_mod.ContentionDenied):
        with backend._dispatch(request=_REQ):
            pass
    assert acquired == []
    assert 3 in gate.seen  # the pick was gate-evaluated, then vetoed


def test_selector_none_keeps_head_queued(monkeypatch):
    backend, gate, acquired = _setup(monkeypatch, backfill=True, gate_admits={3})
    calls = _spy_selector(monkeypatch, None)
    with pytest.raises(gate_mod.ContentionDenied) as ei:
        with backend._dispatch(request=_REQ):
            pass
    assert calls and acquired == []
    assert "placement_gate_timeout" in str(ei.value)  # poll-budget deadline semantics unchanged


def test_selector_exception_fails_closed(monkeypatch):
    backend, gate, acquired = _setup(monkeypatch, backfill=True, gate_admits={3})

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(contention, "select_backfill_candidate", boom)
    with pytest.raises(gate_mod.ContentionDenied):
        with backend._dispatch(request=_REQ):
            pass
    assert acquired == []


def test_real_selector_overlap_veto_all_held(monkeypatch):
    """Real select_backfill_candidate: every region held => nothing disjoint =>
    no backfill even with the flag on (overlap veto)."""
    held = {"architect_coding": frozenset({"q0", "q1", "q2", "q3"})}
    backend, gate, acquired = _setup(
        monkeypatch, backfill=True, gate_admits={1, 2, 3, 4}, places_topo=(), held=held
    )
    with pytest.raises(gate_mod.ContentionDenied):
        with backend._dispatch(request=_REQ):
            pass
    assert acquired == []
