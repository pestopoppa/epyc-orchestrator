"""Phase B — admission gate behavior + active_region_holders helper."""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

contention = importlib.import_module("src.scheduling.contention")
gate_mod = importlib.import_module("src.scheduling.contention_gate")
cpu_region_lock = importlib.import_module("src.runtime.cpu_region_lock")


@pytest.fixture
def real_matrix_path() -> Path:
    return ROOT / "orchestration" / "contention_matrix.yaml"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Each test gets a fresh ContentionGate singleton."""
    gate_mod.reset_gate()
    yield
    gate_mod.reset_gate()


# ── active_region_holders (helper factoring per handoff line 35) ───────


def test_active_region_holders_empty_when_no_topology() -> None:
    """No instance_regions → empty dict."""
    holders = cpu_region_lock.active_region_holders(instance_regions={})
    assert holders == {}


def test_active_region_holders_empty_when_no_locks(tmp_path, monkeypatch) -> None:
    """Topology present but no lock files exist → empty (nothing actively decoding)."""
    monkeypatch.setattr(cpu_region_lock, "_tmp_dir", lambda: tmp_path)
    instance_regions = {
        ("frontdoor", 0): frozenset({"q0", "q1"}),
        ("frontdoor", 1): frozenset({"q0"}),
        ("worker_general", 0): frozenset({"q0", "q1", "q2", "q3"}),
    }
    holders = cpu_region_lock.active_region_holders(instance_regions=instance_regions)
    assert holders == {}


def test_active_region_holders_returns_instances_with_held_regions(tmp_path, monkeypatch) -> None:
    """Create a held lock file → its (role, idx) shows up in the result."""
    monkeypatch.setattr(cpu_region_lock, "_tmp_dir", lambda: tmp_path)
    # Stub _current_lock_owner_pids: q0 of frontdoor is held; q1 isn't.
    def fake_owners(path):
        return ["12345"] if path.name == "cpu_region.frontdoor.q0.lock" else []
    monkeypatch.setattr(cpu_region_lock, "_current_lock_owner_pids", fake_owners)
    # Touch the lock files so .exists() passes
    (tmp_path / "cpu_region.frontdoor.q0.lock").touch()
    (tmp_path / "cpu_region.frontdoor.q1.lock").touch()
    (tmp_path / "cpu_region.worker_general.q0.lock").touch()

    instance_regions = {
        ("frontdoor", 0): frozenset({"q0", "q1"}),  # has q0 held → active
        ("frontdoor", 1): frozenset({"q1"}),         # only q1 → not active
        ("worker_general", 0): frozenset({"q0"}),    # different role, no held lock
    }
    holders = cpu_region_lock.active_region_holders(instance_regions=instance_regions)
    # Only frontdoor instance 0 is active (its q0 is held)
    assert holders == {"frontdoor": [0]}


def test_active_region_holders_groups_multiple_instances(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cpu_region_lock, "_tmp_dir", lambda: tmp_path)
    def fake_owners(path):
        return ["111"] if "q0" in path.name or "q2" in path.name else []
    monkeypatch.setattr(cpu_region_lock, "_current_lock_owner_pids", fake_owners)
    for region in ("q0", "q1", "q2", "q3"):
        (tmp_path / f"cpu_region.frontdoor.{region}.lock").touch()

    instance_regions = {
        ("frontdoor", 1): frozenset({"q0"}),  # held
        ("frontdoor", 2): frozenset({"q1"}),  # not held
        ("frontdoor", 3): frozenset({"q2"}),  # held
    }
    holders = cpu_region_lock.active_region_holders(instance_regions=instance_regions)
    assert holders == {"frontdoor": [1, 3]}  # sorted


# ── ContentionGate.evaluate ────────────────────────────────────────────


def _fake_active_factory(holders):
    return lambda: dict(holders)


def test_gate_admits_when_no_active_decodes(real_matrix_path) -> None:
    m = contention.load_contention_matrix(real_matrix_path)
    gate = gate_mod.ContentionGate(matrix=m, active_holders_fn=_fake_active_factory({}))
    d = gate.evaluate("frontdoor", contention.TrafficClass.FOREGROUND_INTERACTIVE)
    assert d.admitted
    assert d.decision == contention.PairDecision.ALLOW


def test_gate_blocks_background_when_pair_catastrophic(real_matrix_path) -> None:
    m = contention.load_contention_matrix(real_matrix_path)
    # ingest_long_context is actively decoding
    gate = gate_mod.ContentionGate(
        matrix=m, active_holders_fn=_fake_active_factory({"ingest_long_context": [0]})
    )
    # Background frontdoor request — should be QUEUE (ratio 0.37)
    d = gate.evaluate("frontdoor", contention.TrafficClass.BACKGROUND)
    assert not d.admitted
    assert d.decision == contention.PairDecision.QUEUE
    assert "ingest_long_context" in d.blocking_roles


def test_gate_blocks_foreground_when_pair_below_floor(real_matrix_path) -> None:
    """frontdoor+architect = 0.50 < 0.85 → foreground also queues (per handoff:
    'block or delay unless explicit low-latency override')."""
    m = contention.load_contention_matrix(real_matrix_path)
    gate = gate_mod.ContentionGate(
        matrix=m, active_holders_fn=_fake_active_factory({"architect_general": [0]})
    )
    d = gate.evaluate("frontdoor", contention.TrafficClass.FOREGROUND_INTERACTIVE)
    assert not d.admitted
    assert d.decision == contention.PairDecision.QUEUE


def test_gate_allows_known_good_pair(real_matrix_path) -> None:
    """frontdoor+worker_general = 1.28 → ALLOW even for background."""
    m = contention.load_contention_matrix(real_matrix_path)
    gate = gate_mod.ContentionGate(
        matrix=m, active_holders_fn=_fake_active_factory({"worker_general": [0]})
    )
    d = gate.evaluate("frontdoor", contention.TrafficClass.BACKGROUND)
    assert d.admitted
    assert d.decision == contention.PairDecision.ALLOW


def test_gate_allows_same_role_when_matrix_says_allow(real_matrix_path) -> None:
    """frontdoor is same-role-allow (4-quarters 1.88×). A second frontdoor
    request should be admitted even while frontdoor is decoding."""
    m = contention.load_contention_matrix(real_matrix_path)
    gate = gate_mod.ContentionGate(
        matrix=m, active_holders_fn=_fake_active_factory({"frontdoor": [0]})
    )
    d = gate.evaluate("frontdoor", contention.TrafficClass.FOREGROUND_INTERACTIVE)
    assert d.admitted


def test_gate_allows_vision_escalation_self_pair_on_certified_matrix(real_matrix_path) -> None:
    """vision_escalation same-role is ALLOW on the certified-affinity matrix. The earlier
    full+quarter "block" was a BAD-AFFINITY ARTIFACT (REFUTED 2026-05-26 — quarters were pinned
    to the wrong cores; on certified disjoint quarters the co-run allows). A second vision request
    is admitted in both traffic classes.
    NOTE: the block→QUEUE mechanism is no longer exercised by the real matrix (no measured block
    remains); cover that path with a synthetic-fixture matrix if regression protection is needed."""
    m = contention.load_contention_matrix(real_matrix_path)
    gate = gate_mod.ContentionGate(
        matrix=m, active_holders_fn=_fake_active_factory({"vision_escalation": [0]})
    )
    bg = gate.evaluate("vision_escalation", contention.TrafficClass.BACKGROUND)
    # ADMITTED is the invariant (no longer blocked/queued). The decision may be degraded_allow
    # rather than allow because the same_role entry still carries structural full+q2/full+q3
    # cpuset-overlap markers (vision's "full" half1 contains those quarters) — that is a
    # placement-overlap signal, not a contention block.
    assert bg.admitted
    fg = gate.evaluate("vision_escalation", contention.TrafficClass.FOREGROUND_INTERACTIVE)
    assert fg.admitted


def test_gate_picks_worst_pair_in_multi_active(real_matrix_path) -> None:
    """When multiple roles are active, gate picks the most-restrictive decision."""
    m = contention.load_contention_matrix(real_matrix_path)
    # worker_general is fine; architect is catastrophic
    gate = gate_mod.ContentionGate(
        matrix=m,
        active_holders_fn=_fake_active_factory({
            "worker_general": [0],
            "architect_general": [0],
        }),
    )
    d = gate.evaluate("frontdoor", contention.TrafficClass.BACKGROUND)
    assert not d.admitted
    assert "architect_general" in d.blocking_roles
    # worker_general is allow-pair so shouldn't appear as blocker
    assert "worker_general" not in d.blocking_roles


def test_gate_unknown_pair_blocks_background_allows_foreground(real_matrix_path) -> None:
    """Per handoff: unknown pair → background QUEUE, foreground ALLOW."""
    m = contention.load_contention_matrix(real_matrix_path)
    # worker_vision + ingest_long_context is in unknown_pairs
    gate = gate_mod.ContentionGate(
        matrix=m, active_holders_fn=_fake_active_factory({"worker_vision": [0]})
    )
    bg = gate.evaluate("ingest_long_context", contention.TrafficClass.BACKGROUND)
    assert not bg.admitted
    fg = gate.evaluate("ingest_long_context", contention.TrafficClass.FOREGROUND_INTERACTIVE)
    assert fg.admitted


# ── ContentionGate.admit (waiting behavior) ───────────────────────────


def test_admit_returns_immediately_when_ok(real_matrix_path) -> None:
    m = contention.load_contention_matrix(real_matrix_path)
    gate = gate_mod.ContentionGate(matrix=m, active_holders_fn=_fake_active_factory({}))
    t0 = time.monotonic()
    d = gate.admit("frontdoor", contention.TrafficClass.FOREGROUND_INTERACTIVE, max_queue_wait_ms=100)
    assert d.admitted
    assert (time.monotonic() - t0) < 0.5


def test_admit_times_out_when_persistently_blocked(real_matrix_path) -> None:
    m = contention.load_contention_matrix(real_matrix_path)
    gate = gate_mod.ContentionGate(
        matrix=m,
        active_holders_fn=_fake_active_factory({"architect_general": [0]}),
    )
    t0 = time.monotonic()
    d = gate.admit("frontdoor", contention.TrafficClass.BACKGROUND, max_queue_wait_ms=400)
    elapsed = time.monotonic() - t0
    assert not d.admitted
    assert "timeout" in d.reason
    assert 0.4 <= elapsed < 1.5  # respected the 400ms budget


def test_admit_unblocks_when_active_clears(real_matrix_path) -> None:
    m = contention.load_contention_matrix(real_matrix_path)
    # Mutable holders dict — flip to empty after 200 ms
    holders = {"architect_general": [0]}
    def get_holders():
        return dict(holders)
    gate = gate_mod.ContentionGate(matrix=m, active_holders_fn=get_holders)

    def clear_later():
        time.sleep(0.20)
        holders.clear()
    threading.Thread(target=clear_later, daemon=True).start()

    t0 = time.monotonic()
    d = gate.admit("frontdoor", contention.TrafficClass.FOREGROUND_INTERACTIVE, max_queue_wait_ms=2000)
    elapsed = time.monotonic() - t0
    assert d.admitted
    assert 0.15 <= elapsed < 1.0


# ── Singleton + metrics ────────────────────────────────────────────────


def test_get_gate_returns_singleton() -> None:
    g1 = gate_mod.get_gate()
    g2 = gate_mod.get_gate()
    assert g1 is g2


def test_metrics_snapshot_includes_required_keys(real_matrix_path) -> None:
    m = contention.load_contention_matrix(real_matrix_path)
    gate = gate_mod.ContentionGate(matrix=m, active_holders_fn=_fake_active_factory({}))
    snap = gate.metrics_snapshot()
    # Acceptance: handoff requires these counter keys
    for key in (
        "contention_blocked_count",
        "contention_wait_seconds",
        "contention_unknown_pair_count",
        "contention_admitted_count",
        "active_decodes_by_role",
        "matrix_status",
    ):
        assert key in snap


def test_metrics_record_blocked_count(real_matrix_path) -> None:
    m = contention.load_contention_matrix(real_matrix_path)
    gate = gate_mod.ContentionGate(
        matrix=m,
        active_holders_fn=_fake_active_factory({"architect_general": [0]}),
    )
    # Force one short-budget admit that times out → records blocked count
    gate.admit("frontdoor", contention.TrafficClass.BACKGROUND, max_queue_wait_ms=200)
    snap = gate.metrics_snapshot()
    # Pair key is sorted: ("architect_general", "frontdoor")
    assert any("architect_general" in k and "frontdoor" in k
               for k in snap["contention_blocked_count"].keys())
    assert snap["contention_timeout_count"] == 1
