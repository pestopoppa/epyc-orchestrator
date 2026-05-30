"""B call-site wiring: ContentionGate.evaluate consults seam_admit ONLY when a
candidate_topology_idx is supplied AND both dual flags are on. When consulted
and it returns a non-None verdict, that placement-aware verdict is AUTHORITATIVE
— it REPLACES the legacy role-keyed result (audit #1/#2), in BOTH directions: it
can admit a disjoint placement the stale role-keyed pair layer would falsely
QUEUE, and can queue an overlap the pair layer would allow. (An earlier draft
wired this tightening-only, which could never unlock B's target false-negative;
that was corrected and the bug-enshrining test removed.)

This tests the GATE'S INTEGRATION (when/how it calls the seam + combines), not
seam_admit's internals (covered by test_shape_aware_seam.py). seam_admit is
monkeypatched to a known verdict so the wiring logic is isolated, plus two
end-to-end cases use the real seam_admit with monkeypatched region helpers.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

contention = importlib.import_module("src.scheduling.contention")
gate_mod = importlib.import_module("src.scheduling.contention_gate")

PairDecision = contention.PairDecision
TrafficClass = contention.TrafficClass


@pytest.fixture(autouse=True)
def reset_singleton():
    gate_mod.reset_gate()
    yield
    gate_mod.reset_gate()


@pytest.fixture
def real_matrix():
    return contention.load_contention_matrix(ROOT / "orchestration" / "contention_matrix.yaml")


def _gate(matrix, holders):
    return gate_mod.ContentionGate(matrix=matrix, active_holders_fn=lambda: dict(holders))


@pytest.fixture
def dual_flags_on(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", "1")
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")


# A scenario where the LEGACY verdict is ALLOW: frontdoor + worker_general is
# allow pairwise AND n_way in the real matrix. We then have the seam return a
# tighter verdict and assert the gate honors it only under the right conditions.
_LEGACY_ALLOW_HOLDERS = {"worker_general": [1]}


def test_gate_ignores_seam_when_no_candidate_idx(real_matrix, dual_flags_on, monkeypatch) -> None:
    """Flags ON but candidate_topology_idx=None → seam NOT consulted → legacy."""
    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return PairDecision.QUEUE

    monkeypatch.setattr(gate_mod, "seam_admit", _spy)
    gate = _gate(real_matrix, _LEGACY_ALLOW_HOLDERS)
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND)  # no idx
    assert called["n"] == 0
    assert d.admitted and d.decision == PairDecision.ALLOW


def test_gate_ignores_seam_when_flags_off(real_matrix, monkeypatch) -> None:
    """idx supplied but flags OFF → seam returns None (disabled) → legacy.
    Also assert the gate respects shape_aware_contention_enabled()=False by
    not letting a (hypothetical) seam verdict through."""
    monkeypatch.delenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", raising=False)
    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return PairDecision.QUEUE

    monkeypatch.setattr(gate_mod, "seam_admit", _spy)
    gate = _gate(real_matrix, _LEGACY_ALLOW_HOLDERS)
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND, candidate_topology_idx=3)
    # gate guards on shape_aware_contention_enabled() before calling seam → not called
    assert called["n"] == 0
    assert d.admitted and d.decision == PairDecision.ALLOW


def test_gate_applies_seam_tightening(real_matrix, dual_flags_on, monkeypatch) -> None:
    """Flags ON + idx supplied + seam returns QUEUE where legacy ALLOWs →
    gate returns the TIGHTENED verdict (QUEUE, not admitted)."""
    monkeypatch.setattr(gate_mod, "seam_admit", lambda *a, **k: PairDecision.QUEUE)
    gate = _gate(real_matrix, _LEGACY_ALLOW_HOLDERS)
    # sanity: legacy alone admits this
    legacy = _gate(real_matrix, _LEGACY_ALLOW_HOLDERS).evaluate(
        "frontdoor", TrafficClass.BACKGROUND
    )
    assert legacy.admitted
    # with seam consulted → tightened to QUEUE
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND, candidate_topology_idx=3)
    assert not d.admitted
    assert d.decision == PairDecision.QUEUE


def test_gate_seam_authoritative_overrides_legacy_queue(real_matrix, dual_flags_on, monkeypatch) -> None:
    """THE CORE OF B (audit-corrected): legacy role-keyed pair_policy QUEUEs
    frontdoor+ingest (stale 0.37 primary), but the placement-aware seam, given
    a DISJOINT candidate, returns ALLOW. When both flags + idx are supplied the
    seam is AUTHORITATIVE — the gate must ADMIT, overriding the stale legacy
    QUEUE. (This is the false-negative B exists to fix; tightening-only could
    never do it.)"""
    monkeypatch.setattr(gate_mod, "seam_admit", lambda *a, **k: PairDecision.ALLOW)
    # frontdoor + ingest_long_context background = 0.37 → legacy QUEUE
    gate = _gate(real_matrix, {"ingest_long_context": [0]})
    # sanity: legacy alone (no idx) QUEUEs this
    legacy = _gate(real_matrix, {"ingest_long_context": [0]}).evaluate(
        "frontdoor", TrafficClass.BACKGROUND
    )
    assert not legacy.admitted and legacy.decision == PairDecision.QUEUE
    # with the placement-aware seam authoritative → ADMITTED
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND, candidate_topology_idx=3)
    assert d.admitted
    assert d.decision == PairDecision.ALLOW


def test_gate_seam_authoritative_can_also_tighten(real_matrix, dual_flags_on, monkeypatch) -> None:
    """Authoritative both ways: where legacy ALLOWs but the seam (physical
    overlap / measured block) returns QUEUE, the gate QUEUEs. The seam replaces
    the legacy verdict in both directions — it is the trusted placement-aware
    source, fail-closed inside seam_admit."""
    monkeypatch.setattr(gate_mod, "seam_admit", lambda *a, **k: PairDecision.QUEUE)
    gate = _gate(real_matrix, _LEGACY_ALLOW_HOLDERS)
    legacy = _gate(real_matrix, _LEGACY_ALLOW_HOLDERS).evaluate(
        "frontdoor", TrafficClass.BACKGROUND
    )
    assert legacy.admitted
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND, candidate_topology_idx=3)
    assert not d.admitted
    assert d.decision == PairDecision.QUEUE


def test_gate_seam_none_keeps_legacy(real_matrix, dual_flags_on, monkeypatch) -> None:
    """seam_admit returning None (e.g. unknown placement foreground) → gate
    keeps the legacy verdict unchanged."""
    monkeypatch.setattr(gate_mod, "seam_admit", lambda *a, **k: None)
    gate = _gate(real_matrix, _LEGACY_ALLOW_HOLDERS)
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND, candidate_topology_idx=3)
    assert d.admitted and d.decision == PairDecision.ALLOW


def test_gate_seam_exception_falls_back_to_legacy(real_matrix, dual_flags_on, monkeypatch) -> None:
    """If seam_admit raises, the gate must not crash — it logs and uses the
    legacy verdict."""
    def _boom(*a, **k):
        raise RuntimeError("seam blew up")

    monkeypatch.setattr(gate_mod, "seam_admit", _boom)
    gate = _gate(real_matrix, _LEGACY_ALLOW_HOLDERS)
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND, candidate_topology_idx=3)
    assert d.admitted and d.decision == PairDecision.ALLOW


def test_gate_real_seam_unlocks_disjoint_placement(real_matrix, dual_flags_on, monkeypatch) -> None:
    """END-TO-END with the REAL seam_admit (not mocked): legacy QUEUEs
    frontdoor+ingest (0.37 stale primary), but frontdoor's candidate q2 is
    physically disjoint from ingest's held node0-half {q0,q1}, and the matrix's
    n_way frontdoor+ingest is 1.716 ALLOW. The wired gate must ADMIT.

    Hermetic: monkeypatch the runtime region helpers seam_admit consults so the
    test doesn't depend on a live NUMA_CONFIG, but the genuine seam_admit logic
    runs (overlap test + nway delegation)."""
    import src.runtime.cpu_region_lock as crl
    import src.runtime.instance_topology as it

    regions = {
        ("frontdoor", 0): frozenset({"q0", "q1"}),
        ("frontdoor", 3): frozenset({"q2"}),   # candidate idx 3 → q2 (disjoint)
        ("ingest_long_context", 0): frozenset({"q0", "q1"}),
    }
    # ingest holds its node0-half {q0,q1}; exact held-region view.
    monkeypatch.setattr(crl, "held_regions_by_role",
                        lambda instance_regions=None: {"ingest_long_context": frozenset({"q0", "q1"})})
    monkeypatch.setattr(it, "get_instance_regions", lambda: dict(regions))

    gate = _gate(real_matrix, {"ingest_long_context": [0]})
    # legacy alone → QUEUE (stale pair 0.37)
    legacy = _gate(real_matrix, {"ingest_long_context": [0]}).evaluate(
        "frontdoor", TrafficClass.BACKGROUND
    )
    assert not legacy.admitted and legacy.decision == PairDecision.QUEUE
    # real seam, disjoint candidate q2 → nway 1.716 allow → gate ADMITS
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND, candidate_topology_idx=3)
    assert d.admitted, f"expected admit via disjoint placement, got {d.decision}"
    assert d.decision == PairDecision.ALLOW


def test_gate_real_seam_overlap_still_queues(real_matrix, dual_flags_on, monkeypatch) -> None:
    """Real seam, but candidate OVERLAPS the held region → seam QUEUEs (physical
    conflict, fail-closed) → gate QUEUEs. Confirms the override stays safe."""
    import src.runtime.cpu_region_lock as crl
    import src.runtime.instance_topology as it

    regions = {
        ("frontdoor", 1): frozenset({"q0"}),   # candidate idx 1 → q0 (OVERLAPS)
        ("ingest_long_context", 0): frozenset({"q0", "q1"}),
    }
    monkeypatch.setattr(crl, "held_regions_by_role",
                        lambda instance_regions=None: {"ingest_long_context": frozenset({"q0", "q1"})})
    monkeypatch.setattr(it, "get_instance_regions", lambda: dict(regions))

    gate = _gate(real_matrix, {"ingest_long_context": [0]})
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND, candidate_topology_idx=1)
    assert not d.admitted
    assert d.decision == PairDecision.QUEUE


def test_gate_no_active_decodes_short_circuits_before_seam(real_matrix, dual_flags_on, monkeypatch) -> None:
    """No holders → gate returns ALLOW early (before the seam block), seam not
    consulted even with flags+idx."""
    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return PairDecision.QUEUE

    monkeypatch.setattr(gate_mod, "seam_admit", _spy)
    gate = _gate(real_matrix, {})  # no active decodes
    d = gate.evaluate("frontdoor", TrafficClass.BACKGROUND, candidate_topology_idx=3)
    assert called["n"] == 0
    assert d.admitted and d.decision == PairDecision.ALLOW
