"""Phase A — contention matrix loader + policy decisions."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

contention = importlib.import_module("src.scheduling.contention")


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def real_matrix_path() -> Path:
    """The actual committed contention matrix."""
    return ROOT / "orchestration" / "contention_matrix.yaml"


@pytest.fixture
def minimal_matrix_yaml(tmp_path: Path) -> Path:
    """Tiny synthetic matrix for isolated testing."""
    p = tmp_path / "contention_matrix.yaml"
    p.write_text(
        """
version: 1
measured_at: "2026-05-24T12:00:00Z"
host: "test"
topology_hash: "TEST_HASH"
default_floor: 0.85
pairs:
  - roles: ["frontdoor", "architect_general"]
    instance_a: {port: 8070}
    instance_b: {port: 8083}
    seq_aggregate_tps: 17.0
    parallel_aggregate_tps: 8.5
    ratio: 0.50
    verdict: "block"
  - roles: ["frontdoor", "worker_general"]
    instance_a: {port: 8070}
    instance_b: {port: 8072}
    seq_aggregate_tps: 35.0
    parallel_aggregate_tps: 45.0
    ratio: 1.28
    verdict: "allow"
  - roles: ["frontdoor", "vision_escalation"]
    instance_a: {port: 8070}
    instance_b: {port: 8087}
    seq_aggregate_tps: 26.6
    parallel_aggregate_tps: 22.3
    ratio: 0.84
    verdict: "borderline"
same_role:
  - role: "frontdoor"
    verdict: "allow"
  - role: "vision_escalation"
    verdict: "block"
    note: "4-quarter anomaly"
unknown_pairs:
  - roles: ["ingest_long_context", "worker_vision"]
    reason: "not_measured"
"""
    )
    return p


# ── Loader ───────────────────────────────────────────────────────────


def test_load_real_matrix(real_matrix_path: Path) -> None:
    """The committed matrix should parse cleanly."""
    m = contention.load_contention_matrix(real_matrix_path)
    assert m.version == 1
    assert m.host == "Beelzebub"
    assert m.default_floor == 0.85
    # v6 full/primary refresh measured every cross-role pair in this layer.
    assert len(m.pairs) == 15
    assert len(m.unknown_pairs) == 0
    assert m.get_pair("architect_general", "worker_vision") is not None
    assert m.get_pair("ingest_long_context", "worker_vision") is not None


def test_real_matrix_declares_current_nway_role_classes(real_matrix_path: Path) -> None:
    """The committed matrix should expose current light/heavy role classes."""
    m = contention.load_contention_matrix(real_matrix_path)

    assert {
        "frontdoor",
        "vision_escalation",
        "worker_general",
        "worker_vision",
    } <= m.light_roles
    assert {"ingest_long_context", "architect_general"} <= m.heavy_roles
    assert "worker_fast" not in m.light_roles


def test_real_matrix_vision_escalation_same_role_allows(real_matrix_path: Path) -> None:
    """Certified matrix no longer treats same-role vision_escalation as blocked."""
    m = contention.load_contention_matrix(real_matrix_path)

    same_role = m.get_same_role("vision_escalation")

    assert same_role is not None
    assert same_role.verdict == "allow"
    assert (
        contention.pair_policy(
            "vision_escalation",
            "vision_escalation",
            contention.TrafficClass.BACKGROUND,
            matrix=m,
        )
        == contention.PairDecision.ALLOW
    )


def test_load_minimal_matrix(minimal_matrix_yaml: Path) -> None:
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert m.topology_hash == "TEST_HASH"
    assert len(m.pairs) == 3
    assert len(m.same_role) == 2


def test_pair_key_is_sorted(minimal_matrix_yaml: Path) -> None:
    """(A, B) and (B, A) should resolve to the same pair."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    p1 = m.get_pair("frontdoor", "architect_general")
    p2 = m.get_pair("architect_general", "frontdoor")
    assert p1 is not None and p2 is not None
    assert p1.ratio == p2.ratio == 0.50


def test_unknown_pair_detection(minimal_matrix_yaml: Path) -> None:
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert m.is_unknown_pair("ingest_long_context", "worker_vision")
    assert m.is_unknown_pair("worker_vision", "ingest_long_context")  # order-agnostic
    assert not m.is_unknown_pair("frontdoor", "architect_general")  # known


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        contention.load_contention_matrix(Path("/tmp/does_not_exist_xyz.yaml"))


# ── matrix_status ────────────────────────────────────────────────────


def test_matrix_status_missing(tmp_path: Path) -> None:
    assert contention.matrix_status(tmp_path / "missing.yaml") == contention.MatrixStatus.MISSING


def test_matrix_status_ok(minimal_matrix_yaml: Path) -> None:
    status = contention.matrix_status(minimal_matrix_yaml, current_topology_hash="TEST_HASH")
    assert status == contention.MatrixStatus.OK


def test_matrix_status_stale_topology(minimal_matrix_yaml: Path) -> None:
    status = contention.matrix_status(minimal_matrix_yaml, current_topology_hash="DIFFERENT_HASH")
    assert status == contention.MatrixStatus.STALE


def test_matrix_status_stale_age(minimal_matrix_yaml: Path) -> None:
    # File just-created; with max_age_days=-1 anything is stale
    status = contention.matrix_status(minimal_matrix_yaml, current_topology_hash="TEST_HASH", max_age_days=-1)
    assert status == contention.MatrixStatus.STALE


def test_matrix_status_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [a, valid\n  yaml mapping at root")
    assert contention.matrix_status(bad) == contention.MatrixStatus.INVALID


# ── pair_policy ──────────────────────────────────────────────────────


def test_policy_block_below_floor(minimal_matrix_yaml: Path) -> None:
    """frontdoor + architect at 0.50 → QUEUE for all classes."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    for tc in [
        contention.TrafficClass.FOREGROUND_INTERACTIVE,
        contention.TrafficClass.FOREGROUND_SPECIALIST,
        contention.TrafficClass.BACKGROUND,
    ]:
        assert contention.pair_policy("frontdoor", "architect_general", tc, matrix=m) == contention.PairDecision.QUEUE


def test_policy_allow_above_one(minimal_matrix_yaml: Path) -> None:
    """frontdoor + worker_general at 1.28 → ALLOW for all classes."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    for tc in [
        contention.TrafficClass.FOREGROUND_INTERACTIVE,
        contention.TrafficClass.BACKGROUND,
    ]:
        assert contention.pair_policy("frontdoor", "worker_general", tc, matrix=m) == contention.PairDecision.ALLOW


def test_policy_borderline_below_floor_queues_all(minimal_matrix_yaml: Path) -> None:
    """frontdoor + vision_escalation at 0.84 is BELOW the 0.85 floor → QUEUE for
    all classes. Foreground caller can promote to DEGRADED_ALLOW based on SLO."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert contention.pair_policy(
        "frontdoor", "vision_escalation",
        contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m,
    ) == contention.PairDecision.QUEUE
    assert contention.pair_policy(
        "frontdoor", "vision_escalation",
        contention.TrafficClass.BACKGROUND, matrix=m,
    ) == contention.PairDecision.QUEUE


def test_policy_at_floor_allows_foreground(minimal_matrix_yaml: Path) -> None:
    """A pair exactly at the floor (0.85) should ALLOW foreground, QUEUE background.
    Test by passing a custom floor that puts vision_escalation at-or-just-below."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    # With floor=0.80, the 0.84 pair becomes "at-or-above" → foreground ALLOW
    assert contention.pair_policy(
        "frontdoor", "vision_escalation",
        contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m, floor=0.80,
    ) == contention.PairDecision.ALLOW
    assert contention.pair_policy(
        "frontdoor", "vision_escalation",
        contention.TrafficClass.BACKGROUND, matrix=m, floor=0.80,
    ) == contention.PairDecision.QUEUE


def test_policy_unknown_pair(minimal_matrix_yaml: Path) -> None:
    """Unknown pair → ALLOW foreground, QUEUE background."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert contention.pair_policy(
        "ingest_long_context", "worker_vision",
        contention.TrafficClass.BACKGROUND, matrix=m,
    ) == contention.PairDecision.QUEUE
    assert contention.pair_policy(
        "ingest_long_context", "worker_vision",
        contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m,
    ) == contention.PairDecision.ALLOW


def test_policy_same_role_allowed(minimal_matrix_yaml: Path) -> None:
    """frontdoor + frontdoor → ALLOW (matrix says same-role frontdoor is allow)."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert contention.pair_policy(
        "frontdoor", "frontdoor",
        contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m,
    ) == contention.PairDecision.ALLOW


def test_policy_same_role_blocked(minimal_matrix_yaml: Path) -> None:
    """vision_escalation + vision_escalation → QUEUE background, DEGRADED_ALLOW foreground."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert contention.pair_policy(
        "vision_escalation", "vision_escalation",
        contention.TrafficClass.BACKGROUND, matrix=m,
    ) == contention.PairDecision.QUEUE
    assert contention.pair_policy(
        "vision_escalation", "vision_escalation",
        contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m,
    ) == contention.PairDecision.DEGRADED_ALLOW


def test_policy_fail_open_on_missing_matrix(monkeypatch, tmp_path: Path) -> None:
    """When matrix is missing, foreground ALLOW, background QUEUE."""
    monkeypatch.setattr(contention, "DEFAULT_MATRIX_PATH", tmp_path / "absent.yaml")
    assert contention.pair_policy(
        "frontdoor", "architect_general",
        contention.TrafficClass.FOREGROUND_INTERACTIVE,
    ) == contention.PairDecision.ALLOW
    assert contention.pair_policy(
        "frontdoor", "architect_general",
        contention.TrafficClass.BACKGROUND,
    ) == contention.PairDecision.QUEUE


def test_traffic_class_string_coerces() -> None:
    """Passing a string for traffic_class should be coerced to TrafficClass."""
    m = contention.load_contention_matrix(
        Path(__file__).resolve().parents[2] / "orchestration" / "contention_matrix.yaml"
    )
    # "background" string should work
    decision = contention.pair_policy("frontdoor", "architect_general", "background", matrix=m)
    assert decision == contention.PairDecision.QUEUE
    # "foreground_interactive" string should work
    decision = contention.pair_policy("frontdoor", "architect_general", "foreground_interactive", matrix=m)
    assert decision == contention.PairDecision.QUEUE


# ── topology_fingerprint ─────────────────────────────────────────────


def test_topology_fingerprint_deterministic() -> None:
    config = {
        "frontdoor": {"instances": [("0-47,96-143", 8070, 96), ("0-23,96-119", 8080, 48)]},
        "worker_general": {"instances": [("0-95", 8072, 96)]},
    }
    h1 = contention.topology_fingerprint(config)
    h2 = contention.topology_fingerprint(config)
    assert h1 == h2 and len(h1) == 16


def test_topology_fingerprint_order_independent() -> None:
    """Same content in different insertion order should hash identically."""
    a = {"frontdoor": {"instances": [("0-47", 8070, 96)]}, "worker_general": {"instances": [("0-95", 8072, 96)]}}
    b = {"worker_general": {"instances": [("0-95", 8072, 96)]}, "frontdoor": {"instances": [("0-47", 8070, 96)]}}
    assert contention.topology_fingerprint(a) == contention.topology_fingerprint(b)


def test_topology_fingerprint_detects_change() -> None:
    base = {"frontdoor": {"instances": [("0-47", 8070, 96)]}}
    # Different thread count
    changed_threads = {"frontdoor": {"instances": [("0-47", 8070, 48)]}}
    assert contention.topology_fingerprint(base) != contention.topology_fingerprint(changed_threads)
    # Different cpu_list
    changed_cpus = {"frontdoor": {"instances": [("0-95", 8070, 96)]}}
    assert contention.topology_fingerprint(base) != contention.topology_fingerprint(changed_cpus)
    # New instance
    added = {"frontdoor": {"instances": [("0-47", 8070, 96), ("48-95", 8080, 96)]}}
    assert contention.topology_fingerprint(base) != contention.topology_fingerprint(added)


def test_topology_fingerprint_ignores_metadata() -> None:
    """mlock and numactl_policy should NOT change the topology fingerprint."""
    a = {"frontdoor": {"instances": [("0-47", 8070, 96)], "mlock": True}}
    b = {"frontdoor": {"instances": [("0-47", 8070, 96)], "mlock": False, "numactl_policy": "interleave=all"}}
    assert contention.topology_fingerprint(a) == contention.topology_fingerprint(b)


def test_topology_fingerprint_for_matrix_excludes_unmeasured_auxiliary_role() -> None:
    measured = {
        "frontdoor": {"instances": [("0-1", 8070, 2)]},
        "worker_general": {"instances": [("2-3", 8072, 2)]},
    }
    live = {
        **measured,
        "eval_batch_frontdoor": {"instances": [("0-1", 18070, 2)]},
    }
    matrix = contention.ContentionMatrix(
        version=1,
        measured_at="",
        host="",
        topology_hash=contention.topology_fingerprint(measured),
        default_floor=0.85,
        pairs={
            ("frontdoor", "worker_general"): contention.Pair(
                roles=("frontdoor", "worker_general"),
                ratio=1.0,
                verdict="allow",
            )
        },
    )

    assert contention.topology_fingerprint_for_matrix(live, matrix) == (
        contention.topology_fingerprint(measured)
    )
    assert contention.topology_fingerprint_for_matrix(live, matrix) != (
        contention.topology_fingerprint(live)
    )


def test_real_matrix_against_live_numa_config() -> None:
    """Smoke: the committed matrix should at least parse alongside the live NUMA_CONFIG.
    Topology hashes don't have to match (matrix uses a placeholder), but
    the topology_fingerprint of NUMA_CONFIG should be a sensible non-empty hash."""
    sys.path.insert(0, str(ROOT / "scripts" / "server"))
    import stack_numa
    h = contention.topology_fingerprint(stack_numa.NUMA_CONFIG)
    assert isinstance(h, str) and len(h) == 16


# ── nway_policy (J4c — N-way active-set admission) ───────────────────


def _nway_test_matrix():
    """Synthetic matrix: one measured-block triple + one measured-allow triple,
    with the production light/heavy role classification."""
    Nway = contention.Nway
    return contention.ContentionMatrix(
        version=1, measured_at="", host="", topology_hash="t", default_floor=0.85,
        n_way={
            ("frontdoor", "ingest_long_context", "vision_escalation"): Nway(
                roles=("frontdoor", "ingest_long_context", "vision_escalation"),
                ratio=0.847, verdict="block", contains_heavy=True),
            ("frontdoor", "vision_escalation", "worker_general"): Nway(
                roles=("frontdoor", "vision_escalation", "worker_general"),
                ratio=1.607, verdict="allow"),
        },
        light_roles=frozenset({"frontdoor", "vision_escalation", "worker_general", "worker_vision"}),
        heavy_roles=frozenset({"ingest_long_context", "architect_general"}),
    )


def test_nway_policy_measured_block_queues_even_when_pairs_allow():
    """The crux of J4c: a measured-block N-way set queues even though every
    constituent pair is allow ({frontdoor,ingest,vision} = 0.847)."""
    m = _nway_test_matrix()
    roles = ["frontdoor", "ingest_long_context", "vision_escalation"]
    assert contention.nway_policy(roles, contention.TrafficClass.BACKGROUND, matrix=m) == contention.PairDecision.QUEUE
    assert contention.nway_policy(roles, contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m) == contention.PairDecision.QUEUE


def test_nway_policy_measured_allow():
    m = _nway_test_matrix()
    assert contention.nway_policy(
        ["frontdoor", "vision_escalation", "worker_general"],
        contention.TrafficClass.BACKGROUND, matrix=m) == contention.PairDecision.ALLOW


def test_nway_policy_all_light_unmeasured_allows_both_classes():
    """Unmeasured but all-light (covers mixed multi-instance via role-set dedup)."""
    m = _nway_test_matrix()
    roles = ["frontdoor", "worker_general", "worker_vision"]  # not in n_way, all light
    assert contention.nway_policy(roles, contention.TrafficClass.BACKGROUND, matrix=m) == contention.PairDecision.ALLOW
    assert contention.nway_policy(roles, contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m) == contention.PairDecision.ALLOW


def test_nway_policy_heavy_unmeasured_fail_open_fg_closed_bg():
    m = _nway_test_matrix()
    roles = ["ingest_long_context", "worker_vision"]  # not in n_way, contains heavy
    assert contention.nway_policy(roles, contention.TrafficClass.BACKGROUND, matrix=m) == contention.PairDecision.QUEUE
    assert contention.nway_policy(roles, contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m) == contention.PairDecision.ALLOW


def test_nway_policy_single_role_is_allow():
    m = _nway_test_matrix()
    assert contention.nway_policy(["frontdoor"], matrix=m) == contention.PairDecision.ALLOW
    assert contention.nway_policy([], matrix=m) == contention.PairDecision.ALLOW


def test_nway_policy_order_independent():
    m = _nway_test_matrix()
    a = contention.nway_policy(["vision_escalation", "frontdoor", "ingest_long_context"], "background", matrix=m)
    b = contention.nway_policy(["ingest_long_context", "vision_escalation", "frontdoor"], "background", matrix=m)
    assert a == b == contention.PairDecision.QUEUE
