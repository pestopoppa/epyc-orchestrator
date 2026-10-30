"""Synthetic rates, actual writer geometry and public matrix-consumer policy."""

from types import SimpleNamespace

import pytest
import yaml

from scripts.server import contention_matrix as writer
from src.scheduling.contention import (
    PairDecision,
    TrafficClass,
    load_contention_matrix,
    pair_policy,
    topology_fingerprint_for_matrix,
)


def _write_matrix(monkeypatch, tmp_path, live_ports, *, witness=True):
    config = {
        "frontdoor": {"instances": [("0-95", 8070, 96), ("0-47,96-143", 8080, 48)]},
        "ingest_long_context": {
            "instances": [("0-47,96-143", 8085, 48), ("48-95,144-191", 8285, 48)]
        },
    }
    if witness:
        config["witness"] = {"instances": [("184-191", 8090, 8)]}
        live_ports = {*live_ports, 8090}
    monkeypatch.setattr("stack_numa.NUMA_CONFIG", config)
    monkeypatch.setattr(writer, "_port_healthy", lambda port, timeout_s=0.5: port in live_ports)
    calls = []

    def bench(role_a, port_a, role_b, port_b, *, instance_a=None, instance_b=None):
        calls.append(((role_a, role_b), (port_a, port_b)))
        return writer.PairBench(
            roles=tuple(sorted((role_a, role_b))),
            instance_a=instance_a,
            instance_b=instance_b,
            solo_a=writer.BenchResult(port_a, role_a, 10, 1),
            solo_b=writer.BenchResult(port_b, role_b, 10, 1),
            parallel_a=writer.BenchResult(port_a, role_a, 18.9, 1),
            parallel_b=writer.BenchResult(port_b, role_b, 18.9, 1),
            seq_aggregate_tps=10,
            parallel_aggregate_tps=18.9,
            ratio=1.89,
        )

    monkeypatch.setattr(writer, "_bench_pair", bench)
    monkeypatch.setattr(writer, "_binary_metadata", lambda _path: {"git_commit": "MOCK_ONLY"})
    monkeypatch.setattr(writer, "_host_metadata", lambda: {"hostname": "MOCK_ONLY"})
    output = tmp_path / "matrix.yaml"
    assert (
        writer.cmd_run(SimpleNamespace(roles=list(config), dry_run=False, output=str(output))) == 0
    )
    return yaml.safe_load(output.read_text()), load_contention_matrix(output), calls, config


def test_healthy_primary_overlap_is_measured_even_when_disjoint_fallback_is_live(
    monkeypatch, tmp_path
):
    raw, matrix, calls, config = _write_matrix(monkeypatch, tmp_path, {8070, 8085, 8080, 8285})
    assert [ports for roles, ports in calls if roles == ("frontdoor", "ingest_long_context")] == [
        (8070, 8085)
    ]
    pair = next(p for p in raw["pairs"] if p["roles"] == ["frontdoor", "ingest_long_context"])
    assert pair["instance_a"]["instance_idx"] == pair["instance_b"]["instance_idx"] == 0
    assert set(pair["instance_a"]["regions"]) & set(pair["instance_b"]["regions"])
    for traffic in TrafficClass:
        assert (
            pair_policy("frontdoor", "ingest_long_context", traffic, matrix=matrix)
            is PairDecision.ALLOW
        )


@pytest.mark.parametrize("live_ports", [{8070, 8285}, {8080, 8085}])
def test_substituted_live_geometry_is_unknown_to_existing_consumer(
    monkeypatch, tmp_path, live_ports
):
    raw, matrix, calls, config = _write_matrix(monkeypatch, tmp_path, live_ports)
    assert not any(roles == ("frontdoor", "ingest_long_context") for roles, _ in calls)
    assert all(
        p["instance_a"]["instance_idx"] == p["instance_b"]["instance_idx"] == 0
        for p in raw["pairs"]
    )
    assert not any(p["roles"] == ["frontdoor", "ingest_long_context"] for p in raw["pairs"])
    assert raw["topology_hash"] == topology_fingerprint_for_matrix(config, matrix)
    assert matrix.is_unknown_pair("frontdoor", "ingest_long_context")
    assert (
        pair_policy("frontdoor", "ingest_long_context", TrafficClass.BACKGROUND, matrix=matrix)
        is PairDecision.QUEUE
    )
    assert (
        pair_policy("frontdoor", "ingest_long_context", TrafficClass.MAINTENANCE, matrix=matrix)
        is PairDecision.ALLOW
    )
    for traffic in (TrafficClass.FOREGROUND_INTERACTIVE, TrafficClass.FOREGROUND_SPECIALIST):
        assert (
            pair_policy("frontdoor", "ingest_long_context", traffic, matrix=matrix)
            is PairDecision.ALLOW
        )


def test_unknown_only_output_is_not_qualified_and_preserves_unknown_policy(monkeypatch, tmp_path):
    raw, matrix, calls, config = _write_matrix(monkeypatch, tmp_path, {8080, 8285}, witness=False)
    assert calls == []
    assert not raw["pairs"] and matrix.get_pair("frontdoor", "ingest_long_context") is None
    assert raw["decision_grade"] is False
    assert (
        "no configured-primary pair was measured; unknown pairs only"
        in raw["decision_grade_blockers"]
    )
    assert writer.read_matrix_host_health(tmp_path / "matrix.yaml")["decision_grade"] is False
    assert raw["topology_hash"] == topology_fingerprint_for_matrix(config, matrix)
    assert (
        pair_policy("frontdoor", "ingest_long_context", TrafficClass.BACKGROUND, matrix=matrix)
        is PairDecision.QUEUE
    )
    assert (
        pair_policy(
            "frontdoor", "ingest_long_context", TrafficClass.FOREGROUND_INTERACTIVE, matrix=matrix
        )
        is PairDecision.ALLOW
    )
