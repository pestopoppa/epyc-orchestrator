"""W-40: derived cpuset-intersection placement feasibility (no measurement, no inference)."""
from __future__ import annotations

import json
import subprocess
import sys

from scripts.server import placement_feasibility as pf
from scripts.server import stack_numa


def _pair(art, a, b):
    for p in art["pairs"]:
        if {p["a"], p["b"]} == {a, b}:
            return p
    raise AssertionError((a, b))


def test_halves_disjoint_full_exclusive():
    art = pf.build_artifact(stack_numa)
    assert _pair(art, "NUMA_HALF_A", "NUMA_HALF_B")["may_co_run"] is True
    assert _pair(art, "NUMA_FULL", "NUMA_HALF_A")["verdict"] == "intersecting"
    assert _pair(art, "NUMA_FULL", "NUMA_HALF_B")["may_co_run"] is False
    assert art["measured"] is False


def test_quarters_pairwise_disjoint_and_half_a_is_q0_plus_q1():
    art = pf.build_artifact(stack_numa)
    qs = ["NUMA_Q0A", "NUMA_Q0B", "NUMA_Q1A", "NUMA_Q1B"]
    for i, x in enumerate(qs):
        for y in qs[i + 1:]:
            assert _pair(art, x, y)["may_co_run"]
    assert not _pair(art, "NUMA_HALF_A", "NUMA_Q0A")["may_co_run"]


def test_smt_sibling_overlap_is_intersecting():
    assert pf.pair_verdict("0-3", "96-99")["may_co_run"] is False
    assert pf.pair_verdict("0-3", "4-7")["may_co_run"] is True


def test_committed_artifact_is_current():
    rc, msg = pf.check()
    assert rc == 0, msg


def test_check_detects_stale_and_missing(tmp_path):
    p = tmp_path / "pf.json"
    assert pf.check(p)[0] == 2
    art = pf.build_artifact(stack_numa)
    art["topology_hash"] = "0" * 64
    p.write_text(json.dumps(art))
    assert pf.check(p)[0] == 2
    p.write_text("{")
    assert pf.check(p)[0] == 3
    p.write_text(pf.render(pf.build_artifact(stack_numa)))
    assert pf.check(p)[0] == 0


def test_hash_tracks_topology_change():
    class Fake:
        _SHAPE_CLASSES = stack_numa._SHAPE_CLASSES
        NUMA_CONFIG = {"r": {"instances": [("0-23,96-119", 8000, 48)]}}
        NUMA_FULL = stack_numa.NUMA_FULL
        NUMA_HALF_A = stack_numa.NUMA_HALF_A
        NUMA_HALF_B = stack_numa.NUMA_HALF_B
        NUMA_Q0A = stack_numa.NUMA_Q0A
        NUMA_Q0B = stack_numa.NUMA_Q0B
        NUMA_Q1A = stack_numa.NUMA_Q1A
        NUMA_Q1B = stack_numa.NUMA_Q1B
        GPU_HOST_LANE = stack_numa.GPU_HOST_LANE

    assert pf.build_artifact(Fake)["topology_hash"] != pf.build_artifact(stack_numa)["topology_hash"]


def test_precommit_gate_consults_derived_artifact_not_benchmarked_matrix():
    r = subprocess.run(
        [sys.executable, str(pf.REPO_ROOT / "scripts/validate/check_contention_matrix_fresh.py")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "placement feasibility" in r.stdout
