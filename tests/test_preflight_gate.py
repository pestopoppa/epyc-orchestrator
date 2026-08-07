#!/usr/bin/env python3
"""Tests for preflight_gate.py (B4 inference-batch preflight gate).

Hermetic: probe subprocesses (affinity_preflight, contention-freshness checker)
are either monkeypatched or replaced with tiny temp scripts run for real, so no
live stack, servers, or heavy probes are touched. NO inference.

Covers:
  * attestation shape (all required top-level keys)
  * PASS when every check ok; FAIL aggregation with mocked probe failures
  * topology-hash drift gate (expected match / mismatch / missing topology)
  * contention-freshness exit-code mapping (0 fresh / non-0 stale)
  * live-affinity artifact parsing (verified true/false)
  * write_attestation round-trip
"""
from __future__ import annotations

import json

from scripts.server import preflight_gate as pg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _all_pass_checks() -> dict:
    return {
        "live_affinity": {"ok": True, "live_affinity_verified": True},
        "health": {"ok": True, "health_ok": True},
        "topology": {"ok": True, "topology_hash": "abc123", "registry_hash": "def456"},
        "contention_matrix": {"ok": True, "contention_matrix_fresh": True},
    }


_REQUIRED_KEYS = {
    "ts", "topology_hash", "registry_hash", "expected_topology_hash",
    "live_affinity_verified", "contention_matrix_fresh", "health_ok",
    "checks", "overall", "fail_reasons",
}


# --------------------------------------------------------------------------- #
# Shape + PASS
# --------------------------------------------------------------------------- #
def test_attest_shape_and_pass():
    att = pg.attest(checks=_all_pass_checks())
    assert _REQUIRED_KEYS <= set(att)
    assert att["overall"] == "PASS"
    assert att["fail_reasons"] == []
    assert att["live_affinity_verified"] is True
    assert att["contention_matrix_fresh"] is True
    assert att["health_ok"] is True
    assert att["topology_hash"] == "abc123"
    assert att["registry_hash"] == "def456"


# --------------------------------------------------------------------------- #
# FAIL aggregation
# --------------------------------------------------------------------------- #
def test_attest_fail_aggregation_multiple():
    checks = _all_pass_checks()
    checks["live_affinity"] = {"ok": False, "live_affinity_verified": False,
                               "error": "no live process"}
    checks["contention_matrix"] = {"ok": False, "contention_matrix_fresh": False,
                                   "error": "stale"}
    att = pg.attest(checks=checks)
    assert att["overall"] == "FAIL"
    assert att["live_affinity_verified"] is False
    assert att["contention_matrix_fresh"] is False
    assert any(r.startswith("live_affinity:") and "no live process" in r
               for r in att["fail_reasons"])
    assert any(r.startswith("contention_matrix:") and "stale" in r
               for r in att["fail_reasons"])
    # health + topology were ok ⇒ not in fail_reasons
    assert not any(r.startswith("health:") for r in att["fail_reasons"])


def test_attest_default_path_with_monkeypatched_probes(monkeypatch):
    monkeypatch.setattr(pg, "check_live_affinity",
                        lambda **k: {"ok": True, "live_affinity_verified": True})
    monkeypatch.setattr(pg, "check_health",
                        lambda **k: {"ok": True, "health_ok": True})
    monkeypatch.setattr(pg, "check_topology_hashes",
                        lambda **k: {"ok": True, "topology_hash": "h", "registry_hash": "r"})
    monkeypatch.setattr(pg, "check_contention_matrix_fresh",
                        lambda **k: {"ok": False, "contention_matrix_fresh": False,
                                     "error": "STALE"})
    att = pg.attest()
    assert att["overall"] == "FAIL"
    assert att["health_ok"] is True
    assert att["topology_hash"] == "h"
    assert any("contention_matrix" in r and "STALE" in r for r in att["fail_reasons"])


def test_role_scoped_require_servers_uses_primary_role_ports(monkeypatch):
    seen = {}

    monkeypatch.setattr(pg, "check_live_affinity",
                        lambda **k: {"ok": True, "live_affinity_verified": True})
    monkeypatch.setattr(pg, "check_topology_hashes",
                        lambda **k: {"ok": True, "topology_hash": "h", "registry_hash": "r"})
    monkeypatch.setattr(pg, "check_contention_matrix_fresh",
                        lambda **k: {"ok": True, "contention_matrix_fresh": True})

    def fake_health(**kwargs):
        seen.update(kwargs)
        return {"ok": True, "health_ok": True}

    monkeypatch.setattr(pg, "check_health", fake_health)

    att = pg.attest(roles=["frontdoor", "worker_general"], require_servers=True)

    assert att["overall"] == "PASS"
    assert seen["require_servers"] is True
    assert seen["ports"] == [
        8070,
        8080,
        8180,
        8072,
        8082,
        8182,
    ]


def test_role_scoped_live_only_ports_skip_nonlive_configured_ports(monkeypatch):
    monkeypatch.setattr(pg, "_pid_on_port", lambda port: "123" if port in {8080, 8082} else None)

    assert pg.ports_for_roles(["frontdoor", "worker_general"], live_only=True) == [
        8080,
        8082,
    ]


def test_server_health_only_skips_structural_health(monkeypatch):
    monkeypatch.setattr(pg, "_probe_ports", lambda ports, timeout: {
        "ports": {"8070": True},
        "error": None,
    })

    res = pg.check_health(
        require_servers=True,
        ports=[8070],
        script=pg.ORCH / "does-not-exist.sh",
        server_health_only=True,
    )

    assert res["ok"] is True
    assert res["health_ok"] is True
    assert res["structural_ok"] is None
    assert res["structural_skipped"] is True


def test_health_check_uses_batch_profile(tmp_path):
    script = tmp_path / "health.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"--profile\" && \"$2\" == \"batch\" ]]; then exit 0; fi\n"
        "exit 7\n"
    )

    res = pg.check_health(script=script)

    assert res["ok"] is True
    assert res["health_ok"] is True
    assert res["health_profile"] == "batch"
    assert res["health_check_rc"] == 0


def test_contention_observation_only_records_stale_without_failing(tmp_path):
    stale_script = tmp_path / "stale.py"
    stale_script.write_text("import sys; print('stale'); sys.exit(2)\n")

    res = pg.check_contention_matrix_fresh(
        script=stale_script,
        observation_only=True,
    )

    assert res["ok"] is True
    assert res["contention_matrix_fresh"] is False
    assert res["warning"] == "contention matrix not fresh (rc=2)"
    assert res["error"] is None


def test_live_affinity_live_only_passes_when_live_instances_match(tmp_path):
    script = tmp_path / "aff_live_only.py"
    script.write_text(
        "import argparse, json, sys\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--output')\n"
        "ap.add_argument('--roles', nargs='*')\n"
        "args = ap.parse_args()\n"
        "json.dump({\n"
        "  'live_affinity_verified': False,\n"
        "  'live_memory_placement_verified': True,\n"
        "  'instances': [\n"
        "    {'port': 8070, 'pid': '123', 'match': True},\n"
        "    {'port': 8080, 'pid': None, 'match': False},\n"
        "  ],\n"
        "}, open(args.output, 'w'))\n"
        "sys.exit(1)\n"
    )

    res = pg.check_live_affinity(script=script, live_only=True)

    assert res["ok"] is True
    assert res["live_affinity_verified"] is True
    assert res["configured_affinity_verified"] is False
    assert res["artifact_summary"]["live_instances"] == 1
    assert res["artifact_summary"]["live_matched"] == 1


def test_live_affinity_memory_observation_does_not_fail_without_requirement(tmp_path):
    script = tmp_path / "aff_memory_observation.py"
    script.write_text(
        "import argparse, json, sys\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--output')\n"
        "ap.add_argument('--roles', nargs='*')\n"
        "args = ap.parse_args()\n"
        "json.dump({\n"
        "  'live_affinity_verified': False,\n"
        "  'live_memory_placement_verified': False,\n"
        "  'memory_locality_required': False,\n"
        "  'instances': [{'port': 8082, 'pid': '123', 'match': True}],\n"
        "}, open(args.output, 'w'))\n"
        "sys.exit(1)\n"
    )

    res = pg.check_live_affinity(script=script, live_only=True)

    assert res["ok"] is True
    assert res["live_affinity_verified"] is True
    assert res["memory_locality_required"] is False


# --------------------------------------------------------------------------- #
# Topology-hash gate
# --------------------------------------------------------------------------- #
def test_topology_hash_recorded_without_expected(tmp_path):
    reg = tmp_path / "reg.yaml"
    reg.write_text("roles: {}\n")
    matrix = tmp_path / "matrix.yaml"
    res = pg.check_topology_hashes(live_registry=reg, matrix_path=matrix)
    assert res["ok"] is True  # observation-only when no expected hash
    assert res["topology_hash"] is not None
    assert res["registry_hash"] == pg._sha256(reg)
    assert res["topology_source"] == str(matrix)
    assert res["registry_source"] == str(reg)
    assert res["topology_match"] is None


def test_topology_hash_match_and_drift(tmp_path, monkeypatch):
    reg = tmp_path / "reg.yaml"
    reg.write_text("roles: {a: 1}\n")
    monkeypatch.setattr(pg, "_live_topology_hash", lambda matrix_path: "8c8cfcbb13d2611d")
    good = pg.check_topology_hashes(expected_topology_hash="8c8cfcbb13d2611d",
                                    live_registry=reg)
    assert good["ok"] is True and good["topology_match"] is True
    assert good["registry_hash"] == pg._sha256(reg)
    bad = pg.check_topology_hashes(expected_topology_hash="deadbeef",
                                   live_registry=reg)
    assert bad["ok"] is False and bad["topology_match"] is False
    assert "drift" in bad["error"]


def test_topology_hash_missing_live_topology(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "_live_topology_hash", lambda matrix_path: None)
    missing = tmp_path / "does_not_exist.yaml"
    res = pg.check_topology_hashes(live_registry=missing, matrix_path=missing)
    assert res["ok"] is False
    assert res["topology_hash"] is None
    assert "live topology hash unavailable" in res["error"]


def test_attest_topology_hash_is_not_registry_hash():
    checks = _all_pass_checks()
    checks["topology"] = {
        "ok": True,
        "topology_hash": "8c8cfcbb13d2611d",
        "registry_hash": "f09dc260" * 8,
        "topology_match": True,
    }

    att = pg.attest(
        expected_topology_hash="8c8cfcbb13d2611d",
        checks=checks,
    )

    assert att["overall"] == "PASS"
    assert att["topology_hash"] == "8c8cfcbb13d2611d"
    assert att["registry_hash"] == "f09dc260" * 8
    assert att["expected_topology_hash"] == "8c8cfcbb13d2611d"


# --------------------------------------------------------------------------- #
# Contention-freshness exit-code mapping (real subprocess, tiny temp scripts)
# --------------------------------------------------------------------------- #
def test_contention_fresh_exit_code_mapping(tmp_path):
    ok_script = tmp_path / "fresh_ok.py"
    ok_script.write_text("import sys; sys.exit(0)\n")
    stale_script = tmp_path / "stale.py"
    stale_script.write_text("import sys; sys.exit(2)\n")

    fresh = pg.check_contention_matrix_fresh(script=ok_script)
    assert fresh["ok"] is True and fresh["contention_matrix_fresh"] is True
    assert fresh["returncode"] == 0

    stale = pg.check_contention_matrix_fresh(script=stale_script)
    assert stale["ok"] is False and stale["contention_matrix_fresh"] is False
    assert stale["returncode"] == 2


def test_contention_fresh_missing_script(tmp_path):
    res = pg.check_contention_matrix_fresh(script=tmp_path / "nope.py")
    assert res["ok"] is False
    assert "missing" in res["error"]


# --------------------------------------------------------------------------- #
# Live-affinity artifact parsing (real subprocess, tiny temp script)
# --------------------------------------------------------------------------- #
def _affinity_stub(verified: bool) -> str:
    artifact = json.dumps({
        "live_affinity_verified": verified,
        "live_memory_placement_verified": True,
        "instances": [{"match": True}],
    })
    return (
        "import sys, json, argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--output')\n"
        "ap.add_argument('--roles', nargs='*')\n"
        "a = ap.parse_args()\n"
        "open(a.output, 'w').write(" + repr(artifact) + ")\n"
        "sys.exit(" + ("0" if verified else "1") + ")\n"
    )


def test_live_affinity_verified_true(tmp_path):
    script = tmp_path / "aff_ok.py"
    script.write_text(_affinity_stub(True))
    res = pg.check_live_affinity(script=script)
    assert res["ok"] is True
    assert res["live_affinity_verified"] is True
    assert res["artifact_summary"]["instances"] == 1


def test_live_affinity_verified_false(tmp_path):
    script = tmp_path / "aff_bad.py"
    script.write_text(_affinity_stub(False))
    res = pg.check_live_affinity(script=script)
    assert res["ok"] is False
    assert res["live_affinity_verified"] is False


def test_live_affinity_missing_script(tmp_path):
    res = pg.check_live_affinity(script=tmp_path / "nope.py")
    assert res["ok"] is False
    assert res["live_affinity_verified"] is False
    assert "missing" in res["error"]


# --------------------------------------------------------------------------- #
# write_attestation round-trip
# --------------------------------------------------------------------------- #
def test_write_attestation_roundtrip(tmp_path):
    att = pg.attest(checks=_all_pass_checks())
    path = pg.write_attestation(att, output_dir=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["overall"] == "PASS"
    assert _REQUIRED_KEYS <= set(data)
