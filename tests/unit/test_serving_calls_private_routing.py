"""ws58 2026-11-02: only production ports write the production serving-call log.

Private / experimental servers (8102, 8199, 18283, 18973) appended to
``logs/serving_calls/serving_calls.jsonl``, the evidence base of gates such as
UFH14-B4a's 200-call window. The writer now routes by ``server.port``; the read-only
``scripts/analysis/serving_calls_nonprod_report.py`` reports what is already there.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from src.backends import serving_calls as sc
from src.registry import stack_priors as sp


@pytest.fixture
def logdir(monkeypatch, tmp_path) -> Path:
    monkeypatch.delenv(sc.LOG_ENV, raising=False)   # the suite sets it to "off"
    monkeypatch.delenv(sc.PRIVATE_LOG_ENV, raising=False)
    monkeypatch.setenv(sc.PRODUCTION_PORTS_ENV, "8070,8080,8083")
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    sc._SIDECAR_CACHE.clear()
    sc.clear_staged()
    yield tmp_path
    sc.clear_staged()


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _call(port: int | None) -> bool:
    now = time.time()
    return sc.record_http_call(
        method="test_lane", role="frontdoor",
        base_url=f"http://localhost:{port}" if port else None,
        endpoint="/completion", ts_start=now - 0.5, ts_end=now, prompt="p", status=200,
        timings={"prompt_n": 3, "predicted_n": 2},
    )


def test_production_port_writes_production_log(logdir):
    assert _call(8070)
    prod = logdir / "serving_calls" / "serving_calls.jsonl"
    assert [r["server"]["port"] for r in _records(prod)] == [8070]
    assert not (logdir / "serving_calls" / sc.PRIVATE_LOG_NAME).exists()


@pytest.mark.parametrize("port", [8102, 8199, 18283, 18973])
def test_private_port_never_writes_production_log(logdir, port):
    assert _call(port)
    prod = logdir / "serving_calls" / "serving_calls.jsonl"
    private = logdir / "serving_calls" / sc.PRIVATE_LOG_NAME
    assert _records(prod) == []
    recs = _records(private)
    assert [r["server"]["port"] for r in recs] == [port]
    assert recs[0]["record_sha256"] == sc.record_digest(recs[0])


def test_portless_record_stays_in_production_log(logdir):
    assert _call(None)
    prod = logdir / "serving_calls" / "serving_calls.jsonl"
    assert len(_records(prod)) == 1


def test_private_env_path_is_run_local(logdir, monkeypatch, tmp_path):
    run_local = tmp_path / "scratch" / "run-x" / "serving_calls.jsonl"
    monkeypatch.setenv(sc.PRIVATE_LOG_ENV, str(run_local))
    assert _call(18973)
    assert [r["server"]["port"] for r in _records(run_local)] == [18973]
    assert not (logdir / "serving_calls" / "serving_calls.jsonl").exists()


def test_private_off_drops_and_counts(logdir, monkeypatch):
    monkeypatch.setenv(sc.PRIVATE_LOG_ENV, "off")
    before = sc.health()["skipped_private_disabled"]
    assert not _call(8102)
    assert sc.health()["skipped_private_disabled"] == before + 1
    assert not (logdir / "serving_calls" / "serving_calls.jsonl").exists()


def test_explicit_log_override_receives_every_record(logdir, monkeypatch, tmp_path):
    path = tmp_path / "chosen.jsonl"
    monkeypatch.setenv(sc.LOG_ENV, str(path))
    assert _call(8102) and _call(8070)
    assert [r["server"]["port"] for r in _records(path)] == [8102, 8070]


def test_production_ports_from_stack_priors(logdir, monkeypatch, tmp_path):
    monkeypatch.delenv(sc.PRODUCTION_PORTS_ENV)
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        "roles:\n"
        "  frontdoor: {deployment_status: live_stack, serving: {ports: [8070, 8080, 8180], endpoint: 'http://localhost:8070'}}\n"
        "  architect_general: {deployment_status: live_stack, serving: {endpoint: 'http://localhost:8074'}}\n"
        "  retired: {deployment_status: retired, serving: {ports: [8102]}}\n"
    )
    monkeypatch.setattr(sp, "DEFAULT_OUTPUT", priors)
    sc._PROD_PORTS_CACHE.update(key=None, ports=None)
    assert sc.production_ports() == frozenset({8070, 8074, 8080, 8180})
    assert sc.destination({"server": {"port": 8074}})[1] == "production"
    assert sc.destination({"server": {"port": 8102}})[1] == "private"


def test_unresolvable_production_ports_fail_closed_to_private(logdir, monkeypatch, tmp_path):
    monkeypatch.delenv(sc.PRODUCTION_PORTS_ENV)
    monkeypatch.setattr(sp, "DEFAULT_OUTPUT", tmp_path / "missing.yaml")
    sc._PROD_PORTS_CACHE.update(key=None, ports=None)
    before = sc.health()["routed_private_unresolved"]
    assert _call(8070)
    assert sc.health()["routed_private_unresolved"] == before + 1
    assert not (logdir / "serving_calls" / "serving_calls.jsonl").exists()


def test_live_stack_priors_exclude_the_private_ports():
    """The repo's own compiled stack priors never name the incident's private ports."""
    sc._PROD_PORTS_CACHE.update(key=None, ports=None)
    ports = sc.production_ports()
    assert ports, "orchestration/derived/stack_priors.yaml yields no live_stack ports"
    assert not ports & {8102, 8199, 18283, 18973, 19198}


def test_nonprod_report_counts_per_port_read_only(tmp_path):
    from scripts.analysis import serving_calls_nonprod_report as rep

    log = tmp_path / "serving_calls.jsonl"
    rows = [{"server": {"port": p}, "record_sha256": f"d{i}"} for i, p in
            enumerate([8070, 8102, 8102, 18973, None, 8083])]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows[:3]))
    log.with_name(log.name + ".1").write_text("".join(json.dumps(r) + "\n" for r in rows[3:]) + "not json\n")
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    out = rep.classify(rep.shards(log), frozenset({8070, 8083}))
    assert out["non_production_by_port"] == {"8102": 2, "18973": 1}
    assert out["production"] == 3 and out["non_production"] == 3 and out["unparsed_lines"] == 1
    assert out["_digests"]["8102"] == ["d1", "d2"]
    assert rep.main(["--log", str(log), "--production-ports", "8070,8083", "--json"]) == 0
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before
