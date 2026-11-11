"""INF-45 P1-8: region-lock wait reader over dispatch.v1 fixture JSONL."""

from __future__ import annotations

import json

from scripts.autopilot import lock_wait_reader as lw
from scripts.autopilot import optimization_brief as ob


def _rec(pid, run, snap):
    return {
        "schema": "epyc.orchestrator.dispatch.v1",
        "region_lock_wait_s_by_holder": snap,
        "provenance": {"host": "h", "pid": pid, "run_id": run},
    }


def _write(tmp_path, recs, extra=""):
    p = tmp_path / "dispatch.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs) + extra)
    return p


def test_latest_per_process_summed_across_processes(tmp_path):
    p = _write(
        tmp_path,
        [
            _rec(1, "a", {"coder": {"wait_s": 1.0, "waits": 1.0}}),
            _rec(1, "a", {"coder": {"wait_s": 3.0, "waits": 2.0}, "arch": {"wait_s": 0.5, "waits": 1.0}}),
            _rec(2, "b", {"coder": {"wait_s": 2.0, "waits": 4.0}}),
            _rec(1, "a", {"coder": {"wait_s": 2.5, "waits": 2.0}}),  # out of order: max wins
        ],
        extra="not json\n",
    )
    r = lw.read_region_lock_wait(p)
    assert r["read_only"] and r["processes"] == 2 and r["records_seen"] == 4
    assert r["by_holder"]["coder"] == {"wait_s": 5.0, "waits": 6}
    assert r["by_holder"]["arch"] == {"wait_s": 0.5, "waits": 1}
    assert list(r["by_holder"]) == ["coder", "arch"]
    assert r["total_wait_s"] == 5.5 and r["total_waits"] == 7


def test_missing_or_empty_never_raises(tmp_path):
    assert lw.read_region_lock_wait(tmp_path / "nope.jsonl")["by_holder"] == {}
    p = _write(tmp_path, [{"schema": "x"}, _rec(1, "a", {})])
    assert lw.read_region_lock_wait(p)["total_wait_s"] == 0.0


def test_brief_surfaces_section(tmp_path):
    p = _write(tmp_path, [_rec(1, "a", {"coder": {"wait_s": 1.5, "waits": 3.0}})])
    brief = ob.build_optimization_brief(
        state_path=tmp_path / "s.json",
        journal_paths=[tmp_path / "j.jsonl"],
        strategy_db=tmp_path / "x.db",
        digest_dir=tmp_path,
        digest_text="",
        dispatch_log_path=p,
    )
    assert brief["region_lock_wait"]["by_holder"]["coder"]["wait_s"] == 1.5
