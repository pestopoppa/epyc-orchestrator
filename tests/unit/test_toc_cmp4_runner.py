"""TOC-CMP-4 runner: determinism, frozen draw, arm permutation, invalid-run policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))

import toc_cmp4_runner as r  # noqa: E402

POOL = [f"t{i:02d}" for i in range(50)]


def test_draw_is_deterministic_stratified_and_30_points():
    a, b = r.draw_points(POOL), r.draw_points(list(reversed(POOL)))
    assert a == b and len(a) == 30
    assert {s: sum(p["stratum"] == s for p in a) for s in r.STRATA} == {"early": 10, "middle": 10, "late": 10}
    assert all(p["boundary_turn"] in r.STRATA[p["stratum"]] for p in a)
    assert len({p["point_id"] for p in a}) == 30
    assert r.points_sha256(a) == r.points_sha256(b)
    assert r.points_sha256(a) != r.points_sha256(r.draw_points(POOL, seed=1))


def test_schedule_covers_every_point_arm_repeat_with_seeded_order():
    pts = r.draw_points(POOL)
    rows = r.build_schedule(pts)
    assert len(rows) == 30 * 3 * 8 == 720
    assert len({x["run_id"] for x in rows}) == 720
    first = [x for x in rows if x["point_id"] == pts[0]["point_id"] and x["repeat"] == 0]
    assert [x["arm"] for x in first] == r.arm_permutation(pts[0]["point_id"])
    assert set(r.arm_permutation(pts[0]["point_id"])) == set(r.ARMS_A)
    # repeats differ only in sampling seed (same env except TRIAL_INDEX)
    e0 = next(x for x in rows if x["repeat"] == 0)
    e1 = next(x for x in rows if x["repeat"] == 1 and x["run_id"].split("|r")[0] == e0["run_id"].split("|r")[0])
    assert e0["sampling_seed"] != e1["sampling_seed"]
    assert {k: v for k, v in e0["env"].items() if k != "ORCHESTRATOR_CONTEXT_TRIAL_INDEX"} == {
        k: v for k, v in e1["env"].items() if k != "ORCHESTRATOR_CONTEXT_TRIAL_INDEX"}
    assert not {"raw_tail", "fifo"} & {x["arm"] for x in rows}


def _good(row):
    return {"pins": {k: "x" for k in r.REQUIRED_PINS}, "blocked": 0}


def test_invalid_run_policy_and_resume(tmp_path):
    rows = r.build_schedule(r.draw_points(POOL)[:1], repeats=1)  # 3 rows
    calls = []

    def submit(row):
        calls.append(row["run_id"])
        if row["arm"] == rows[0]["arm"] and calls.count(row["run_id"]) == 1:
            raise OSError("transport")  # first failure re-run once, then ok
        if row["arm"] == rows[1]["arm"]:
            return {"pins": {}}  # missing pins twice -> dropped
        return _good(row)

    ledger = tmp_path / "l.jsonl"
    counts = r.run_schedule(rows, submit, ledger)
    assert (counts["ok"], counts["rerun"], counts["dropped"], counts["skipped_done"]) == (2, 2, 1, 0)
    assert r.run_schedule(rows, submit, ledger)["skipped_done"] == 3  # resume
    assert len(ledger.read_text().splitlines()) == 3


def test_cli_draw_and_plan(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text('{"__core_metadata__": true}\n' + "".join(json.dumps({"id": i}) + "\n" for i in POOL))
    pts, sched = tmp_path / "p.json", tmp_path / "s.jsonl"
    assert r.main(["draw", "--pool-jsonl", str(pool), "--out", str(pts)]) == 0
    assert r.main(["plan", "--points", str(pts), "--out", str(sched)]) == 0
    assert len(sched.read_text().splitlines()) == 720
    d = json.loads(pts.read_text())
    d["points"][0]["boundary_turn"] = 99
    pts.write_text(json.dumps(d))
    assert r.main(["plan", "--points", str(pts), "--out", str(sched)]) == 3


def test_void_when_boundary_not_reached_replaced_from_seeded_reserve(tmp_path):
    pts = r.draw_points(POOL)
    reserves = r.draw_reserves(POOL)
    assert all(len(v) == 40 for v in reserves.values())
    assert not {p["point_id"] for p in pts} & {q["point_id"] for v in reserves.values() for q in v if q["task_id"] in {p["task_id"] for p in pts if p["stratum"] == q["stratum"]}}
    victim = pts[0]
    first_reserve = reserves[victim["stratum"]][0]

    def submit(row):
        res = {"pins": {k: "x" for k in r.REQUIRED_PINS}}
        if row["point_id"] == victim["point_id"] and row["arm"] == row["arm"]:
            res["boundary_reached"] = False
        return res

    ledger = tmp_path / "l.jsonl"
    counts = r.run_schedule(r.build_schedule([victim], repeats=2), submit, ledger, reserves=reserves, repeats=2)
    assert counts["voided_points"] == [victim["point_id"]]
    assert counts["replacements"] == [{"void": victim["point_id"], "replacement": first_reserve["point_id"]}]
    assert counts["ok"] == 6 and counts["reserve_exhausted"] == 0  # replacement: 3 arms x 2 repeats
    recs = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert sum(x["status"] == "void_point" for x in recs) == 1
    # resume: voided point stays voided, nothing re-run
    again = r.run_schedule(r.build_schedule([victim], repeats=2), submit, ledger, reserves=reserves, repeats=2)
    assert again["ok"] == 0
