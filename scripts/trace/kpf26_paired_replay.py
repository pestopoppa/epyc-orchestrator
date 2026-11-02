#!/usr/bin/env python3
"""Manifest-bound KPF-26 paired replay; executor provides native witnesses.

No gate threshold or promotion decision is implemented.  See the adjacent contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "epyc.kpf26.paired.v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v) and v >= 0


def validate_manifest(m: Any, directory: Path) -> None:
    if not isinstance(m, dict) or m.get("schema") != SCHEMA:
        raise ValueError("bad manifest schema")
    tasks = m.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks required")
    ids = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"]:
            raise ValueError("invalid task id")
        ids.append(task["id"])
        if type(task.get("siblings")) is not int or task["siblings"] < 1:
            raise ValueError("invalid sibling count")
        for key in ("request", "expected"):
            ref = task.get(key)
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
                raise ValueError(f"missing {key} pin")
            p = (directory / ref["path"]).resolve()
            if not p.is_file() or sha(p) != ref["sha256"]:
                raise ValueError(f"{key} pin mismatch")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task id")
    if not number(m.get("arm_timeout_s")) or not m["arm_timeout_s"]:
        raise ValueError("positive arm timeout required")
    if not number(m.get("max_occupancy_sample_gap_s")) or not m["max_occupancy_sample_gap_s"]:
        raise ValueError("positive occupancy sample bound required")
    pins = m.get("pins")
    if not isinstance(pins, dict) or not all(isinstance(pins.get(k), str) and pins[k] for k in
                                              ("model", "server_binary_sha256", "source_commit", "argv", "executor_sha256")):
        raise ValueError("missing run pins")
    for key in ("server_binary_sha256", "executor_sha256"):
        if len(pins[key]) != 64 or any(c not in "0123456789abcdef" for c in pins[key]):
            raise ValueError("invalid digest")
    if m.get("arm_order") not in (["off", "on"], ["on", "off"]):
        raise ValueError("explicit paired arm order required")


def validate_evidence(e: Any, m: dict, manifest_hash: str, arm: str) -> list[str]:
    errors = []
    if not isinstance(e, dict):
        return ["evidence is not an object"]
    for key, value in (("arm", arm), ("manifest_sha256", manifest_hash), ("pins", m["pins"])):
        if e.get(key) != value:
            errors.append(f"{key} mismatch")
    if not isinstance(e.get("window_id"), str) or not e["window_id"]:
        errors.append("missing window id")
    if e.get("flags") != {"prefix_index": arm == "on", "fork": arm == "on", "lpm_max_skips": 1}:
        errors.append("flag readback mismatch")
    rows = e.get("tasks")
    if not isinstance(rows, list):
        return errors + ["tasks missing"]
    ids = [x.get("id") if isinstance(x, dict) else None for x in rows]
    expected_ids = [x["id"] for x in m["tasks"]]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        errors.append("task coverage mismatch")
    expected = {t["id"]: t for t in m["tasks"]}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") not in expected:
            continue
        task = expected[row["id"]]
        if row.get("request_sha256") != task["request"]["sha256"]:
            errors.append(f"{row['id']}: request mismatch")
        if row.get("expected_sha256") != task["expected"]["sha256"]:
            errors.append(f"{row['id']}: scorer input mismatch")
        children = row.get("children")
        if not isinstance(children, list) or len(children) != task["siblings"]:
            errors.append(f"{row['id']}: sibling coverage mismatch")
            continue
        child_ids = [c.get("id") if isinstance(c, dict) else None for c in children]
        if any(not isinstance(x, str) or not x for x in child_ids) or len(set(child_ids)) != len(child_ids):
            errors.append(f"{row['id']}: invalid child IDs")
        for c in children:
            if not isinstance(c, dict):
                errors.append(f"{row['id']}: bad child")
                continue
            if c.get("status") != "ok" or not all(number(c.get(k)) for k in ("ttft_ms", "admission_wait_ms", "prefill_tokens")):
                errors.append(f"{row['id']}: incomplete child outcome or metrics")
            if not isinstance(c.get("output_sha256"), str) or len(c["output_sha256"]) != 64 or type(c.get("outcome")) is not bool:
                errors.append(f"{row['id']}: missing scored outcome/output")
            if arm == "on" and (not number(c.get("n_fork_tokens")) or c["n_fork_tokens"] <= 0 or not c.get("fork_task_slot_witness")):
                errors.append(f"{row['id']}: missing native fork witness")
        if arm == "on":
            trunk = row.get("trunk")
            if not isinstance(trunk, dict) or trunk.get("count") != 1 or not trunk.get("checkpoint_witness") or not number(trunk.get("prefill_tokens")):
                errors.append(f"{row['id']}: missing trunk/checkpoint witness")
        start, end = row.get("start_s"), row.get("end_s")
        samples = row.get("unique_cells_samples")
        if not number(start) or not number(end) or end < start or not isinstance(samples, list) or not samples:
            errors.append(f"{row['id']}: missing occupancy interval/samples")
            continue
        if any(not isinstance(x, dict) or not number(x.get("t")) or not number(x.get("cells")) for x in samples):
            errors.append(f"{row['id']}: invalid occupancy samples")
            continue
        times = [x["t"] for x in samples]
        if times != sorted(times) or times[0] > start or times[-1] < end or any(
                b - a > m["max_occupancy_sample_gap_s"] for a, b in zip(times, times[1:])):
            errors.append(f"{row['id']}: occupancy sampling does not cover interval")
    return errors


def run(manifest_path: Path, executor: Path, output: Path) -> int:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    validate_manifest(manifest, manifest_path.parent)
    digest = hashlib.sha256(raw).hexdigest()
    if sha(executor) != manifest["pins"]["executor_sha256"]:
        raise ValueError("executor digest mismatch")
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(raw)
    all_errors: list[str] = []
    windows = []
    for arm in manifest["arm_order"]:
        target = output / f"{arm}.json"
        start = time.time()
        with (output / f"{arm}.stdout").open("wb") as out, (output / f"{arm}.stderr").open("wb") as err:
            try:
                proc = subprocess.run([str(executor), "--manifest", str(manifest_path.resolve()), "--arm", arm,
                                       "--evidence", str(target.resolve())], stdout=out, stderr=err, check=False,
                                      env=os.environ.copy(), timeout=manifest["arm_timeout_s"])
            except subprocess.TimeoutExpired:
                all_errors.append(f"{arm}: executor timed out")
                continue
        if proc.returncode != 0 or not target.is_file():
            all_errors.append(f"{arm}: executor exit {proc.returncode} or missing evidence")
            continue
        try:
            e = json.loads(target.read_text())
            all_errors.extend(f"{arm}: {x}" for x in validate_evidence(e, manifest, digest, arm))
            windows.append(e.get("window_id"))
        except (ValueError, OSError) as exc:
            all_errors.append(f"{arm}: invalid evidence: {exc}")
        (output / f"{arm}.runner.json").write_text(json.dumps({"start_s": start, "end_s": time.time(),
                                      "exit_code": proc.returncode, "evidence_sha256": sha(target)}) + "\n")
    if len(windows) != 2 or windows[0] == windows[1]:
        all_errors.append("distinct window IDs required")
    result = {"schema": SCHEMA, "manifest_sha256": digest, "errors": all_errors,
              "status": "INVALID" if all_errors else "OBSERVATION_COMPLETE",
              "gate_pass": None}
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 1 if all_errors else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--executor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        return run(args.manifest, args.executor.resolve(), args.output.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"kpf26: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
