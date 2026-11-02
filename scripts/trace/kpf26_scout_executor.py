#!/usr/bin/env python3
"""Collect KPF-26 evidence from the actual scout stage; never infer missing witnesses.

The parent paired runner validates the result. The runtime grant and server process are
inputs, not authority conferred by this script. See KPF26_PAIRED_REPLAY.md.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trace.kpf26_paired_replay import validate_evidence, validate_manifest  # noqa: E402

_SCOUT_ID: contextvars.ContextVar[int | None] = contextvars.ContextVar("kpf26_scout_id", default=None)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pinned_file(base: Path, ref: dict[str, str]) -> bytes:
    raw = (base / ref["path"]).read_bytes()
    if digest(raw) != ref["sha256"]:
        raise ValueError("input digest changed")
    return raw


def process_snapshot(pid: int) -> dict[str, Any]:
    """Bind the real executable, argv and mapped ggml bytes, twice per arm."""
    proc = Path(f"/proc/{pid}")
    exe = (proc / "exe").resolve(strict=True)
    argv = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    stat = (proc / "stat").read_text()
    start_ticks = stat[stat.rfind(")") + 2:].split()[19]
    maps = (proc / "maps").read_text()
    library_lines = [line for line in maps.splitlines() if "libggml" in line or "libllama" in line]
    if any(line.endswith(" (deleted)") for line in library_lines):
        raise ValueError("mapped ggml/llama library was deleted")
    libraries = sorted({line.split()[-1] for line in library_lines if line.split()[-1].startswith("/")})
    return {"pid": pid, "start_ticks": start_ticks, "exe": str(exe),
            "binary_sha256": digest(exe.read_bytes()),
            "argv": [x.decode(errors="surrogateescape") for x in argv],
            "ggml": {p: digest(Path(p).read_bytes()) for p in libraries}}


def _pool(slots: Any) -> int | None:
    if not isinstance(slots, list):
        return None
    pools = [x.get("kv_pool") for x in slots if isinstance(x, dict) and isinstance(x.get("kv_pool"), dict)]
    values = [x.get("used") for x in pools]
    return values[0] if values and all(type(x) is int and x >= 0 and x == values[0] for x in values) else None


class Sampler:
    def __init__(self, url: str, interval: float):
        self.url, self.interval = url.rstrip("/"), interval
        self.samples: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready = threading.Event()

    def _sample(self) -> None:
        import httpx
        started = time.time()
        try:
            response = httpx.get(self.url + "/slots", timeout=2.0)
            response.raise_for_status()
            slots = response.json()
            self.samples.append({"t": time.time(), "request_start_s": started, "cells": _pool(slots), "slots": slots})
        except Exception as exc:  # retain failure, never substitute zero
            self.samples.append({"t": time.time(), "request_start_s": started, "cells": None, "error": f"{type(exc).__name__}: {exc}"})
        self.ready.set()

    def _run(self) -> None:
        self._sample()
        while not self.stop.wait(self.interval):
            self._sample()
        self._sample()

    def __enter__(self):
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("occupancy sampler did not start")
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("occupancy sampler did not stop")


def _slot_witness(samples: list[dict[str, Any]], timing: dict[str, Any], start: float, end: float) -> dict[str, Any] | None:
    slot, task = timing.get("id_slot"), timing.get("id_task")
    if type(slot) is not int or type(task) is not int:
        return None
    for sample in samples:
        if not start <= sample["t"] <= end:
            continue
        for row in sample.get("slots", []):
            if isinstance(row, dict) and row.get("id") == slot and row.get("id_task") == task:
                return {"sample_t": sample["t"], "slot": slot, "task": task}
    return None


def _checkpoint_witness(samples: list[dict[str, Any]], timing: dict[str, Any], start: float, end: float) -> dict[str, Any] | None:
    witness = _slot_witness(samples, timing, start, end)
    if witness is None:
        return None
    for sample in samples:
        if sample["t"] != witness["sample_t"]:
            continue
        for row in sample.get("slots", []):
            if isinstance(row, dict) and row.get("id") == witness["slot"] and row.get("id_task") == witness["task"]:
                checkpoints = row.get("checkpoints")
                if isinstance(checkpoints, list) and checkpoints:
                    return {**witness, "checkpoints": checkpoints}
    return None


def collect_task(task: dict[str, Any], request: dict[str, Any], expected: dict[str, Any],
                 runtime: dict[str, Any], base: Path, output: Path) -> dict[str, Any]:
    from src.api.routes.chat_pipeline import scout_stage as stage
    from src.repl_environment.task_root import TaskScope

    spec = request.get("spec")
    if not isinstance(spec, dict) or spec.get("max_turns") != 1 or len(spec.get("targets", [])) != task["siblings"]:
        raise ValueError("spec must declare exactly the pinned siblings and max_turns=1")
    scope = Path(request["scope_root"]).resolve(strict=True)
    files = request.get("scope_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("scope_files must pin every file in the scope")
    actual = {str(p.relative_to(scope)): digest(p.read_bytes()) for p in scope.rglob("*") if p.is_file() and not p.is_symlink()}
    if any(p.is_symlink() for p in scope.rglob("*")) or files != actual:
        raise ValueError("scope file set or digest mismatch")
    summaries = expected.get("summaries")
    if not isinstance(summaries, list) or len(summaries) != task["siblings"] or not all(isinstance(x, str) for x in summaries):
        raise ValueError("expected summaries must contain one string per scout")

    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    original_record, original_run = stage._record_scout_call, stage.run_scout

    def record(method, url, messages, max_tokens, ts_start, status, finish, timings, usage, exc, notes=None):
        with lock:
            records.append({"scout": _SCOUT_ID.get(), "method": method, "url": url,
                            "ts_start": ts_start, "ts_end": time.time(), "status": status,
                            "finish": finish, "timings": timings, "usage": usage,
                            "error": None if exc is None else repr(exc), "notes": notes})
        original_record(method, url, messages, max_tokens, ts_start, status, finish, timings, usage, exc, notes)

    def run(index, *args, **kwargs):
        token = _SCOUT_ID.set(index)
        try:
            return original_run(index, *args, **kwargs)
        finally:
            _SCOUT_ID.reset(token)

    stage._record_scout_call, stage.run_scout = record, run
    try:
        with Sampler(runtime["url"], runtime["sample_interval_s"]) as sampler:
            start = time.time()
            result = asyncio.run(stage.run_scouts(spec, scope=TaskScope(root=str(scope), edit_mode="none"),
                                                  role=runtime["role"], url=runtime["url"]))
            end = time.time()
    finally:
        stage._record_scout_call, stage.run_scout = original_record, original_run
    samples = sampler.samples
    children = []
    for r in result.results:
        calls = [x for x in records if x["scout"] == r.index and x["method"] == "scout_stage_chat_stream"]
        call = calls[0] if len(calls) == 1 else {}
        timing = call.get("timings") if isinstance(call.get("timings"), dict) else {}
        notes = call.get("notes") if isinstance(call.get("notes"), dict) else {}
        admission = notes.get("scout_admission") if isinstance(notes.get("scout_admission"), dict) else {}
        summary = r.summary
        children.append({"id": str(r.index), "status": r.status, "ttft_ms": notes.get("first_output_ms"),
                         "admission_wait_ms": admission.get("wait_ms"), "prefill_tokens": timing.get("prompt_n"),
                         "output_sha256": digest(summary.encode()) if summary is not None else None,
                         "outcome": summary == summaries[r.index] if summary is not None else None,
                         "n_fork_tokens": timing.get("n_fork_tokens"),
                         "fork_task_slot_witness": _slot_witness(samples, timing, call.get("ts_start", start), call.get("ts_end", end))})
    trunks = [x for x in records if x["method"] == "scout_stage_prefill_trunk"]
    trunk_timing = trunks[0].get("timings") if len(trunks) == 1 and isinstance(trunks[0].get("timings"), dict) else {}
    trunk = {"count": len(trunks), "checkpoint_witness": _checkpoint_witness(samples, trunk_timing, trunks[0]["ts_start"] if trunks else start, end),
             "prefill_tokens": trunk_timing.get("prompt_n")}
    raw = {"request": request, "report": result.report,
           "summaries": [r.summary for r in result.results], "calls": records, "samples": samples}
    raw_path = output / (digest(task["id"].encode()) + ".native.json")
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return {"id": task["id"], "request_sha256": task["request"]["sha256"],
            "expected_sha256": task["expected"]["sha256"], "start_s": start, "end_s": end,
            "unique_cells_samples": [{"t": x["t"], "cells": x.get("cells")} for x in samples],
            "trunk": trunk, "children": children,
            "native_path": raw_path.name, "native_sha256": digest(raw_path.read_bytes())}


def run(manifest: Path, arm: str, evidence: Path) -> int:
    raw = manifest.read_bytes()
    m = json.loads(raw)
    validate_manifest(m, manifest.parent)
    runtime = m.get("runtime", {}).get(arm)
    if not isinstance(runtime, dict) or not all(k in runtime for k in ("url", "role", "server_pid", "window_id", "sample_interval_s", "grant")):
        raise ValueError("runtime must pin server, role, window, sample interval and grant for each arm")
    interval = runtime["sample_interval_s"]
    if type(interval) not in (int, float) or not 0 < interval <= m["max_occupancy_sample_gap_s"] / 2:
        raise ValueError("sampling interval must be at most half the declared sample gap")
    grant = pinned_file(manifest.parent, runtime["grant"])
    before = process_snapshot(runtime["server_pid"])
    pins = m["pins"]
    if before["binary_sha256"] != pins["server_binary_sha256"] or " ".join(before["argv"]) != pins["argv"] or not before["ggml"]:
        raise ValueError("server binary, argv or mapped ggml identity mismatch")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if source != pins["source_commit"] or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("orchestrator source commit mismatch or tracked files dirty")
    os.environ["ORCHESTRATOR_PREFIX_INDEX"] = "1" if arm == "on" else "0"
    os.environ["ORCHESTRATOR_PREFIX_INDEX_FORK"] = "1" if arm == "on" else "0"
    os.environ["ORCHESTRATOR_PREFIX_INDEX_LPM_MAX_SKIPS"] = "1"
    flags = {"prefix_index": os.environ["ORCHESTRATOR_PREFIX_INDEX"] == "1",
             "fork": os.environ["ORCHESTRATOR_PREFIX_INDEX_FORK"] == "1", "lpm_max_skips": 1}
    out = evidence.parent / (evidence.stem + "-native")
    out.mkdir(parents=True, exist_ok=False)
    (out / "grant.json").write_bytes(grant)
    tasks = []
    for task in m["tasks"]:
        request = json.loads(pinned_file(manifest.parent, task["request"]))
        expected = json.loads(pinned_file(manifest.parent, task["expected"]))
        tasks.append(collect_task(task, request, expected, runtime, manifest.parent, out))
    after = process_snapshot(runtime["server_pid"])
    e = {"arm": arm, "manifest_sha256": digest(raw), "pins": pins, "flags": flags,
         "window_id": runtime["window_id"], "tasks": tasks, "server_before": before,
         "server_after": after, "grant_sha256": digest(grant), "grant_authority_verified": False}
    errors = validate_evidence(e, m, digest(raw), arm)
    if before != after:
        errors.append("server process or mapped library identity changed")
    # A grant digest proves file custody, not authority or active-window coverage.
    # A future owner-bound validator must establish those facts before a gate can pass.
    errors.append("grant authority and window coverage not verified")
    errors.append("server PID to listening endpoint binding not verified")
    e["collection_errors"] = errors
    evidence.write_text(json.dumps(e, indent=2, sort_keys=True) + "\n")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("off", "on"), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args.manifest.resolve(), args.arm, args.evidence.resolve())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"kpf26-scout: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
