#!/usr/bin/env python3
"""Collect KPF-26 evidence from the actual scout stage; never infer missing witnesses.

The parent paired runner validates the result. The runtime grant and server process are
inputs, not authority conferred by this script. See KPF26_PAIRED_REPLAY.md.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trace.kpf26_paired_replay import validate_evidence, validate_manifest  # noqa: E402

_SCOUT_ID: contextvars.ContextVar[int | None] = contextvars.ContextVar("kpf26_scout_id", default=None)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


GRANTS = Path("/mnt/raid0/llm/tmp/compute-flags/grants")
_GRANT = re.compile(r"regions=([^ ]+) start=(\d{4}-\d\d-\d\dT\d\d:\d\dZ) end=(\d{4}-\d\d-\d\dT\d\d:\d\dZ)(?: holds=([^ ]*))?\n?")


def active_grant(grant_id: str) -> dict[str, Any]:
    """Require the live rowflow grant and its launcher-enforced hard end."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", grant_id):
        raise ValueError("invalid grant id")
    if os.environ.get("FILLER_ID") != grant_id:
        raise ValueError("executor is not inside the matching filler launcher")
    path = GRANTS / (grant_id + ".grant")
    if path.is_symlink() or not path.is_file():
        raise ValueError("canonical active grant absent")
    raw = path.read_bytes()
    match = _GRANT.fullmatch(raw.decode("ascii"))
    if not match:
        raise ValueError("invalid grant format")
    regions, start, end, _holds = match.groups()
    def parse(value: str) -> float:
        return dt.datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc).timestamp()

    start_s, end_s = parse(start), parse(end)
    now = time.time()
    if not start_s <= now < end_s or os.environ.get("FILLER_END") != str(int(end_s)):
        raise ValueError("grant window inactive or launcher hard end disagrees")
    stop = os.environ.get("FILLER_STOPFILE")
    if (not os.environ.get("FILLER_DEADLINE", "").isdigit()
            or not now < int(os.environ["FILLER_DEADLINE"]) <= int(end_s)):
        raise ValueError("missing or invalid launcher deadline")
    if not stop or Path(stop).exists():
        raise ValueError("missing or active launcher stop file")
    return {"grant_id": grant_id, "path": str(path), "sha256": digest(raw),
            "regions": regions, "start_s": start_s, "end_s": end_s}


def endpoint_identity(pid: int, url: str) -> dict[str, Any]:
    """Bind a local HTTP endpoint to the listener in the declared PID's fd table."""
    target = urlsplit(url)
    if target.scheme != "http" or target.hostname != "127.0.0.1" or target.port is None:
        raise ValueError("only a local explicit HTTP endpoint can be bound")
    inodes = set()
    for name in ("/proc/self/net/tcp", "/proc/self/net/tcp6"):
        for line in Path(name).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == target.port:
                inodes.add(fields[9])
    owned = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            match = re.fullmatch(r"socket:\[(\d+)\]", os.readlink(fd))
        except FileNotFoundError:
            continue
        if match:
            owned.add(match.group(1))
    if not inodes or not inodes <= owned:
        raise ValueError("endpoint listener is missing or not exclusively owned by the pinned PID")
    return {"url": url, "pid": pid, "listener_inodes": sorted(inodes)}


def utc(seconds: float) -> str:
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_belief(out: Path, tasks: list[dict[str, Any]], arm: str, manifest_hash: str,
                 runtime: dict[str, Any], server: dict[str, Any], start: float, end: float) -> None:
    """Write the observation at collection time; preserve each original native row."""
    rows = out / "belief_rows.jsonl"
    with rows.open("x") as stream:
        for task in tasks:
            stream.write(json.dumps(task, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    children = [c for task in tasks for c in task["children"]]
    row = {"schema": "epyc.kpf.run.v1", "kpf_task": "KPF-26",
           "run_id": runtime["grant_id"], "arm": "fork_on" if arm == "on" else "fork_off",
           "binary": server["exe"], "binary_realpath": server["exe"], "binary_sha256": server["binary_sha256"],
           "argv": server["argv"], "prompt_set_sha256": manifest_hash, "window_id": runtime["window_id"],
           "host": socket.gethostname(), "start_utc": utc(start), "end_utc": utc(end), "category": "CANDIDATE",
           "equivalence": {"not_measured": True}, "rows": str(rows), "rows_sha256": digest(rows.read_bytes()),
           "kpf26": {"tasks": len(tasks), "children": len(children),
                     "completed": sum(c["status"] == "ok" for c in children)},
           "ttft_ms": [c["ttft_ms"] for c in children if isinstance(c["ttft_ms"], (int, float))],
           "unique_cells_peak": max(s["cells"] for task in tasks for s in task["unique_cells_samples"]),
           "instrument_class": "serving-observation", "model_sha256": server["model_sha256"]}
    row["record_sha256"] = digest(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
    path = out / "belief_measurements.jsonl"
    with path.open("x") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _pool(slots: Any) -> int | None:
    if not isinstance(slots, list):
        return None
    pools = [x.get("kv_pool") for x in slots if isinstance(x, dict) and isinstance(x.get("kv_pool"), dict)]
    values = [x.get("used") for x in pools]
    return values[0] if values and all(type(x) is int and x >= 0 and x == values[0] for x in values) else None


class Sampler:
    def __init__(self, url: str, interval: float, cancel: threading.Event):
        self.url, self.interval, self.cancel = url.rstrip("/"), interval, cancel
        self.samples: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready = threading.Event()

    def _sample(self) -> None:
        import httpx
        started = time.time()
        try:
            response = httpx.get(self.url + "/slots", timeout=2.0, trust_env=False)
            response.raise_for_status()
            slots = response.json()
            self.samples.append({"t": time.time(), "request_start_s": started, "cells": _pool(slots), "slots": slots})
        except Exception as exc:  # retain failure, never substitute zero
            self.samples.append({"t": time.time(), "request_start_s": started, "cells": None, "error": f"{type(exc).__name__}: {exc}"})
        self.ready.set()

    def _run(self) -> None:
        self._sample()
        while not self.stop.wait(self.interval):
            if Path(os.environ["FILLER_STOPFILE"]).exists() or time.time() >= float(os.environ["FILLER_DEADLINE"]):
                self.cancel.set()
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


def validate_scorer(expected: dict[str, Any], siblings: int) -> list[dict[str, Any]]:
    """Validate the pinned scorer. Legacy exact-summary records remain supported."""
    if expected.get("schema") == "epyc.kpf26.scorer.unscored.v1":
        if set(expected) != {"schema"} or type(siblings) is not int or siblings < 1:
            raise ValueError("unscored diagnostic requires only its schema and a positive sibling count")
        return [{"mode": "unscored"} for _ in range(siblings)]
    if "summaries" in expected:
        values = expected["summaries"]
        if not isinstance(values, list) or len(values) != siblings or not all(isinstance(x, str) for x in values):
            raise ValueError("expected summaries must contain one string per scout")
        return [{"mode": "exact", "value": value} for value in values]
    values = expected.get("required_terms")
    if (expected.get("schema") != "epyc.kpf26.scorer.required-terms.v1" or
            not isinstance(values, list) or len(values) != siblings or
            not all(isinstance(row, list) and row and all(isinstance(x, str) and x.strip() for x in row)
                    for row in values)):
        raise ValueError("required_terms must contain a nonempty string list per scout")
    return [{"mode": "required_terms", "value": row} for row in values]


def score_summary(summary: str | None, rule: dict[str, Any]) -> bool | None:
    if summary is None or rule["mode"] == "unscored":
        return None
    if rule["mode"] == "exact":
        return summary == rule["value"]
    return all(term.casefold() in summary.casefold() for term in rule["value"])


def collect_task(task: dict[str, Any], request: dict[str, Any], expected: dict[str, Any],
                 runtime: dict[str, Any], base: Path, output: Path) -> dict[str, Any]:
    import httpx
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
    scorer = validate_scorer(expected, task["siblings"])

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
        cancel = threading.Event()
        with Sampler(runtime["url"], runtime["sample_interval_s"], cancel) as sampler:
            start = time.time()
            with httpx.Client(trust_env=False) as client:
                transport = stage.ChatCompletionsTransport(runtime["url"], client=client)
                result = asyncio.run(stage.run_scouts(spec, scope=TaskScope(root=str(scope), edit_mode="none"),
                                                      role=runtime["role"], url=runtime["url"], transport=transport,
                                                      cancel_event=cancel, request_deadline_s=time.perf_counter() +
                                                      max(0, float(os.environ["FILLER_DEADLINE"]) - time.time())))
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
                         "outcome": score_summary(summary, scorer[r.index]),
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
    if not isinstance(runtime, dict) or not all(k in runtime for k in ("url", "role", "server_pid", "window_id", "sample_interval_s", "grant_id")):
        raise ValueError("runtime must pin server, role, window, sample interval and grant for each arm")
    interval = runtime["sample_interval_s"]
    if type(interval) not in (int, float) or not 0 < interval <= m["max_occupancy_sample_gap_s"] / 2:
        raise ValueError("sampling interval must be at most half the declared sample gap")
    grant_before = active_grant(runtime["grant_id"])
    endpoint_before = endpoint_identity(runtime["server_pid"], runtime["url"])
    before = process_snapshot(runtime["server_pid"])
    started = time.time()
    pins = m["pins"]
    model = Path(pins["model"]).resolve(strict=True)
    model_sha = pins.get("model_sha256")
    if not isinstance(model_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha) or digest_file(model) != model_sha:
        raise ValueError("model full-file digest missing or mismatched")
    if before["binary_sha256"] != pins["server_binary_sha256"] or " ".join(before["argv"]) != pins["argv"] or not before["ggml"] or pins["model"] not in before["argv"]:
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
    (out / "grant.txt").write_bytes(Path(grant_before["path"]).read_bytes())
    tasks = []
    for task in m["tasks"]:
        request = json.loads(pinned_file(manifest.parent, task["request"]))
        expected = json.loads(pinned_file(manifest.parent, task["expected"]))
        active_grant(runtime["grant_id"])
        tasks.append(collect_task(task, request, expected, runtime, manifest.parent, out))
    ended = time.time()
    if digest_file(model) != model_sha:
        raise ValueError("model bytes changed during collection")
    grant_after = active_grant(runtime["grant_id"])
    endpoint_after = endpoint_identity(runtime["server_pid"], runtime["url"])
    after = process_snapshot(runtime["server_pid"])
    e = {"arm": arm, "manifest_sha256": digest(raw), "pins": pins, "flags": flags,
         "window_id": runtime["window_id"], "tasks": tasks, "server_before": before,
         "server_after": after, "grant": grant_before, "grant_authority_verified": True,
         "endpoint": endpoint_before}
    errors = validate_evidence(e, m, digest(raw), arm)
    if before != after:
        errors.append("server process or mapped library identity changed")
    if grant_before != grant_after:
        errors.append("grant changed during collection")
    if endpoint_before != endpoint_after:
        errors.append("server endpoint listener changed during collection")
    e["collection_errors"] = errors
    if not errors:
        before["model_sha256"] = model_sha
        write_belief(out, tasks, arm, digest(raw), runtime, before, started, ended)
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
