#!/usr/bin/env python3
"""Bounded, single-window KPF-26 /chat route smoke; never asserts the paired gate."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trace.kpf26_paired_replay import validate_manifest  # noqa: E402
from scripts.trace.kpf26_scout_executor import (  # noqa: E402
    Sampler, active_grant, digest_file, endpoint_identity, pinned_file, process_snapshot,
    score_summary, validate_scorer, _slot_witness, _checkpoint_witness, utc,
)


def api_process(pid: int) -> dict[str, Any]:
    proc = Path(f"/proc/{pid}")
    stat = (proc / "stat").read_text()
    return {"pid": pid, "start_ticks": stat[stat.rfind(")") + 2:].split()[19],
            "exe": str((proc / "exe").resolve(strict=True)),
            "argv": (proc / "cmdline").read_bytes().rstrip(b"\0").decode(errors="surrogateescape").split("\0"),
            "cwd": str((proc / "cwd").resolve(strict=True))}


def summarize(body: Any, expected: int, role: str, url: str) -> dict[str, Any]:
    """API exposes preview only: report scout metadata; never score a truncated summary."""
    if not isinstance(body, dict) or not isinstance(body.get("scouts"), dict):
        return {"status": "INCOMPLETE", "reason": "missing scout report", "gate_pass": None}
    report = body["scouts"]
    scouts = report.get("scouts")
    reported_url = report.get("url")
    if (report.get("schema") != "epyc.orchestrator.scouts.v1" or report.get("role") != role
            or not isinstance(reported_url, str) or reported_url.rstrip("/") != url.rstrip("/")
            or report.get("requested") != expected
            or report.get("launched") != expected or report.get("completed") != expected
            or not isinstance(scouts, list) or len(scouts) != expected
            or any(not isinstance(x, dict) or x.get("status") != "ok" for x in scouts)):
        return {"status": "INCOMPLETE", "reason": "missing or incomplete scouts", "gate_pass": None}
    # A preview may be truncated. It is retained in the original API response but is never scored here.
    return {"status": "API_SCOUTS_OBSERVED", "outcome": None, "gate_pass": None,
            "reason": "single route smoke; no paired fork/checkpoint or full-summary outcome verdict"}


async def _post_bounded(url: str, request: dict[str, Any], timeout_s: float,
                        cancel: threading.Event) -> httpx.Response:
    """Closing the HTTP connection triggers /chat's disconnect cancellation watcher."""
    async with httpx.AsyncClient(trust_env=False, timeout=timeout_s) as client:
        task = asyncio.create_task(client.post(url, json=request))
        try:
            while not task.done():
                if cancel.is_set():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise RuntimeError("launcher stop/deadline while request was active")
                await asyncio.sleep(0.1)
            return await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


def _descends_from(pid: int, root_pid: int) -> bool:
    for _ in range(16):
        if pid == root_pid:
            return True
        if pid <= 1:
            return False
        stat = Path(f"/proc/{pid}/stat").read_text()
        pid = int(stat[stat.rfind(")") + 2:].split()[1])
    return False


def read_trace(directory: Path, request_id: str, api_pid: int, body: dict[str, Any],
               start: float, end: float, samples: list[dict], scorer: list[dict]) -> tuple[bytes, dict]:
    """Accept only the private exact-request capture and independently matched slot samples."""
    import stat
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.stat(f"/proc/{api_pid}").st_uid or info.st_mode & 0o077:
        raise ValueError("trace directory must be private and owned by the API user")
    path = directory / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.stat(f"/proc/{api_pid}").st_uid or info.st_mode & 0o077:
            raise ValueError("trace file must be private and owned by the API user")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            raw = stream.read(16 * 1024 * 1024 + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("trace exceeds capture limit")
    trace = json.loads(raw)
    if (not isinstance(trace, dict) or trace.get("schema") != "epyc.kpf26.api-scout-trace.v1"
            or trace.get("complete") is not True or trace.get("request_id") != request_id
            or type(trace.get("pid")) is not int or not _descends_from(trace["pid"], api_pid)
            or not start <= trace.get("start_s", 0) <= trace.get("end_s", 0) <= end
            or trace.get("report") != body.get("scouts")):
        raise ValueError("trace does not bind to this completed API request")
    summaries, calls = trace.get("summaries"), trace.get("calls")
    if not isinstance(summaries, list) or len(summaries) != len(scorer) or not isinstance(calls, list):
        raise ValueError("trace scout count or calls missing")
    children = []
    report_rows = trace["report"].get("scouts", [])
    for i, (summary, rule) in enumerate(zip(summaries, scorer, strict=True)):
        if not isinstance(summary, dict) or summary.get("index") != i or i >= len(report_rows):
            raise ValueError("trace scout index mismatch")
        text = summary.get("summary")
        if text is not None and not isinstance(text, str):
            raise ValueError("invalid summary")
        expected = hashlib.sha256(text.encode()).hexdigest() if text else None
        if report_rows[i].get("summary_sha256") != expected or report_rows[i].get("status") != summary.get("status"):
            raise ValueError("trace summary does not match API provenance")
        rows = [r for r in calls if isinstance(r, dict) and r.get("scout") == i
                and r.get("method") == "scout_stage_chat_stream"]
        call = rows[0] if len(rows) == 1 else {}
        timing = call.get("timings") if isinstance(call.get("timings"), dict) else {}
        t0, t1 = call.get("ts_start"), call.get("ts_end")
        witness = _slot_witness(samples, timing, t0, t1) if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) else None
        children.append({"index": i, "outcome": score_summary(text, rule), "summary_sha256": expected,
                         "n_fork_tokens": timing.get("n_fork_tokens"), "slot_witness": witness})
    trunks = [r for r in calls if isinstance(r, dict) and r.get("method") == "scout_stage_prefill_trunk"]
    trunk = trunks[0] if len(trunks) == 1 else {}
    tt = trunk.get("timings") if isinstance(trunk.get("timings"), dict) else {}
    t0, t1 = trunk.get("ts_start"), trunk.get("ts_end")
    checkpoint = _checkpoint_witness(samples, tt, t0, t1) if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) else None
    return raw, {"trace_sha256": hashlib.sha256(raw).hexdigest(), "children": children,
                 "checkpoint_witness": checkpoint, "flags_raw": trace.get("flags_raw"),
                 "note": "raw flags only; absent task/slot samples remain unknown"}


def write_api_observation(out: Path, result: dict[str, Any], response_raw: bytes,
                          runtime: dict[str, Any], server: dict[str, Any],
                          model_sha256: str, api_url: str) -> None:
    """Seal a completed API response at capture time; never infer fork or quality."""
    if result.get("status") != "API_SCOUTS_OBSERVED" or result.get("http_status") != 200:
        raise ValueError("completed API response required")
    body = json.loads(response_raw)
    report = body.get("scouts") if isinstance(body, dict) else None
    scouts = report.get("scouts") if isinstance(report, dict) else None
    if not isinstance(scouts, list) or not scouts or any(not isinstance(c, dict) for c in scouts):
        raise ValueError("native scout rows missing")
    completed = sum(c.get("status") == "ok" for c in scouts)
    if summarize(body, len(scouts), runtime["role"], runtime["url"])["status"] != "API_SCOUTS_OBSERVED":
        raise ValueError("native API report incomplete or scope mismatched")
    rows = out / "response.bin"
    if rows.read_bytes() != response_raw:
        raise ValueError("response bytes changed before sealing")
    response_sha = hashlib.sha256(response_raw).hexdigest()
    if result.get("response_sha256") != response_sha:
        raise ValueError("response digest mismatch")
    row = {"schema": "epyc.kpf26.api-observation.v1", "run_id": runtime["grant_id"],
           "window_id": runtime["window_id"], "host": socket.gethostname(), "api_url": api_url,
           "role": runtime["role"], "server_url": runtime["url"],
           "start_utc": utc(result["start_s"]), "end_utc": utc(result["end_s"]),
           "binary": server["exe"], "binary_realpath": server["exe"],
           "binary_sha256": server["binary_sha256"], "model_sha256": model_sha256,
           "argv": server["argv"], "prompt_set_sha256": result["manifest_sha256"],
           "request_sha256": result["request_sha256"], "grant_sha256": result["grant"]["sha256"],
           "http_status": 200, "flags": "unverified", "category": "CANDIDATE",
           "rows": str(rows), "rows_sha256": response_sha,
           "kpf26": {"tasks": 1, "children": len(scouts), "completed": completed}}
    trace = result.get("scout_trace")
    if trace is not None:
        trace_path = out / "scout-trace.json"
        if not isinstance(trace, dict) or not trace_path.is_file():
            raise ValueError("matched scout trace missing")
        trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        if trace.get("trace_sha256") != trace_sha:
            raise ValueError("matched scout trace digest mismatch")
        row["scout_trace"] = {"path": str(trace_path), "sha256": trace_sha}
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
    row["record_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    with (out / "belief_measurements.jsonl").open("x") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(packet: Path, api_url: str, api_pid: int, out: Path, trace_dir: Path | None = None) -> int:
    raw = (packet / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    validate_manifest(manifest, packet)
    request = json.loads(pinned_file(packet, manifest["api_request"]))
    runtime = manifest["runtime"]["off"]
    grant_before = active_grant(runtime["grant_id"])
    api_endpoint_before = endpoint_identity(api_pid, api_url)
    api_before = api_process(api_pid)
    server_endpoint_before = endpoint_identity(runtime["server_pid"], runtime["url"])
    before = process_snapshot(runtime["server_pid"])
    pins = manifest["pins"]
    model = Path(pins["model"]).resolve(strict=True)
    if (before["binary_sha256"] != pins["server_binary_sha256"] or
            " ".join(before["argv"]) != pins["argv"] or not before["ggml"] or
            str(model) not in before["argv"] or digest_file(model) != pins["model_sha256"]):
        raise ValueError("server/model identity mismatch")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (source != pins["source_commit"] or
            subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()):
        raise ValueError("collector source changed or dirty")
    task = manifest["tasks"][0]
    scout_request = json.loads(pinned_file(packet, task["request"]))
    scope = Path(scout_request["scope_root"]).resolve(strict=True)
    paths = list(scope.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("scope symlink refused")
    files = {str(p.relative_to(scope)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    if files != scout_request["scope_files"] or request.get("task_root") != str(scope):
        raise ValueError("scope bytes or path changed")
    deadline = min(float(os.environ["FILLER_DEADLINE"]), grant_before["end_s"]) - 15.0
    remaining = deadline - time.time()
    if remaining < 30:
        raise ValueError("grant cannot fit bounded route smoke")
    request["client_deadline_unix_s"] = deadline
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    request_raw = (json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (out / "request.json").write_bytes(request_raw)
    start = time.time()
    cancel = threading.Event()
    response_status = None
    response_raw = b""
    error = None
    with Sampler(runtime["url"], runtime["sample_interval_s"], cancel) as sampler:
        try:
            response = asyncio.run(_post_bounded(api_url.rstrip("/") + "/chat", request, remaining + 5.0, cancel))
            response_status, response_raw = response.status_code, response.content
        except (httpx.HTTPError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    end = time.time()
    (out / "response.bin").write_bytes(response_raw)
    (out / "slots.json").write_text(json.dumps(sampler.samples, sort_keys=True) + "\n")
    try:
        body = json.loads(response_raw)
    except (ValueError, UnicodeDecodeError):
        body = None
    result = summarize(body, task["siblings"], runtime["role"], runtime["url"]) if response_status == 200 else {
        "status": "INCOMPLETE", "reason": error or f"API HTTP {response_status}", "gate_pass": None}
    try:
        if (grant_before != active_grant(runtime["grant_id"]) or
                api_endpoint_before != endpoint_identity(api_pid, api_url) or api_before != api_process(api_pid) or
                server_endpoint_before != endpoint_identity(runtime["server_pid"], runtime["url"]) or
                before != process_snapshot(runtime["server_pid"]) or digest_file(model) != pins["model_sha256"]):
            raise ValueError("grant, API or target identity changed")
    except (OSError, ValueError) as exc:
        result = {"status": "INCOMPLETE", "reason": str(exc), "gate_pass": None}
    if trace_dir is not None:
        try:
            if not isinstance(body, dict) or result.get("status") != "API_SCOUTS_OBSERVED":
                raise ValueError("completed API scout report missing")
            request_id = request.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request id missing")
            expected = json.loads(pinned_file(packet, task["expected"]))
            trace_raw, trace_result = read_trace(trace_dir, request_id, api_pid, body, start, end,
                                                 sampler.samples, validate_scorer(expected, task["siblings"]))
            trace_output = out / "scout-trace.json"
            trace_output.write_bytes(trace_raw)
            os.chmod(trace_output, 0o600)
            result["scout_trace"] = trace_result
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            result = {"status": "INCOMPLETE", "reason": f"trace: {exc}", "gate_pass": None}
    result.update({"manifest_sha256": hashlib.sha256(raw).hexdigest(), "request_sha256": hashlib.sha256(request_raw).hexdigest(),
                   "response_sha256": hashlib.sha256(response_raw).hexdigest(), "http_status": response_status,
                   "start_s": start, "end_s": end, "grant": grant_before, "api": api_before,
                   "server": before, "api_url": api_url, "server_url": runtime["url"],
                   "scope": files, "flags": "unverified", "outcome": None})
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] == "API_SCOUTS_OBSERVED":
        write_api_observation(out, result, response_raw, runtime, before, pins["model_sha256"], api_url)
        return 0
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--api-url", required=True)
    p.add_argument("--api-pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trace-dir", type=Path, help="private directory configured on the API for this exact request")
    args = p.parse_args()
    try:
        return run(args.packet.resolve(), args.api_url, args.api_pid, args.output.resolve(), args.trace_dir)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"kpf26-api-smoke: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
