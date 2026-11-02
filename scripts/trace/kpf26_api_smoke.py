#!/usr/bin/env python3
"""Bounded, single-window KPF-26 /chat route smoke; never asserts the paired gate."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
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


def run(packet: Path, api_url: str, api_pid: int, out: Path) -> int:
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
    out.mkdir(parents=True, exist_ok=False)
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
    result.update({"manifest_sha256": hashlib.sha256(raw).hexdigest(), "request_sha256": hashlib.sha256(request_raw).hexdigest(),
                   "response_sha256": hashlib.sha256(response_raw).hexdigest(), "http_status": response_status,
                   "start_s": start, "end_s": end, "grant": grant_before, "api": api_before,
                   "server": before, "api_url": api_url, "server_url": runtime["url"],
                   "scope": files, "flags": "unverified", "outcome": None})
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"] == "API_SCOUTS_OBSERVED" else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--api-url", required=True)
    p.add_argument("--api-pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        return run(args.packet.resolve(), args.api_url, args.api_pid, args.output.resolve())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"kpf26-api-smoke: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
