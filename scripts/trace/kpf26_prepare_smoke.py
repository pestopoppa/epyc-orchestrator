#!/usr/bin/env python3
"""Freeze a bounded prospective scout smoke from the retained HS-19a task.

This creates a packet only; it never loads a model or creates a grant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trace.kpf26_scout_executor import digest_file, endpoint_identity, process_snapshot  # noqa: E402
from scripts.trace.kpf26_paired_replay import validate_manifest  # noqa: E402

FIXTURE = ROOT / "scripts/trace/fixtures/kpf26_hs19a"
EXECUTOR = ROOT / "scripts/trace/kpf26_scout_executor.py"
RUNNER = ROOT / "scripts/trace/kpf26_paired_replay.py"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> dict[str, str]:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return {"path": path.name, "sha256": sha(data)}


def prepare(output: Path, *, pid: int, url: str, role: str, model: Path, grant: str,
            source: str | None = None, snapshot: dict | None = None, verify_endpoint: bool = True) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", grant):
        raise ValueError("invalid grant id")
    if not role or pid <= 0:
        raise ValueError("invalid role or server PID")
    if output.exists():
        raise ValueError("output already exists")
    model = model.resolve(strict=True)
    source = source or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("source commit must be a full SHA-1")
    if snapshot is None:
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
            raise ValueError("tracked orchestrator checkout is dirty")
        snapshot = process_snapshot(pid)
    if verify_endpoint:
        endpoint_identity(pid, url)
    if str(model) not in snapshot["argv"] or not snapshot.get("ggml"):
        raise ValueError("server model or mapped ggml identity mismatch")
    raw = (FIXTURE / "scope/config/release.toml").read_bytes()
    original = (FIXTURE / "original-prompt.txt").read_bytes()
    if b'RELEASE_CODENAME = "amber-heron-47"' not in raw or b"RELEASE_CODENAME" not in original:
        raise ValueError("retained HS-19a source fixture does not match task")
    # Packets have a fresh, isolated, immutable-by-hash scope and refuse overwrite.
    output.mkdir(parents=True, exist_ok=False)
    scope = output / "scope"
    (scope / "config").mkdir(parents=True)
    (scope / "config/release.toml").write_bytes(raw)
    (output / "original-prompt.txt").write_bytes(original)
    request = {"scope_root": str(scope.resolve()), "scope_files": {"config/release.toml": sha(raw)},
               "spec": {"enabled": True, "max": 2, "max_turns": 1, "budget_s": 600,
                        "reserve_slots": 1, "summary_tokens": 400,
                        "targets": [{"file": "config/release.toml", "label": label}
                                    for label in ("Find the release codename", "Verify the release codename")]}}
    expected = {"schema": "epyc.kpf26.scorer.required-terms.v1",
                "required_terms": [["RELEASE_CODENAME", "amber-heron-47"]] * 2}
    request_ref = write_json(output / "request.json", request)
    expected_ref = write_json(output / "expected.json", expected)
    api_request = {"request_id": "kpf26-" + grant + "-" + secrets.token_hex(16),
                   "prompt": "Summarize the release metadata from the scoped repository.",
                   "real_mode": True, "mock_mode": False, "force_mode": "direct", "force_role": role,
                   "allow_delegation": False, "task_root": str(scope.resolve()), "scouts": request["spec"],
                   "max_turns": 1, "max_tokens": 256, "timeout_s": 600, "x_request_source": "smoke"}
    api_ref = write_json(output / "api-request.json", api_request)
    runtime = {"url": url, "role": role, "server_pid": pid, "sample_interval_s": 0.5, "grant_id": grant}
    manifest = {"schema": "epyc.kpf26.paired.v1", "arm_order": ["off", "on"], "arm_timeout_s": 900,
                "max_occupancy_sample_gap_s": 2,
                "pins": {"model": str(model), "model_sha256": digest_file(model),
                         "server_binary_sha256": snapshot["binary_sha256"], "source_commit": source,
                         "argv": " ".join(snapshot["argv"]), "executor_sha256": digest_file(EXECUTOR)},
                "runtime": {arm: {**runtime, "window_id": f"{grant}-{arm}"} for arm in ("off", "on")},
                "tasks": [{"id": "hs19a-release-codename-scout-smoke", "siblings": 2,
                           "request": request_ref, "expected": expected_ref}],
                "api_request": api_ref,
                "provenance": {"kind": "prospective-scout-smoke-derived-from-hs19a",
                               "source_prompt_sha256": sha(original), "scope_sha256": sha(raw),
                               "source": "HS-19a live acceptance fixture 2026-09-27",
                               "interpretation": "Two new scouts derived from the original one-subagent task; not a historical scout replay or decision-grade gate."}}
    validate_manifest(manifest, output)
    write_json(output / "manifest.json", manifest)
    (output / "README.txt").write_text(
        "Prospective smoke from retained HS-19a inputs; not the original scout cohort or KPF-26 gate.\n"
        "Run only within the matching active grant on the exact pinned server. This packet never starts or reloads a server.\n"
        f"python3 {RUNNER} --manifest {output.resolve() / 'manifest.json'} --executor {EXECUTOR} --output {output.resolve() / 'result'}\n")
    return output / "manifest.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server-pid", type=int, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--grant-id", required=True)
    args = p.parse_args()
    try:
        print(prepare(args.output.resolve(), pid=args.server_pid, url=args.url, role=args.role,
                      model=args.model, grant=args.grant_id))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"kpf26-prepare: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
