#!/usr/bin/env python3
"""Execute live A9 audit-target seeding commands in a quiet window.

This helper reads the audit-target collection manifest used by
collect_offline_reward_pairwise_audit_target.sh, removes ``--dry-run`` from each
seed_specialist_routing command, and runs the four batches in sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ORCH_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_MANIFEST = (
    ORCH_ROOT
    / "orchestration"
    / "reports"
    / "offline_reward_oracle_token_coverage_final_labels_20260621"
    / "offline_reward_pairwise_audit_target_collection_manifest.json"
)
AUTOPILOT_PATTERN = "scripts/autopilot/autopilot.py start"
TIMESTAMP_DEFAULT_ENV = "A9_COLLECTION_TIMESTAMP"
TIMESTAMP_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
TIMESTAMP_PLACEHOLDER = "<YYYYMMDDTHHMMSSZ>"
REFUSAL_EXIT_CODE = 75
INVALID_TIMESTAMP_EXIT_CODE = 64


def _now_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_timestamp(run_ts: str) -> None:
    if not TIMESTAMP_RE.fullmatch(run_ts):
        raise ValueError(f"invalid A9 collection timestamp: {run_ts}")


def _active_autopilot_processes(pattern: str = AUTOPILOT_PATTERN) -> list[str]:
    proc = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        stdout = proc.stdout or ""
        return [line.strip() for line in stdout.splitlines() if line.strip()]
    if proc.returncode == 1:
        return []
    raise RuntimeError(f"AutoPilot process probe failed with exit {proc.returncode}")


def _remove_dry_run(tokens: list[str], *, command: str) -> list[str]:
    if "--dry-run" not in tokens:
        raise ValueError(f"command is not dry-run guarded: {command}")
    return [token for token in tokens if token != "--dry-run"]


def build_live_batch_commands(
    manifest_path: Path, *, timestamp: str
) -> list[tuple[str, list[str], Path]]:
    _validate_timestamp(timestamp)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    batches = payload.get("batches")
    if not isinstance(batches, list):
        raise ValueError("collection manifest batches missing or invalid")
    if payload.get("batch_count") != len(batches):
        raise ValueError("collection manifest batch_count does not match runnable batches")

    out: list[tuple[str, list[str], Path]] = []
    for batch in batches:
        if not isinstance(batch, dict):
            raise ValueError("collection manifest batch is not an object")
        target = str(batch.get("target") or "batch")
        raw_command = str(batch.get("command") or "").replace(TIMESTAMP_PLACEHOLDER, timestamp)
        if "seed_specialist_routing.py" not in raw_command:
            raise ValueError(f"{target}: command is not a seed_specialist_routing invocation")

        tokens = shlex.split(raw_command)
        live_tokens = _remove_dry_run(tokens, command=raw_command)
        if "--output" not in live_tokens:
            raise ValueError(f"{target}: command is missing --output")
        output_index = live_tokens.index("--output")
        if output_index + 1 >= len(live_tokens):
            raise ValueError(f"{target}: command has malformed --output")
        output = Path(live_tokens[output_index + 1])

        out.append((target, live_tokens, output))
    return out


def run_live_collection(
    *,
    manifest_path: Path,
    timestamp: str,
) -> int:
    run_ts = timestamp
    _validate_timestamp(run_ts)

    active = _active_autopilot_processes()
    if active:
        print("refusing A9 collection while AutoPilot is active", file=sys.stderr)
        return REFUSAL_EXIT_CODE

    try:
        batches = build_live_batch_commands(manifest_path, timestamp=run_ts)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"failed to read A9 collection manifest: {exc}", file=sys.stderr)
        return 2

    for index, (target, command, output_path) in enumerate(batches, start=1):
        print(f"A9 collection batch {index}/{len(batches)}: {target}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, check=False, cwd=str(ORCH_ROOT))
        if result.returncode != 0:
            return result.returncode
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Collection manifest with dry-run A9 batch commands.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # keep timestamp behavior aligned with the shell script:
    # A9_COLLECTION_TIMESTAMP overrides default UTC timestamp.
    run_ts = os.environ.get(TIMESTAMP_DEFAULT_ENV, _now_timestamp())

    try:
        _validate_timestamp(run_ts)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return INVALID_TIMESTAMP_EXIT_CODE

    return run_live_collection(manifest_path=args.manifest, timestamp=run_ts)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
