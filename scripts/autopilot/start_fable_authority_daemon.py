#!/usr/bin/env python3
"""Start AutoPilot with the Fable authority/restart environment.

This wrapper exists because the daemon's authority/tool telemetry state is
process-environment gated. A bare ``autopilot.py start`` can look healthy while
silently dropping sequential verdicts, W6 audit accrual, planner hints, or tool
sentinels from the live process.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_LOG_DIR = Path("/mnt/raid0/llm/tmp")
LIVE_PROCESS_PATTERN = "scripts/autopilot/autopilot.py start"
RUNNING_REFUSAL_EXIT = 75
REPO_READINESS_PICKUP_ENV = "AUTOPILOT_REPO_READINESS_PICKUP"
REPO_READINESS_DIR_ENV = "AUTOPILOT_REPO_READINESS_DIR"
DEFAULT_REPO_READINESS_DIRS = (
    Path("/mnt/raid0/llm/epyc-root/data/repo_readiness"),
    Path("/workspace/repos/epyc-root/data/repo_readiness"),
)
REPO_READINESS_PICKUP_GLOB = "repo_readiness_autopilot_pickup_*.json"

FABLE_AUTHORITY_ENV: dict[str, str] = {
    "AUTOPILOT_SEQ_VERDICT": "1",
    "AUTOPILOT_W6_AUDIT_BLOCK": "1",
    "AUTOPILOT_W6_AUDIT_N": "10",
    "AUTOPILOT_W6_AUDIT_EVERY_N_TRIALS": "1",
    "AUTOPILOT_W6_AUDIT_SHADOW_ONLY": "1",
    "AUTOPILOT_PLANNER_TIMEOUT": "600",
    "AUTOPILOT_PLANNER_HINTS": "1",
    "AUTOPILOT_TOOL_SENTINELS": "1",
    "AUTOPILOT_STEPPING_STONES": "1",
}


def authority_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment with required Fable authority keys enforced."""
    env = dict(os.environ if base is None else base)
    env.update(FABLE_AUTHORITY_ENV)
    if REPO_READINESS_PICKUP_ENV not in env:
        pickup = latest_repo_readiness_pickup(env)
        if pickup is not None:
            env[REPO_READINESS_PICKUP_ENV] = str(pickup)
    return env


def _repo_readiness_dirs(env: dict[str, str]) -> list[Path]:
    raw_override = env.get(REPO_READINESS_DIR_ENV, "").strip()
    if raw_override:
        return [Path(raw_override).expanduser()]
    return list(DEFAULT_REPO_READINESS_DIRS)


def latest_repo_readiness_pickup(env: dict[str, str] | None = None) -> Path | None:
    """Return the newest passive repo-readiness pickup artifact, if present."""
    env = dict(os.environ if env is None else env)
    candidates: list[Path] = []
    for data_dir in _repo_readiness_dirs(env):
        if data_dir.exists():
            candidates.extend(data_dir.glob(REPO_READINESS_PICKUP_GLOB))
    if not candidates:
        return None
    return sorted(candidates)[-1]


def python_executable() -> str:
    venv_python = ORCH_ROOT / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def build_command(max_trials: int, extra_args: list[str] | None = None) -> list[str]:
    command = [
        python_executable(),
        "scripts/autopilot/autopilot.py",
        "start",
        "--max-trials",
        str(max_trials),
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def live_autopilot_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", LIVE_PROCESS_PATTERN],
        cwd=ORCH_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _env_subset(env: dict[str, str]) -> dict[str, str]:
    keys = set(FABLE_AUTHORITY_ENV)
    keys.add(REPO_READINESS_PICKUP_ENV)
    return {key: env[key] for key in sorted(keys) if key in env}


def _payload(
    *,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    pid: int | None = None,
) -> dict[str, Any]:
    return {
        "pid": pid,
        "cwd": str(ORCH_ROOT),
        "command": command,
        "log_path": str(log_path),
        "env": _env_subset(env),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start AutoPilot with the Fable authority/tool-sentinel env."
    )
    parser.add_argument("--max-trials", type=int, default=2000)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Do not refuse when a live autopilot.py start process is detected.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command/env payload without starting a process.",
    )
    parser.add_argument(
        "autopilot_args",
        nargs=argparse.REMAINDER,
        help="Additional args appended after 'autopilot.py start --max-trials N'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env = authority_env()
    extra_args = list(args.autopilot_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command = build_command(args.max_trials, extra_args)
    log_path = args.log_dir / f"autopilot_fable_authority_{_timestamp()}.log"

    live = live_autopilot_processes()
    if live and not args.allow_existing and not args.dry_run:
        print(
            "Refusing to start another AutoPilot; live process(es):\n"
            + "\n".join(live),
            file=sys.stderr,
        )
        return RUNNING_REFUSAL_EXIT

    payload = _payload(command=command, env=env, log_path=log_path)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            command,
            cwd=ORCH_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_file.close()

    print(
        json.dumps(
            _payload(command=command, env=env, log_path=log_path, pid=proc.pid),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
