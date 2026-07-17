#!/usr/bin/env python3
"""Bounded AutoPilot supervisor with a durable death-cause ledger."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


ORCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEATH_LEDGER_PATH = Path("/mnt/raid0/llm/tmp/autopilot_death_ledger.jsonl")

_child: subprocess.Popen[bytes] | None = None
_stopping = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_death_ledger(
    event: dict[str, Any],
    *,
    path: Path = DEFAULT_DEATH_LEDGER_PATH,
) -> None:
    """Append one supervisor event to the death-cause ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(event, sort_keys=True, default=str))
            fh.write("\n")
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _returncode_cause(returncode: int, *, stopping: bool) -> str:
    if returncode == 0:
        return "clean_exit"
    if returncode < 0:
        signum = -returncode
        try:
            label = signal.Signals(signum).name
        except ValueError:
            label = str(signum)
        return f"{'requested_' if stopping else ''}signal_{label}"
    return "nonzero_exit"


def _forward_signal(signum: int, _frame: Any) -> None:
    global _stopping
    _stopping = True
    child = _child
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        return


def supervise(
    command: list[str],
    *,
    max_restarts: int,
    restart_delay_s: float,
    death_ledger_path: Path = DEFAULT_DEATH_LEDGER_PATH,
) -> int:
    """Run command, record death cause, and restart unexpected failures."""
    global _child
    restarts = 0
    run_index = 0
    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    while True:
        run_index += 1
        started_at = time.time()
        started_at_iso = _utc_now()
        _child = subprocess.Popen(
            command,
            cwd=ORCH_ROOT,
            start_new_session=True,
        )
        returncode = _child.wait()
        ended_at = time.time()
        cause = _returncode_cause(returncode, stopping=_stopping)
        should_restart = (
            not _stopping
            and returncode != 0
            and restarts < max(0, max_restarts)
        )
        event = {
            "event": "autopilot_child_exit",
            "schema_version": 1,
            "pid": _child.pid,
            "supervisor_pid": os.getpid(),
            "run_index": run_index,
            "returncode": returncode,
            "cause": cause,
            "started_at": started_at,
            "started_at_iso": started_at_iso,
            "ended_at": ended_at,
            "ended_at_iso": _utc_now(),
            "duration_s": round(max(0.0, ended_at - started_at), 3),
            "command": command,
            "restart_scheduled": should_restart,
            "restart_count": restarts,
            "max_restarts": max_restarts,
        }
        append_death_ledger(event, path=death_ledger_path)
        if not should_restart:
            return returncode
        restarts += 1
        time.sleep(max(0.0, restart_delay_s))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--restart-delay-s", type=float, default=30.0)
    parser.add_argument("--death-ledger-path", type=Path, default=DEFAULT_DEATH_LEDGER_PATH)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return supervise(
        list(args.command),
        max_restarts=args.max_restarts,
        restart_delay_s=args.restart_delay_s,
        death_ledger_path=args.death_ledger_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
