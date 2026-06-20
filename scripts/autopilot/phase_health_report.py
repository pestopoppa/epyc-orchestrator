#!/usr/bin/env python3
"""Read-only AutoPilot phase heartbeat health report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from phase_status import (  # noqa: E402
    DEFAULT_STALE_AFTER_S,
    PHASE_PATH,
    build_phase_health_report,
    format_phase_health_report,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report AutoPilot phase-heartbeat liveness without writing state."
    )
    parser.add_argument(
        "--phase-path",
        type=Path,
        default=PHASE_PATH,
        help=f"Phase heartbeat JSON path (default: {PHASE_PATH})",
    )
    parser.add_argument(
        "--stale-after-s",
        type=float,
        default=DEFAULT_STALE_AFTER_S,
        help="Heartbeat age threshold that marks an active phase stale.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when the heartbeat is missing, dead, or stale.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_phase_health_report(
            path=args.phase_path.expanduser().resolve(),
            stale_after_s=args.stale_after_s,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print("\n".join(format_phase_health_report(report)))

    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
