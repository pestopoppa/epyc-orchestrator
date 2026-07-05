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
    DEFAULT_AUTOPILOT_LOG_PATH,
    DEFAULT_JOURNAL_DIR,
    DEFAULT_OUTCOME_STALL_FRONTIER_TRIALS,
    DEFAULT_OUTCOME_STALL_PROMOTION_TRIALS,
    DEFAULT_OUTCOME_RECENT_WINDOW_TRIALS,
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
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help=(
            "AutoPilot log path used to fill missing in-flight eval counters "
            f"(default for the live phase path: {DEFAULT_AUTOPILOT_LOG_PATH})"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when the heartbeat is missing, dead, or stale.",
    )
    parser.add_argument(
        "--require-current-code",
        action="store_true",
        help=(
            "Also fail strict checks when the live AutoPilot process predates "
            "runtime AutoPilot source changes."
        ),
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=DEFAULT_JOURNAL_DIR,
        help=f"AutoPilot journal shard directory (default: {DEFAULT_JOURNAL_DIR})",
    )
    parser.add_argument(
        "--require-outcome-progress",
        action="store_true",
        help=(
            "Also fail strict checks when journal-derived frontier/baseline "
            "promotion progress is stale."
        ),
    )
    parser.add_argument(
        "--max-trials-since-frontier",
        type=int,
        default=DEFAULT_OUTCOME_STALL_FRONTIER_TRIALS,
        help=(
            "Outcome-progress threshold for trials since the latest frontier "
            "admission."
        ),
    )
    parser.add_argument(
        "--max-trials-since-promotion",
        type=int,
        default=DEFAULT_OUTCOME_STALL_PROMOTION_TRIALS,
        help=(
            "Outcome-progress threshold for trials since the latest baseline "
            "promotion."
        ),
    )
    parser.add_argument(
        "--recent-window-trials",
        type=int,
        default=DEFAULT_OUTCOME_RECENT_WINDOW_TRIALS,
        help="Recent trial window for keepable/wasted-eval/learning-excluded rates.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        log_path = args.log_path.expanduser().resolve() if args.log_path else None
        report = build_phase_health_report(
            path=args.phase_path.expanduser().resolve(),
            log_path=log_path,
            journal_dir=args.journal_dir.expanduser().resolve(),
            require_current_code=args.require_current_code,
            require_outcome_progress=args.require_outcome_progress,
            max_trials_since_frontier=args.max_trials_since_frontier,
            max_trials_since_promotion=args.max_trials_since_promotion,
            recent_window_trials=args.recent_window_trials,
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
