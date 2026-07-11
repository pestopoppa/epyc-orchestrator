#!/usr/bin/env python3
"""Read-only advice for stale AutoPilot restart timing.

This complements restart_readiness_report.py. Readiness answers whether the
ledger/baseline/cutover state is internally consistent for a restart. This
advisor answers the operational question: if the live daemon is stale, is the
current phase a reasonable restart boundary or should the operator wait?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from phase_status import (  # noqa: E402
    DEFAULT_JOURNAL_DIR,
    DEFAULT_STALE_AFTER_S,
    PHASE_PATH,
    build_phase_health_report,
)

ADVISOR_VERSION = "autopilot_restart_advisor.v1"
DEFAULT_MAX_TRIALS = 3000

# Phases before a metric-bearing action starts, after a trial has fully
# completed, or while the daemon is explicitly idle/latched. Restarting here may
# lose a planner attempt, but should not corrupt an in-flight eval row.
SAFE_RESTART_PHASES = frozenset(
    {
        "stopped",
        "paused",
        "loop_start",
        "health_check",
        "health_backoff",
        "preflight",
        "observe",
        "planner_prompt_build",
        "autonomous_select",
        "max_trials_reached",
        "critic_unavailable_halt",
        "critic_reject_loop_halt",
        "planners_offline_halt",
        "meta_loop_halt",
        "skip_loop_halt",
        "async_plots_scheduled",
        "async_digest_scheduled",
    }
)

# These phases are in or immediately around action execution, safety gating,
# journal construction, artifact generation, or final state save. A stale daemon
# should be restarted after the loop reaches a later boundary.
WAIT_FOR_BOUNDARY_PHASES = frozenset(
    {
        "dispatch_action",
        "dispatch_complete",
        "dispatch_precheck_skip",
        "dispatch_skip",
        "dispatch_dry_run",
        "safety_gate",
        "self_criticism",
        "record_trial",
        "post_trial_artifacts",
        "checkpoint",
        "save_state",
        "shutting_down",
    }
)


def _stamp_landed_gate(advice: dict[str, Any]) -> dict[str, Any]:
    advice["pid_age_verified_landed"] = (
        advice.get("ok") is True
        and advice.get("status") == "no_action"
        and advice.get("restart_needed") is False
        and advice.get("pid_alive") is True
    )
    return advice


def _recommended_start_command(max_trials: int) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/autopilot/start_fable_authority_daemon.py",
        "--max-trials",
        str(max_trials),
    ]


def _phase_name(report: dict[str, Any]) -> str:
    return str(report.get("phase") or "").strip()


def _phase_is_safe_boundary(report: dict[str, Any]) -> bool:
    phase = _phase_name(report)
    if phase in SAFE_RESTART_PHASES:
        return True
    return phase.endswith(":complete")


def _phase_is_active_work(report: dict[str, Any]) -> bool:
    phase = _phase_name(report)
    if phase in WAIT_FOR_BOUNDARY_PHASES:
        return True
    idle_reason = str(report.get("idle_reason") or "").lower()
    return "evaluating question" in idle_reason or "running selected action" in idle_reason


def build_restart_advice(
    phase_report: dict[str, Any],
    *,
    max_trials: int = DEFAULT_MAX_TRIALS,
) -> dict[str, Any]:
    """Classify live AutoPilot restart timing from phase-health telemetry."""
    blockers = list(phase_report.get("blockers") or [])
    phase = _phase_name(phase_report) or None
    pid_alive = phase_report.get("pid_alive")
    code_stale = bool(phase_report.get("code_stale"))
    status = str(phase_report.get("status") or "")

    advice: dict[str, Any] = {
        "advisor_version": ADVISOR_VERSION,
        "ok": True,
        "status": "no_action",
        "restart_needed": False,
        "safe_to_restart_now": False,
        "reason": "live AutoPilot appears current",
        "phase": phase,
        "pid": phase_report.get("pid"),
        "pid_alive": pid_alive,
        "trial_id": phase_report.get("trial_id"),
        "action_type": phase_report.get("action_type"),
        "idle_reason": phase_report.get("idle_reason"),
        "code_stale": code_stale,
        "phase_health_status": status,
        "phase_health_ok": phase_report.get("ok"),
        "phase_health_blockers": blockers,
        "blockers": [],
        "stop_command": [],
        "start_command": _recommended_start_command(max_trials),
    }

    if status == "missing":
        advice.update(
            {
                "ok": False,
                "status": "manual_attention",
                "restart_needed": False,
                "reason": "phase heartbeat is missing; inspect before starting",
                "blockers": blockers or ["phase heartbeat missing"],
            }
        )
        return _stamp_landed_gate(advice)

    if pid_alive is False or phase == "stopped":
        advice.update(
            {
                "status": "restart_recommended",
                "restart_needed": True,
                "safe_to_restart_now": True,
                "reason": "AutoPilot is stopped or its heartbeat PID is dead",
            }
        )
        return _stamp_landed_gate(advice)

    if not code_stale:
        return _stamp_landed_gate(advice)

    advice["restart_needed"] = True
    if _phase_is_safe_boundary(phase_report):
        advice.update(
            {
                "status": "restart_recommended",
                "safe_to_restart_now": True,
                "reason": f"runtime code is stale and phase {phase!r} is a restart boundary",
            }
        )
    elif _phase_is_active_work(phase_report):
        advice.update(
            {
                "status": "wait_for_boundary",
                "safe_to_restart_now": False,
                "reason": f"runtime code is stale but phase {phase!r} is active work",
            }
        )
    else:
        advice.update(
            {
                "status": "restart_recommended",
                "safe_to_restart_now": True,
                "reason": (
                    "runtime code is stale and the current phase is not known "
                    "to be a metric/journal critical section"
                ),
            }
        )

    pid = advice.get("pid")
    if advice["safe_to_restart_now"] and isinstance(pid, int) and pid > 0:
        advice["stop_command"] = ["kill", "-TERM", str(pid)]
    return _stamp_landed_gate(advice)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advise whether a stale live AutoPilot should restart now or at a boundary."
    )
    parser.add_argument(
        "--phase-path",
        type=Path,
        default=PHASE_PATH,
        help=f"Phase heartbeat JSON path (default: {PHASE_PATH})",
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=DEFAULT_JOURNAL_DIR,
        help=f"AutoPilot journal shard directory (default: {DEFAULT_JOURNAL_DIR})",
    )
    parser.add_argument(
        "--stale-after-s",
        type=float,
        default=DEFAULT_STALE_AFTER_S,
        help="Heartbeat age threshold passed through to phase health.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=DEFAULT_MAX_TRIALS,
        help="Max-trials value shown in the recommended restart command.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit nonzero unless the live AutoPilot PID is age-verified "
            "against current runtime sources."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_trials < 1:
        print("--max-trials must be >= 1", file=sys.stderr)
        return 2
    if args.stale_after_s < 0:
        print("--stale-after-s must be non-negative", file=sys.stderr)
        return 2
    phase_report = build_phase_health_report(
        path=args.phase_path.expanduser().resolve(),
        journal_dir=args.journal_dir.expanduser().resolve(),
        require_current_code=True,
        stale_after_s=args.stale_after_s,
    )
    advice = build_restart_advice(phase_report, max_trials=args.max_trials)
    if args.json:
        print(json.dumps(advice, sort_keys=True, default=str))
    else:
        print(f"status: {advice['status']}")
        print(f"restart_needed: {str(advice['restart_needed']).lower()}")
        print(f"safe_to_restart_now: {str(advice['safe_to_restart_now']).lower()}")
        print(f"reason: {advice['reason']}")
        if advice["blockers"]:
            print("blockers:")
            for blocker in advice["blockers"]:
                print(f"- {blocker}")
    if args.strict and not advice.get("pid_age_verified_landed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
