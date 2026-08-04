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
LIVE_SUPERVISOR_PATTERN = "scripts/autopilot/autopilot_supervisor.py"
RUNNING_REFUSAL_EXIT = 75
REPO_READINESS_PICKUP_ENV = "AUTOPILOT_REPO_READINESS_PICKUP"
REPO_READINESS_DIR_ENV = "AUTOPILOT_REPO_READINESS_DIR"
DEFAULT_REPO_READINESS_DIRS = (
    Path("/mnt/raid0/llm/epyc-root/data/repo_readiness"),
    Path("/workspace/repos/epyc-root/data/repo_readiness"),
)
REPO_READINESS_PICKUP_GLOB = "repo_readiness_autopilot_pickup_*.json"

FABLE_AUTHORITY_ENV: dict[str, str] = {
    # 2026-08-04 OPERATOR UNBLOCK — the sequential gate is UNREACHABLE, so leaving it
    # armed means AutoPilot cannot ratchet a baseline at all. Measured over the whole
    # journal (1,362 seq rows, 396 sequenced trials):
    #
    #     E_quality                     max  11.5507   (bar 20.0)
    #     E_rate_noninf                 max   1.1100   (bar 20.0)   <- never accumulates
    #     baseline_promotion_required_E max 100.0000   (a SECOND, 5x higher bar)
    #     confirmed                       0 of 396
    #
    # Promotion needs `E_quality >= confirm_e AND E_rate_noninf >= confirm_e`. The rate
    # e-process sits at its initial wealth of ~1.0 forever, so the conjunction is
    # unsatisfiable by construction — no configuration, however good, can promote. The
    # quality arm reached 11.55 and could never have mattered. This is SEQ-B, filed
    # 2026-07-28 in autopilot-sequential-allocation.md and open since.
    #
    # Turning it OFF drops to the legacy promotion path, which is NOT ungated: the
    # absolute quality floor, the reliability floor, same-tier regression, per-suite
    # regression, the MAD noise band, the throughput floor, the routing-diversity cap
    # and the consecutive-failure breaker all still bind. What is lost is the
    # anytime-valid confirmation on top of them — a gate that has said "no" to
    # everything for its entire life.
    #
    # RESTORE TO "1" once E_rate_noninf is fixed. The bug is in the EVIDENCE, not the
    # threshold: an e-process pinned at 1.0 is multiplying a likelihood ratio of ~1
    # every step, meaning the alternative is mis-specified or the rate statistic is not
    # being fed. Lowering confirm_e instead would let unconfirmed configs promote, which
    # is precisely what this gate exists to prevent.
    #
    # ── 2026-08-04, SEQ-B ROOT-CAUSED AND FIXED (uncommitted; awaiting operator) ──
    # The diagnosis above was right that the EVIDENCE was broken, and wrong about which
    # part. The rate statistic WAS being fed — with a mismatched pair of numbers:
    #
    #   `EvalTower._aggregate_decision_partitions` returns an EvalResult whose
    #   `n_questions` counts only the DECISION partition (55) while `eval_wall_s` is the
    #   FULL batch's wall clock (65 questions), and the incumbent comparator counted the
    #   full 65. Candidate rate = 55/wall, incumbent rate = 65/wall, on the SAME trial.
    #   An unchanged config therefore measured 0.846x its own throughput => z_rate =
    #   -0.208 every trial => `next_lambda` clipped the negative running mean to 0 =>
    #   the wealth factor became EXACTLY 1.0 and froze. That is the "likelihood ratio of
    #   ~1 every step" — not a mis-specified alternative, a mis-paired measurement.
    #
    # Fixed in tier_specs.seq_task_rate_qph_* (one measurement, both sides), a median +
    # validity-floor incumbent comparator, a skip-don't-fabricate guard for unmeasured
    # rates, and a Ville-validity repair to `rate_noninferiority_z`. NO THRESHOLD WAS
    # CHANGED. Historical replay of all 396 sequenced trials: z_rate at its clip floor
    # drops 50% -> 0%, positive-evidence trials 8% -> 69%, three candidates cross
    # E_rate = 20 and one reaches 222, ZERO false confirms (nothing confirms — the
    # QUALITY axis, max 11.55, is now the binding constraint).
    #
    # RE-ARMING IS AN OPERATOR DECISION, deliberately left un-flipped here: it changes
    # what counts as a promotion, which is human-amendment-only per MEASUREMENT.md.
    # Two coupled switches:
    #   * AUTOPILOT_SEQ_VERDICT "0" -> "1"  re-arms the sequential gate.
    #   * AUTOPILOT_SEQ_P0_2_BRIDGE "1" -> "0" makes the rate axis BINDING again
    #     (it is currently advisory, a bridge added while the axis was dead).
    "AUTOPILOT_SEQ_VERDICT": "0",
    "AUTOPILOT_SEQ_P0_2_BRIDGE": "1",
    "AUTOPILOT_W6_AUDIT_BLOCK": "1",
    "AUTOPILOT_W6_AUDIT_N": "10",
    "AUTOPILOT_W6_AUDIT_EVERY_N_TRIALS": "1",
    "AUTOPILOT_W6_AUDIT_SHADOW_ONLY": "1",
    "AUTOPILOT_PLANNER_TIMEOUT": "600",
    "AUTOPILOT_PLANNER_SPEND_BREAKER": "0",
    "AUTOPILOT_PLANNER_HINTS": "1",
    "AUTOPILOT_TOOL_SENTINELS": "1",
    "AUTOPILOT_STEPPING_STONES": "1",
}

LOCAL_PLANNER_DEFAULT_ENV: dict[str, str] = {
    "AUTOPILOT_PLANNER_PRIMARY": "claude",
    "AUTOPILOT_PLANNER_CRITIC": "codex_critic",
    "AUTOPILOT_PLANNER_CRITIC_FALLBACK": "claude",
    "AUTOPILOT_PLANNER_SPEND_BREAKER_PRIMARY": "local_frontdoor",
    "AUTOPILOT_PLANNER_SPEND_BREAKER_CRITIC": "local_ingest",
    "AUTOPILOT_LOCAL_PLANNER_ROLE": "ingest_long_context",
    "AUTOPILOT_LOCAL_PLANNER_MODEL": "ingest_long_context",
    "AUTOPILOT_LOCAL_PLANNER_TEMPERATURE": "0",
    "AUTOPILOT_LOCAL_PLANNER_MAX_TOKENS": "2048",
}


def authority_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment with required Fable authority keys enforced."""
    env = dict(os.environ if base is None else base)
    env.update(FABLE_AUTHORITY_ENV)
    for key, value in LOCAL_PLANNER_DEFAULT_ENV.items():
        env.setdefault(key, value)
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


def build_supervisor_command(
    child_command: list[str],
    *,
    max_restarts: int = 3,
    restart_delay_s: float = 30.0,
) -> list[str]:
    """Return the bounded supervisor command for a child AutoPilot command."""
    return [
        python_executable(),
        "scripts/autopilot/autopilot_supervisor.py",
        "--max-restarts",
        str(max_restarts),
        "--restart-delay-s",
        str(restart_delay_s),
        "--",
        *child_command,
    ]


def live_autopilot_processes() -> list[str]:
    live: list[str] = []
    for pattern in (LIVE_PROCESS_PATTERN, LIVE_SUPERVISOR_PATTERN):
        result = subprocess.run(
            ["pgrep", "-af", pattern],
            cwd=ORCH_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        live.extend(line.strip() for line in result.stdout.splitlines() if line.strip())
    return sorted(set(live))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _env_subset(env: dict[str, str]) -> dict[str, str]:
    keys = set(FABLE_AUTHORITY_ENV)
    keys.update(LOCAL_PLANNER_DEFAULT_ENV)
    keys.add(REPO_READINESS_PICKUP_ENV)
    return {key: env[key] for key in sorted(keys) if key in env}


def _payload(
    *,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    pid: int | None = None,
    child_command: list[str] | None = None,
    supervised: bool = False,
) -> dict[str, Any]:
    return {
        "pid": pid,
        "cwd": str(ORCH_ROOT),
        "command": command,
        "child_command": child_command or command,
        "supervised": supervised,
        "log_path": str(log_path),
        "env": _env_subset(env),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start AutoPilot with the Fable authority/tool-sentinel env."
    )
    parser.add_argument("--max-trials", type=int, default=3000)
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
        "--no-supervisor",
        action="store_true",
        help="Start autopilot.py directly without the bounded supervisor/death ledger.",
    )
    parser.add_argument("--supervisor-max-restarts", type=int, default=3)
    parser.add_argument("--supervisor-restart-delay-s", type=float, default=30.0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Print read-only stale-daemon restart advice and exit nonzero "
            "unless the live PID is age-verified against current code."
        ),
    )
    parser.add_argument(
        "autopilot_args",
        nargs=argparse.REMAINDER,
        help="Additional args appended after 'autopilot.py start --max-trials N'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight:
        from autopilot_restart_advisor import (  # type: ignore
            build_restart_advice,
        )
        from phase_status import build_phase_health_report  # type: ignore

        advice = build_restart_advice(
            build_phase_health_report(require_current_code=True),
            max_trials=args.max_trials,
        )
        print(json.dumps(advice, indent=2, sort_keys=True))
        return 0 if advice.get("pid_age_verified_landed") else 1

    env = authority_env()
    extra_args = list(args.autopilot_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    child_command = build_command(args.max_trials, extra_args)
    command = (
        child_command
        if args.no_supervisor
        else build_supervisor_command(
            child_command,
            max_restarts=args.supervisor_max_restarts,
            restart_delay_s=args.supervisor_restart_delay_s,
        )
    )
    log_path = args.log_dir / f"autopilot_fable_authority_{_timestamp()}.log"

    live = live_autopilot_processes()
    if live and not args.allow_existing and not args.dry_run:
        print(
            "Refusing to start another AutoPilot; live process(es):\n" + "\n".join(live),
            file=sys.stderr,
        )
        return RUNNING_REFUSAL_EXIT

    supervised = not args.no_supervisor
    payload = _payload(
        command=command,
        child_command=child_command,
        supervised=supervised,
        env=env,
        log_path=log_path,
    )
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
            _payload(
                command=command,
                child_command=child_command,
                supervised=supervised,
                env=env,
                log_path=log_path,
                pid=proc.pid,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
