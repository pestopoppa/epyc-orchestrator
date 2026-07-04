#!/usr/bin/env python3
"""Run or plan the A7 eval-batch serving activation window.

The eval-batch serving lane is intentionally default-off.  This wrapper turns
the handoff's manual activation sequence into one guarded operation:

1. start the warm ``eval_batch_frontdoor`` auxiliary server,
2. reload the orchestrator API with eval-batch serving enabled,
3. run the existing smoke probe and verify tap attribution,
4. roll back automatically unless ``--keep-enabled`` is explicit.

By default this script is plan-only and writes the exact commands it would run.
Live mutation requires both ``--apply`` and ``--confirm-clean-window``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_EVAL_BATCH_URL = "http://localhost:18070"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "orchestration" / "reports"
PROBE_PATH = PROJECT_ROOT / "scripts" / "benchmark" / "eval_batch_serving_probe.py"


@dataclass(frozen=True)
class PlannedCommand:
    name: str
    argv: list[str]
    env: dict[str, str] | None = None

    def display(self) -> str:
        env_prefix = ""
        if self.env:
            env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env.items())
            env_prefix += " "
        return env_prefix + " ".join(shlex.quote(part) for part in self.argv)


@dataclass
class StepResult:
    name: str
    command: str
    returncode: int
    elapsed_s: float
    stdout_tail: str
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_probe_module() -> Any:
    spec = importlib.util.spec_from_file_location("eval_batch_serving_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load probe module from {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_batch_serving_probe"] = module
    spec.loader.exec_module(module)
    return module


def _python() -> str:
    return sys.executable


def start_command(*, mode: str) -> PlannedCommand:
    if mode == "include-warm":
        argv = [
            _python(),
            "scripts/server/orchestrator_stack.py",
            "start",
            "--include-warm",
            "eval_batch_frontdoor",
        ]
    else:
        argv = [
            _python(),
            "scripts/server/orchestrator_stack.py",
            "start",
            "--only",
            "eval_batch_frontdoor",
        ]
    return PlannedCommand("start_eval_batch_frontdoor", argv)


def enable_reload_command(eval_batch_url: str) -> PlannedCommand:
    return PlannedCommand(
        "reload_orchestrator_eval_batch_enabled",
        [_python(), "scripts/server/orchestrator_stack.py", "reload", "orchestrator"],
        env={
            "ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING": "1",
            "ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL": eval_batch_url.rstrip("/"),
        },
    )


def smoke_command(args: argparse.Namespace, output_dir: Path) -> PlannedCommand:
    argv = [
        _python(),
        "scripts/benchmark/eval_batch_serving_probe.py",
        "--smoke",
        "--confirm-clean-window",
        "--require-enabled",
        "--api-url",
        args.api_url.rstrip("/"),
        "--eval-batch-url",
        args.eval_batch_url.rstrip("/"),
        "--output-dir",
        str(output_dir),
        "--summary-only",
    ]
    if args.allow_autopilot_active:
        argv.append("--allow-autopilot-active")
    if args.tap_events:
        argv.extend(["--tap-events", args.tap_events])
    return PlannedCommand("smoke_probe", argv)


def rollback_commands() -> list[PlannedCommand]:
    return [
        PlannedCommand(
            "rollback_reload_orchestrator_eval_batch_disabled",
            [_python(), "scripts/server/orchestrator_stack.py", "reload", "orchestrator"],
            env={"ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING": "0"},
        ),
        PlannedCommand(
            "rollback_stop_eval_batch_frontdoor",
            [_python(), "scripts/server/orchestrator_stack.py", "stop", "eval_batch_frontdoor"],
        ),
    ]


def activation_plan(args: argparse.Namespace, *, output_dir: Path) -> dict[str, list[PlannedCommand]]:
    smoke_dir = output_dir / "smoke_probe"
    return {
        "activation": [
            start_command(mode=args.start_mode),
            enable_reload_command(args.eval_batch_url),
            smoke_command(args, smoke_dir),
        ],
        "rollback": rollback_commands(),
    }


def _probe_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        api_url=args.api_url.rstrip("/"),
        eval_batch_url=args.eval_batch_url.rstrip("/"),
        http_timeout_s=args.http_timeout_s,
        attest_samples=args.attest_samples,
    )


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    probe = _load_probe_module()
    return probe.build_preflight(_probe_namespace(args))


def pre_apply_blockers(args: argparse.Namespace, preflight: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not args.confirm_clean_window:
        blockers.append("--apply requires --confirm-clean-window")
    if not (preflight.get("api_health") or {}).get("ok"):
        blockers.append("orchestrator API health is not OK")
    if preflight.get("autopilot_active") and not args.allow_autopilot_active:
        blockers.append("AutoPilot appears active; activation would contaminate live eval resources")
    return blockers


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def run_command(command: PlannedCommand, *, timeout_s: float) -> StepResult:
    env = os.environ.copy()
    if command.env:
        env.update(command.env)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command.argv,
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return StepResult(
            name=command.name,
            command=command.display(),
            returncode=completed.returncode,
            elapsed_s=time.perf_counter() - started,
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        return StepResult(
            name=command.name,
            command=command.display(),
            returncode=124,
            elapsed_s=time.perf_counter() - started,
            stdout_tail=_tail(exc.stdout or ""),
            stderr_tail=_tail(exc.stderr or f"timed out after {timeout_s}s"),
        )


def _load_probe_summary(output_dir: Path) -> dict[str, Any] | None:
    summary_path = output_dir / "smoke_probe" / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"invalid JSON in {summary_path}"}


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Eval-Batch Serving Activation Window",
        "",
        f"- status: `{report['status']}`",
        f"- decision_grade: `{report['decision_grade']}`",
        f"- applied: `{report['applied']}`",
        f"- keep_enabled: `{report['keep_enabled']}`",
        f"- eval_batch_url: `{report['eval_batch_url']}`",
        f"- autopilot_active: `{report['preflight'].get('autopilot_active')}`",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    lines.extend(["", "## Activation Plan", ""])
    lines.extend(f"```bash\n{cmd}\n```" for cmd in report["activation_commands"])
    lines.extend(["", "## Rollback Plan", ""])
    lines.extend(f"```bash\n{cmd}\n```" for cmd in report["rollback_commands"])
    if report.get("steps"):
        lines.extend(["", "## Steps", ""])
        for step in report["steps"]:
            lines.append(
                f"- `{step['name']}` rc=`{step['returncode']}` elapsed_s=`{step['elapsed_s']:.3f}`"
            )
    probe_summary = report.get("probe_summary")
    if isinstance(probe_summary, dict):
        lines.extend(
            [
                "",
                "## Smoke Probe",
                "",
                f"- status: `{probe_summary.get('status')}`",
                f"- decision_grade: `{probe_summary.get('decision_grade')}`",
                f"- blockers: `{probe_summary.get('blockers')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    return json_path, md_path


def execute_activation(args: argparse.Namespace, *, output_dir: Path) -> tuple[list[StepResult], list[str]]:
    plan = activation_plan(args, output_dir=output_dir)
    steps: list[StepResult] = []
    errors: list[str] = []
    for command in plan["activation"]:
        result = run_command(command, timeout_s=args.command_timeout_s)
        steps.append(result)
        if not result.ok:
            errors.append(f"{command.name} failed with rc={result.returncode}")
            break
    return steps, errors


def execute_rollback(args: argparse.Namespace) -> list[StepResult]:
    return [
        run_command(command, timeout_s=args.command_timeout_s)
        for command in rollback_commands()
    ]


def build_report(args: argparse.Namespace, *, output_dir: Path) -> tuple[dict[str, Any], int]:
    preflight = build_preflight(args)
    plan = activation_plan(args, output_dir=output_dir)
    blockers = pre_apply_blockers(args, preflight) if args.apply else []
    steps: list[StepResult] = []
    rollback_steps: list[StepResult] = []
    probe_summary: dict[str, Any] | None = None
    status = "plan_only"
    rc = 0

    if args.apply and blockers:
        status = "blocked"
        rc = 75 if any("AutoPilot appears active" in item for item in blockers) else 2
    elif args.apply:
        steps, errors = execute_activation(args, output_dir=output_dir)
        probe_summary = _load_probe_summary(output_dir)
        probe_passed = bool(probe_summary and probe_summary.get("decision_grade"))
        if errors or not probe_passed:
            status = "activation_failed"
            blockers.extend(errors or ["smoke probe did not produce decision-grade evidence"])
            rc = 75
        elif args.keep_enabled:
            status = "smoke_passed_enabled_left_on"
            rc = 0
        else:
            rollback_steps = execute_rollback(args)
            failed_rollback = [step for step in rollback_steps if not step.ok]
            if failed_rollback:
                status = "smoke_passed_rollback_failed"
                blockers.extend(f"rollback step {step.name} failed" for step in failed_rollback)
                rc = 75
            else:
                status = "smoke_passed_rolled_back"
                rc = 0

        if status == "activation_failed" and not args.skip_rollback:
            rollback_steps = execute_rollback(args)

    decision_grade = bool(probe_summary and probe_summary.get("decision_grade"))
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "decision_grade": decision_grade,
        "applied": bool(args.apply),
        "keep_enabled": bool(args.keep_enabled),
        "eval_batch_url": args.eval_batch_url.rstrip("/"),
        "start_mode": args.start_mode,
        "blockers": blockers,
        "preflight": preflight,
        "activation_commands": [command.display() for command in plan["activation"]],
        "rollback_commands": [command.display() for command in plan["rollback"]],
        "steps": [asdict(step) for step in steps],
        "rollback_steps": [asdict(step) for step in rollback_steps],
        "probe_summary": probe_summary,
    }
    return report, rc


def default_output_dir(stamp: str | None = None) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"eval_batch_serving_activation_{stamp or utc_stamp()}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.environ.get("ORCHESTRATOR_API_URL", DEFAULT_API_URL))
    parser.add_argument(
        "--eval-batch-url",
        default=os.environ.get("ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL", DEFAULT_EVAL_BATCH_URL),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-clean-window", action="store_true")
    parser.add_argument("--allow-autopilot-active", action="store_true")
    parser.add_argument("--keep-enabled", action="store_true")
    parser.add_argument("--skip-rollback", action="store_true")
    parser.add_argument("--tap-events", default=None)
    parser.add_argument("--attest-samples", type=int, default=12)
    parser.add_argument("--http-timeout-s", type=float, default=5.0)
    parser.add_argument("--command-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--start-mode",
        choices=("only", "include-warm"),
        default="only",
        help=(
            "Use 'only' to start just eval_batch_frontdoor in an already-live stack. "
            "Use 'include-warm' for the historical full-stack start command."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or default_output_dir()
    report, rc = build_report(args, output_dir=output_dir)
    json_path, md_path = write_report(report, output_dir)
    if not args.summary_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {json_path}")
        print(f"wrote {md_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
