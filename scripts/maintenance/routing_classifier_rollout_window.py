#!/usr/bin/env python3
"""Run or plan the N9 routing-classifier rollout window.

The routing classifier weights are staged, but the production feature remains
default-off. This wrapper turns the manual rollout sequence into one guarded,
auditable operation:

1. run the existing offline routing-wiring verifier,
2. reload the orchestrator API with ``routing_classifier`` enabled,
3. attest that sampled API workers agree on the feature flag,
4. roll back automatically unless ``--keep-enabled`` is explicit.

Default mode is plan-only. Live mutation requires both ``--apply`` and
``--confirm-clean-window``. The script sends no model prompts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "orchestration" / "reports"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "orchestration/repl_memory/routing_classifier_weights.npz"


@dataclass(frozen=True)
class PlannedCommand:
    name: str
    argv: list[str]
    env: dict[str, str] | None = None

    def display(self) -> str:
        env_prefix = ""
        if self.env:
            env_prefix = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in self.env.items()
            )
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


@dataclass
class HttpResult:
    url: str
    status: int | None
    ok: bool
    elapsed_s: float
    json_body: Any = None
    error: str | None = None


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def default_output_dir(stamp: str | None = None) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"routing_classifier_rollout_{stamp or utc_stamp()}"


def _python() -> str:
    return sys.executable


def _active_autopilot() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "scripts/autopilot/autopilot.py start"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _request_json(method: str, url: str, *, timeout_s: float) -> HttpResult:
    req = request.Request(url, method=method)
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = {"raw": raw[:1000]}
            status = int(getattr(resp, "status", 0))
            return HttpResult(
                url=url,
                status=status,
                ok=200 <= status < 300,
                elapsed_s=time.perf_counter() - started,
                json_body=parsed,
            )
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return HttpResult(
            url=url,
            status=int(exc.code),
            ok=False,
            elapsed_s=time.perf_counter() - started,
            error=raw[:1000] or str(exc),
        )
    except OSError as exc:
        return HttpResult(
            url=url,
            status=None,
            ok=False,
            elapsed_s=time.perf_counter() - started,
            error=str(exc),
        )


def _collect_config_attest(api_url: str, *, samples: int, timeout_s: float) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for _ in range(max(1, samples)):
        result = _request_json(
            "GET",
            f"{api_url.rstrip('/')}/config/attest",
            timeout_s=timeout_s,
        )
        body = result.json_body
        if isinstance(body, dict):
            pid = body.get("pid")
            if isinstance(pid, int):
                seen[pid] = {
                    "pid": pid,
                    "flags": body.get("flags") if isinstance(body.get("flags"), dict) else {},
                    "sources": body.get("sources") if isinstance(body.get("sources"), dict) else {},
                    "status": result.status,
                    "ok": result.ok,
                }
        time.sleep(0.05)
    return list(seen.values())


def _routing_flag_by_pid(attest_rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        str(row["pid"]): bool((row.get("flags") or {}).get("routing_classifier"))
        for row in attest_rows
        if isinstance(row.get("pid"), int)
    }


def _routing_flag_sources_by_pid(attest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(row["pid"]): (row.get("sources") or {}).get("routing_classifier")
        for row in attest_rows
        if isinstance(row.get("pid"), int)
    }


def verifier_command() -> PlannedCommand:
    return PlannedCommand(
        "verify_routing_wiring",
        [_python(), "scripts/maintenance/verify_routing_wiring.py"],
    )


def reload_command(*, enabled: bool, weights_path: Path) -> PlannedCommand:
    return PlannedCommand(
        "reload_orchestrator_routing_classifier_enabled"
        if enabled
        else "reload_orchestrator_routing_classifier_disabled",
        [_python(), "scripts/server/orchestrator_stack.py", "reload", "orchestrator"],
        env={
            "ORCHESTRATOR_FEATURE_ROUTING_CLASSIFIER": "1" if enabled else "0",
            "ROUTING_CLASSIFIER_WEIGHTS": str(weights_path),
        },
    )


def attest_command(args: argparse.Namespace, *, enabled: bool) -> PlannedCommand:
    return PlannedCommand(
        "attest_routing_classifier_enabled" if enabled else "attest_routing_classifier_disabled",
        [
            _python(),
            "scripts/validate/attest_flags.py",
            "--url",
            args.api_url.rstrip("/"),
            "--polls",
            str(args.attest_polls),
            "--delay-s",
            str(args.attest_delay_s),
            "--min-workers",
            str(args.min_workers),
            "--expect",
            f"routing_classifier={'true' if enabled else 'false'}",
        ],
    )


def rollout_plan(args: argparse.Namespace) -> dict[str, list[PlannedCommand]]:
    weights_path = Path(args.weights)
    return {
        "activation": [
            verifier_command(),
            reload_command(enabled=True, weights_path=weights_path),
            attest_command(args, enabled=True),
        ],
        "rollback": [
            reload_command(enabled=False, weights_path=weights_path),
            attest_command(args, enabled=False),
        ],
    }


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    api_url = args.api_url.rstrip("/")
    api_health = _request_json("GET", f"{api_url}/health", timeout_s=args.http_timeout_s)
    attest = _collect_config_attest(
        api_url,
        samples=args.preflight_attest_samples,
        timeout_s=args.http_timeout_s,
    )
    values = _routing_flag_by_pid(attest)
    return {
        "api_url": api_url,
        "api_health": asdict(api_health),
        "autopilot_active": _active_autopilot(),
        "weights_path": str(Path(args.weights)),
        "weights_present": Path(args.weights).exists(),
        "config_attest": {
            "samples_requested": args.preflight_attest_samples,
            "workers_seen": len(attest),
            "routing_classifier_by_pid": values,
            "routing_classifier_sources_by_pid": _routing_flag_sources_by_pid(attest),
            "all_sampled_workers_enabled": bool(values) and all(values.values()),
            "any_sampled_worker_enabled": any(values.values()),
        },
    }


def pre_apply_blockers(args: argparse.Namespace, preflight: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not args.confirm_clean_window:
        blockers.append("--apply requires --confirm-clean-window")
    if not (preflight.get("api_health") or {}).get("ok"):
        blockers.append("orchestrator API health is not OK")
    if not preflight.get("weights_present"):
        blockers.append(f"routing classifier weights are missing: {preflight.get('weights_path')}")
    if preflight.get("autopilot_active") and not args.allow_autopilot_active:
        blockers.append("AutoPilot appears active; rollout would change routing during live evidence accrual")
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


def _json_from_tail(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        parsed = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def execute_commands(
    commands: list[PlannedCommand],
    *,
    timeout_s: float,
) -> tuple[list[StepResult], list[str]]:
    steps: list[StepResult] = []
    errors: list[str] = []
    for command in commands:
        result = run_command(command, timeout_s=timeout_s)
        steps.append(result)
        if not result.ok:
            errors.append(f"{command.name} failed with rc={result.returncode}")
            break
    return steps, errors


def _attest_reports(steps: list[StepResult]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for step in steps:
        if step.name.startswith("attest_"):
            reports[step.name] = _json_from_tail(step.stdout_tail)
    return reports


def build_report(args: argparse.Namespace, *, output_dir: Path) -> tuple[dict[str, Any], int]:
    preflight = build_preflight(args)
    plan = rollout_plan(args)
    blockers = pre_apply_blockers(args, preflight) if args.apply else []
    steps: list[StepResult] = []
    rollback_steps: list[StepResult] = []
    status = "plan_only"
    rc = 0

    if args.apply and blockers:
        status = "blocked"
        rc = 75 if any("AutoPilot appears active" in item for item in blockers) else 2
    elif args.apply:
        steps, errors = execute_commands(
            plan["activation"],
            timeout_s=args.command_timeout_s,
        )
        if errors:
            status = "activation_failed"
            blockers.extend(errors)
            rc = 75
        elif args.keep_enabled:
            status = "attestation_passed_enabled_left_on"
            rc = 0
        else:
            rollback_steps, rollback_errors = execute_commands(
                plan["rollback"],
                timeout_s=args.command_timeout_s,
            )
            if rollback_errors:
                status = "attestation_passed_rollback_failed"
                blockers.extend(rollback_errors)
                rc = 75
            else:
                status = "attestation_passed_rolled_back"
                rc = 0

        if status == "activation_failed" and not args.skip_rollback:
            rollback_steps, rollback_errors = execute_commands(
                plan["rollback"],
                timeout_s=args.command_timeout_s,
            )
            blockers.extend(rollback_errors)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "applied": bool(args.apply),
        "keep_enabled": bool(args.keep_enabled),
        "allow_autopilot_active": bool(args.allow_autopilot_active),
        "decision_grade": False,
        "rollout_attested": bool(args.apply and not blockers and "attestation_passed" in status),
        "notes": [
            "No model prompts are sent by this rollout harness.",
            "decision_grade=false because this attests feature rollout only, not routing quality.",
        ],
        "blockers": blockers,
        "preflight": preflight,
        "activation_commands": [command.display() for command in plan["activation"]],
        "rollback_commands": [command.display() for command in plan["rollback"]],
        "steps": [asdict(step) for step in steps],
        "rollback_steps": [asdict(step) for step in rollback_steps],
        "attestation_reports": {
            **_attest_reports(steps),
            **_attest_reports(rollback_steps),
        },
    }
    return report, rc


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Routing Classifier Rollout Window",
        "",
        f"- status: `{report['status']}`",
        f"- applied: `{report['applied']}`",
        f"- keep_enabled: `{report['keep_enabled']}`",
        f"- rollout_attested: `{report['rollout_attested']}`",
        f"- decision_grade: `{report['decision_grade']}`",
        f"- autopilot_active: `{report['preflight'].get('autopilot_active')}`",
        f"- weights_present: `{report['preflight'].get('weights_present')}`",
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
    if report.get("rollback_steps"):
        lines.extend(["", "## Rollback Steps", ""])
        for step in report["rollback_steps"]:
            lines.append(
                f"- `{step['name']}` rc=`{step['returncode']}` elapsed_s=`{step['elapsed_s']:.3f}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    return json_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.environ.get("ORCHESTRATOR_API_URL", DEFAULT_API_URL))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(os.environ.get("ROUTING_CLASSIFIER_WEIGHTS", str(DEFAULT_WEIGHTS_PATH))),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-clean-window", action="store_true")
    parser.add_argument("--allow-autopilot-active", action="store_true")
    parser.add_argument("--keep-enabled", action="store_true")
    parser.add_argument("--skip-rollback", action="store_true")
    parser.add_argument("--http-timeout-s", type=float, default=5.0)
    parser.add_argument("--command-timeout-s", type=float, default=900.0)
    parser.add_argument("--preflight-attest-samples", type=int, default=12)
    parser.add_argument("--attest-polls", type=int, default=120)
    parser.add_argument("--attest-delay-s", type=float, default=0.05)
    parser.add_argument("--min-workers", type=int, default=1)
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
