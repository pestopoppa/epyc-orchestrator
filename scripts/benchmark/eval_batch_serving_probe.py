#!/usr/bin/env python3
"""Guarded probe for the default-off eval-batch serving lane.

The A7/P-BENCH-3 E2 result showed that a single full frontdoor server with
continuous batching can make EvalTower substantially faster. The production
hook remains default-off. This script turns the next activation window into a
repeatable gate: preflight the warm endpoint/API flag state, optionally send one
eval-batch request, and verify route attribution from the structured tap.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_EVAL_BATCH_PORT = 18070
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "orchestration" / "reports"
TAP_SENTINEL = Path("/mnt/raid0/llm/tmp/.inference_tap_active")


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


def default_eval_batch_url() -> str:
    raw = os.environ.get("ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL", "").strip()
    if raw:
        return raw.rstrip("/")
    return f"http://localhost:{DEFAULT_EVAL_BATCH_PORT}"


def default_output_dir(stamp: str | None = None) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"eval_batch_serving_probe_{stamp or utc_stamp()}"


def _active_autopilot() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "scripts/autopilot/autopilot.py start"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> HttpResult:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
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


def _collect_config_attest(
    api_url: str,
    *,
    samples: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
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


def _tap_events_path_from_tap_path(tap_path: str) -> Path | None:
    value = tap_path.strip()
    if not value or value == os.devnull:
        return None
    path = Path(value)
    if path.name == "inference_tap.log":
        return path.with_name("inference_tap_events.jsonl")
    suffix = path.suffix or ".log"
    return path.with_suffix(f"{suffix}.events.jsonl")


def resolve_tap_events_path(explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    override = os.environ.get("INFERENCE_TAP_EVENTS_FILE", "").strip()
    if override:
        return Path(override)
    tap_path = os.environ.get("INFERENCE_TAP_FILE", "").strip()
    if not tap_path and TAP_SENTINEL.exists():
        try:
            tap_path = TAP_SENTINEL.read_text(encoding="utf-8").strip()
        except OSError:
            tap_path = ""
    return _tap_events_path_from_tap_path(tap_path) if tap_path else None


def _recent_text(path: Path, *, max_bytes: int) -> str:
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            fh.readline()
        return fh.read().decode("utf-8", errors="replace")


def load_recent_tap_events(
    path: Path | None,
    *,
    batch_id: str,
    max_bytes: int = 32 * 1024 * 1024,
) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in _recent_text(path, max_bytes=max_bytes).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and str(event.get("batch_id") or "") == batch_id:
            events.append(event)
    return events


def summarize_tap_events(
    events: list[dict[str, Any]],
    *,
    expected_port: int = DEFAULT_EVAL_BATCH_PORT,
) -> dict[str, Any]:
    ports = sorted(
        {
            int(event["port"])
            for event in events
            if isinstance(event.get("port"), int)
        }
    )
    roles = sorted({str(event.get("role")) for event in events if event.get("role")})
    request_ids = sorted(
        {str(event.get("request_id")) for event in events if event.get("request_id")}
    )
    timing_events = [event for event in events if event.get("event") == "timings"]
    tps_values = [
        float(event["tps"])
        for event in timing_events
        if isinstance(event.get("tps"), (int, float))
    ]
    return {
        "event_count": len(events),
        "request_ids": request_ids,
        "roles": roles,
        "ports": ports,
        "expected_port": expected_port,
        "hit_expected_port": expected_port in ports,
        "timing_event_count": len(timing_events),
        "tps_values": tps_values,
        "median_tps": _median(tps_values),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def activation_commands(eval_batch_url: str) -> list[str]:
    return [
        "cd /mnt/raid0/llm/epyc-orchestrator && "
        "uv run python scripts/server/orchestrator_stack.py start --include-warm eval_batch_frontdoor",
        "cd /mnt/raid0/llm/epyc-orchestrator && "
        f"ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=1 "
        f"ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL={eval_batch_url.rstrip('/')} "
        "uv run python scripts/server/orchestrator_stack.py reload orchestrator",
        "cd /mnt/raid0/llm/epyc-orchestrator && "
        "uv run python scripts/benchmark/eval_batch_serving_probe.py "
        "--smoke --confirm-clean-window --require-enabled",
    ]


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    api_url = args.api_url.rstrip("/")
    eval_url = args.eval_batch_url.rstrip("/")
    api_health = _request_json(
        "GET",
        f"{api_url}/health",
        timeout_s=args.http_timeout_s,
    )
    eval_health = _request_json(
        "GET",
        f"{eval_url}/health",
        timeout_s=args.http_timeout_s,
    )
    attest = _collect_config_attest(
        api_url,
        samples=args.attest_samples,
        timeout_s=args.http_timeout_s,
    )
    feature_values = {
        str(row["pid"]): bool((row.get("flags") or {}).get("eval_batch_serving"))
        for row in attest
        if isinstance(row.get("pid"), int)
    }
    source_values = {
        str(row["pid"]): (row.get("sources") or {}).get("eval_batch_serving")
        for row in attest
        if isinstance(row.get("pid"), int)
    }
    all_enabled = bool(feature_values) and all(feature_values.values())
    any_disabled = any(value is False for value in feature_values.values())
    return {
        "api_url": api_url,
        "eval_batch_url": eval_url,
        "api_health": asdict(api_health),
        "eval_batch_frontdoor_health": asdict(eval_health),
        "autopilot_active": _active_autopilot(),
        "config_attest": {
            "samples_requested": args.attest_samples,
            "workers_seen": len(attest),
            "eval_batch_serving_by_pid": feature_values,
            "eval_batch_serving_sources_by_pid": source_values,
            "all_sampled_workers_enabled": all_enabled,
            "any_sampled_worker_disabled": any_disabled,
        },
        "activation_commands": activation_commands(eval_url),
    }


def _smoke_payload(args: argparse.Namespace, *, batch_id: str, request_id: str) -> dict[str, Any]:
    return {
        "prompt": args.prompt,
        "mock_mode": False,
        "real_mode": True,
        "force_role": args.force_role,
        "force_mode": "direct",
        "max_turns": 1,
        "cache_prompt": False,
        "request_id": request_id,
        "batch_id": batch_id,
        "request_priority": "background",
        "workload_class": "eval_batch",
        "timeout_s": args.request_timeout_s,
    }


def run_smoke(args: argparse.Namespace, *, stamp: str) -> dict[str, Any]:
    batch_id = args.batch_id or f"eval-batch-probe-{stamp}"
    request_id = args.request_id or f"{batch_id}:request"
    payload = _smoke_payload(args, batch_id=batch_id, request_id=request_id)
    response = _request_json(
        "POST",
        f"{args.api_url.rstrip('/')}/chat",
        payload=payload,
        timeout_s=args.client_timeout_s,
    )
    time.sleep(max(0.0, args.tap_grace_s))
    tap_path = resolve_tap_events_path(args.tap_events)
    events = load_recent_tap_events(tap_path, batch_id=batch_id)
    tap_summary = summarize_tap_events(events, expected_port=args.expected_port)
    return {
        "batch_id": batch_id,
        "request_id": request_id,
        "payload": payload,
        "response": asdict(response),
        "tap_events_path": str(tap_path) if tap_path else None,
        "tap_summary": tap_summary,
    }


def _preflight_blockers(args: argparse.Namespace, preflight: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not preflight["api_health"]["ok"]:
        blockers.append("orchestrator API health is not OK")
    if not preflight["eval_batch_frontdoor_health"]["ok"]:
        blockers.append("eval_batch_frontdoor health is not OK")
    if args.require_enabled and not preflight["config_attest"]["all_sampled_workers_enabled"]:
        blockers.append("eval_batch_serving is not enabled on every sampled API worker")
    if args.smoke and not args.confirm_clean_window:
        blockers.append("--smoke requires --confirm-clean-window")
    if args.smoke and preflight["autopilot_active"] and not args.allow_autopilot_active:
        blockers.append("AutoPilot appears active; smoke would contaminate live eval resources")
    return blockers


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Eval-Batch Serving Probe",
        "",
        f"- status: `{report['status']}`",
        f"- decision_grade: `{report['decision_grade']}`",
        f"- api: `{report['preflight']['api_url']}`",
        f"- eval_batch_frontdoor: `{report['preflight']['eval_batch_url']}`",
        f"- autopilot_active: `{report['preflight']['autopilot_active']}`",
        f"- api_health_ok: `{report['preflight']['api_health']['ok']}`",
        f"- eval_frontdoor_health_ok: `{report['preflight']['eval_batch_frontdoor_health']['ok']}`",
        f"- sampled_workers_enabled: `{report['preflight']['config_attest']['all_sampled_workers_enabled']}`",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    if report.get("smoke"):
        smoke = report["smoke"]
        lines.extend(
            [
                "",
                "## Smoke",
                "",
                f"- response_ok: `{smoke['response']['ok']}`",
                f"- status_code: `{smoke['response']['status']}`",
                f"- batch_id: `{smoke['batch_id']}`",
                f"- tap_events_path: `{smoke['tap_events_path']}`",
                f"- tap_hit_expected_port: `{smoke['tap_summary']['hit_expected_port']}`",
                f"- tap_ports: `{smoke['tap_summary']['ports']}`",
                f"- median_tps: `{smoke['tap_summary']['median_tps']}`",
            ]
        )
    lines.extend(["", "## Activation Commands", ""])
    lines.extend(f"```bash\n{cmd}\n```" for cmd in report["preflight"]["activation_commands"])
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
    parser.add_argument("--eval-batch-url", default=default_eval_batch_url())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Send one real eval-batch /chat request.")
    parser.add_argument("--confirm-clean-window", action="store_true")
    parser.add_argument("--allow-autopilot-active", action="store_true")
    parser.add_argument("--require-enabled", action="store_true")
    parser.add_argument("--attest-samples", type=int, default=12)
    parser.add_argument("--http-timeout-s", type=float, default=5.0)
    parser.add_argument("--client-timeout-s", type=float, default=240.0)
    parser.add_argument("--request-timeout-s", type=int, default=180)
    parser.add_argument("--tap-grace-s", type=float, default=1.0)
    parser.add_argument("--tap-events", default=None)
    parser.add_argument("--expected-port", type=int, default=DEFAULT_EVAL_BATCH_PORT)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--force-role", default="frontdoor")
    parser.add_argument(
        "--prompt",
        default=(
            "Answer only the final number: a worker processes 18 items in 6 minutes. "
            "How many items per minute?"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = utc_stamp()
    output_dir = args.output_dir or default_output_dir(stamp)
    preflight = build_preflight(args)
    blockers = _preflight_blockers(args, preflight)

    smoke = None
    if args.smoke and not blockers:
        smoke = run_smoke(args, stamp=stamp)
        if not smoke["response"]["ok"]:
            blockers.append("smoke /chat response was not OK")
        if not smoke["tap_summary"]["hit_expected_port"]:
            blockers.append("structured tap did not show eval-batch traffic on expected port")

    decision_grade = bool(args.smoke and args.confirm_clean_window and not blockers)
    if blockers:
        status = "blocked"
    elif smoke:
        status = "smoke_passed"
    else:
        status = "preflight_only"

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "decision_grade": decision_grade,
        "blockers": blockers,
        "preflight": preflight,
        "smoke": smoke,
    }
    json_path, md_path = write_report(report, output_dir)
    if not args.summary_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {json_path}")
        print(f"wrote {md_path}")
    if args.smoke and "--smoke requires --confirm-clean-window" in blockers:
        return 2
    if blockers and args.smoke:
        return 75
    return 0


if __name__ == "__main__":
    sys.exit(main())
