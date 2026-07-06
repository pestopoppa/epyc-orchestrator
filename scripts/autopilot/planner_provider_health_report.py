#!/usr/bin/env python3
"""Read-only planner provider health report for local AutoPilot operation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "autopilot_planner_provider_health.v1"
DEFAULT_TAP_PATH = Path("/mnt/raid0/llm/tmp/planner_tap.log")
DEFAULT_TAIL_BYTES = 1_500_000
DEFAULT_STALE_AFTER_S = 7_200.0
LOCAL_PROVIDER_PREFIX = "local_"
START_RE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\s+PLANNER\s+"
    r"provider=(?P<provider>\S+)\s+role=(?P<role>\S+)\s+start",
    re.MULTILINE,
)
END_RE = re.compile(
    r"^\[END\s+provider=(?P<provider>\S+)\s+role=(?P<role>[^\]]+)\]\s+"
    r"result_chars=(?P<result_chars>\d+)",
    re.MULTILINE,
)
FAIL_RE = re.compile(
    r"^\[FAIL\s+provider=(?P<provider>\S+)\s+role=(?P<role>[^\]]+)\]\s*(?P<message>.*)",
    re.MULTILINE,
)
RETRY_RE = re.compile(
    r"^\[RETRY\s+provider=(?P<provider>\S+)\s+role=(?P<role>[^\]]+)\]\s*(?P<message>.*)",
    re.MULTILINE,
)
ACTION_FENCE_RE = re.compile(r"```json:autopilot_actions\s*(?P<payload>.*?)```", re.DOTALL)
CRITIQUE_FENCE_RE = re.compile(r"```json:autopilot_critique\s*(?P<payload>.*?)```", re.DOTALL)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _read_tail(path: Path, tail_bytes: int) -> str:
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - max(1, tail_bytes)))
        return fh.read().decode("utf-8", errors="replace")


def _split_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n={20,}\n", text) if block.strip()]


def _parse_json_payload(payload: str) -> tuple[dict[str, Any] | None, str]:
    payload = payload.strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start < 0 or end <= start:
            return None, "json_object_not_found"
        try:
            value = json.loads(payload[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"json_parse_error:{exc.msg}"
    if not isinstance(value, dict):
        return None, "json_payload_not_object"
    return value, ""


def _parse_event_block(block: str) -> dict[str, Any] | None:
    start = START_RE.search(block)
    if not start:
        return None
    provider = start.group("provider")
    role = start.group("role")
    event: dict[str, Any] = {
        "timestamp": start.group("timestamp"),
        "provider": provider,
        "role": role,
        "status": "started",
        "retry_count": 0,
        "result_chars": None,
        "failure": "",
        "parse_error": "",
        "action_type": None,
        "critique_decision": None,
        "critique_confidence": None,
        "revised_action_type": None,
        "issues": [],
    }

    retries = RETRY_RE.findall(block)
    event["retry_count"] = len(retries)
    failure = FAIL_RE.search(block)
    end = END_RE.search(block)
    if failure:
        event["status"] = "failed"
        event["failure"] = failure.group("message").strip()
    elif end:
        event["status"] = "ended"
        event["result_chars"] = int(end.group("result_chars"))

    if role == "draft":
        match = ACTION_FENCE_RE.search(block)
        if match:
            action, parse_error = _parse_json_payload(match.group("payload"))
            event["parse_error"] = parse_error
            if action:
                event["action_type"] = action.get("type")
        elif event["status"] == "ended":
            event["parse_error"] = "autopilot_actions_fence_missing"

    if role == "critique":
        match = CRITIQUE_FENCE_RE.search(block)
        if match:
            critique, parse_error = _parse_json_payload(match.group("payload"))
            event["parse_error"] = parse_error
            if critique:
                event["critique_decision"] = critique.get("decision")
                event["critique_confidence"] = critique.get("confidence")
                issues = critique.get("issues")
                if isinstance(issues, list):
                    event["issues"] = [str(item) for item in issues]
                revised_action = critique.get("revised_action")
                if isinstance(revised_action, dict):
                    event["revised_action_type"] = revised_action.get("type")
        elif event["status"] == "ended":
            event["parse_error"] = "autopilot_critique_fence_missing"

    return event


def _count_key(mapping: dict[str, int], key: Any) -> None:
    if key is None:
        return
    text = str(key)
    if not text:
        return
    mapping[text] = mapping.get(text, 0) + 1


def _provider_stats(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    providers: dict[str, dict[str, Any]] = {}
    for event in events:
        provider = str(event["provider"])
        stats = providers.setdefault(
            provider,
            {
                "starts": 0,
                "ends": 0,
                "failures": 0,
                "retries": 0,
                "draft_successes": 0,
                "critique_successes": 0,
                "last_status": "",
                "last_role": "",
                "last_timestamp": "",
                "roles": {},
            },
        )
        stats["starts"] += 1
        stats["retries"] += int(event.get("retry_count") or 0)
        if event["status"] == "ended":
            stats["ends"] += 1
        if event["status"] == "failed":
            stats["failures"] += 1
        if event["role"] == "draft" and event["status"] == "ended" and event.get("action_type"):
            stats["draft_successes"] += 1
        if (
            event["role"] == "critique"
            and event["status"] == "ended"
            and event.get("critique_decision")
        ):
            stats["critique_successes"] += 1
        stats["last_status"] = event["status"]
        stats["last_role"] = event["role"]
        stats["last_timestamp"] = event["timestamp"]
        role_counts = stats["roles"].setdefault(event["role"], {"starts": 0, "ends": 0, "failures": 0})
        role_counts["starts"] += 1
        if event["status"] == "ended":
            role_counts["ends"] += 1
        if event["status"] == "failed":
            role_counts["failures"] += 1
    return providers


def _recent_issues(events: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for event in reversed(events):
        provider = str(event["provider"])
        role = str(event["role"])
        base = {
            "timestamp": event["timestamp"],
            "provider": provider,
            "role": role,
        }
        if event.get("failure"):
            issues.append({**base, "kind": "failure", "message": event["failure"]})
        if event.get("parse_error"):
            issues.append({**base, "kind": "parse_error", "message": event["parse_error"]})
        for item in event.get("issues") or []:
            issues.append({**base, "kind": "critic_issue", "message": item})
        if len(issues) >= max_items:
            return issues[:max_items]
    return issues


def _age_seconds(path: Path) -> float | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return max(0.0, dt.datetime.now(dt.timezone.utc).timestamp() - mtime)


def build_report(
    *,
    tap_path: Path = DEFAULT_TAP_PATH,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
    max_issues: int = 8,
) -> dict[str, Any]:
    tap_path = tap_path.expanduser().resolve()
    generated_at = _utc_now()
    if not tap_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "ok": False,
            "status": "missing",
            "blockers": [f"planner tap log missing: {tap_path}"],
            "tap_path": str(tap_path),
            "window": {"tail_bytes": tail_bytes, "event_count": 0},
            "providers": {},
            "local": {},
            "critic_decisions": {},
            "draft_actions": {},
            "recent_issues": [],
        }

    text = _read_tail(tap_path, tail_bytes)
    events = [event for block in _split_blocks(text) if (event := _parse_event_block(block))]
    last_age_s = _age_seconds(tap_path)
    blockers: list[str] = []
    if not events:
        blockers.append("no planner provider events parsed from tap log window")

    providers = _provider_stats(events)
    critic_decisions: dict[str, int] = {}
    draft_actions: dict[str, int] = {}
    revised_actions: dict[str, int] = {}
    local_draft_successes = 0
    local_critique_successes = 0
    local_failures = 0
    fallback_starts = 0
    for event in events:
        provider = str(event["provider"])
        is_local = provider.startswith(LOCAL_PROVIDER_PREFIX)
        if is_local and event["role"] == "draft" and event["status"] == "ended" and event.get("action_type"):
            local_draft_successes += 1
        if (
            is_local
            and event["role"] == "critique"
            and event["status"] == "ended"
            and event.get("critique_decision")
        ):
            local_critique_successes += 1
        if is_local and event["status"] == "failed":
            local_failures += 1
        if not is_local:
            fallback_starts += 1
        _count_key(critic_decisions, event.get("critique_decision"))
        _count_key(draft_actions, event.get("action_type"))
        _count_key(revised_actions, event.get("revised_action_type"))

    if events and local_draft_successes == 0:
        blockers.append("no successful local draft event in planner tap window")
    if events and local_critique_successes == 0:
        blockers.append("no successful local critique event in planner tap window")
    if events and events[-1]["status"] == "failed":
        blockers.append("latest planner provider event failed")

    stale = bool(last_age_s is not None and last_age_s > stale_after_s)
    if blockers:
        status = "attention"
    elif stale:
        status = "stale"
    else:
        status = "healthy"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "ok": not blockers,
        "status": status,
        "blockers": blockers,
        "tap_path": str(tap_path),
        "window": {
            "tail_bytes": tail_bytes,
            "event_count": len(events),
            "last_event_timestamp": events[-1]["timestamp"] if events else None,
            "tap_mtime_age_s": round(last_age_s, 3) if last_age_s is not None else None,
            "stale_after_s": stale_after_s,
        },
        "providers": providers,
        "local": {
            "draft_successes": local_draft_successes,
            "critique_successes": local_critique_successes,
            "failures": local_failures,
            "fallback_provider_starts": fallback_starts,
        },
        "critic_decisions": critic_decisions,
        "draft_actions": draft_actions,
        "revised_actions": revised_actions,
        "recent_issues": _recent_issues(events, max_issues),
    }


def format_report(report: dict[str, Any]) -> list[str]:
    lines = [
        f"planner-provider-health: {report['status']} ok={report['ok']}",
        f"tap: {report['tap_path']}",
        f"events: {report['window'].get('event_count', 0)}",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.append("blockers:")
        lines.extend(f"- {item}" for item in blockers)
    providers = report.get("providers") or {}
    if providers:
        lines.append("providers:")
        for provider, stats in sorted(providers.items()):
            lines.append(
                "- "
                f"{provider}: starts={stats['starts']} ends={stats['ends']} "
                f"failures={stats['failures']} retries={stats['retries']} "
                f"last={stats['last_status']}/{stats['last_role']}"
            )
    decisions = report.get("critic_decisions") or {}
    if decisions:
        lines.append(f"critic decisions: {decisions}")
    return lines


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report local/cloud planner provider health from planner_tap.log."
    )
    parser.add_argument(
        "--tap-path",
        type=Path,
        default=DEFAULT_TAP_PATH,
        help=f"Planner tap log path (default: {DEFAULT_TAP_PATH})",
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=DEFAULT_TAIL_BYTES,
        help="Number of bytes to inspect from the tail of the planner tap log.",
    )
    parser.add_argument(
        "--stale-after-s",
        type=float,
        default=DEFAULT_STALE_AFTER_S,
        help="Mark the report stale when the tap log is older than this many seconds.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=8,
        help="Maximum recent failure/parse/critic issues to include.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when not ok.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(
        tap_path=args.tap_path,
        tail_bytes=args.tail_bytes,
        stale_after_s=args.stale_after_s,
        max_issues=args.max_issues,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print("\n".join(format_report(report)))
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
