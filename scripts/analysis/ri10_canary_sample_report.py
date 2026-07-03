#!/usr/bin/env python3
"""Summarize RI-10 factual-risk canary sample coverage from routing logs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

SCRIPT_PATH = Path(__file__).resolve()
ORCH_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_LOG_DIR = ORCH_ROOT / "logs" / "progress"
DEFAULT_CANARY_START = "2026-04-06"
DEFAULT_GATE = 50
DEFAULT_MIN_ARM_SAMPLES = 10
DEFAULT_CANARY_ROLES = ("frontdoor",)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _date_key(value: str | None) -> str:
    if not value:
        return ""
    return value[:10]


def _iter_progress_records(log_dir: Path) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in sorted(log_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield path, line_no, record


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _action_key(data: dict[str, Any]) -> str:
    action = str(data.get("risk_gate_action") or "")
    reason = str(data.get("risk_gate_reason") or "")
    if reason:
        return f"{action or '<missing>'}:{reason}"
    return action or "<missing>"


def _routing_roles(data: dict[str, Any]) -> set[str]:
    routing = data.get("routing") or []
    if isinstance(routing, str):
        routing = [routing]
    if isinstance(routing, list):
        return {str(role) for role in routing if role}
    return set()


def _is_canary_participant(data: dict[str, Any], canary_roles: set[str]) -> bool:
    if not canary_roles:
        return True
    return bool(_routing_roles(data) & canary_roles)


def _is_enforce_arm(data: dict[str, Any], canary_roles: set[str]) -> bool:
    if not _is_canary_participant(data, canary_roles):
        return False
    mode = str(data.get("factual_risk_mode") or data.get("canary_mode") or "")
    action = str(data.get("risk_gate_action") or "")
    return mode == "enforce" or action in {"enforce", "forced_enforce", "risk_enforced"}


def _is_shadow_arm(data: dict[str, Any], canary_roles: set[str]) -> bool:
    if not _is_canary_participant(data, canary_roles):
        return False
    mode = str(data.get("factual_risk_mode") or data.get("canary_mode") or "")
    action = str(data.get("risk_gate_action") or "")
    return mode == "shadow" or action in {"shadow", "not_enforced_shadow"}


def build_report(
    log_dir: Path = DEFAULT_LOG_DIR,
    *,
    canary_start: str = DEFAULT_CANARY_START,
    decision_gate: int = DEFAULT_GATE,
    min_arm_samples: int = DEFAULT_MIN_ARM_SAMPLES,
    canary_roles: Iterable[str] = DEFAULT_CANARY_ROLES,
) -> dict[str, Any]:
    canary_role_set = {str(role) for role in canary_roles if str(role)}
    routing_rows = 0
    since_rows = 0
    high_total = 0
    high_since = 0
    frontdoor_high_since = 0
    canary_role_high_since = 0
    band_counts = Counter()
    since_band_counts = Counter()
    high_by_date = Counter()
    high_actions = Counter()
    high_sources = Counter()
    enforce_high = 0
    shadow_high = 0
    examples: list[dict[str, Any]] = []

    for path, line_no, record in _iter_progress_records(log_dir):
        if record.get("event_type") != "routing_decision":
            continue
        routing_rows += 1
        data = record.get("data") or {}
        if not isinstance(data, dict):
            continue
        date = _date_key(str(record.get("timestamp") or path.stem))
        band = str(data.get("factual_risk_band") or "")
        band_counts[band or "<missing>"] += 1
        in_window = date >= canary_start
        if in_window:
            since_rows += 1
            since_band_counts[band or "<missing>"] += 1
        if band != "high":
            continue

        high_total += 1
        if not in_window:
            continue

        high_since += 1
        routing = sorted(_routing_roles(data))
        if "frontdoor" in routing:
            frontdoor_high_since += 1
        if _is_canary_participant(data, canary_role_set):
            canary_role_high_since += 1
        high_by_date[date] += 1
        high_actions[_action_key(data)] += 1
        high_sources[str(data.get("decision_source") or data.get("strategy") or "<missing>")] += 1
        if _is_enforce_arm(data, canary_role_set):
            enforce_high += 1
        if _is_shadow_arm(data, canary_role_set):
            shadow_high += 1
        if len(examples) < 12:
            examples.append(
                {
                    "path": str(path),
                    "line": line_no,
                    "timestamp": record.get("timestamp"),
                    "routing": routing,
                    "factual_risk_score": data.get("factual_risk_score"),
                    "risk_gate_action": data.get("risk_gate_action"),
                    "risk_gate_reason": data.get("risk_gate_reason"),
                    "decision_source": data.get("decision_source") or data.get("strategy"),
                }
            )

    arm_attributed_high = enforce_high + shadow_high
    risk_control_disabled_high = high_actions.get("not_enforced:risk_control_disabled", 0)
    non_evaluable_high = max(0, canary_role_high_since - arm_attributed_high)
    non_canary_role_high = max(0, high_since - canary_role_high_since)
    sample_count_ready = high_since >= decision_gate
    arm_sample_count_ready = arm_attributed_high >= decision_gate
    arm_balance_ready = enforce_high >= min_arm_samples and shadow_high >= min_arm_samples
    canary_decision_ready = sample_count_ready and arm_sample_count_ready and arm_balance_ready
    if not sample_count_ready:
        decision_reason = f"high-risk sample count {high_since} is below gate {decision_gate}"
    elif not arm_sample_count_ready:
        decision_reason = (
            f"only {arm_attributed_high} high-risk rows have observable enforce/shadow "
            f"canary arms; gate requires {decision_gate}"
        )
    elif not arm_balance_ready:
        decision_reason = (
            "enforce/shadow canary arm counts are below the per-arm gate "
            f"{min_arm_samples} (enforce={enforce_high}, shadow={shadow_high})"
        )
    else:
        decision_reason = (
            "high-risk sample count and enforce/shadow arm-attributed telemetry "
            "are decision-grade"
        )

    return {
        "generated_at": _iso_now(),
        "source_glob": str(log_dir / "*.jsonl"),
        "canary_start": canary_start,
        "decision_gate_high_risk_samples": decision_gate,
        "min_canary_arm_samples": min_arm_samples,
        "canary_roles": sorted(canary_role_set),
        "routing_decision_rows": routing_rows,
        "routing_decision_rows_since_canary_start": since_rows,
        "high_risk_rows_total": high_total,
        "high_risk_rows_since_canary_start": high_since,
        "frontdoor_high_risk_rows_since_canary_start": frontdoor_high_since,
        "canary_role_high_risk_rows_since_canary_start": canary_role_high_since,
        "evaluable_canary_arm_high_risk_rows": arm_attributed_high,
        "non_evaluable_high_risk_rows_since_canary_start": non_evaluable_high,
        "non_canary_role_high_risk_rows_since_canary_start": non_canary_role_high,
        "risk_control_disabled_high_risk_rows_since_canary_start": risk_control_disabled_high,
        "sample_count_ready": sample_count_ready,
        "canary_arm_sample_count_ready": arm_sample_count_ready,
        "canary_arm_balance_ready": arm_balance_ready,
        "canary_decision_ready": canary_decision_ready,
        "decision_reason": decision_reason,
        "band_counts": _counter_dict(band_counts),
        "band_counts_since_canary_start": _counter_dict(since_band_counts),
        "high_risk_by_date_since_canary_start": _counter_dict(high_by_date),
        "high_risk_gate_actions_since_canary_start": _counter_dict(high_actions),
        "high_risk_decision_sources_since_canary_start": _counter_dict(high_sources),
        "canary_arm_counts_since_canary_start": {
            "enforce_high_risk": enforce_high,
            "shadow_high_risk": shadow_high,
        },
        "example_high_risk_rows": examples,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--canary-start", default=DEFAULT_CANARY_START)
    parser.add_argument("--decision-gate", type=int, default=DEFAULT_GATE)
    parser.add_argument(
        "--canary-role",
        action="append",
        default=None,
        help=(
            "Role participating in the RI-10 canary. Repeat for multiple roles; "
            "default: frontdoor. Use --all-canary-roles to disable role filtering."
        ),
    )
    parser.add_argument(
        "--all-canary-roles",
        action="store_true",
        help="Treat every route as canary-eligible; use only if canary_roles is empty in config.",
    )
    parser.add_argument(
        "--min-arm-samples",
        type=int,
        default=DEFAULT_MIN_ARM_SAMPLES,
        help="Minimum high-risk rows required in each observable canary arm.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON report to this path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(
        args.log_dir,
        canary_start=args.canary_start,
        decision_gate=args.decision_gate,
        min_arm_samples=args.min_arm_samples,
        canary_roles=() if args.all_canary_roles else (args.canary_role or DEFAULT_CANARY_ROLES),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
