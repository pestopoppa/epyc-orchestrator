#!/usr/bin/env python3
"""Summarize RI-10 factual-risk canary sample coverage from routing logs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

SCRIPT_PATH = Path(__file__).resolve()
ORCH_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_LOG_DIR = ORCH_ROOT / "logs" / "progress"
DEFAULT_CLASSIFIER_CONFIG = ORCH_ROOT / "orchestration" / "classifier_config.yaml"
DEFAULT_CANARY_START = "2026-04-06"
DEFAULT_TELEMETRY_HEALTH_START = "2026-06-20"
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


def _factual_risk_mode(data: dict[str, Any]) -> str:
    return str(data.get("factual_risk_mode") or data.get("canary_mode") or "")


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
    mode = _factual_risk_mode(data)
    action = str(data.get("risk_gate_action") or "")
    return mode == "enforce" or action in {"enforce", "forced_enforce", "risk_enforced"}


def _is_shadow_arm(data: dict[str, Any], canary_roles: set[str]) -> bool:
    if not _is_canary_participant(data, canary_roles):
        return False
    mode = _factual_risk_mode(data)
    action = str(data.get("risk_gate_action") or "")
    return mode == "shadow" or action in {"shadow", "not_enforced_shadow"}


def _configured_canary_roles(config_path: Path = DEFAULT_CLASSIFIER_CONFIG) -> list[str] | None:
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    factual = loaded.get("factual_risk") or {}
    if not isinstance(factual, dict):
        return None
    roles = factual.get("canary_roles")
    if roles is None:
        return None
    if not isinstance(roles, list):
        return None
    return [str(role) for role in roles if str(role)]


def build_report(
    log_dir: Path = DEFAULT_LOG_DIR,
    *,
    canary_start: str = DEFAULT_CANARY_START,
    telemetry_health_start: str = DEFAULT_TELEMETRY_HEALTH_START,
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
    high_by_role = Counter()
    high_actions = Counter()
    high_modes = Counter()
    canary_role_high_by_role = Counter()
    canary_arm_counts_by_role = Counter()
    canary_role_modes = Counter()
    high_sources = Counter()
    enforce_high = 0
    shadow_high = 0
    telemetry_high_since = 0
    telemetry_frontdoor_high_since = 0
    telemetry_canary_role_high_since = 0
    telemetry_non_canary_role_high_since = 0
    telemetry_high_by_date = Counter()
    telemetry_high_by_role = Counter()
    telemetry_high_actions = Counter()
    telemetry_high_modes = Counter()
    telemetry_canary_role_high_by_role = Counter()
    telemetry_canary_arm_counts_by_role = Counter()
    telemetry_canary_role_modes = Counter()
    telemetry_enforce_high = 0
    telemetry_shadow_high = 0
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
        factual_mode = _factual_risk_mode(data) or "<missing>"
        in_telemetry_window = date >= telemetry_health_start
        is_canary_participant = _is_canary_participant(data, canary_role_set)
        is_enforce_arm = _is_enforce_arm(data, canary_role_set)
        is_shadow_arm = _is_shadow_arm(data, canary_role_set)
        high_modes[factual_mode] += 1
        if "frontdoor" in routing:
            frontdoor_high_since += 1
        if is_canary_participant:
            canary_role_high_since += 1
            canary_role_modes[factual_mode] += 1
        high_by_date[date] += 1
        high_by_role.update(routing or ["<missing>"])
        for role in routing:
            if not canary_role_set or role in canary_role_set:
                canary_role_high_by_role[role] += 1
        high_actions[_action_key(data)] += 1
        high_sources[str(data.get("decision_source") or data.get("strategy") or "<missing>")] += 1
        if is_enforce_arm:
            enforce_high += 1
            for role in routing:
                if not canary_role_set or role in canary_role_set:
                    canary_arm_counts_by_role[(role, "enforce_high_risk")] += 1
        if is_shadow_arm:
            shadow_high += 1
            for role in routing:
                if not canary_role_set or role in canary_role_set:
                    canary_arm_counts_by_role[(role, "shadow_high_risk")] += 1
        if in_telemetry_window:
            telemetry_high_since += 1
            telemetry_high_by_date[date] += 1
            telemetry_high_by_role.update(routing or ["<missing>"])
            telemetry_high_modes[factual_mode] += 1
            telemetry_high_actions[_action_key(data)] += 1
            if "frontdoor" in routing:
                telemetry_frontdoor_high_since += 1
            if is_canary_participant:
                telemetry_canary_role_high_since += 1
                telemetry_canary_role_modes[factual_mode] += 1
                for role in routing:
                    if not canary_role_set or role in canary_role_set:
                        telemetry_canary_role_high_by_role[role] += 1
            else:
                telemetry_non_canary_role_high_since += 1
            if is_enforce_arm:
                telemetry_enforce_high += 1
                for role in routing:
                    if not canary_role_set or role in canary_role_set:
                        telemetry_canary_arm_counts_by_role[(role, "enforce_high_risk")] += 1
            if is_shadow_arm:
                telemetry_shadow_high += 1
                for role in routing:
                    if not canary_role_set or role in canary_role_set:
                        telemetry_canary_arm_counts_by_role[(role, "shadow_high_risk")] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "path": str(path),
                    "line": line_no,
                    "timestamp": record.get("timestamp"),
                    "routing": routing,
                    "factual_risk_score": data.get("factual_risk_score"),
                    "factual_risk_mode": data.get("factual_risk_mode") or data.get("canary_mode"),
                    "risk_gate_action": data.get("risk_gate_action"),
                    "risk_gate_reason": data.get("risk_gate_reason"),
                    "decision_source": data.get("decision_source") or data.get("strategy"),
                }
            )

    arm_attributed_high = enforce_high + shadow_high
    risk_control_disabled_high = high_actions.get("not_enforced:risk_control_disabled", 0)
    non_evaluable_high = max(0, canary_role_high_since - arm_attributed_high)
    non_canary_role_high = max(0, high_since - canary_role_high_since)
    canary_role_missing_mode_high = canary_role_modes.get("<missing>", 0)
    canary_role_observable_mode_high = canary_role_high_since - canary_role_missing_mode_high
    telemetry_arm_attributed_high = telemetry_enforce_high + telemetry_shadow_high
    telemetry_missing_mode_high = telemetry_high_modes.get("<missing>", 0)
    telemetry_observable_mode_high = telemetry_high_since - telemetry_missing_mode_high
    telemetry_canary_role_missing_mode_high = telemetry_canary_role_modes.get("<missing>", 0)
    telemetry_canary_role_observable_mode_high = (
        telemetry_canary_role_high_since - telemetry_canary_role_missing_mode_high
    )
    telemetry_producer_currently_healthy = (
        telemetry_high_since > 0 and telemetry_missing_mode_high == 0
    )
    telemetry_canary_role_scope_starved = (
        telemetry_producer_currently_healthy
        and telemetry_non_canary_role_high_since > telemetry_canary_role_high_since
        and telemetry_canary_role_high_since < decision_gate
    )
    if telemetry_high_since == 0:
        telemetry_collection_blocker = "no_recent_high_risk_rows"
        telemetry_collection_reason = (
            "no high-risk routing rows were observed in the telemetry health window"
        )
    elif telemetry_missing_mode_high:
        telemetry_collection_blocker = "current_missing_factual_risk_mode"
        telemetry_collection_reason = (
            f"{telemetry_missing_mode_high} current high-risk row(s) still lack "
            "factual_risk_mode/canary_mode"
        )
    elif telemetry_canary_role_scope_starved:
        telemetry_collection_blocker = "canary_role_scope_starved"
        telemetry_collection_reason = (
            "current factual-risk telemetry is populated, but most recent high-risk "
            "traffic routes outside the configured canary_roles"
        )
    elif telemetry_canary_role_high_since < decision_gate:
        telemetry_collection_blocker = "canary_role_sample_count_insufficient"
        telemetry_collection_reason = (
            f"only {telemetry_canary_role_high_since} current high-risk row(s) "
            f"matched configured canary_roles; gate requires {decision_gate}"
        )
    elif telemetry_arm_attributed_high < decision_gate:
        telemetry_collection_blocker = "canary_arm_volume_insufficient"
        telemetry_collection_reason = (
            f"only {telemetry_arm_attributed_high} current high-risk row(s) have "
            f"observable enforce/shadow canary arms; gate requires {decision_gate}"
        )
    elif telemetry_enforce_high < min_arm_samples or telemetry_shadow_high < min_arm_samples:
        telemetry_collection_blocker = "canary_arm_balance_insufficient"
        telemetry_collection_reason = (
            "current enforce/shadow canary arm counts are below the per-arm gate "
            f"{min_arm_samples} (enforce={telemetry_enforce_high}, "
            f"shadow={telemetry_shadow_high})"
        )
    else:
        telemetry_collection_blocker = "decision_ready"
        telemetry_collection_reason = (
            "current high-risk telemetry has decision-grade canary arm coverage"
        )
    sample_count_ready = high_since >= decision_gate
    arm_sample_count_ready = arm_attributed_high >= decision_gate
    arm_balance_ready = enforce_high >= min_arm_samples and shadow_high >= min_arm_samples
    canary_decision_ready = sample_count_ready and arm_sample_count_ready and arm_balance_ready
    telemetry_canary_role_sample_deficit = max(0, decision_gate - telemetry_canary_role_high_since)
    telemetry_canary_arm_volume_deficit = max(0, decision_gate - telemetry_arm_attributed_high)
    telemetry_canary_enforce_arm_deficit = max(0, min_arm_samples - telemetry_enforce_high)
    telemetry_canary_shadow_arm_deficit = max(0, min_arm_samples - telemetry_shadow_high)
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
        "telemetry_health_start": telemetry_health_start,
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
        "canary_role_observable_factual_risk_mode_high_risk_rows": canary_role_observable_mode_high,
        "canary_role_missing_factual_risk_mode_high_risk_rows": canary_role_missing_mode_high,
        "risk_control_disabled_high_risk_rows_since_canary_start": risk_control_disabled_high,
        "high_risk_rows_since_telemetry_health_start": telemetry_high_since,
        "frontdoor_high_risk_rows_since_telemetry_health_start": (
            telemetry_frontdoor_high_since
        ),
        "canary_role_high_risk_rows_since_telemetry_health_start": (
            telemetry_canary_role_high_since
        ),
        "non_canary_role_high_risk_rows_since_telemetry_health_start": (
            telemetry_non_canary_role_high_since
        ),
        "evaluable_canary_arm_high_risk_rows_since_telemetry_health_start": (
            telemetry_arm_attributed_high
        ),
        "observable_factual_risk_mode_high_risk_rows_since_telemetry_health_start": (
            telemetry_observable_mode_high
        ),
        "missing_factual_risk_mode_high_risk_rows_since_telemetry_health_start": (
            telemetry_missing_mode_high
        ),
        "canary_role_observable_factual_risk_mode_high_risk_rows_since_telemetry_health_start": (
            telemetry_canary_role_observable_mode_high
        ),
        "canary_role_missing_factual_risk_mode_high_risk_rows_since_telemetry_health_start": (
            telemetry_canary_role_missing_mode_high
        ),
        "telemetry_producer_currently_healthy": telemetry_producer_currently_healthy,
        "telemetry_canary_role_scope_starved": telemetry_canary_role_scope_starved,
        "telemetry_collection_blocker": telemetry_collection_blocker,
        "telemetry_collection_reason": telemetry_collection_reason,
        "sample_count_ready": sample_count_ready,
        "canary_arm_sample_count_ready": arm_sample_count_ready,
        "canary_arm_balance_ready": arm_balance_ready,
        "canary_decision_ready": canary_decision_ready,
        "decision_reason": decision_reason,
        "band_counts": _counter_dict(band_counts),
        "band_counts_since_canary_start": _counter_dict(since_band_counts),
        "high_risk_by_date_since_canary_start": _counter_dict(high_by_date),
        "high_risk_by_date_since_telemetry_health_start": _counter_dict(
            telemetry_high_by_date
        ),
        "high_risk_by_role_since_canary_start": _counter_dict(high_by_role),
        "high_risk_by_role_since_telemetry_health_start": _counter_dict(
            telemetry_high_by_role
        ),
        "canary_role_high_risk_by_role_since_canary_start": _counter_dict(
            canary_role_high_by_role
        ),
        "canary_role_high_risk_by_role_since_telemetry_health_start": _counter_dict(
            telemetry_canary_role_high_by_role
        ),
        "high_risk_factual_risk_modes_since_canary_start": _counter_dict(high_modes),
        "canary_role_factual_risk_modes_since_canary_start": _counter_dict(canary_role_modes),
        "high_risk_factual_risk_modes_since_telemetry_health_start": _counter_dict(
            telemetry_high_modes
        ),
        "canary_role_factual_risk_modes_since_telemetry_health_start": _counter_dict(
            telemetry_canary_role_modes
        ),
        "high_risk_gate_actions_since_canary_start": _counter_dict(high_actions),
        "high_risk_gate_actions_since_telemetry_health_start": _counter_dict(
            telemetry_high_actions
        ),
        "memory_risk_gate_actions_since_canary_start": _counter_dict(high_actions),
        "high_risk_decision_sources_since_canary_start": _counter_dict(high_sources),
        "canary_arm_counts_since_canary_start": {
            "enforce_high_risk": enforce_high,
            "shadow_high_risk": shadow_high,
        },
        "canary_arm_counts_since_telemetry_health_start": {
            "enforce_high_risk": telemetry_enforce_high,
            "shadow_high_risk": telemetry_shadow_high,
        },
        "canary_arm_counts_by_role_since_canary_start": _role_arm_counts(
            canary_arm_counts_by_role
        ),
        "canary_arm_counts_by_role_since_telemetry_health_start": _role_arm_counts(
            telemetry_canary_arm_counts_by_role
        ),
        "canary_role_sample_deficit_since_telemetry_health_start": (
            telemetry_canary_role_sample_deficit
        ),
        "canary_arm_volume_deficit_since_telemetry_health_start": (
            telemetry_canary_arm_volume_deficit
        ),
        "canary_arm_balance_deficits_since_telemetry_health_start": {
            "enforce_high_risk": telemetry_canary_enforce_arm_deficit,
            "shadow_high_risk": telemetry_canary_shadow_arm_deficit,
        },
        "example_high_risk_rows": examples,
    }


def _role_arm_counts(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    by_role: dict[str, dict[str, int]] = {}
    for (role, arm), count in sorted(counter.items(), key=lambda item: item[0]):
        by_role.setdefault(str(role), {})[str(arm)] = int(count)
    for counts in by_role.values():
        counts.setdefault("enforce_high_risk", 0)
        counts.setdefault("shadow_high_risk", 0)
    return by_role


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--classifier-config", type=Path, default=DEFAULT_CLASSIFIER_CONFIG)
    parser.add_argument("--canary-start", default=DEFAULT_CANARY_START)
    parser.add_argument("--telemetry-health-start", default=DEFAULT_TELEMETRY_HEALTH_START)
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
    if args.all_canary_roles:
        canary_roles: Iterable[str] = ()
    elif args.canary_role is not None:
        canary_roles = args.canary_role
    else:
        canary_roles = _configured_canary_roles(args.classifier_config) or DEFAULT_CANARY_ROLES
    report = build_report(
        args.log_dir,
        canary_start=args.canary_start,
        telemetry_health_start=args.telemetry_health_start,
        decision_gate=args.decision_gate,
        min_arm_samples=args.min_arm_samples,
        canary_roles=canary_roles,
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
