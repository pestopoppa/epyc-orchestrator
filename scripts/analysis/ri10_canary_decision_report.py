#!/usr/bin/env python3
"""Build the RI-10 canary rollout decision packet from progress logs.

The companion ``ri10_canary_sample_report.py`` answers whether the enforce and
shadow arms have enough current high-risk rows. This report answers the next
question: whether the observed arms support a rollout decision. It deliberately
separates computable operational proxies from missing factuality evidence so a
decision-ready sample is not mistaken for a promotion-grade decision.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import mean, median
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
LATENCY_REGRESSION_THRESHOLD = 1.10
COST_REGRESSION_THRESHOLD = 1.05
RATE_INFLATION_THRESHOLD = 1.20
SCORED_SUMMARY_SCHEMA = "ri10_canary_scored_response_report.v1"


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _date_key(value: str | None) -> str:
    return (value or "")[:10]


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
    if not isinstance(roles, list):
        return None
    return [str(role) for role in roles if str(role)]


def _routing_roles(data: dict[str, Any]) -> set[str]:
    routing = data.get("routing") or []
    if isinstance(routing, str):
        routing = [routing]
    if isinstance(routing, list):
        return {str(role) for role in routing if role}
    return set()


def _factual_risk_mode(data: dict[str, Any]) -> str:
    return str(data.get("factual_risk_mode") or data.get("canary_mode") or "")


def _is_canary_participant(data: dict[str, Any], canary_roles: set[str]) -> bool:
    if not canary_roles:
        return True
    return bool(_routing_roles(data) & canary_roles)


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (k - low) * (ordered[high] - ordered[low])


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _extract_elapsed_seconds(details: str | None) -> float | None:
    if not details:
        return None
    match = re.search(r"(?:^|,\s*)(\d+(?:\.\d+)?)s(?:,|$)", details)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)s", details)
    return float(match.group(1)) if match else None


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _quality_value(record: dict[str, Any] | None) -> float | None:
    if not record:
        return None
    data = record.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    for source in (data, record):
        for key in ("factuality_score", "quality_score", "accuracy_score", "reward"):
            value = _float_value(source.get(key))
            if value is not None:
                return value
    return None


def _scored_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(bucket.get("rows") or 0),
        "scored": int(bucket.get("scored") or 0),
        "missing": int(bucket.get("missing") or 0),
        "correct": int(bucket.get("correct") or 0),
        "accuracy": _round(_float_value(bucket.get("accuracy"))),
        "mean_token_f1": _round(_float_value(bucket.get("mean_token_f1"))),
    }


def _load_scored_quality_evidence(scored_summary_path: Path | None) -> dict[str, Any]:
    if scored_summary_path is None:
        return {"status": "not_provided"}
    loaded = json.loads(scored_summary_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"scored summary must be a JSON object: {scored_summary_path}")
    schema = loaded.get("schema_version")
    if schema != SCORED_SUMMARY_SCHEMA:
        raise ValueError(
            f"unsupported scored summary schema {schema!r}; expected {SCORED_SUMMARY_SCHEMA!r}"
        )
    buckets = loaded.get("buckets") or {}
    if not isinstance(buckets, dict):
        buckets = {}
    comparison = loaded.get("arm_comparison") or {}
    if not isinstance(comparison, dict):
        comparison = {}
    arms = {
        "enforce": _scored_bucket(buckets.get("arm:enforce") or {}),
        "shadow": _scored_bucket(buckets.get("arm:shadow") or {}),
    }
    ready = (
        loaded.get("status") == "ready"
        and comparison.get("status") == "ready"
        and arms["enforce"]["scored"] > 0
        and arms["shadow"]["scored"] > 0
    )
    return {
        "status": "ready" if ready else "not_ready",
        "source_path": str(scored_summary_path),
        "summary_status": loaded.get("status"),
        "arm_comparison_status": comparison.get("status"),
        "rows": int(loaded.get("rows") or 0),
        "status_counts": loaded.get("status_counts") or {},
        "f1_threshold": _float_value(loaded.get("f1_threshold")),
        "arms": arms,
        "comparison": {
            "accuracy_delta_enforce_minus_shadow": _round(
                _float_value(comparison.get("accuracy_delta_enforce_minus_shadow"))
            ),
            "mean_token_f1_delta_enforce_minus_shadow": _round(
                _float_value(comparison.get("mean_token_f1_delta_enforce_minus_shadow"))
            ),
        },
    }


def _completion_success(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return record.get("event_type") == "task_completed" and record.get("outcome") == "success"


def _completion_failed(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return record.get("event_type") == "task_failed" or (
        record.get("event_type") == "task_completed" and record.get("outcome") not in (None, "success")
    )


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_s"] for row in rows if row["latency_s"] is not None]
    costs = [row["estimated_cost"] for row in rows if row["estimated_cost"] is not None]
    quality = [row["quality_score"] for row in rows if row["quality_score"] is not None]
    n = len(rows)
    completed = sum(1 for row in rows if row["outcome_event_type"] == "task_completed")
    failed = sum(1 for row in rows if row["outcome_event_type"] == "task_failed")
    missing = sum(1 for row in rows if not row["outcome_event_type"])
    success = sum(1 for row in rows if row["success"])
    operational_errors = n - success
    escalation_tasks = sum(1 for row in rows if row["escalation_event_count"] > 0)
    review_tasks = sum(1 for row in rows if row["plan_review_event_count"] > 0)
    return {
        "rows": n,
        "unique_tasks": len({row["task_id"] for row in rows}),
        "completed": completed,
        "failed": failed,
        "missing_outcome": missing,
        "success": success,
        "operational_error_or_missing": operational_errors,
        "operational_error_or_missing_rate": _round(operational_errors / n if n else None),
        "latency_count": len(latencies),
        "latency_p50_s": _round(median(latencies) if latencies else None, 3),
        "latency_p95_s": _round(_percentile(latencies, 95), 3),
        "latency_mean_s": _round(mean(latencies) if latencies else None, 3),
        "estimated_cost_count": len(costs),
        "estimated_cost_mean": _round(mean(costs) if costs else None, 9),
        "estimated_cost_sum": _round(sum(costs) if costs else None, 9),
        "quality_count": len(quality),
        "quality_mean": _round(mean(quality) if quality else None, 6),
        "escalation_task_count": escalation_tasks,
        "escalation_event_count": sum(row["escalation_event_count"] for row in rows),
        "escalation_task_rate": _round(escalation_tasks / n if n else None),
        "plan_review_task_count": review_tasks,
        "plan_review_event_count": sum(row["plan_review_event_count"] for row in rows),
        "plan_review_task_rate": _round(review_tasks / n if n else None),
        "roles": _counter_dict(Counter(role for row in rows for role in row["routing"])),
        "risk_gate_actions": _counter_dict(Counter(row["risk_gate_action"] or "<missing>" for row in rows)),
        "plan_review_decisions": _counter_dict(
            Counter(decision for row in rows for decision in row["plan_review_decisions"])
        ),
    }


def _comparison(enforce: dict[str, Any], shadow: dict[str, Any]) -> dict[str, Any]:
    return {
        "latency_p95_ratio_enforce_over_shadow": _round(
            _safe_ratio(enforce.get("latency_p95_s"), shadow.get("latency_p95_s"))
        ),
        "latency_mean_ratio_enforce_over_shadow": _round(
            _safe_ratio(enforce.get("latency_mean_s"), shadow.get("latency_mean_s"))
        ),
        "estimated_cost_mean_ratio_enforce_over_shadow": _round(
            _safe_ratio(enforce.get("estimated_cost_mean"), shadow.get("estimated_cost_mean"))
        ),
        "operational_error_rate_delta": _round(
            (enforce.get("operational_error_or_missing_rate") or 0.0)
            - (shadow.get("operational_error_or_missing_rate") or 0.0)
        ),
        "escalation_task_rate_ratio_enforce_over_shadow": _round(
            _safe_ratio(enforce.get("escalation_task_rate"), shadow.get("escalation_task_rate"))
        ),
        "plan_review_task_rate_ratio_enforce_over_shadow": _round(
            _safe_ratio(enforce.get("plan_review_task_rate"), shadow.get("plan_review_task_rate"))
        ),
        "quality_mean_delta_enforce_minus_shadow": _round(
            enforce.get("quality_mean") - shadow.get("quality_mean")
            if enforce.get("quality_mean") is not None and shadow.get("quality_mean") is not None
            else None
        ),
    }


def _rate_inflated(enforce_rate: float | None, shadow_rate: float | None) -> bool:
    if enforce_rate is None or shadow_rate is None:
        return False
    if shadow_rate == 0:
        return enforce_rate > 0
    return enforce_rate > shadow_rate * RATE_INFLATION_THRESHOLD


def _decision(
    sample_report: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    cmp: dict[str, Any],
    quality_evidence: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    notes: list[str] = []
    if sample_report.get("canary_decision_ready") is not True:
        blockers.append("telemetry_not_decision_ready")
    enforce = arms.get("enforce", {})
    shadow = arms.get("shadow", {})
    quality_status = quality_evidence.get("status")
    if quality_status == "ready":
        quality_cmp = quality_evidence.get("comparison") or {}
        accuracy_delta = quality_cmp.get("accuracy_delta_enforce_minus_shadow")
        token_f1_delta = quality_cmp.get("mean_token_f1_delta_enforce_minus_shadow")
        if accuracy_delta is not None and accuracy_delta < 0:
            blockers.append("factuality_regression")
        elif token_f1_delta is not None and token_f1_delta < 0:
            blockers.append("factuality_regression")
        elif accuracy_delta is None or accuracy_delta <= 0:
            blockers.append("factuality_no_enforce_lift")
            notes.append(
                "Attached scored RI-10 evidence does not show an enforce-arm factuality lift; "
                "hold classifier/risk-routing expansion frozen."
            )
    elif quality_status == "not_ready":
        blockers.append("factuality_scored_summary_not_ready")
        notes.append(
            "A scored RI-10 summary was provided, but its arm comparison is not ready for a rollout decision."
        )
    elif enforce.get("quality_count", 0) == 0 or shadow.get("quality_count", 0) == 0:
        blockers.append("factuality_not_scored")
        notes.append(
            "Progress logs contain operational outcomes, but no scored factuality/accuracy field "
            "for both RI-10 arms; success/failure is not a factuality substitute."
        )
    elif cmp.get("quality_mean_delta_enforce_minus_shadow", 0.0) < 0:
        blockers.append("factuality_regression")
    latency_ratio = cmp.get("latency_p95_ratio_enforce_over_shadow")
    if latency_ratio is not None and latency_ratio > LATENCY_REGRESSION_THRESHOLD:
        blockers.append("p95_latency_regression")
    cost_ratio = cmp.get("estimated_cost_mean_ratio_enforce_over_shadow")
    if cost_ratio is not None and cost_ratio > COST_REGRESSION_THRESHOLD:
        blockers.append("cost_regression")
    if _rate_inflated(enforce.get("escalation_task_rate"), shadow.get("escalation_task_rate")):
        blockers.append("escalation_rate_inflation")
    if _rate_inflated(enforce.get("plan_review_task_rate"), shadow.get("plan_review_task_rate")):
        blockers.append("review_rate_inflation")
    if (
        enforce.get("operational_error_or_missing", 0) > 0
        and enforce.get("operational_error_or_missing_rate", 0.0)
        > shadow.get("operational_error_or_missing_rate", 0.0)
    ):
        blockers.append("operational_error_cluster")

    if "telemetry_not_decision_ready" in blockers:
        status = "awaiting_telemetry"
    elif blockers == ["factuality_not_scored"]:
        status = "hold_quality_unscored"
    elif blockers == ["factuality_no_enforce_lift"]:
        status = "hold_quality_scored_no_lift"
    elif blockers:
        status = "hold"
    else:
        status = "promote_candidate"
    return {
        "status": status,
        "blockers": blockers,
        "notes": notes,
        "thresholds": {
            "max_p95_latency_ratio": LATENCY_REGRESSION_THRESHOLD,
            "max_estimated_cost_mean_ratio": COST_REGRESSION_THRESHOLD,
            "max_escalation_or_review_rate_ratio": RATE_INFLATION_THRESHOLD,
        },
    }


def build_report(
    log_dir: Path = DEFAULT_LOG_DIR,
    *,
    canary_start: str = DEFAULT_CANARY_START,
    telemetry_health_start: str = DEFAULT_TELEMETRY_HEALTH_START,
    decision_gate: int = DEFAULT_GATE,
    min_arm_samples: int = DEFAULT_MIN_ARM_SAMPLES,
    canary_roles: Iterable[str] = DEFAULT_CANARY_ROLES,
    scored_summary_path: Path | None = None,
) -> dict[str, Any]:
    canary_role_set = {str(role) for role in canary_roles if str(role)}
    routing_rows: list[dict[str, Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    escalations: dict[str, int] = {}
    plan_reviews: dict[str, list[str]] = {}

    for path, line_no, record in _iter_progress_records(log_dir):
        task_id = str(record.get("task_id") or "")
        if not task_id:
            continue
        event_type = str(record.get("event_type") or "")
        data = record.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if event_type == "routing_decision":
            mode = _factual_risk_mode(data)
            date = _date_key(str(record.get("timestamp") or path.stem))
            if (
                date >= telemetry_health_start
                and data.get("factual_risk_band") == "high"
                and mode in {"enforce", "shadow"}
                and _is_canary_participant(data, canary_role_set)
            ):
                routing_rows.append(
                    {
                        "task_id": task_id,
                        "timestamp": record.get("timestamp"),
                        "source": f"{path}:{line_no}",
                        "mode": mode,
                        "routing": sorted(_routing_roles(data)),
                        "risk_gate_action": str(data.get("risk_gate_action") or ""),
                        "estimated_cost": _float_value(data.get("estimated_cost")),
                        "factual_risk_score": _float_value(data.get("factual_risk_score")),
                    }
                )
        elif event_type in {"task_completed", "task_failed"}:
            outcomes[task_id] = record
        elif event_type == "escalation_triggered":
            escalations[task_id] = escalations.get(task_id, 0) + 1
        elif event_type == "plan_reviewed":
            decision = str(data.get("decision") or "<missing>")
            plan_reviews.setdefault(task_id, []).append(decision)

    rows_by_arm: dict[str, list[dict[str, Any]]] = {"enforce": [], "shadow": []}
    for row in routing_rows:
        outcome = outcomes.get(row["task_id"])
        row = dict(row)
        row["outcome_event_type"] = outcome.get("event_type") if outcome else ""
        row["outcome"] = outcome.get("outcome") if outcome else None
        row["success"] = _completion_success(outcome)
        row["failed"] = _completion_failed(outcome)
        row["latency_s"] = _extract_elapsed_seconds(outcome.get("outcome_details") if outcome else None)
        row["quality_score"] = _quality_value(outcome)
        row["escalation_event_count"] = escalations.get(row["task_id"], 0)
        row["plan_review_decisions"] = plan_reviews.get(row["task_id"], [])
        row["plan_review_event_count"] = len(row["plan_review_decisions"])
        rows_by_arm[row["mode"]].append(row)

    arm_summaries = {mode: _arm_summary(rows) for mode, rows in rows_by_arm.items()}
    cmp = _comparison(arm_summaries["enforce"], arm_summaries["shadow"])

    # Keep sample coverage local so one report is self-contained and generated
    # under identical role/date gates.
    from scripts.analysis import ri10_canary_sample_report as sample_report_mod

    sample_report = sample_report_mod.build_report(
        log_dir,
        canary_start=canary_start,
        telemetry_health_start=telemetry_health_start,
        decision_gate=decision_gate,
        min_arm_samples=min_arm_samples,
        canary_roles=canary_roles,
    )
    quality_evidence = _load_scored_quality_evidence(scored_summary_path)
    decision = _decision(sample_report, arm_summaries, cmp, quality_evidence)
    return {
        "generated_at": _iso_now(),
        "source_glob": str(log_dir / "*.jsonl"),
        "canary_start": canary_start,
        "telemetry_health_start": telemetry_health_start,
        "canary_roles": sorted(canary_role_set),
        "decision_gate_high_risk_samples": decision_gate,
        "min_canary_arm_samples": min_arm_samples,
        "decision": decision,
        "sample_coverage": {
            "canary_decision_ready": sample_report.get("canary_decision_ready"),
            "telemetry_collection_blocker": sample_report.get("telemetry_collection_blocker"),
            "telemetry_collection_reason": sample_report.get("telemetry_collection_reason"),
            "canary_arm_counts_since_telemetry_health_start": sample_report.get(
                "canary_arm_counts_since_telemetry_health_start"
            ),
            "canary_arm_counts_by_role_since_telemetry_health_start": sample_report.get(
                "canary_arm_counts_by_role_since_telemetry_health_start"
            ),
        },
        "arms": arm_summaries,
        "comparison": cmp,
        "quality_evidence": quality_evidence,
        "measurement_notes": [
            "This is an observational live-traffic canary comparison, not paired prompt A/B.",
            "Operational task success is reported separately from factuality/accuracy; it does not satisfy RI-10 factuality evidence.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    arms = report["arms"]
    cmp = report["comparison"]
    quality_evidence = report.get("quality_evidence") or {"status": "not_provided"}
    lines = [
        "# RI-10 Canary Decision Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Status: `{decision['status']}`",
        f"- Blockers: {', '.join(f'`{b}`' for b in decision['blockers']) or '`none`'}",
        f"- Sample coverage: `{report['sample_coverage']['telemetry_collection_blocker']}`",
        "",
        "## Arm Summary",
        "",
        "| Arm | Rows | Success | Error/missing | p50 s | p95 s | Mean cost | Escalation rate | Review rate | Quality rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("enforce", "shadow"):
        stats = arms[arm]
        lines.append(
            "| {arm} | {rows} | {success} | {err} | {p50} | {p95} | {cost} | {esc} | {rev} | {qual} |".format(
                arm=arm,
                rows=stats["rows"],
                success=stats["success"],
                err=stats["operational_error_or_missing"],
                p50=stats["latency_p50_s"],
                p95=stats["latency_p95_s"],
                cost=stats["estimated_cost_mean"],
                esc=stats["escalation_task_rate"],
                rev=stats["plan_review_task_rate"],
                qual=stats["quality_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- p95 latency ratio enforce/shadow: `{cmp['latency_p95_ratio_enforce_over_shadow']}`",
            f"- mean estimated-cost ratio enforce/shadow: `{cmp['estimated_cost_mean_ratio_enforce_over_shadow']}`",
            f"- operational error-rate delta: `{cmp['operational_error_rate_delta']}`",
            f"- escalation-rate ratio enforce/shadow: `{cmp['escalation_task_rate_ratio_enforce_over_shadow']}`",
            f"- review-rate ratio enforce/shadow: `{cmp['plan_review_task_rate_ratio_enforce_over_shadow']}`",
            f"- quality delta enforce-shadow: `{cmp['quality_mean_delta_enforce_minus_shadow']}`",
            "",
            "## Scored Factuality Evidence",
            "",
            f"- status: `{quality_evidence.get('status')}`",
        ]
    )
    if quality_evidence.get("source_path"):
        quality_cmp = quality_evidence.get("comparison") or {}
        quality_arms = quality_evidence.get("arms") or {}
        lines.extend(
            [
                f"- source: `{quality_evidence['source_path']}`",
                f"- rows: `{quality_evidence.get('rows')}`",
                f"- accuracy delta enforce-shadow: `{quality_cmp.get('accuracy_delta_enforce_minus_shadow')}`",
                f"- token-F1 delta enforce-shadow: `{quality_cmp.get('mean_token_f1_delta_enforce_minus_shadow')}`",
                "",
                "| Arm | Rows | Scored | Missing | Correct | Accuracy | Mean Token F1 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in ("enforce", "shadow"):
            stats = quality_arms.get(arm) or {}
            lines.append(
                "| {arm} | {rows} | {scored} | {missing} | {correct} | {accuracy} | {f1} |".format(
                    arm=arm,
                    rows=stats.get("rows"),
                    scored=stats.get("scored"),
                    missing=stats.get("missing"),
                    correct=stats.get("correct"),
                    accuracy=stats.get("accuracy"),
                    f1=stats.get("mean_token_f1"),
                )
            )
    lines.extend(
        [
            "",
            "## Measurement Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in report["measurement_notes"])
    lines.extend(f"- {note}" for note in decision["notes"])
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--classifier-config", type=Path, default=DEFAULT_CLASSIFIER_CONFIG)
    parser.add_argument("--canary-start", default=DEFAULT_CANARY_START)
    parser.add_argument("--telemetry-health-start", default=DEFAULT_TELEMETRY_HEALTH_START)
    parser.add_argument("--decision-gate", type=int, default=DEFAULT_GATE)
    parser.add_argument("--min-arm-samples", type=int, default=DEFAULT_MIN_ARM_SAMPLES)
    parser.add_argument("--canary-role", action="append", default=None)
    parser.add_argument("--all-canary-roles", action="store_true")
    parser.add_argument(
        "--scored-summary",
        type=Path,
        help="Attach an RI-10 scored response summary JSON as factuality evidence.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON report to this path.")
    parser.add_argument("--markdown-output", type=Path, help="Write Markdown report to this path.")
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
        scored_summary_path=args.scored_summary,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
