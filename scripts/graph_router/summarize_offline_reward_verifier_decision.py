#!/usr/bin/env python3
"""Summarize A9 offline reward-verifier decision evidence.

This tool is intentionally offline-only. It reads existing robustness and
model-family summaries, then emits a stop/continue decision for the current
offline verifier family without training models or changing runtime gates.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "offline_reward_verifier_decision.v1"


@dataclass(frozen=True)
class Attempt:
    path: str
    schema_version: str
    data_path: str | None
    status: str
    best_label: str
    best_pass_count: int
    best_pass_rate: float
    best_auc_mean: float | None
    best_ece_mean: float | None
    best_brier_mean: float | None
    blockers: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_mean(stats: dict[str, Any], metric: str) -> float | None:
    value = stats.get(metric)
    if isinstance(value, dict):
        return _float_or_none(value.get("mean"))
    return None


def _attempt_sort_key(attempt: Attempt) -> tuple[float, float, float]:
    auc = attempt.best_auc_mean if attempt.best_auc_mean is not None else -1.0
    ece = attempt.best_ece_mean if attempt.best_ece_mean is not None else 1.0
    brier = attempt.best_brier_mean if attempt.best_brier_mean is not None else 1.0
    return (attempt.best_pass_rate, auc, -ece - brier)


def _best_from_calibration_summary(payload: dict[str, Any], path: Path) -> Attempt:
    aggregate = payload.get("aggregate", {})
    decision = aggregate.get("decision", {})
    methods = aggregate.get("methods", {})
    best_label = ""
    best_stats: dict[str, Any] = {}
    best_pass_count = -1
    best_pass_rate = -1.0
    for method, stats in methods.items():
        pass_count = int(stats.get("calibrated_pass_count", 0))
        pass_rate = float(stats.get("calibrated_pass_rate", 0.0))
        candidate = (pass_rate, pass_count, _metric_mean(stats, "calibrated_auc") or -1.0)
        current = (
            best_pass_rate,
            best_pass_count,
            _metric_mean(best_stats, "calibrated_auc") or -1.0,
        )
        if candidate > current:
            best_label = str(method)
            best_stats = stats
            best_pass_count = pass_count
            best_pass_rate = pass_rate
    return Attempt(
        path=str(path),
        schema_version=str(payload.get("schema_version", "")),
        data_path=payload.get("data_path"),
        status=str(decision.get("status", "unknown")),
        best_label=best_label,
        best_pass_count=max(best_pass_count, 0),
        best_pass_rate=max(best_pass_rate, 0.0),
        best_auc_mean=_metric_mean(best_stats, "calibrated_auc"),
        best_ece_mean=_metric_mean(best_stats, "calibrated_ece"),
        best_brier_mean=_metric_mean(best_stats, "calibrated_brier"),
        blockers=[str(item) for item in decision.get("blockers", [])],
    )


def _best_from_model_family_summary(payload: dict[str, Any], path: Path) -> Attempt:
    aggregate = payload.get("aggregate", {})
    decision = aggregate.get("decision", {})
    families = aggregate.get("families", {})
    best_label = ""
    best_stats: dict[str, Any] = {}
    best_pass_count = -1
    best_pass_rate = -1.0
    for family, family_stats in families.items():
        for method, stats in family_stats.get("methods", {}).items():
            pass_count = int(stats.get("pass_count", 0))
            pass_rate = float(stats.get("pass_rate", 0.0))
            candidate = (pass_rate, pass_count, _metric_mean(stats, "auc") or -1.0)
            current = (best_pass_rate, best_pass_count, _metric_mean(best_stats, "auc") or -1.0)
            if candidate > current:
                best_label = f"{family}:{method}"
                best_stats = stats
                best_pass_count = pass_count
                best_pass_rate = pass_rate
    return Attempt(
        path=str(path),
        schema_version=str(payload.get("schema_version", "")),
        data_path=payload.get("data_path"),
        status=str(decision.get("status", "unknown")),
        best_label=best_label,
        best_pass_count=max(best_pass_count, 0),
        best_pass_rate=max(best_pass_rate, 0.0),
        best_auc_mean=_metric_mean(best_stats, "auc"),
        best_ece_mean=_metric_mean(best_stats, "ece"),
        best_brier_mean=_metric_mean(best_stats, "brier"),
        blockers=[str(item) for item in decision.get("blockers", [])],
    )


def parse_attempt(path: Path) -> Attempt:
    payload = _read_json(path)
    schema = str(payload.get("schema_version", ""))
    if schema == "verifier_calibration_robustness.v1":
        return _best_from_calibration_summary(payload, path)
    if schema == "offline_reward_verifier_model_family_robustness.v1":
        return _best_from_model_family_summary(payload, path)
    raise ValueError(f"unsupported summary schema in {path}: {schema}")


def collect_summary_paths(summary_paths: list[str], summary_globs: list[str]) -> list[Path]:
    paths = [Path(item) for item in summary_paths]
    for pattern in summary_globs:
        paths.extend(Path(item) for item in sorted(glob.glob(pattern)))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


def build_decision(
    attempts: list[Attempt],
    *,
    min_failed_artifacts: int,
    generated_at: str,
) -> dict[str, Any]:
    if not attempts:
        raise ValueError("at least one summary artifact is required")
    promotion_grade = [attempt for attempt in attempts if attempt.status == "promotion_grade"]
    failed = [attempt for attempt in attempts if attempt.status != "promotion_grade"]
    best_attempt = max(attempts, key=_attempt_sort_key)
    schema_counts: dict[str, int] = {}
    for attempt in attempts:
        schema_counts[attempt.schema_version] = schema_counts.get(attempt.schema_version, 0) + 1

    has_model_family_sweep = "offline_reward_verifier_model_family_robustness.v1" in schema_counts
    has_calibration_sweep = "verifier_calibration_robustness.v1" in schema_counts
    enough_failures = len(failed) >= min_failed_artifacts

    blockers: list[str] = []
    if promotion_grade:
        status = "promotion_candidate"
        recommended_next = "review_promotion_artifact_before_runtime_gate"
    elif enough_failures and has_model_family_sweep and has_calibration_sweep:
        status = "stop_current_verifier_family"
        blockers.extend(
            [
                "no_promotion_grade_artifact",
                "calibration_sweeps_failed",
                "model_family_sweeps_failed",
            ]
        )
        recommended_next = "design_materially_different_reward_oracle_or_balanced_label_contract"
    else:
        status = "continue_evidence_collection"
        if not enough_failures:
            blockers.append("insufficient_failed_artifacts_for_stop_condition")
        if not has_calibration_sweep:
            blockers.append("missing_calibration_sweep")
        if not has_model_family_sweep:
            blockers.append("missing_model_family_sweep")
        recommended_next = "collect_required_summaries_before_stopping_current_family"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "criteria": {
            "min_failed_artifacts": min_failed_artifacts,
            "requires_calibration_sweep": True,
            "requires_model_family_sweep": True,
        },
        "artifact_counts": {
            "total": len(attempts),
            "failed": len(failed),
            "promotion_grade": len(promotion_grade),
            "by_schema": schema_counts,
        },
        "best_attempt": attempt_to_json(best_attempt),
        "attempts": [attempt_to_json(attempt) for attempt in sorted(attempts, key=lambda item: item.path)],
        "decision": {
            "status": status,
            "blockers": blockers,
            "recommended_next": recommended_next,
            "runtime_gate_change_allowed": bool(promotion_grade),
        },
        "guardrails": [
            "Do not enable a runtime verifier gate from a not_promotion_grade artifact.",
            "Do not spend further cycles retuning the same prompt/action verifier family after the stop condition fires.",
            "Next evidence should change the reward-oracle/label contract, not only the classifier family or calibrator.",
        ],
    }


def attempt_to_json(attempt: Attempt) -> dict[str, Any]:
    return {
        "path": attempt.path,
        "schema_version": attempt.schema_version,
        "data_path": attempt.data_path,
        "status": attempt.status,
        "best_label": attempt.best_label,
        "best_pass_count": attempt.best_pass_count,
        "best_pass_rate": round(attempt.best_pass_rate, 6),
        "best_auc_mean": None if attempt.best_auc_mean is None else round(attempt.best_auc_mean, 6),
        "best_ece_mean": None if attempt.best_ece_mean is None else round(attempt.best_ece_mean, 6),
        "best_brier_mean": None if attempt.best_brier_mean is None else round(attempt.best_brier_mean, 6),
        "blockers": attempt.blockers,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    best = summary["best_attempt"]
    lines = [
        "# Offline Reward Verifier Decision",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Decision: `{summary['decision']['status']}`",
        f"- Recommended next: `{summary['decision']['recommended_next']}`",
        f"- Runtime gate change allowed: `{summary['decision']['runtime_gate_change_allowed']}`",
        f"- Artifacts: `{summary['artifact_counts']['total']}` total, "
        f"`{summary['artifact_counts']['failed']}` failed, "
        f"`{summary['artifact_counts']['promotion_grade']}` promotion-grade",
        f"- Best attempt: `{best['best_label']}` from `{best['path']}`",
        f"- Best pass rate: `{best['best_pass_rate']}`",
        f"- Best AUC/ECE/Brier means: `{best['best_auc_mean']}` / "
        f"`{best['best_ece_mean']}` / `{best['best_brier_mean']}`",
        "",
        "## Guardrails",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["guardrails"])
    lines.extend(["", "## Attempt Summary", ""])
    for attempt in summary["attempts"]:
        lines.extend(
            [
                f"### `{Path(attempt['path']).name}`",
                "",
                f"- Status: `{attempt['status']}`",
                f"- Schema: `{attempt['schema_version']}`",
                f"- Best label: `{attempt['best_label']}`",
                f"- Best pass count/rate: `{attempt['best_pass_count']}` / "
                f"`{attempt['best_pass_rate']}`",
                f"- AUC/ECE/Brier means: `{attempt['best_auc_mean']}` / "
                f"`{attempt['best_ece_mean']}` / `{attempt['best_brier_mean']}`",
                f"- Blockers: `{attempt['blockers']}`",
                "",
            ]
        )
    lines.extend(
        [
            "This is an offline decision artifact. It does not train weights,",
            "change classifier configuration, or enable a runtime verifier gate.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = collect_summary_paths(args.summary, args.summary_glob)
    attempts = [parse_attempt(path) for path in paths]
    generated_at = args.generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    summary = build_decision(
        attempts,
        min_failed_artifacts=args.min_failed_artifacts,
        generated_at=generated_at,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", default=[], help="Summary JSON path")
    parser.add_argument("--summary-glob", action="append", default=[], help="Glob for summary JSONs")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--generated-at")
    parser.add_argument("--min-failed-artifacts", type=int, default=3)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
