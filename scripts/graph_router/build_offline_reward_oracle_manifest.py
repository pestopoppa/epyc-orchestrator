#!/usr/bin/env python3
"""Build an adoption manifest for decision-grade offline reward oracles.

The manifest is an explicit handoff artifact for learned-routing verifier
training. It is intentionally offline-only: a report must already pass the
offline decision gate before this script will write an adoption manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = "offline_reward_oracle_adoption_manifest.v1"
EVAL_SCHEMA_VERSION = "offline_reward_oracle_eval.v1"
DECISION_GATE_SCHEMA_VERSION = "offline_reward_oracle_decision_gate.v1"
ADOPTABLE_STATUS = "adoptable_offline_oracle"
REQUIRED_DECISION_STATUS = "decision_grade"
DEFAULT_INTENDED_USE = (
    "offline NEXT-A2/A3 reward-signal labeling and verifier-target "
    "preparation only"
)
FORBIDDEN_USE = (
    "live routing, serve-time request gating, production policy flips, or "
    "online reward updates without a separate deployment gate"
)


class ManifestError(ValueError):
    """Raised when an oracle report is not eligible for adoption."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: expected a JSON object")
    return value


def _expect_schema(
    payload: dict[str, Any],
    *,
    path: Path,
    expected: str,
    label: str,
) -> None:
    actual = payload.get("schema_version")
    if actual != expected:
        raise ManifestError(
            f"{path}: expected {label} schema_version={expected!r}, got {actual!r}"
        )


def _require_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    path: Path,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: expected object at {key!r}")
    return value


def _require_number(payload: dict[str, Any], key: str, *, path: Path) -> int | float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ManifestError(f"{path}: expected numeric {key!r}")
    return value


def _required_score_summary(
    score_summary: dict[str, Any],
    *,
    score_summary_path: Path,
) -> dict[str, Any]:
    model_id = score_summary.get("model_id")
    source = score_summary.get("oracle_score_source")
    rows = score_summary.get("rows")
    if not isinstance(model_id, str) or not model_id:
        raise ManifestError(f"{score_summary_path}: model_id is required")
    if not isinstance(source, str) or not source:
        raise ManifestError(f"{score_summary_path}: oracle_score_source is required")
    if not isinstance(rows, int) or rows <= 0:
        raise ManifestError(f"{score_summary_path}: positive integer rows is required")
    return {
        "model_id": model_id,
        "oracle_score_source": source,
        "rows": rows,
        "schema_version": score_summary.get("schema_version"),
        "score_definition": score_summary.get("score_definition"),
        "score_min": score_summary.get("score_min"),
        "score_max": score_summary.get("score_max"),
        "score_mean": score_summary.get("score_mean"),
        "stats": score_summary.get("stats", {}),
    }


def _best_threshold(eval_summary: dict[str, Any], *, eval_path: Path) -> float:
    calibration = _require_mapping(eval_summary, "calibration", path=eval_path)
    best = _require_mapping(calibration, "best", path=eval_path)
    balanced = _require_mapping(best, "balanced_accuracy", path=eval_path)
    threshold = _require_number(balanced, "threshold", path=eval_path)
    return float(threshold)


def _build_required_target_sources(
    eval_summary: dict[str, Any],
    *,
    eval_path: Path,
) -> dict[str, Any]:
    decision_gate = _require_mapping(eval_summary, "decision_gate", path=eval_path)
    slice_checks = _require_mapping(decision_gate, "slice_checks", path=eval_path)
    criteria = _require_mapping(decision_gate, "criteria", path=eval_path)
    required = criteria.get("required_target_sources", {})
    if not isinstance(required, dict):
        raise ManifestError(
            f"{eval_path}: decision_gate.criteria.required_target_sources must be an object"
        )
    slices = _require_mapping(eval_summary, "slices", path=eval_path)
    target_source_slices = _require_mapping(slices, "target_source", path=eval_path)
    result: dict[str, Any] = {}
    for source in sorted(required):
        checks = slice_checks.get(source)
        metrics = target_source_slices.get(source)
        if not isinstance(checks, dict):
            raise ManifestError(f"{eval_path}: missing slice checks for {source!r}")
        if not isinstance(metrics, dict):
            raise ManifestError(f"{eval_path}: missing target_source slice for {source!r}")
        result[source] = {
            "checks": checks,
            "metrics": {
                "n": metrics.get("n"),
                "target_positive": metrics.get("target_positive"),
                "target_negative": metrics.get("target_negative"),
                "agreement_at_threshold": metrics.get("agreement_at_threshold"),
                "spearman": metrics.get("spearman"),
                "confusion": metrics.get("confusion"),
            },
        }
    return result


def build_manifest(
    eval_summary: dict[str, Any],
    score_summary: dict[str, Any],
    *,
    eval_path: Path,
    score_summary_path: Path,
    intended_use: str = DEFAULT_INTENDED_USE,
) -> dict[str, Any]:
    _expect_schema(
        eval_summary,
        path=eval_path,
        expected=EVAL_SCHEMA_VERSION,
        label="eval",
    )
    decision_gate = _require_mapping(eval_summary, "decision_gate", path=eval_path)
    _expect_schema(
        decision_gate,
        path=eval_path,
        expected=DECISION_GATE_SCHEMA_VERSION,
        label="decision_gate",
    )
    decision_status = decision_gate.get("status")
    if decision_status != REQUIRED_DECISION_STATUS:
        blockers = decision_gate.get("blockers", [])
        raise ManifestError(
            f"{eval_path}: decision_gate.status must be "
            f"{REQUIRED_DECISION_STATUS!r}, got {decision_status!r}; "
            f"blockers={blockers!r}"
        )

    score_identity = _required_score_summary(
        score_summary,
        score_summary_path=score_summary_path,
    )
    eval_rows = _require_number(eval_summary, "n", path=eval_path)
    if int(eval_rows) != score_identity["rows"]:
        raise ManifestError(
            f"row-count mismatch: eval n={eval_rows!r}, "
            f"score_summary rows={score_identity['rows']!r}"
        )

    score = _require_mapping(eval_summary, "score", path=eval_path)
    stress = _require_mapping(eval_summary, "stress", path=eval_path)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": ADOPTABLE_STATUS,
        "oracle": {
            "model_id": score_identity["model_id"],
            "oracle_score_source": score_identity["oracle_score_source"],
            "oracle_threshold": _best_threshold(eval_summary, eval_path=eval_path),
            "score_schema_version": score_identity["schema_version"],
            "score_definition": score_identity["score_definition"],
            "score_summary": {
                "rows": score_identity["rows"],
                "score_min": score_identity["score_min"],
                "score_max": score_identity["score_max"],
                "score_mean": score_identity["score_mean"],
                "stats": score_identity["stats"],
            },
        },
        "evidence": {
            "eval_json": str(eval_path),
            "score_summary_json": str(score_summary_path),
            "eval_schema_version": eval_summary["schema_version"],
            "rows": int(eval_rows),
            "target_positive": eval_summary.get("target_positive"),
            "target_negative": eval_summary.get("target_negative"),
            "score": {
                "agreement_at_threshold": score.get("agreement_at_threshold"),
                "spearman": score.get("spearman"),
                "pearson": score.get("pearson"),
                "mean_abs_error": score.get("mean_abs_error"),
                "confusion": score.get("confusion"),
            },
            "decision_gate": {
                "status": decision_status,
                "criteria": decision_gate.get("criteria", {}),
                "checks": decision_gate.get("checks", {}),
                "blockers": decision_gate.get("blockers", []),
            },
            "required_target_sources": _build_required_target_sources(
                eval_summary,
                eval_path=eval_path,
            ),
            "stress": {
                "groups_evaluated": stress.get("groups_evaluated"),
                "paraphrase_total": stress.get("paraphrase_total"),
                "paraphrase_penalized": stress.get("paraphrase_penalized"),
                "paraphrase_penalty_rate": stress.get("paraphrase_penalty_rate"),
                "confound_total": stress.get("confound_total"),
                "confound_fooled": stress.get("confound_fooled"),
                "confound_fooled_rate": stress.get("confound_fooled_rate"),
                "variant_counts": stress.get("variant_counts", {}),
            },
        },
        "privacy": {
            "commits_private_rows": False,
            "row_text_fields_excluded": ["prompt", "reference", "response", "answer"],
            "note": (
                "This manifest references aggregate reports only; scored JSONL rows "
                "remain private input/output artifacts."
            ),
        },
        "intended_use": intended_use,
        "forbidden_use": FORBIDDEN_USE,
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed adoption manifest for an offline reward oracle",
    )
    parser.add_argument("--eval-json", required=True, type=Path)
    parser.add_argument("--score-summary-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--intended-use", default=DEFAULT_INTENDED_USE)
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(
            load_json(args.eval_json),
            load_json(args.score_summary_json),
            eval_path=args.eval_json,
            score_summary_path=args.score_summary_json,
            intended_use=args.intended_use,
        )
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_manifest(manifest, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
