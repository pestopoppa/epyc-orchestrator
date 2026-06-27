#!/usr/bin/env python3
"""Offline IRT cold-start subset audit against a full baseline artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.irt_scorer import predict_irt_from_embeddings


@dataclass(frozen=True)
class BaselineRecord:
    suite: str
    question_key: str
    question_id: str
    prompt: str
    prompt_hash: str
    algorithmic_score: float | None
    tokens_per_second: float | None
    total_time_ms: float | None


@dataclass(frozen=True)
class ScoredRecord:
    record: BaselineRecord
    latent_difficulty: float
    latent_discrimination: float


ACCEPTANCE_METRICS = ("avg_algorithmic_score", "avg_tokens_per_second", "pass_rate")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_baseline_records(path: Path) -> tuple[dict[str, Any], list[BaselineRecord]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[BaselineRecord] = []
    for suite, suite_records in data.get("results", {}).items():
        if not isinstance(suite_records, dict):
            continue
        for question_key, raw in suite_records.items():
            if not isinstance(raw, dict):
                continue
            prompt = str(raw.get("prompt") or "")
            records.append(
                BaselineRecord(
                    suite=str(suite),
                    question_key=str(question_key),
                    question_id=str(raw.get("question_id") or question_key),
                    prompt=prompt,
                    prompt_hash=prompt_hash(prompt),
                    algorithmic_score=_as_float(raw.get("algorithmic_score")),
                    tokens_per_second=_as_float(raw.get("tokens_per_second")),
                    total_time_ms=_as_float(raw.get("total_time_ms")),
                )
            )
    return data, records


def load_prompt_embeddings(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data.files:
        raise ValueError(f"{path} does not contain an embeddings array")
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if "prompt_hashes" in data.files:
        keys = [str(item) for item in data["prompt_hashes"].tolist()]
    elif "question_ids" in data.files:
        keys = [str(item) for item in data["question_ids"].tolist()]
    else:
        raise ValueError(f"{path} must contain prompt_hashes or question_ids")
    if len(keys) != embeddings.shape[0]:
        raise ValueError("embedding key count does not match embedding rows")
    return {key: embeddings[idx] for idx, key in enumerate(keys)}


def load_keyed_irt_scores(path: Path) -> dict[str, tuple[float, float]]:
    data = np.load(path, allow_pickle=True)
    if "latent_difficulty" not in data.files or "latent_discrimination" not in data.files:
        raise ValueError(f"{path} does not contain latent IRT score arrays")
    if "prompt_hashes" in data.files:
        keys = [str(item) for item in data["prompt_hashes"].tolist()]
    elif "question_ids" in data.files:
        keys = [str(item) for item in data["question_ids"].tolist()]
    else:
        return {}
    difficulty = np.asarray(data["latent_difficulty"], dtype=np.float32)
    discrimination = np.asarray(data["latent_discrimination"], dtype=np.float32)
    if len(keys) != difficulty.shape[0]:
        raise ValueError("IRT key count does not match score rows")
    return {
        key: (float(difficulty[idx]), float(discrimination[idx]))
        for idx, key in enumerate(keys)
    }


def score_records(
    records: list[BaselineRecord],
    *,
    irt_scores_path: Path,
    prompt_embeddings_path: Path | None = None,
) -> list[ScoredRecord]:
    keyed_scores = load_keyed_irt_scores(irt_scores_path)
    scored: list[ScoredRecord] = []
    missing: list[str] = []
    for record in records:
        score = keyed_scores.get(record.prompt_hash) or keyed_scores.get(record.question_id)
        if score is None:
            missing.append(record.prompt_hash)
            continue
        scored.append(ScoredRecord(record, score[0], score[1]))

    if scored or prompt_embeddings_path is None:
        return scored

    embeddings_by_key = load_prompt_embeddings(prompt_embeddings_path)
    irt = np.load(irt_scores_path, allow_pickle=True)
    required = {
        "embedding_dim",
        "projector_feature_mean",
        "projector_feature_scale",
        "projector_weights",
        "projector_target_mean",
        "projector_target_scale",
    }
    if not required.issubset(set(irt.files)):
        raise ValueError("IRT artifact is not keyed and does not include a persisted embedding projector")

    matched_records: list[BaselineRecord] = []
    embeddings: list[np.ndarray] = []
    for record in records:
        embedding = embeddings_by_key.get(record.prompt_hash)
        if embedding is None:
            embedding = embeddings_by_key.get(record.question_id)
        if embedding is None:
            continue
        matched_records.append(record)
        embeddings.append(embedding)
    if not embeddings:
        return []

    difficulty, discrimination = predict_irt_from_embeddings(
        np.stack(embeddings),
        embedding_dim=int(irt["embedding_dim"]),
        feature_mean=irt["projector_feature_mean"],
        feature_scale=irt["projector_feature_scale"],
        weights=irt["projector_weights"],
        target_mean=irt["projector_target_mean"],
        target_scale=irt["projector_target_scale"],
    )
    return [
        ScoredRecord(record, float(difficulty[idx]), float(discrimination[idx]))
        for idx, record in enumerate(matched_records)
    ]


def select_irt_stratified(
    scored: list[ScoredRecord],
    *,
    sample_size: int = 50,
    difficulty_bins: int = 5,
) -> list[ScoredRecord]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if difficulty_bins <= 0:
        raise ValueError("difficulty_bins must be positive")
    if len(scored) <= sample_size:
        return sorted(scored, key=lambda item: (-item.latent_discrimination, item.record.prompt_hash))

    ordered = sorted(scored, key=lambda item: (item.latent_difficulty, item.record.prompt_hash))
    bins = np.array_split(np.array(ordered, dtype=object), min(difficulty_bins, len(ordered)))
    selected: list[ScoredRecord] = []
    base_quota = sample_size // len(bins)
    remainder = sample_size % len(bins)
    for idx, bin_items in enumerate(bins):
        quota = base_quota + (1 if idx < remainder else 0)
        ranked = sorted(bin_items.tolist(), key=lambda item: (-item.latent_discrimination, item.record.prompt_hash))
        selected.extend(ranked[:quota])

    if len(selected) < sample_size:
        selected_hashes = {item.record.prompt_hash for item in selected}
        fill = [
            item for item in sorted(scored, key=lambda item: (-item.latent_discrimination, item.record.prompt_hash))
            if item.record.prompt_hash not in selected_hashes
        ]
        selected.extend(fill[: sample_size - len(selected)])
    return selected[:sample_size]


def summarize(records: list[BaselineRecord]) -> dict[str, Any]:
    scores = [record.algorithmic_score for record in records if record.algorithmic_score is not None]
    speeds = [record.tokens_per_second for record in records if record.tokens_per_second is not None]
    total_times = [record.total_time_ms for record in records if record.total_time_ms is not None]
    passed = sum(1 for score in scores if score is not None and score >= 2.5)
    return {
        "questions_tested": len(records),
        "score_count": len(scores),
        "speed_count": len(speeds),
        "total_time_ms_count": len(total_times),
        "avg_algorithmic_score": float(np.mean(scores)) if scores else None,
        "avg_tokens_per_second": float(np.mean(speeds)) if speeds else None,
        "avg_total_time_ms": float(np.mean(total_times)) if total_times else None,
        "questions_passed": passed,
        "sum_total_time_ms": float(np.sum(total_times)) if total_times else None,
        "pass_rate": passed / len(scores) if scores else None,
    }


def compare_summary(full: dict[str, Any], subset: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("avg_algorithmic_score", "avg_tokens_per_second", "pass_rate"):
        full_value = full.get(key)
        subset_value = subset.get(key)
        if full_value is None or subset_value is None:
            metrics[key] = {"full": full_value, "subset": subset_value, "abs_error": None, "rel_error": None}
            continue
        abs_error = abs(float(subset_value) - float(full_value))
        rel_error = abs_error / abs(float(full_value)) if float(full_value) != 0 else None
        metrics[key] = {
            "full": full_value,
            "subset": subset_value,
            "abs_error": abs_error,
            "rel_error": rel_error,
        }
    return metrics


def _parse_acceptance_metrics(raw: str) -> tuple[str, ...]:
    metrics = tuple(item.strip() for item in raw.split(",") if item.strip())
    return metrics or ACCEPTANCE_METRICS


def evaluate_acceptance(
    full: dict[str, Any],
    subset: dict[str, Any],
    *,
    metrics: tuple[str, ...] = ACCEPTANCE_METRICS,
    max_relative_error: float = 0.05,
    min_speedup_ratio: float = 5.0,
) -> dict[str, Any]:
    metric_checks: dict[str, Any] = {}
    for key in metrics:
        full_value = full.get(key)
        subset_value = subset.get(key)
        if full_value is None or subset_value is None:
            metric_checks[key] = {
                "full": full_value,
                "subset": subset_value,
                "abs_error": None,
                "rel_error": None,
                "within_threshold": False,
            }
            continue
        full_value_float = float(full_value)
        subset_value_float = float(subset_value)
        abs_error = abs(subset_value_float - full_value_float)
        rel_error = abs_error / abs(full_value_float) if full_value_float else (0.0 if abs_error == 0.0 else None)
        metric_checks[key] = {
            "full": full_value_float,
            "subset": subset_value_float,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "within_threshold": rel_error is not None and rel_error <= max_relative_error,
        }

    full_total_ms = full.get("sum_total_time_ms")
    subset_total_ms = subset.get("sum_total_time_ms")
    speedup: float | None
    if full_total_ms is None or subset_total_ms is None:
        speedup = None
    elif full_total_ms <= 0 or subset_total_ms <= 0:
        speedup = None
    else:
        speedup = float(full_total_ms) / float(subset_total_ms)

    all_metrics_within = all(
        details["within_threshold"] for details in metric_checks.values()
    )
    meets_speedup = speedup is not None and speedup >= min_speedup_ratio

    if not metric_checks:
        status = "blocked_missing_acceptance_metrics"
    elif speedup is None:
        status = "blocked_missing_wall_clock"
    elif all_metrics_within and meets_speedup:
        status = "accepted"
    else:
        status = "rejected"

    return {
        "status": status,
        "max_relative_error": max_relative_error,
        "min_speedup_ratio": min_speedup_ratio,
        "metric_checks": metric_checks,
        "speedup_ratio": speedup,
        "meets_speedup_threshold": meets_speedup,
        "meets_metric_thresholds": all_metrics_within,
    }


def run_audit(
    baseline_path: Path,
    irt_scores_path: Path,
    *,
    prompt_embeddings_path: Path | None = None,
    sample_size: int = 50,
    difficulty_bins: int = 5,
    enable_acceptance_gate: bool = False,
    acceptance_metrics: tuple[str, ...] | str = ACCEPTANCE_METRICS,
    max_relative_error: float = 0.05,
    min_speedup_ratio: float = 5.0,
) -> dict[str, Any]:
    baseline, records = load_baseline_records(baseline_path)
    scored = score_records(
        records,
        irt_scores_path=irt_scores_path,
        prompt_embeddings_path=prompt_embeddings_path,
    )
    if not scored:
        return {
            "status": "blocked_missing_irt_scores",
            "baseline_path": str(baseline_path),
            "irt_scores_path": str(irt_scores_path),
            "prompt_embeddings_path": str(prompt_embeddings_path) if prompt_embeddings_path else None,
            "full_records": len(records),
            "scored_records": 0,
            "reason": "No baseline prompts matched keyed IRT scores or prompt embeddings.",
        }

    selected = select_irt_stratified(scored, sample_size=sample_size, difficulty_bins=difficulty_bins)
    full_summary = summarize(records)
    subset_summary = summarize([item.record for item in selected])
    comparison = compare_summary(full_summary, subset_summary)
    report: dict[str, Any] = {
        "status": "ok",
        "model_role": baseline.get("model_role"),
        "baseline_path": str(baseline_path),
        "irt_scores_path": str(irt_scores_path),
        "prompt_embeddings_path": str(prompt_embeddings_path) if prompt_embeddings_path else None,
        "full_records": len(records),
        "scored_records": len(scored),
        "sample_size": len(selected),
        "difficulty_bins": difficulty_bins,
        "full_summary": full_summary,
        "subset_summary": subset_summary,
        "comparison": comparison,
        "selected": [
            {
                **asdict(item.record),
                "latent_difficulty": item.latent_difficulty,
                "latent_discrimination": item.latent_discrimination,
            }
            for item in selected
        ],
    }
    if enable_acceptance_gate:
        parsed_metrics = (
            _parse_acceptance_metrics(acceptance_metrics)
            if isinstance(acceptance_metrics, str)
            else tuple(acceptance_metrics)
        )
        acceptance = evaluate_acceptance(
            full_summary,
            subset_summary,
            metrics=parsed_metrics,
            max_relative_error=max_relative_error,
            min_speedup_ratio=min_speedup_ratio,
        )
        report["acceptance"] = acceptance
        report["status"] = acceptance["status"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline IRT cold-start A/B audit")
    parser.add_argument("--baseline", type=Path, required=True, help="Full baseline JSON artifact")
    parser.add_argument("--irt-scores", type=Path, required=True, help="IRT scores NPZ artifact")
    parser.add_argument("--prompt-embeddings", type=Path, default=None, help="Optional keyed prompt embeddings NPZ")
    parser.add_argument("--sample-size", type=int, default=50, help="IRT-stratified subset size")
    parser.add_argument("--difficulty-bins", type=int, default=5, help="Difficulty strata count")
    parser.add_argument("--acceptance-gate", action="store_true", help="Enable production gate status for full-vs-subset A/B")
    parser.add_argument(
        "--acceptance-metrics",
        type=str,
        default=",".join(ACCEPTANCE_METRICS),
        help="Comma-separated metrics to gate against during acceptance mode",
    )
    parser.add_argument(
        "--max-relative-error",
        type=float,
        default=0.05,
        help="Maximum tolerated relative error for gated metrics",
    )
    parser.add_argument(
        "--min-speedup-ratio",
        type=float,
        default=5.0,
        help="Minimum full-vs-subset wall-clock speedup ratio for gated acceptance",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report = run_audit(
        args.baseline,
        args.irt_scores,
        prompt_embeddings_path=args.prompt_embeddings,
        sample_size=args.sample_size,
        difficulty_bins=args.difficulty_bins,
        enable_acceptance_gate=args.acceptance_gate,
        acceptance_metrics=_parse_acceptance_metrics(args.acceptance_metrics),
        max_relative_error=args.max_relative_error,
        min_speedup_ratio=args.min_speedup_ratio,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
