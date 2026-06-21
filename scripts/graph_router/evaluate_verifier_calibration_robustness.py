#!/usr/bin/env python3
"""Evaluate offline verifier calibration robustness across split seeds.

This is an offline A9/NEXT-A2 preparation tool. It repeatedly trains the
verifier head on disjoint train/calibration/test splits and summarizes whether
calibrated gates are stable enough to justify a later promotion-grade run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.train_verifier_head import train_and_eval

SCHEMA_VERSION = "verifier_calibration_robustness.v1"


def _parse_csv_ints(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return seeds


def _parse_csv_strings(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    if not methods:
        raise argparse.ArgumentTypeError("expected at least one method")
    allowed = {"temperature_bias", "ece_temperature_bias", "quantile_histogram", "isotonic"}
    invalid = sorted(set(methods) - allowed)
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported method(s): {', '.join(invalid)}")
    return methods


def _metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
    }


def aggregate_runs(
    runs: list[dict[str, Any]],
    min_calibrated_pass_rate: float,
    min_action_rows: int,
) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_method.setdefault(run["method"], []).append(run)

    methods: dict[str, Any] = {}
    for method, method_runs in sorted(by_method.items()):
        calibrated = [run["calibration"]["calibrated_verifier"] for run in method_runs]
        calibrated_gates = [run["calibration"]["gates"] for run in method_runs]
        raw_gates = [run["gates"] for run in method_runs]
        pass_count = sum(1 for gate in calibrated_gates if gate["pass"])
        raw_pass_count = sum(1 for gate in raw_gates if gate["pass"])
        total = len(method_runs)
        methods[method] = {
            "runs": total,
            "raw_pass_count": raw_pass_count,
            "raw_pass_rate": raw_pass_count / total,
            "calibrated_pass_count": pass_count,
            "calibrated_pass_rate": pass_count / total,
            "calibrated_brier": _metric_stats([m["brier"] for m in calibrated]),
            "calibrated_auc": _metric_stats([m["auc"] for m in calibrated]),
            "calibrated_ece": _metric_stats([m["ece"] for m in calibrated]),
            "calibrated_acc": _metric_stats([m["acc"] for m in calibrated]),
            "meets_pass_rate": (pass_count / total) >= min_calibrated_pass_rate,
        }

    action_counts: dict[str, int] = {}
    if runs:
        action_counts = {str(k): int(v) for k, v in runs[0]["action_counts"].items()}
    sparse_actions = {
        action: count
        for action, count in action_counts.items()
        if count < min_action_rows
    }
    blockers: list[str] = []
    for method, stats in methods.items():
        if not stats["meets_pass_rate"]:
            blockers.append(f"{method}_calibrated_pass_rate_below_threshold")
    if sparse_actions:
        blockers.append("sparse_action_coverage")

    return {
        "methods": methods,
        "action_counts": action_counts,
        "sparse_actions": sparse_actions,
        "criteria": {
            "min_calibrated_pass_rate": min_calibrated_pass_rate,
            "min_action_rows": min_action_rows,
        },
        "decision": {
            "status": "promotion_grade" if not blockers else "not_promotion_grade",
            "blockers": blockers,
        },
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Offline Verifier Calibration Robustness",
        "",
        f"- Data: `{summary['data_path']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Methods: `{summary['methods_requested']}`",
        f"- Training params: `{summary['training_params']}`",
        f"- Decision: `{summary['aggregate']['decision']['status']}`",
        f"- Blockers: `{summary['aggregate']['decision']['blockers']}`",
        f"- Action counts: `{summary['aggregate']['action_counts']}`",
        f"- Sparse actions: `{summary['aggregate']['sparse_actions']}`",
        "",
        "## Method Summary",
        "",
    ]
    for method, stats in summary["aggregate"]["methods"].items():
        lines.extend(
            [
                f"### `{method}`",
                "",
                f"- Runs: `{stats['runs']}`",
                f"- Raw pass count: `{stats['raw_pass_count']}`",
                f"- Calibrated pass count: `{stats['calibrated_pass_count']}`",
                f"- Calibrated pass rate: `{stats['calibrated_pass_rate']:.4f}`",
                f"- Calibrated Brier mean/range: `{stats['calibrated_brier']['mean']:.4f}` "
                f"(`{stats['calibrated_brier']['min']:.4f}`-`{stats['calibrated_brier']['max']:.4f}`)",
                f"- Calibrated ROC-AUC mean/range: `{stats['calibrated_auc']['mean']:.4f}` "
                f"(`{stats['calibrated_auc']['min']:.4f}`-`{stats['calibrated_auc']['max']:.4f}`)",
                f"- Calibrated ECE mean/range: `{stats['calibrated_ece']['mean']:.4f}` "
                f"(`{stats['calibrated_ece']['min']:.4f}`-`{stats['calibrated_ece']['max']:.4f}`)",
                "",
            ]
        )
    lines.extend(
        [
            "This is an offline robustness artifact. It does not adopt live",
            "verifier weights or enable a runtime verifier gate.",
            "",
        ]
    )
    return "\n".join(lines)


def run_robustness(args: argparse.Namespace) -> dict[str, Any]:
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        for method in args.methods:
            weight_path = weights_dir / f"verifier_{method}_seed{seed}.npz"
            result = train_and_eval(
                data_path=Path(args.data),
                classifier_weights_path=Path(args.classifier_weights),
                classifier_data_path=Path(args.classifier_data) if args.classifier_data else None,
                output_path=weight_path,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                patience=args.patience,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                val_seed=seed,
                val_split=args.val_split,
                calibration_split=args.calibration_split,
                test_split=args.test_split,
                calibration_method=method,
                calibration_bins=args.calibration_bins,
                calibration_alpha=args.calibration_alpha,
                normalize_features=args.normalize_features,
            )
            runs.append(
                {
                    "seed": seed,
                    "method": method,
                    "eval_split": result["eval_split"],
                    "train_rows": result["train_rows"],
                    "calibration_rows": result["calibration_rows"],
                    "test_rows": result["test_rows"],
                    "action_counts": result["action_counts"],
                    "verifier": result["verifier"],
                    "gates": result["gates"],
                    "calibration": result["calibration"],
                    "best_softmax_baseline_name": result["best_softmax_baseline_name"],
                    "brier_delta_vs_best_softmax_baseline": result[
                        "brier_delta_vs_best_softmax_baseline"
                    ],
                    "brier_delta_vs_constant_baseline": result[
                        "brier_delta_vs_constant_baseline"
                    ],
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "data_path": args.data,
        "classifier_weights_path": args.classifier_weights,
        "classifier_data_path": args.classifier_data,
        "seeds": args.seeds,
        "methods_requested": args.methods,
        "split": {
            "val_split": args.val_split,
            "calibration_split": args.calibration_split,
            "test_split": args.test_split,
        },
        "training_params": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "normalize_features": args.normalize_features,
        },
        "calibration_params": {
            "calibration_bins": args.calibration_bins,
            "calibration_alpha": args.calibration_alpha,
        },
        "runs": runs,
        "aggregate": aggregate_runs(
            runs,
            min_calibrated_pass_rate=args.min_calibrated_pass_rate,
            min_action_rows=args.min_action_rows,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate verifier calibration stability across split seeds",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--classifier-weights", required=True)
    parser.add_argument("--classifier-data")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument(
        "--weights-dir",
        default="/mnt/raid0/llm/tmp/a9-verifier-calibration-robustness-weights",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_csv_ints,
        default=_parse_csv_ints("42,7,13,101,2026,31415,2718,9001,123,55"),
    )
    parser.add_argument(
        "--methods",
        type=_parse_csv_strings,
        default=_parse_csv_strings("temperature_bias,quantile_histogram"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--calibration-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--calibration-bins", type=int, default=7)
    parser.add_argument("--calibration-alpha", type=float, default=0.0)
    parser.add_argument("--min-calibrated-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-action-rows", type=int, default=30)
    args = parser.parse_args()

    summary = run_robustness(args)
    json_path = Path(args.summary_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = Path(args.summary_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_summary_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
