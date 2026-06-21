#!/usr/bin/env python3
"""Evaluate offline reward verifier model families on the same NPZ contract.

This is an offline A9 diagnostic. It does not write verifier weights and does
not enable any runtime gate. The goal is to distinguish "bad feature/evidence
contract" from "bad MLP-head model family" after calibration-only repairs fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.routing_classifier import RoutingClassifier
from scripts.graph_router.train_verifier_head import (
    _apply_isotonic_calibrator,
    _apply_quantile_histogram_calibrator,
    _apply_temperature_bias_calibrator,
    _brier,
    _fit_isotonic_calibrator,
    _fit_quantile_histogram_calibrator,
    _fit_temperature_bias_calibrator,
    _metrics,
    _roc_auc,
)

SCHEMA_VERSION = "offline_reward_verifier_model_family_robustness.v1"
DEFAULT_SEEDS = [42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]
MODEL_FAMILIES = ("logistic_l2", "hist_gradient_boosting", "random_forest", "mlp_sklearn")
CALIBRATION_METHODS = ("temperature_bias", "ece_temperature_bias", "quantile_histogram", "isotonic")
SOURCE_FAMILIES = ("orchestrator_live_seed", "seeding_eval", "three_way_eval", "other")


def _parse_csv(value: str, *, allowed: tuple[str, ...] | None = None) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one value")
    if allowed is not None:
        invalid = sorted(set(items) - set(allowed))
        if invalid:
            raise argparse.ArgumentTypeError(f"unsupported value(s): {', '.join(invalid)}")
    return items


def _parse_csv_ints(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return seeds


def _metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {"min": min(values), "max": max(values), "mean": mean(values)}


def _source_family(source_path: str) -> str:
    path = Path(source_path)
    stem = path.stem.lower()
    parts = {part.lower() for part in path.parts}
    if "orchestrator" in parts and stem.startswith("seeding_live"):
        return "orchestrator_live_seed"
    if stem.startswith("seeding_"):
        return "seeding_eval"
    if stem.startswith("3way_"):
        return "three_way_eval"
    return "other"


def _split_indices(
    n_rows: int,
    *,
    seed: int,
    calibration_split: float,
    test_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    n_test = int(n_rows * test_split)
    n_cal = int(n_rows * calibration_split)
    if n_test < 2 or n_cal < 2:
        raise ValueError("calibration/test split requires at least two rows each")
    if n_test + n_cal >= n_rows:
        raise ValueError("calibration/test splits leave no rows for training")
    return idx[n_test + n_cal :], idx[n_test : n_test + n_cal], idx[:n_test]


def _model_for_family(family: str, seed: int) -> Any:
    if family == "logistic_l2":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            l2_regularization=0.01,
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            random_state=seed,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if family == "mlp_sklearn":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-4,
            early_stopping=True,
            max_iter=500,
            random_state=seed,
        )
    raise ValueError(f"unsupported model family: {family}")


def _fit_model(model: Any, x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> Any:
    try:
        return model.fit(x_train, y_train, sample_weight=weights)
    except TypeError:
        return model.fit(x_train, y_train)


def _predict_positive(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = [int(value) for value in model.classes_.tolist()]
        if 1 not in classes:
            return np.zeros(x.shape[0], dtype=np.float32)
        return proba[:, classes.index(1)].astype(np.float32)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x).astype(np.float32)
        return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)
    raise ValueError(f"model {type(model).__name__} cannot emit probabilities")


def _calibrate(
    method: str,
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    *,
    calibration_bins: int,
    calibration_alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method in {"temperature_bias", "ece_temperature_bias"}:
        objective = "ece" if method == "ece_temperature_bias" else "nll"
        calibrator = _fit_temperature_bias_calibrator(p_cal, y_cal, objective=objective)
        return _apply_temperature_bias_calibrator(p_test, calibrator), {
            "method": "ece_temperature_bias_grid" if objective == "ece" else "temperature_bias_grid",
            "temperature": calibrator["temperature"],
            "bias": calibrator["bias"],
            "calibration_objective": calibrator["objective"],
            "calibration_nll": calibrator["nll"],
            "calibration_brier": calibrator["brier"],
            "calibration_ece": calibrator["ece"],
        }
    if method == "quantile_histogram":
        calibrator = _fit_quantile_histogram_calibrator(
            p_cal,
            y_cal,
            n_bins=calibration_bins,
            smoothing_alpha=calibration_alpha,
        )
        return _apply_quantile_histogram_calibrator(p_test, calibrator), calibrator
    if method == "isotonic":
        calibrator = _fit_isotonic_calibrator(p_cal, y_cal)
        return _apply_isotonic_calibrator(p_test, calibrator), calibrator
    raise ValueError(f"unsupported calibration method: {method}")


def _gates(metrics: dict[str, float], brier_delta: float) -> dict[str, bool]:
    return {
        "brier_delta_ge_0_02": bool(brier_delta >= 0.02),
        "auc_ge_0_75": bool(metrics["auc"] >= 0.75),
        "ece_le_0_05": bool(metrics["ece"] <= 0.05),
        "pass": bool(
            (brier_delta >= 0.02)
            and (metrics["auc"] >= 0.75)
            and (metrics["ece"] <= 0.05)
        ),
    }


def _stratum_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    strata: list[str],
    *,
    min_rows: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stratum in sorted(set(strata)):
        indexes = np.asarray([idx for idx, value in enumerate(strata) if value == stratum], dtype=np.int64)
        y = labels[indexes]
        p = probs[indexes]
        entry: dict[str, Any] = {
            "rows": int(indexes.shape[0]),
            "positive_rows": int(y.sum()),
            "negative_rows": int(indexes.shape[0] - int(y.sum())),
        }
        if indexes.shape[0] >= min_rows and len(set(int(v) for v in y.tolist())) == 2:
            entry["metrics"] = _metrics(p, y)
        else:
            entry["metrics"] = None
        out[stratum] = entry
    return out


def _load_classifier_baseline(
    classifier_weights_path: Path,
    data: Any,
    z: np.ndarray,
    feature_dim: int,
    actions: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[dict[str, float], float, str]:
    classifier_feature_dim = (
        int(data["classifier_feature_dim"]) if "classifier_feature_dim" in data.files else feature_dim
    )
    clf = RoutingClassifier.load(classifier_weights_path)
    if clf is None:
        raise ValueError(f"failed to load classifier weights from {classifier_weights_path}")
    actions_eval = actions[eval_idx]
    if len(actions_eval) and int(actions_eval.max()) >= clf.n_actions:
        raise ValueError(
            f"classifier has {clf.n_actions} actions but eval labels include action {int(actions_eval.max())}"
        )
    x_clf = z[:, :classifier_feature_dim].astype(np.float32)
    probs, _ = clf.forward(x_clf[eval_idx])
    softmax_taken = probs[np.arange(len(actions_eval)), actions_eval]
    softmax_max = probs.max(axis=1)
    labels = data["correct"].astype(np.float32)[eval_idx]
    taken = {
        "brier": _brier(softmax_taken, labels),
        "auc": _roc_auc(softmax_taken, labels.astype(int)),
    }
    maxed = {
        "brier": _brier(softmax_max, labels),
        "auc": _roc_auc(softmax_max, labels.astype(int)),
    }
    if taken["brier"] <= maxed["brier"]:
        return taken, taken["brier"], "softmax_taken"
    return maxed, maxed["brier"], "softmax_max"


def aggregate_runs(
    runs: list[dict[str, Any]],
    *,
    min_calibrated_pass_rate: float,
) -> dict[str, Any]:
    by_family_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        by_family_method.setdefault((run["family"], run["method"]), []).append(run)

    families: dict[str, Any] = {}
    blockers: list[str] = []
    for (family, method), group in sorted(by_family_method.items()):
        fam = families.setdefault(family, {"methods": {}})
        pass_count = sum(1 for run in group if run["gates"]["pass"])
        pass_rate = pass_count / len(group)
        fam["methods"][method] = {
            "runs": len(group),
            "pass_count": pass_count,
            "pass_rate": pass_rate,
            "meets_pass_rate": pass_rate >= min_calibrated_pass_rate,
            "brier": _metric_stats([run["metrics"]["brier"] for run in group]),
            "auc": _metric_stats([run["metrics"]["auc"] for run in group]),
            "ece": _metric_stats([run["metrics"]["ece"] for run in group]),
            "acc": _metric_stats([run["metrics"]["acc"] for run in group]),
            "brier_delta_vs_best_softmax_baseline": _metric_stats(
                [run["brier_delta_vs_best_softmax_baseline"] for run in group]
            ),
        }
        if pass_rate < min_calibrated_pass_rate:
            blockers.append(f"{family}_{method}_pass_rate_below_threshold")

    return {
        "families": families,
        "criteria": {"min_calibrated_pass_rate": min_calibrated_pass_rate},
        "decision": {
            "status": "promotion_grade" if not blockers else "not_promotion_grade",
            "blockers": blockers,
        },
    }


def run_model_family_robustness(args: argparse.Namespace) -> dict[str, Any]:
    data = np.load(args.data, allow_pickle=True)
    z = data["Z"].astype(np.float32)
    correct = data["correct"].astype(np.float32)
    sample_weights = data["sample_weights"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    feature_dim = int(data["feature_dim"])
    metadata = data["metadata"].tolist() if "metadata" in data.files else [{} for _ in range(z.shape[0])]
    source_families = [_source_family(str(row.get("source_path") or "")) for row in metadata]

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        train_idx, cal_idx, test_idx = _split_indices(
            z.shape[0],
            seed=seed,
            calibration_split=args.calibration_split,
            test_split=args.test_split,
        )
        baseline_metrics, baseline_brier, baseline_name = _load_classifier_baseline(
            Path(args.classifier_weights),
            data,
            z,
            feature_dim,
            actions,
            test_idx,
        )
        x_all = z
        scaler: StandardScaler | None = None
        if args.normalize_features:
            scaler = StandardScaler()
            x_all = z.copy()
            scaler.fit(z[train_idx, :feature_dim])
            x_all[:, :feature_dim] = scaler.transform(z[:, :feature_dim])
            x_all[:, feature_dim:] = z[:, feature_dim:]
        x_train = x_all[train_idx]
        x_cal = x_all[cal_idx]
        x_test = x_all[test_idx]
        y_train = correct[train_idx].astype(np.int64)
        y_cal = correct[cal_idx]
        y_test = correct[test_idx]

        for family in args.families:
            model = _model_for_family(family, seed)
            _fit_model(model, x_train, y_train, sample_weights[train_idx])
            p_cal = _predict_positive(model, x_cal)
            p_test = _predict_positive(model, x_test)
            raw_metrics = _metrics(p_test, y_test)
            raw_delta = baseline_brier - raw_metrics["brier"]
            runs.append(
                {
                    "seed": seed,
                    "family": family,
                    "method": "raw",
                    "train_rows": int(train_idx.shape[0]),
                    "calibration_rows": int(cal_idx.shape[0]),
                    "test_rows": int(test_idx.shape[0]),
                    "best_softmax_baseline_name": baseline_name,
                    "best_softmax_baseline": baseline_metrics,
                    "metrics": raw_metrics,
                    "brier_delta_vs_best_softmax_baseline": raw_delta,
                    "gates": _gates(raw_metrics, raw_delta),
                    "source_family_metrics": _stratum_metrics(
                        p_test,
                        y_test,
                        [source_families[idx] for idx in test_idx.tolist()],
                        min_rows=args.min_stratum_rows,
                    ),
                }
            )
            for method in args.methods:
                p_calibrated, calibrator = _calibrate(
                    method,
                    p_cal,
                    y_cal,
                    p_test,
                    calibration_bins=args.calibration_bins,
                    calibration_alpha=args.calibration_alpha,
                )
                metrics = _metrics(p_calibrated, y_test)
                delta = baseline_brier - metrics["brier"]
                runs.append(
                    {
                        "seed": seed,
                        "family": family,
                        "method": method,
                        "train_rows": int(train_idx.shape[0]),
                        "calibration_rows": int(cal_idx.shape[0]),
                        "test_rows": int(test_idx.shape[0]),
                        "best_softmax_baseline_name": baseline_name,
                        "best_softmax_baseline": baseline_metrics,
                        "calibrator": calibrator,
                        "metrics": metrics,
                        "brier_delta_vs_best_softmax_baseline": delta,
                        "gates": _gates(metrics, delta),
                        "source_family_metrics": _stratum_metrics(
                            p_calibrated,
                            y_test,
                            [source_families[idx] for idx in test_idx.tolist()],
                            min_rows=args.min_stratum_rows,
                        ),
                    }
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "data_path": args.data,
        "classifier_weights_path": args.classifier_weights,
        "seeds": args.seeds,
        "families_requested": args.families,
        "methods_requested": args.methods,
        "split": {
            "calibration_split": args.calibration_split,
            "test_split": args.test_split,
        },
        "normalize_features": args.normalize_features,
        "rows": int(z.shape[0]),
        "feature_dim": feature_dim,
        "source_family_counts": dict(sorted(Counter(source_families).items())),
        "runs": runs,
        "aggregate": aggregate_runs(
            runs,
            min_calibrated_pass_rate=args.min_calibrated_pass_rate,
        ),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Offline Reward Verifier Model-Family Robustness",
        "",
        f"- Data: `{summary['data_path']}`",
        f"- Families: `{summary['families_requested']}`",
        f"- Methods: `['raw', *{summary['methods_requested']}]`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Normalize features: `{summary['normalize_features']}`",
        f"- Decision: `{summary['aggregate']['decision']['status']}`",
        f"- Blockers: `{summary['aggregate']['decision']['blockers']}`",
        f"- Source-family counts: `{summary['source_family_counts']}`",
        "",
        "## Family Summary",
        "",
    ]
    for family, family_summary in summary["aggregate"]["families"].items():
        lines.extend([f"### `{family}`", ""])
        for method, stats in family_summary["methods"].items():
            lines.extend(
                [
                    f"- `{method}`: pass `{stats['pass_count']}/{stats['runs']}`, "
                    f"Brier mean `{stats['brier']['mean']:.4f}`, "
                    f"AUC mean `{stats['auc']['mean']:.4f}`, "
                    f"ECE mean `{stats['ece']['mean']:.4f}`, "
                    f"delta-Brier mean `{stats['brier_delta_vs_best_softmax_baseline']['mean']:.4f}`",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "This is an offline diagnostic artifact. It does not adopt live",
            "verifier weights or enable a runtime verifier gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare offline reward verifier model families across split seeds",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--classifier-weights", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument(
        "--families",
        type=lambda value: _parse_csv(value, allowed=MODEL_FAMILIES),
        default=list(MODEL_FAMILIES),
    )
    parser.add_argument(
        "--methods",
        type=lambda value: _parse_csv(value, allowed=CALIBRATION_METHODS),
        default=["temperature_bias", "ece_temperature_bias", "quantile_histogram", "isotonic"],
    )
    parser.add_argument("--seeds", type=_parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--calibration-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--calibration-bins", type=int, default=7)
    parser.add_argument("--calibration-alpha", type=float, default=0.0)
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--min-calibrated-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-stratum-rows", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        summary = run_model_family_robustness(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json_path = Path(args.summary_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = Path(args.summary_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
