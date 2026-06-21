#!/usr/bin/env python3
"""P6.2.3 / P6.2.4 — Train the verifier head and evaluate against P6.2.5 gates.

Loads verifier training NPZ (Z, correct, sample_weights), trains a VerifierHead
with BCE + inverse-frequency class weighting, then evaluates calibration on the
val split:

    - Brier score
    - ROC-AUC
    - ECE (10-bin)
    - Reliability bins (printed)

Compares against three baselines on the SAME val split:

    a. Softmax magnitude of the proposed action (current threshold mechanism's input)
    b. Constant 0.5 (uninformative reference)
    c. Constant mean(correct) — empirical base rate

P6.2.5 decision gate: Brier improvement ≥ 0.02 AND ROC-AUC ≥ 0.75 AND ECE ≤ 0.05.

Usage:
    python3 scripts/graph_router/train_verifier_head.py \
        --data    /tmp/p6_2_verifier_training_data.npz \
        --classifier-weights /path/to/routing_classifier_weights.npz \
        --classifier-data    /tmp/p6_4_training_data.npz \
        --output  /tmp/verifier_head_weights.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.routing_classifier import RoutingClassifier
from orchestration.repl_memory.verifier_head import VerifierHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_verifier")


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos_n = int(labels.sum())
    neg_n = len(labels) - pos_n
    if pos_n == 0 or neg_n == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(scores))
    rank_pos = ranks[labels == 1].sum()
    u = rank_pos - pos_n * (pos_n - 1) / 2
    return float(u / (pos_n * neg_n))


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / N) * abs(bin_conf - bin_acc)
    return float(ece)


def _metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "brier": _brier(probs, labels),
        "auc": _roc_auc(probs, labels.astype(int)),
        "ece": _ece(probs, labels),
        "acc": float(((probs >= 0.5).astype(np.float32) == labels).mean()),
    }


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype(np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid_np(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    pos = values >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-values[pos]))
    exp_x = np.exp(values[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _fit_temperature_bias_calibrator(
    probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Fit a Platt-style calibrator by deterministic NLL grid search."""
    x = _logit(probs)
    y = labels.astype(np.float32)
    best = {"temperature": 1.0, "bias": 0.0, "nll": float("inf")}
    for temperature in np.geomspace(0.25, 8.0, 65):
        scaled = x / float(temperature)
        for bias in np.linspace(-3.0, 3.0, 121):
            p = _sigmoid_np(scaled + float(bias))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            if nll < best["nll"]:
                best = {
                    "temperature": float(temperature),
                    "bias": float(bias),
                    "nll": nll,
                }
    return best


def _apply_temperature_bias_calibrator(
    probs: np.ndarray,
    calibrator: dict[str, float],
) -> np.ndarray:
    return _sigmoid_np(
        _logit(probs) / float(calibrator["temperature"]) + float(calibrator["bias"])
    )


def _fit_quantile_histogram_calibrator(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    smoothing_alpha: float,
) -> dict[str, Any]:
    """Fit empirical quantile-bin calibration on a held-out calibration split."""
    if n_bins < 2:
        raise ValueError("quantile histogram calibration requires at least two bins")
    if smoothing_alpha < 0.0:
        raise ValueError("quantile histogram smoothing alpha must be non-negative")

    p = np.asarray(probs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(p, quantiles).astype(np.float32)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.maximum.accumulate(edges)
    base_rate = float(y.mean())
    values: list[float] = []
    counts: list[int] = []
    positives: list[int] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        count = int(mask.sum())
        positive = int(y[mask].sum())
        counts.append(count)
        positives.append(positive)
        if count:
            numerator = positive + smoothing_alpha * base_rate
            denominator = count + smoothing_alpha
            values.append(float(numerator / denominator))
        else:
            values.append(base_rate)
    return {
        "method": "quantile_histogram",
        "n_bins": int(n_bins),
        "smoothing_alpha": float(smoothing_alpha),
        "base_rate": base_rate,
        "edges": edges.tolist(),
        "values": values,
        "bin_counts": counts,
        "bin_positives": positives,
    }


def _apply_quantile_histogram_calibrator(
    probs: np.ndarray,
    calibrator: dict[str, Any],
) -> np.ndarray:
    edges = np.asarray(calibrator["edges"], dtype=np.float32)
    values = np.asarray(calibrator["values"], dtype=np.float32)
    idx = np.searchsorted(edges[1:-1], probs.astype(np.float32), side="right")
    return values[idx]


def _fit_isotonic_calibrator(
    probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Fit a monotone isotonic calibrator with pool-adjacent-violators."""
    p = np.asarray(probs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    order = np.argsort(p)
    sorted_p = p[order]
    sorted_y = y[order]

    blocks: list[dict[str, float]] = []
    for score, label in zip(sorted_p, sorted_y):
        blocks.append(
            {
                "lo": float(score),
                "hi": float(score),
                "weight": 1.0,
                "value": float(label),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            value = (
                (left["value"] * left["weight"] + right["value"] * right["weight"])
                / weight
            )
            blocks.append(
                {
                    "lo": left["lo"],
                    "hi": right["hi"],
                    "weight": weight,
                    "value": value,
                }
            )
    return {
        "method": "isotonic_regression",
        "thresholds": [block["hi"] for block in blocks[:-1]],
        "values": [block["value"] for block in blocks],
        "block_count": len(blocks),
    }


def _apply_isotonic_calibrator(
    probs: np.ndarray,
    calibrator: dict[str, Any],
) -> np.ndarray:
    thresholds = np.asarray(calibrator["thresholds"], dtype=np.float32)
    values = np.asarray(calibrator["values"], dtype=np.float32)
    idx = np.searchsorted(thresholds, probs.astype(np.float32), side="right")
    return values[idx]


def _fit_feature_normalizer(
    Z: np.ndarray,
    train_idx: np.ndarray,
    feature_dim: int,
) -> dict[str, Any]:
    features = Z[train_idx, :feature_dim].astype(np.float32)
    mean = features.mean(axis=0).astype(np.float32)
    scale = features.std(axis=0).astype(np.float32)
    zero_scale_mask = scale < 1e-6
    scale = np.where(zero_scale_mask, 1.0, scale).astype(np.float32)
    return {
        "mean": mean,
        "scale": scale,
        "zero_scale_count": int(zero_scale_mask.sum()),
    }


def _apply_feature_normalizer(
    Z: np.ndarray,
    feature_dim: int,
    normalizer: dict[str, Any],
) -> np.ndarray:
    Z_norm = Z.copy()
    Z_norm[:, :feature_dim] = (
        Z_norm[:, :feature_dim] - normalizer["mean"]
    ) / normalizer["scale"]
    return Z_norm


def _reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    print(f"\n{'bin':<14} {'n':>8} {'mean_p':>10} {'frac_pos':>10} {'|gap|':>10}")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        n = int(mask.sum())
        if n == 0:
            print(f"[{lo:.2f},{hi:.2f}]   {n:>8} {'-':>10} {'-':>10} {'-':>10}")
            continue
        mean_p = probs[mask].mean()
        frac_pos = labels[mask].mean()
        gap = abs(mean_p - frac_pos)
        print(f"[{lo:.2f},{hi:.2f}]   {n:>8} {mean_p:>10.3f} {frac_pos:>10.3f} {gap:>10.3f}")


def _load_classifier_features(
    classifier_data_path: Path | None,
    verifier_npz: Any,
    Z: np.ndarray,
    feature_dim: int,
    expected_rows: int,
) -> tuple[np.ndarray, str]:
    if classifier_data_path is not None:
        raw = np.load(classifier_data_path, allow_pickle=True)
        if "X" in raw.files:
            X_raw = raw["X"].astype(np.float32)
            if X_raw.shape[0] != expected_rows:
                raise SystemExit(
                    f"classifier-data row count {X_raw.shape[0]} does not match verifier rows {expected_rows}"
                )
            return X_raw, f"{classifier_data_path}:X"
        logger.info(
            "Classifier data %s has no X matrix; using verifier feature prefix",
            classifier_data_path,
        )

    if "X" in verifier_npz.files:
        X_raw = verifier_npz["X"].astype(np.float32)
        if X_raw.shape[0] != expected_rows:
            raise SystemExit(
                f"verifier X row count {X_raw.shape[0]} does not match verifier rows {expected_rows}"
            )
        return X_raw, f"{classifier_data_path or 'verifier_npz'}:X"

    source = f"{classifier_data_path}:Z_feature_prefix" if classifier_data_path else "verifier_npz:Z_feature_prefix"
    return Z[:, :feature_dim].astype(np.float32), source


def train_and_eval(
    data_path: Path,
    classifier_weights_path: Path,
    classifier_data_path: Path | None,
    output_path: Path,
    epochs: int = 100,
    lr: float = 0.05,
    batch_size: int = 256,
    patience: int = 20,
    hidden1: int = 64,
    hidden2: int = 32,
    val_seed: int = 42,
    val_split: float = 0.2,
    calibration_split: float = 0.0,
    test_split: float = 0.0,
    calibration_method: str = "temperature_bias",
    calibration_bins: int = 7,
    calibration_alpha: float = 0.0,
    normalize_features: bool = False,
) -> dict:
    # ── Load data ──
    logger.info("Loading verifier data from %s", data_path)
    d = np.load(data_path, allow_pickle=True)
    Z = d["Z"].astype(np.float32)
    correct = d["correct"].astype(np.float32)
    sample_weights = d["sample_weights"].astype(np.float32)
    actions = d["actions"].astype(np.int64)
    feature_dim = int(d["feature_dim"])
    n_actions = int(d["n_actions"])
    N = Z.shape[0]
    logger.info("Loaded %d samples, Z.shape=%s, feature_dim=%d, n_actions=%d",
                N, Z.shape, feature_dim, n_actions)
    logger.info("Correctness base rate: %.4f", correct.mean())

    rng = np.random.default_rng(val_seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(N * test_split) if test_split > 0.0 else 0
    n_cal = int(N * calibration_split) if calibration_split > 0.0 else 0
    if n_test or n_cal:
        if n_test < 2 or n_cal < 2:
            raise SystemExit(
                "calibration/test mode requires at least two rows in each requested split"
            )
        if n_test + n_cal >= N:
            raise SystemExit("calibration/test splits leave no rows for training")
        test_idx = idx[:n_test]
        cal_idx = idx[n_test:n_test + n_cal]
        train_idx = idx[n_test + n_cal:]
        eval_idx = test_idx
        eval_split_name = "test"
    else:
        train_idx = idx
        cal_idx = np.array([], dtype=np.int64)
        n_val = max(1, int(N * val_split))
        eval_idx = idx[:n_val]
        eval_split_name = "val"

    feature_normalizer: dict[str, Any] | None = None
    if normalize_features:
        feature_normalizer = _fit_feature_normalizer(Z, train_idx, feature_dim)
        Z = _apply_feature_normalizer(Z, feature_dim, feature_normalizer)

    Z_fit = Z[train_idx]
    y_fit = correct[train_idx]
    w_fit = sample_weights[train_idx]

    # ── Train ──
    verifier = VerifierHead(
        feature_dim=feature_dim,
        n_actions=n_actions,
        hidden1=hidden1,
        hidden2=hidden2,
    )
    if feature_normalizer is not None:
        verifier.set_feature_normalizer(
            feature_normalizer["mean"],
            feature_normalizer["scale"],
        )
    logger.info("Verifier params: %d (input_dim=%d)", verifier.param_count, verifier.input_dim)

    history = verifier.train(
        Z_fit, y_fit, w_fit,
        epochs=epochs,
        lr=lr,
        val_split=val_split,
        patience=patience,
        batch_size=batch_size,
        rng_seed=val_seed,
    )

    # ── Evaluate on the selected holdout split ──
    Z_val = Z[eval_idx]
    y_val = correct[eval_idx]
    actions_val = actions[eval_idx]

    # Verifier predictions on val
    p_verifier, _ = verifier.forward(Z_val)
    pred_verifier = (p_verifier >= 0.5).astype(np.float32)
    acc_verifier = float((pred_verifier == y_val).mean())
    brier_verifier = _brier(p_verifier, y_val)
    auc_verifier = _roc_auc(p_verifier, y_val.astype(int))
    ece_verifier = _ece(p_verifier, y_val)
    calibration: dict[str, Any] | None = None
    if n_cal:
        p_cal, _ = verifier.forward(Z[cal_idx])
        y_cal = correct[cal_idx]
        if calibration_method == "temperature_bias":
            calibrator = _fit_temperature_bias_calibrator(p_cal, y_cal)
            p_calibrated = _apply_temperature_bias_calibrator(p_verifier, calibrator)
            calibration = {
                "method": "temperature_bias_grid",
                "calibration_rows": int(n_cal),
                "test_rows": int(len(y_val)),
                "temperature": calibrator["temperature"],
                "bias": calibrator["bias"],
                "calibration_nll": calibrator["nll"],
                "calibrated_verifier": _metrics(p_calibrated, y_val),
            }
        elif calibration_method == "quantile_histogram":
            calibrator = _fit_quantile_histogram_calibrator(
                p_cal,
                y_cal,
                n_bins=calibration_bins,
                smoothing_alpha=calibration_alpha,
            )
            p_calibrated = _apply_quantile_histogram_calibrator(p_verifier, calibrator)
            calibration = {
                **calibrator,
                "calibration_rows": int(n_cal),
                "test_rows": int(len(y_val)),
                "calibrated_verifier": _metrics(p_calibrated, y_val),
            }
        elif calibration_method == "isotonic":
            calibrator = _fit_isotonic_calibrator(p_cal, y_cal)
            p_calibrated = _apply_isotonic_calibrator(p_verifier, calibrator)
            calibration = {
                **calibrator,
                "calibration_rows": int(n_cal),
                "test_rows": int(len(y_val)),
                "calibrated_verifier": _metrics(p_calibrated, y_val),
            }
        else:
            raise SystemExit(f"Unsupported calibration method: {calibration_method}")

    # ── Baseline: softmax magnitude of the proposed action ──
    # Load classifier + cached features, recover softmax(taken_action | features)
    logger.info("Loading classifier from %s for baseline comparison", classifier_weights_path)
    clf = RoutingClassifier.load(classifier_weights_path)
    if clf is None:
        raise SystemExit(f"Failed to load classifier weights from {classifier_weights_path}")
    if len(actions_val) and int(actions_val.max()) >= clf.n_actions:
        raise SystemExit(
            f"classifier has {clf.n_actions} actions but validation labels include action "
            f"{int(actions_val.max())}"
        )
    X_raw, classifier_feature_source = _load_classifier_features(
        classifier_data_path, d, Z, feature_dim, N
    )
    # ── Critical: align the eval split.  The verifier extractor preserves row
    # order, so eval_idx indexes the same rows in X_raw.
    X_val = X_raw[eval_idx]
    probs_clf, _ = clf.forward(X_val)
    softmax_taken = probs_clf[np.arange(len(actions_val)), actions_val]
    max_softmax = probs_clf.max(axis=1)

    brier_sm_taken = _brier(softmax_taken, y_val)
    auc_sm_taken = _roc_auc(softmax_taken, y_val.astype(int))
    ece_sm_taken = _ece(softmax_taken, y_val)

    brier_sm_max = _brier(max_softmax, y_val)
    auc_sm_max = _roc_auc(max_softmax, y_val.astype(int))
    ece_sm_max = _ece(max_softmax, y_val)

    # Base rate baseline (constant prediction)
    base_rate = float(y_val.mean())
    brier_base = _brier(np.full_like(y_val, base_rate), y_val)

    # ── Report ──
    print("\n" + "=" * 90)
    print(f"{'Predictor':<32} {'Brier ↓':>10} {'ROC-AUC ↑':>10} {'ECE ↓':>10}  {'Acc@0.5':>8}")
    print("=" * 90)
    print(f"{'verifier (P6.2)':<32} {brier_verifier:>10.4f} {auc_verifier:>10.4f} {ece_verifier:>10.4f}  {acc_verifier:>8.4f}")
    print(f"{'softmax_taken (clf p(a|x))':<32} {brier_sm_taken:>10.4f} {auc_sm_taken:>10.4f} {ece_sm_taken:>10.4f}  {'-':>8}")
    print(f"{'softmax_max (clf top-1 prob)':<32} {brier_sm_max:>10.4f} {auc_sm_max:>10.4f} {ece_sm_max:>10.4f}  {'-':>8}")
    print(f"{'constant base rate':<32} {brier_base:>10.4f} {'-':>10} {'-':>10}  {'-':>8}")
    if calibration is not None:
        calibrated = calibration["calibrated_verifier"]
        print(f"{'verifier + calibration':<32} {calibrated['brier']:>10.4f} {calibrated['auc']:>10.4f} {calibrated['ece']:>10.4f}  {calibrated['acc']:>8.4f}")
    print("=" * 90)

    _reliability(p_verifier, y_val, n_bins=10)

    # ── Decision gate ──
    # Pick the stronger of the two baselines for the Brier comparison
    base_brier = min(brier_sm_taken, brier_sm_max)
    base_brier_name = "softmax_taken" if brier_sm_taken < brier_sm_max else "softmax_max"
    brier_delta = base_brier - brier_verifier   # positive = improvement
    brier_delta_constant = brier_base - brier_verifier
    if calibration is not None:
        calibrated = calibration["calibrated_verifier"]
        calibrated_brier_delta = base_brier - calibrated["brier"]
        calibrated_brier_delta_constant = brier_base - calibrated["brier"]
        calibration["brier_delta_vs_best_softmax_baseline"] = calibrated_brier_delta
        calibration["brier_delta_vs_constant_baseline"] = calibrated_brier_delta_constant
        calibration["gates"] = {
            "brier_delta_ge_0_02": bool(calibrated_brier_delta >= 0.02),
            "auc_ge_0_75": bool(calibrated["auc"] >= 0.75),
            "ece_le_0_05": bool(calibrated["ece"] <= 0.05),
            "pass": bool(
                (calibrated_brier_delta >= 0.02)
                and (calibrated["auc"] >= 0.75)
                and (calibrated["ece"] <= 0.05)
            ),
        }
    print(
        f"\nGates (P6.2.5):\n"
        f"  ΔBrier vs best baseline ({base_brier_name}): {brier_delta:+.4f}  (gate: ≥ +0.02)  "
        f"{'PASS' if brier_delta >= 0.02 else 'FAIL'}\n"
        f"  ROC-AUC:                                       {auc_verifier:.4f}        (gate: ≥ 0.75)   "
        f"{'PASS' if auc_verifier >= 0.75 else 'FAIL'}\n"
        f"  ECE:                                           {ece_verifier:.4f}        (gate: ≤ 0.05)   "
        f"{'PASS' if ece_verifier <= 0.05 else 'FAIL'}"
    )
    all_pass = (brier_delta >= 0.02) and (auc_verifier >= 0.75) and (ece_verifier <= 0.05)
    if calibration is not None:
        calibrated = calibration["calibrated_verifier"]
        calibrated_gates = calibration["gates"]
        print(
            f"\nCalibrated gates ({calibration['method']}):\n"
            f"  ΔBrier vs best baseline ({base_brier_name}): "
            f"{calibration['brier_delta_vs_best_softmax_baseline']:+.4f}  (gate: ≥ +0.02)  "
            f"{'PASS' if calibrated_gates['brier_delta_ge_0_02'] else 'FAIL'}\n"
            f"  ROC-AUC:                                       {calibrated['auc']:.4f}        "
            f"(gate: ≥ 0.75)   {'PASS' if calibrated_gates['auc_ge_0_75'] else 'FAIL'}\n"
            f"  ECE:                                           {calibrated['ece']:.4f}        "
            f"(gate: ≤ 0.05)   {'PASS' if calibrated_gates['ece_le_0_05'] else 'FAIL'}\n"
            f"  Overall:                                       "
            f"{'PASS' if calibrated_gates['pass'] else 'FAIL'}"
        )
    print(
        f"\nRaw P6.2.5 overall: "
        f"{'PASS — wire ORCHESTRATOR_VERIFIER_GATE (default OFF)' if all_pass else 'FAIL'}"
    )

    verifier.save(output_path)

    return {
        "schema_version": "verifier_head_eval.v1",
        "data_path": str(data_path),
        "classifier_weights_path": str(classifier_weights_path),
        "classifier_data_path": str(classifier_data_path) if classifier_data_path else None,
        "classifier_feature_source": classifier_feature_source,
        "output_path": str(output_path),
        "rows": int(N),
        "eval_split": eval_split_name,
        "eval_rows": int(len(y_val)),
        "val_rows": int(len(y_val)),
        "train_rows": int(len(train_idx)),
        "calibration_rows": int(n_cal),
        "test_rows": int(n_test),
        "feature_dim": int(feature_dim),
        "n_actions": int(n_actions),
        "hidden1": int(hidden1),
        "hidden2": int(hidden2),
        "normalize_features": bool(normalize_features),
        "feature_normalizer": (
            {
                "scope": "feature_prefix",
                "zero_scale_count": feature_normalizer["zero_scale_count"],
            }
            if feature_normalizer is not None
            else None
        ),
        "positive_rows": int(correct.sum()),
        "negative_rows": int(N - int(correct.sum())),
        "val_positive_rate": float(y_val.mean()),
        "action_counts": {
            str(action): int((actions == action).sum())
            for action in sorted(set(int(a) for a in actions.tolist()))
        },
        "action_positive_counts": {
            str(action): int(correct[actions == action].sum())
            for action in sorted(set(int(a) for a in actions.tolist()))
        },
        "best_softmax_baseline_name": base_brier_name,
        "verifier": {"brier": brier_verifier, "auc": auc_verifier, "ece": ece_verifier, "acc": acc_verifier},
        "softmax_taken": {"brier": brier_sm_taken, "auc": auc_sm_taken, "ece": ece_sm_taken},
        "softmax_max": {"brier": brier_sm_max, "auc": auc_sm_max, "ece": ece_sm_max},
        "constant_base_rate": {"brier": brier_base},
        "brier_delta_vs_best_softmax_baseline": brier_delta,
        "brier_delta_vs_constant_baseline": brier_delta_constant,
        "gates": {
            "brier_delta_ge_0_02": bool(brier_delta >= 0.02),
            "auc_ge_0_75": bool(auc_verifier >= 0.75),
            "ece_le_0_05": bool(ece_verifier <= 0.05),
            "pass": bool(all_pass),
        },
        "calibration": calibration,
        "history_final": {k: v[-1] for k, v in history.items()},
    }


def _summary_markdown(summary: dict) -> str:
    verifier = summary["verifier"]
    lines = [
        "# Offline Multi-Action Verifier Evaluation",
        "",
        f"- Data: `{summary['data_path']}`",
        f"- Output weights: `{summary['output_path']}`",
        f"- Classifier feature source: `{summary['classifier_feature_source']}`",
        f"- Rows: `{summary['rows']}` ({summary['positive_rows']} positive / {summary['negative_rows']} negative)",
        f"- Evaluation split: `{summary.get('eval_split', 'val')}`",
        f"- Evaluation rows: `{summary.get('eval_rows', summary['val_rows'])}`",
        f"- Hidden widths: `{summary['hidden1']}` / `{summary['hidden2']}`",
        f"- Feature normalization: `{summary['normalize_features']}`",
        f"- Actions represented: `{summary['action_counts']}`",
        f"- Brier: `{verifier['brier']:.4f}`",
        f"- ROC-AUC: `{verifier['auc']:.4f}`",
        f"- ECE: `{verifier['ece']:.4f}`",
        f"- Accuracy@0.5: `{verifier['acc']:.4f}`",
        f"- Delta Brier vs best softmax baseline: "
        f"`{summary['brier_delta_vs_best_softmax_baseline']:+.4f}`",
        f"- Delta Brier vs constant base-rate baseline: "
        f"`{summary['brier_delta_vs_constant_baseline']:+.4f}`",
        f"- Gates passed: `{summary['gates']['pass']}`",
    ]
    if summary.get("calibration"):
        calibrated = summary["calibration"]["calibrated_verifier"]
        lines.extend(
            [
                f"- Calibration method: `{summary['calibration']['method']}`",
                f"- Calibrated Brier: `{calibrated['brier']:.4f}`",
                f"- Calibrated ROC-AUC: `{calibrated['auc']:.4f}`",
                f"- Calibrated ECE: `{calibrated['ece']:.4f}`",
                f"- Calibrated Accuracy@0.5: `{calibrated['acc']:.4f}`",
                f"- Calibrated delta Brier vs best softmax baseline: "
                f"`{summary['calibration']['brier_delta_vs_best_softmax_baseline']:+.4f}`",
                f"- Calibrated delta Brier vs constant base-rate baseline: "
                f"`{summary['calibration']['brier_delta_vs_constant_baseline']:+.4f}`",
                f"- Calibrated gates passed: `{summary['calibration']['gates']['pass']}`",
            ]
        )
    lines.extend(
        [
            "",
            "This is an offline evaluation artifact. It is not a live verifier",
            "weight promotion and does not enable the verifier gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="P6.2.3/4: train + evaluate verifier head")
    parser.add_argument("--data", type=str, required=True,
                        help="Verifier training NPZ (from extract_verifier_training_data.py)")
    parser.add_argument("--classifier-weights", type=str, required=True,
                        help="Classifier weights NPZ for softmax-baseline comparison")
    parser.add_argument("--classifier-data", type=str,
                        help=(
                            "Classifier training NPZ. If omitted or if it has no X matrix, "
                            "use the verifier NPZ feature prefix for the classifier baseline."
                        ))
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for trained verifier weights")
    parser.add_argument("--summary-json", type=str)
    parser.add_argument("--summary-md", type=str)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--calibration-split", type=float, default=0.0)
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument(
        "--calibration-method",
        choices=("temperature_bias", "quantile_histogram", "isotonic"),
        default="temperature_bias",
        help="Held-out calibration method to fit when --calibration-split is set.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=7,
        help="Quantile bins for --calibration-method quantile_histogram.",
    )
    parser.add_argument(
        "--calibration-alpha",
        type=float,
        default=0.0,
        help="Base-rate smoothing alpha for quantile histogram calibration.",
    )
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Standardize the verifier feature prefix using the train split.",
    )
    args = parser.parse_args()

    summary = train_and_eval(
        data_path=Path(args.data),
        classifier_weights_path=Path(args.classifier_weights),
        classifier_data_path=Path(args.classifier_data) if args.classifier_data else None,
        output_path=Path(args.output),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        calibration_split=args.calibration_split,
        test_split=args.test_split,
        calibration_method=args.calibration_method,
        calibration_bins=args.calibration_bins,
        calibration_alpha=args.calibration_alpha,
        normalize_features=args.normalize_features,
    )
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_md:
        path = Path(args.summary_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_summary_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
