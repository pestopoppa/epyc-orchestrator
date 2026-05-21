#!/usr/bin/env python3
"""P6.4 — MC-dropout uncertainty vs softmax magnitude as correctness predictor.

Loads a trained RoutingClassifier checkpoint and a training NPZ, recreates the
80/20 train/val split with the same RNG seed used by the training pipeline,
then evaluates two candidate correctness predictors on the val set:

    1. softmax_magnitude — max softmax probability from deterministic forward
       (baseline: this is what the per-class threshold uses today)
    2. mc_consistency    — derived from N=10 MC-dropout forward passes:
       a. var_top  = variance of top-class probability across samples (low = stable)
       b. ent_mean = entropy of mean predictive distribution (low = stable)

Correctness label: did the deterministic argmax match the val label?

Reports per-predictor Brier, ROC-AUC, ECE (10-bin).

Usage:
    python3 scripts/graph_router/mc_dropout_eval.py \
        --weights /path/to/routing_classifier_weights.npz \
        --data    /path/to/training_data.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.routing_classifier import RoutingClassifier

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("mc_dropout_eval")


def _recreate_val_split(N: int, val_split: float = 0.2, seed: int = 42) -> np.ndarray:
    """Reproduce the val split from RoutingClassifier.train()."""
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    n_val = max(1, int(N * val_split))
    return indices[:n_val]


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score for a binary correctness predictor (probs ∈ [0,1])."""
    return float(np.mean((probs - labels) ** 2))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via Mann–Whitney U (rank-based, no sklearn dep)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(scores))
    rank_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = rank_pos - n_pos * (n_pos - 1) / 2
    return float(u / (n_pos * n_neg))


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (10-bin equal-width)."""
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


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy with safe log."""
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def _normalize_unit(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def evaluate(
    weights_path: Path,
    data_path: Path,
    n_samples: int = 10,
    p_drop: float = 0.1,
    seed: int = 0,
    val_split: float = 0.2,
    val_seed: int = 42,
) -> dict:
    clf = RoutingClassifier.load(weights_path)
    if clf is None:
        raise SystemExit(f"Failed to load classifier weights from {weights_path}")
    logger.info(
        "Loaded classifier: input_dim=%d hidden=%d/%d n_actions=%d params=%d",
        clf.input_dim, clf.hidden1, clf.hidden2, clf.n_actions, clf.param_count,
    )

    data = np.load(data_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    logger.info("Loaded training data: X=%s y=%s", X.shape, y.shape)

    val_idx = _recreate_val_split(len(X), val_split=val_split, seed=val_seed)
    X_val, y_val = X[val_idx], y[val_idx]
    logger.info("Val split: %d samples (seed=%d, val_split=%.2f)", len(val_idx), val_seed, val_split)

    # 1. Deterministic forward (current production behavior)
    probs_det, _ = clf.forward(X_val)
    pred_det = probs_det.argmax(axis=1)
    max_prob_det = probs_det.max(axis=1)
    val_acc = float((pred_det == y_val).mean())
    logger.info("Deterministic val accuracy: %.4f", val_acc)

    # Correctness labels (binary): did deterministic argmax match the val label?
    correct = (pred_det == y_val).astype(np.float32)

    # 2. MC-dropout forward
    logger.info("Running MC-dropout: n_samples=%d p_drop=%.2f seed=%d", n_samples, p_drop, seed)
    mc_probs = clf.mc_predict(X_val, p_drop=p_drop, n_samples=n_samples, seed=seed)  # (S, N, A)
    mc_mean = mc_probs.mean(axis=0)
    mc_var = mc_probs.var(axis=0)
    pred_mc = mc_mean.argmax(axis=1)
    mc_val_acc = float((pred_mc == y_val).mean())
    logger.info("MC-mean val accuracy:      %.4f (delta vs det: %+.4f)", mc_val_acc, mc_val_acc - val_acc)
    flip_rate = float((pred_mc != pred_det).mean())
    logger.info("MC-vs-det argmax flip rate: %.4f", flip_rate)

    # Per-example uncertainty derivations
    top_idx = pred_det
    top_prob_var = mc_var[np.arange(len(top_idx)), top_idx]  # variance of top class prob across samples
    mean_entropy = _entropy(mc_mean, axis=-1)  # entropy of mean predictive dist

    # 3. Candidate correctness predictors (higher = more likely correct)
    predictors = {
        "softmax_magnitude (baseline)": max_prob_det,                      # raw max prob ∈ [0,1]
        "mc_stability_var":             -top_prob_var,                    # high var → low stability
        "mc_stability_entropy":         -mean_entropy,                    # high entropy → low stability
        "mc_max_prob_mean":             mc_mean.max(axis=1),              # max of MC-mean dist
    }

    # 4. Metrics
    results = {}
    print("\n" + "=" * 90)
    print(f"{'Predictor':<32} {'Brier':>10} {'ROC-AUC':>10} {'ECE':>10}  Description")
    print("=" * 90)
    for name, raw_score in predictors.items():
        score_unit = _normalize_unit(raw_score)  # map to [0,1] for Brier/ECE
        brier = _brier(score_unit, correct)
        auc = _roc_auc(raw_score, correct.astype(int))
        ece = _ece(score_unit, correct, n_bins=10)
        results[name] = {"brier": brier, "auc": auc, "ece": ece}
        print(f"{name:<32} {brier:>10.4f} {auc:>10.4f} {ece:>10.4f}")
    print("=" * 90)

    # Decision-gate comparison
    base = results["softmax_magnitude (baseline)"]
    best_mc_name = max(
        [k for k in results if k.startswith("mc_")],
        key=lambda k: results[k]["auc"],
    )
    best_mc = results[best_mc_name]
    auc_delta = best_mc["auc"] - base["auc"]
    brier_delta = base["brier"] - best_mc["brier"]  # lower brier is better → positive delta is improvement

    print(
        f"\nBest MC predictor: {best_mc_name}\n"
        f"  ΔROC-AUC vs softmax_magnitude: {auc_delta:+.4f}  (P6.4.4 gate: ≥ +0.05)\n"
        f"  ΔBrier   vs softmax_magnitude: {brier_delta:+.4f}\n"
        f"  ΔECE     vs softmax_magnitude: {base['ece'] - best_mc['ece']:+.4f}\n"
    )
    gate_pass = auc_delta >= 0.05
    print(f"P6.4 decision gate: {'PASS — wire MC-dropout uncertainty as fallback gate' if gate_pass else 'FAIL — record null result; rely on P6.2 verifier head'}")

    return {
        "val_size": int(len(val_idx)),
        "det_val_acc": val_acc,
        "mc_val_acc": mc_val_acc,
        "argmax_flip_rate": flip_rate,
        "predictors": results,
        "best_mc_predictor": best_mc_name,
        "auc_delta": auc_delta,
        "brier_delta": brier_delta,
        "gate_pass": gate_pass,
    }


def main():
    parser = argparse.ArgumentParser(description="P6.4: MC-dropout vs softmax-magnitude correctness predictor")
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to routing_classifier_weights.npz",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to training_data.npz",
    )
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--p-drop", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluate(
        weights_path=Path(args.weights),
        data_path=Path(args.data),
        n_samples=args.n_samples,
        p_drop=args.p_drop,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
