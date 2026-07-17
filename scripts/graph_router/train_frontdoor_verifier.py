#!/usr/bin/env python3
"""P6.2 / A2 — Frontdoor-specialist binary success predictor.

Avoids the label-leakage trap from the multi-action verifier by training a
single-action specialist: input = features only (1031-d), output = P(frontdoor
success). No action one-hot, no per-action marginal to memorize — any signal
the model achieves is genuinely embedding-conditional.

Training data: frontdoor (action[0]) subset of the debiased verifier NPZ,
correctness = outcome == 'success' from episodic.db.backup-20260415.
Train/val split: same seed=42 as the multi-action verifier for direct
comparability.

Baselines compared on the same val split:
    - constant base rate (uninformative)
    - softmax max prob from the existing RoutingClassifier
    - softmax p(frontdoor | x) from the existing RoutingClassifier

Decision gate (P6.2 A2): Brier improvement >= 0.02 AND ROC-AUC >= 0.75 AND
ECE <= 0.05 vs the stronger softmax baseline. The summary also records the
constant base-rate Brier comparison so null results are not over-read.

Usage:
    python3 scripts/graph_router/train_frontdoor_verifier.py \
        --data /tmp/p6_2_verifier_training_data_debiased.npz \
        --classifier-weights /mnt/raid0/llm/epyc-orchestrator/orchestration/autopilot_checkpoints/20260416_134815/routing_classifier_weights.npz \
        --classifier-data /tmp/p6_4_training_data.npz \
        --output /tmp/frontdoor_verifier_weights.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_primitives.stat_tests import (  # noqa: E402
    expected_calibration_error as _stat_ece,
    roc_auc as _stat_roc_auc,
)

from orchestration.repl_memory.routing_classifier import RoutingClassifier
from orchestration.repl_memory.verifier_head import VerifierHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_frontdoor_verifier")


def _brier(p, y): return float(np.mean((p - y) ** 2))


def _roc_auc(scores, labels):
    """Consolidated onto ``stat_tests.roc_auc`` (tie-averaged, == sklearn);
    ``float('nan')`` on a single-class input preserved."""
    auc = _stat_roc_auc(scores, labels)
    return float("nan") if auc is None else float(auc)


def _ece(p, y, nb=10):
    """Consolidated onto ``stat_tests.expected_calibration_error`` (identical
    binning); ``0.0`` on empty input preserved."""
    e = _stat_ece(p, y, n_bins=nb)
    return 0.0 if e is None else float(e)


def _reliability(p, y, nb=10):
    bins = np.linspace(0.0, 1.0, nb + 1)
    print(f"\n{'bin':<14} {'n':>8} {'mean_p':>10} {'frac_pos':>10} {'|gap|':>10}")
    for i in range(nb):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi if i < nb - 1 else p <= hi)
        n = int(m.sum())
        if n == 0:
            print(f"[{lo:.2f},{hi:.2f}]   {n:>8} {'-':>10} {'-':>10} {'-':>10}")
            continue
        print(f"[{lo:.2f},{hi:.2f}]   {n:>8} {p[m].mean():>10.3f} {y[m].mean():>10.3f} {abs(p[m].mean() - y[m].mean()):>10.3f}")


def train_and_eval(
    data_path: Path,
    classifier_weights_path: Path,
    classifier_data_path: Path,
    output_path: Path,
    epochs: int = 100,
    lr: float = 0.05,
    batch_size: int = 256,
    patience: int = 15,
    val_seed: int = 42,
    val_split: float = 0.2,
    frontdoor_action_idx: int = 0,
) -> dict:
    # ── Load debiased data ──
    logger.info("Loading debiased verifier data from %s", data_path)
    d = np.load(data_path, allow_pickle=True)
    Z = d["Z"].astype(np.float32)              # (N, 1031 + 8)
    correct = d["correct"].astype(np.float32)
    actions = d["actions"].astype(np.int64)
    feature_dim = int(d["feature_dim"])        # 1031
    N_full = Z.shape[0]
    logger.info("Loaded %d samples; filtering to action=%d (frontdoor)",
                N_full, frontdoor_action_idx)

    # ── Filter to frontdoor + strip the action one-hot ──
    fd_mask = actions == frontdoor_action_idx
    X_fd = Z[fd_mask, :feature_dim]            # (N_fd, 1031) — drop one-hot
    y_fd = correct[fd_mask]
    N = X_fd.shape[0]
    n_pos = int(y_fd.sum())
    n_neg = N - n_pos
    logger.info(
        "Frontdoor subset: %d samples (%.1f%% of total). Pos=%d (%.1f%%) Neg=%d (%.1f%%)",
        N, 100 * N / N_full, n_pos, 100 * n_pos / N, n_neg, 100 * n_neg / N,
    )

    # Inverse-frequency sample weights — class imbalance ~22% failure in frontdoor
    pos_weight = N / (2.0 * n_pos) if n_pos else 0.0
    neg_weight = N / (2.0 * n_neg) if n_neg else 0.0
    sample_weights = np.where(y_fd == 1.0, pos_weight, neg_weight).astype(np.float32)

    # ── Train: VerifierHead with n_actions=0 → input_dim=feature_dim, no one-hot ──
    verifier = VerifierHead(
        feature_dim=feature_dim,
        n_actions=0,
        hidden1=64,
        hidden2=32,
    )
    logger.info(
        "Frontdoor verifier: input_dim=%d (no action one-hot), params=%d",
        verifier.input_dim, verifier.param_count,
    )

    verifier.train(
        X_fd, y_fd, sample_weights,
        epochs=epochs, lr=lr, val_split=val_split,
        patience=patience, batch_size=batch_size, rng_seed=val_seed,
    )

    # ── Evaluate on val split (recreated with same seed) ──
    rng = np.random.default_rng(val_seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(1, int(N * val_split))
    val_idx = idx[:n_val]
    X_val = X_fd[val_idx]
    y_val = y_fd[val_idx]

    p_fd, _ = verifier.forward(X_val)
    brier_fd = _brier(p_fd, y_val)
    auc_fd = _roc_auc(p_fd, y_val.astype(int))
    ece_fd = _ece(p_fd, y_val)
    acc_fd = float(((p_fd >= 0.5).astype(np.float32) == y_val).mean())

    # ── Baseline: softmax magnitude from classifier on the SAME frontdoor val rows ──
    logger.info("Loading classifier from %s for baseline comparison", classifier_weights_path)
    clf = RoutingClassifier.load(classifier_weights_path)
    if clf is None:
        raise SystemExit(f"Failed to load classifier weights from {classifier_weights_path}")

    # ── Align: X_fd already has the classifier-input features (1031-d) for frontdoor
    # rows in the debiased NPZ ordering. val_idx indexes within X_fd. So X_fd[val_idx]
    # is the right val feature matrix for the classifier baseline — no need to
    # re-join through the (mis-sized) raw classifier NPZ.
    X_clf_fd_val = X_fd[val_idx]
    probs_clf, _ = clf.forward(X_clf_fd_val)
    max_softmax = probs_clf.max(axis=1)
    fd_softmax = probs_clf[:, frontdoor_action_idx]

    base_rate = float(y_val.mean())
    constant_pred = np.full_like(y_val, base_rate)

    brier_sm_max = _brier(max_softmax, y_val)
    auc_sm_max = _roc_auc(max_softmax, y_val.astype(int))
    ece_sm_max = _ece(max_softmax, y_val)

    brier_sm_fd = _brier(fd_softmax, y_val)
    auc_sm_fd = _roc_auc(fd_softmax, y_val.astype(int))
    ece_sm_fd = _ece(fd_softmax, y_val)

    brier_constant = _brier(constant_pred, y_val)

    print("\n" + "=" * 90)
    print("Frontdoor-specialist verifier — intra-action val metrics (action[0] only)")
    print(f"  Val subset: n={len(y_val)} ({y_val.mean()*100:.1f}% positive)")
    print("=" * 90)
    print(f"{'Predictor':<32} {'Brier ↓':>10} {'ROC-AUC ↑':>10} {'ECE ↓':>10}  {'Acc@0.5':>8}")
    print("-" * 90)
    print(f"{'frontdoor verifier (A2)':<32} {brier_fd:>10.4f} {auc_fd:>10.4f} {ece_fd:>10.4f}  {acc_fd:>8.4f}")
    print(f"{'softmax max prob':<32} {brier_sm_max:>10.4f} {auc_sm_max:>10.4f} {ece_sm_max:>10.4f}  {'-':>8}")
    print(f"{'softmax p(frontdoor|x)':<32} {brier_sm_fd:>10.4f} {auc_sm_fd:>10.4f} {ece_sm_fd:>10.4f}  {'-':>8}")
    print(f"{'constant base rate':<32} {brier_constant:>10.4f} {'-':>10} {'-':>10}  {'-':>8}")
    print("=" * 90)

    _reliability(p_fd, y_val, nb=10)

    base_brier = min(brier_sm_max, brier_sm_fd)
    base_name = "softmax_max" if brier_sm_max < brier_sm_fd else "softmax_p(fd|x)"
    brier_delta = base_brier - brier_fd
    brier_delta_constant = brier_constant - brier_fd
    print(
        f"\nA2 decision gates:\n"
        f"  ΔBrier vs best baseline ({base_name}): {brier_delta:+.4f}  (gate: ≥ +0.02)  "
        f"{'PASS' if brier_delta >= 0.02 else 'FAIL'}\n"
        f"  ROC-AUC:                                       {auc_fd:.4f}        (gate: ≥ 0.75)   "
        f"{'PASS' if auc_fd >= 0.75 else 'FAIL'}\n"
        f"  ECE:                                           {ece_fd:.4f}        (gate: ≤ 0.05)   "
        f"{'PASS' if ece_fd <= 0.05 else 'FAIL'}"
    )
    all_pass = (brier_delta >= 0.02) and (auc_fd >= 0.75) and (ece_fd <= 0.05)
    print(f"\nA2 overall: {'PASS — frontdoor specialist outperforms baselines' if all_pass else 'FAIL — frontdoor specialist did not clear gates'}")

    verifier.save(output_path)
    return {
        "schema_version": "frontdoor_verifier_eval.v1",
        "data_path": str(data_path),
        "classifier_weights_path": str(classifier_weights_path),
        "output_path": str(output_path),
        "frontdoor_action_idx": frontdoor_action_idx,
        "total_rows": int(N_full),
        "frontdoor_rows": int(N),
        "frontdoor_positives": int(n_pos),
        "frontdoor_negatives": int(n_neg),
        "val_rows": int(len(y_val)),
        "val_positive_rate": float(y_val.mean()),
        "best_softmax_baseline_name": base_name,
        "frontdoor": {"brier": brier_fd, "auc": auc_fd, "ece": ece_fd, "acc": acc_fd},
        "softmax_max": {"brier": brier_sm_max, "auc": auc_sm_max, "ece": ece_sm_max},
        "softmax_fd": {"brier": brier_sm_fd, "auc": auc_sm_fd, "ece": ece_sm_fd},
        "constant_base_rate": {"brier": brier_constant},
        "brier_delta_vs_best_softmax_baseline": brier_delta,
        "brier_delta_vs_constant_baseline": brier_delta_constant,
        "gates_passed": all_pass,
        "gates": {
            "brier_delta_ge_0_02": bool(brier_delta >= 0.02),
            "auc_ge_0_75": bool(auc_fd >= 0.75),
            "ece_le_0_05": bool(ece_fd <= 0.05),
        },
    }


def _summary_markdown(summary: dict) -> str:
    frontdoor = summary["frontdoor"]
    return "\n".join(
        [
            "# Offline Frontdoor Verifier Evaluation",
            "",
            f"- Data: `{summary['data_path']}`",
            f"- Output weights: `{summary['output_path']}`",
            f"- Frontdoor rows: `{summary['frontdoor_rows']}` "
            f"({summary['frontdoor_positives']} positive / {summary['frontdoor_negatives']} negative)",
            f"- Validation rows: `{summary['val_rows']}`",
            f"- Brier: `{frontdoor['brier']:.4f}`",
            f"- ROC-AUC: `{frontdoor['auc']:.4f}`",
            f"- ECE: `{frontdoor['ece']:.4f}`",
            f"- Accuracy@0.5: `{frontdoor['acc']:.4f}`",
            f"- Delta Brier vs best softmax baseline: "
            f"`{summary['brier_delta_vs_best_softmax_baseline']:+.4f}`",
            f"- Delta Brier vs constant base-rate baseline: "
            f"`{summary['brier_delta_vs_constant_baseline']:+.4f}`",
            f"- Gates passed: `{summary['gates_passed']}`",
            "",
            "This is an offline evaluation artifact. It is not a live verifier",
            "weight promotion and does not enable the frontdoor verifier gate.",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="P6.2 A2: frontdoor-specialist verifier")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--classifier-weights", type=str, required=True)
    parser.add_argument("--classifier-data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--summary-json", type=str)
    parser.add_argument("--summary-md", type=str)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    summary = train_and_eval(
        data_path=Path(args.data),
        classifier_weights_path=Path(args.classifier_weights),
        classifier_data_path=Path(args.classifier_data),
        output_path=Path(args.output),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
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
