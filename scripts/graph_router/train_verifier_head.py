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
    val_seed: int = 42,
    val_split: float = 0.2,
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

    # ── Train ──
    verifier = VerifierHead(
        feature_dim=feature_dim,
        n_actions=n_actions,
        hidden1=64,
        hidden2=32,
    )
    logger.info("Verifier params: %d (input_dim=%d)", verifier.param_count, verifier.input_dim)

    history = verifier.train(
        Z, correct, sample_weights,
        epochs=epochs,
        lr=lr,
        val_split=val_split,
        patience=patience,
        batch_size=batch_size,
        rng_seed=val_seed,
    )

    # ── Recompute the same val split for evaluation ──
    rng = np.random.default_rng(val_seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(1, int(N * val_split))
    val_idx = idx[:n_val]
    Z_val = Z[val_idx]
    y_val = correct[val_idx]
    actions_val = actions[val_idx]

    # Verifier predictions on val
    p_verifier, _ = verifier.forward(Z_val)
    pred_verifier = (p_verifier >= 0.5).astype(np.float32)
    acc_verifier = float((pred_verifier == y_val).mean())
    brier_verifier = _brier(p_verifier, y_val)
    auc_verifier = _roc_auc(p_verifier, y_val.astype(int))
    ece_verifier = _ece(p_verifier, y_val)

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
    # ── Critical: align the val split.  The verifier extractor preserves row
    # order, so val_idx (computed above) indexes the same rows in X_raw.
    X_val = X_raw[val_idx]
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
    print("=" * 90)

    _reliability(p_verifier, y_val, n_bins=10)

    # ── Decision gate ──
    # Pick the stronger of the two baselines for the Brier comparison
    base_brier = min(brier_sm_taken, brier_sm_max)
    base_brier_name = "softmax_taken" if brier_sm_taken < brier_sm_max else "softmax_max"
    brier_delta = base_brier - brier_verifier   # positive = improvement
    brier_delta_constant = brier_base - brier_verifier
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
    print(
        f"\nP6.2.5 overall: {'PASS — wire ORCHESTRATOR_VERIFIER_GATE (default OFF)' if all_pass else 'FAIL — archive Hypothesis C; close Phase 6'}"
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
        "val_rows": int(len(y_val)),
        "feature_dim": int(feature_dim),
        "n_actions": int(n_actions),
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
        f"- Validation rows: `{summary['val_rows']}`",
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
        "",
        "This is an offline evaluation artifact. It is not a live verifier",
        "weight promotion and does not enable the verifier gate.",
        "",
    ]
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
