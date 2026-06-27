#!/usr/bin/env python3
"""Observed-outcome IRT feature ablation for cached P4.5 soft-label data.

This is a diagnostic follow-up to the label-proxy P4.1.3 ablation. It uses the
cached journal-derived soft-label dataset, fits IRT item scores from the
training split's observed per-role correctness matrix, projects those scores
from BGE features, and compares the same hard-label classifier with and without
the two projected IRT columns.

The script never writes production routing weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.irt_scorer import estimate_irt_scores, fit_embedding_projector  # noqa: E402
from scripts.graph_router.train_routing_classifier_kl import (  # noqa: E402
    _argmax_match_acc,
    _role_success_acc,
    _train_arm,
)

DEFAULT_DATA = PROJECT_ROOT / "orchestration/reports/p45_soft_labels/soft_labels_embedded.npz"
DEFAULT_LABELS = PROJECT_ROOT / "orchestration/reports/p45_soft_labels/soft_labels.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "orchestration/reports/p413_observed_irt_feature_ablation/report.json"

logger = logging.getLogger("observed_irt_feature_ablation")


def _label_map_from_npz(data: Any) -> dict[int, str]:
    raw = data["label_map"]
    return {int(row[0]): str(row[1]) for row in raw}


def _observed_mask_from_jsonl(
    labels_path: Path,
    *,
    qids: np.ndarray,
    label_map: dict[int, str],
) -> np.ndarray:
    """Build a qid x action mask distinguishing observed failures from unseen roles."""

    qid_to_idx = {str(qid): idx for idx, qid in enumerate(qids)}
    role_to_idx = {role: idx for idx, role in label_map.items()}
    mask = np.zeros((len(qids), len(label_map)), dtype=bool)
    if not labels_path.exists():
        return mask

    with labels_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            row_idx = qid_to_idx.get(str(record.get("qid", "")))
            if row_idx is None:
                continue
            for role in record.get("roles_seen", []):
                col_idx = role_to_idx.get(str(role))
                if col_idx is not None:
                    mask[row_idx, col_idx] = True
    return mask


def load_observed_split(
    data_path: Path,
    labels_path: Path,
    *,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Load cached soft-label embeddings and create a seeded train/val split."""

    data = np.load(data_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    hard = np.asarray(data["hard_labels"], dtype=np.int64)
    correctness = np.asarray(data["correctness"], dtype=np.float32)
    qids = np.asarray(data["qids"], dtype=object)
    suites = np.asarray(data["suites"], dtype=object)
    label_map = _label_map_from_npz(data)
    observed_mask = _observed_mask_from_jsonl(labels_path, qids=qids, label_map=label_map)
    if not np.any(observed_mask):
        observed_mask = correctness > 0.0

    n = X.shape[0]
    if n == 0:
        raise ValueError("observed soft-label dataset is empty")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return {
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "hard_train": hard[train_idx],
        "hard_val": hard[val_idx],
        "correctness_train": correctness[train_idx],
        "correctness_val": correctness[val_idx],
        "observed_mask_train": observed_mask[train_idx],
        "observed_mask_val": observed_mask[val_idx],
        "qids_train": qids[train_idx],
        "qids_val": qids[val_idx],
        "suites_val": suites[val_idx],
        "label_map": label_map,
        "source_rows": int(n),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "observed_cells": int(observed_mask.sum()),
    }


def _one_hot(labels: np.ndarray, n_actions: int) -> np.ndarray:
    targets = np.zeros((len(labels), n_actions), dtype=np.float32)
    targets[np.arange(len(labels)), labels] = 1.0
    return targets


def _train_hard_arm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    correctness_val: np.ndarray,
    *,
    label_map: dict[int, str],
    epochs: int,
    lr: float,
    patience: int,
    batch_size: int,
    seed: int,
    arm_name: str,
) -> dict[str, Any]:
    n_actions = len(label_map)
    clf, history = _train_arm(
        X_train,
        _one_hot(y_train, n_actions),
        X_val,
        _one_hot(y_val, n_actions),
        n_actions=n_actions,
        label_map=label_map,
        epochs=epochs,
        lr=lr,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        arm_name=arm_name,
    )
    probs, _ = clf.forward(X_val)
    return {
        "input_dim": int(X_train.shape[1]),
        "role_success_acc": round(_role_success_acc(clf, X_val, correctness_val), 6),
        "argmax_match_acc": round(_argmax_match_acc(clf, X_val, y_val), 6),
        "val_loss": round(float(-np.log(np.clip(probs[np.arange(len(y_val)), y_val], 1e-7, 1.0)).mean()), 6),
        "epochs_run": len(history),
        "history_tail": history[-5:],
    }


def append_observed_irt_features(
    split: dict[str, Any],
    *,
    embedding_dim: int = 1024,
    platt_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Append IRT features fitted from train-split observed outcome cells only."""

    responses = np.where(
        split["observed_mask_train"],
        split["correctness_train"],
        np.nan,
    ).astype(np.float32)
    weights = np.where(split["observed_mask_train"], 1.0, 0.0).astype(np.float32)
    scores = estimate_irt_scores(
        responses,
        mask=split["observed_mask_train"],
        sample_weights=weights,
        platt_iterations=platt_iterations,
    )
    projector = fit_embedding_projector(split["X_train"], scores, embedding_dim=embedding_dim)
    train_difficulty, train_discrimination = projector.predict(split["X_train"])
    val_difficulty, val_discrimination = projector.predict(split["X_val"])

    train_extra = np.stack([train_difficulty, train_discrimination], axis=1).astype(np.float32)
    val_extra = np.stack([val_difficulty, val_discrimination], axis=1).astype(np.float32)
    mean = train_extra.mean(axis=0)
    scale = train_extra.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    train_extra = (train_extra - mean) / scale
    val_extra = (val_extra - mean) / scale
    metadata = {
        "embedding_dim": int(projector.embedding_dim),
        "train_feature_mean": [float(v) for v in mean],
        "train_feature_scale": [float(v) for v in scale],
        "platt_slope": float(scores.platt_slope),
        "platt_intercept": float(scores.platt_intercept),
        "observed_response_cells_train": int(split["observed_mask_train"].sum()),
    }
    return (
        np.concatenate([split["X_train"], train_extra], axis=1),
        np.concatenate([split["X_val"], val_extra], axis=1),
        metadata,
    )


def run_observed_ablation(
    data_path: Path = DEFAULT_DATA,
    labels_path: Path = DEFAULT_LABELS,
    *,
    val_split: float = 0.2,
    embedding_dim: int = 1024,
    epochs: int = 300,
    lr: float = 0.01,
    batch_size: int = 64,
    patience: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.time()
    split = load_observed_split(data_path, labels_path, val_split=val_split, seed=seed)
    common = {
        "label_map": split["label_map"],
        "epochs": epochs,
        "lr": lr,
        "patience": patience,
        "batch_size": batch_size,
        "seed": seed,
    }
    baseline = _train_hard_arm(
        split["X_train"],
        split["hard_train"],
        split["X_val"],
        split["hard_val"],
        split["correctness_val"],
        arm_name="observed_hard_baseline",
        **common,
    )
    X_train_irt, X_val_irt, irt_metadata = append_observed_irt_features(
        split,
        embedding_dim=embedding_dim,
    )
    observed_irt = _train_hard_arm(
        X_train_irt,
        split["hard_train"],
        X_val_irt,
        split["hard_val"],
        split["correctness_val"],
        arm_name="observed_irt_augmented",
        **common,
    )
    delta_pp = (observed_irt["role_success_acc"] - baseline["role_success_acc"]) * 100.0
    decision = "escalate_observed_irt_features" if delta_pp >= 1.0 else "do_not_escalate_observed_irt_features"
    return {
        "schema": "epyc.graph_router.observed_irt_feature_ablation.v1",
        "source": str(data_path),
        "labels_source": str(labels_path),
        "response_source": "journal_observed_per_role_correctness_cached_embeddings",
        "source_rows": split["source_rows"],
        "train_rows": split["train_rows"],
        "val_rows": split["val_rows"],
        "observed_cells": split["observed_cells"],
        "actions": {str(k): v for k, v in split["label_map"].items()},
        "baseline": baseline,
        "observed_irt_augmented": observed_irt,
        "delta_role_success_acc_pp": round(delta_pp, 4),
        "decision_gate": {
            "metric": "role_success_acc",
            "threshold_pp": 1.0,
            "decision": decision,
            "promotion_grade": False,
            "note": "cached observed-outcome diagnostic only; no production weights written",
        },
        "irt_metadata": irt_metadata,
        "seconds": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run observed-outcome IRT feature ablation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    report = run_observed_ablation(
        data_path=args.data,
        labels_path=args.labels,
        val_split=args.val_split,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        logger.info("Wrote report: %s", args.report)
        logger.info(
            "baseline=%.4f observed_irt=%.4f delta=%.2f pp decision=%s",
            report["baseline"]["role_success_acc"],
            report["observed_irt_augmented"]["role_success_acc"],
            report["delta_role_success_acc_pp"],
            report["decision_gate"]["decision"],
        )


if __name__ == "__main__":
    main()
