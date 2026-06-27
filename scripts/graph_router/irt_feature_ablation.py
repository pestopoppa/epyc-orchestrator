#!/usr/bin/env python3
"""P4.1.3 IRT-feature ablation for the routing classifier.

This is an evidence-only diagnostic. It never writes production classifier
weights. The IRT targets are fit from the training split only, projected from
BGE features, then appended to both train and validation rows as two extra
features: latent difficulty and latent discrimination.
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

from orchestration.repl_memory.routing_classifier import RoutingClassifier  # noqa: E402
from scripts.graph_router.irt_scorer import estimate_irt_scores, fit_embedding_projector  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "orchestration/repl_memory/training_data.npz"
DEFAULT_REPORT = PROJECT_ROOT / "orchestration/reports/p413_irt_feature_ablation/report.json"

logger = logging.getLogger("irt_feature_ablation")


def _label_map_from_npz(data: Any) -> dict[int, str]:
    raw = data["label_map"]
    return {int(row[0]): str(row[1]) for row in raw}


def _weighted_loss(probs: np.ndarray, y: np.ndarray, q_weights: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-7, 1.0)
    logs = np.log(clipped[np.arange(len(y)), y])
    denom = float(np.sum(q_weights))
    if denom <= 0.0:
        q_weights = np.ones_like(q_weights, dtype=np.float32)
        denom = float(len(q_weights))
    return float(-np.sum(q_weights * logs) / denom)


def _accuracy(clf: RoutingClassifier, X: np.ndarray, y: np.ndarray) -> float:
    probs, _ = clf.forward(X)
    return float((np.argmax(probs, axis=1) == y).mean())


def load_training_split(
    data_path: Path,
    *,
    max_samples: int | None = 80_000,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    data = np.load(data_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    q = np.asarray(data["q_weights"], dtype=np.float32)
    label_map = _label_map_from_npz(data)

    rng = np.random.default_rng(seed)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[: min(max_samples, len(indices))]
    n_val = max(1, int(len(indices) * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "q_train": q[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "q_val": q[val_idx],
        "label_map": label_map,
        "source_rows": int(X.shape[0]),
        "sampled_rows": int(len(indices)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
    }


def _one_hot_responses(y: np.ndarray, n_actions: int) -> np.ndarray:
    responses = np.zeros((len(y), n_actions), dtype=np.float32)
    responses[np.arange(len(y)), y] = 1.0
    return responses


def append_projected_irt_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    q_train: np.ndarray,
    X_val: np.ndarray,
    *,
    n_actions: int,
    embedding_dim: int = 1024,
    platt_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit train-only IRT features and append projected scores to X matrices."""

    responses = _one_hot_responses(y_train, n_actions)
    weights = np.maximum(q_train[:, None], 0.01) * np.ones_like(responses, dtype=np.float32)
    scores = estimate_irt_scores(
        responses,
        sample_weights=weights,
        platt_iterations=platt_iterations,
    )
    projector = fit_embedding_projector(X_train, scores, embedding_dim=embedding_dim)
    train_difficulty, train_discrimination = projector.predict(X_train)
    val_difficulty, val_discrimination = projector.predict(X_val)

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
    }
    return (
        np.concatenate([X_train, train_extra], axis=1),
        np.concatenate([X_val, val_extra], axis=1),
        metadata,
    )


def train_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    q_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    q_val: np.ndarray,
    *,
    label_map: dict[int, str],
    epochs: int = 80,
    lr: float = 0.01,
    batch_size: int = 256,
    patience: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    clf = RoutingClassifier(
        input_dim=X_train.shape[1],
        n_actions=len(label_map),
        label_map=label_map,
    )
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_weights: dict[str, np.ndarray] | None = None
    no_improve = 0
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(epochs):
        perm = rng.permutation(len(X_train))
        current_lr = lr * 0.5 * (1.0 + np.cos(np.pi * epoch / max(epochs, 1)))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            probs, cache = clf.forward(X_train[batch_idx])
            loss, grads = clf._backward(probs, cache, y_train[batch_idx], q_train[batch_idx])
            epoch_loss += loss
            n_batches += 1
            for key in clf._weights:
                clf._weights[key] -= current_lr * grads[key]

        val_probs, _ = clf.forward(X_val)
        val_loss = _weighted_loss(val_probs, y_val, q_val)
        val_acc = float((np.argmax(val_probs, axis=1) == y_val).mean())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(epoch_loss / max(n_batches, 1)),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.copy() for k, v in clf._weights.items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_weights is not None:
        clf._weights = best_weights
    probs, _ = clf.forward(X_val)
    preds = np.argmax(probs, axis=1)
    per_action: dict[str, Any] = {}
    for idx, action in sorted(label_map.items()):
        mask = y_val == idx
        if np.any(mask):
            per_action[action] = {
                "samples": int(mask.sum()),
                "accuracy": float((preds[mask] == y_val[mask]).mean()),
            }
    return {
        "input_dim": int(X_train.shape[1]),
        "val_acc": _accuracy(clf, X_val, y_val),
        "val_loss": _weighted_loss(probs, y_val, q_val),
        "epochs_run": len(history),
        "seconds": round(time.time() - started, 3),
        "history_tail": history[-5:],
        "per_action": per_action,
    }


def run_ablation(
    data_path: Path = DEFAULT_DATA,
    *,
    max_samples: int | None = 80_000,
    embedding_dim: int = 1024,
    epochs: int = 80,
    lr: float = 0.01,
    batch_size: int = 256,
    patience: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    split = load_training_split(data_path, max_samples=max_samples, seed=seed)
    label_map = split["label_map"]
    common = {
        "label_map": label_map,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "seed": seed,
    }

    baseline = train_head(
        split["X_train"],
        split["y_train"],
        split["q_train"],
        split["X_val"],
        split["y_val"],
        split["q_val"],
        **common,
    )
    X_train_irt, X_val_irt, irt_metadata = append_projected_irt_features(
        split["X_train"],
        split["y_train"],
        split["q_train"],
        split["X_val"],
        n_actions=len(label_map),
        embedding_dim=embedding_dim,
    )
    augmented = train_head(
        X_train_irt,
        split["y_train"],
        split["q_train"],
        X_val_irt,
        split["y_val"],
        split["q_val"],
        **common,
    )
    delta_pp = (augmented["val_acc"] - baseline["val_acc"]) * 100.0
    decision = "escalate_irt_features" if delta_pp >= 1.0 else "do_not_escalate_label_proxy_irt"
    return {
        "schema": "epyc.graph_router.irt_feature_ablation.v1",
        "source": str(data_path),
        "response_source": "label_proxy_train_split_only",
        "source_rows": split["source_rows"],
        "sampled_rows": split["sampled_rows"],
        "train_rows": split["train_rows"],
        "val_rows": split["val_rows"],
        "actions": {str(k): v for k, v in label_map.items()},
        "baseline": baseline,
        "irt_augmented": augmented,
        "delta_val_acc_pp": round(delta_pp, 4),
        "decision_gate": {
            "threshold_pp": 1.0,
            "decision": decision,
            "promotion_grade": False,
            "note": "label-proxy evidence only; observed per-model outcome matrix required for promotion",
        },
        "irt_metadata": irt_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P4.1.3 IRT-feature routing ablation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-samples", type=int, default=80_000)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    report = run_ablation(
        data_path=args.data,
        max_samples=args.max_samples,
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
            "baseline=%.4f irt=%.4f delta=%.2f pp decision=%s",
            report["baseline"]["val_acc"],
            report["irt_augmented"]["val_acc"],
            report["delta_val_acc_pp"],
            report["decision_gate"]["decision"],
        )


if __name__ == "__main__":
    main()
