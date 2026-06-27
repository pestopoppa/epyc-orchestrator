#!/usr/bin/env python3
"""P4.2 block-epsilon separability diagnostic for the routing classifier.

This trains the existing numpy RoutingClassifier architecture under three
connectivity regimes on the same data split:

- full: normal dense MLP.
- block10: block-masked feature trunk with a dense classifier head.
- diagonal: one-to-one-ish trunk connections with a dense classifier head.

The output is an evidence artifact only. It never writes production classifier
weights.
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

DEFAULT_DATA = PROJECT_ROOT / "orchestration/repl_memory/training_data.npz"
DEFAULT_REPORT = PROJECT_ROOT / "orchestration/reports/p42_block_separability/report.json"

logger = logging.getLogger("block_separability_diagnostic")


def _label_map_from_npz(data: Any) -> dict[int, str]:
    raw = data["label_map"]
    return {int(row[0]): str(row[1]) for row in raw}


def _partition(size: int, blocks: int) -> np.ndarray:
    if size < 1:
        raise ValueError("partition size must be positive")
    if blocks < 1:
        raise ValueError("blocks must be positive")
    return np.arange(size, dtype=np.int64) * blocks // size


def _diagonal_mask(rows: int, cols: int) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=np.float32)
    for col in range(cols):
        start = col * rows // cols
        end = (col + 1) * rows // cols
        if end <= start:
            end = min(rows, start + 1)
        mask[start:end, col] = 1.0
    return mask


def connectivity_masks(
    *,
    input_dim: int,
    hidden1: int,
    hidden2: int,
    n_actions: int,
    variant: str,
    blocks: int = 10,
) -> dict[str, np.ndarray]:
    """Return multiplicative masks for classifier weights."""
    if variant == "full":
        return {
            "W1": np.ones((input_dim, hidden1), dtype=np.float32),
            "W2": np.ones((hidden1, hidden2), dtype=np.float32),
            "W3": np.ones((hidden2, n_actions), dtype=np.float32),
        }
    if variant == "block10":
        input_blocks = _partition(input_dim, blocks)
        h1_blocks = _partition(hidden1, blocks)
        h2_blocks = _partition(hidden2, blocks)
        return {
            "W1": (input_blocks[:, None] == h1_blocks[None, :]).astype(np.float32),
            "W2": (h1_blocks[:, None] == h2_blocks[None, :]).astype(np.float32),
            # Keep the final action head dense. Otherwise n_actions < blocks makes
            # the diagnostic mostly a class-partition test rather than a feature
            # separability test.
            "W3": np.ones((hidden2, n_actions), dtype=np.float32),
        }
    if variant == "diagonal":
        return {
            "W1": _diagonal_mask(input_dim, hidden1),
            "W2": _diagonal_mask(hidden1, hidden2),
            "W3": np.ones((hidden2, n_actions), dtype=np.float32),
        }
    raise ValueError(f"unknown variant: {variant}")


def _apply_masks(clf: RoutingClassifier, masks: dict[str, np.ndarray]) -> None:
    for key, mask in masks.items():
        clf._weights[key] *= mask


def _loss(probs: np.ndarray, y: np.ndarray, q_weights: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-7, 1.0)
    logs = np.log(clipped[np.arange(len(y)), y])
    denom = float(np.sum(q_weights))
    if denom <= 0.0:
        denom = float(len(q_weights))
        q_weights = np.ones_like(q_weights, dtype=np.float32)
    return float(-np.sum(q_weights * logs) / denom)


def _accuracy(clf: RoutingClassifier, X: np.ndarray, y: np.ndarray) -> float:
    probs, _ = clf.forward(X)
    return float((np.argmax(probs, axis=1) == y).mean())


def train_variant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    q_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    q_val: np.ndarray,
    *,
    label_map: dict[int, str],
    variant: str,
    hidden1: int = 128,
    hidden2: int = 64,
    blocks: int = 10,
    epochs: int = 80,
    lr: float = 0.01,
    batch_size: int = 256,
    patience: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    clf = RoutingClassifier(
        input_dim=X_train.shape[1],
        hidden1=hidden1,
        hidden2=hidden2,
        n_actions=len(label_map),
        label_map=label_map,
    )
    masks = connectivity_masks(
        input_dim=X_train.shape[1],
        hidden1=hidden1,
        hidden2=hidden2,
        n_actions=len(label_map),
        variant=variant,
        blocks=blocks,
    )
    _apply_masks(clf, masks)

    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_weights: dict[str, np.ndarray] | None = None
    no_improve = 0
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(epochs):
        perm = rng.permutation(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        current_lr = lr * 0.5 * (1.0 + np.cos(np.pi * epoch / max(epochs, 1)))

        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            X_b = X_train[batch_idx]
            y_b = y_train[batch_idx]
            q_b = q_train[batch_idx]
            probs, cache = clf.forward(X_b)
            loss, grads = clf._backward(probs, cache, y_b, q_b)
            epoch_loss += loss
            n_batches += 1
            for key in clf._weights:
                grad = grads[key]
                if key in masks:
                    grad = grad * masks[key]
                clf._weights[key] -= current_lr * grad
            _apply_masks(clf, masks)

        val_probs, _ = clf.forward(X_val)
        val_loss = _loss(val_probs, y_val, q_val)
        val_acc = _accuracy(clf, X_val, y_val)
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
    return {
        "variant": variant,
        "val_acc": _accuracy(clf, X_val, y_val),
        "val_loss": _loss(clf.forward(X_val)[0], y_val, q_val),
        "epochs_run": len(history),
        "seconds": round(time.time() - started, 3),
        "active_weight_fraction": {
            key: round(float(mask.mean()), 6)
            for key, mask in masks.items()
        },
        "history_tail": history[-5:],
    }


def load_training_split(
    data_path: Path,
    *,
    max_samples: int | None,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    data = np.load(data_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    q = data["q_weights"].astype(np.float32)
    label_map = _label_map_from_npz(data)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    if max_samples is not None and max_samples > 0:
        idx = idx[: min(max_samples, len(idx))]
    n_val = max(1, int(len(idx) * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], q[train_idx], X[val_idx], y[val_idx], q[val_idx], label_map


def _class_distribution(y: np.ndarray, label_map: dict[int, str]) -> dict[str, int]:
    counts = np.bincount(y, minlength=len(label_map))
    return {
        label_map.get(i, f"action_{i}"): int(count)
        for i, count in enumerate(counts)
        if count
    }


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    X_train, y_train, q_train, X_val, y_val, q_val, label_map = load_training_split(
        args.data,
        max_samples=args.max_samples,
        val_split=args.val_split,
        seed=args.seed,
    )
    variants = ["full", "block10", "diagonal"]
    results = {}
    for variant in variants:
        logger.info("Training %s arm", variant)
        results[variant] = train_variant(
            X_train,
            y_train,
            q_train,
            X_val,
            y_val,
            q_val,
            label_map=label_map,
            variant=variant,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            blocks=args.blocks,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
        )

    full_acc = float(results["full"]["val_acc"])
    block_acc = float(results["block10"]["val_acc"])
    diagonal_acc = float(results["diagonal"]["val_acc"])
    class_counts = _class_distribution(y_val, label_map)
    majority_count = max(class_counts.values()) if class_counts else 0
    majority_acc = majority_count / max(len(y_val), 1)
    tolerance = args.tolerance_pp / 100.0
    min_full_lift = args.min_full_lift_pp / 100.0
    full_has_signal = full_acc - majority_acc >= min_full_lift
    block_within_gate = full_has_signal and full_acc - block_acc <= tolerance
    if not full_has_signal:
        interpretation = "insufficient_full_rank_signal"
    elif block_within_gate:
        interpretation = "block_epsilon_separable_candidate"
    else:
        interpretation = "full_rank_dominates"
    report = {
        "schema": "epyc.lrc.block_separability.v1",
        "data": str(args.data),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "input_dim": int(X_train.shape[1]),
        "n_actions": int(len(label_map)),
        "label_map": label_map,
        "validation_class_counts": class_counts,
        "majority_baseline_acc": round(float(majority_acc), 6),
        "config": {
            "max_samples": args.max_samples,
            "val_split": args.val_split,
            "seed": args.seed,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "blocks": args.blocks,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "tolerance_pp": args.tolerance_pp,
            "min_full_lift_pp": args.min_full_lift_pp,
        },
        "results": results,
        "deltas": {
            "block10_minus_full_acc": round(block_acc - full_acc, 6),
            "diagonal_minus_full_acc": round(diagonal_acc - full_acc, 6),
            "full_minus_majority_acc": round(full_acc - majority_acc, 6),
        },
        "decision": {
            "block10_within_tolerance": bool(block_within_gate),
            "full_rank_signal_above_majority": bool(full_has_signal),
            "interpretation": interpretation,
        },
    }
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-samples", type=int, default=80000)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--tolerance-pp", type=float, default=2.0)
    parser.add_argument("--min-full-lift-pp", type=float, default=2.0)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"training data not found: {args.data}")
    report = run_diagnostic(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
