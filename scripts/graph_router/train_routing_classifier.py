#!/usr/bin/env python3
"""Train routing classifier from extracted training data.

Loads the training data produced by extract_training_data.py, trains a 2-layer
MLP with Q-value weighted cross-entropy loss, and saves weights.

Usage:
    python3 scripts/graph_router/train_routing_classifier.py [--data PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_routing_classifier")

DEFAULT_DATA_PATH = PROJECT_ROOT / "orchestration/repl_memory/training_data.npz"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "orchestration/repl_memory/routing_classifier_weights.npz"


def train(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_WEIGHTS_PATH,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 30,
    batch_size: int = 64,
) -> bool:
    """Train routing classifier.

    Returns:
        True on success.
    """
    from orchestration.repl_memory.routing_classifier import RoutingClassifier

    logger.info("=== Routing Classifier Training ===")
    t0 = time.time()

    # Load training data
    if not data_path.exists():
        logger.error("Training data not found at %s. Run extract_training_data.py first.", data_path)
        return False

    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    q_weights = data["q_weights"]
    label_map_raw = data["label_map"]

    # Reconstruct label_map
    label_map = {int(row[0]): str(row[1]) for row in label_map_raw}

    logger.info(
        "Training data: %d samples, %d features, %d actions",
        X.shape[0], X.shape[1], len(label_map),
    )
    logger.info("Actions: %s", label_map)

    if X.shape[0] < 50:
        logger.warning("Very few training samples (%d). Classifier may not generalize well.", X.shape[0])

    # Create and train classifier
    clf = RoutingClassifier(
        input_dim=X.shape[1],
        n_actions=len(label_map),
        label_map=label_map,
    )
    logger.info("Classifier: %d parameters", clf.param_count)

    history = clf.train(
        X, y, q_weights,
        epochs=epochs,
        lr=lr,
        patience=patience,
        batch_size=batch_size,
    )

    # ── Evaluation on held-out validation set ──
    # Re-split to get a clean val set for evaluation + calibration
    rng = np.random.default_rng(42)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    n_val = max(1, int(X.shape[0] * 0.2))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_val, y_val = X[val_idx], y[val_idx]

    # Train accuracy
    probs_train, _ = clf.forward(X[train_idx])
    preds_train = np.argmax(probs_train, axis=1)
    train_acc = float((preds_train == y[train_idx]).mean())

    # Val accuracy
    probs_val, _ = clf.forward(X_val)
    preds_val = np.argmax(probs_val, axis=1)
    val_acc = float((preds_val == y_val).mean())

    # Per-action accuracy (on val set)
    logger.info("\nPer-action accuracy (validation set, %d samples):", len(y_val))
    for idx, action in sorted(label_map.items()):
        mask = y_val == idx
        if mask.sum() > 0:
            acc = float((preds_val[mask] == y_val[mask]).mean())
            logger.info("  [%d] %s: %.1f%% (%d samples)", idx, action, acc * 100, int(mask.sum()))

    # Confusion matrix on val set
    n_actions = len(label_map)
    confusion = np.zeros((n_actions, n_actions), dtype=np.int32)
    for true, pred in zip(y_val, preds_val):
        confusion[true, pred] += 1

    logger.info("\nConfusion matrix — validation (rows=true, cols=predicted):")
    header = "      " + "".join(f"{i:>6}" for i in range(n_actions))
    logger.info(header)
    for i in range(n_actions):
        if confusion[i].sum() == 0:
            continue
        row = f"  [{i}] " + "".join(f"{confusion[i, j]:>6}" for j in range(n_actions))
        logger.info(row)

    # ── Calibrate per-class confidence thresholds ──
    logger.info("\nCalibrating per-class confidence thresholds...")
    thresholds = clf.calibrate_thresholds(X_val, y_val, target_precision=0.9)

    # Random baseline
    random_baseline = 1.0 / max(n_actions, 1)

    # Save weights + thresholds
    clf.save(output_path)

    elapsed = time.time() - t0
    logger.info(
        "\n=== Training complete in %.1fs ===\n"
        "  Train accuracy: %.1f%%\n"
        "  Val accuracy: %.1f%% (random baseline: %.1f%%)\n"
        "  Best val loss: %.4f\n"
        "  Weights: %s",
        elapsed,
        train_acc * 100,
        val_acc * 100,
        random_baseline * 100,
        min(history["val_loss"]) if history["val_loss"] else float("inf"),
        output_path,
    )

    return True


def main():
    parser = argparse.ArgumentParser(description="Train routing classifier")
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA_PATH),
        help="Path to training data (.npz)",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_WEIGHTS_PATH),
        help="Output path for classifier weights",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stop patience")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    args = parser.parse_args()

    success = train(
        data_path=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
