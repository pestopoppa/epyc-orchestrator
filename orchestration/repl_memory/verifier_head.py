"""VerifierHead: per-decision correctness probability for routing decisions.

P6.2 (Hypothesis C from research/deep-dives/2026-05-21-recursive-reasoning-routing.md).

A small numpy MLP that takes (request_features ⊕ action_one_hot) and emits
P(action is correct) ∈ [0, 1]. Trained on (existing-features, action_taken,
q_value > 0.5) triples mined from the same reembedded.npz the routing
classifier was trained from.

Architecture: input(D+A) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid).
Pure numpy — no PyTorch dependency. Inference: <0.5ms on CPU.

Wiring (P6.2.5 gate-pass conditional):
    classifier top_class ─► verifier(features ⊕ one_hot(top_class)) ─► P_correct
        P_correct ≥ τ ─► route via classifier
        P_correct < τ ─► fall through to KNN retriever
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/verifier_head_weights.npz"
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


class VerifierHead:
    """2-layer numpy MLP — joint (features, action) → P(correct).

    Args:
        feature_dim: dimension of the request feature vector (matches classifier input_dim).
        n_actions: number of action classes (one-hot concatenated to features).
        hidden1, hidden2: hidden layer widths.
    """

    def __init__(
        self,
        feature_dim: int = 1031,
        n_actions: int = 8,
        hidden1: int = 64,
        hidden2: int = 32,
    ):
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.input_dim = feature_dim + n_actions
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        rng = np.random.default_rng(42)

        def xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

        self._weights: Dict[str, np.ndarray] = {
            "W1": xavier(self.input_dim, hidden1),
            "b1": np.zeros(hidden1, dtype=np.float32),
            "W2": xavier(hidden1, hidden2),
            "b2": np.zeros(hidden2, dtype=np.float32),
            "W3": xavier(hidden2, 1),
            "b3": np.zeros(1, dtype=np.float32),
        }

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self._weights.values())

    @staticmethod
    def join(features: np.ndarray, action_idx: int, n_actions: int) -> np.ndarray:
        """Concatenate a feature vector with a one-hot action.

        When n_actions == 0 (single-action specialist, e.g. frontdoor-only),
        action_idx is ignored and the features are returned unchanged.
        """
        feats = features.astype(np.float32)
        if n_actions <= 0:
            return feats
        oh = np.zeros(n_actions, dtype=np.float32)
        oh[action_idx] = 1.0
        return np.concatenate([feats, oh], axis=-1)

    @staticmethod
    def join_batch(X: np.ndarray, actions: np.ndarray, n_actions: int) -> np.ndarray:
        """Vectorized join — X is (N, D), actions is (N,) integer.

        When n_actions == 0, actions is ignored and X is returned unchanged.
        """
        X = X.astype(np.float32)
        if n_actions <= 0:
            return X
        N = X.shape[0]
        oh = np.zeros((N, n_actions), dtype=np.float32)
        oh[np.arange(N), actions] = 1.0
        return np.concatenate([X, oh], axis=-1)

    def forward(self, Z: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass — Z is the JOINED input (N, feature_dim + n_actions).

        Returns:
            (p_correct, cache) where p_correct is (N,) in [0,1].
        """
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)

        z1 = Z @ self._weights["W1"] + self._weights["b1"]
        a1 = _relu(z1)
        z2 = a1 @ self._weights["W2"] + self._weights["b2"]
        a2 = _relu(z2)
        z3 = a2 @ self._weights["W3"] + self._weights["b3"]
        p = _sigmoid(z3).reshape(-1)

        cache = {"Z": Z, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}
        return p, cache

    def predict(self, features: np.ndarray, action_idx: int) -> float:
        """Predict P(correct) for a single (features, action) pair."""
        z = self.join(features, action_idx, self.n_actions)
        p, _ = self.forward(z)
        return float(p[0])

    def _compute_loss(
        self,
        p: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        """Weighted binary cross-entropy."""
        eps = 1e-7
        p_clipped = np.clip(p, eps, 1.0 - eps)
        loss = -(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
        return float(np.sum(sample_weights * loss) / np.sum(sample_weights))

    def _backward(
        self,
        p: np.ndarray,
        cache: Dict[str, np.ndarray],
        y: np.ndarray,
        sample_weights: np.ndarray,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        loss = self._compute_loss(p, y, sample_weights)

        w_norm = sample_weights / np.sum(sample_weights)
        dz3 = ((p - y) * w_norm).reshape(-1, 1)  # (N, 1)

        grads: Dict[str, np.ndarray] = {}
        grads["W3"] = cache["a2"].T @ dz3
        grads["b3"] = dz3.sum(axis=0)

        da2 = dz3 @ self._weights["W3"].T
        dz2 = da2 * _relu_grad(cache["z2"])
        grads["W2"] = cache["a1"].T @ dz2
        grads["b2"] = dz2.sum(axis=0)

        da1 = dz2 @ self._weights["W2"].T
        dz1 = da1 * _relu_grad(cache["z1"])
        grads["W1"] = cache["Z"].T @ dz1
        grads["b1"] = dz1.sum(axis=0)

        return loss, grads

    def train(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        epochs: int = 200,
        lr: float = 0.01,
        val_split: float = 0.2,
        patience: int = 30,
        batch_size: int = 256,
        rng_seed: int = 42,
    ) -> Dict[str, List[float]]:
        """Train with mini-batch SGD and weighted BCE."""
        N = Z.shape[0]
        if sample_weights is None:
            sample_weights = np.ones(N, dtype=np.float32)

        rng = np.random.default_rng(rng_seed)
        indices = np.arange(N)
        rng.shuffle(indices)
        n_val = max(1, int(N * val_split))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        Z_train, y_train, w_train = Z[train_idx], y[train_idx], sample_weights[train_idx]
        Z_val, y_val, w_val = Z[val_idx], y[val_idx], sample_weights[val_idx]

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_weights = None
        no_improve = 0

        for epoch in range(epochs):
            perm = rng.permutation(len(Z_train))
            Z_train, y_train, w_train = Z_train[perm], y_train[perm], w_train[perm]

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(Z_train), batch_size):
                end = min(start + batch_size, len(Z_train))
                Z_b, y_b, w_b = Z_train[start:end], y_train[start:end], w_train[start:end]

                p, cache = self.forward(Z_b)
                loss, grads = self._backward(p, cache, y_b, w_b)
                epoch_loss += loss
                n_batches += 1

                current_lr = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
                for key in self._weights:
                    self._weights[key] -= current_lr * grads[key]

            train_loss = epoch_loss / max(n_batches, 1)

            p_val, _ = self.forward(Z_val)
            val_loss = self._compute_loss(p_val, y_val, w_val)
            val_acc = float(((p_val >= 0.5).astype(np.float32) == y_val).mean())

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.copy() for k, v in self._weights.items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  patience=%d/%d",
                    epoch, epochs, train_loss, val_loss, val_acc, no_improve, patience,
                )

            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_weights:
            self._weights = best_weights

        return history

    def save(self, path: Path = DEFAULT_WEIGHTS_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = dict(self._weights)
        save_dict["_config"] = np.array(
            [self.feature_dim, self.n_actions, self.hidden1, self.hidden2],
            dtype=np.int64,
        )
        np.savez_compressed(path, **save_dict)
        logger.info("Saved verifier weights to %s (%d params)", path, self.param_count)

    @classmethod
    def load(cls, path: Path = DEFAULT_WEIGHTS_PATH) -> Optional["VerifierHead"]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=True)
            cfg = data["_config"]
            v = cls(
                feature_dim=int(cfg[0]),
                n_actions=int(cfg[1]),
                hidden1=int(cfg[2]),
                hidden2=int(cfg[3]),
            )
            for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
                v._weights[key] = data[key].astype(np.float32)
            logger.info("Loaded verifier: %d params", v.param_count)
            return v
        except Exception as e:
            logger.warning("Failed to load verifier weights: %s", e)
            return None
