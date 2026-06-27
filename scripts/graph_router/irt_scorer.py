#!/usr/bin/env python3
"""Estimate IRT-style prompt difficulty and discrimination scores.

The scorer accepts either a direct response matrix (`responses`: prompts x
models/actions, values in {0, 1}, NaN for missing) or the existing graph-router
training NPZ (`X`, `y`, `q_weights`, `label_map`).  The latter is treated as a
label-proxy response matrix so the output is suitable for cold-start selection
experiments, not as promotion-grade model-outcome evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.action_space import canonical_actions_from_npz, infer_n_actions

logger = logging.getLogger("irt_scorer")

DEFAULT_DATA_PATH = PROJECT_ROOT / "orchestration/repl_memory/training_data.npz"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "orchestration/repl_memory/irt_prompt_scores.npz"


@dataclass(frozen=True)
class IRTScores:
    """Prompt-level IRT-style score bundle."""

    latent_difficulty: np.ndarray
    latent_discrimination: np.ndarray
    calibrated_success: np.ndarray
    action_abilities: np.ndarray
    platt_slope: float
    platt_intercept: float


@dataclass(frozen=True)
class EmbeddingIRTProjector:
    """Linear prompt-embedding projector for future prompt scoring."""

    embedding_dim: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    def predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] < self.embedding_dim:
            raise ValueError(f"expected at least {self.embedding_dim} embedding features")
        emb = matrix[:, : self.embedding_dim]
        scaled = (emb - self.feature_mean) / self.feature_scale
        design = np.concatenate([scaled, np.ones((scaled.shape[0], 1), dtype=np.float64)], axis=1)
        pred = design @ self.weights
        pred = pred * self.target_scale + self.target_mean
        return pred[:, 0].astype(np.float32), np.maximum(pred[:, 1], 0.0).astype(np.float32)


def _clip_prob(values: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return np.clip(values.astype(np.float64), eps, 1.0 - eps)


def _logit(values: np.ndarray) -> np.ndarray:
    probs = _clip_prob(values)
    return np.log(probs / (1.0 - probs))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_platt_scaler(
    raw_logits: np.ndarray,
    responses: np.ndarray,
    mask: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    lr: float = 0.1,
    iterations: int = 200,
) -> tuple[float, float]:
    """Fit a global Platt scaling layer for raw item/action logits."""

    observed = mask & np.isfinite(raw_logits) & np.isfinite(responses)
    if not np.any(observed):
        return 1.0, 0.0

    x = raw_logits[observed].astype(np.float64)
    y = responses[observed].astype(np.float64)
    if sample_weights is None:
        weights = np.ones_like(y)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)[observed]
        weights = np.maximum(weights, 0.0)
    denom = max(float(weights.sum()), 1e-8)

    slope = 1.0
    intercept = 0.0
    for _ in range(iterations):
        pred = _sigmoid(slope * x + intercept)
        err = (pred - y) * weights
        slope -= lr * float(np.dot(err, x) / denom)
        intercept -= lr * float(err.sum() / denom)

    return float(slope), float(intercept)


def estimate_irt_scores(
    responses: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
    platt_iterations: int = 200,
) -> IRTScores:
    """Estimate prompt difficulty and discrimination from response outcomes."""

    response_matrix = np.asarray(responses, dtype=np.float64)
    if response_matrix.ndim != 2:
        raise ValueError("responses must be a 2-D prompt x action/model matrix")
    if response_matrix.size == 0:
        raise ValueError("responses matrix is empty")

    observed = np.isfinite(response_matrix)
    if mask is not None:
        observed &= np.asarray(mask, dtype=bool)
    if not np.any(observed):
        raise ValueError("responses matrix has no observed entries")

    if sample_weights is None:
        weights = np.ones_like(response_matrix, dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != response_matrix.shape:
            raise ValueError("sample_weights must match responses shape")
        weights = np.maximum(weights, 0.0)
    weights = np.where(observed, weights, 0.0)

    action_weight = np.maximum(weights.sum(axis=0), 1e-8)
    action_success = (np.where(observed, response_matrix, 0.0) * weights).sum(axis=0) / action_weight
    action_abilities = _logit(action_success)
    action_abilities -= float(action_abilities.mean())

    item_weight = np.maximum(weights.sum(axis=1), 1e-8)
    item_success = (np.where(observed, response_matrix, 0.0) * weights).sum(axis=1) / item_weight
    difficulty = -_logit(item_success)

    centered_actions = np.where(observed, action_abilities[None, :], 0.0)
    action_mean = (centered_actions * weights).sum(axis=1) / item_weight
    action_delta = np.where(observed, action_abilities[None, :] - action_mean[:, None], 0.0)
    response_delta = np.where(observed, response_matrix - item_success[:, None], 0.0)
    cov = (weights * action_delta * response_delta).sum(axis=1) / item_weight
    var = (weights * action_delta * action_delta).sum(axis=1) / item_weight
    discrimination = np.abs(cov / np.maximum(var, 1e-8))
    discrimination = np.nan_to_num(discrimination, nan=0.0, posinf=0.0, neginf=0.0)

    raw_logits = discrimination[:, None] * (action_abilities[None, :] - difficulty[:, None])
    slope, intercept = fit_platt_scaler(
        raw_logits,
        response_matrix,
        observed,
        sample_weights=weights,
        iterations=platt_iterations,
    )
    calibrated_matrix = _sigmoid(slope * raw_logits + intercept)
    calibrated_success = (calibrated_matrix * weights).sum(axis=1) / item_weight

    return IRTScores(
        latent_difficulty=difficulty.astype(np.float32),
        latent_discrimination=discrimination.astype(np.float32),
        calibrated_success=calibrated_success.astype(np.float32),
        action_abilities=action_abilities.astype(np.float32),
        platt_slope=slope,
        platt_intercept=intercept,
    )


def fit_embedding_projector(
    embeddings: np.ndarray,
    scores: IRTScores,
    *,
    embedding_dim: int = 1024,
    ridge_lambda: float = 1e-2,
) -> EmbeddingIRTProjector:
    """Fit a small ridge regressor from BGE embeddings to IRT item scores."""

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2-D matrix")
    if matrix.shape[1] < embedding_dim:
        raise ValueError(f"embeddings has {matrix.shape[1]} columns; need {embedding_dim}")

    emb = matrix[:, :embedding_dim]
    targets = np.stack(
        [
            np.asarray(scores.latent_difficulty, dtype=np.float64),
            np.asarray(scores.latent_discrimination, dtype=np.float64),
        ],
        axis=1,
    )
    if emb.shape[0] != targets.shape[0]:
        raise ValueError("embeddings and score arrays must have the same row count")

    feature_mean = emb.mean(axis=0)
    feature_scale = emb.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
    target_mean = targets.mean(axis=0)
    target_scale = targets.std(axis=0)
    target_scale = np.where(target_scale < 1e-6, 1.0, target_scale)

    design = (emb - feature_mean) / feature_scale
    design = np.concatenate([design, np.ones((design.shape[0], 1), dtype=np.float64)], axis=1)
    target_scaled = (targets - target_mean) / target_scale
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target_scaled)

    return EmbeddingIRTProjector(
        embedding_dim=embedding_dim,
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
        weights=weights.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_scale=target_scale.astype(np.float32),
    )


def predict_irt_from_embeddings(
    embeddings: np.ndarray,
    *,
    embedding_dim: int,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    weights: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score embeddings using a persisted IRT projector."""

    projector = EmbeddingIRTProjector(
        embedding_dim=embedding_dim,
        feature_mean=np.asarray(feature_mean),
        feature_scale=np.asarray(feature_scale),
        weights=np.asarray(weights),
        target_mean=np.asarray(target_mean),
        target_scale=np.asarray(target_scale),
    )
    return projector.predict(embeddings)


def _label_map_from_npz(data: Any) -> dict[int, str]:
    actions = canonical_actions_from_npz(data)
    if actions:
        return {idx: action for idx, action in enumerate(actions)}
    if "label_map" not in data.files:
        return {}
    raw = data["label_map"]
    return {int(row[0]): str(row[1]) for row in raw}


def _features_from_npz(data: Any, max_items: int | None) -> np.ndarray | None:
    if "X" not in data.files:
        return None
    features = data["X"]
    if max_items is not None:
        features = features[:max_items]
    return features


def load_response_dataset(data_path: Path, *, max_items: int | None = None) -> dict[str, Any]:
    """Load direct responses or build a label-proxy response matrix."""

    data = np.load(data_path, allow_pickle=True)
    if "responses" in data.files:
        responses = np.asarray(data["responses"], dtype=np.float64)
        if max_items is not None:
            responses = responses[:max_items]
        mask = np.isfinite(responses)
        label_map = _label_map_from_npz(data)
        return {
            "responses": responses,
            "mask": mask,
            "sample_weights": None,
            "label_map": label_map,
            "response_source": "observed_matrix",
            "feature_rows": responses.shape[0],
            "features": _features_from_npz(data, max_items),
        }

    required = {"X", "y"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{data_path} is missing required arrays: {sorted(missing)}")

    y = np.asarray(data["y"], dtype=np.int64)
    if max_items is not None:
        y = y[:max_items]
    label_map = _label_map_from_npz(data)
    n_actions = max(len(label_map), infer_n_actions(data, y))
    responses = np.zeros((y.shape[0], n_actions), dtype=np.float32)
    responses[np.arange(y.shape[0]), y] = 1.0
    weights = np.ones_like(responses, dtype=np.float32)
    if "q_weights" in data.files:
        q_weights = np.asarray(data["q_weights"], dtype=np.float32)
        if max_items is not None:
            q_weights = q_weights[:max_items]
        weights *= np.maximum(q_weights[:, None], 0.01)

    feature_rows = int(data["X"].shape[0])
    if max_items is not None:
        feature_rows = min(feature_rows, max_items)
    return {
        "responses": responses,
        "mask": np.ones_like(responses, dtype=bool),
        "sample_weights": weights,
        "label_map": label_map,
        "response_source": "label_proxy",
        "feature_rows": feature_rows,
        "features": data["X"][:feature_rows],
    }


def score_file(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    max_items: int | None = None,
    platt_iterations: int = 200,
) -> dict[str, Any]:
    """Score a response/training NPZ and write prompt-level IRT arrays."""

    t0 = time.time()
    dataset = load_response_dataset(data_path, max_items=max_items)
    scores = estimate_irt_scores(
        dataset["responses"],
        mask=dataset["mask"],
        sample_weights=dataset["sample_weights"],
        platt_iterations=platt_iterations,
    )
    projector = None
    if dataset.get("features") is not None:
        projector = fit_embedding_projector(dataset["features"], scores)

    metadata = {
        "schema": "epyc.graph_router.irt_prompt_scores.v1",
        "source": str(data_path),
        "response_source": dataset["response_source"],
        "items": int(scores.latent_difficulty.shape[0]),
        "actions": int(scores.action_abilities.shape[0]),
        "feature_rows": int(dataset["feature_rows"]),
        "max_items": max_items,
        "platt_slope": scores.platt_slope,
        "platt_intercept": scores.platt_intercept,
        "embedding_projector": projector is not None,
        "embedding_dim": projector.embedding_dim if projector is not None else None,
        "elapsed_s": round(time.time() - t0, 3),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "latent_difficulty": scores.latent_difficulty,
        "latent_discrimination": scores.latent_discrimination,
        "calibrated_success": scores.calibrated_success,
        "action_abilities": scores.action_abilities,
        "label_map": np.array(list(dataset["label_map"].items()), dtype=object),
        "metadata": np.array(metadata, dtype=object),
    }
    if projector is not None:
        payload.update(
            {
                "embedding_dim": np.array(projector.embedding_dim, dtype=np.int64),
                "projector_feature_mean": projector.feature_mean,
                "projector_feature_scale": projector.feature_scale,
                "projector_weights": projector.weights,
                "projector_target_mean": projector.target_mean,
                "projector_target_scale": projector.target_scale,
            }
        )
    np.savez_compressed(output_path, **payload)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate IRT prompt difficulty/discrimination scores")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input response/training NPZ")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output IRT scores NPZ")
    parser.add_argument("--max-items", type=int, default=None, help="Optional row cap for smoke tests")
    parser.add_argument("--platt-iterations", type=int, default=200, help="Platt scaler gradient steps")
    parser.add_argument("--json", action="store_true", help="Print metadata as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    metadata = score_file(
        data_path=args.data,
        output_path=args.output,
        max_items=args.max_items,
        platt_iterations=args.platt_iterations,
    )
    if args.json:
        print(json.dumps(metadata, sort_keys=True))
    else:
        logger.info("Wrote IRT scores: %s", args.output)
        logger.info("Metadata: %s", metadata)


if __name__ == "__main__":
    main()
