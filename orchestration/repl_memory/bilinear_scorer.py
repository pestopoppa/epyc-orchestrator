"""DAR-4: Bilinear Q-scorer for zero cold-start model routing.

Replaces per-action Q-values with a feature-conditioned scorer:

    Q(prompt, model) = sigmoid(v_model^T W v_prompt + b)

where v_model is a fixed feature vector derived from model registry specs
(baseline_tps, baseline_quality, quality_known, memory_cost,
param_count_log, is_moe, quant_bits), and v_prompt is derived from the prompt embedding or task
features. W is a learned interaction matrix.

Key advantage: new models score immediately from their spec features,
with no routing history needed. Training data comes from the episodic
store's (embedding, action, reward) tuples.

Feature-flagged: BILINEAR_SCORER_ENABLED=1 to activate.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.roles import Role
from src.registry.stack_priors import live_stack_role_records
from src.registry.model_descriptors import compile_model_descriptors

_REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

BILINEAR_SCORER_ENABLED = os.environ.get("BILINEAR_SCORER_ENABLED", "0") == "1"

# Model feature dimension (7 features per model).
# 7, not 6: baseline_quality gained a companion quality_known mask on
# 2026-08-02. Any persisted W from before that is dimensionally stale and
# must be retrained rather than loaded.
MODEL_FEATURE_DIM = 7

# Prompt feature dimension (extracted from task IR)
PROMPT_FEATURE_DIM = 8

DEFAULT_STACK_PRIORS_PATH = _REPO_ROOT / "orchestration/derived/stack_priors.yaml"
DEFAULT_REGISTRY_PATH = _REPO_ROOT / "orchestration/model_registry.yaml"

UNKNOWN_MODEL_FEATURES = {"params_b": 30.0, "is_moe": False, "quant_bits": 4.0}


@dataclass
class ModelFeatures:
    """Feature vector for a model/role, derived from registry specs."""

    role: str
    baseline_tps: float = 0.0
    baseline_quality: float = 0.0
    # Missing-data mask for baseline_quality. Without it, "no measurement" and
    # "measured this value" are the same float and the model cannot tell them
    # apart — the standard way an imputed default gets learned as if it were
    # evidence.
    quality_known: float = 0.0
    memory_cost: float = 0.0
    param_count_log: float = 0.0  # log2(param_count_B)
    is_moe: float = 0.0  # 1.0 for MoE, 0.0 for dense
    quant_bits: float = 4.0  # Q4=4, Q8=8, FP16=16

    def to_vector(self) -> np.ndarray:
        """Convert to normalized feature vector."""
        raw = np.array([
            self.baseline_tps / 40.0,  # normalize to ~[0,1]
            self.baseline_quality,
            self.quality_known,
            self.memory_cost / 5.0,  # normalize
            self.param_count_log / 10.0,  # log2(1024B) ≈ 10
            self.is_moe,
            self.quant_bits / 16.0,
        ], dtype=np.float32)
        return raw


def _quant_bits(quant: Any) -> float:
    value = str(quant or "").upper()
    if "Q8" in value:
        return 8.0
    if "Q6" in value:
        return 6.0
    if "Q5" in value:
        return 5.0
    if "Q4" in value:
        return 4.0
    if "F16" in value or "FP16" in value:
        return 16.0
    return 4.0


def _stack_prior_model_features(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> dict[str, dict[str, float | bool]]:
    features: dict[str, dict[str, float | bool]] = {}
    for role, record in live_stack_role_records(stack_priors_path).items():
        model = record.get("model")
        if not isinstance(model, dict):
            continue
        params_b = model.get("params_b")
        if not isinstance(params_b, (int, float)) or params_b <= 0:
            continue
        arch = str(model.get("arch") or "").lower()
        active_b = model.get("active_b")
        is_moe = "moe" in arch or isinstance(active_b, (int, float))
        features[role] = {
            "params_b": float(params_b),
            "is_moe": bool(is_moe),
            "quant_bits": _quant_bits(model.get("quant")),
        }
    return features


def _descriptor_model_features(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    *,
    active_roles: set[str] | None = None,
) -> dict[str, dict[str, float | bool]]:
    features: dict[str, dict[str, float | bool]] = {}
    try:
        descriptors = compile_model_descriptors(
            lean_registry_path=registry_path,
            research_registry_path=None,
            active_roles=active_roles,
            allow_incomplete=True,
        )
    except Exception as exc:
        logger.debug("Could not compile degraded bilinear model descriptors: %s", exc)
        return features

    models = descriptors.get("models")
    if not isinstance(models, list):
        return features

    for descriptor in models:
        if not isinstance(descriptor, dict):
            continue
        params_b = descriptor.get("params_b")
        if not isinstance(params_b, (int, float)) or params_b <= 0:
            continue
        arch = str(descriptor.get("arch") or "").lower()
        active_b = descriptor.get("active_b")
        specs = {
            "params_b": float(params_b),
            "is_moe": bool("moe" in arch or isinstance(active_b, (int, float))),
            "quant_bits": _quant_bits(descriptor.get("quant")),
        }
        bindings = descriptor.get("role_bindings")
        if not isinstance(bindings, dict):
            continue
        for field in ("roles", "server_roles"):
            values = bindings.get(field)
            if not isinstance(values, list):
                continue
            for role in values:
                if not isinstance(role, str):
                    continue
                canonical = _canonical_role_name(role)
                features.setdefault(canonical, specs)
    return features


def _canonical_role_name(role: str) -> str:
    canonical = Role.from_string(role)
    return canonical.value if canonical is not None else role


def extract_model_features(
    scoring_config,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> Dict[str, ModelFeatures]:
    """Extract model feature vectors from ScoringConfig baselines.

    Args:
        scoring_config: ScoringConfig instance with per-role baselines.

    Returns:
        Dict mapping role name to ModelFeatures.
    """
    features = {}
    tps = scoring_config.baseline_tps_by_role
    quality = scoring_config.baseline_quality_by_role
    memory = scoring_config.memory_cost_by_role

    stack_features = _stack_prior_model_features(stack_priors_path)
    canonical_roles = {_canonical_role_name(role) for role in tps}
    degraded_features = (
        {}
        if stack_features
        else _descriptor_model_features(registry_path, active_roles=canonical_roles)
    )

    for role in tps:
        canonical_role = _canonical_role_name(role)
        specs = stack_features.get(
            canonical_role,
            degraded_features.get(canonical_role, UNKNOWN_MODEL_FEATURES),
        )
        features[canonical_role] = ModelFeatures(
            role=canonical_role,
            baseline_tps=tps.get(role, 10.0),
            # 0.0 with an explicit known-flag, NOT a 0.75 default. 0.75 is the
            # worker baseline the reward's quality_gap is measured against, so
            # defaulting an UNMEASURED role to it asserted "this model is exactly
            # worker-grade" — a specific, load-bearing claim made by omission.
            # architect_general sat at that default while having no measured
            # entry at all. quality_known lets a learner ignore the feature when
            # it is absent instead of training on a fabricated value.
            baseline_quality=float(quality.get(role, 0.0)),
            quality_known=1.0 if role in quality else 0.0,
            memory_cost=memory.get(role, 1.0),
            param_count_log=math.log2(max(float(specs["params_b"]), 1.0)),
            is_moe=float(specs["is_moe"]),
            quant_bits=float(specs["quant_bits"]),
        )

    return features


def extract_prompt_features(task_ir: Dict[str, Any]) -> np.ndarray:
    """Extract prompt feature vector from task IR.

    Uses lightweight heuristics (no model call):
    - prompt length (log-scaled)
    - code presence
    - math presence
    - multi-step indicators
    - constraint count
    - question mark count
    - nesting depth estimate
    - ambiguity markers

    Args:
        task_ir: TaskIR dictionary.

    Returns:
        Normalized feature vector of dimension PROMPT_FEATURE_DIM.
    """
    objective = str(task_ir.get("objective", ""))
    task_type = str(task_ir.get("task_type", ""))
    combined = f"{objective} {task_type}".lower()

    # Feature extraction (mirrors difficulty_signal.py heuristics)
    prompt_len = math.log(max(len(combined), 1)) / 10.0  # log-scaled, ~[0,1]
    code_present = float(any(kw in combined for kw in ["```", "def ", "class ", "import ", "function"]))
    math_present = float(any(kw in combined for kw in ["solve", "equation", "∑", "integral", "derivative", "proof"]))
    multi_step = float(any(kw in combined for kw in ["step by step", "first", "then", "finally", "analyze and"]))
    constraints = min(combined.count("must") + combined.count("should") + combined.count("ensure"), 5) / 5.0
    questions = min(combined.count("?"), 3) / 3.0
    nesting = min(combined.count("(") + combined.count("[") + combined.count("{"), 5) / 5.0
    ambiguity = float(any(kw in combined for kw in ["might", "perhaps", "could be", "unclear"]))

    return np.array([
        prompt_len, code_present, math_present, multi_step,
        constraints, questions, nesting, ambiguity,
    ], dtype=np.float32)


class BilinearScorer:
    """Bilinear Q-scorer: Q(prompt, model) = sigmoid(v_m^T W v_p + b).

    Enables zero cold-start scoring for new models using their spec features.
    Training uses (prompt_features, model_features, reward) tuples from the
    episodic store.

    Args:
        model_features: Dict of role → ModelFeatures.
        learning_rate: SGD learning rate for online updates.
        weight_decay: L2 regularization coefficient.
    """

    def __init__(
        self,
        model_features: Dict[str, ModelFeatures],
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
    ):
        self.model_features = {
            _canonical_role_name(role): feature
            for role, feature in model_features.items()
        }
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Initialize interaction matrix W and bias
        self.W = np.random.randn(MODEL_FEATURE_DIM, PROMPT_FEATURE_DIM).astype(np.float32) * 0.01
        self.b = np.float32(0.0)

        # Training stats
        self.updates = 0

    def predict(self, prompt_features: np.ndarray, role: str) -> float:
        """Predict Q-value for a (prompt, model) pair.

        Args:
            prompt_features: Prompt feature vector (dim PROMPT_FEATURE_DIM).
            role: Model role name.

        Returns:
            Q-value in [0, 1].
        """
        role = _canonical_role_name(role)
        if role not in self.model_features:
            return 0.5  # Neutral for unknown roles

        v_m = self.model_features[role].to_vector()
        v_p = prompt_features

        logit = float(v_m @ self.W @ v_p + self.b)
        return _sigmoid(logit)

    def predict_all(self, prompt_features: np.ndarray) -> Dict[str, float]:
        """Predict Q-values for all known models.

        Args:
            prompt_features: Prompt feature vector.

        Returns:
            Dict of role → Q-value, sorted by Q descending.
        """
        scores = {}
        for role in self.model_features:
            scores[role] = self.predict(prompt_features, role)
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def update(
        self,
        prompt_features: np.ndarray,
        role: str,
        reward: float,
    ) -> float:
        """Online SGD update from an observed (prompt, model, reward) tuple.

        Args:
            prompt_features: Prompt feature vector.
            role: Model role that was used.
            reward: Observed reward in [-1, 1].

        Returns:
            Updated Q-value prediction.
        """
        role = _canonical_role_name(role)
        if role not in self.model_features:
            return 0.5

        v_m = self.model_features[role].to_vector()
        v_p = prompt_features

        # Forward
        logit = float(v_m @ self.W @ v_p + self.b)
        q_hat = _sigmoid(logit)

        # Target: map reward to [0, 1]
        target = 0.5 + reward * 0.5

        # Gradient of BCE loss: d/d_logit = q_hat - target
        error = q_hat - target

        # Gradient for W: outer product of model and prompt features scaled by error
        grad_W = error * np.outer(v_m, v_p)

        # Update with weight decay (L2 regularization)
        self.W -= self.learning_rate * (grad_W + self.weight_decay * self.W)
        self.b -= self.learning_rate * error

        self.updates += 1

        return self.predict(prompt_features, role)

    def get_best_role(self, prompt_features: np.ndarray) -> Optional[str]:
        """Get the highest-scoring role for a prompt.

        Args:
            prompt_features: Prompt feature vector.

        Returns:
            Role name with highest Q-value, or None if no models loaded.
        """
        scores = self.predict_all(prompt_features)
        if not scores:
            return None
        return max(scores, key=scores.get)

    def save(self, path: str) -> None:
        """Save weights to npz file."""
        np.savez(
            path,
            W=self.W,
            b=np.array([float(self.b)]),
            updates=np.array([self.updates]),
        )
        logger.info("BilinearScorer saved to %s (%d updates)", path, self.updates)

    def load(self, path: str) -> bool:
        """Load weights from npz file.

        Returns:
            True if loaded successfully.
        """
        try:
            data = np.load(path)
            self.W = data["W"]
            self.b = np.float32(data["b"][0])
            self.updates = int(data["updates"][0])
            logger.info("BilinearScorer loaded from %s (%d updates)", path, self.updates)
            return True
        except (FileNotFoundError, KeyError) as e:
            logger.debug("BilinearScorer load failed: %s", e)
            return False


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)
