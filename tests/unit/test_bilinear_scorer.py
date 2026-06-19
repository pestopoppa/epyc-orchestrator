"""Tests for DAR-4 BilinearScorer."""

from pathlib import Path

import numpy as np
import pytest

from orchestration.repl_memory.bilinear_scorer import (
    BilinearScorer,
    ModelFeatures,
    extract_model_features,
    extract_prompt_features,
    PROMPT_FEATURE_DIM,
    MODEL_FEATURE_DIM,
)
from orchestration.repl_memory.q_scorer import ScoringConfig


class TestModelFeatures:
    """Test ModelFeatures vector extraction."""

    def test_to_vector_shape(self):
        mf = ModelFeatures(role="test", baseline_tps=10.0, baseline_quality=0.8)
        v = mf.to_vector()
        assert v.shape == (MODEL_FEATURE_DIM,)

    def test_to_vector_normalized(self):
        mf = ModelFeatures(
            role="test",
            baseline_tps=40.0,  # max → 1.0
            baseline_quality=1.0,
            memory_cost=5.0,  # max → 1.0
            param_count_log=10.0,  # max → 1.0
            is_moe=1.0,
            quant_bits=16.0,  # max → 1.0
        )
        v = mf.to_vector()
        assert all(v <= 1.0 + 1e-6)

    def test_extract_from_scoring_config(self):
        cfg = ScoringConfig()
        features = extract_model_features(cfg)
        assert "frontdoor" in features
        assert "architect_general" in features
        assert "worker_general" in features
        assert "worker_explore" not in features
        assert features["frontdoor"].baseline_tps == 24.3
        assert features["frontdoor"].is_moe == 1.0
        assert features["frontdoor"].quant_bits == 8.0

    def test_extract_from_stack_priors(self, tmp_path: Path):
        stack_priors = tmp_path / "stack_priors.yaml"
        stack_priors.write_text(
            """
roles:
  frontdoor:
    model:
      arch: moe-a3b
      params_b: 35
      active_b: 3
      quant: Q8_0
  worker_general:
    model:
      arch: moe-a4b
      params_b: 26
      active_b: 4
      quant: Q4_K_M
""",
            encoding="utf-8",
        )
        cfg = ScoringConfig(
            baseline_tps_by_role={"frontdoor": 24.3, "worker_general": 60.7},
            baseline_quality_by_role={"frontdoor": 0.9, "worker_general": 0.8},
            memory_cost_by_role={"frontdoor": 1.0, "worker_general": 1.0},
        )

        features = extract_model_features(cfg, stack_priors_path=stack_priors)

        assert features["frontdoor"].param_count_log == pytest.approx(np.log2(35))
        assert features["frontdoor"].is_moe == 1.0
        assert features["frontdoor"].quant_bits == 8.0
        assert features["worker_general"].param_count_log == pytest.approx(np.log2(26))
        assert features["worker_general"].quant_bits == 4.0

    def test_extract_model_features_uses_degraded_fallback_when_priors_missing(self, tmp_path: Path):
        cfg = ScoringConfig(
            baseline_tps_by_role={"worker_general": 60.7},
            baseline_quality_by_role={"worker_general": 0.745},
            memory_cost_by_role={"worker_general": 1.0},
        )

        features = extract_model_features(cfg, stack_priors_path=tmp_path / "missing.yaml")

        assert features["worker_general"].param_count_log == pytest.approx(np.log2(26))
        assert features["worker_general"].is_moe == 1.0
        assert features["worker_general"].quant_bits == 4.0

    def test_extract_model_features_degraded_fallback_uses_registry_descriptors(
        self,
        tmp_path: Path,
    ):
        registry = tmp_path / "model_registry.yaml"
        registry.write_text(
            """
roles:
  novel_router:
    model:
      name: NovelRouter-42B-MoE-Q5_K_M
      quant: Q5_K_M
      size_gb: 42
      architecture: custom_moe
server_mode: {}
""",
            encoding="utf-8",
        )
        cfg = ScoringConfig(
            baseline_tps_by_role={"novel_router": 12.5},
            baseline_quality_by_role={"novel_router": 0.81},
            memory_cost_by_role={"novel_router": 2.0},
        )

        features = extract_model_features(
            cfg,
            stack_priors_path=tmp_path / "missing.yaml",
            registry_path=registry,
        )

        assert features["novel_router"].param_count_log == pytest.approx(np.log2(42))
        assert features["novel_router"].is_moe == 1.0
        assert features["novel_router"].quant_bits == 5.0


class TestPromptFeatures:
    """Test prompt feature extraction."""

    def test_shape(self):
        task_ir = {"objective": "Write a function to sort a list", "task_type": "coding"}
        v = extract_prompt_features(task_ir)
        assert v.shape == (PROMPT_FEATURE_DIM,)

    def test_code_detection(self):
        code_ir = {"objective": "def sort_list(): implement the function"}
        plain_ir = {"objective": "What is the capital of France?"}
        code_feat = extract_prompt_features(code_ir)
        plain_feat = extract_prompt_features(plain_ir)
        # Code feature (index 1) should be higher for code task
        assert code_feat[1] > plain_feat[1]

    def test_math_detection(self):
        math_ir = {"objective": "Solve the integral of x^2 dx"}
        plain_ir = {"objective": "Tell me about dogs"}
        math_feat = extract_prompt_features(math_ir)
        plain_feat = extract_prompt_features(plain_ir)
        assert math_feat[2] > plain_feat[2]

    def test_empty_task_ir(self):
        v = extract_prompt_features({})
        assert v.shape == (PROMPT_FEATURE_DIM,)
        assert np.isfinite(v).all()


class TestBilinearScorer:
    """Test BilinearScorer core functionality."""

    def _make_scorer(self):
        features = {
            "frontdoor": ModelFeatures(
                role="frontdoor", baseline_tps=12.7, baseline_quality=0.895,
                memory_cost=1.0, param_count_log=5.13, is_moe=1.0, quant_bits=4.0,
            ),
            "architect_general": ModelFeatures(
                role="architect_general", baseline_tps=4.3, baseline_quality=0.94,
                memory_cost=3.0, param_count_log=6.93, is_moe=1.0, quant_bits=4.0,
            ),
            "worker_general": ModelFeatures(
                role="worker_general", baseline_tps=39.1, baseline_quality=0.745,
                memory_cost=0.5, param_count_log=4.91, is_moe=1.0, quant_bits=4.0,
            ),
        }
        return BilinearScorer(features)

    def test_predict_returns_valid_range(self):
        scorer = self._make_scorer()
        prompt = np.random.randn(PROMPT_FEATURE_DIM).astype(np.float32)
        q = scorer.predict(prompt, "frontdoor")
        assert 0.0 <= q <= 1.0

    def test_predict_unknown_role_returns_neutral(self):
        scorer = self._make_scorer()
        prompt = np.random.randn(PROMPT_FEATURE_DIM).astype(np.float32)
        q = scorer.predict(prompt, "unknown_role")
        assert q == 0.5

    def test_predict_all_returns_all_roles(self):
        scorer = self._make_scorer()
        prompt = np.random.randn(PROMPT_FEATURE_DIM).astype(np.float32)
        scores = scorer.predict_all(prompt)
        assert set(scores.keys()) == {"frontdoor", "architect_general", "worker_general"}
        for q in scores.values():
            assert 0.0 <= q <= 1.0

    def test_legacy_worker_explore_alias_canonicalizes_to_worker_general(self):
        scorer = self._make_scorer()
        prompt = extract_prompt_features({"objective": "Solve a complex proof"})

        for _ in range(10):
            scorer.update(prompt, "worker_explore", reward=1.0)

        assert "worker_explore" not in scorer.model_features
        assert scorer.predict(prompt, "worker_explore") == pytest.approx(
            scorer.predict(prompt, "worker_general")
        )
        assert scorer.predict_all(prompt).keys() == {"frontdoor", "architect_general", "worker_general"}

    def test_update_changes_prediction(self):
        scorer = self._make_scorer()
        prompt = extract_prompt_features({"objective": "Write a sort function"})

        q_before = scorer.predict(prompt, "frontdoor")

        # Strong positive signal
        for _ in range(50):
            scorer.update(prompt, "frontdoor", reward=1.0)

        q_after = scorer.predict(prompt, "frontdoor")
        assert q_after > q_before

    def test_update_with_negative_reward_decreases_q(self):
        scorer = self._make_scorer()
        prompt = extract_prompt_features({"objective": "Solve a complex proof"})

        # First push Q up
        for _ in range(20):
            scorer.update(prompt, "worker_general", reward=1.0)
        q_high = scorer.predict(prompt, "worker_general")

        # Then push Q down
        for _ in range(50):
            scorer.update(prompt, "worker_general", reward=-1.0)
        q_low = scorer.predict(prompt, "worker_general")

        assert q_low < q_high

    def test_get_best_role(self):
        scorer = self._make_scorer()
        prompt = extract_prompt_features({"objective": "Write code"})

        # Train one role to be clearly best, others clearly worst
        for _ in range(200):
            scorer.update(prompt, "frontdoor", reward=1.0)
            scorer.update(prompt, "worker_general", reward=-1.0)
            scorer.update(prompt, "architect_general", reward=-1.0)

        best = scorer.get_best_role(prompt)
        assert best == "frontdoor"

    def test_zero_cold_start_new_model(self):
        """New model with known specs can score immediately."""
        scorer = self._make_scorer()
        # Add a new model at runtime
        scorer.model_features["new_model"] = ModelFeatures(
            role="new_model", baseline_tps=25.0, baseline_quality=0.88,
            memory_cost=1.5, param_count_log=5.0, is_moe=False, quant_bits=8.0,
        )
        prompt = extract_prompt_features({"objective": "Translate text"})
        q = scorer.predict(prompt, "new_model")
        assert 0.0 <= q <= 1.0  # Can score without any training data

    def test_save_load_roundtrip(self, tmp_path):
        scorer = self._make_scorer()
        prompt = extract_prompt_features({"objective": "Test"})

        # Train
        for _ in range(10):
            scorer.update(prompt, "frontdoor", reward=0.8)

        q_before = scorer.predict(prompt, "frontdoor")
        path = str(tmp_path / "bilinear.npz")
        scorer.save(path)

        # Load into new scorer
        scorer2 = self._make_scorer()
        assert scorer2.load(path)
        q_after = scorer2.predict(prompt, "frontdoor")
        assert q_before == pytest.approx(q_after, abs=1e-6)

    def test_load_nonexistent_returns_false(self):
        scorer = self._make_scorer()
        assert scorer.load("/nonexistent/path.npz") is False
