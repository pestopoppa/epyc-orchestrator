"""Smoke tests for GEPA ↔ autopilot integration (AP-19).

Tests adapter structure, PromptMutation compatibility, and dispatch wiring.
Does NOT require live model servers — all inference is mocked.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add autopilot to path
_autopilot_dir = Path(__file__).resolve().parents[1] / "scripts" / "autopilot"
sys.path.insert(0, str(_autopilot_dir))


# ── Import tests ──────────────────────────────────────────────────


def test_gepa_optimizer_imports():
    """gepa_optimizer module imports without error."""
    from species.gepa_optimizer import (
        GEPAOptResult,
        GEPAPromptOptimizer,
        OrchestratorGEPAAdapter,
    )
    assert GEPAOptResult is not None
    assert GEPAPromptOptimizer is not None
    assert OrchestratorGEPAAdapter is not None


def test_gepa_in_mutation_types():
    """'gepa' is a valid PromptForge mutation type."""
    from species.prompt_forge import MUTATION_TYPES
    assert "gepa" in MUTATION_TYPES


# ── GEPAOptResult tests ──────────────────────────────────────────


def test_gepa_opt_result_to_prompt_mutation():
    """GEPAOptResult converts to a valid PromptMutation."""
    from species.gepa_optimizer import GEPAOptResult

    result = GEPAOptResult(
        target_file="frontdoor.md",
        original_content="old prompt",
        best_content="new prompt",
        best_score=0.85,
        baseline_score=0.70,
        n_evals=50,
        elapsed_s=120.0,
        improvement=0.15,
    )

    mutation = result.to_prompt_mutation()
    assert mutation.file == "frontdoor.md"
    assert mutation.mutation_type == "gepa"
    assert mutation.original_content == "old prompt"
    assert mutation.mutated_content == "new prompt"
    assert "0.700" in mutation.description
    assert "0.850" in mutation.description


def test_gepa_opt_result_no_improvement():
    """GEPAOptResult with no improvement reports correctly."""
    from species.gepa_optimizer import GEPAOptResult

    result = GEPAOptResult(
        target_file="frontdoor.md",
        original_content="prompt",
        best_content="prompt",
        best_score=0.70,
        baseline_score=0.70,
        n_evals=50,
        elapsed_s=60.0,
        improvement=0.0,
    )
    assert not result.improved


# ── OrchestratorGEPAAdapter tests ────────────────────────────────


def test_adapter_evaluate_returns_evaluation_batch():
    """Adapter.evaluate() returns properly structured EvaluationBatch."""
    from gepa.core.adapter import EvaluationBatch
    from species.gepa_optimizer import OrchestratorGEPAAdapter

    # Mock eval_tower and prompt_forge
    mock_tower = MagicMock()
    mock_tower.timeout = 30

    # Mock _eval_question to return a QuestionResult-like object
    mock_result = MagicMock()
    mock_result.correct = True
    mock_result.answer = "Canberra"
    mock_result.route_used = "frontdoor"
    mock_result.error = None
    mock_result.elapsed_s = 1.5
    mock_tower._eval_question.return_value = mock_result

    mock_forge = MagicMock()

    adapter = OrchestratorGEPAAdapter(
        eval_tower=mock_tower,
        prompt_forge=mock_forge,
        target_file="frontdoor.md",
    )

    batch = [
        {"prompt": "What is the capital of Australia?", "expected": "Canberra", "suite": "general"},
    ]
    candidate = {"prompt": "You are a helpful routing assistant."}

    result = adapter.evaluate(batch, candidate, capture_traces=True)

    assert isinstance(result, EvaluationBatch)
    assert len(result.scores) == 1
    assert result.scores[0] == 1.0
    assert result.trajectories is not None
    assert len(result.trajectories) == 1
    assert result.trajectories[0]["correct"] is True

    # Verify prompt was written to disk
    mock_forge.write_prompt.assert_called_once_with(
        "frontdoor.md", "You are a helpful routing assistant."
    )


def test_adapter_evaluate_wrong_answer():
    """Adapter scores 0.0 for incorrect answers."""
    from species.gepa_optimizer import OrchestratorGEPAAdapter

    mock_tower = MagicMock()
    mock_tower.timeout = 30
    mock_result = MagicMock()
    mock_result.correct = False
    mock_result.answer = "Sydney"
    mock_result.route_used = "frontdoor"
    mock_result.error = None
    mock_result.elapsed_s = 2.0
    mock_tower._eval_question.return_value = mock_result

    mock_forge = MagicMock()

    adapter = OrchestratorGEPAAdapter(
        eval_tower=mock_tower,
        prompt_forge=mock_forge,
    )

    batch = [{"prompt": "Capital?", "expected": "Canberra", "suite": "general"}]
    candidate = {"prompt": "system prompt"}

    result = adapter.evaluate(batch, candidate)
    assert result.scores[0] == 0.0


def test_adapter_reflective_dataset():
    """make_reflective_dataset produces per-component feedback."""
    from gepa.core.adapter import EvaluationBatch
    from species.gepa_optimizer import OrchestratorGEPAAdapter

    mock_tower = MagicMock()
    mock_forge = MagicMock()
    adapter = OrchestratorGEPAAdapter(mock_tower, mock_forge)

    eval_batch = EvaluationBatch(
        outputs=[{"answer": "Canberra"}],
        scores=[1.0],
        trajectories=[{
            "question": "Capital of Australia?",
            "expected": "Canberra",
            "suite": "general",
            "answer": "Canberra",
            "route": "frontdoor",
            "correct": True,
            "error": None,
        }],
    )

    dataset = adapter.make_reflective_dataset(
        candidate={"prompt": "test"},
        eval_batch=eval_batch,
        components_to_update=["prompt"],
    )

    assert "prompt" in dataset
    assert len(dataset["prompt"]) == 1
    assert "CORRECT" in dataset["prompt"][0]["feedback"]


# ── PromptForge GEPA dispatch tests ─────────────────────────────


def test_prompt_forge_gepa_mutation_type():
    """PromptForge.propose_mutation dispatches to GEPA for mutation_type='gepa'."""
    from species.prompt_forge import PromptForge

    forge = PromptForge(prompts_dir=Path("/tmp/test_prompts"))

    # Should raise ValueError if no eval_tower
    with pytest.raises(ValueError, match="eval_tower"):
        forge.propose_mutation(
            target_file="frontdoor.md",
            mutation_type="gepa",
            eval_tower=None,
        )


# ── Autopilot dispatch tests ────────────────────────────────────


def test_auto_action_gepa_split():
    """_auto_action produces gepa_optimize actions ~30% of the time."""
    from autopilot import _auto_action

    mock_seeder = MagicMock()
    results = []
    # Run enough times to check distribution
    for _ in range(1000):
        action = _auto_action("prompt_forge", 600, True, mock_seeder)
        results.append(action["type"])

    gepa_count = results.count("gepa_optimize")
    llm_count = results.count("prompt_mutation")
    total = gepa_count + llm_count

    # Should be roughly 30% GEPA (allow 20-40% range for randomness)
    ratio = gepa_count / total
    assert 0.20 < ratio < 0.40, f"GEPA ratio {ratio:.2f} outside expected 0.20-0.40"


def test_validate_single_variable_gepa():
    """gepa_optimize passes single-variable validation."""
    from autopilot import _validate_single_variable

    # Valid
    assert _validate_single_variable(
        {"type": "gepa_optimize", "file": "frontdoor.md"}
    ) is None

    # Invalid: no file
    assert _validate_single_variable(
        {"type": "gepa_optimize"}
    ) is not None

    # Invalid: multiple files
    assert _validate_single_variable(
        {"type": "gepa_optimize", "file": "a.md,b.md"}
    ) is not None
