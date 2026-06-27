"""Tests for EV-9 pure rubric-scoring helpers."""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator/scripts/autopilot")
sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator")

from rubric_scoring import (  # noqa: E402
    DRACO_CONTENT_DIMENSIONS,
    MINDDR_PROCESS_DIMENSIONS,
    RubricCriterion,
    aggregate_rubric_score,
    judge_ranking_stability,
    saturated_items,
)
from safety_gate import EvalResult  # noqa: E402


def test_aggregate_rubric_score_separates_positive_and_negative_axes() -> None:
    result = aggregate_rubric_score(
        {"accuracy": 0.9, "missing_citations": 0.25},
        (
            RubricCriterion("accuracy", weight=2.0, polarity="positive"),
            RubricCriterion("missing_citations", weight=1.0, polarity="negative"),
        ),
    )

    assert result.positive_score == 0.9
    assert result.negative_penalty == 0.25
    assert result.score == 0.65
    assert result.missing_criteria == ()


def test_aggregate_rubric_score_reports_missing_criteria() -> None:
    result = aggregate_rubric_score(
        {"reasoning_trajectory": 0.8},
        (
            RubricCriterion("reasoning_trajectory"),
            RubricCriterion("citation"),
        ),
    )

    assert result.score == 0.8
    assert result.missing_criteria == ("citation",)


def test_default_rubric_dimensions_include_minddr_and_draco_axes() -> None:
    criteria_names = {
        *MINDDR_PROCESS_DIMENSIONS,
        *DRACO_CONTENT_DIMENSIONS,
    }
    scores = {name: 0.5 for name in criteria_names}

    result = aggregate_rubric_score(scores)

    assert result.score == 0.5
    assert result.missing_criteria == ()


def test_saturated_items_flags_any_candidate_above_threshold() -> None:
    saturated = saturated_items({
        "easy": {"frontdoor": 0.91, "worker": 0.75},
        "hard": {"frontdoor": 0.62, "worker": 0.70},
    })

    assert saturated == {"easy": 0.91}


def test_judge_ranking_stability_uses_bradley_terry_consensus() -> None:
    stability = judge_ranking_stability({
        "judge_a": {"alpha": 0.9, "beta": 0.7, "gamma": 0.2},
        "judge_b": {"alpha": 0.8, "beta": 0.6, "gamma": 0.1},
    })

    assert stability.consensus.ranking[0] == "alpha"
    assert stability.top_choice_agreement is True
    assert stability.mean_spearman == 1.0


def test_judge_ranking_stability_surfaces_disagreement() -> None:
    stability = judge_ranking_stability({
        "judge_a": {"alpha": 0.9, "beta": 0.2, "gamma": 0.1},
        "judge_b": {"alpha": 0.1, "beta": 0.9, "gamma": 0.2},
    })

    assert stability.top_choice_agreement is False
    assert stability.mean_spearman < 1.0


def test_autopilot_eval_result_emits_populated_rubric_metrics_only() -> None:
    result = EvalResult(
        tier=2,
        quality=1.0,
        speed=10.0,
        cost=0.5,
        reliability=0.9,
        rubric_reasoning_trajectory=0.75,
        rubric_outline=0.5,
    )

    lines = result.to_grep_lines()

    assert "METRIC rubric_reasoning_trajectory: 0.7500" in lines
    assert "METRIC rubric_outline: 0.5000" in lines
    assert "rubric_tool_calls" not in lines
    assert "rubric_content_stage" not in lines
    assert math.isnan(result.rubric_tool_calls)
