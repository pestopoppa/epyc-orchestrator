"""Pure rubric-scoring helpers for EV-9 deep-research evaluation.

The LLM judge prompt is inference-gated elsewhere. This module only defines the
deterministic scoring contract: positive/negative rubric aggregation, saturation
screening, and multi-judge ranking stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Mapping, Sequence

from src.bradley_terry import BTResult, bradley_terry_from_scores

Polarity = Literal["positive", "negative"]

MINDDR_PROCESS_DIMENSIONS = (
    "reasoning_trajectory",
    "tool_calls",
    "outline",
    "content_stage",
)

DRACO_CONTENT_DIMENSIONS = (
    "factual_accuracy",
    "breadth_depth",
    "presentation",
    "citation",
)


@dataclass(frozen=True)
class RubricCriterion:
    """One rubric criterion with explicit polarity and weight."""

    name: str
    weight: float = 1.0
    polarity: Polarity = "positive"


@dataclass(frozen=True)
class RubricScore:
    """Aggregated positive/negative rubric score in [0, 1]."""

    score: float
    positive_score: float
    negative_penalty: float
    missing_criteria: tuple[str, ...] = ()


@dataclass(frozen=True)
class JudgeStability:
    """Ranking-stability diagnostics across multiple rubric judges."""

    consensus: BTResult
    judge_rankings: dict[str, list[str]] = field(default_factory=dict)
    mean_spearman: float = 1.0
    top_choice_agreement: bool = True


DEFAULT_RUBRIC_CRITERIA: tuple[RubricCriterion, ...] = (
    *(RubricCriterion(name) for name in MINDDR_PROCESS_DIMENSIONS),
    *(RubricCriterion(name) for name in DRACO_CONTENT_DIMENSIONS),
)


def aggregate_rubric_score(
    scores: Mapping[str, float],
    criteria: Sequence[RubricCriterion] = DEFAULT_RUBRIC_CRITERIA,
) -> RubricScore:
    """Combine positive and negative rubric criteria without symmetry leakage.

    Positive criteria reward evidence. Negative criteria are independent penalty
    axes, following the DRACO recommendation not to mix reward-bearing and
    penalty-bearing rubric text into one symmetric score.
    """

    positive_total = 0.0
    positive_weight = 0.0
    negative_total = 0.0
    negative_weight = 0.0
    missing: list[str] = []
    for criterion in criteria:
        weight = max(0.0, float(criterion.weight))
        if weight == 0.0:
            continue
        value = scores.get(criterion.name)
        if value is None or not isfinite(float(value)):
            missing.append(criterion.name)
            continue
        bounded = min(max(float(value), 0.0), 1.0)
        if criterion.polarity == "negative":
            negative_total += bounded * weight
            negative_weight += weight
        else:
            positive_total += bounded * weight
            positive_weight += weight

    positive_score = positive_total / positive_weight if positive_weight else 1.0
    negative_penalty = negative_total / negative_weight if negative_weight else 0.0
    score = min(max(positive_score - negative_penalty, 0.0), 1.0)
    return RubricScore(
        score=score,
        positive_score=positive_score,
        negative_penalty=negative_penalty,
        missing_criteria=tuple(missing),
    )


def saturated_items(
    item_candidate_scores: Mapping[str, Mapping[str, float]],
    *,
    threshold: float = 0.90,
) -> dict[str, float]:
    """Return items where any candidate score is at or above saturation."""

    saturated: dict[str, float] = {}
    for item_id, scores in item_candidate_scores.items():
        finite_scores = [float(v) for v in scores.values() if isfinite(float(v))]
        if not finite_scores:
            continue
        top = max(finite_scores)
        if top >= threshold:
            saturated[item_id] = top
    return saturated


def judge_ranking_stability(
    judge_candidate_scores: Mapping[str, Mapping[str, float]],
) -> JudgeStability:
    """Fit per-judge and consensus Bradley-Terry rankings from scalar scores."""

    candidate_ids = sorted({
        candidate
        for scores in judge_candidate_scores.values()
        for candidate in scores
        if isfinite(float(scores[candidate]))
    })
    if len(candidate_ids) < 2 or not judge_candidate_scores:
        consensus = bradley_terry_from_scores(candidate_ids, {})
        return JudgeStability(consensus=consensus)

    judge_rankings: dict[str, list[str]] = {}
    consensus_scores: dict[tuple[str, str], float] = {}
    consensus_counts: dict[tuple[str, str], int] = {}
    for judge_id, scores in judge_candidate_scores.items():
        pairwise = _pairwise_scores_from_scalar_scores(candidate_ids, scores)
        if not pairwise:
            continue
        judge_result = bradley_terry_from_scores(candidate_ids, pairwise)
        judge_rankings[judge_id] = [str(item) for item in judge_result.ranking]
        for pair, score in pairwise.items():
            consensus_scores[pair] = consensus_scores.get(pair, 0.0) + score
            consensus_counts[pair] = consensus_counts.get(pair, 0) + 1

    averaged = {
        pair: score / consensus_counts[pair]
        for pair, score in consensus_scores.items()
        if consensus_counts[pair] > 0
    }
    consensus = bradley_terry_from_scores(candidate_ids, averaged)
    rankings = list(judge_rankings.values())
    mean_spearman = _mean_pairwise_spearman(rankings)
    top_choice_agreement = len({ranking[0] for ranking in rankings if ranking}) <= 1
    return JudgeStability(
        consensus=consensus,
        judge_rankings=judge_rankings,
        mean_spearman=mean_spearman,
        top_choice_agreement=top_choice_agreement,
    )


def _pairwise_scores_from_scalar_scores(
    candidate_ids: Sequence[str],
    scores: Mapping[str, float],
) -> dict[tuple[str, str], float]:
    pairwise: dict[tuple[str, str], float] = {}
    for i, left in enumerate(candidate_ids):
        left_score = scores.get(left)
        if left_score is None or not isfinite(float(left_score)):
            continue
        for right in candidate_ids[i + 1:]:
            right_score = scores.get(right)
            if right_score is None or not isfinite(float(right_score)):
                continue
            if float(left_score) > float(right_score):
                pairwise[(left, right)] = 1.0
            elif float(left_score) < float(right_score):
                pairwise[(left, right)] = 0.0
            else:
                pairwise[(left, right)] = 0.5
    return pairwise


def _mean_pairwise_spearman(rankings: Sequence[Sequence[str]]) -> float:
    if len(rankings) < 2:
        return 1.0
    values: list[float] = []
    for i, left in enumerate(rankings):
        for right in rankings[i + 1:]:
            values.append(_spearman(left, right))
    return sum(values) / len(values) if values else 1.0


def _spearman(left: Sequence[str], right: Sequence[str]) -> float:
    shared = [item for item in left if item in set(right)]
    n = len(shared)
    if n < 2:
        return 1.0
    left_rank = {item: rank for rank, item in enumerate(left)}
    right_rank = {item: rank for rank, item in enumerate(right)}
    d2 = sum((left_rank[item] - right_rank[item]) ** 2 for item in shared)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))
