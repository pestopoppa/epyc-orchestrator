"""Pure rubric-scoring helpers for EV-9 deep-research evaluation.

The LLM judge prompt is inference-gated elsewhere. This module only defines the
deterministic scoring contract: positive/negative rubric aggregation, saturation
screening, and multi-judge ranking stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
import re
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


@dataclass(frozen=True)
class RubricPrompt:
    """Structured judge prompt payload for EV-9 rubric scoring."""

    prompt: str
    criteria: tuple[RubricCriterion, ...]


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


def build_rubric_judge_prompt(
    *,
    task_prompt: str,
    answer: str,
    expected_contains: Sequence[str] = (),
    criteria: Sequence[RubricCriterion] = DEFAULT_RUBRIC_CRITERIA,
) -> RubricPrompt:
    """Build the model-judge prompt without invoking a model."""

    criteria_tuple = tuple(criteria)
    criteria_lines = "\n".join(
        f"- {criterion.name} ({criterion.polarity}, weight={criterion.weight:g})"
        for criterion in criteria_tuple
    )
    expected_lines = "\n".join(f"- {item}" for item in expected_contains) or "- none"
    prompt = (
        "Score the research answer against the rubric. Return only JSON with a "
        "`scores` object mapping each criterion name to a number in [0, 1], "
        "plus `rationale` with one short sentence per criterion.\n\n"
        "Criteria:\n"
        f"{criteria_lines}\n\n"
        "Expected structural hints:\n"
        f"{expected_lines}\n\n"
        "Task prompt:\n"
        f"{task_prompt.strip()}\n\n"
        "Answer:\n"
        f"{answer.strip()}\n"
    )
    return RubricPrompt(prompt=prompt, criteria=criteria_tuple)


def deterministic_rubric_fallback(
    answer: str,
    *,
    expected_contains: Sequence[str] = (),
    tool_events: Sequence[object] = (),
) -> dict[str, float]:
    """Cheap T1 fallback scores from structure and expected-hint coverage."""

    text = answer or ""
    normalized = text.lower()
    expected_coverage = _expected_hint_coverage(normalized, expected_contains)
    outline_score = _outline_score(text)
    citation_score = _citation_score(text)
    tool_score = _tool_score(text, tool_events)
    reasoning_score = _reasoning_score(normalized)
    breadth_score = min(1.0, (expected_coverage * 0.7) + (outline_score * 0.3))
    presentation_score = min(1.0, (outline_score * 0.7) + (_length_score(text) * 0.3))
    content_stage = min(1.0, (expected_coverage * 0.8) + (citation_score * 0.2))
    return {
        "reasoning_trajectory": reasoning_score,
        "tool_calls": tool_score,
        "outline": outline_score,
        "content_stage": content_stage,
        "factual_accuracy": expected_coverage,
        "breadth_depth": breadth_score,
        "presentation": presentation_score,
        "citation": citation_score,
    }


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


def _expected_hint_coverage(text: str, expected_contains: Sequence[str]) -> float:
    if not expected_contains:
        return 0.0
    covered = 0
    for hint in expected_contains:
        tokens = _content_tokens(hint)
        if not tokens:
            continue
        token_hits = sum(1 for token in tokens if token in text)
        if token_hits / len(tokens) >= 0.5:
            covered += 1
    return covered / len(expected_contains)


def _content_tokens(text: str) -> list[str]:
    stop = {
        "and", "are", "for", "from", "must", "per", "the", "with",
        "what", "which", "that", "this", "into", "each", "etc",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9.+-]{2,}", text.lower())
        if token not in stop
    ]


def _outline_score(text: str) -> float:
    heading_count = len(re.findall(r"(?m)^\s{0,3}#{1,4}\s+\S", text))
    bullet_count = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S", text))
    table_count = len(re.findall(r"(?m)^\s*\|.+\|\s*$", text))
    return min(1.0, (heading_count * 0.25) + (bullet_count * 0.12) + (table_count * 0.2))


def _citation_score(text: str) -> float:
    urls = len(re.findall(r"https?://", text))
    citations = len(re.findall(r"\[[^\]]+\]\([^)]+\)|\b(?:arxiv|doi)\s*[:/]", text, re.I))
    named_years = len(re.findall(r"\b[A-Z][A-Za-z0-9.-]+\s+\(?20\d{2}\)?", text))
    return min(1.0, (urls * 0.25) + (citations * 0.25) + (named_years * 0.12))


def _tool_score(text: str, tool_events: Sequence[object]) -> float:
    if tool_events:
        return min(1.0, len(tool_events) / 3.0)
    tool_markers = len(re.findall(r"\b(?:searched|queried|retrieved|source|citation)\b", text, re.I))
    return min(1.0, tool_markers / 4.0)


def _reasoning_score(text: str) -> float:
    markers = len(re.findall(
        r"\b(?:because|therefore|however|compare|evidence|limitation|caveat|tradeoff|first|second|finally)\b",
        text,
    ))
    return min(1.0, markers / 8.0)


def _length_score(text: str) -> float:
    words = len(re.findall(r"\S+", text))
    return min(1.0, words / 250.0)


def _mean_pairwise_spearman(rankings: Sequence[Sequence[str]]) -> float:
    if len(rankings) < 2:
        return 1.0
    values: list[float] = []
    for i, left in enumerate(rankings):
        for right in rankings[i + 1:]:
            values.append(_spearman(left, right))
    return sum(values) / len(values) if values else 1.0


def _spearman(left: Sequence[str], right: Sequence[str]) -> float:
    right_set = set(right)
    shared = [item for item in left if item in right_set]
    n = len(shared)
    if n < 2:
        return 1.0
    left_rank = {item: rank for rank, item in enumerate(left)}
    right_rank = {item: rank for rank, item in enumerate(right)}
    d2 = sum((left_rank[item] - right_rank[item]) ** 2 for item in shared)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))
