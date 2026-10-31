"""EV-15e/f offline text-rating instrument; never calls or selects a model.

Clean-room equations: arXiv:2603.04304v1, section 4 Eq. (1), Appendix D
Algorithm 3 (CC BY 4.0). A reported winner, independently of rating magnitude,
determines the vote: A=1, tie=0.5, B=0. Confidence is max(rating gap/9, 0.1).
EV-15e additionally requires both presentation orders of every candidate pair.
No scheduler, selector deployment, acceptance threshold or historical rescore.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from math import fsum, isfinite
from pathlib import Path
from typing import Literal, Sequence


@dataclass(frozen=True)
class TextComparison:
    """One parsed judgment, in its actual displayed order; ratings are 1..10."""

    left: str
    right: str
    rating_left: float
    rating_right: float
    winner: Literal["A", "B", "tie"]


def comparison_plan(candidate_ids: Sequence[str]) -> list[tuple[str, str]]:
    """Exhaustive paired-order plan; swapping only a prompt instruction is insufficient."""
    ids = list(candidate_ids)
    if len(ids) < 2 or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("at least two nonempty candidate IDs required")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    return [order for a, b in combinations(ids, 2) for order in ((a, b), (b, a))]


def aggregate_text_comparisons(
    candidate_ids: Sequence[str], comparisons: Sequence[TextComparison]
) -> dict:
    """Refuse incomplete/duplicate orders before computing any candidate win rate.

    Both orders contribute their own weight and identity-mapped vote. Pair-level
    numerator/denominator records are combined before candidate aggregation;
    neither order may be silently dropped or replaced by an inferred complement.
    """
    plan = comparison_plan(candidate_ids)
    expected = set(plan)
    rows: dict[tuple[str, str], TextComparison] = {}
    for row in comparisons:
        key = (row.left, row.right)
        if key not in expected:
            raise ValueError(f"unexpected candidate/order: {key!r}")
        if key in rows:
            raise ValueError(f"duplicate presentation order: {key!r}")
        for rating in (row.rating_left, row.rating_right):
            if (
                isinstance(rating, bool)
                or not isinstance(rating, (int, float))
                or not isfinite(rating)
                or not 1 <= rating <= 10
            ):
                raise ValueError("ratings must be finite numbers in [1,10]")
        if row.winner not in ("A", "B", "tie"):
            raise ValueError("winner must be A, B, or tie")
        rows[key] = row
    missing = expected - rows.keys()
    if missing:
        raise ValueError(f"both presentation orders required; missing {sorted(missing)!r}")
    numerators: dict[str, list[float]] = {i: [] for i in candidate_ids}
    denominators: dict[str, list[float]] = {i: [] for i in candidate_ids}
    pairs = []
    for a, b in combinations(candidate_ids, 2):
        weights, votes, observations = [], [], []
        for key in ((a, b), (b, a)):
            row = rows[key]
            weight = max(abs(row.rating_left - row.rating_right) / 9, 0.1)
            left_vote = {"A": 1.0, "B": 0.0, "tie": 0.5}[row.winner]
            vote_a = left_vote if row.left == a else 1 - left_vote
            weights.append(weight)
            votes.append(weight * vote_a)
            observations.append({**asdict(row), "weight": weight, "vote_for_first_id": vote_a})
        denominator = fsum(weights)
        numerator = fsum(votes)
        numerators[a].append(numerator)
        numerators[b].append(denominator - numerator)
        denominators[a].append(denominator)
        denominators[b].append(denominator)
        pairs.append(
            {
                "candidate_ids": [a, b],
                "observations": observations,
                "first_numerator": numerator,
                "denominator": denominator,
            }
        )
    return {
        "instrument": "ev15_text_rating_paired_orders.v1",
        "score_direction": "higher_better",
        "weight_floor": 0.1,
        "candidate_scores": {i: fsum(numerators[i]) / fsum(denominators[i]) for i in candidate_ids},
        "judgment_count": len(rows),
        "pairs": pairs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.input.read_text())
        result = aggregate_text_comparisons(
            data["candidate_ids"], [TextComparison(**r) for r in data["comparisons"]]
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    with args.output.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
