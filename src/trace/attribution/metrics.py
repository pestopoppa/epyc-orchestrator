"""Ranking and accuracy metrics against a human consensus ranking.

Gold for one trace is a relevance map ``{step_index: consensus_rate}`` (HIL-12 item 6: the fraction
of annotators who flagged the step; steps nobody flagged are absent = 0). A single-label set such as
Who&When Pro is the special case ``{gold_step: 1.0}``.

nDCG@K follows MP-Bench (intake-2332#record): DCG = sum_i gain(rel_i) / log2(i + 1), i 1-based,
gain ``exp`` = 2^rel - 1 or ``linear`` = rel, normalised by the ideal ordering of the gold map.
A predicted ranking is a score map ``{step: score}`` (vote share); unscored steps are unranked.
Ties: ``tie="expected"`` (default) charges a tied block the EXPECTED DCG over all orderings of the
block (exact, by linearity: every position in the block gets the block's mean gain), so no tie-break
rule can flatter a method; ``tie="index"`` breaks ties by ascending step index.
The random baseline is a uniformly random permutation of all attributable steps; its expected
nDCG@K is exact (``random_ndcg_expected``), with a seeded Monte Carlo twin for cross-checking.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def gain(rel: float, kind: str = "exp") -> float:
    if kind == "exp":
        return 2.0**rel - 1.0
    if kind == "linear":
        return float(rel)
    raise ValueError(f"unknown gain {kind!r}")


def _disc(pos: int) -> float:  # pos is 1-based
    return 1.0 / math.log2(pos + 1)


def ideal_dcg(relevance: dict[int, float], k: int | None, kind: str = "exp") -> float:
    rels = sorted((r for r in relevance.values() if r > 0), reverse=True)
    if k is not None:
        rels = rels[:k]
    return sum(gain(r, kind) * _disc(i + 1) for i, r in enumerate(rels))


def ordered_blocks(scores: dict[int, float], tie: str = "expected") -> list[list[int]]:
    """Score map -> ranked blocks of tied steps (descending score). ``tie="index"`` = singletons."""
    if tie not in ("expected", "index"):
        raise ValueError(f"unknown tie rule {tie!r}")
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if tie == "index":
        return [[s] for s, _ in items]
    blocks: list[list[int]] = []
    last = None
    for s, v in items:
        if blocks and v == last:
            blocks[-1].append(s)
        else:
            blocks.append([s])
        last = v
    return blocks


def dcg(
    scores: dict[int, float],
    relevance: dict[int, float],
    k: int | None,
    kind: str = "exp",
    tie: str = "expected",
) -> float:
    total, pos = 0.0, 0
    for block in ordered_blocks(scores, tie):
        mean_gain = sum(gain(relevance.get(s, 0.0), kind) for s in block) / len(block)
        for _ in block:
            pos += 1
            if k is not None and pos > k:
                return total
            total += mean_gain * _disc(pos)
    return total


def ndcg(
    scores: dict[int, float],
    relevance: dict[int, float],
    k: int | None = 5,
    kind: str = "exp",
    tie: str = "expected",
) -> float | None:
    """None when the gold map has no positive relevance (undefined; excluded from means)."""
    idcg = ideal_dcg(relevance, k, kind)
    if idcg <= 0:
        return None
    return dcg(scores, relevance, k, kind, tie) / idcg


def random_ndcg_expected(
    steps: Sequence[int], relevance: dict[int, float], k: int | None = 5, kind: str = "exp"
) -> float | None:
    """Exact E[nDCG@K] of a uniformly random permutation of ``steps`` (every position holds a uniformly
    random step, so its expected gain is the mean gain over ``steps``)."""
    idcg = ideal_dcg(relevance, k, kind)
    if idcg <= 0 or not steps:
        return None
    mean_gain = sum(gain(relevance.get(s, 0.0), kind) for s in steps) / len(steps)
    n = len(steps) if k is None else min(k, len(steps))
    return mean_gain * sum(_disc(i + 1) for i in range(n)) / idcg


def random_ndcg_mc(
    steps: Sequence[int],
    relevance: dict[int, float],
    k: int | None = 5,
    kind: str = "exp",
    draws: int = 2000,
    seed: int = 0,
) -> float | None:
    if ideal_dcg(relevance, k, kind) <= 0 or not steps:
        return None
    rng = random.Random(seed)
    acc = 0.0
    st = list(steps)
    for _ in range(draws):
        rng.shuffle(st)
        n = len(st)
        acc += ndcg({s: float(n - i) for i, s in enumerate(st)}, relevance, k, kind, tie="index")  # type: ignore[operator]
    return acc / draws


# ------------------------------------------------------------------------------ single-pick accuracy
def top_steps(relevance: dict[int, float]) -> set[int]:
    pos = {s: r for s, r in relevance.items() if r > 0}
    if not pos:
        return set()
    m = max(pos.values())
    return {s for s, r in pos.items() if r == m}


def pick_accuracy(pred: int | None, relevance: dict[int, float]) -> dict:
    """One predicted step vs the consensus map. ``exact`` = a top-consensus step; ``within1`` = within
    one step index of a top-consensus step; ``hit`` = any step with consensus > 0. Abstain = miss."""
    tops = top_steps(relevance)
    if pred is None:
        return {"abstain": True, "exact": False, "within1": False, "hit": False}
    return {
        "abstain": False,
        "exact": pred in tops,
        "within1": any(abs(pred - t) <= 1 for t in tops),
        "hit": relevance.get(pred, 0.0) > 0,
    }


def mean(xs: Iterable[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None
