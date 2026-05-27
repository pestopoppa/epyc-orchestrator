"""Bradley-Terry pairwise-ranking aggregation.

Shared module consumed by:
  - autopilot P17 (AP-37/38): tiebreak top-K Pareto candidates when
    hypervolume slope falls below the auto-calibrated noise floor.
  - decision-aware-routing DAR-6.4: aggregate N concurrent-serve completions
    on high-injection-risk prompts.
  - swarm-dataset-distillation Phase 3: filter swarm-generated candidate
    outputs by pairwise-judged BT rank.

All three call sites share this one implementation per the cross-handoff
invariant recorded in handoffs/active/{autopilot-continuous-optimization,
swarm-dataset-distillation,decision-aware-routing,peer-verifier-speculation-spike}.md.

Background
----------
The Bradley-Terry model assigns each item i a latent "skill" theta_i such that
the probability that i beats j in a pairwise comparison is

    P(i beats j) = exp(theta_i) / (exp(theta_i) + exp(theta_j))
                 = pi_i / (pi_i + pi_j)        with pi_i = exp(theta_i)

Given a matrix W where W[i][j] = number of times i beat j (or a continuous
weight in [0, count]), the MLE for pi can be obtained by Zermelo's iteration
(Hunter 2004; Zermelo 1929):

    pi_i_new = W_i  /  sum_{j != i} ( (W_ij + W_ji) / (pi_i + pi_j) )

where W_i = sum_j W_ij is i's total wins.

We use Zermelo iteration because it is simple, converges monotonically, and
has no external dependencies. For very small N (typical autopilot tiebreaks
top-K with K<=8) it converges in <100 iterations to 1e-8 tolerance.

Diagnostics returned with the ranking:

  - `comparison_graph_connected`: False means the comparison graph has
    disconnected components — items in different components have no
    transitive evidence linking them and their relative scores are
    artefacts of the regularization prior. Treat as a warning.

  - `condorcet_cycles`: triples (a, b, c) where a beats b, b beats c, and
    c beats a with all three pairwise win-fractions > 0.5. BT can fit
    such data but the assumed transitive skill model is violated; the
    ranking should not be taken as decisive.

  - `dominance_skew`: max absolute log-skill difference between adjacent
    ranked items. A value > 3 (about 20x odds) usually means one item
    sweeps the field — capability-skew failure mode of intake-615.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log
from typing import Hashable, Iterable, Sequence


__all__ = [
    "BTResult",
    "bradley_terry_rank",
    "bradley_terry_from_pairs",
    "bradley_terry_from_scores",
]


# Sentinel: smallest positive prior added to every pair's W to keep the
# comparison graph "weakly connected" under regularization. Picked small
# enough that real comparisons dominate (typical W_ij >= 1) but large
# enough to avoid div-by-zero. Documented at intake-615 capability-skew
# failure mode.
_REGULARIZATION_PRIOR = 1e-3


@dataclass
class BTResult:
    """Bradley-Terry fit result.

    Attributes
    ----------
    ranking:
        Items ordered from highest to lowest fitted skill.
    log_skills:
        Dict mapping item -> log(pi_i). Anchored so the lowest-skill item
        is 0.0; differences are odds ratios on log scale.
    iterations:
        Zermelo iterations used to reach tolerance.
    converged:
        True iff the iteration converged within max_iterations.
    comparison_graph_connected:
        False if the directed win-graph has more than one weakly-connected
        component. Disconnected components mean some pairs are entirely
        un-compared; their relative skills come from the regularization prior.
    condorcet_cycles:
        List of triples (a, b, c) forming a Condorcet cycle in the empirical
        win fractions. Empty for fully transitive data.
    dominance_skew:
        Maximum absolute difference in log_skills between adjacent ranked
        items. >3 means one item dominates the field (capability-skew flag).
    """

    ranking: list[Hashable]
    log_skills: dict[Hashable, float]
    iterations: int
    converged: bool
    comparison_graph_connected: bool
    condorcet_cycles: list[tuple[Hashable, Hashable, Hashable]] = field(default_factory=list)
    dominance_skew: float = 0.0

    @property
    def warnings(self) -> list[str]:
        msgs: list[str] = []
        if not self.converged:
            msgs.append(f"Zermelo iteration did not converge in {self.iterations} steps")
        if not self.comparison_graph_connected:
            msgs.append("comparison graph has disconnected components — some scores reflect only the regularization prior")
        if self.condorcet_cycles:
            msgs.append(f"{len(self.condorcet_cycles)} Condorcet cycle(s) detected — transitive-skill assumption violated")
        if self.dominance_skew > 3.0:
            msgs.append(f"dominance skew {self.dominance_skew:.2f} (>3 log-odds) — one item sweeps the field; ranking may be capability-skewed (intake-615)")
        return msgs


def bradley_terry_rank(
    items: Sequence[Hashable],
    win_matrix: dict[tuple[Hashable, Hashable], float],
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> BTResult:
    """Fit Bradley-Terry on a sparse win matrix.

    Parameters
    ----------
    items:
        Sequence of item identifiers (any hashable). Order is not significant.
    win_matrix:
        Mapping (i, j) -> w where w is the fractional or integer count of
        times i beat j. Missing pairs are treated as zero observations.
        Diagonal entries (i, i) are ignored.
    max_iterations:
        Hard cap on Zermelo iterations.
    tolerance:
        Convergence threshold on max absolute change in pi per iteration.

    Returns
    -------
    BTResult with ranking, log_skills, and diagnostics.

    Notes
    -----
    A small regularization prior (1e-3) is added to every off-diagonal pair
    so the iteration is defined even when the comparison graph has
    disconnected components. Connectivity is reported as a diagnostic.
    """
    if len(items) < 2:
        # Trivial cases: empty or singleton input.
        items_list = list(items)
        return BTResult(
            ranking=items_list,
            log_skills={i: 0.0 for i in items_list},
            iterations=0,
            converged=True,
            comparison_graph_connected=True,
        )

    items_list = list(items)
    n = len(items_list)
    idx = {item: k for k, item in enumerate(items_list)}

    # Build dense W with regularization. Skip diagonal.
    W = [[0.0] * n for _ in range(n)]
    raw_W = [[0.0] * n for _ in range(n)]
    for (i, j), w in win_matrix.items():
        if i not in idx or j not in idx or i == j:
            continue
        if not isfinite(w) or w < 0:
            continue
        raw_W[idx[i]][idx[j]] += w
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            W[a][b] = raw_W[a][b] + _REGULARIZATION_PRIOR

    # Connectivity check uses RAW wins (un-regularized) so the prior cannot
    # mask a genuinely disconnected graph.
    connected = _weakly_connected(raw_W)
    cycle_indices = _condorcet_cycles(raw_W)
    cycles = [
        (items_list[a], items_list[b], items_list[c])
        for a, b, c in cycle_indices
    ]

    # Zermelo iteration on pi. Two convergence criteria, either declares
    # success:
    #   (1) Tight numerical convergence: max delta on anchored log-skills
    #       below `tolerance`. This is the well-conditioned case.
    #   (2) Ranking-stability convergence: when data is perfectly
    #       separable the MLE is at pi=infinity and absolute deltas never
    #       reach `tolerance`. Detect this by holding the ranking stable
    #       for `ranking_stability_required` consecutive iterations after
    #       a minimum warm-up; the ranking is the operative output for
    #       all three consumer call sites.
    pi = [1.0] * n
    prev_anchored_logs = [0.0] * n
    prev_ranking_key: tuple[int, ...] | None = None
    stable_ranking_count = 0
    ranking_stability_required = 5
    ranking_stability_warmup = 20
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        new_pi = [0.0] * n
        for a in range(n):
            wins_a = sum(W[a][b] for b in range(n) if b != a)
            denom = 0.0
            for b in range(n):
                if b == a:
                    continue
                pair_count = W[a][b] + W[b][a]
                denom += pair_count / (pi[a] + pi[b])
            new_pi[a] = wins_a / denom if denom > 0 else pi[a]
        # Normalize to keep numbers bounded (BT is scale-invariant in pi).
        total = sum(new_pi)
        if total > 0:
            new_pi = [v * n / total for v in new_pi]
        # Anchored log-skills for convergence test.
        anchored_logs = [log(max(v, 1e-300)) for v in new_pi]
        floor_l = min(anchored_logs)
        anchored_logs = [v - floor_l for v in anchored_logs]
        max_delta = max(
            abs(anchored_logs[a] - prev_anchored_logs[a]) for a in range(n)
        )
        # Ranking-stability check.
        current_ranking_key = tuple(
            sorted(range(n), key=lambda a: anchored_logs[a], reverse=True)
        )
        if current_ranking_key == prev_ranking_key:
            stable_ranking_count += 1
        else:
            stable_ranking_count = 0
        prev_ranking_key = current_ranking_key
        prev_anchored_logs = anchored_logs
        pi = new_pi
        # Criterion 1: tight numerical convergence.
        if max_delta < tolerance:
            converged = True
            break
        # Criterion 2: ranking has stabilized after warm-up — MLE may be
        # at infinity but the ordering is settled, which is what callers
        # consume.
        if (
            iterations >= ranking_stability_warmup
            and stable_ranking_count >= ranking_stability_required
        ):
            converged = True
            break

    # Convert pi -> log-skill, anchored so min = 0.
    log_pi = [log(max(p, 1e-300)) for p in pi]
    floor = min(log_pi)
    log_skills = {items_list[k]: log_pi[k] - floor for k in range(n)}

    # Rank high -> low.
    ranking = sorted(items_list, key=lambda x: log_skills[x], reverse=True)

    # Dominance skew = largest adjacent gap.
    skew = 0.0
    for a, b in zip(ranking, ranking[1:]):
        skew = max(skew, log_skills[a] - log_skills[b])

    return BTResult(
        ranking=ranking,
        log_skills=log_skills,
        iterations=iterations,
        converged=converged,
        comparison_graph_connected=connected,
        condorcet_cycles=cycles,
        dominance_skew=skew,
    )


def bradley_terry_from_pairs(
    items: Sequence[Hashable],
    pairs: Iterable[tuple[Hashable, Hashable]],
    **kwargs,
) -> BTResult:
    """Convenience wrapper: BT fit from a list of (winner, loser) pairs.

    Each pair contributes exactly 1.0 to the win matrix.
    """
    W: dict[tuple[Hashable, Hashable], float] = {}
    for winner, loser in pairs:
        W[(winner, loser)] = W.get((winner, loser), 0.0) + 1.0
    return bradley_terry_rank(items, W, **kwargs)


def bradley_terry_from_scores(
    items: Sequence[Hashable],
    pairwise_scores: dict[tuple[Hashable, Hashable], float],
    **kwargs,
) -> BTResult:
    """Convenience wrapper: BT fit from continuous pairwise scores in [0, 1].

    pairwise_scores[(i, j)] = p where p is the judge's reported probability
    that i beats j. Symmetry is NOT assumed — if both (i, j) and (j, i) are
    given, each contributes independently (i.e., the judge ranked both
    directions). If only (i, j) is given, (j, i) is inferred as 1 - p.

    Each score contributes that fractional win to the matrix.
    """
    W: dict[tuple[Hashable, Hashable], float] = {}
    for (i, j), p in pairwise_scores.items():
        if not isfinite(p):
            continue
        p = max(0.0, min(1.0, p))
        W[(i, j)] = W.get((i, j), 0.0) + p
        reverse = (j, i)
        if reverse not in pairwise_scores:
            W[reverse] = W.get(reverse, 0.0) + (1.0 - p)
    return bradley_terry_rank(items, W, **kwargs)


# ── internal helpers ──────────────────────────────────────────────


def _weakly_connected(raw_W: list[list[float]]) -> bool:
    """True iff every item is reachable from item 0 ignoring edge direction."""
    n = len(raw_W)
    if n == 0:
        return True
    seen = {0}
    stack = [0]
    while stack:
        a = stack.pop()
        for b in range(n):
            if b in seen:
                continue
            if raw_W[a][b] > 0 or raw_W[b][a] > 0:
                seen.add(b)
                stack.append(b)
    return len(seen) == n


def _condorcet_cycles(
    raw_W: list[list[float]],
) -> list[tuple[int, int, int]]:
    """Enumerate (a, b, c) triples forming a Condorcet cycle on the raw wins.

    For each triple, a beats b, b beats c, c beats a in the empirical
    win-fraction sense (W_ab / (W_ab + W_ba) > 0.5 for each pair). Pairs
    with no comparisons are skipped.
    """
    n = len(raw_W)
    cycles: list[tuple[int, int, int]] = []

    def beats(a: int, b: int) -> bool:
        total = raw_W[a][b] + raw_W[b][a]
        if total == 0:
            return False
        return raw_W[a][b] / total > 0.5

    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                if beats(a, b) and beats(b, c) and beats(c, a):
                    cycles.append((a, b, c))
    return cycles
