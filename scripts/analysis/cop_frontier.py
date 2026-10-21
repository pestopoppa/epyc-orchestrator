"""ROUTE-COP-GUARD: cost-of-pass frontier with a mandatory reliability floor.

Cost-of-pass (expected spend per solved task = cost_per_attempt / pass_rate) used as the SOLE
routing objective is gamed by a cheap near-random strategy: a free candidate that passes 2%
of the time has a finite, sometimes frontier-leading, cost-of-pass. So no candidate enters
the frontier unless it clears a reliability floor: pass rate >= tau, or pass^k >= tau
(all k independent attempts pass) when the caller states k.

Retries are NOT independent when context carries over (a retry inherits the failed attempt's
transcript, errors and anchoring), so ``pass_pow_k`` here is the optimistic independent
bound; a caller that retries with carried context must measure the k-attempt pass rate
directly and pass it as ``pass_rate`` instead (intake-2262#record).

Pure functions, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class Candidate:
    name: str
    cost_per_attempt: float
    pass_rate: float
    n: int = 0


def cost_of_pass(c: Candidate) -> float:
    return inf if c.pass_rate <= 0 else c.cost_per_attempt / c.pass_rate


def pass_pow_k(pass_rate: float, k: int) -> float:
    """Independent-retry optimistic bound for 'all k attempts pass'."""
    return pass_rate ** k


def meets_floor(c: Candidate, tau: float, k: int = 1) -> bool:
    return pass_pow_k(c.pass_rate, k) >= tau


def frontier(candidates: list[Candidate], *, tau: float, k: int = 1) -> dict:
    """Return {'admitted', 'excluded', 'frontier'}; only floor-clearing candidates can be on
    the cost-of-pass frontier (non-dominated in (cost_of_pass, pass_rate))."""
    if not 0.0 < tau <= 1.0:
        raise ValueError("reliability floor tau must be in (0, 1]; there is no floor-free frontier")
    admitted = [c for c in candidates if meets_floor(c, tau, k)]
    excluded = [c.name for c in candidates if not meets_floor(c, tau, k)]
    front = [a for a in admitted
             if not any(cost_of_pass(b) <= cost_of_pass(a) and b.pass_rate >= a.pass_rate
                        and (cost_of_pass(b) < cost_of_pass(a) or b.pass_rate > a.pass_rate)
                        for b in admitted)]
    return {"admitted": [c.name for c in admitted], "excluded": excluded,
            "frontier": [c.name for c in sorted(front, key=cost_of_pass)], "tau": tau, "k": k,
            "retry_independence": "assumed; invalid when context carries over"}
