"""Anytime-valid sequential verdict primitives for AutoPilot candidates.

The module is intentionally pure. W4 wiring can fold journal rows into these
states without changing the safety gate in the same patch that introduces the
math.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any


STATE_ACCUMULATING = "accumulating"
STATE_CONFIRMED = "confirmed"
STATE_REFUTED = "refuted"


@dataclass(frozen=True)
class SequentialPolicy:
    """Default seq-v1 e-process policy from the Fable 5 01c spec."""

    version: str = "seq-v1"
    alpha: float = 0.05
    confirm_e: float = 20.0
    futility_e: float = 0.05
    budget: int = 8
    budget_min_e: float = 2.0
    first_lambda: float = 0.1
    lambda_cap: float = 0.5
    rate_noninferiority_margin: float = 0.05


DEFAULT_POLICY = SequentialPolicy()


@dataclass(frozen=True)
class TrialStatistic:
    """A single trial-level observation used for one e-process update."""

    s: float
    r_eff: int
    z: float
    qids: tuple[str, ...]


@dataclass(frozen=True)
class EProcessUpdate:
    """Audit record for one wealth update."""

    k: int
    z: float
    lambda_t: float
    factor: float
    wealth: float
    state: str


@dataclass(frozen=True)
class CandidateSequentialView:
    """Rebuildable per-candidate view derived by folding trial observations."""

    fingerprint: str
    core_id: str
    trials: tuple[int, ...]
    quality_state: EProcessState
    quality_updates: tuple[EProcessUpdate, ...]
    state: str
    policy_version: str
    expected_axis: str = "quality"


@dataclass(frozen=True)
class EProcessState:
    """Foldable e-process state for one candidate and one evidence axis."""

    wealth: float = 1.0
    k: int = 0
    sum_z: float = 0.0
    sum_z2: float = 0.0
    wealth_history: tuple[tuple[int | None, float], ...] = ()

    @property
    def mean_z(self) -> float:
        return self.sum_z / self.k if self.k else 0.0

    @property
    def var_z(self) -> float:
        if not self.k:
            return 0.0
        return max(0.0, self.sum_z2 / self.k - self.mean_z * self.mean_z)

    def next_lambda(self, policy: SequentialPolicy = DEFAULT_POLICY) -> float:
        """Predictable capped-Kelly bet using only past observations."""
        if self.k == 0:
            return _clip(policy.first_lambda, 0.0, policy.lambda_cap)
        mu = self.mean_z
        denom = self.var_z + mu * mu
        if denom <= 0.0:
            return 0.0
        return _clip(mu / denom, 0.0, policy.lambda_cap)

    def state_name(self, policy: SequentialPolicy = DEFAULT_POLICY) -> str:
        if self.wealth >= policy.confirm_e:
            return STATE_CONFIRMED
        if self.wealth <= policy.futility_e:
            return STATE_REFUTED
        if self.k >= policy.budget and self.wealth < policy.budget_min_e:
            return STATE_REFUTED
        return STATE_ACCUMULATING

    def update(
        self,
        z: float,
        *,
        policy: SequentialPolicy = DEFAULT_POLICY,
        trial_id: int | None = None,
    ) -> tuple["EProcessState", EProcessUpdate]:
        """Return the next e-process state and an audit update row."""
        z = float(z)
        lambda_t = self.next_lambda(policy)
        factor = 1.0 + lambda_t * z
        if factor < 0.0:
            raise ValueError(
                f"e-process update must stay nonnegative; lambda={lambda_t}, z={z}"
            )
        wealth = self.wealth * factor
        next_state = replace(
            self,
            wealth=wealth,
            k=self.k + 1,
            sum_z=self.sum_z + z,
            sum_z2=self.sum_z2 + z * z,
            wealth_history=self.wealth_history + ((trial_id, wealth),),
        )
        update = EProcessUpdate(
            k=next_state.k,
            z=z,
            lambda_t=lambda_t,
            factor=factor,
            wealth=wealth,
            state=next_state.state_name(policy),
        )
        return next_state, update


def baseline_profile_from_trials(
    trials: Iterable[Mapping[str, bool | int | float]],
) -> dict[str, float]:
    """Mean correctness per qid from baseline trial outcome maps."""
    values: dict[str, list[float]] = defaultdict(list)
    for trial in trials:
        for qid, correct in trial.items():
            values[str(qid)].append(_clip(float(correct), 0.0, 1.0))
    return {
        qid: sum(outcomes) / len(outcomes)
        for qid, outcomes in values.items()
        if outcomes
    }


def quality_trial_statistic(
    question_results: Mapping[str, bool | int | float] | Sequence[Mapping[str, Any]],
    baseline_profile: Mapping[str, float],
) -> TrialStatistic:
    """Compute the trial-level centered quality statistic from qid outcomes."""
    outcomes = _coerce_question_results(question_results)
    s = 0.0
    qids: list[str] = []
    for qid, observed in outcomes.items():
        if qid not in baseline_profile:
            continue
        p_base = _clip(float(baseline_profile[qid]), 0.0, 1.0)
        x = float(bool(observed))
        delta = x - p_base
        if 0.0 < p_base < 1.0 or x != round(p_base):
            qids.append(qid)
            s += delta

    r_eff = len(qids)
    z = s / r_eff if r_eff else 0.0
    return TrialStatistic(s=s, r_eff=r_eff, z=z, qids=tuple(sorted(qids)))


def rate_noninferiority_z(
    task_rate: float,
    baseline_task_rate: float,
    *,
    margin: float = DEFAULT_POLICY.rate_noninferiority_margin,
) -> float:
    """Task-rate non-inferiority statistic from 01c §3.

    H0 is ``E[y] <= -margin`` where ``y`` is relative task-rate lift. The
    returned value has nonpositive expectation under that null.
    """
    if baseline_task_rate <= 0.0:
        raise ValueError("baseline_task_rate must be positive")
    y = _clip((task_rate - baseline_task_rate) / baseline_task_rate, -0.5, 0.5)
    return (y + margin) / 0.5


def journal_seq_block(
    *,
    candidate: str,
    core_id: str,
    quality_update: EProcessUpdate,
    quality_state: EProcessState,
    policy: SequentialPolicy = DEFAULT_POLICY,
    rate_noninf_update: EProcessUpdate | None = None,
) -> dict[str, Any]:
    """Small JSON-serializable ``seq`` block for an evaluated trial row."""
    block: dict[str, Any] = {
        "candidate": candidate,
        "core_id": core_id,
        "k": quality_update.k,
        "z": round(quality_update.z, 6),
        "lambda": round(quality_update.lambda_t, 6),
        "E_quality": round(quality_state.wealth, 6),
        "state": quality_update.state,
        "policy_version": policy.version,
    }
    if rate_noninf_update is not None:
        block["E_rate_noninf"] = round(rate_noninf_update.wealth, 6)
    return block


def rebuild_candidate_view(
    *,
    candidate: str,
    core_id: str,
    observations: Iterable[Mapping[str, Any] | tuple[int | None, float]],
    policy: SequentialPolicy = DEFAULT_POLICY,
    expected_axis: str = "quality",
) -> CandidateSequentialView:
    """Fold journal rows or ``(trial_id, z)`` pairs into a candidate view."""
    state = EProcessState()
    updates: list[EProcessUpdate] = []
    trials: list[int] = []
    for observation in observations:
        parsed = _coerce_seq_observation(observation, candidate=candidate, core_id=core_id)
        if parsed is None:
            continue
        trial_id, z = parsed
        state, update = state.update(z, policy=policy, trial_id=trial_id)
        updates.append(update)
        if trial_id is not None:
            trials.append(trial_id)
    return CandidateSequentialView(
        fingerprint=candidate,
        core_id=core_id,
        trials=tuple(trials),
        quality_state=state,
        quality_updates=tuple(updates),
        state=state.state_name(policy),
        policy_version=policy.version,
        expected_axis=expected_axis,
    )


def empirical_ville_false_positive_rate(
    *,
    runs: int = 100_000,
    horizon: int = 12,
    policy: SequentialPolicy = DEFAULT_POLICY,
    seed: int = 20260612,
) -> float:
    """Simulate a mean-zero Rademacher null and return ``P(max E >= threshold)``."""
    rng = random.Random(seed)
    hits = 0
    for _ in range(runs):
        state = EProcessState()
        hit = False
        for _trial_idx in range(horizon):
            z = 1.0 if rng.random() < 0.5 else -1.0
            state, update = state.update(z, policy=policy)
            if update.state == STATE_CONFIRMED:
                hit = True
                break
        hits += int(hit)
    return hits / runs


def _coerce_question_results(
    question_results: Mapping[str, bool | int | float] | Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    if isinstance(question_results, Mapping):
        return {str(qid): bool(correct) for qid, correct in question_results.items()}
    outcomes: dict[str, bool] = {}
    for item in question_results:
        qid = str(item.get("qid") or item.get("question_id") or "").strip()
        if not qid:
            continue
        outcomes[qid] = bool(item.get("correct"))
    return outcomes


def _coerce_seq_observation(
    observation: Mapping[str, Any] | tuple[int | None, float],
    *,
    candidate: str,
    core_id: str,
) -> tuple[int | None, float] | None:
    if isinstance(observation, tuple):
        trial_id, z = observation
        return trial_id, float(z)
    seq = observation.get("seq")
    block = seq if isinstance(seq, Mapping) else observation
    block_candidate = block.get("candidate")
    if block_candidate is not None and str(block_candidate) != candidate:
        return None
    block_core = block.get("core_id")
    if block_core is not None and str(block_core) != core_id:
        return None
    if "z" not in block:
        return None
    raw_trial_id = observation.get("trial_id", block.get("trial_id"))
    trial_id = int(raw_trial_id) if raw_trial_id is not None else None
    return trial_id, float(block["z"])


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
