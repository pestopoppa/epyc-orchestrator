"""Anytime-valid sequential verdict primitives for AutoPilot candidates.

The module is intentionally pure. W4 wiring can fold journal rows into these
states without changing the safety gate in the same patch that introduces the
math.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any


log = logging.getLogger(__name__)

STATE_ACCUMULATING = "accumulating"
STATE_CONFIRMED = "confirmed"
STATE_REFUTED = "refuted"

# SEQ-3a: validity floor for a per-trial e-process statistic ``z``.
#
# The wealth process is a nonnegative supermartingale (the property Ville's
# inequality — hence the anytime-valid type-I guarantee — rests on) only while every
# applied factor ``1 + lambda_t * z`` stays >= 0. With ``lambda_t`` capped at
# ``policy.lambda_cap``, that is threatened ONLY on the negative side: it requires
# ``z >= -1/lambda_cap``. This is EXACTLY the condition under which
# ``EProcessState.update`` raises ValueError. A journal-derived z below that floor
# (an old-schema / mis-scaled / corrupt value) would crash the rebuild.
#
# We deliberately impose NO upper bound: a large-positive z keeps the factor >= 0 and
# is legitimate strong evidence — it does not threaten the nonnegative-supermartingale
# validity guarantee, and the statistics that produce z are already bounded by
# construction (quality in [-1, 1]; rate in ~[-0.9, 1.1]). Rejecting large-positive
# values would instead break legitimate accumulation. The public candidate-view rebuild
# entry point SKIPS (not clamps) non-finite / below-floor values and counts them.
_SEQ_Z_RANGE_EPS = 1e-9


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

    def __post_init__(self) -> None:
        """SEQ-B: fail loudly if the policy can produce a negative wealth factor.

        ``rate_noninferiority_z`` bottoms out at ``(-1 + margin) / 0.5``. The wealth
        factor ``1 + lambda_t * z`` stays nonnegative — the property the anytime-valid
        guarantee rests on — only while that floor is ``>= -1/lambda_cap``. Checked here
        so a future policy edit cannot silently break Ville; ``replace()`` re-runs it.
        """
        if self.lambda_cap > 0.0:
            rate_z_floor = (-1.0 + self.rate_noninferiority_margin) / 0.5
            if rate_z_floor < -1.0 / self.lambda_cap - _SEQ_Z_RANGE_EPS:
                raise ValueError(
                    "SequentialPolicy would admit a negative wealth factor on the rate "
                    f"axis: rate z floor {rate_z_floor} < -1/lambda_cap "
                    f"{-1.0 / self.lambda_cap} (lambda_cap must be <= "
                    f"{0.5 / (1.0 - self.rate_noninferiority_margin)} for margin "
                    f"{self.rate_noninferiority_margin})"
                )


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
    # SEQ-3a: count of journal-derived z observations rejected as non-finite /
    # out-of-domain during this rebuild (0 for clean evidence).
    out_of_range_skipped: int = 0


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

    SEQ-B validity fix — the lower clip was at ``-0.5`` and that docstring promise did
    not hold. Clipping is only mean-decreasing when it truncates the UPPER tail; a
    two-sided clip centered on 0 truncates the null-side (lower) tail too, which pulls
    ``E[y]`` UP toward 0 and can drive ``E[z]`` strictly POSITIVE under H0. The wealth
    process is then a SUBmartingale and Ville's inequality does not apply. Measured by
    simulating the exact code path at the null boundary (``true lift = -margin``) with
    Gaussian per-trial noise at the dispersion actually observed in the journal:
    ``P(sup E >= 20)`` was 0.066 at horizon 120 and 0.146 at horizon 400, against an
    alpha = 0.05 bound. This was latent only because the axis had never accumulated.

    The fix moves the LOWER clip to ``-1.0``, which is not a truncation at all: both
    rates are questions-per-hour and therefore nonnegative, so ``y = (r - b)/b >= -1``
    identically and the lower bound is an unreachable numerical guard. What remains is
    the UPPER clip at ``+0.5``, and ``E[clip_upper(y)] <= E[y] <= -margin`` under H0, so
    ``E[z] <= 0`` and the wealth is a genuine nonnegative supermartingale.

    The change is strictly CONSERVATIVE: ``z`` is unchanged for every ``y >= -0.5`` and
    strictly lower (more negative) below it, so a genuinely slow candidate is now
    penalized instead of having its penalty capped at the old floor.

    Factor nonnegativity is preserved. ``z`` now ranges over ``[-2 + 2*margin, 1.1]``;
    ``EProcessState.update`` requires ``z >= -1/lambda_cap``, which holds iff
    ``lambda_cap <= 0.5 / (1 - margin)`` (0.526 for margin=0.05, vs the policy's 0.5).
    The assertion below fails loudly rather than letting a future policy edit silently
    produce a negative wealth factor.
    """
    if baseline_task_rate <= 0.0:
        raise ValueError("baseline_task_rate must be positive")
    y = _clip((task_rate - baseline_task_rate) / baseline_task_rate, -1.0, 0.5)
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
        block["z_rate"] = round(rate_noninf_update.z, 6)
        # SEQ-B: the rate axis previously journaled only its wealth and its z. ``k`` and
        # ``lambda`` in this block are the QUALITY axis's, so a rate e-process stuck at
        # k=1, or frozen by lambda_rate=0 (which multiplies wealth by exactly 1.0 forever),
        # was indistinguishable in the journal from one that was accumulating normally.
        # That is why "E_rate_noninf never leaves ~1.0" survived undiagnosed for the
        # statistic's entire life. Journal the rate axis's own k and lambda.
        block["k_rate"] = rate_noninf_update.k
        block["lambda_rate"] = round(rate_noninf_update.lambda_t, 6)
    return block


def _z_lower_bound(policy: SequentialPolicy) -> float:
    """SEQ-3a factor-nonnegativity floor for ``z``: ``-1/lambda_cap`` (or -inf if uncapped)."""
    if policy.lambda_cap <= 0.0:
        return float("-inf")
    return -1.0 / policy.lambda_cap


def rebuild_candidate_view(
    *,
    candidate: str,
    core_id: str,
    observations: Iterable[Mapping[str, Any] | tuple[int | None, float]],
    policy: SequentialPolicy = DEFAULT_POLICY,
    expected_axis: str = "quality",
) -> CandidateSequentialView:
    """Fold journal rows or ``(trial_id, z)`` pairs into a candidate view.

    SEQ-3a: this is the PUBLIC entry point that accepts journal-derived z sequences.
    Each observed z is validated against the factor-nonnegativity floor
    (``_z_lower_bound`` = ``-1/lambda_cap``) BEFORE it reaches
    ``EProcessState.update`` — a non-finite or below-floor value (corrupt / mis-scaled
    journal evidence that would drive ``1 + lambda*z`` negative and raise) is SKIPPED
    and counted, never fed to the wealth update. Skipping (rather than clamping) is
    what preserves the anytime-valid guarantee: a skipped observation multiplies wealth
    by exactly 1.0, which cannot inflate the Ville false-confirm bound, whereas clamping
    would fabricate an observation that was never actually measured. No upper bound is
    imposed — a large-positive z keeps the factor nonnegative and is legitimate evidence.
    """
    state = EProcessState()
    updates: list[EProcessUpdate] = []
    trials: list[int] = []
    z_floor = _z_lower_bound(policy)
    skipped_out_of_range = 0
    for observation in observations:
        parsed = _coerce_seq_observation(
            observation,
            candidate=candidate,
            core_id=core_id,
            expected_axis=expected_axis,
        )
        if parsed is None:
            continue
        trial_id, z = parsed
        if not math.isfinite(z) or z < z_floor - _SEQ_Z_RANGE_EPS:
            skipped_out_of_range += 1
            continue
        state, update = state.update(z, policy=policy, trial_id=trial_id)
        updates.append(update)
        if trial_id is not None:
            trials.append(trial_id)
    if skipped_out_of_range:
        log.warning(
            "rebuild_candidate_view skipped %d non-finite / below-floor z value(s) for "
            "candidate=%s core_id=%s axis=%s (z floor=%.4f = -1/lambda_cap); corrupt "
            "journal evidence excluded to preserve e-process validity (SEQ-3a)",
            skipped_out_of_range,
            candidate,
            core_id,
            expected_axis,
            z_floor,
        )
    return CandidateSequentialView(
        fingerprint=candidate,
        core_id=core_id,
        trials=tuple(trials),
        quality_state=state,
        quality_updates=tuple(updates),
        state=state.state_name(policy),
        policy_version=policy.version,
        expected_axis=expected_axis,
        out_of_range_skipped=skipped_out_of_range,
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


# SEQ-B: which journal field carries each axis's per-trial statistic. `journal_seq_block`
# writes the quality statistic as ``z`` and the rate statistic as ``z_rate``, so a
# rate-axis rebuild fed journal ROWS must read ``z_rate``. It previously always read
# ``z`` — `expected_axis` was accepted, stored on the view, and never consulted — so a
# rate rebuild from journal rows would have silently folded QUALITY evidence into the
# rate wealth. The live gate happens to pass ``(trial_id, z)`` tuples, so this was latent
# rather than firing; it is fixed here because `rebuild_candidate_view` is the documented
# public entry point for journal-derived evidence.
_SEQ_AXIS_Z_FIELDS: dict[str, tuple[str, ...]] = {
    "quality": ("z",),
    "rate": ("z_rate",),
}


def _coerce_seq_observation(
    observation: Mapping[str, Any] | tuple[int | None, float],
    *,
    candidate: str,
    core_id: str,
    expected_axis: str = "quality",
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
    z_fields = _SEQ_AXIS_Z_FIELDS.get(expected_axis, _SEQ_AXIS_Z_FIELDS["quality"])
    z_field = next((name for name in z_fields if name in block), None)
    if z_field is None:
        return None
    raw_trial_id = observation.get("trial_id", block.get("trial_id"))
    trial_id = int(raw_trial_id) if raw_trial_id is not None else None
    return trial_id, float(block[z_field])


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
