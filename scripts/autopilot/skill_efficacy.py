#!/usr/bin/env python3
"""EV-10 — skill/prompt efficacy gating (EV-10a) + leak-free surrogate-verifier
scoring (EV-10b). Sidecar module: pure functions, no inference, no live wiring.

Background (research/intake_index.yaml intake-096 SkillsBench, intake-628 CoEvoSkills;
epyc-root handoffs/active/eval-tower-verification.md EV-10):

  SkillsBench (arxiv:2602.12670) showed empirically that (a) self-generated skills are
  net-NEGATIVE on average (-1.3pp vs no-skill) and (b) curated skills REGRESS 16/84
  individual tasks even when the aggregate improves. So before any autopilot
  self-optimization (a SkillOpt-style PromptForge edit) is trusted, the accept-path
  must measure the *paired* skill-vs-no-skill delta PER SUITE with an explicit
  negative-delta guard — an aggregate gain must not hide a per-suite regression.

EV-10a (evaluate_skill_efficacy / *_split): the paired-efficacy decision. This is the
  structured-verdict counterpart to scripts/benchmark/skill_transfer_regression.py
  (whose compare_and_report() already flags `delta < -threshold` cells for the
  model-swap case but only PRINTS). Here `without_artifact` = the no-skill arm and
  `with_artifact` = the with-skill arm of the same candidate.

EV-10b (surrogate_proxy_reward / surrogate_feedback / require_cross_family): the
  CoEvoSkills leak-free scoring pattern — a self-authored assertion suite gives a dense
  proxy reward, and when the surrogate passes but the ground-truth oracle fails, only an
  opaque bit is returned (no detail) so the generator cannot overfit to held-out tests.
  The LLM that authors the assertions is NOT called here; the caller injects the
  assertion outcomes (same convention as scripts/autopilot/verbalized_sampling.py).

WIRING IS DEFERRED (sidecar pattern, matching AP-29/30/31): these functions are not yet
  called from the live autopilot accept-path. The intended hook is
  scripts/autopilot/species/prompt_forge.py `apply_mutation_isolated` -> ctx.accept(),
  to be added at the next AR-3 restart so a running campaign is not perturbed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

__all__ = [
    "EfficacyVerdict",
    "evaluate_skill_efficacy",
    "evaluate_skill_efficacy_split",
    "surrogate_proxy_reward",
    "SurrogateFeedback",
    "surrogate_feedback",
    "require_cross_family",
]


# ──────────────────────────────────────────────────────────────────────────
# EV-10a — paired skill-vs-no-skill efficacy gate (negative-delta guard)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class EfficacyVerdict:
    """Outcome of a paired skill-vs-no-skill efficacy check.

    accept            -- True iff the artifact is safe to keep under the gate.
    aggregate_delta   -- mean per-suite (with - without) over comparable suites.
    per_suite_delta   -- {suite: with - without} for every suite present in both arms.
    regressed_suites  -- [(suite, delta)] for suites dropping below -regress_threshold,
                         worst first. Non-empty => the SkillsBench "16/84 regression"
                         pattern is present and the gate rejects.
    reason            -- human-readable explanation of the decision.
    """

    accept: bool
    aggregate_delta: float
    per_suite_delta: dict[str, float] = field(default_factory=dict)
    regressed_suites: list[tuple[str, float]] = field(default_factory=list)
    reason: str = ""


def _comparable_suites(
    without_artifact: Mapping[str, float],
    with_artifact: Mapping[str, float],
) -> list[str]:
    """Suites present in BOTH arms with finite scores, sorted for determinism."""
    out: list[str] = []
    for suite in without_artifact:
        if suite not in with_artifact:
            continue
        b = without_artifact[suite]
        a = with_artifact[suite]
        if b is None or a is None:
            continue
        if isinstance(b, float) and math.isnan(b):
            continue
        if isinstance(a, float) and math.isnan(a):
            continue
        out.append(suite)
    return sorted(out)


def evaluate_skill_efficacy(
    without_artifact: Mapping[str, float],
    with_artifact: Mapping[str, float],
    *,
    regress_threshold: float = 0.10,
    require_aggregate_gain: bool = True,
) -> EfficacyVerdict:
    """Decide whether a skill/agent-file/prompt artifact is worth keeping.

    Compares the with-artifact arm against the no-artifact (no-skill) arm on a per-suite
    basis. Scores are pass-rates/qualities on a comparable scale (any consistent units;
    threshold is in the same units).

    Gate (both must hold to accept):
      1. NEGATIVE-DELTA GUARD: no comparable suite may drop by more than
         `regress_threshold` (the SkillsBench per-task-regression protection). A single
         regressed suite rejects, even if the aggregate improves.
      2. AGGREGATE GAIN: mean per-suite delta must be > 0 (skippable via
         require_aggregate_gain=False, e.g. for a neutrality probe).

    A no-op artifact (all deltas ~0) does NOT accept when require_aggregate_gain is True —
    this is intentional, mirroring SkillOpt's strict-improvement acceptance.
    """
    suites = _comparable_suites(without_artifact, with_artifact)
    if not suites:
        return EfficacyVerdict(
            accept=False,
            aggregate_delta=0.0,
            reason="no comparable suites present in both arms",
        )

    per_suite = {s: float(with_artifact[s]) - float(without_artifact[s]) for s in suites}
    aggregate = sum(per_suite.values()) / len(per_suite)
    regressed = sorted(
        ((s, d) for s, d in per_suite.items() if d < -regress_threshold),
        key=lambda kv: kv[1],
    )

    if regressed:
        worst_s, worst_d = regressed[0]
        return EfficacyVerdict(
            accept=False,
            aggregate_delta=aggregate,
            per_suite_delta=per_suite,
            regressed_suites=regressed,
            reason=(
                f"{len(regressed)} suite(s) regressed beyond {regress_threshold:+.3f} "
                f"(worst: {worst_s} {worst_d:+.3f}); reject despite aggregate "
                f"{aggregate:+.3f}"
            ),
        )

    if require_aggregate_gain and aggregate <= 0:
        return EfficacyVerdict(
            accept=False,
            aggregate_delta=aggregate,
            per_suite_delta=per_suite,
            reason=f"no aggregate gain (mean delta {aggregate:+.3f} <= 0)",
        )

    return EfficacyVerdict(
        accept=True,
        aggregate_delta=aggregate,
        per_suite_delta=per_suite,
        reason=(
            f"accept: aggregate {aggregate:+.3f}, no suite below "
            f"{-regress_threshold:+.3f} across {len(suites)} suite(s)"
        ),
    )


def evaluate_skill_efficacy_split(
    dev_without: Mapping[str, float],
    dev_with: Mapping[str, float],
    test_without: Mapping[str, float],
    test_with: Mapping[str, float],
    *,
    regress_threshold: float = 0.10,
    require_aggregate_gain: bool = True,
) -> EfficacyVerdict:
    """dev/test_normal split discipline: require ACCEPT on BOTH dev and test.

    Guards against an artifact that overfits the dev split (the AppWorld dev/test_normal
    convention adopted in eval-tower-verification.md 2026-04-30). The returned verdict
    carries the merged per-suite deltas (dev:/test: prefixed) and the union of regressed
    suites; accept is the logical AND of the two arm verdicts.
    """
    dev_v = evaluate_skill_efficacy(
        dev_without, dev_with,
        regress_threshold=regress_threshold,
        require_aggregate_gain=require_aggregate_gain,
    )
    test_v = evaluate_skill_efficacy(
        test_without, test_with,
        regress_threshold=regress_threshold,
        require_aggregate_gain=require_aggregate_gain,
    )
    merged_delta: dict[str, float] = {}
    merged_delta.update({f"dev:{s}": d for s, d in dev_v.per_suite_delta.items()})
    merged_delta.update({f"test:{s}": d for s, d in test_v.per_suite_delta.items()})
    regressed = (
        [(f"dev:{s}", d) for s, d in dev_v.regressed_suites]
        + [(f"test:{s}", d) for s, d in test_v.regressed_suites]
    )
    accept = dev_v.accept and test_v.accept
    if accept:
        reason = (
            f"accept on BOTH splits (dev {dev_v.aggregate_delta:+.3f}, "
            f"test {test_v.aggregate_delta:+.3f})"
        )
    else:
        failed = []
        if not dev_v.accept:
            failed.append(f"dev: {dev_v.reason}")
        if not test_v.accept:
            failed.append(f"test: {test_v.reason}")
        reason = "reject — " + " | ".join(failed)
    return EfficacyVerdict(
        accept=accept,
        aggregate_delta=(dev_v.aggregate_delta + test_v.aggregate_delta) / 2.0,
        per_suite_delta=merged_delta,
        regressed_suites=regressed,
        reason=reason,
    )


# ──────────────────────────────────────────────────────────────────────────
# EV-10b — leak-free surrogate-verifier scoring (CoEvoSkills, intake-628)
# ──────────────────────────────────────────────────────────────────────────


def surrogate_proxy_reward(assertion_pass: Sequence[bool]) -> float:
    """Proxy reward = fraction of the self-authored assertion suite that passes.

    The assertions are produced by an independent verifier LLM session that sees only
    the task instruction and the agent's output files (not the ground-truth tests). That
    LLM call happens in the caller; this function only aggregates the boolean outcomes.
    Returns 0.0 for an empty suite (no evidence => no reward).
    """
    if not assertion_pass:
        return 0.0
    return sum(1 for p in assertion_pass if p) / len(assertion_pass)


@dataclass
class SurrogateFeedback:
    """Decision about what to feed back to the skill generator.

    proxy_reward   -- fraction of surrogate assertions passing.
    dense_feedback -- True when the surrogate found failures (reward < 1): the generator
                      may receive per-assertion detail to guide the next edit.
    opaque_only    -- True when the surrogate PASSED but the ground-truth oracle FAILED:
                      return only an opaque pass/fail bit (no content) so the generator
                      cannot overfit to the held-out tests (CoEvoSkills anti-overfit).
    accepted       -- True only when surrogate passed AND the oracle did not contradict it.
    """

    proxy_reward: float
    dense_feedback: bool
    opaque_only: bool
    accepted: bool


def surrogate_feedback(
    proxy_reward: float,
    oracle_pass: bool | None = None,
) -> SurrogateFeedback:
    """Map (surrogate reward, optional oracle bit) to a feedback decision.

    - reward < 1.0                       -> dense feedback (surrogate found failures).
    - reward == 1.0 and oracle_pass False -> opaque-only (anti-overfit): surrogate is
                                            over-optimistic vs the authoritative oracle;
                                            emit only the failing bit, no detail.
    - reward == 1.0 and oracle_pass in {None, True} -> accepted.

    The ground-truth oracle (when available) remains the authoritative arbiter; the
    surrogate exists to give DENSE feedback where the oracle returns only pass/fail.
    """
    passed = proxy_reward >= 1.0
    if not passed:
        return SurrogateFeedback(
            proxy_reward=proxy_reward,
            dense_feedback=True,
            opaque_only=False,
            accepted=False,
        )
    if oracle_pass is False:
        return SurrogateFeedback(
            proxy_reward=proxy_reward,
            dense_feedback=False,
            opaque_only=True,
            accepted=False,
        )
    return SurrogateFeedback(
        proxy_reward=proxy_reward,
        dense_feedback=False,
        opaque_only=False,
        accepted=True,
    )


def require_cross_family(
    generator_model: str,
    verifier_model: str,
    cross_family_fn: Callable[[str, str], bool],
) -> bool:
    """Enforce that the surrogate verifier is a DIFFERENT model family than the generator.

    `cross_family_fn` is injected (production: scripts/autopilot/eval_tower.check_cross_family)
    so this module does not import the live eval_tower (which avoids coupling the sidecar to
    in-flight edits there). Returns True when the pairing is cross-family (safe); raises
    ValueError when same-family, since same-family self-verification amplifies bias
    (eval-tower-verification.md confirmation-bias mitigation #1).
    """
    if not cross_family_fn(generator_model, verifier_model):
        raise ValueError(
            f"same-family verification rejected: generator '{generator_model}' and "
            f"verifier '{verifier_model}' are the same family; pick a cross-family verifier"
        )
    return True
