"""HIL-6 pilot requirements and the HIL-2 pre-first-suite-run checklist (items 1-22) as a validator.

A :class:`PilotDesign` is the declared design of one pilot. :func:`unmet_requirements` lists every
requirement the design does not yet satisfy; the driver refuses to plan a cycle while the list is
non-empty. Each checklist item is satisfied in exactly one of three ways (:data:`CHECKLIST`):

* ``code``: enforced mechanically by this package at run time (named function);
* ``design``: a field of :class:`PilotDesign` that must be declared before arm generation;
* ``report``: a report-time column the finalize/report step must carry (no code gate possible
  before a run exists).

Declaring a field satisfies the DESIGN half only. Nothing here is a protocol, metric definition,
sample unit or MDE/noise rule; results without a cited human-ratified protocol stay observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.harness_loop.interfaces import MEMORY_ACCUMULATING_HOMES, MUTABLE_HOMES, SplitRef, splits_disjoint

#: item -> (title, how, where). ``where`` names the enforcing function or the design field.
CHECKLIST: Mapping[int, tuple[str, str, str]] = {
    1: ("matched-rollout task-level baseline arm", "design", "PilotDesign.matched_rollout_baseline"),
    2: ("proposer/meta/summariser tokens in the budget", "code", "driver: TokenCount per seat summed"),
    3: ("disjoint train/validation/test; select on validation; test touched once", "code",
        "PilotDesign splits + driver.final_test single-use"),
    4: ("report with and without the verification signal", "design", "PilotDesign.verification_arms"),
    5: (">=2 runs, CI, stated MDE", "design", "PilotDesign.runs / mde"),
    6: ("unmodified-harness control on the same split", "design", "PilotDesign.unmodified_control"),
    7: ("executor/proposer isolation: fence both arms, tripwire, decoy canary, label timing, "
        "index-style view", "code", "driver: fenced_eval_request, require_fence_active, tripwire, "
        "ProposerView outcome-free"),
    8: ("context-policy A/B reporting contract", "design",
        "PilotDesign.context_policy_controls (home 3.6 only)"),
    9: ("arrival split and commitment; stopping rule declared", "code",
        "driver: arrival eval with no tuning; PilotDesign.stopping_rule"),
    10: ("harness growth per cycle", "code", "driver: incumbent bytes per cycle in the cycle event"),
    11: ("validation power statement", "design", "PilotDesign.paired_n / expected_discordant / min_win_share"),
    12: ("feature actually exercised (treatment-fired count)", "code", "driver: ArmResult.treatment_fired required"),
    13: ("headroom and failure decomposition note", "design", "PilotDesign.headroom_note"),
    14: ("pinned arm identity (item-14 arm_pins)", "code", "driver: ArmSpec.pins must be the closed set"),
    15: ("exclusions and slices declared before the run", "design", "PilotDesign.exclusions_declared"),
    16: ("token-cost columns (uncached prefill + decode, cached share)", "code", "ArmResult.tokens"),
    17: ("label-free/judged arms: incumbent wins ties, verdict matrix, argmax check", "code",
         "IncumbentWinsTies, VerdictMatrix.validate / choice_is_argmax"),
    18: ("regression readout beside the capability axis", "design", "PilotDesign.regression_set_id"),
    19: ("per-role arm design; local reviewer for reviewer arms", "design",
         "PilotDesign.per_role_arm_set / reviewer_model"),
    20: ("memory-accumulating stream positions", "design", "PilotDesign.stream_positions (homes 3.1/3.2/3.4)"),
    21: ("attempt ledger and protected layer", "code", "AttemptLedger"),
    22: ("arm prompt interference lint recorded", "code", "driver: prompt_interference_lint warnings in the run record"),
}

VERIFICATION_ARMS = frozenset({"with", "none", "self_judged"})
TOKEN_SEATS = frozenset({"proposer", "judge", "solver"})


def model_family(model_id: str) -> str:
    """Coarse vendor family for the cross-model judge check (pilot requirement 3)."""
    m = model_id.lower().removeprefix("cloud:")
    for prefix, fam in (("claude", "anthropic"), ("sonnet", "anthropic"), ("opus", "anthropic"),
                        ("haiku", "anthropic"), ("gpt", "openai"), ("codex", "openai"),
                        ("qwen", "qwen"), ("gemma", "google"), ("glm", "zhipu"),
                        ("deepseek", "deepseek"), ("llama", "meta")):
        if m.startswith(prefix):
            return fam
    return m.split("-")[0] or m


@dataclass(frozen=True)
class PilotDesign:
    feature_id: str
    home: str
    primary_metric: str
    primary_unit: str
    primary_direction: str
    train: SplitRef
    validation: SplitRef
    arrival: SplitRef
    test: SplitRef
    cycles: int
    stopping_rule: str
    local_editor_model: str
    reference_editor_model: str
    judge_model: str
    matched_rollout_baseline: bool = False
    unmodified_control: bool = False
    verification_arms: frozenset[str] = frozenset()
    runs: int = 0
    mde: float | None = None
    paired_n: int | None = None
    expected_discordant: int | None = None
    min_win_share: float | None = None
    token_seats: frozenset[str] = frozenset()
    headroom_note: str = ""
    exclusions_declared: bool = False
    regression_set_id: str = ""
    per_role_arm_set: bool = False
    reviewer_model: str = ""  # only for features whose mechanism is a reviewer/critic
    served_local_models: frozenset[str] = frozenset()
    context_policy_controls: frozenset[str] = frozenset()
    stream_positions: tuple[str, ...] = ()
    alpha_budget: float = 0.05
    notes: Mapping[str, str] = field(default_factory=dict)


CONTEXT_POLICY_CONTROLS = frozenset({"no_management", "wording_only"})
STREAM_POSITIONS = ("first_exposure", "repeat", "variant", "variant_after_interference")


def unmet_requirements(d: PilotDesign) -> list[tuple[str, str]]:
    """Every unmet HIL-6 pilot requirement (``P<n>``) or HIL-2 design item (``I<n>``)."""
    out: list[tuple[str, str]] = []

    def need(key: str, ok: bool, why: str) -> None:
        if not ok:
            out.append((key, why))

    need("home", d.home in MUTABLE_HOMES, f"home {d.home!r} has no loop-mutable arm (HIL-1)")
    need("objective", bool(d.primary_metric and d.primary_unit)
         and d.primary_direction in {"higher_better", "lower_better"},
         "primary objective (metric, unit, direction) declared before arm generation")
    # P1 / I9 / I3
    need("P1", d.cycles >= 2, "at least two cycles with an arriving split between them")
    need("P1", d.arrival.role == "arrival", "the arrival split has role 'arrival'")
    need("I3", d.train.role == "train" and d.validation.role == "validation" and d.test.role == "test",
         "train/validation/test roles")
    need("I3", splits_disjoint(d.train, d.validation, d.arrival, d.test),
         "train, validation, arrival and test splits are pairwise disjoint")
    need("I9", bool(d.stopping_rule.strip()), "stopping rule declared in advance")
    # P2 editor-capability arm
    need("P2", bool(d.local_editor_model and d.reference_editor_model)
         and d.local_editor_model != d.reference_editor_model,
         "editor-capability arm: a local editor and a different reference editor")
    need("P2", not d.served_local_models or d.local_editor_model in d.served_local_models,
         "the local editor is a model served on EPYC")
    # P3 cross-model judge
    need("P3", bool(d.judge_model), "a judge arm")
    need("P3", d.judge_model not in {d.local_editor_model, d.reference_editor_model},
         "the judge is a different model from every proposer")
    need("P3", model_family(d.judge_model) != model_family(d.local_editor_model),
         "the judge is a different model family from the local proposer (correlated error)")
    # P4 / I20
    if d.home in MEMORY_ACCUMULATING_HOMES:
        need("I20", tuple(d.stream_positions) == STREAM_POSITIONS,
             f"memory-accumulating home: stream positions {STREAM_POSITIONS}")
    # I8
    if d.home == "3.6":
        need("I8", CONTEXT_POLICY_CONTROLS <= d.context_policy_controls,
             "context-policy arm: no-management and wording-only controls")
    # remaining design items (P5: items 7-22 all apply; 1-6 per HIL-7)
    need("I1", d.matched_rollout_baseline, "matched-rollout task-level baseline arm")
    need("I2", TOKEN_SEATS <= d.token_seats, f"token seats counted: {sorted(TOKEN_SEATS)}")
    need("I4", VERIFICATION_ARMS <= d.verification_arms,
         "with / without / self-judged verification-signal arms")
    need("I5", d.runs >= 2 and d.mde is not None, "at least two runs and a stated MDE")
    need("I6", d.unmodified_control, "unmodified-harness control on the same split")
    need("I11", d.paired_n is not None and d.expected_discordant is not None
         and d.min_win_share is not None, "power statement: paired n, discordant pairs, win share")
    need("I13", bool(d.headroom_note.strip()), "headroom / expected-null note before arms")
    need("I15", d.exclusions_declared, "exclusions and robust slices declared before the run")
    need("I18", bool(d.regression_set_id), "regression readout set named")
    need("I19", d.per_role_arm_set, "per-role leave-one-out + all-off + all-on arm set")
    if d.reviewer_model:
        need("I19", d.reviewer_model in d.served_local_models,
             "a reviewer/critic mechanism runs with the served local model")
    need("I21", 0.0 < d.alpha_budget <= 1.0, "attempt-ledger alpha budget in (0, 1]")
    return out
