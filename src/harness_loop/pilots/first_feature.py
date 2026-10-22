"""HIL-6i: the pilot design for the FIRST feature of the sibling harness-improvement loop.

Build only. The operator chose HIL-4 option B on 2026-10-22 (coordinator session 3c4ca599); the
2026-09-16 deferment on RUNNING a harness-improvement loop is still in force, and HS-5b / P0.4
acceptance and freeze plus the applicable protocols gate any run. Nothing here reaches a model,
reads the question pool, or schedules anything. Every decision this design produces is an
observation unless a human-ratified protocol id is cited (MEASUREMENT.md).

Design document (epyc-root): ``docs/design/hil6i-pilot-design-first-feature-20261022.md``.

Chosen home: HS-4 section 3.5 (delegation). The mutable target is the architect's phase-A
decision prompt ``orchestration/prompts/architect_investigate.md``: it is live by default
(``architect_delegation`` production default True in ``src/features.py``), resolved per request
from a prompt root (``x_orchestrator_prompt_root`` on ``/chat``, so arms never write the live
file), consumed by exactly one role (``architect_general``; ``delegation_stage.py``), scored by
deterministic pool scorers, and its treatment signal is mechanical (the phase-A decision mode
``direct`` / ``investigate`` recorded in the delegation stats).

What this module provides:

* the declared constants of the design (ids are the ones in the orchestrator config at
  ``origin/main`` df9d2eca; see the design document for each source path);
* :func:`build_splits`: a pure, deterministic, stratified draw over question-pool rows the CALLER
  supplies, run once at freeze time; it never opens the pool itself;
* :func:`first_feature_design`: the filled :class:`PilotDesign` for cycle 1 or cycle 2;
* :func:`protected_layer_reason` / :func:`make_candidate_screen`: the HIL-1 protected layer of
  the target file, as a pure check a later entrypoint may pass to the driver's ``leakage`` seam;
* :func:`sign_test_critical_wins` / :func:`sign_test_power`: the item-11 power arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from math import comb
from typing import Any, Callable, Iterable, Mapping

from src.harness_loop.interfaces import SplitRef, splits_disjoint
from src.harness_loop.pilot import PilotDesign

DESIGN_DOC = "docs/design/hil6i-pilot-design-first-feature-20261022.md"  # epyc-root

# ── feature, home, target ─────────────────────────────────────────────────────

FEATURE_ID = "HS4-3.5-architect-investigate-prompt"
HOME = "3.5"
TARGET_PATH = "orchestration/prompts/architect_investigate.md"
TARGET_KIND = "prompt"
#: sha256 of the target at orchestrator origin/main df9d2eca (2026-10-22). Informational: the
#: freeze re-reads the file and records the bytes it actually pins as the unmodified control.
TARGET_SHA256_AT_DESIGN = "82dcf9d0cdece2e22f621ba9fe181001e6524a11f7dc2bb7e23f7414da902090"
CONSUMER_ROLE = "architect_general"
CONSUMER_MODEL = "Qwen3.8-Flash-Next-UD-IQ4_XS"  # model_registry.yaml roles.architect_general

PRIMARY_METRIC = "task_pass_share"
PRIMARY_UNIT = "fraction"
PRIMARY_DIRECTION = "higher_better"

# ── editors and judge (P2, P3) ────────────────────────────────────────────────

#: P2 local editor: the 35B-A3B MoE served by frontdoor/worker_general (model_registry.yaml).
LOCAL_EDITOR_MODEL = "Qwen3.6-35B-A3B-MTP-Q8_0"
#: P2 reference editor: OpenAI family through the codex CLI (cloud_judges.yaml codex-luna-low
#: ``model``). Chosen so the pinned Anthropic judge shares a family with NEITHER editor.
REFERENCE_EDITOR_MODEL = "gpt-5.6-luna"
#: P3 judge: the exact-version pin in cloud_judges.yaml (``sonnet-5-5-low`` → claude-sonnet-5-5).
JUDGE_MODEL = "cloud:sonnet-5-5-low"
JUDGE_PINNED_MODEL = "claude-sonnet-5-5"
#: The cross-family judge option, valid ONLY if the reference editor were Anthropic-family.
CROSS_FAMILY_JUDGE_OPTION = "cloud:codex-luna-low"

#: Models with a live (non-deprecated) role row in orchestration/model_registry.yaml at df9d2eca.
#: Registered-to-serve, not proof of a running process: the run's infra fingerprint proves that.
SERVED_LOCAL_MODELS = frozenset({
    "Qwen3.6-35B-A3B-MTP-Q8_0",
    "Qwen3.8-Flash-Next-UD-IQ4_XS",
    "Qwen3.8-27B-Q8_0",
    "Qwen3-VL-30B-A3B-Instruct",
})

#: Published editor prompt the local editor's meta-prompt is ported from (M9): GEPA 0.0.26
#: ``InstructionProposalSignature.default_prompt_template`` as installed in the orchestrator
#: .venv. The port replaces the inputs/outputs/feedback slot with the outcome-free dossier and
#: drops the "include niche factual information" sentence (an eval-content leakage vector).
EDITOR_PROMPT_SOURCE = "gepa==0.0.26:gepa/strategies/instruction_proposal.py:default_prompt_template"
EDITOR_PROMPT_SOURCE_SHA256 = "7949f860c308b9d99a2091d2cadc2c18824f684ca373260d285eecd3f3e63336"

# ── splits ────────────────────────────────────────────────────────────────────

#: epyc-inference-research question pool and the corpus version read at design time.
POOL_PATH = "benchmarks/prompts/question_pool.jsonl"
POOL_CORPUS_VERSION_AT_DESIGN = "corpus-ev19b-20261021"
#: One stratum per decision rule the prompt states: direct MC answer, obscure fact (search),
#: computation in the REPL, code reasoning, code implementation (delegate to coder_escalation).
STRATA = ("mmlu_pro", "simpleqa", "math", "cruxeval", "livecodebench")
#: Per-stratum sizes, assigned as consecutive slices of one seeded order, in this order.
PER_STRATUM = (("train", 30), ("validation", 84), ("arrival", 30), ("arrival_2", 30), ("test", 48))
SPLIT_SEED = "hil6i-first-feature-v1"
#: ``build_architect_investigate_prompt`` slices ``question[:2000]`` and ``context[:3000]``
#: (src/prompt_builders/review.py). Items longer than that would measure the cap, not the prompt.
QUESTION_CAP_CHARS = 2000
EVIDENCE_CAP_CHARS = 3000

# ── power, runs, alpha (items 5, 11, 21) ──────────────────────────────────────

LINEAGES = ("local_editor", "reference_editor")
CYCLES = 2
MAX_CANDIDATES_PER_CYCLE = 2
ATTEMPTS_PLANNED = len(LINEAGES) * CYCLES * MAX_CANDIDATES_PER_CYCLE  # 8
ALPHA_BUDGET = 0.05
ALPHA_PER_ATTEMPT = ALPHA_BUDGET / ATTEMPTS_PLANNED  # 0.00625
RUNS = 2
PAIRED_N = dict(PER_STRATUM)["validation"] * len(STRATA)  # 420
ASSUMED_DISCORDANCE = 0.20  # replaced by the measured incumbent A/A discordance before arms
EXPECTED_DISCORDANT = round(PAIRED_N * ASSUMED_DISCORDANCE)  # 84
MDE = 0.075  # absolute pass-share gain with >= 0.8 power at the assumed discordance


def sign_test_critical_wins(discordant: int, alpha: float) -> int:
    """Smallest win count k with P(X >= k | Binomial(discordant, 1/2)) <= alpha (one-sided)."""
    if discordant <= 0 or not (0.0 < alpha < 1.0):
        raise ValueError("discordant must be positive and alpha in (0, 1)")
    tail = 0.0
    for k in range(discordant, -1, -1):
        tail += comb(discordant, k) / 2 ** discordant
        if tail > alpha:
            return k + 1
    return 0


def sign_test_power(discordant: int, critical_wins: int, win_share: float) -> float:
    """P(X >= critical_wins | Binomial(discordant, win_share)), conditional on the discordant count."""
    return float(sum(comb(discordant, j) * win_share ** j * (1 - win_share) ** (discordant - j)
                     for j in range(critical_wins, discordant + 1)))


CRITICAL_WINS = sign_test_critical_wins(EXPECTED_DISCORDANT, ALPHA_PER_ATTEMPT)  # 54
MIN_WIN_SHARE = CRITICAL_WINS / EXPECTED_DISCORDANT  # 0.643
#: IncumbentWinsTies margin equivalent to the critical count AT the expected discordant count.
DECISION_MARGIN = (2 * CRITICAL_WINS - EXPECTED_DISCORDANT) / PAIRED_N  # 0.057
#: Above this incumbent validation pass share, n*(1-p) < CRITICAL_WINS: no candidate can pass.
CEILING_PASS_SHARE = 1.0 - CRITICAL_WINS / PAIRED_N  # 0.871

REGRESSION_SET_ID = "hil6i-arch-deleg-regress-v1"
TOKEN_SEATS = frozenset({"proposer", "judge", "solver", "solver_child"})

STOPPING_RULE = (
    "Fixed horizon: exactly 2 cycles per editor lineage (local_editor, reference_editor), at most "
    f"{MAX_CANDIDATES_PER_CYCLE} candidates per cycle, {ATTEMPTS_PLANNED} attempts in all at "
    f"alpha {ALPHA_PER_ATTEMPT:g} each from one shared ledger (budget {ALPHA_BUDGET:g}). Before any "
    "candidate is evaluated, the two incumbent validation runs (A/A) give the measured discordance; "
    "if a candidate at the declared MDE would then pass with power < 0.8, or the incumbent pass "
    f"share exceeds {CEILING_PASS_SHARE:.3f}, the run is labelled underpowered and no candidate is "
    "evaluated (n is never enlarged mid-run; a larger n is a re-freeze). Early stop only on "
    "AlphaBudgetExhausted or an infra NON_COMPARABLE incumbent. A cycle-2 KEEP_BASE is an outcome. "
    "The reported state is the committed final state after cycle 2 plus the per-cycle trajectory; "
    "no best-of-rounds figure is reported."
)

HEADROOM_NOTE = (
    "Before arm generation, scan the incumbent's existing architect_general delegated-mode eval "
    "traces (chat_delegation stats: break_reason in {max_loops, semantic_dedup, role_repetition, "
    "token_budget, wall_clock_budget, pre_delegation_architect_error}, decision-line parse "
    "fallback, decision-guard rewrites) and the HIL-12 L2 categories, and count the share of "
    "baseline failures a prompt rule could prevent. If that share times the baseline failure rate "
    f"is below the declared MDE ({MDE:g}), record an expected-null note and do not generate arms. "
    "Not yet counted: the scan reads traces only and is run at freeze, not at design time."
)

NOTES: Mapping[str, str] = {
    # M9 extras (docs/design/hil6-sibling-loop-implementation.md §5)
    "m9.near_ceiling_power": (
        f"paired_n={PAIRED_N}, expected_discordant={EXPECTED_DISCORDANT} at assumed discordance "
        f"{ASSUMED_DISCORDANCE:g}, critical wins {CRITICAL_WINS} (one-sided exact sign test, alpha "
        f"{ALPHA_PER_ATTEMPT:g}), min win share {MIN_WIN_SHARE:.3f}. If the incumbent validation pass "
        f"share p > 0.9, expected_discordant is restated from headroom (at most n*(1-p) wins); for "
        f"p > {CEILING_PASS_SHARE:.3f} no candidate can reach {CRITICAL_WINS} wins, so the run is "
        "labelled underpowered and its null is uninformative. Per-stratum readouts are diagnostic "
        "and are labelled underpowered whenever their stratum is above 0.9."),
    "m9.zero_adoption": (
        "treatment_fired = items whose phase-A architect call resolved the arm's prompt bytes "
        "(resolver provenance under the arm's prompt root). A result with treatment_fired == 0 is "
        "'not exercised' and its null says nothing about content. Reported beside it: the I| "
        "(investigate) decision rate per arm, the share of decisions rewritten by "
        "_apply_decision_guards, and the share answered with no model call."),
    "m9.reference_trajectory_arm": (
        "Diagnostic only: a separate local-editor proposal in which the dossier also carries one "
        "T1 trajectory whose delegated path reached a scorer-passing answer. Showing it implies an "
        "outcome, so the arm sits outside the item-7 selection path: never committed, never in the "
        "attempt ledger, reported apart."),
    "m9.offline_then_freeze": (
        "Arms are byte-pinned prompt trees under the orchestrator tmp/gepa_prompt_roots base, sent "
        "per request as x_orchestrator_prompt_root; the loop never writes "
        "orchestration/prompts/architect_investigate.md. Promotion of a committed candidate is a "
        "separate human-applied change after the pilot."),
    "m9.local_editor_prompt_port": (
        f"Local editor meta-prompt = port of {EDITOR_PROMPT_SOURCE} (sha256 "
        f"{EDITOR_PROMPT_SOURCE_SHA256}); the port's own sha256 is pinned when HIL-6h lands it. "
        "The reference editor receives the same ported meta-prompt and the same dossier."),
    "m9.planner_self_modification": (
        "Excluded: the proposer, its meta-prompt, the judge, the evaluator, the splits, the "
        "reference arms and the ledger are the protected layer (item 21)."),
    "m9.regression_set": (
        f"{REGRESSION_SET_ID} = validation items the incumbent passed in both runs (right-to-wrong "
        "count per arm) plus the non-degeneracy readouts: empty-answer share, decision-line parse "
        "failure share, investigate-rate shift. The chosen strata hold no labelled unsafe-control "
        "items; none are invented."),
    # P2 / P3
    "p2.editor_arm": (
        "Two lineages from the same incumbent, same splits, executor and dossier rule: "
        f"{LOCAL_EDITOR_MODEL} (local) and {REFERENCE_EDITOR_MODEL} (reference). Gain is reported "
        "with run-to-run noise; no local-feasibility claim is drawn from the sources."),
    "p3.family_caveat": (
        f"Judge {JUDGE_MODEL} ({JUDGE_PINNED_MODEL}) is Anthropic; the local editor and the "
        "consumer are Qwen, the reference editor is OpenAI, so the judge shares a family with no "
        f"proposer. {CROSS_FAMILY_JUDGE_OPTION} would share the reference editor's family and is "
        "not used; it becomes the judge only if the reference editor is switched to Anthropic."),
    "p3.egress": (
        "Judge calls send validation items to Anthropic (claude CLI); reference-editor calls send "
        "T1 dossier excerpts to OpenAI (codex CLI). Both judge entries are verified_live: false. "
        "Egress of the chosen strata is cleared by the operator before any run; every call log "
        "records its egress. The typed credibility-matrix judge schema is HIL-6h."),
    # items 1, 4, 6, 15, 19
    "i1.matched_rollout_baseline": (
        "Incumbent sampled k times per validation item, k = solver rollouts one lineage spends per "
        f"item ({CYCLES} cycles x {MAX_CANDIDATES_PER_CYCLE} candidates x {RUNS} runs = 8), final "
        "answer by majority over extracted answers (label-free); own run_id. If the role sampler is "
        "deterministic the k samples coincide and this is reported, not hidden."),
    "i4.verification_arms": (
        "Selection modes over the same validation evaluations: with = scorer outcomes plus the "
        "sign-test gate; none = the first screened candidate in proposal order, committed without "
        "looking; self_judged = the local editor picks among its candidates without labels; the "
        "P3 judge matrix is the fourth, cross-model mode."),
    "i6.unmodified_control": (
        "The target bytes at freeze. Equal to the incumbent in cycle 1; differs from it in cycle 2 "
        "whenever cycle 1 committed. Stated per run."),
    "i15.exclusions": (
        "Pre-draw: other suites; rows with image_path; question > 2000 or context > 3000 chars; "
        "EV-19 gold self-check outcome wrong/scorer_error; core_v2 members; qids seen in any eval "
        "within 60 days (P-QUAL-PROMO); livecodebench dropped as a whole stratum before freeze if "
        "the code-execution sandbox is unavailable. In-run: none removed from the primary slice "
        "(infra failures count as failures, classified by reason per P-AB-1); robust slice drops "
        "items with an infra failure in either arm, diagnostic only, with per-arm counts."),
    "i19.per_role_arm_set": (
        f"One consuming role ({CONSUMER_ROLE}, {CONSUMER_MODEL}); n=1 feature, so the n+2 set is "
        "{all-on (delegation on, arm prompt), all-off (allow_delegation=False)}, leave-one-out "
        "being all-off. No verdict transfers to another role or model. Not a reviewer mechanism."),
    # gaps in the skeleton, recorded rather than patched (pilot.py / driver.py are not edited)
    "gap.arrival_per_cycle": (
        "PilotDesign has one arrival split and run_cycle refuses a tune split overlapping it, so P1's "
        "cycle-2 re-tune on T1+T2 needs a second design (train=T1+T2, arrival=T3) on the same "
        "journal; see first_feature_design(cycle=2)."),
    "gap.paired_final_test": (
        "SiblingLoopDriver.final_test evaluates one arm and then refuses; a paired test claim needs "
        "the committed arm and the unmodified control in the same single touch."),
    "gap.decision_rule": (
        "No paired sign-test DecisionRule and no PilotDesign.decision_rule field: IncumbentWinsTies"
        f"(margin={DECISION_MARGIN:.4f}) equals the critical count only at the expected discordant "
        "count."),
    "gap.design_fields": (
        "PilotDesign has no fields for the split manifest digest, the target sha256, the editor "
        "meta-prompt pin, egress clearance or the per-cycle arrival split; they live in notes."),
}


# ── splits: pure, deterministic, caller-supplied rows ─────────────────────────


@dataclass(frozen=True)
class FrozenSplits:
    train: SplitRef
    validation: SplitRef
    arrival: SplitRef  # T2: released after cycle 1 commits; joins tuning in cycle 2
    arrival_2: SplitRef  # T3: released after cycle 2 commits
    test: SplitRef
    seed: str
    manifest_sha256: str


def _eligible(row: Mapping[str, Any], exclude: frozenset[str]) -> bool:
    qid, suite = row.get("id"), row.get("suite")
    if not qid or suite not in STRATA or qid in exclude:
        return False
    if row.get("image_path"):
        return False
    if len(row.get("prompt") or "") > QUESTION_CAP_CHARS:
        return False
    return len(row.get("context") or "") <= EVIDENCE_CAP_CHARS


def _order_key(seed: str, qid: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}\x00{qid}".encode("utf-8")).hexdigest(), qid


def build_splits(rows: Iterable[Mapping[str, Any]], *, seed: str = SPLIT_SEED,
                 exclude_qids: Iterable[str] = (), frozen_at: str = "") -> FrozenSplits:
    """Stratified, seeded, disjoint draw. Each stratum's eligible qids are ordered by
    sha256(seed, qid) and cut into consecutive slices in :data:`PER_STRATUM` order. A stratum
    short of its total refuses (a re-freeze decision, never a silent shrink)."""
    exclude = frozenset(exclude_qids)
    by_stratum: dict[str, set[str]] = {s: set() for s in STRATA}
    for row in rows:
        if _eligible(row, exclude):
            by_stratum[row["suite"]].add(str(row["id"]))
    need = sum(n for _, n in PER_STRATUM)
    picked: dict[str, set[str]] = {role: set() for role, _ in PER_STRATUM}
    for stratum in STRATA:
        ordered = sorted(by_stratum[stratum], key=lambda q: _order_key(seed, q))
        if len(ordered) < need:
            raise ValueError(f"stratum {stratum!r}: {len(ordered)} eligible < {need} needed")
        at = 0
        for role, n in PER_STRATUM:
            picked[role].update(ordered[at:at + n])
            at += n
    manifest = {
        "seed": seed, "strata": list(STRATA), "per_stratum": [list(p) for p in PER_STRATUM],
        "caps": [QUESTION_CAP_CHARS, EVIDENCE_CAP_CHARS], "excluded": len(exclude),
        "splits": {role: sorted(q) for role, q in picked.items()},
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                            .encode("utf-8")).hexdigest()
    tag = digest[:12]

    def ref(role: str, name: str, split_role: str, arrived: str) -> SplitRef:
        return SplitRef(f"hil6i-{name}@{tag}", split_role, frozenset(picked[role]),
                        arrived_at=arrived)

    out = FrozenSplits(
        train=ref("train", "T1", "train", frozen_at),
        validation=ref("validation", "V", "validation", frozen_at),
        arrival=ref("arrival", "T2", "arrival", ""),  # sealed until released
        arrival_2=ref("arrival_2", "T3", "arrival", ""),
        test=ref("test", "TEST", "test", frozen_at),
        seed=seed, manifest_sha256=digest,
    )
    if not splits_disjoint(out.train, out.validation, out.arrival, out.arrival_2, out.test):
        raise AssertionError("split slices overlap")  # unreachable: slices are disjoint by construction
    return out


# ── the design ────────────────────────────────────────────────────────────────


def first_feature_design(splits: FrozenSplits, *, cycle: int = 1) -> PilotDesign:
    """The filled design. Cycle 1 tunes on T1 with arrival T2; cycle 2 re-tunes on T1+T2 from the
    cycle-1 harness with arrival T3 (P1). Both designs share validation and test."""
    if cycle == 1:
        train, arrival = splits.train, splits.arrival
    elif cycle == 2:
        train = SplitRef(f"{splits.train.split_id}+{splits.arrival.split_id}", "train",
                         splits.train.qids | splits.arrival.qids,
                         arrived_at=splits.arrival.arrived_at)
        arrival = splits.arrival_2
    else:
        raise ValueError(f"cycle must be 1 or 2, got {cycle!r}")
    return PilotDesign(
        feature_id=FEATURE_ID,
        home=HOME,
        primary_metric=PRIMARY_METRIC,
        primary_unit=PRIMARY_UNIT,
        primary_direction=PRIMARY_DIRECTION,
        train=train,
        validation=splits.validation,
        arrival=arrival,
        test=splits.test,
        cycles=CYCLES,
        stopping_rule=STOPPING_RULE,
        local_editor_model=LOCAL_EDITOR_MODEL,
        reference_editor_model=REFERENCE_EDITOR_MODEL,
        judge_model=JUDGE_MODEL,
        matched_rollout_baseline=True,
        unmodified_control=True,
        verification_arms=frozenset({"with", "none", "self_judged"}),
        runs=RUNS,
        mde=MDE,
        paired_n=PAIRED_N,
        expected_discordant=EXPECTED_DISCORDANT,
        min_win_share=MIN_WIN_SHARE,
        token_seats=TOKEN_SEATS,
        headroom_note=HEADROOM_NOTE,
        exclusions_declared=True,
        regression_set_id=REGRESSION_SET_ID,
        per_role_arm_set=True,
        reviewer_model="",
        served_local_models=SERVED_LOCAL_MODELS,
        context_policy_controls=frozenset(),
        stream_positions=(),
        alpha_budget=ALPHA_BUDGET,
        notes={**NOTES, "splits.manifest_sha256": splits.manifest_sha256,
               "splits.seed": splits.seed, "cycle": str(cycle)},
    )


# ── protected layer of the target (HIL-1 §3.5, item 21) ───────────────────────

#: Template slots filled by build_architect_investigate_prompt; each must survive exactly.
PLACEHOLDERS = ("{valid_roles_section}", "{context_section}", "{question}")
#: The decision grammar the parser reads (D|<answer>, I|brief:<text>|to:<role>).
GRAMMAR_MARKERS = ("D|", "I|brief:")
FINAL_ANCHOR = "Decision:"
_ROLE_BINDING = re.compile(r"\bto:([A-Za-z_][A-Za-z0-9_]*)")
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def protected_layer_reason(reference_text: str, mutated_text: str) -> str | None:
    """Refusal reason when a candidate touches the protected layer, else None.

    Protected: the template placeholders (count unchanged, none added), the decision grammar
    markers, the set of ``to:<role>`` bindings (role choice is routing policy, HIL-1 §3.5) and
    the final ``Decision:`` anchor. The wording around them is the mutable surface.
    ``reference_text`` is the unmodified control (the bytes pinned at freeze), not the current
    incumbent, so the layer cannot drift across cycles.
    """
    for ph in PLACEHOLDERS:
        if mutated_text.count(ph) != reference_text.count(ph):
            return f"protected_layer:placeholder:{ph}"
    added = set(_PLACEHOLDER.findall(mutated_text)) - set(_PLACEHOLDER.findall(reference_text))
    if added:
        return f"protected_layer:new_placeholder:{sorted(added)[0]}"
    for marker in GRAMMAR_MARKERS:
        if marker in reference_text and marker not in mutated_text:
            return f"protected_layer:grammar:{marker}"
    if set(_ROLE_BINDING.findall(mutated_text)) != set(_ROLE_BINDING.findall(reference_text)):
        return "protected_layer:role_binding"
    if reference_text.rstrip().endswith(FINAL_ANCHOR) and not mutated_text.rstrip().endswith(FINAL_ANCHOR):
        return "protected_layer:final_anchor"
    return None


def make_candidate_screen(reference_text: str,
                          leakage: Callable[..., str | None] | None = None) -> Callable[..., str | None]:
    """A text screen for the driver's ``leakage`` seam: the protected layer first, then the MHS-3
    leakage check (``substrate.leakage_reason`` by default: the real fail-closed guard since seam
    S2). The driver's ``original=`` / ``trace_context=`` keywords are passed through to it."""
    if leakage is None:
        from src.harness_loop import substrate

        leakage = substrate.leakage_reason

    def screen(mutated_text: str, **kw: Any) -> str | None:
        return protected_layer_reason(reference_text, mutated_text) or leakage(mutated_text, **kw)

    return screen
