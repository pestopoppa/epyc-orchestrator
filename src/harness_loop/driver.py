"""Sibling-loop cycle driver skeleton (HIL-6 option B). Pure orchestration over injected parts.

``run_cycle`` composes the shared substrate (guards, AP-53 ledger, fence, fingerprint, journal)
with the sibling's own proposer/evaluator/judge/decision rule. It contains no model client, no
eval runner and no scheduler hook; with no concrete proposer or evaluator in the codebase it
cannot reach inference. A later, separately authorized change supplies those and the entrypoint.

One cycle (HIL-6 pilot requirement 1; HIL-2 items 3, 7, 9, 21):

1. tripwire every string handed to the proposer (answer-key markers + decoy-rubric canary);
2. propose; screen each candidate (HIL-1 policy boundary, MHS-4 risk gate, MHS-3 leakage
   (fail-closed until seam S2), MHS-15 lint recorded); rejections go to the LOOP's AP-53 ledger;
3. evaluate the incumbent and each surviving candidate on the VALIDATION split only, fence
   armed and echoed active on both arms, infra regimes compared (NON_COMPARABLE refuses);
4. one attempt-ledger row per candidate, alpha reserved before evaluation;
5. decide with the declared rule (incumbent wins ties); optional judge matrix + argmax check;
6. commit (or KEEP_BASE), then evaluate the committed arm on the ARRIVAL split with no tuning;
7. journal one ``hil_cycle_commit`` event with harness bytes, per-seat tokens and lint warnings.
The test split is touched once, by :meth:`SiblingLoopDriver.final_test`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.harness_loop import substrate
from src.harness_loop.attempt_ledger import AttemptLedger
from src.harness_loop.interfaces import (
    ArmResult,
    ArmSpec,
    Candidate,
    Decision,
    DecisionRule,
    Evaluator,
    IncumbentWinsTies,
    Judge,
    Proposer,
    ProposerView,
    SplitRef,
    TokenCount,
    VerdictMatrix,
    outcome_keys_in,
    policy_document_reason,
    sha256_text,
)
from src.harness_loop.pilot import PilotDesign, unmet_requirements

#: HIL-2 item 14 closed pin set; equal to ``scripts/benchmark/hil_ab_sidecar.ARM_PIN_FIELDS``
#: (a unit test asserts it, so the two cannot drift).
ARM_PIN_FIELDS = ("tool_surface_sha256", "served_context_window", "compaction_policy_sha256",
                  "wall_time_budget_s", "turn_budget", "final_result_selection_rule")
#: Seeded into the evaluator rubric; it must never reach proposer context (HIL-2 item 7).
DECOY_CANARY = "HIL-DECOY-RUBRIC-CANARY-5e1d"

CYCLE_EVENT = "hil_cycle_commit"
TEST_EVENT = "hil_test_touched"


class PilotDesignIncomplete(ValueError):
    def __init__(self, unmet: list[tuple[str, str]]):
        self.unmet = unmet
        super().__init__("; ".join(f"{k}: {v}" for k, v in unmet))


class TripwireTripped(RuntimeError):
    pass


class ResultRefused(ValueError):
    pass


def tripwire(texts: Sequence[str], markers: Sequence[str]) -> list[str]:
    """Markers (answer keys, label paths, the decoy canary) found in any proposer-bound text."""
    hits = []
    for m in markers:
        if m and any(m in (t or "") for t in texts):
            hits.append(m)
    return hits


@dataclass(frozen=True)
class CycleSpec:
    cycle_index: int
    tune_split: SplitRef
    incumbent: ArmSpec
    incumbent_text: str
    pins: Mapping[str, Any]
    max_candidates: int = 4
    alpha_per_attempt: float = 0.01


@dataclass
class CycleOutcome:
    cycle_index: int
    outcome: str  # keep | keep_base
    committed_arm_id: str
    committed_text: str
    validation: dict[str, ArmResult] = field(default_factory=dict)
    arrival: ArmResult | None = None
    decisions: dict[str, Decision] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    lint: dict[str, list[str]] = field(default_factory=dict)
    judge_argmax_ok: bool | None = None
    tokens_by_seat: dict[str, int | None] = field(default_factory=dict)


def _sum_tokens(by_seat: dict[str, int | None], counts: Sequence[TokenCount], default_seat: str) -> None:
    for c in counts:
        seat = c.seat or default_seat
        t = c.total()
        prev = by_seat.get(seat, 0)
        by_seat[seat] = None if (t is None or prev is None) else prev + t


class SiblingLoopDriver:
    def __init__(
        self,
        *,
        design: PilotDesign,
        journal: Any,
        journal_dir: Path | str,
        proposer: Proposer,
        evaluator: Evaluator,
        decision_rule: DecisionRule | None = None,
        judge: Judge | None = None,
        leakage: Callable[[str], str | None] = substrate.leakage_reason,
        answer_markers: Sequence[str] = (),
    ) -> None:
        unmet = unmet_requirements(design)
        if unmet:
            raise PilotDesignIncomplete(unmet)
        self.design = design
        self.journal = journal
        self.journal_dir = Path(journal_dir)
        self.proposer = proposer
        self.evaluator = evaluator
        self.decision_rule = decision_rule or IncumbentWinsTies()
        self.judge = judge
        self.leakage = leakage
        self.markers = tuple(answer_markers) + (DECOY_CANARY,)
        self.ledger = AttemptLedger(journal, alpha_budget=design.alpha_budget)

    # ── checks ────────────────────────────────────────────────────────────

    def _check_view(self, view: ProposerView) -> None:
        hits = tripwire(view.context_texts(), self.markers)
        if hits:
            raise TripwireTripped(f"item 7: proposer context carries {hits}")
        for t in view.traces:
            leaked = outcome_keys_in({"events": list(t.events)})
            if leaked:
                raise TripwireTripped(f"item 7: trace {t.trace_id} carries outcome keys {leaked}")

    def _check_arm(self, arm: ArmSpec) -> None:
        if sorted(arm.pins) != sorted(ARM_PIN_FIELDS):
            raise ResultRefused(f"item 14: arm {arm.arm_id} pins {sorted(arm.pins)}")

    def _check_result(self, arm: ArmSpec, split: SplitRef, r: ArmResult) -> ArmResult:
        substrate.require_fence_active({"state": r.fence_state})
        if r.arm_id != arm.arm_id or r.split_id != split.split_id:
            raise ResultRefused(f"result for {r.arm_id}/{r.split_id}, asked {arm.arm_id}/{split.split_id}")
        if r.treatment_fired is None:
            raise ResultRefused(f"item 12: arm {arm.arm_id} reported no treatment-fired count")
        if not r.tokens:
            raise ResultRefused(f"item 16: arm {arm.arm_id} reported no token columns")
        if r.scored > r.attempted:
            raise ResultRefused(f"arm {arm.arm_id}: scored {r.scored} > attempted {r.attempted}")
        return r

    def _screen(self, cand: Candidate, view: ProposerView) -> tuple[str | None, list[str]]:
        if cand.feature_id != view.feature_id or cand.target_path != view.target_path:
            return "target_mismatch", []
        if cand.original_text != view.incumbent_text:
            return "stale_original", []
        reason = policy_document_reason(cand.target_path)
        if reason:
            return reason, []
        lint = substrate.prompt_interference_lint(cand.original_text, cand.mutated_text)
        effect = substrate.classify_prompt_effect(cand.original_text, cand.mutated_text)
        reason = substrate.mutation_risk_gate_reason(effect)
        if reason:
            return reason, lint
        reason = self.leakage(cand.mutated_text)
        if reason:
            return reason, lint
        return None, lint

    def _evaluate(self, arm: ArmSpec, split: SplitRef) -> ArmResult:
        self._check_arm(arm)
        return self._check_result(arm, split, self.evaluator.evaluate(arm, split))

    # ── one cycle ─────────────────────────────────────────────────────────

    def run_cycle(self, view: ProposerView, spec: CycleSpec) -> CycleOutcome:
        d = self.design
        if spec.tune_split.qids & (d.validation.qids | d.arrival.qids | d.test.qids):
            raise ResultRefused("item 3: the tuning split overlaps validation/arrival/test")
        self._check_view(view)
        tokens: dict[str, int | None] = {}
        proposal = self.proposer.propose(view, max_candidates=spec.max_candidates)
        _sum_tokens(tokens, proposal.tokens, "proposer")

        out = CycleOutcome(spec.cycle_index, "keep_base", spec.incumbent.arm_id, spec.incumbent_text,
                           tokens_by_seat=tokens)
        survivors: list[Candidate] = []
        for cand in proposal.candidates:
            reason, lint = self._screen(cand, view)
            if lint:
                out.lint[cand.candidate_id] = lint
            if reason:
                out.rejected[cand.candidate_id] = reason
                substrate.append_rejection(
                    self.journal_dir, target=cand.target_path, mutation_type="hil_prompt",
                    artifact_kind=cand.target_kind, rejecting_gate=reason.split(":")[0],
                    gate_detail=reason, original_content=cand.original_text,
                    mutated_content=cand.mutated_text, safety_warnings=lint)
                continue
            survivors.append(cand)
        # MHS-4 ordering, safest first (stable: ties keep proposal order).
        survivors.sort(key=lambda c: substrate.mutation_effect_risk(
            substrate.classify_prompt_effect(c.original_text, c.mutated_text)))

        inc = self._evaluate(spec.incumbent, d.validation)
        out.validation[spec.incumbent.arm_id] = inc
        _sum_tokens(tokens, inc.tokens, "solver")
        best: tuple[float, Candidate] | None = None
        for cand in survivors:
            arm = ArmSpec(cand.candidate_id, "candidate", cand.feature_id, cand.target_path,
                          sha256_text(cand.mutated_text), dict(spec.pins))
            idx = self.ledger.open(cand.candidate_id, spec.alpha_per_attempt,
                                   cycle=spec.cycle_index, split=d.validation.split_id)
            res = self._evaluate(arm, d.validation)
            _sum_tokens(tokens, res.tokens, "solver")
            out.validation[arm.arm_id] = res
            comp = substrate.compare_infra_fingerprints(res.infra_fingerprint, inc.infra_fingerprint)
            if comp.get("status") == substrate.NON_COMPARABLE:
                out.decisions[arm.arm_id] = Decision(False, "infra_non_comparable")
                self.ledger.close(idx, "refused", infra=comp.get("status"))
                continue
            dec = self.decision_rule.decide(inc, res)
            out.decisions[arm.arm_id] = dec
            self.ledger.close(idx, "keep" if dec.keep else "keep_base", reason=dec.reason,
                              infra=comp.get("status"), claim_grade=dec.claim_grade)
            if dec.keep and res.value is not None and inc.value is not None:
                gain = res.value - inc.value
                if res.direction == "lower_better":
                    gain = -gain
                if best is None or gain > best[0]:
                    best = (gain, cand)

        if self.judge is not None and survivors:
            matrix: VerdictMatrix = self.judge.judge(survivors, sorted(d.validation.qids))
            matrix.validate([c.candidate_id for c in survivors], sorted(d.validation.qids))
            out.judge_argmax_ok = matrix.choice_is_argmax()
            _sum_tokens(tokens, matrix.tokens, "judge")

        committed_arm = spec.incumbent
        if best is not None:
            cand = best[1]
            out.outcome, out.committed_arm_id, out.committed_text = "keep", cand.candidate_id, cand.mutated_text
            committed_arm = ArmSpec(cand.candidate_id, "candidate", cand.feature_id, cand.target_path,
                                    sha256_text(cand.mutated_text), dict(spec.pins))
        out.arrival = self._evaluate(committed_arm, d.arrival)
        _sum_tokens(tokens, out.arrival.tokens, "solver")

        self.journal.append_ledger_event({
            "type": CYCLE_EVENT,
            "cycle": spec.cycle_index,
            "feature_id": d.feature_id,
            "outcome": out.outcome,
            "committed_arm_id": out.committed_arm_id,
            "committed_sha256": sha256_text(out.committed_text),
            "harness_bytes": len(out.committed_text.encode("utf-8")),
            "arrival": {"split": d.arrival.split_id, "metric": out.arrival.metric,
                        "value": out.arrival.value, "scored": out.arrival.scored,
                        "attempted": out.arrival.attempted},
            "rejected": dict(out.rejected),
            "lint": dict(out.lint),
            "tokens_by_seat": dict(tokens),
            "judge_argmax_ok": out.judge_argmax_ok,
            "decision_grade": "observation",
        })
        return out

    def final_test(self, arm: ArmSpec) -> ArmResult:
        """Item 3: the held-out test split is evaluated once, for the reported claim."""
        if self.journal.ledger_events(TEST_EVENT):
            raise ResultRefused("item 3: the test split was already touched")
        self.journal.append_ledger_event({"type": TEST_EVENT, "arm_id": arm.arm_id,
                                          "split": self.design.test.split_id})
        return self._evaluate(arm, self.design.test)
