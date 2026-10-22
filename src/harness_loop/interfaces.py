"""The sibling loop's OWN parts: proposer, evaluator, judge, decision rule, and their data.

These are interfaces and frozen value types only. No concrete proposer, evaluator or judge lives
in this package: each one reaches a model, and running a loop is deferred (2026-09-16). The value
types carry the HIL-2 checklist fields that are write-side and cannot be recovered later
(item 14 arm identity, item 16 token split, item 12 treatment-fired count, item 7 fence state).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

# ── HIL-1: feature homes and the policy-document boundary ──────────────────────

#: HS-4 section-3 homes with a loop-mutable measured arm (HIL-1). 3.7 skills and 3.8 "not
#: rebuilt" have none; a candidate naming them is refused.
MUTABLE_HOMES: Mapping[str, str] = {
    "3.1": "user-profile memory",
    "3.2": "agent notes",
    "3.3": "session search",
    "3.4": "background review",
    "3.5": "delegation",
    "3.6": "compaction",
}
#: Memory-accumulating homes: HIL-2 item 20 applies at evaluation (HIL-6 pilot requirement 4).
MEMORY_ACCUMULATING_HOMES = frozenset({"3.1", "3.2", "3.4"})
#: Policy documents are proposal-only and human-applied (HIL-1); never a mutation target.
POLICY_DOCUMENT_NAMES = frozenset({"AGENTS.md", "SKILL.md", "HARNESS_RUN_POLICY.md", "CLAUDE.md"})

TARGET_KINDS = frozenset({"prompt", "parameter"})


def policy_document_reason(target_path: str) -> str | None:
    """Refusal reason when ``target_path`` is a policy document or inside a skill folder."""
    p = PurePosixPath(str(target_path))
    if p.name in POLICY_DOCUMENT_NAMES:
        return f"policy_document:{p.name}"
    if any(part.lower() in {"skills", ".skills"} for part in p.parts[:-1]):
        return f"skill_folder:{target_path}"
    return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── splits (items 3, 7, 9) ─────────────────────────────────────────────────────

SPLIT_ROLES = frozenset({"train", "validation", "test", "arrival"})


@dataclass(frozen=True)
class SplitRef:
    """A frozen question set. ``arrived_at`` is when its items became visible (item 9); labels
    are never visible to the proposer before ``labels_resolved_at`` (item 7)."""

    split_id: str
    role: str
    qids: frozenset[str]
    arrived_at: str = ""
    labels_resolved_at: str = ""

    def __post_init__(self) -> None:
        if self.role not in SPLIT_ROLES:
            raise ValueError(f"split role {self.role!r} not in {sorted(SPLIT_ROLES)}")
        if not self.qids:
            raise ValueError(f"split {self.split_id!r} is empty")


def splits_disjoint(*splits: SplitRef) -> bool:
    seen: set[str] = set()
    for s in splits:
        if seen & s.qids:
            return False
        seen |= s.qids
    return True


# ── proposer view (item 7: index-style, no outcome) ────────────────────────────

#: Keys that would leak an outcome/label to the proposer (item 7: "the proposer's trajectory view
#: carries no success, score, feedback or judge claim"). Matched case-insensitively on any key at
#: any depth of a trace record.
OUTCOME_KEYS = frozenset({
    "score", "scores", "success", "passed", "pass", "correct", "is_correct", "reward",
    "feedback", "judge", "verdict", "label", "labels", "gold", "answer", "expected",
    "expected_answer", "reference", "rubric", "accuracy", "outcome",
})


def outcome_keys_in(record: Any, _path: str = "") -> list[str]:
    """Paths of every outcome-bearing key inside ``record`` (empty list == clean)."""
    hits: list[str] = []
    if isinstance(record, Mapping):
        for k, v in record.items():
            key = str(k)
            here = f"{_path}.{key}" if _path else key
            if key.lower() in OUTCOME_KEYS:
                hits.append(here)
            hits.extend(outcome_keys_in(v, here))
    elif isinstance(record, (list, tuple)):
        for i, v in enumerate(record):
            hits.extend(outcome_keys_in(v, f"{_path}[{i}]"))
    return hits


@dataclass(frozen=True)
class TraceRef:
    """One trajectory as the proposer sees it: an index entry, never an outcome.

    ``events`` are mechanical event names (L2 scan categories such as ``tool_error`` or
    ``truncation``; HIL-12) or tool names. The rule that selected which traces appear is declared
    on the view, not inferred from a per-trace outcome field.
    """

    trace_id: str
    role: str
    events: tuple[str, ...] = ()
    excerpt: str = ""


@dataclass(frozen=True)
class ProposerView:
    feature_id: str
    home: str
    target_path: str
    target_kind: str
    incumbent_text: str
    traces: tuple[TraceRef, ...] = ()
    selection_rule: str = ""
    rejected_history: tuple[str, ...] = ()  # AP-53 diffs for this target (no scores)

    def __post_init__(self) -> None:
        if self.home not in MUTABLE_HOMES:
            raise ValueError(f"home {self.home!r} has no loop-mutable arm (HIL-1)")
        if self.target_kind not in TARGET_KINDS:
            raise ValueError(f"target_kind {self.target_kind!r} not in {sorted(TARGET_KINDS)}")
        reason = policy_document_reason(self.target_path)
        if reason:
            raise ValueError(f"HIL-1: {reason} is proposal-only, never a loop target")
        if self.traces and not self.selection_rule:
            raise ValueError("item 7: a view with traces must declare its selection rule")

    def context_texts(self) -> list[str]:
        """Every string handed to the proposer (the tripwire input)."""
        texts = [self.incumbent_text, self.selection_rule, *self.rejected_history]
        for t in self.traces:
            texts.extend([t.excerpt, *t.events])
        return texts


# ── token accounting (items 2, 16) ─────────────────────────────────────────────


@dataclass(frozen=True)
class TokenCount:
    """Uncached prefill + decode tokens with the cached share beside them (item 16). ``None`` is
    "not captured", never zero (HIL-20 addendum item 2)."""

    uncached_prefill: int | None = None
    decode: int | None = None
    cached_prefill: int | None = None
    seat: str = ""  # proposer | meta | summariser | judge | solver (item 2: all seats counted)

    def total(self) -> int | None:
        parts = (self.uncached_prefill, self.decode, self.cached_prefill)
        if any(p is None for p in parts):
            return None
        return int(sum(parts))  # type: ignore[arg-type]


# ── candidates and arm results ────────────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    feature_id: str
    home: str
    target_path: str
    target_kind: str
    original_text: str
    mutated_text: str
    proposer_model: str
    rationale: str = ""  # item 17c: an untrusted self-report; never evidence


@dataclass(frozen=True)
class ProposalResult:
    candidates: tuple[Candidate, ...]
    tokens: tuple[TokenCount, ...] = ()


@dataclass(frozen=True)
class ArmSpec:
    """One arm: the incumbent or a candidate, pinned by bytes (items 6, 14)."""

    arm_id: str
    role: str  # incumbent | candidate | unmodified_control | matched_rollout_baseline
    feature_id: str
    target_path: str
    content_sha256: str
    pins: Mapping[str, Any] = field(default_factory=dict)  # HIL-2 item 14 arm_pins


ARM_ROLES = frozenset({"incumbent", "candidate", "unmodified_control", "matched_rollout_baseline"})


@dataclass(frozen=True)
class ArmResult:
    arm_id: str
    split_id: str
    metric: str
    value: float | None
    unit: str
    direction: str  # higher_better | lower_better
    scored: int
    attempted: int
    item_outcomes: Mapping[str, bool | None] = field(default_factory=dict)
    tokens: tuple[TokenCount, ...] = ()
    treatment_fired: int | None = None  # item 12
    fence_state: str = ""  # item 7: must be "active"
    infra_fingerprint: Mapping[str, Any] | None = None
    verification_signal: str = "with"  # item 4/17d: with | none | self_judged


# ── protocols ─────────────────────────────────────────────────────────────────


@runtime_checkable
class Proposer(Protocol):
    model_id: str

    def propose(self, view: ProposerView, *, max_candidates: int) -> ProposalResult: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, arm: ArmSpec, split: SplitRef) -> ArmResult: ...


VERDICT_VALUES = ("credible", "not_credible", "unresolved")


@dataclass(frozen=True)
class VerdictMatrix:
    """Item 17b: candidate x instance verdicts, no empty cell, written before choosing."""

    judge_model: str
    cells: Mapping[str, Mapping[str, str]]  # candidate_id -> instance_id -> verdict
    chosen: str | None = None
    tokens: tuple[TokenCount, ...] = ()  # item 2: judge seat counted

    def validate(self, candidate_ids: Sequence[str], instance_ids: Sequence[str]) -> None:
        for c in candidate_ids:
            row = self.cells.get(c)
            if row is None:
                raise ValueError(f"item 17b: no verdict row for candidate {c!r}")
            for i in instance_ids:
                v = row.get(i)
                if v not in VERDICT_VALUES:
                    raise ValueError(f"item 17b: cell ({c!r},{i!r}) = {v!r}")

    def credible_counts(self) -> dict[str, int]:
        return {c: sum(1 for v in row.values() if v == "credible") for c, row in self.cells.items()}

    def choice_is_argmax(self) -> bool:
        """Zero-LLM check (item 17b): the named choice is an argmax of its own counts."""
        if self.chosen is None:
            return True
        counts = self.credible_counts()
        if self.chosen not in counts:
            return False
        return counts[self.chosen] == max(counts.values())


@runtime_checkable
class Judge(Protocol):
    model_id: str

    def judge(self, candidates: Sequence[Candidate], instances: Sequence[str]) -> VerdictMatrix: ...


@dataclass(frozen=True)
class Decision:
    """A keep/revert outcome. ``claim_grade`` is ``observation`` unless a cited, human-ratified
    protocol applies (MEASUREMENT.md; HIL-2 protocol governance)."""

    keep: bool
    reason: str
    protocol_id: str | None = None

    @property
    def claim_grade(self) -> str:
        return "protocol" if self.protocol_id else "observation"


@runtime_checkable
class DecisionRule(Protocol):
    def decide(self, incumbent: ArmResult, candidate: ArmResult) -> Decision: ...


def sign_test_p(wins: int, discordant: int) -> float:
    """One-sided exact sign test: P(X >= wins | Binomial(discordant, 1/2)); 1.0 when no pairs."""
    if discordant <= 0:
        return 1.0
    from math import comb

    return sum(comb(discordant, k) for k in range(max(wins, 0), discordant + 1)) / 2**discordant


@dataclass(frozen=True)
class PairedWinShare:
    """Paired decision over per-item outcomes: an exact one-sided sign test on discordant items.

    Only items BOTH arms scored count (``None`` outcomes are excluded). A win is an item the
    candidate passes and the incumbent fails; a loss is the reverse. The candidate is kept only when
    the sign-test tail ``P(X >= wins | Bin(D, 1/2)) <= alpha`` AND ``wins / D >= min_win_share``;
    anything else, ties included, goes to the incumbent (item 17a). ``alpha`` is the attempt's
    reserved share of the ledger budget (item 21). Not a protocol: decisions are observations.
    """

    alpha: float
    min_win_share: float = 0.5

    def decide(self, incumbent: ArmResult, candidate: ArmResult) -> Decision:
        if incumbent.metric != candidate.metric or incumbent.direction != candidate.direction:
            return Decision(False, "metric_mismatch")
        paired = [
            (bool(incumbent.item_outcomes[q]), bool(candidate.item_outcomes[q]))
            for q in sorted(set(incumbent.item_outcomes) & set(candidate.item_outcomes))
            if incumbent.item_outcomes[q] is not None and candidate.item_outcomes[q] is not None
        ]
        if not paired:
            return Decision(False, "no_paired_items")
        wins = sum(1 for inc, cand in paired if cand and not inc)
        losses = sum(1 for inc, cand in paired if inc and not cand)
        discordant = wins + losses
        p = sign_test_p(wins, discordant)
        share = wins / discordant if discordant else 0.0
        detail = f"paired_wins:{wins}/{discordant} n:{len(paired)} p:{p:.3g} alpha:{self.alpha:g}"
        if discordant and p <= self.alpha and share >= self.min_win_share:
            return Decision(True, f"candidate_better:{detail}")
        return Decision(False, f"incumbent_holds:{detail}")


@dataclass(frozen=True)
class IncumbentWinsTies:
    """Item 17a: the incumbent carries the burden of proof; ties (and anything inside the declared
    margin) go to the incumbent. Not a protocol: its decisions are observations."""

    margin: float = 0.0

    def decide(self, incumbent: ArmResult, candidate: ArmResult) -> Decision:
        if incumbent.metric != candidate.metric or incumbent.direction != candidate.direction:
            return Decision(False, "metric_mismatch")
        if incumbent.value is None or candidate.value is None:
            return Decision(False, "value_missing")
        delta = candidate.value - incumbent.value
        if candidate.direction == "lower_better":
            delta = -delta
        if delta > self.margin:
            return Decision(True, f"candidate_better_by:{delta:g}>margin:{self.margin:g}")
        return Decision(False, f"incumbent_holds:delta:{delta:g}<=margin:{self.margin:g}")
