"""BSV-3: conflict-aware acceptance — mutation-dependency ledger + semantic-conflict classifier.

Per `handoffs/active/autopilot-continuous-optimization.md` BSV-3 (intake-607 §5.2.4). When two
independently-accepted autopilot mutations touch the same subsystem, syntactic composition can
produce a *semantically* inconsistent harness ("syntactically mergeable, behaviorally
inconsistent"). This module records accepted mutations and flags potential conflicts for review
rather than blind-composing them.

Pure (no autopilot/disk imports); the autopilot accept-path constructs MutationRecords and
consults the ledger before composing a new mutation onto the live config.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ConflictLevel:
    NONE = "none"        # disjoint surfaces, independent — safe to compose
    REVIEW = "review"    # shared mutable surface — flag for human/coordinated review
    BLOCK = "block"      # opposing improvements / direct regression on shared state — do not auto-compose

    RANK = {NONE: 0, REVIEW: 1, BLOCK: 2}


@dataclass
class MutationRecord:
    """An accepted (or proposed) mutation, keyed for conflict detection (BSV-3)."""

    mutation_id: str
    parent_trial: int | None = None
    subsystem: str | None = None              # e.g. 'prompt', 'routing', 'tool_policy', 'context_packer'
    files_touched: set[str] = field(default_factory=set)
    prompt_sections_touched: set[str] = field(default_factory=set)
    feature_flags: set[str] = field(default_factory=set)
    behavior_signature_hash: str | None = None
    improved_sentinels: set[str] = field(default_factory=set)
    regressed_sentinels: set[str] = field(default_factory=set)
    accepted: bool = True

    def __post_init__(self) -> None:
        # tolerate list inputs (JSON-decoded) → coerce to sets
        self.files_touched = set(self.files_touched)
        self.prompt_sections_touched = set(self.prompt_sections_touched)
        self.feature_flags = set(self.feature_flags)
        self.improved_sentinels = set(self.improved_sentinels)
        self.regressed_sentinels = set(self.regressed_sentinels)


def _shared_surface(a: MutationRecord, b: MutationRecord) -> dict[str, set[str]]:
    return {
        "files": a.files_touched & b.files_touched,
        "prompt_sections": a.prompt_sections_touched & b.prompt_sections_touched,
        "feature_flags": a.feature_flags & b.feature_flags,
    }


def classify_mutation_conflict(a: MutationRecord, b: MutationRecord) -> tuple[str, list[str]]:
    """Classify the conflict risk of composing mutations a and b. Returns (level, reasons)."""
    if a.mutation_id == b.mutation_id:
        return ConflictLevel.NONE, ["same mutation"]

    reasons: list[str] = []
    level = ConflictLevel.NONE

    def bump(new_level: str, reason: str) -> None:
        nonlocal level
        reasons.append(reason)
        if ConflictLevel.RANK[new_level] > ConflictLevel.RANK[level]:
            level = new_level

    shared = _shared_surface(a, b)
    any_shared_surface = any(shared.values())
    same_subsystem = a.subsystem is not None and a.subsystem == b.subsystem

    for kind, overlap in shared.items():
        if overlap:
            bump(ConflictLevel.REVIEW, f"shared {kind}: {sorted(overlap)}")
    if same_subsystem and not any_shared_surface:
        bump(ConflictLevel.REVIEW, f"same subsystem '{a.subsystem}' (no explicit file/flag overlap)")

    # Direct opposition: one improves a sentinel the other regresses → blocking.
    crossed = (a.improved_sentinels & b.regressed_sentinels) | (b.improved_sentinels & a.regressed_sentinels)
    if crossed:
        bump(ConflictLevel.BLOCK, f"direct opposition on sentinels {sorted(crossed)} (one improves, other regresses)")

    # Opposing improvements on shared state: both improve, but DIFFERENT sentinels, while
    # sharing a mutable surface and producing different behavior signatures → semantic conflict.
    if (any_shared_surface or same_subsystem) and a.improved_sentinels and b.improved_sentinels:
        if a.improved_sentinels.isdisjoint(b.improved_sentinels):
            if a.behavior_signature_hash != b.behavior_signature_hash:
                bump(
                    ConflictLevel.BLOCK,
                    "opposing improvements: disjoint improved sentinels + divergent behavior "
                    "signatures on shared surface",
                )

    if not reasons:
        reasons.append("disjoint surfaces, independent subsystems")
    return level, reasons


class MutationLedger:
    """Append-only ledger of accepted mutations with conflict lookup (BSV-3)."""

    def __init__(self) -> None:
        self._records: list[MutationRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def register(self, record: MutationRecord) -> list[tuple[str, str, list[str]]]:
        """Check `record` against prior ACCEPTED records, then store it.

        Returns a list of (other_mutation_id, level, reasons) for every prior accepted
        mutation with a non-NONE conflict. Storing happens regardless (the ledger is the
        audit trail); the caller decides whether to compose, gate, or escalate.
        """
        conflicts: list[tuple[str, str, list[str]]] = []
        for other in self._records:
            if not other.accepted:
                continue
            level, reasons = classify_mutation_conflict(record, other)
            if level != ConflictLevel.NONE:
                conflicts.append((other.mutation_id, level, reasons))
        self._records.append(record)
        return conflicts

    def worst_level(self, conflicts: list[tuple[str, str, list[str]]]) -> str:
        if not conflicts:
            return ConflictLevel.NONE
        return max((c[1] for c in conflicts), key=lambda lv: ConflictLevel.RANK[lv])

    def records(self) -> list[MutationRecord]:
        return list(self._records)
