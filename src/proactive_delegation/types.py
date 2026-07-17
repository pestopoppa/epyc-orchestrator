"""Data types for proactive delegation workflow."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Custom Exceptions ─────────────────────────────────────────────────


class DelegationError(Exception):
    """Base exception for proactive delegation failures."""


class ArchitectPlanError(DelegationError):
    """Architect failed to generate a valid plan."""


class StepExecutionError(DelegationError):
    """A delegation step failed execution."""

    def __init__(self, step_id: str, role: str, cause: Exception | None = None):
        self.step_id = step_id
        self.role = role
        self.cause = cause
        msg = f"Step {step_id} ({role}) failed"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


# ── Enums ─────────────────────────────────────────────────────────────


class ReviewDecision(Enum):
    """Reviewer's bounded-authority decision.

    Mirrors the ``decision`` enum in ``orchestration/review_decision.schema.json``.
    REQUEST_EVIDENCE and REJECT_TO_EMPTY (RA-6) are additive — existing consumers
    that branch on APPROVE/REJECT/REQUEST_CHANGES/ESCALATE keep working; unhandled
    members fall through their else-branches (verified in delegator.py /
    parallel_step_executor.py).
    """

    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"
    REJECT = "reject"
    # RA-6 additions (evidence-linked control plane):
    REQUEST_EVIDENCE = "request_evidence"  # verdict withheld pending verifier_requests
    REJECT_TO_EMPTY = "reject_to_empty"  # bad plan/output worse than none; discard, don't iterate


class TaskComplexity(Enum):
    """Task complexity level determining delegation path."""

    TRIVIAL = "trivial"  # Frontdoor answers directly (factual, chat)
    SIMPLE = "simple"  # Frontdoor executes in REPL (single code task)
    MODERATE = "moderate"  # Frontdoor delegates to single specialist
    COMPLEX = "complex"  # Architect generates TaskIR, multi-specialist


@dataclass
class ComplexitySignals:
    """Signals used to estimate task complexity and routing."""

    word_count: int = 0
    has_code_keywords: bool = False
    has_multi_step_keywords: bool = False
    has_architecture_keywords: bool = False
    question_type: str = "unknown"  # factual, how-to, implementation, design, architect_requested, thinking_requested
    estimated_files: int = 0
    # Escalation flags (orthogonal to complexity)
    thinking_requested: bool = False  # /think, ultrathink -> architect-grade reasoning
    architect_requested: bool = False  # /architect, /plan -> use architect for planning


def _deleg_cfg():
    from src.config import get_config

    return get_config().delegation


@dataclass
class IterationContext:
    """Track iteration state to prevent infinite loops.

    Attributes:
        max_iterations: Maximum review-fix cycles per subtask
        max_total_iterations: Maximum total iterations across all subtasks
        current_iteration: Current iteration count for active subtask
        total_iterations: Total iterations across all subtasks
        iteration_history: Log of iteration decisions
    """

    max_iterations: int = field(default_factory=lambda: _deleg_cfg().max_iterations)
    max_total_iterations: int = field(default_factory=lambda: _deleg_cfg().max_total_iterations)
    current_iteration: int = 0
    total_iterations: int = 0
    subtask_iterations: dict[str, int] = field(default_factory=dict)
    iteration_history: list[dict[str, Any]] = field(default_factory=list)
    # RD-10c sticky decision cache + RD-11/LB-5 shadow budget seam.
    # ``max_iterations`` / ``max_total_iterations`` above ARE the wire-points for
    # the ``max_review_iterations`` / ``max_total_review_iterations`` knobs
    # (LB-2 reuses IterationContext semantics — review turns count on the same
    # ledger, so a runaway Architect<->Reviewer handshake trips the existing cap).
    # All fields below default to inert so behavior is byte-identical until a knob
    # flips them on.
    decision_cache_enabled: bool = False
    decision_cache: dict[str, str] = field(default_factory=dict)
    budget_violations: list[dict[str, Any]] = field(default_factory=list)

    def can_iterate(self, subtask_id: str) -> bool:
        """Check if another iteration is allowed for this subtask."""
        subtask_count = self.subtask_iterations.get(subtask_id, 0)
        return (
            subtask_count < self.max_iterations
            and self.total_iterations < self.max_total_iterations
        )

    # ── RD-10c sticky decision cache ─────────────────────────────────────
    def subtask_signature(
        self,
        task: dict[str, Any] | None,
        subtask: dict[str, Any] | None,
        candidate: str | None,
    ) -> str:
        """Stable signature over the *sanitized* task shape + candidate shape.

        RD-10c keys the sticky cache on a hash of ``(task, subtask, candidate)``
        with volatile detail sanitized out: the objective/action are whitespace-
        normalized and truncated, and the candidate is reduced to a coarse
        *shape* fingerprint (digit-count length bucket + normalized head/tail)
        rather than its exact bytes — so an approved *pattern* can skip re-review
        for structurally-equivalent candidates within the same run/wave.
        """
        obj = " ".join(str((task or {}).get("objective", "")).lower().split())[:200]
        action = " ".join(str((subtask or {}).get("action", "")).lower().split())[:200]
        cand = str(candidate or "")
        n = len(cand)
        len_bucket = 0 if n == 0 else len(str(n))  # order-of-magnitude bucket
        head = " ".join(cand[:80].lower().split())
        tail = " ".join(cand[-80:].lower().split())
        payload = json.dumps(
            {
                "objective": obj,
                "action": action,
                "shape": {"len_bucket": len_bucket, "head": head, "tail": tail},
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cached_decision(self, signature: str) -> "ReviewDecision | None":
        """Return a sticky APPROVE for this signature, or None. Off by default."""
        if not self.decision_cache_enabled:
            return None
        value = self.decision_cache.get(signature)
        if value is None:
            return None
        try:
            return ReviewDecision(value)
        except ValueError:
            return None

    def remember_decision(self, signature: str, decision: "ReviewDecision") -> None:
        """Cache an APPROVE pattern. REJECT/REJECT_TO_EMPTY are NEVER sticky."""
        if not self.decision_cache_enabled:
            return
        if decision == ReviewDecision.APPROVE:
            self.decision_cache[signature] = decision.value

    # ── RD-11 / LB-5 shadow budget-check seam ────────────────────────────
    def check_token_budget(
        self,
        decision_type: str,
        *,
        tokens_used: int | None = None,
        token_budget: int | None = None,
        latency_ms: float | None = None,
        latency_budget_ms: float | None = None,
        actor: str = "",
    ) -> dict[str, Any] | None:
        """Record a per-decision token/latency budget breach — NEVER blocks.

        The LB-2 budgets (plan-review ≤350 tok, candidate-review ≤300,
        rubric-authoring ≤800 amortized, rubric-grading ≤180, plus their latency
        ceilings) are *observation-grade / proposed*. This hook is the enforcement
        seam: in the shadow era it appends a VIOLATION record and logs it, but it
        never raises and never alters control flow. Returns the record on breach,
        else None.
        """
        breaches: dict[str, Any] = {}
        if (
            token_budget is not None
            and tokens_used is not None
            and tokens_used > token_budget
        ):
            breaches["tokens"] = {"used": tokens_used, "budget": token_budget}
        if (
            latency_budget_ms is not None
            and latency_ms is not None
            and latency_ms > latency_budget_ms
        ):
            breaches["latency_ms"] = {"used": latency_ms, "budget": latency_budget_ms}
        if not breaches:
            return None
        record = {
            "decision_type": decision_type,
            "actor": actor,
            "breaches": breaches,
            "timestamp": datetime.now().isoformat(),
        }
        self.budget_violations.append(record)
        logger.info("review budget VIOLATION (shadow, non-blocking): %s", record)
        return record

    def record_iteration(
        self,
        subtask_id: str,
        decision: ReviewDecision,
        feedback: str | None = None,
    ) -> None:
        """Record an iteration for tracking."""
        self.subtask_iterations[subtask_id] = self.subtask_iterations.get(subtask_id, 0) + 1
        self.total_iterations += 1
        self.current_iteration = self.subtask_iterations[subtask_id]

        self.iteration_history.append(
            {
                "subtask_id": subtask_id,
                "iteration": self.current_iteration,
                "total": self.total_iterations,
                "decision": decision.value,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_summary(self) -> dict[str, Any]:
        """Get iteration summary for logging."""
        return {
            "total_iterations": self.total_iterations,
            "subtask_counts": dict(self.subtask_iterations),
            "max_reached": self.total_iterations >= self.max_total_iterations,
        }


@dataclass
class ArchitectReview:
    """Result of a reviewer adjudicating specialist output.

    RA-6 extends this with the evidence-linked control-plane fields while keeping
    every legacy field/default so existing consumers (review_service.py,
    delegator.py, parallel_step_executor.py, chat_review.py) are unaffected.

    Score vs confidence semantics (score-vs-confidence is an open operator
    decision, documented here for now):
      * ``score``      — advisory quality of the candidate in [0, 1]
                         (how good is the output). Legacy field, kept for compat.
      * ``confidence`` — the reviewer's calibrated confidence in its own VERDICT
                         in [0, 1] (how sure am I this decision is correct). Feeds
                         the FA/FR calibration ledger, not the quality signal.

    ``tripwire`` is the hard-stop channel (orthogonal to ``score``): a violated
    invariant blocks regardless of advisory score (safety_gate.py semantics).
    """

    subtask_id: str
    decision: ReviewDecision
    feedback: str = ""
    score: float = 0.0
    suggested_changes: list[str] = field(default_factory=list)
    approved_output: str | None = None
    # RA-6 evidence-linked control-plane fields (all optional / defaulted):
    confidence: float = 0.0
    tripwire: bool = False
    evidence: list[dict[str, Any]] = field(default_factory=list)
    verifier_requests: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Legacy keys are preserved verbatim; the new keys are additive so existing
        readers of the dict are unaffected.
        """
        return {
            "subtask_id": self.subtask_id,
            "decision": self.decision.value,
            "feedback": self.feedback,
            "score": self.score,
            "suggested_changes": self.suggested_changes,
            "approved_output": self.approved_output,
            "confidence": self.confidence,
            "tripwire": self.tripwire,
            "evidence": self.evidence,
            "verifier_requests": self.verifier_requests,
        }


@dataclass
class SubtaskResult:
    """Result from a specialist executing a subtask."""

    subtask_id: str
    role: str
    output: str
    success: bool
    error: str | None = None
    tokens_used: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class AggregatedResult:
    """Final aggregated result from multiple specialists."""

    task_id: str
    objective: str
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    aggregated_output: str = ""
    all_approved: bool = False
    total_iterations: int = 0
    roles_used: list[str] = field(default_factory=list)
    delegation_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "subtask_results": [
                {
                    "subtask_id": r.subtask_id,
                    "role": r.role,
                    "output": r.output[:500] + "..." if len(r.output) > 500 else r.output,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.subtask_results
            ],
            "aggregated_output": self.aggregated_output,
            "all_approved": self.all_approved,
            "total_iterations": self.total_iterations,
            "roles_used": self.roles_used,
            "delegation_events": self.delegation_events,
        }


@dataclass
class PlanReviewResult:
    """Result of architect reviewing a plan before execution."""

    decision: str = "ok"
    score: float = 1.0
    feedback: str = ""
    patches: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""

    @property
    def is_ok(self) -> bool:
        """True if architect approved the plan without changes."""
        return self.decision == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision": self.decision,
            "score": self.score,
            "feedback": self.feedback,
            "patches": self.patches,
        }
