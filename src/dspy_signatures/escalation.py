"""DSPy Signature for escalation decisions (AP-18).

Maps to the orchestrator's rule-based EscalationPolicy.decide().
The DSPy Signature enables future A/B testing: rule-based vs
LLM-augmented escalation with the same interface.

Source code: src/escalation.py (EscalationContext, EscalationPolicy.decide())
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

if TYPE_CHECKING:
    from src.escalation import EscalationContext


class EscalationDecider(dspy.Signature):
    """Decide whether to retry, escalate, delegate, or fail after a role failure.

    Given the current failure context (which role failed, how many times,
    what kind of error), determine the best recovery action and target role.
    """

    current_role: str = dspy.InputField(
        desc="Role that failed (e.g. 'coder', 'architect', 'frontdoor')",
    )
    error_category: str = dspy.InputField(
        desc="Error type: schema, timeout, early_abort, gate_failure, unknown",
    )
    error_message: str = dspy.InputField(
        desc="Error details or empty string",
        default="",
    )
    failure_count: int = dspy.InputField(
        desc="Number of failures at this role for this task",
    )
    escalation_count: int = dspy.InputField(
        desc="Previous escalations for this task",
    )

    action: str = dspy.OutputField(
        desc="One of: retry, think_harder, escalate, delegate, review, fail, skip, explore",
    )
    target_role: str = dspy.OutputField(
        desc="Target role for escalation/delegation (empty if retry/fail)",
    )
    reasoning: str = dspy.OutputField(
        desc="Why this action was chosen",
    )


def from_escalation_context(ctx: "EscalationContext") -> dict:
    """Convert an existing EscalationContext dataclass to DSPy Signature inputs.

    Usage:
        inputs = from_escalation_context(ctx)
        result = dspy.Predict(EscalationDecider)(**inputs)
    """
    return {
        "current_role": str(ctx.current_role),
        "error_category": str(ctx.error_category),
        "error_message": ctx.error_message,
        "failure_count": ctx.failure_count,
        "escalation_count": ctx.escalation_count,
    }


def create_module(use_cot: bool = False) -> dspy.Module:
    """Create a DSPy module for escalation decisions."""
    if use_cot:
        return dspy.ChainOfThought(EscalationDecider)
    return dspy.Predict(EscalationDecider)
