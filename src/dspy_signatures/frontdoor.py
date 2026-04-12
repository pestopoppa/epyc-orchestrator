"""DSPy Signature for frontdoor classification (AP-18).

Maps to the orchestrator's frontdoor.md prompt which receives a user
request and produces a TaskIR JSON routing plan.

Source prompt: orchestration/prompts/frontdoor.md
Existing code: src/prompt_builders/resolver.py
"""

from __future__ import annotations

import dspy


class FrontdoorClassifier(dspy.Signature):
    """Classify a user request and produce a routing plan.

    Given a user's natural language request and the available agent roles,
    determine the task type, optimal specialist role, priority level,
    and success criteria. This maps to the orchestrator's TaskIR schema.
    """

    user_prompt: str = dspy.InputField(desc="User's natural language request")
    context: str = dspy.InputField(
        desc="Prior conversation context (empty for first turn)",
        default="",
    )
    available_roles: str = dspy.InputField(
        desc="JSON list of available agent roles with tier and capabilities",
    )

    task_type: str = dspy.OutputField(
        desc="One of: chat, code, math, ingest, vision, manage, research",
    )
    primary_role: str = dspy.OutputField(
        desc="Best specialist role for this request (e.g. coder, architect, math)",
    )
    priority: str = dspy.OutputField(
        desc="'interactive' (user waiting) or 'batch' (async OK)",
    )
    objective: str = dspy.OutputField(
        desc="Clear statement of what constitutes success for this request",
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of routing decision",
    )


def create_module(use_cot: bool = False) -> dspy.Module:
    """Create a DSPy module for frontdoor classification.

    Args:
        use_cot: If True, use ChainOfThought for reasoning trace.

    Returns:
        A dspy.Predict or dspy.ChainOfThought module.
    """
    if use_cot:
        return dspy.ChainOfThought(FrontdoorClassifier)
    return dspy.Predict(FrontdoorClassifier)
