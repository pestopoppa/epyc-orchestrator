"""DSPy Signature for mode selection and routing (AP-18).

Maps to the orchestrator's _select_mode() and _classify_and_route()
functions in src/api/routes/chat_routing.py.

These are currently zero-latency keyword heuristics. The DSPy Signature
enables future LLM-augmented routing with optimization via GEPA.
"""

from __future__ import annotations

import dspy


class ModeSelector(dspy.Signature):
    """Select execution mode (direct vs REPL) and specialist role for a request.

    Given the user's prompt and context, determine whether to use direct
    generation or REPL-based exploration, and which specialist role is best.
    """

    prompt: str = dspy.InputField(desc="User's prompt text")
    context: str = dspy.InputField(
        desc="Conversation context (prior messages)",
        default="",
    )
    has_image: bool = dspy.InputField(
        desc="Whether the request includes image data",
        default=False,
    )
    prompt_length: int = dspy.InputField(
        desc="Character count of the prompt",
    )

    mode: str = dspy.OutputField(
        desc="'direct' (single generation) or 'repl' (tool-augmented exploration)",
    )
    role: str = dspy.OutputField(
        desc="Specialist role to route to (e.g. coder, math, architect, frontdoor)",
    )
    confidence: float = dspy.OutputField(
        desc="Routing confidence 0.0-1.0",
    )
    reasoning: str = dspy.OutputField(
        desc="Brief routing rationale",
    )


def create_module(use_cot: bool = False) -> dspy.Module:
    """Create a DSPy module for mode selection."""
    if use_cot:
        return dspy.ChainOfThought(ModeSelector)
    return dspy.Predict(ModeSelector)
