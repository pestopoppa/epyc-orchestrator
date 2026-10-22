"""Prompt data types and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PromptStyle(Enum):
    """Prompt formatting styles."""

    MINIMAL = "minimal"
    """Minimal formatting, just the essentials."""

    STRUCTURED = "structured"
    """Structured with headers and sections."""

    DETAILED = "detailed"
    """Full detail with examples and context."""


@dataclass
class PromptConfig:
    """Configuration for prompt building."""

    style: PromptStyle = PromptStyle.STRUCTURED
    """Default prompt style."""

    max_context_chars: int = 4000
    """Maximum characters for context sections."""

    max_output_preview: int = 1500
    """Maximum characters for output previews (bumped for spill-to-file summaries)."""

    max_error_preview: int = 500
    """Maximum characters for error previews."""

    use_toon_encoding: bool = True
    """Use TOON encoding for structured data. Benchmark: 55.6% token reduction, 41.8% TTFT improvement."""

    include_examples: bool = False
    """Include usage examples in prompts."""

    tools_file: str | None = None
    """Path to external tools prompt file. Overrides style-based selection when set."""

    rules_file: str | None = None
    """Path to external rules prompt file. Overrides DEFAULT_ROOT_LM_RULES when set."""


@dataclass
class RootLMPrompt:
    """Structured prompt for the Root LM.

    Contains all sections that make up a Root LM prompt.
    """

    system: str = ""
    """System instructions."""

    tools: str = ""
    """Available tools description."""

    rules: str = ""
    """Rules the LLM must follow."""

    state: str = ""
    """Current REPL state."""

    context: str = ""
    """Additional context (error, output, etc.)."""

    reference_code: str = ""
    """Corpus-retrieved code snippets for prompt-lookup acceleration."""

    task: str = ""
    """The user's task."""

    instruction: str = ""
    """Final instruction."""

    turn_tail: str = ""
    """RTE-PFX-3: per-turn volatile line(s) rendered LAST (flag ``repl_turn_counter_tail``).

    Empty (the default) renders nothing, so every existing prompt is byte-identical."""

    def to_string(self, prefix_stable: bool = False) -> str:
        """Convert to a single prompt string.

        Args:
            prefix_stable: When True, render the FIXED sections (task,
                instruction) before the per-turn-mutating sections (state,
                context, reference_code) so that successive REPL turns share a
                longer prompt prefix and llama-server KV-cache reuse is
                maximized (RTE-Prefix). The mutating sections carry the turn
                delta (Turn N state, last output/error, corpus), so they must
                stay AFTER the fixed prefix for the cache to help at all.
                Default False preserves the legacy order.
        """
        return "\n".join(text for _, text in self._parts(prefix_stable))

    def _parts(self, prefix_stable: bool = False) -> list[tuple[str, str]]:
        """Ordered ``(kind, line)`` pairs; ``to_string`` joins the lines with newlines.

        ``kind`` is the RTE-LOG-1 block kind (``src/backends/prompt_layout.py``);
        separator lines belong to the section that follows them.
        """
        parts: list[tuple[str, str]] = []
        if self.system:
            parts.append(("system", self.system))
        if self.tools:
            parts.extend([("tools", ""), ("tools", "## Available Tools"), ("tools", self.tools)])
        if self.rules:
            parts.extend([("rules", ""), ("rules", "## Rules"), ("rules", self.rules)])
        if prefix_stable:
            # Fixed sections first — shared across every turn of a session.
            if self.task:
                parts.extend([("task", ""), ("task", "## Task"), ("task", self.task)])
            if self.instruction:
                parts.extend([("task", ""), ("task", "## Your Code"), ("task", self.instruction)])
            # Per-turn-mutating sections last — the delta between turns.
            if self.state:
                parts.extend([("state", ""), ("state", "## Current State"), ("state", self.state)])
            if self.context:
                parts.extend([("context", ""), ("context", self.context)])
            if self.reference_code:
                parts.extend([("reference_code", ""), ("reference_code", "## Reference Code"),
                              ("reference_code", self.reference_code)])
        else:
            if self.state:
                parts.extend([("state", ""), ("state", "## Current State"), ("state", self.state)])
            if self.context:
                parts.extend([("context", ""), ("context", self.context)])
            if self.reference_code:
                parts.extend([("reference_code", ""), ("reference_code", "## Reference Code"),
                              ("reference_code", self.reference_code)])
            if self.task:
                parts.extend([("task", ""), ("task", "## Task"), ("task", self.task)])
            if self.instruction:
                parts.extend([("task", ""), ("task", "## Your Code"), ("task", self.instruction)])
        if self.turn_tail:  # RTE-PFX-3: volatile tail, after every section in both orders
            parts.extend([("state", ""), ("state", self.turn_tail)])
        return parts

    def section_spans(self, prefix_stable: bool = False) -> list[tuple[str, int, int]]:
        """``(kind, start, end)`` character spans of each section in ``to_string``.

        Adjacent lines of one kind merge; a span starts after its leading blank
        separator line so it covers the heading and body only.
        """
        spans: list[tuple[str, int, int]] = []
        pos = 0
        for i, (kind, text) in enumerate(self._parts(prefix_stable)):
            start, end = pos, pos + len(text)
            pos = end + 1  # the "\n" join
            if text == "":
                continue
            if spans and spans[-1][0] == kind and start - spans[-1][2] <= 2:
                spans[-1] = (kind, spans[-1][1], end)
            else:
                spans.append((kind, start, end))
        return spans


@dataclass
class EscalationPrompt:
    """Structured prompt for escalation scenarios.

    Contains all sections for an escalation prompt.
    """

    header: str = ""
    """Escalation header with source role."""

    failure_info: str = ""
    """Information about the failure."""

    error_details: str = ""
    """Detailed error information."""

    state: str = ""
    """Current state."""

    task: str = ""
    """Original task."""

    instructions: str = ""
    """Instructions for the escalated role."""

    def to_string(self) -> str:
        """Convert to a single prompt string."""
        parts = []
        if self.header:
            parts.append(self.header)
        if self.failure_info:
            parts.extend(["", self.failure_info])
        if self.error_details:
            parts.extend(["", "## Error Details", self.error_details])
        if self.state:
            parts.extend(["", "## Current State", self.state])
        if self.task:
            parts.extend(["", "## Original Task", self.task])
        if self.instructions:
            parts.extend(["", "## Instructions", self.instructions])
        return "\n".join(parts)


@dataclass
class StepPrompt:
    """Structured prompt for step execution.

    Contains all sections for a step execution prompt.
    """

    action: str = ""
    """The action to perform."""

    inputs: str = ""
    """Input context."""

    outputs: str = ""
    """Expected outputs."""

    constraints: str = ""
    """Any constraints or requirements."""

    def to_string(self) -> str:
        """Convert to a single prompt string."""
        parts = []
        if self.action:
            parts.append(f"Task: {self.action}")
        if self.inputs:
            parts.append(f"\nInputs:{self.inputs}")
        if self.outputs:
            parts.append(f"\nExpected outputs: {self.outputs}")
        if self.constraints:
            parts.append(f"\nConstraints: {self.constraints}")
        return "\n".join(parts)
