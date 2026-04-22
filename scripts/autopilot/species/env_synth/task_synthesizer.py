"""Task synthesizer — composes verifiable Agent-World tasks (NIB2-44 AW-1).

A ``SynthesizedTask`` bundles:

  environment_id       link back to the discovery provenance
  tool_set             the MCP tools the agent may use
  prompt               the task statement
  difficulty_band      easy | medium | hard (tool-call count + chain depth)
  verifier             a VerifierSpec — scored by the built scorer
  ground_truth_hint    optional hint string (used by the solvability gate)
  metadata             free-form (distractors, expected tool-call count, …)

The synthesizer is LLM-backed: the caller injects an ``llm`` callable
that accepts a system+user pair and returns a JSON string with the task
fields. A seed-based deterministic ``FakeLLM`` is provided for tests so
the rest of the species can be validated without real inference.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from scripts.autopilot.species.env_synth.mcp_tool_registry import MCPToolEntry
from scripts.autopilot.species.env_synth.verifier_builder import (
    VerifierBuilder,
    VerifierSpec,
    VerifierType,
)

log = logging.getLogger("autopilot.env_synth.task_synthesizer")


class DifficultyBand(str, Enum):
    EASY = "easy"        # 1-2 tool calls, linear chain
    MEDIUM = "medium"    # 3-5 tool calls, some branching
    HARD = "hard"        # 6+ tool calls, branching + distractors


_BAND_TOOL_CALLS = {
    DifficultyBand.EASY: (1, 2),
    DifficultyBand.MEDIUM: (3, 5),
    DifficultyBand.HARD: (6, 10),
}


@dataclass
class SynthesizedTask:
    environment_id: str
    tool_set: list[str]                     # tool_ids from the registry
    prompt: str
    difficulty_band: DifficultyBand
    verifier: VerifierSpec
    ground_truth_hint: str = ""
    expected_tool_calls: tuple[int, int] = (1, 2)
    metadata: dict[str, Any] = field(default_factory=dict)


LLMCall = Callable[[str, str], Awaitable[str]]


@dataclass
class TaskSynthesizer:
    """Compose tasks for a given (environment, tool_subset) with a
    requested difficulty band. Caller injects ``llm`` — a FakeLLM is
    provided below for offline scaffolding / tests.
    """

    llm: LLMCall
    max_retries: int = 2

    async def synthesize(
        self,
        environment_id: str,
        tools: list[MCPToolEntry],
        band: DifficultyBand,
        *,
        seed: Optional[int] = None,
    ) -> Optional[SynthesizedTask]:
        """Produce one verifiable task, or ``None`` if synthesis fails."""
        if not tools:
            log.warning("task_synthesizer: empty tool set for env %s", environment_id)
            return None

        system = (
            "You are a synthetic-task generator for an Agent-World environment. "
            "Given a list of tools and a difficulty band, produce ONE verifiable "
            "task. Respond with a single JSON object containing keys: "
            "prompt, verifier (type in {regex, exact_match, f1}; reference; "
            "pattern; allowlist), ground_truth_hint, metadata. "
            "Reject vague or unverifiable tasks — the verifier MUST be cheap and "
            "deterministic."
        )

        tool_section = "\n".join(
            f"- {t.tool_id}: {t.name} — {t.description[:120]}"
            for t in tools
        )
        user = (
            f"Environment: {environment_id}\n"
            f"Difficulty band: {band.value}\n"
            f"Tool call count target: {_BAND_TOOL_CALLS[band]}\n"
            f"Tools available:\n{tool_section}\n\n"
            "Generate the JSON task."
        )

        for attempt in range(self.max_retries + 1):
            try:
                raw = await self.llm(system, user)
                task = self._parse(
                    raw,
                    environment_id=environment_id,
                    tool_set=[t.tool_id for t in tools],
                    band=band,
                    seed=seed,
                )
                if task is not None:
                    return task
            except Exception as e:
                log.warning(
                    "synthesize attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries + 1, e,
                )
        return None

    # ── parsing ────────────────────────────────────────────────────

    def _parse(
        self,
        raw: str,
        *,
        environment_id: str,
        tool_set: list[str],
        band: DifficultyBand,
        seed: Optional[int],
    ) -> Optional[SynthesizedTask]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("task synthesizer: non-JSON LLM output")
            return None

        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return None

        verifier = self._build_verifier(payload.get("verifier") or {})
        if verifier is None:
            return None

        return SynthesizedTask(
            environment_id=environment_id,
            tool_set=tool_set,
            prompt=prompt,
            difficulty_band=band,
            verifier=verifier,
            ground_truth_hint=(payload.get("ground_truth_hint") or "").strip(),
            expected_tool_calls=_BAND_TOOL_CALLS[band],
            metadata={
                "seed": seed,
                "raw_metadata": payload.get("metadata") or {},
            },
        )

    @staticmethod
    def _build_verifier(data: dict[str, Any]) -> Optional[VerifierSpec]:
        type_str = (data.get("type") or "").lower()
        try:
            vtype = VerifierType(type_str)
        except ValueError:
            log.warning("unknown verifier type %s", type_str)
            return None
        spec = VerifierSpec(
            type=vtype,
            reference=data.get("reference", ""),
            pattern=data.get("pattern", ""),
            allowlist=list(data.get("allowlist", []) or []),
            case_sensitive=bool(data.get("case_sensitive", False)),
            min_tokens=int(data.get("min_tokens", 1)),
        )
        try:
            VerifierBuilder.build(spec)  # raises on degenerate spec
        except ValueError as e:
            log.warning("verifier validation failed: %s", e)
            return None
        return spec


# ── fake LLM for offline scaffolding + tests ────────────────────────


def make_fake_llm(
    verifier_type: VerifierType = VerifierType.EXACT_MATCH,
    reference: str = "42",
    pattern: str = "",
    rng: Optional[random.Random] = None,
) -> LLMCall:
    """Deterministic fake LLM that returns a valid JSON task payload.

    Useful for tests + smoke runs without live inference.
    """
    rng = rng or random.Random(0)

    async def _llm(system: str, user: str) -> str:
        topic = f"topic_{rng.randint(0, 9999)}"
        payload: dict[str, Any] = {
            "prompt": f"Using the provided tools, compute the expected answer for {topic}.",
            "verifier": {
                "type": verifier_type.value,
                "reference": reference,
                "pattern": pattern or r"^\d+$",
                "allowlist": [],
                "case_sensitive": False,
                "min_tokens": 1,
            },
            "ground_truth_hint": reference,
            "metadata": {"topic": topic},
        }
        return json.dumps(payload)

    return _llm
