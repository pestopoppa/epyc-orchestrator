#!/usr/bin/env python3
"""effort_ladder_spec.py — E-1: the reasoning-effort ladder, as an executable spec.

Task: handoffs/active/reasoning-effort-levels.md E-1 (L173). Draft of 4-5 effort levels as PROMPT
variants (the accuracy lever, per R2a/E-6) each with a BUDGET BACKSTOP (a graceful-close safety net,
never the primary knob — a bare max_tokens cap truncates mid-derivation and scores a garbage wrong
answer; L59-68 of the handoff).

This module is the canonical, machine-readable ladder that E-2's sweep runner
(`v7_quality_gate_runner.py`) can import, so the prompt text lives in ONE place. Companion design
doc: orchestration/reports/reasoning_effort/E1_EFFORT_LADDER_DESIGN.md.

Design invariants encoded here:
  * PROMPT is the lever. `budget_backstop_tokens` is generous headroom, sized from TB-7 realized
    demand (+ headroom), NOT a target. A level that hits its backstop is DISQUALIFIED, not scored
    wrong (see `disqualify`).
  * Cheap levels (L0/L1) put the ANSWER FIRST -> truncation-robust (a well-formed answer survives a
    tight budget). Expensive levels (L2/L3) put REASONING FIRST -> require a generous backstop +
    graceful close; truncation there => disqualify.
  * Levels are DEFINED generically but CERTIFIED PER (model, quant) — no level is "enabled" for a
    model until its own E-2 curve is measured. `budget_backstop_tokens` here are model-agnostic
    STARTING points to be re-certified per model (the per-model INVARIANT, handoff L36).
  * The native-<think> level (L4) is a SAFETY-capped option, not an accuracy win: E-6 showed capped
    native thinking only ties think-off. It MUST run with a reasoning-budget force-close.

All numbers are OBSERVATIONS per MEASUREMENT.md (backstops seeded from TB-7 mined demand; the curve
that certifies each level per model is E-2, not yet run).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass(frozen=True)
class EffortLevel:
    id: str
    name: str
    intent: str
    # Prompt instruction appended to the task. `{task}` is the task body slot.
    prompt_template: str
    answer_ordering: str            # "answer_first" (truncation-robust) | "reasoning_first"
    enable_thinking: bool           # native <think> channel on/off
    budget_backstop_tokens: int     # generous graceful-close backstop, NOT a target
    reasoning_budget_tokens: Optional[int]  # only for native-think levels (force-close via
    #                                          --reasoning-budget + --reasoning-budget-message)
    expected_token_regime: str      # grounded in R2a endpoints + TB-7 realized demand
    fits: str                       # which task-classes / roles this level suits


# Backstops are sized from TB-7 realized demand (per_task_class p95 + headroom) so a well-behaved
# response never hits them; they exist only to force a clean close instead of a runaway generation.
LADDER: List[EffortLevel] = [
    EffortLevel(
        id="L0",
        name="answer-only",
        intent="Cheapest. No reasoning emitted; format-only answer. Baseline low end (R2a 52%/537tok).",
        prompt_template=(
            "{task}\n\n"
            "Respond with ONLY the final answer in the exact required format. "
            "Do not explain, justify, or show any work."
        ),
        answer_ordering="answer_first",
        enable_thinking=False,
        budget_backstop_tokens=256,
        reasoning_budget_tokens=None,
        expected_token_regime="~1-50 tokens realized (TB-7: general/thinking/vl/tool_use classes).",
        fits="Saturated / lookup / classification classes: general, simpleqa, tool_use, routing.",
    ),
    EffortLevel(
        id="L1",
        name="terse-justify",
        intent="Answer first, then one short justification sentence. Cheap sanity without full CoT.",
        prompt_template=(
            "{task}\n\n"
            "Give the final answer in the required format FIRST, then add at most ONE short sentence "
            "of justification. Keep it under 40 words total."
        ),
        answer_ordering="answer_first",
        enable_thinking=False,
        budget_backstop_tokens=512,
        reasoning_budget_tokens=None,
        expected_token_regime="~50-350 tokens (TB-7: hotpotqa/agentic/long_context p95 ~300-500).",
        fits="Light-reasoning retrieval/QA: hotpotqa, agentic, long_context, instruction_precision(easy).",
    ),
    EffortLevel(
        id="L2",
        name="bounded-reasoning",
        intent="A few explicit steps, length-bounded by the PROMPT (not a token cap). The dial's mid.",
        prompt_template=(
            "{task}\n\n"
            "Reason briefly: at most 3 short steps, about 120 words of working, then give the final "
            "answer in the required format on its own line. Be concise; do not restate the problem."
        ),
        answer_ordering="reasoning_first",
        enable_thinking=False,
        budget_backstop_tokens=1024,
        reasoning_budget_tokens=None,
        expected_token_regime="~300-1000 tokens (TB-7: math/skill_transfer/instruction_precision p95).",
        fits="Moderate reasoning: math (non-olympiad), skill_transfer, gpqa(easy), instruction_precision.",
    ),
    EffortLevel(
        id="L3",
        name="full-cot",
        intent="Full chain-of-thought then answer. Measured accuracy high end (R2a 84%/2150tok, +32pp).",
        prompt_template=(
            "{task}\n\n"
            "Reason step by step, working through the problem carefully, then state the final answer "
            "in the required format on its own line at the end."
        ),
        answer_ordering="reasoning_first",
        enable_thinking=False,
        budget_backstop_tokens=4096,  # generous: R2a mean 2150; TB-7 code/math p99 up to ~3080 and
        #                               right-censored at 2048 -> keep the backstop well above the knee.
        reasoning_budget_tokens=None,
        expected_token_regime="~1000-3000+ tokens (TB-7 code/math censored; true knee via TB-1).",
        fits="Hard reasoning/code: livecodebench, debugbench, bigcodebench, gpqa, mode_advantage_hard.",
    ),
    EffortLevel(
        id="L4",
        name="native-think-capped",
        intent=("Native <think> channel, FORCE-CLOSED by a reasoning budget. SAFETY option, not an "
                "accuracy win: E-6 showed capped native thinking only ties think-off, never beats "
                "the L3 prompt lever. Use only where a model's native channel is wanted AND proven "
                "to terminate under cap for that (model, quant)."),
        prompt_template="{task}",  # the native channel supplies the reasoning; prompt stays clean
        answer_ordering="reasoning_first",
        enable_thinking=True,
        budget_backstop_tokens=512,          # answer budget AFTER the think block closes
        reasoning_budget_tokens=2048,        # --reasoning-budget 2048 --reasoning-budget-message ...
        expected_token_regime="~1.6-3x tokens vs think-off (E-6), 0% non-termination once capped.",
        fits="Models whose native <think> terminates under cap (certify per model — R2b: 35B-A3B "
             "does NOT terminate 48% of the time even at 16k; do not enable there).",
    ),
]

# Mandatory graceful-close message for any level with a reasoning budget (L4). Without it the model
# is cut mid-token and the response scores wrong (the exact artifact the handoff warns about).
REASONING_BUDGET_MESSAGE = (
    "You have reached your reasoning budget. Stop reasoning now and immediately give your best final "
    "answer in the required format."
)


def render(level: EffortLevel, task: str) -> str:
    """Render the effort-conditioned prompt for a task body."""
    return level.prompt_template.format(task=task)


def disqualify(level: EffortLevel, *, tokens_generated: int, finish_reason: Optional[str],
               content: Optional[str], answer_parsed: bool) -> Optional[str]:
    """Return a disqualification reason (level result is DROPPED, not scored wrong) or None.

    Encodes the handoff rule: a level that truncates, fails to terminate, or emits empty/unparseable
    content is DISQUALIFIED, not scored as incorrect — scoring it wrong is the artifact that has
    already bitten twice (L61-64, L181).
    """
    if finish_reason in ("length", "truncated"):
        return "truncated_at_backstop"
    if tokens_generated >= level.budget_backstop_tokens + (level.reasoning_budget_tokens or 0):
        return "hit_budget_backstop"
    if not content or not content.strip():
        return "empty_content"
    if not answer_parsed:
        return "parse_failure"
    return None


def as_dicts() -> List[Dict]:
    return [asdict(l) for l in LADDER]


def _self_test() -> None:
    ids = [l.id for l in LADDER]
    assert ids == ["L0", "L1", "L2", "L3", "L4"], ids
    # backstop monotonic across the prompt-effort levels L0..L3
    prompt_levels = [l for l in LADDER if not l.enable_thinking]
    budgets = [l.budget_backstop_tokens for l in prompt_levels]
    assert budgets == sorted(budgets), f"backstops not monotonic: {budgets}"
    # every native-think level must carry a reasoning budget (force-close)
    for l in LADDER:
        if l.enable_thinking:
            assert l.reasoning_budget_tokens, f"{l.id} native-think without reasoning budget"
    # render smoke test
    assert "step by step" in render(LADDER[3], "Q?")
    print("effort_ladder_spec self-test OK")


if __name__ == "__main__":
    import json
    print(json.dumps({"ladder": as_dicts(),
                      "reasoning_budget_message": REASONING_BUDGET_MESSAGE}, indent=2))
    _self_test()
