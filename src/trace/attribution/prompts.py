"""Prompt harness: trajectory rendering and the four request builders.

Prompt texts are EPYC's own, modelled on the AgentDebugX prompt contracts in arXiv:2607.18754v1
Appendix A (whole-trace localizer, binary-search probe, cross-examination) and on the MP-Bench
all-at-once multi-step output (intake-2332#record). One step renders as ONE line,
``Step <i> | <agent> | <event type> :: <content>``, so a long trace stays line-addressable.

Leakage rule: only an event's input, output and error are rendered. Dataset labels (injection
flags, gold fields) never reach an event, and the reference answer is rendered only when the
caller sets ``with_reference`` (a report must state which; AgentDebugX D4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.trace.attribution.judge import JudgeRequest
from src.trace.attribution.schema import AgentEvent, AgentTrajectory


@dataclass(frozen=True)
class PromptConfig:
    max_step_chars: int = 600
    max_tokens_localize: int = 1024
    max_tokens_bisect: int = 256
    max_tokens_adjudicate: int = 256
    max_tokens_multi: int = 1024
    with_reference: bool = False


def _flat(v: Any) -> str:
    if v is None:
        return ""
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return " ".join(s.split())


def render_step(e: AgentEvent, cfg: PromptConfig) -> str:
    parts = []
    if e.input not in (None, "", {}):
        parts.append(f"input: {_flat(e.input)}")
    if e.output not in (None, ""):
        parts.append(f"output: {_flat(e.output)}")
    if e.error:
        parts.append(f"error: {_flat(e.error)}")
    body = " || ".join(parts) or "(empty)"
    if len(body) > cfg.max_step_chars:
        body = body[: cfg.max_step_chars] + f" ...[{len(body) - cfg.max_step_chars} chars elided]"
    return f"Step {e.step_index} | {e.agent_name} | {e.event_type.value} :: {body}"


def render_steps(events: list[AgentEvent], cfg: PromptConfig) -> str:
    return "\n".join(render_step(e, cfg) for e in events)


def _header(traj: AgentTrajectory, cfg: PromptConfig, reference: str | None) -> str:
    h = f"Goal: {_flat(traj.goal) or '(not recorded)'}\n"
    final = (traj.metadata or {}).get("final_answer")
    if final not in (None, ""):
        h += f"Final answer given by the run: {_flat(final)[: cfg.max_step_chars]}\n"
    if cfg.with_reference and reference:
        h += f"Correct answer: {_flat(reference)}\n"
    return h + "This run FAILED.\n"


_JSON_RULES = (
    "Output ONLY one JSON object: no prose, no markdown fences, no newlines inside strings. "
    "Close the JSON completely."
)

SYSTEM_LOCALIZE = (
    "You analyse the step-by-step record of an AI agent run that failed. Identify the single decisive "
    "error step: the earliest step whose correction would most likely have averted the failure. "
    + _JSON_RULES
    + ' Schema: {"step_index": <int>, "agent_name": "<name or null>", '
    '"confidence": <float 0..1>, "rationale": "<one sentence>"}'
)

SYSTEM_BISECT = (
    "You run a binary-search failure attribution. A segment of a failed agent run is split into an "
    "UPPER HALF (earlier steps) and a LOWER HALF (later steps). Decide which half contains the single "
    "decisive error step that caused the failure. "
    + _JSON_RULES
    + ' Schema: {"half": "upper" | "lower", "rationale": "<one short sentence>"}'
)

SYSTEM_ADJUDICATE = (
    "A run FAILED. Two candidate error steps are shown with their context. Which ONE is the decisive "
    "critical error step, the one whose correction would most likely have averted the failure? "
    + _JSON_RULES
    + ' Schema: {"step": <int, one of the two>}'
)

SYSTEM_MULTI = (
    "You analyse the step-by-step record of an AI agent run that failed. List every step you judge to "
    "be failure-inducing (a step whose correction would have changed the outcome), most important "
    "first. "
    + _JSON_RULES
    + ' Schema: {"steps": [{"step": <int>, "agent": "<name or null>", "reason": "<one sentence>"}]}'
)


def localize_request(
    traj: AgentTrajectory,
    cfg: PromptConfig,
    reference: str | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
) -> JudgeRequest:
    user = (
        _header(traj, cfg, reference)
        + "\n"
        + render_steps(traj.steps(), cfg)
        + "\n\nWhich step is the decisive error step?"
    )
    return JudgeRequest(
        "localize", SYSTEM_LOCALIZE, user, temperature, cfg.max_tokens_localize, seed
    )


def bisect_request(
    traj: AgentTrajectory,
    upper: list[AgentEvent],
    lower: list[AgentEvent],
    cfg: PromptConfig,
    reference: str | None = None,
) -> JudgeRequest:
    user = (
        _header(traj, cfg, reference)
        + f"\n=== UPPER HALF (steps {upper[0].step_index}-{upper[-1].step_index}) ===\n"
        + render_steps(upper, cfg)
        + f"\n=== LOWER HALF (steps {lower[0].step_index}-{lower[-1].step_index}) ===\n"
        + render_steps(lower, cfg)
        + "\n\nWhich half contains the decisive error step?"
    )
    return JudgeRequest("bisect", SYSTEM_BISECT, user, 0.0, cfg.max_tokens_bisect)


def _context(traj: AgentTrajectory, step: int, cfg: PromptConfig) -> str:
    idx = traj.step_indices()
    pos = idx.index(step)
    by = traj.by_step()
    return render_steps([by[i] for i in idx[max(0, pos - 1) : pos + 2]], cfg)


def adjudicate_request(
    traj: AgentTrajectory, a: int, b: int, cfg: PromptConfig, reference: str | None = None
) -> JudgeRequest:
    user = (
        _header(traj, cfg, reference)
        + f"\nCANDIDATE A = step {a}:\n{_context(traj, a, cfg)}"
        + f"\n\nCANDIDATE B = step {b}:\n{_context(traj, b, cfg)}"
        + f"\n\nWhich is the critical error step, {a} or {b}?"
    )
    return JudgeRequest("adjudicate", SYSTEM_ADJUDICATE, user, 0.0, cfg.max_tokens_adjudicate)


def multi_request(
    traj: AgentTrajectory,
    cfg: PromptConfig,
    reference: str | None = None,
    temperature: float = 0.7,
    seed: int | None = None,
) -> JudgeRequest:
    user = (
        _header(traj, cfg, reference)
        + "\n"
        + render_steps(traj.steps(), cfg)
        + "\n\nList the failure-inducing steps."
    )
    return JudgeRequest(
        "multi_attribute", SYSTEM_MULTI, user, temperature, cfg.max_tokens_multi, seed
    )
