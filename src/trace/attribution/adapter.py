"""Orchestrator trace -> AgentDebugX ``AgentTrajectory`` adapter, and the projection back to ``L2Trace``.

Two orchestrator sources:

* ``from_l2_trace``: any ``L2Trace`` (progress logs, ``tool_step`` rows via
  ``load_step_events``, normalised JSONL). One event per step; ``step_index`` = the L2 step idx,
  so an L2 detection and a judge's answer live in the same index space.
* ``from_recorded_run``: an HIL-16 replay recording (``model_turn`` + ``tool_step`` rows + CAS).
  Each turn's model output becomes an ``llm.response`` event followed by that turn's tool calls,
  with argument and output text read back from the CAS. ``step_index`` here is the position in
  that merged order (the tool ``idx`` is kept in ``metadata``).

The diagnosis is never written into the trajectory (the package's layering rule). Fields the
source did not capture stay ``None``; nothing is inferred.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.trace.attribution.schema import AgentEvent, AgentTrajectory, EventType
from src.trace.l2.model import L2Step, L2Trace

_L2_META = (
    "tool",
    "exit_status",
    "truncated",
    "finish_reason",
    "tokens_in",
    "is_final",
    "mutating",
    "verifying",
    "args_hash",
    "state_digest",
    "invocation_error",
    "output_bytes",
    "error_class",
)


def _err_text(s: L2Step) -> str | None:
    if s.invocation_error:
        return f"invalid_invocation:{s.invocation_error}"
    if s.error is True:
        return s.error_class or "error"
    if s.exit_status not in (None, 0):
        return f"exit_status={s.exit_status}"
    return None


def from_l2_trace(t: L2Trace, goal: str | None = None) -> AgentTrajectory:
    agent = t.role or "agent"
    events: list[AgentEvent] = []
    for s in t.steps:
        if s.kind == "tool_call":
            et, inp = (
                EventType.TOOL_CALL,
                (
                    s.args
                    if s.args is not None
                    else (s.args_raw or ({"args_hash": s.args_hash} if s.args_hash else None))
                ),
            )
        elif s.kind == "final" or s.is_final:
            et, inp = EventType.RUN_END, None
        else:
            et, inp = EventType.AGENT_STEP, None
        meta = {k: getattr(s, k) for k in _L2_META if getattr(s, k) is not None}
        meta["l2_kind"] = s.kind
        events.append(
            AgentEvent(
                event_id=f"{t.trace_id}:{s.idx}",
                trace_id=t.trace_id,
                event_type=et,
                agent_name=agent,
                module=s.tool,
                step_index=s.idx,
                input=inp,
                output=s.observation,
                error=_err_text(s),
                duration_ms=s.latency_ms,
                metadata=meta,
            )
        )
    meta = {"source": t.source or "l2_trace", "outcome": t.outcome}
    for k in ("role", "model_pin", "arm", "status", "exception"):
        if getattr(t, k) is not None:
            meta[k] = getattr(t, k)
    return AgentTrajectory(
        trace_id=t.trace_id,
        task_id=t.task_key,
        goal=goal,
        framework="epyc-orchestrator",
        metadata=meta,
        events=events,
    )


def _cas_text(cas: Any, addr: str | None) -> str | None:
    if cas is None or addr is None:
        return None
    try:
        return cas.get_text(addr)
    except Exception:  # missing / corrupt blob: the field is unobserved, not empty
        return None


def from_recorded_run(
    db_path: Path | str,
    run_id: str,
    cas_root: Path | str | None = None,
    goal: str | None = None,
    outcome: str = "unknown",
    include_prompts: bool = False,
) -> AgentTrajectory:
    """Replay recording -> trajectory. Prompts are omitted by default (each one repeats the whole
    history); ``include_prompts`` puts the prompt text in the ``llm.response`` event's input."""
    from src.trace.replay.cas import CAS
    from src.trace.replay.model import load_run

    run = load_run(db_path, run_id)
    cas = CAS(cas_root) if cas_root is not None else None
    events: list[AgentEvent] = []
    pos = 0
    for t in run.turns:
        inp: Any = {"prompt_cas": t.prompt_cas}
        if include_prompts:
            inp["prompt"] = _cas_text(cas, t.prompt_cas)
        events.append(
            AgentEvent(
                event_id=f"{run_id}:turn:{t.turn}",
                trace_id=run_id,
                event_type=EventType.LLM_RESPONSE,
                agent_name=t.role or "agent",
                step_index=pos,
                input=inp,
                output=_cas_text(cas, t.output_cas),
                metadata={"turn": t.turn, "output_cas": t.output_cas, "chain": t.chain},
            )
        )
        turn_event = events[-1].event_id
        pos += 1
        for s in t.steps:
            if s.kind == "tool_call":
                raw = _cas_text(cas, s.args_cas)
                try:
                    args: Any = json.loads(raw) if raw is not None else {"args_hash": s.args_hash}
                except ValueError:
                    args = raw
                et = EventType.TOOL_CALL
            else:
                args, et = None, (EventType.RUN_END if s.kind == "final" else EventType.AGENT_STEP)
            events.append(
                AgentEvent(
                    event_id=f"{run_id}:{s.idx}",
                    trace_id=run_id,
                    event_type=et,
                    agent_name=t.role or "agent",
                    parent_event_id=turn_event,
                    module=s.tool,
                    step_index=pos,
                    input=args,
                    output=_cas_text(cas, s.output_cas),
                    error=None if s.ok else (s.error_class or "error"),
                    metadata={"tool_idx": s.idx, "l2_kind": s.kind, "tool": s.tool},
                )
            )
            pos += 1
    return AgentTrajectory(
        trace_id=run_id,
        task_id=run_id,
        goal=goal,
        framework="epyc-orchestrator",
        metadata={"source": "replay_recording", "outcome": outcome},
        events=events,
    )


def to_l2_trace(traj: AgentTrajectory, outcome: str = "fail") -> L2Trace:
    """Project a trajectory onto ``L2Trace`` so the free rung (L2 rules) can run on any source.

    Step idx = event ``step_index``; ``tool.call`` events become tool calls (``error`` = the event's
    error field is set), ``run.end`` is the final step, everything else is a message. Events without a
    step index are not attributable and are skipped.
    """
    steps: list[L2Step] = []
    for e in traj.steps():
        m = e.metadata or {}
        if e.event_type == EventType.TOOL_CALL:
            steps.append(
                L2Step(
                    idx=e.step_index,
                    kind="tool_call",
                    tool=m.get("tool") or e.module,
                    args=e.input
                    if isinstance(e.input, dict) and "args_hash" not in e.input
                    else None,
                    args_raw=e.input if isinstance(e.input, str) else None,
                    observation=e.output
                    if isinstance(e.output, str)
                    else (
                        None
                        if e.output is None
                        else json.dumps(e.output, sort_keys=True, default=str)
                    ),
                    error=e.error is not None and not str(e.error).startswith("invalid_invocation"),
                    invocation_error=(
                        str(e.error).split(":", 1)[1]
                        if e.error and str(e.error).startswith("invalid_invocation:")
                        else None
                    ),
                    exit_status=m.get("exit_status"),
                    truncated=m.get("truncated"),
                    finish_reason=m.get("finish_reason"),
                    mutating=m.get("mutating"),
                    verifying=m.get("verifying"),
                    args_hash=m.get("args_hash"),
                    state_digest=m.get("state_digest"),
                    output_bytes=m.get("output_bytes"),
                    is_final=bool(m.get("is_final")),
                )
            )
        elif e.event_type == EventType.RUN_END:
            steps.append(L2Step(idx=e.step_index, kind="final", is_final=True))
        else:
            steps.append(
                L2Step(
                    idx=e.step_index,
                    kind="message",
                    observation=e.output if isinstance(e.output, str) else None,
                )
            )
    meta = traj.metadata or {}
    return L2Trace(
        trace_id=traj.trace_id,
        task_key=traj.task_id or traj.trace_id,
        outcome=meta.get("outcome", outcome)
        if meta.get("outcome") not in (None, "unknown")
        else outcome,
        role=meta.get("role"),
        source=meta.get("source"),
        status=meta.get("status"),
        exception=meta.get("exception"),
        steps=steps,
    )
