"""Who&When Pro text split -> attribution cases (JSON only; intake-2340#record).

Each row is a trajectory with ONE injected error at a known unit; gold = ``{unit: 1.0}`` plus the
gold agent(s). The dataset's locators differ per framework, so each is resolved explicitly:

    alfagent, smolagents   ``step`` = the action's ``step_number``; smolagents step 0 = the planning
                           entry right after the task (later planning entries carry no locator)
    magentic-one           ``step`` = "<round>.<position>" of an agent entry
    mathchat               ``round`` + ``position`` (+ ``agent``) of an entry
    metagpt                ``stage`` (+ ``agent``) of an entry
    macnet                 locator fields do not match entry fields; resolved by the entry's
                           ``is_injected`` flag, cross-checked against ``agent``
    debate, dylan          ``round`` + ``agents`` (a set: every named agent's turn in that round is gold)

A case whose gold does not resolve to exactly the unit(s) above is skipped with a counted reason.
Injection flags (``is_injected``, ``is_replayed``, ``is_live``, ``injection_via``) are read for gold
resolution only and never copied into an event, so no prompt can see them. Every trace is a failure by
construction: this measures attribution of INJECTED errors, not natural failures.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.trace.attribution.case import AttributionCase
from src.trace.attribution.schema import AgentEvent, AgentTrajectory, EventType

_FALSY = (None, "", "None", "none", "null")


def _true(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1")


def _jl(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def build_case(row: dict) -> tuple[AttributionCase | None, str | None]:
    """One dataset row -> (case, None) or (None, skip_reason)."""
    fw = row.get("framework")
    task = _jl(row.get("task")) or {}
    gt = _jl(row.get("ground_truth")) or {}
    traj_raw = _jl(row.get("trajectory")) or []
    if not isinstance(traj_raw, list) or not isinstance(gt, dict):
        return None, "malformed_row"
    tid = row["id"]
    events: list[AgentEvent] = []
    gold: list[int] = []
    final_answer = None
    pos = 0
    seen_plan = False

    def add(
        et: EventType,
        agent: str,
        inp: Any,
        out: Any,
        err: Any = None,
        meta: dict | None = None,
        is_gold: bool = False,
    ) -> None:
        nonlocal pos
        events.append(
            AgentEvent(
                event_id=f"{tid}:{pos}",
                trace_id=tid,
                event_type=et,
                agent_name=agent or "agent",
                step_index=pos,
                input=inp,
                output=out,
                error=None if err in _FALSY else str(err),
                metadata=meta or {},
            )
        )
        if is_gold:
            gold.append(pos)
        pos += 1

    gstep = gt.get("step")
    gagent = gt.get("agent")
    gagents = set(gt.get("agents") or ([] if gagent is None else [gagent]))
    for i, x in enumerate(traj_raw):
        if not isinstance(x, dict):
            continue
        k = x.get("kind")
        if k == "user":
            events.append(
                AgentEvent(
                    event_id=f"{tid}:task",
                    trace_id=tid,
                    event_type=EventType.RUN_START,
                    agent_name="user",
                    output=x.get("content"),
                )
            )
        elif k == "final_answer":
            final_answer = x.get("content")
        elif k == "planning":
            first = not seen_plan and i == 1
            seen_plan = True
            add(EventType.PLAN, "agent", None, x.get("plan"), is_gold=first and str(gstep) == "0")
        elif k == "action":
            body_key = "code" if "code" in x else "action"
            inp = {"reasoning": x.get("reasoning"), body_key: x.get(body_key)}
            add(
                EventType.TOOL_CALL,
                "agent",
                inp,
                x.get("observation"),
                x.get("error"),
                {
                    "tool": "python" if body_key == "code" else "env_action",
                    "is_final": _true(x.get("is_final_answer")),
                    "step_number": x.get("step_number"),
                },
                is_gold=gstep is not None and str(x.get("step_number")) == str(gstep),
            )
        elif k == "agent":  # magentic-one
            ta = x.get("tool_actions")
            loc = f"{x.get('round')}.{x.get('position')}"
            add(
                EventType.TOOL_CALL if ta not in _FALSY else EventType.AGENT_STEP,
                x.get("agent"),
                {"tool_actions": ta} if ta not in _FALSY else None,
                x.get("output"),
                meta={"locator": loc},
                is_gold=gstep is not None and loc == str(gstep),
            )
        elif k == "round":  # debate, dylan
            for turn in x.get("turns") or []:
                add(
                    EventType.AGENT_STEP,
                    turn.get("agent_id"),
                    None,
                    turn.get("output"),
                    meta={"round": x.get("round")},
                    is_gold=str(x.get("round")) == str(gt.get("round"))
                    and turn.get("agent_id") in gagents,
                )
        elif k is None and "agent_id" in x:  # mathchat, metagpt, macnet
            if fw == "mathchat":
                hit = (
                    str(x.get("round")) == str(gt.get("round"))
                    and str(x.get("position")) == str(gt.get("position"))
                    and x.get("agent_id") == gagent
                )
            elif fw == "metagpt":
                hit = str(x.get("stage")) == str(gt.get("stage")) and x.get("agent_id") == gagent
            else:
                hit = _true(x.get("is_injected")) and x.get("agent_id") == gagent
            add(
                EventType.AGENT_STEP,
                x.get("agent_id"),
                None,
                x.get("output"),
                meta={"module": x.get("role")},
                is_gold=hit,
            )
    if not any(e.step_index is not None for e in events):
        return None, "no_steps"
    if not gold:
        return None, f"gold_unresolved:{fw}"
    if len(gold) > 1 and fw not in ("debate", "dylan"):
        return None, f"gold_ambiguous:{fw}"
    goal = task.get("query") if isinstance(task, dict) else None
    traj = AgentTrajectory(
        trace_id=tid,
        task_id=tid,
        goal=goal,
        framework=fw,
        metadata={
            "source": "whowhen_pro_text",
            "outcome": "fail",
            "benchmark": row.get("benchmark"),
            "final_answer": final_answer,
        },
        events=events,
    )
    return AttributionCase(
        tid,
        traj,
        {g: 1.0 for g in gold},
        gagents,
        reference=task.get("answer") if isinstance(task, dict) else None,
        meta={"framework": fw, "mode": gt.get("mode"), "benchmark": row.get("benchmark")},
    ), None


def load_cases(path: Path | str, skips: Counter | None = None) -> Iterator[AttributionCase]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            case, why = build_case(json.loads(line))
            if case is not None:
                yield case
            elif skips is not None:
                skips[why or "unknown"] += 1


def select_slice(
    cases: Iterable[AttributionCase],
    per_framework: int,
    seed: int = 0,
    frameworks: Iterable[str] | None = None,
    max_steps: int | None = None,
) -> list[AttributionCase]:
    """Deterministic stratified slice: per framework, a seeded draw of ``per_framework`` cases (sorted by
    id first, so the slice depends only on the file contents and the seed)."""
    want = set(frameworks) if frameworks else None
    by: dict[str, list[AttributionCase]] = {}
    for c in cases:
        fw = c.meta.get("framework") or "?"
        if want is not None and fw not in want:
            continue
        if max_steps is not None and len(c.trajectory.step_indices()) > max_steps:
            continue
        by.setdefault(fw, []).append(c)
    out: list[AttributionCase] = []
    for fw in sorted(by):
        group = sorted(by[fw], key=lambda c: c.case_id)
        random.Random(f"{seed}:{fw}").shuffle(group)
        out.extend(sorted(group[:per_framework], key=lambda c: c.case_id))
    return out


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
