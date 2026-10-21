"""Loaders from recorded sources into ``L2Trace``. Derived flags are derived HERE and named."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from src.trace.l2.model import L2Step, L2Trace

REQUEST_TIMEOUT_S = 120.0     # progress-log failures at/over this wall time are request timeouts
IMMEDIATE_FAIL_S = 0.01


def load_normalised_jsonl(path: Path | str) -> Iterator[L2Trace]:
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield L2Trace.from_dict(json.loads(line))


def load_progress_logs(paths: Iterable[Path | str]) -> Iterator[L2Trace]:
    """Task-level traces from logs/progress/*.jsonl (task_started / task_completed / task_failed).

    These logs carry no per-step tool events, so only trace-level fields exist: outcome, role,
    a derived ``exception`` (request_timeout / immediate_failure / plan_rejected:<reason>) and
    wall time from the outcome_details text. Zero-step violation is therefore NOT claimed for them
    (steps are unobserved, not absent): see ``steps_observed`` handling in the report.
    """
    for p in paths:
        starts: dict[str, dict] = {}
        for line in Path(p).read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            et, tid = d.get("event_type"), d.get("task_id")
            if et == "task_started":
                starts[tid] = d
            elif et in ("task_completed", "task_failed") and tid:
                det = d.get("outcome_details") or ""
                m = re.search(r"([\d.]+)s\s*$", det)
                wall = float(m.group(1)) if m else None
                exc = status = None
                if et == "task_failed":
                    if det.startswith("Plan rejected"):
                        status, exc = "plan_rejected", det.split(":", 1)[-1].strip()[:80]
                    elif wall is not None and wall >= REQUEST_TIMEOUT_S:
                        exc = "request_timeout"
                    elif wall is not None and wall <= IMMEDIATE_FAIL_S:
                        exc = "immediate_failure"
                data = d.get("data") or {}
                yield L2Trace(
                    trace_id=tid, task_key=tid,
                    outcome="pass" if et == "task_completed" else "fail",
                    role=data.get("producer_role") or d.get("agent_role"),
                    source="progress_log", status=status, exception=exc, steps=[],
                )


_ERR_OBS = re.compile(r"^(error|traceback|exception)|exit code [1-9]|command not found|no such file", re.I)


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1")


def _pro_steps(traj: list[dict]) -> list[L2Step]:
    """Map Who&When Pro trajectory elements to steps. Frameworks differ; each branch is explicit.

    action (smolagents, alfagent): a tool call (code or action text) with an observation.
    agent (magentic-one): one step per entry of tool_actions (message-only turns are ``message``).
    final_answer: the final step. Everything else (user, planning, round, ...) is a ``message``
    step that is not a tool call. ``error`` is the row's own error field when present, otherwise
    a fixed regex over the observation (derived flag, named here).
    """
    import ast
    steps: list[L2Step] = []

    def add(**kw):
        steps.append(L2Step(idx=len(steps), **kw))

    for pos, x in enumerate(traj):
        k = x.get("kind")
        if k == "action":
            obs = x.get("observation")
            obs = obs if isinstance(obs, str) else (None if obs is None else json.dumps(obs))
            err = x.get("error")
            has_err = (err not in (None, "None", "")) if "error" in x else bool(obs and _ERR_OBS.search(obs.strip()))
            body = x.get("code") if "code" in x else x.get("action")
            add(kind="tool_call", tool="python" if "code" in x else "env_action",
                args={"code" if "code" in x else "action": body} if body is not None else None,
                observation=obs, error=has_err, is_final=_truthy(x.get("is_final_answer")))
        elif k == "agent":
            ta = x.get("tool_actions")
            try:
                acts = ast.literal_eval(ta) if isinstance(ta, str) else (ta or [])
            except Exception:
                acts = []
            out = x.get("output") if isinstance(x.get("output"), str) else None
            if acts:
                for a in acts:
                    a = a if isinstance(a, dict) else {}
                    args = a.get("args") if isinstance(a.get("args"), dict) else None
                    add(kind="tool_call", tool=a.get("name"), args=args, observation=out,
                        error=bool(out and _ERR_OBS.search(out.strip())))
            else:
                add(kind="message", observation=out)
        elif k == "final_answer":
            add(kind="final", is_final=True)
        else:
            add(kind="message")
    return steps


def load_whowhen_pro(path: Path | str) -> Iterator[tuple[L2Trace, dict]]:
    """Who&When Pro text split. Yields (trace, gold) with gold = ground_truth {mode, step, agent}.

    Every trace is a failure by construction (injected), so outcome is ``fail``. ``step`` is the
    dataset's own step id and is NOT the same index space as our step idx; calibration compares
    modes only, never step ids.
    """
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        gt = json.loads(r["ground_truth"])
        yield (L2Trace(trace_id=r["id"], task_key=r["id"], outcome="fail", source="whowhen_pro",
                       role=r.get("framework"), steps=_pro_steps(json.loads(r["trajectory"]))), gt)


def load_claude_session(path: Path | str) -> L2Trace:
    """Claude-Code session format (KernelBench-Hard traces): one JSONL file = one run.

    tool_use blocks become tool_call steps; the matching tool_result supplies observation and
    ``error`` from its is_error flag (structured, not text-derived). No outcome label is read.
    """
    steps: list[L2Step] = []
    by_id: dict[str, L2Step] = {}
    sid = Path(path).stem
    for line in Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        msg = d.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if b.get("type") == "tool_use":
                s = L2Step(idx=len(steps), tool=b.get("name"), args=b.get("input") if isinstance(b.get("input"), dict) else None)
                steps.append(s)
                by_id[b.get("id")] = s
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in by_id:
                s = by_id[b["tool_use_id"]]
                c = b.get("content")
                s.observation = c if isinstance(c, str) else json.dumps(c)
                s.error = bool(b.get("is_error"))
        if d.get("type") == "assistant" and msg.get("stop_reason") == "max_tokens" and steps:
            steps[-1].finish_reason = "length"
    return L2Trace(trace_id=sid, task_key=sid, outcome="unknown", source="claude_session", steps=steps)
