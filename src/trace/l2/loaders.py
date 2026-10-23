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


def load_step_events(db_path: Path | str, outcomes: dict[str, L2Trace] | None = None) -> Iterator[L2Trace]:
    """HIL-13a: ``tool_step`` rows of the unified trace store -> one ``L2Trace`` per run (session).

    ``outcomes`` optionally maps a run id (= task id) to a task-level trace from
    ``load_progress_logs``; outcome / status / exception / role are taken from it. Without a match
    the outcome is ``unknown`` (never inferred from the steps).

    Derived flags, named here: ``error`` is the capture-time ok/error verdict EXCEPT when the
    registry rejected the invocation itself (``invocation_error``), which is invalid_invocation, not
    a tool failure. ``exit_status`` is not captured and stays None. ``step_cap``, ``ctx_limit`` and
    ``tokens_in`` are not captured, so the near-context-limit wrap-up signature is not computable
    from these traces; ``tool_required`` is None (not captured) unless
    ``apply_tool_required`` sets it from a task file.
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("select detail_json from event where category='tool_step' order by id").fetchall()
    finally:
        conn.close()
    runs: dict[str, list[dict]] = {}
    for (dj,) in rows:
        rec = json.loads(dj)
        runs.setdefault(rec["run_id"], []).append(rec)
    for rid, recs in runs.items():
        recs.sort(key=lambda r: r["idx"])
        steps = []
        for r in recs:
            inv = r.get("error_class") if r.get("error_class") in ("unknown_tool", "invalid_arguments") else None
            steps.append(L2Step(
                idx=r["idx"], kind=r["kind"], tool=r.get("tool"),
                error=bool(r.get("error")) and inv is None, truncated=bool(r.get("truncated")),
                is_final=bool(r.get("is_final")), mutating=r.get("mutating"),
                args_hash=r.get("args_hash"), state_digest=r.get("state_hash"), invocation_error=inv,
                output_bytes=r.get("output_bytes"), latency_ms=r.get("latency_ms"), error_class=r.get("error_class"),
            ))
        base = (outcomes or {}).get(rid)
        yield L2Trace(
            trace_id=rid, task_key=base.task_key if base else rid,
            outcome=base.outcome if base else "unknown",
            role=(base.role if base and base.role else recs[0].get("role")),
            source="step_events", status=base.status if base else None,
            exception=base.exception if base else None, steps=steps,
        )


_REQUIRES_TOOL_RE = re.compile(r"\buse the [\w.-]+ tool\b", re.I)


def tool_required_map(tasks: list[dict]) -> dict[str, bool]:
    """HIL-15c: task_key -> tool_required from a driver ``--tasks`` list. An explicit boolean
    ``tool_required`` wins; otherwise the prompt must instruct a tool call ("use the write_file tool"),
    else the task is recorded as NOT requiring one (so a no-tool pass is not a violation)."""
    out: dict[str, bool] = {}
    for t in tasks:
        v = t.get("tool_required")
        out[t["task_key"]] = v if isinstance(v, bool) else bool(_REQUIRES_TOOL_RE.search(t.get("prompt", "")))
    return out


def apply_tool_required(traces: list[L2Trace], required: dict[str, bool]) -> int:
    """Set ``tool_required`` on step-bearing traces whose task key (run id ``<key>-r<i>`` -> ``<key>``)
    is in ``required``; every other trace stays None (unobserved). Returns the number set."""
    n = 0
    for t in traces:
        for k in (t.task_key, re.sub(r"-r\d+$", "", t.task_key or "")):
            if k in required:
                t.tool_required = required[k]
                n += 1
                break
    return n


def merge_step_traces(progress: list[L2Trace], steps: list[L2Trace]) -> list[L2Trace]:
    """Progress-log traces whose id has step events are replaced by the step-bearing trace."""
    have = {t.trace_id for t in steps}
    return [t for t in progress if t.trace_id not in have] + steps


# ----------------------------------------------------------------------------- TRAIL
def read_json_tolerant(path: Path | str):
    """json.loads, falling back to removal of trailing commas outside strings.

    Some TRAIL ``processed_annotations_*`` files carry an illegal trailing comma before ``]``/``}``.
    The fallback removes only commas that sit directly before a closing bracket, outside strings, so
    no content is altered.
    """
    s = Path(path).read_text()
    try:
        return json.loads(s)
    except ValueError:
        out, i, n, ins = [], 0, len(s), False
        while i < n:
            c = s[i]
            if ins:
                out.append(c)
                if c == "\\" and i + 1 < n:
                    i += 1
                    out.append(s[i])
                elif c == '"':
                    ins = False
            elif c == '"':
                ins = True
                out.append(c)
            elif c == ",":
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                if not (j < n and s[j] in "]}"):
                    out.append(c)
            else:
                out.append(c)
            i += 1
        return json.loads("".join(out))


def _span_kind(s: dict) -> str | None:
    return (s.get("span_attributes") or {}).get("openinference.span.kind")


def _children(s: dict) -> list[dict]:
    return sorted(s.get("child_spans") or [], key=lambda c: c.get("timestamp") or "")


def load_trail_trace(path: Path | str) -> tuple[L2Trace, dict[str, int]]:
    """TRAIL (PatronusAI) span tree -> ``L2Trace`` plus ``{span_id: step_idx}``.

    Format (verified against the real GAIA and SWE Bench files): ``{"trace_id", "spans": [root]}``
    with nested ``child_spans``; smolagents/openinference attributes in ``span_attributes``.
    A step is a ``Step N`` CHAIN span (nested Steps are separate steps, pre-order), and an LLM span
    outside every Step (planning call) is a ``message`` step. Every span maps to its nearest enclosing
    step. Derived flags, all fixed and named here:

    * ``error``: step status is ``Error`` and no nested Step is itself in error (blame goes to the origin);
    * ``invocation_error``: ``unparseable_code`` when the status message starts with ``AgentParsingError``
      or contains ``Code parsing failed``; ``invalid_arguments`` for a TOOL child ``TypeError`` about an
      unexpected/missing argument;
    * ``tool``/``args``: the first direct TOOL child (kwargs from its JSON input) else ``python_interpreter``
      keyed by a hash of the step's LLM output;
    * ``is_final``: the step has a ``FinalAnswerTool`` descendant;
    * ``truncated`` is not derivable (no finish_reason in these spans) and stays ``None``.
    """
    import hashlib
    d = json.loads(Path(path).read_text())
    steps: list[L2Step] = []
    span_step: dict[str, int] = {}
    step_of: dict[int, dict] = {}

    def desc(s):
        yield s
        for c in _children(s):
            yield from desc(c)

    def visit(s: dict, cur: int | None) -> None:
        sid = s.get("span_id")
        nm = s.get("span_name") or ""
        if _span_kind(s) == "CHAIN" and nm.startswith("Step "):
            idx = len(steps)
            steps.append(L2Step(idx=idx, kind="tool_call"))
            step_of[idx] = s
            cur = idx
        elif cur is None and _span_kind(s) == "LLM":
            idx = len(steps)
            steps.append(L2Step(idx=idx, kind="message"))
            step_of[idx] = s
            cur = idx
        if cur is not None and sid:
            span_step[sid] = cur
        for c in _children(s):
            visit(c, cur)

    for root in d.get("spans", []):
        visit(root, None)

    err_steps = {i for i, s in step_of.items() if s.get("status_code") == "Error"}
    for i, st in enumerate(steps):
        sp = step_of[i]
        if st.kind != "tool_call":
            continue
        a = sp.get("span_attributes") or {}
        nested_err = any(span_step.get(x.get("span_id")) != i and span_step.get(x.get("span_id")) in err_steps
                         for x in desc(sp) if x is not sp)
        msg = sp.get("status_message") or ""
        st.error = (i in err_steps) and not nested_err
        if st.error:
            if msg.startswith("AgentParsingError") or "Code parsing failed" in msg:
                st.invocation_error = "unparseable_code"
        tools = [x for x in desc(sp) if x is not sp and _span_kind(x) == "TOOL" and span_step.get(x.get("span_id")) == i]
        llm = [x for x in desc(sp) if x is not sp and _span_kind(x) == "LLM" and span_step.get(x.get("span_id")) == i]
        if tools:
            st.tool = (tools[0].get("span_name") or "tool")
            try:
                v = json.loads((tools[0].get("span_attributes") or {}).get("input.value") or "{}")
                st.args = v.get("kwargs") if isinstance(v, dict) and isinstance(v.get("kwargs"), dict) else {}
            except ValueError:
                st.args = {}
            tm = tools[0].get("status_message") or ""
            if st.error and re.match(r"TypeError: .*(unexpected keyword|missing \d+ required)", tm):
                st.invocation_error = st.invocation_error or "invalid_arguments"
        else:
            st.tool = "python_interpreter"
            lo = str(((llm[0].get("span_attributes") or {}).get("output.value")) if llm else a.get("input.value") or "")
            st.args = {"llm_output_sha1": hashlib.sha1(lo.encode("utf-8", "replace")).hexdigest()}
        st.observation = str(a.get("output.value") or "")[:4000]
        if any(x.get("span_name") == "FinalAnswerTool" for x in desc(sp)):
            st.is_final = True
    t = L2Trace(trace_id=str(d.get("trace_id")), task_key=str(d.get("trace_id")), outcome="unknown",
                source="trail", steps=steps)
    return t, span_step
