"""The three arms (four variants) of the codemode probe. One episode = (model, arm, task, seed).

A   direct calls: native function calling, STATIC tool list (+ ``list_tools`` / ``call_tool`` for discovery,
    so non-static tools are reachable at the same discovery cost B pays).
B   code mode: the REAL REPL (REPLEnvironment) + the REAL root prompt (PromptBuilder, DEFAULT tools text =
    ``src/prompt_builders/constants.py`` static list, ``list_tools()`` for discovery).
C1  B with the tools text replaced by the top-k schemas from the embedding selector (flag
    ORCHESTRATOR_TOOL_SELECTOR=embed through the builder hook).
C2  B with top-k from the one-shot typed-decision classifier (flag ...=oneshot).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from scripts.benchmark.codemode_ab.tasks import Task
from scripts.benchmark.codemode_ab.world import MUTATING, PROBE_ROLE, World

ARMS = ("A", "B", "C1", "C2")
K_SELECTED = 8
MAX_TURNS_REPL = 12
MAX_TURNS_A = 40
WALL_CAP_S = 900.0
TOOL_RESULT_CAP = 12000

_STATIC_NAME_RE = re.compile(r"^([a-z_][a-z0-9_]*)\(|^CALL\(\"([a-z_][a-z0-9_]*)\"", re.M)


def static_tool_names(world: World) -> list[str]:
    """Tools named in today's LIVE static prompt text (COMPACT_ROOT_LM_TOOLS: ``src/graph/helpers.py`` builds the REPL
    prompt with PromptStyle.MINIMAL) that exist in the world; REPL-only names (llm_call, FINAL, ...) drop out."""
    from src.prompt_builders.constants import COMPACT_ROOT_LM_TOOLS

    have = {t["name"] for t in world.tool_dicts()}
    seen: list[str] = []
    for m in _STATIC_NAME_RE.finditer(COMPACT_ROOT_LM_TOOLS):
        n = m.group(1) or m.group(2)
        if n in have and n not in seen:
            seen.append(n)
    return seen


def _json_schema(spec: dict[str, Any]) -> dict[str, Any]:
    t = spec.get("type", "string")
    s: dict[str, Any] = {"type": t}
    if t == "array":
        s["items"] = {}
    if spec.get("description"):
        s["description"] = spec["description"]
    return s


def openai_tool(t: dict[str, Any]) -> dict[str, Any]:
    props = {k: _json_schema(v) for k, v in (t.get("parameters") or {}).items()}
    req = [k for k, v in (t.get("parameters") or {}).items() if v.get("required")]
    return {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


def _capped(x: Any) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= TOOL_RESULT_CAP else s[:TOOL_RESULT_CAP] + "...[truncated]"


# --- metrics from the audit log --------------------------------------------------------------------
def audit_metrics(world: World) -> dict[str, Any]:
    a = world.audit
    ok = sum(1 for r in a if r["ok"])
    seen: set[str] = set()
    dups = writes = 0
    for r in a:
        if r["tool"] in MUTATING and r["ok"]:
            writes += 1
            key = r["tool"] + json.dumps(r["args"], sort_keys=True, default=str)
            dups += key in seen
            seen.add(key)
    return {"tool_calls": len(a), "tool_calls_ok": ok, "param_accuracy": (ok / len(a)) if a else None,
            "write_calls": writes, "duplicate_writes": dups}


def selector_recall(task: Task, selected: list[str]) -> dict[str, Any]:
    cov = [any(n in selected for n in grp) for grp in task.needed]
    return {"recall": sum(cov) / len(cov), "needed_all_in_topk": all(cov), "topk": selected}


# --- arm A -----------------------------------------------------------------------------------------
A_SYSTEM = ("You are an agent that completes tasks by calling functions. Paths are relative to the working directory. "
            "Call as many functions as you need, in parallel when they are independent. When the task is fully done, "
            "reply with a one-line summary and no function call. Use list_tools to discover other tools and "
            "call_tool(name, arguments) to invoke a tool that is not in your function list.")


def run_arm_a(task: Task, world: World, llm: Any, seed: int) -> dict[str, Any]:
    names = static_tool_names(world)
    by = {t["name"]: t for t in world.tool_dicts()}
    tools = [openai_tool(by[n]) for n in names]
    tools.append({"type": "function", "function": {"name": "list_tools", "description": "Discover ALL available tools (name, description, parameters).",
                                                    "parameters": {"type": "object", "properties": {}, "required": []}}})
    tools.append({"type": "function", "function": {"name": "call_tool", "description": "Invoke any registered tool by name.",
                                                    "parameters": {"type": "object", "properties": {
                                                        "name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}}})
    msgs: list[dict[str, Any]] = [{"role": "system", "content": A_SYSTEM}, {"role": "user", "content": task.prompt}]
    rec = {"in_tokens": 0, "out_tokens": 0, "in_cached": 0, "turns": 0, "failed_calls": 0, "end_reason": "max_turns",
           "static_tools": names, "tools_prompt_chars": len(json.dumps(tools))}
    t0 = time.perf_counter()
    for turn in range(MAX_TURNS_A):
        if time.perf_counter() - t0 > WALL_CAP_S:
            rec["end_reason"] = "wall_cap"
            break
        rep = llm.chat(msgs, tools=tools, max_tokens=4096, seed=seed + turn)
        rec["turns"] += 1
        rec["in_tokens"] += rep.in_tokens
        rec["out_tokens"] += rep.out_tokens
        rec["in_cached"] += rep.in_cached
        if rep.error:
            rec["end_reason"] = "llm_error:" + rep.error[:120]
            break
        if not rep.tool_calls:
            rec["end_reason"] = "final"
            break
        msgs.append({"role": "assistant", "content": rep.content or None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in rep.tool_calls]})
        for c in rep.tool_calls:
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": _capped(_dispatch_a(world, c, rec))})
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def _dispatch_a(world: World, call: dict[str, Any], rec: dict[str, Any]) -> Any:
    name, args = call["name"], call["arguments"]
    if call.get("bad_json"):
        rec["failed_calls"] += 1
        return {"error": "arguments were not valid JSON"}
    try:
        if name == "list_tools":
            return [{"name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in world.tool_dicts()]
        if name == "call_tool":
            return world.call(args["name"], **dict(args.get("arguments") or {}))
        return world.call(name, **args)
    except Exception as exc:  # noqa: BLE001 - tool errors go back to the model as the result
        rec["failed_calls"] += 1
        return {"error": f"{type(exc).__name__}: {exc}"[:400]}


# --- arms B / C (REPL) -----------------------------------------------------------------------------
PEEK_NO_FILE_ERROR = ("peek(n, file_path) needs a file_path: it reads a FILE. The task input is the variable `context`; "
                      "use read_file(path) or peek(n, file_path=path) for files")


def make_repl(world: World, bind_names: Any = None) -> Any:
    from src.repl_environment.environment import REPLEnvironment
    from src.repl_environment.types import REPLConfig

    env = REPLEnvironment(context="", tool_registry=world.registry, role=PROBE_ROLE,
                          config=REPLConfig(timeout_seconds=60, spill_dir="/mnt/raid0/llm/tmp/codemode-ab-20261117/work/repl_spill"))
    g = env._globals  # noqa: SLF001 - replace every filesystem/shell builtin with its sandboxed twin
    g["list_dir"] = lambda path: world.call("list_dir", path=path)
    g["file_info"] = lambda path: world.call("file_info", path=path)
    g["peek"] = lambda n, file_path=None: world.call("peek", n=n, file_path=file_path) if file_path else ""
    g["file_write_safe"] = lambda path, content: world.call("file_write_safe", path=path, content=content)
    g["grep"] = lambda pattern, file_path=None: world.call("grep", pattern=pattern, file_path=file_path)

    def blocked(*_a: Any, **_k: Any) -> Any:
        raise PermissionError("not available in the offline probe sandbox")

    for n in ("run_shell", "run_python_code", "fetch_docs", "ocr_document", "analyze_figure", "extract_figure"):
        if n in g:
            g[n] = blocked
    from src.prompt_builders.codemode_stubs import stubs_enabled

    if stubs_enabled():  # fix 4 (ORCHESTRATOR_CODEMODE_STUBS): picked tools are direct functions returning Python objects
        from src.repl_environment.codemode_tools import bind_stub_tools

        if bind_names:
            bind_stub_tools(g, world.call, world.stub_tool_dicts(), names=set(bind_names), override=True)
        inner_peek = g["peek"]

        def strict_peek(n: Any, file_path: Any = None) -> Any:
            if not file_path:
                raise ValueError(PEEK_NO_FILE_ERROR)
            return inner_peek(n, file_path)

        g["peek"] = strict_peek
    from src.prompt_builders.codemode_prompt import prompt_v2_enabled

    if (stubs_enabled() or prompt_v2_enabled()) and "read_file" not in (bind_names or ()):
        # the sandboxed read_file, so the v2 prompt's / stub's read_file(path) reads the world, not the host
        g["read_file"] = lambda path: world.call("read_file", path=path)
    return env


def run_arm_repl(task: Task, world: World, llm: Any, seed: int, arm: str, selection_block: str | None) -> dict[str, Any]:
    from src.prompt_builders import tool_selector as ts
    from src.prompt_builders.builder import PromptBuilder
    from src.prompt_builders.types import PromptConfig, PromptStyle
    from src.prompt_builders.code_utils import auto_wrap_final, extract_code_from_response

    env = make_repl(world)
    builder = PromptBuilder(PromptConfig(style=PromptStyle.MINIMAL, codemode=True))  # (codemode=True only matters with ORCHESTRATOR_CODEMODE_PROMPT=v2) exactly what src/graph/helpers.py:1414 builds live
    rec = {"in_tokens": 0, "out_tokens": 0, "in_cached": 0, "turns": 0, "failed_snippets": 0, "end_reason": "max_turns",
           "tools_prompt_chars": 0}
    prev_flag = os.environ.get(ts.ENV_FLAG)
    if selection_block is not None:  # arms C1/C2: the flag-gated builder hook, exactly as production would use it
        os.environ[ts.ENV_FLAG] = {"C1": ts.VARIANT_EMBED, "C2": ts.VARIANT_ONESHOT}[arm]
        ts.set_selector_provider(lambda _task: selection_block)
    else:
        os.environ.pop(ts.ENV_FLAG, None)
        ts.set_selector_provider(None)
    last_out = last_err = ""
    from src.prompt_builders.turn_memory import TurnMemory, turn_memory_n

    mem = TurnMemory(turn_memory_n())  # ORCHESTRATOR_REPL_TURN_MEMORY=N (default 0 = stateless, as in v1)
    t0 = time.perf_counter()
    try:
        for turn in range(MAX_TURNS_REPL):
            if time.perf_counter() - t0 > WALL_CAP_S:
                rec["end_reason"] = "wall_cap"
                break
            prompt = builder.build_root_lm_prompt(state=env.get_state(), original_prompt=task.prompt, last_output=last_out,
                                                  last_error=last_err, turn=turn,
                                                  **({"turn_history": mem.entries()} if mem.n and mem.entries() else {}))
            if turn == 0:
                rec["tools_prompt_chars"] = len(builder._resolve_tools()) if selection_block is None else len(selection_block)  # noqa: SLF001
            rep = llm.chat([{"role": "user", "content": prompt}], max_tokens=3072, seed=seed + turn)
            rec["turns"] += 1
            rec["in_tokens"] += rep.in_tokens
            rec["out_tokens"] += rep.out_tokens
            rec["in_cached"] += rep.in_cached
            if rep.error:
                rec["end_reason"] = "llm_error:" + rep.error[:120]
                break
            code = auto_wrap_final(extract_code_from_response(rep.content))
            if not code.strip():
                rec["failed_snippets"] += 1
                last_out, last_err = "", "No code found in your response. Write Python code and end with FINAL()."
                mem.add(turn, rep.content or "", "", last_err)
                continue
            res = env.execute(code)
            last_out, last_err = res.output or "", res.error or ""
            mem.add(turn, code, last_out, last_err)
            if res.error:
                rec["failed_snippets"] += 1
            if res.is_final:
                rec["end_reason"] = "final"
                break
    finally:
        ts.set_selector_provider(None)
        if prev_flag is None:
            os.environ.pop(ts.ENV_FLAG, None)
        else:
            os.environ[ts.ENV_FLAG] = prev_flag
    rec["wall_s"] = time.perf_counter() - t0
    return rec


# --- episode ---------------------------------------------------------------------------------------
def run_episode(task: Task, arm: str, world: World, llm: Any, seed: int, *, selectors: dict[str, Any] | None = None) -> dict[str, Any]:
    """Runs one arm on a freshly set-up world and returns the full unit record (no I/O besides the sandbox)."""
    task.setup(world)
    world.audit.clear()
    sel_info: dict[str, Any] | None = None
    block = None
    if arm in ("C1", "C2"):
        from src.prompt_builders.tool_selector import render_tools_block

        sel = (selectors or {})[arm].select(task.prompt, K_SELECTED)
        block = render_tools_block(world.tool_dicts(), sel.selected)
        sel_info = {**selector_recall(task, sel.selected), "variant": sel.variant, "k": sel.k, "elapsed_ms": sel.elapsed_ms,
                    "selector_in_tokens": sel.selector_in_tokens, "selector_out_tokens": sel.selector_out_tokens,
                    "fallback": sel.fallback, "detail": sel.detail}
    rec = run_arm_a(task, world, llm, seed) if arm == "A" else run_arm_repl(task, world, llm, seed, arm, block)
    ok, detail = task.oracle(world)
    am = audit_metrics(world)
    turns = rec["turns"] or 1
    nfail = rec.get("failed_snippets", 0) if arm != "A" else rec.get("failed_calls", 0)
    return {
        "arm": arm, "task_id": task.task_id, "family": task.family, "n": task.n, "seed": seed, "tier": task.tier,
        "stage": task.stage, "success": bool(ok), "oracle": detail,
        "in_tokens": rec["in_tokens"], "out_tokens": rec["out_tokens"], "in_cached_tokens": rec["in_cached"],
        "selector_in_tokens": (sel_info or {}).get("selector_in_tokens", 0),
        "selector_out_tokens": (sel_info or {}).get("selector_out_tokens", 0),
        "wall_s": round(rec["wall_s"], 3), "turns": rec["turns"], "end_reason": rec["end_reason"],
        "failed_snippets": rec.get("failed_snippets"), "failed_calls": rec.get("failed_calls"),
        "failed_snippet_rate": nfail / turns,  # arm A: failed tool calls per turn; B/C: failed snippets per turn (unit differs, see PREDECLARED)
        "unit_of_turn": "assistant_message" if arm == "A" else "repl_turn",
        **am, "selector": sel_info, "tools_prompt_chars": rec["tools_prompt_chars"],
    }
