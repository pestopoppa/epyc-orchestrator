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

import contextlib
import hashlib
import json
import os
import re
import time
from collections.abc import Iterator
from typing import Any

from scripts.benchmark.codemode_ab.tasks import Task
from scripts.benchmark.codemode_ab.world import MUTATING, PROBE_ROLE, World

ARMS_V1 = ("A", "B", "C1", "C2")
# Probe v2 (PREDECLARED_v2.md). A_pick = the direct-call arm with the HARDENED picker; B1 = B + explicit FINAL + prompt v2;
# B2 = B1 + turn memory; C3 = B2 + typed stubs + hardened picker + pre-exec name repair.
ARMS_V2 = ("A_pick", "B1", "B2", "C3")
ARMS = ARMS_V1 + ARMS_V2
REPL_ARMS = ("B", "C1", "C2", "B1", "B2", "C3")
K_SELECTED = 8
MAX_TURNS_REPL = 12
MAX_TURNS_A = 40
MAX_TURNS_V2 = MAX_TURNS_A  # v2 code-mode arms get arm A's cap (the operator's call; v1 B/C stay at 12)
WALL_CAP_S = 900.0
TOOL_RESULT_CAP = 12000
V2_TOOL_RESULT_CAP = 24000  # discovery cap fix: list_tools() is compact AND its cap is wide enough to include syn_*
V2_PREVIEW_CHARS = 4000  # discovery cap fix (REPL): printed-output preview, so a names-only print of all tools fits
V2_MEMORY_TURNS = 4
TRANSCRIPT_OUTPUT_CAP = 20000

# Per-arm flag environment. Everything is flag-gated production code, so the probe measures exactly what production would switch on.
_F = "ORCHESTRATOR_"
ARM_ENV: dict[str, dict[str, str]] = {
    "A_pick": {_F + "TOOL_SELECTOR": "embed"},
    "B1": {_F + "REPL_EXPLICIT_FINAL": "1", _F + "CODEMODE_PROMPT": "v2"},
    "B2": {_F + "REPL_EXPLICIT_FINAL": "1", _F + "CODEMODE_PROMPT": "v2", _F + "REPL_TURN_MEMORY": str(V2_MEMORY_TURNS)},
    "C3": {_F + "REPL_EXPLICIT_FINAL": "1", _F + "CODEMODE_PROMPT": "v2", _F + "REPL_TURN_MEMORY": str(V2_MEMORY_TURNS),
           _F + "CODEMODE_STUBS": "1", _F + "STRUCTURED_READ_ONLY": "1", _F + "TOOL_SELECTOR": "embed"},
}
# Flags that must NOT leak into a v1 arm run in the same process.
_ALL_FLAGS = sorted({k for env in ARM_ENV.values() for k in env} | {_F + "TOOL_SELECTOR"})


@contextlib.contextmanager
def arm_env(arm: str) -> Iterator[None]:
    """Set exactly the flag environment of ``arm`` (every other codemode flag cleared) and restore on exit."""
    prev = {k: os.environ.get(k) for k in _ALL_FLAGS}
    try:
        for k in _ALL_FLAGS:
            os.environ.pop(k, None)
        os.environ.update(ARM_ENV.get(arm, {}))
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

# Per-turn failure reason (from transcripts; PREDECLARED_v2.md). One value per REPL turn / assistant message.
FAILURE_REASONS = ("ok", "ast_refused", "syntax", "name_error", "tool_arg_error", "tool_error", "no_code", "implicit_final",
                   "explicit_final", "other_error")

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


# --- transcripts + per-turn failure reason ----------------------------------------------------------------
def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def classify_turn(*, code: str, error: str, is_final: bool, wrapped: bool) -> str:
    """One value of ``FAILURE_REASONS`` for a REPL turn. ``wrapped``: auto_wrap_final added the FINAL (the model did not write it)."""
    if not (code or "").strip():
        return "no_code"
    e = error or ""
    if e:
        if "REPLSecurityError" in e or "not allowed" in e or "Dangerous operation" in e or "Forbidden" in e:
            return "ast_refused"
        if "SyntaxError" in e or "IndentationError" in e:
            return "syntax"
        if "NameError" in e or "Unknown tool" in e or "unknown tool" in e:
            return "name_error"
        if "TypeError" in e or "Invalid arguments" in e or "missing required" in e or "unexpected keyword" in e or "needs a file_path" in e:
            return "tool_arg_error"
        if "ToolCallError" in e or "unavailable" in e or "PermissionError" in e or "ToolError" in e or "ValueError" in e or "KeyError" in e:
            return "tool_error"
        return "other_error"
    if is_final:
        return "implicit_final" if wrapped else "explicit_final"
    return "ok"


def _cap(text: str, n: int = TRANSCRIPT_OUTPUT_CAP) -> str:
    return text if len(text) <= n else text[:n] + f"...[+{len(text) - n} chars]"


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
           "static_tools": names, "tools_prompt_chars": len(json.dumps(tools)), "transcript": []}
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
        tturn = {"turn": turn, "prompt_hash": prompt_hash(json.dumps(msgs, default=str)), "content": _cap(rep.content or ""),
                 "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in (rep.tool_calls or [])], "results": [],
                 "error": rep.error or "", "in_tokens": rep.in_tokens, "out_tokens": rep.out_tokens}
        if turn == 0:
            tturn["prompt"] = _cap(json.dumps(msgs, default=str))
        rec["transcript"].append(tturn)
        if rep.error:
            rec["end_reason"] = "llm_error:" + rep.error[:120]
            tturn["failure_reason"] = "other_error"
            break
        if not rep.tool_calls:
            rec["end_reason"] = "final"
            tturn["failure_reason"] = "explicit_final"
            break
        msgs.append({"role": "assistant", "content": rep.content or None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in rep.tool_calls]})
        before = rec["failed_calls"]
        for c in rep.tool_calls:
            out = _capped(_dispatch_a(world, c, rec))
            tturn["results"].append(_cap(out, 2000))
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        tturn["failure_reason"] = "tool_error" if rec["failed_calls"] > before else "ok"
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


# --- arm A_pick: direct calls through the PRODUCTION hardened picker ---------------------------------------
def compact_tools(world: World, desc_chars: int = 80) -> list[dict[str, str]]:
    """Discovery cap fix: ``[{name, description[:80]}]`` for every tool (about 11 kB for 123 tools), so ``syn_*`` tools sit inside the
    first 24000 characters of arm A_pick's list_tools result and inside a names-only print of the REPL ``list_tools()``."""
    return [{"name": t["name"], "description": " ".join(str(t.get("description", "")).split())[:desc_chars]} for t in world.tool_dicts()]


def run_arm_a_pick(task: Task, world: World, llm: Any, seed: int) -> dict[str, Any]:
    """Arm A with the tool list produced by ``tool_selector.select_native_tools_ex`` (core pin, adaptive k, family cap, ladder,
    frozen pick, server-side list_tools / search_tools / describe_tool, one re-pick on an unknown name): the same code path as
    ``openai_compat._run_client_tool_completion``, driven turn by turn. The embedder is whatever native provider the runner registered."""
    from src.prompt_builders import tool_picker as tp
    from src.prompt_builders import tool_selector as ts

    tp.clear_frozen()  # the frozen pick is per conversation: no pick or addition may leak from one episode to the next
    ts.LAST_NATIVE_SELECTION.clear()
    all_tools = [openai_tool(t) for t in world.tool_dicts()]
    msgs: list[dict[str, Any]] = [{"role": "system", "content": A_SYSTEM_PICK}, {"role": "user", "content": task.prompt}]
    rec: dict[str, Any] = {"in_tokens": 0, "out_tokens": 0, "in_cached": 0, "turns": 0, "failed_calls": 0, "end_reason": "max_turns",
                           "static_tools": [], "tools_prompt_chars": 0, "transcript": [], "server_tool_calls": 0, "repicks": 0,
                           "picker": {"called": [], "sent_first": None, "ctx": None}}
    t0 = time.perf_counter()
    for turn in range(MAX_TURNS_A):
        if time.perf_counter() - t0 > WALL_CAP_S:
            rec["end_reason"] = "wall_cap"
            break
        send, ctx = ts.select_native_tools_ex(all_tools, msgs, None, scope="probe")
        if turn == 0:
            rec["tools_prompt_chars"] = len(json.dumps(send))
            rec["picker"]["sent_first"] = [t["function"]["name"] for t in send]
            rec["picker"]["ctx"] = ctx
        rep = llm.chat(msgs, tools=send, max_tokens=4096, seed=seed + turn)
        rec["turns"] += 1
        rec["in_tokens"] += rep.in_tokens
        rec["out_tokens"] += rep.out_tokens
        rec["in_cached"] += rep.in_cached
        tturn = {"turn": turn, "prompt_hash": prompt_hash(json.dumps([msgs, [t["function"]["name"] for t in send]], default=str)),
                 "content": _cap(rep.content or ""), "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in (rep.tool_calls or [])],
                 "results": [], "error": rep.error or "", "in_tokens": rep.in_tokens, "out_tokens": rep.out_tokens,
                 "tools_sent": [t["function"]["name"] for t in send]}
        if turn == 0:
            tturn["prompt"] = _cap(json.dumps(msgs, default=str))
        rec["transcript"].append(tturn)
        if rep.error:
            rec["end_reason"] = "llm_error:" + rep.error[:120]
            tturn["failure_reason"] = "other_error"
            break
        if not rep.tool_calls:
            rec["end_reason"] = "final"
            tturn["failure_reason"] = "explicit_final"
            break
        msgs.append({"role": "assistant", "content": rep.content or None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in rep.tool_calls]})
        before = rec["failed_calls"]
        for c in rep.tool_calls:
            out = _capped(_dispatch_a_pick(world, c, rec, ctx, send))
            tturn["results"].append(_cap(out, 2000))
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
            rec["picker"]["called"].append(c["name"])
        tturn["failure_reason"] = "tool_error" if rec["failed_calls"] > before else "ok"
    rec["wall_s"] = time.perf_counter() - t0
    ctx = rec["picker"]["ctx"]
    if ctx is not None:
        tp.record_outcome(ctx, rec["picker"]["called"])
    rec["picker"]["ctx"] = None
    return rec


def _dispatch_a_pick(world: World, call: dict[str, Any], rec: dict[str, Any], ctx: Any, send: list[dict[str, Any]]) -> Any:
    from src.prompt_builders import tool_picker as tp

    name, args = call["name"], call["arguments"]
    if call.get("bad_json"):
        rec["failed_calls"] += 1
        return {"error": "arguments were not valid JSON"}
    if name in tp.SERVER_TOOLS and ctx is not None:
        rec["server_tool_calls"] += 1
        if name == "list_tools":  # compact + wide cap: syn_* must be inside what the model sees
            return json.dumps(compact_tools(world))[:V2_TOOL_RESULT_CAP]
        return tp.execute_server_tool(name, args, ctx)
    sent = {t["function"]["name"] for t in send}
    if ctx is not None and name not in sent:
        # a function that was not offered: the production route re-picks once server-side; here the call fails with the reason
        # and the pick (frozen set) grows, so the NEXT turn offers it when it exists.
        known = {t["name"] for t in world.tool_dicts()}
        if name in known:
            rec["repicks"] += 1
            tp.frozen_add(ctx.key, [name])
        rec["failed_calls"] += 1
        return {"error": f"function {name!r} is not in your function list; use search_tools(query) or describe_tool(name) first"}
    try:
        return world.call(name, **args)
    except Exception as exc:  # noqa: BLE001 - tool errors go back to the model as the result
        rec["failed_calls"] += 1
        return {"error": f"{type(exc).__name__}: {exc}"[:400]}


A_SYSTEM_PICK = ("You are an agent that completes tasks by calling functions. Paths are relative to the working directory. "
                 "Call as many functions as you need, in parallel when they are independent. When the task is fully done, "
                 "reply with a one-line summary and no function call. If the function you need is not in your list, call "
                 "list_tools to see every tool, search_tools(query) to find one, or describe_tool(name) to get its schema; "
                 "a tool you search for or describe becomes callable on your next step.")


# --- arms B / C (REPL) -----------------------------------------------------------------------------
PEEK_NO_FILE_ERROR = ("peek(n, file_path) needs a file_path: it reads a FILE. The task input is the variable `context`; "
                      "use read_file(path) or peek(n, file_path=path) for files")


def make_repl(world: World, bind_names: Any = None, *, compact_list_tools: bool = False) -> Any:
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
    if compact_list_tools:  # v2 discovery cap fix: name + short description, print names only
        g["list_tools"] = lambda: compact_tools(world)

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


def run_arm_repl(task: Task, world: World, llm: Any, seed: int, arm: str, selection_block: str | None,
                 bind_names: Any = None) -> dict[str, Any]:
    from src.prompt_builders import tool_selector as ts
    from src.prompt_builders.builder import PromptBuilder
    from src.prompt_builders.types import PromptConfig, PromptStyle
    from src.prompt_builders.code_utils import auto_wrap_final, extract_code_from_response
    from scripts.benchmark.codemode_ab.validate import repair_tool_names

    v2 = arm in ARMS_V2
    env = make_repl(world, bind_names, compact_list_tools=v2)
    cfg = PromptConfig(style=PromptStyle.MINIMAL, codemode=True)  # (codemode=True only matters with ORCHESTRATOR_CODEMODE_PROMPT=v2) exactly what src/graph/helpers.py builds live
    if v2:
        cfg.max_output_preview = V2_PREVIEW_CHARS
    builder = PromptBuilder(cfg)
    max_turns = MAX_TURNS_V2 if v2 else MAX_TURNS_REPL
    rec = {"in_tokens": 0, "out_tokens": 0, "in_cached": 0, "turns": 0, "failed_snippets": 0, "end_reason": "max_turns",
           "tools_prompt_chars": 0, "transcript": [], "name_repairs": 0}
    prev_flag = os.environ.get(ts.ENV_FLAG)
    if selection_block is not None:  # arms C1/C2/C3: the flag-gated builder hook, exactly as production would use it
        os.environ[ts.ENV_FLAG] = {"C1": ts.VARIANT_EMBED, "C2": ts.VARIANT_ONESHOT, "C3": ts.VARIANT_EMBED}[arm]
        ts.set_selector_provider(lambda _task: selection_block)
    elif arm not in ARM_ENV or ts.ENV_FLAG not in ARM_ENV[arm]:
        os.environ.pop(ts.ENV_FLAG, None)
        ts.set_selector_provider(None)
    last_out = last_err = ""
    from src.prompt_builders.turn_memory import TurnMemory, turn_memory_n

    mem = TurnMemory(turn_memory_n())  # ORCHESTRATOR_REPL_TURN_MEMORY=N (default 0 = stateless, as in v1)
    known = [t["name"] for t in world.tool_dicts()]
    t0 = time.perf_counter()
    try:
        for turn in range(max_turns):
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
            tturn: dict[str, Any] = {"turn": turn, "prompt_hash": prompt_hash(prompt), "completion": _cap(rep.content or ""),
                                     "in_tokens": rep.in_tokens, "out_tokens": rep.out_tokens}
            tturn["prompt"] = prompt  # every REPL turn's whole prompt (disk is cheap): the smoke gate reads turn 0; memory is auditable from turn 1
            rec["transcript"].append(tturn)
            if rep.error:
                rec["end_reason"] = "llm_error:" + rep.error[:120]
                tturn.update(code="", output="", error=rep.error, failure_reason="other_error")
                break
            extracted = extract_code_from_response(rep.content)
            code = auto_wrap_final(extracted)
            wrapped = code != extracted and "FINAL(" in code and "FINAL(" not in extracted
            if arm == "C3" and code.strip():
                code, repairs = repair_tool_names(code, known)
                rec["name_repairs"] += len(repairs)
                tturn["name_repairs"] = repairs
            tturn.update(code=code, code_extracted=extracted, final_wrapped_by_harness=wrapped)
            if not code.strip():
                rec["failed_snippets"] += 1
                last_out, last_err = "", "No code found in your response. Write Python code and end with FINAL()."
                mem.add(turn, rep.content or "", "", last_err)
                tturn.update(output="", error=last_err, is_final=False,
                             failure_reason=classify_turn(code="", error=last_err, is_final=False, wrapped=False))
                continue
            res = env.execute(code)
            last_out, last_err = res.output or "", res.error or ""
            mem.add(turn, code, last_out, last_err)
            if res.error:
                rec["failed_snippets"] += 1
            tturn.update(output=_cap(last_out), error=last_err, is_final=bool(res.is_final),
                         failure_reason=classify_turn(code=code, error=last_err, is_final=bool(res.is_final), wrapped=wrapped))
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
def final_kind(transcript: list[dict[str, Any]], end_reason: str) -> str:
    """``explicit`` | ``implicit`` | ``none``: how a ``final`` episode ended (REPL arms: was FINAL in the model's own code)."""
    if end_reason != "final" or not transcript:
        return "none"
    last = transcript[-1]
    if "final_wrapped_by_harness" in last:
        return "implicit" if last["final_wrapped_by_harness"] else "explicit"
    return "explicit"  # arm A / A_pick: the model replied without a call, that is its own decision


def pick_for_repl(task: Task, world: World, selectors: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """C3's tool set: the PRODUCTION hardened policy (``tool_picker.decide``) over C1 scores."""
    from src.prompt_builders import tool_picker as tp

    tools = world.tool_dicts()
    names = [t["name"] for t in tools]
    sel = selectors["C1"].select(task.prompt, len(names))
    cfg = tp.PickerConfig.from_env()
    core = tp.core_names(names)
    res = tp.decide(sel.scores, names, core, cfg, lexical=bool(sel.fallback and str(sel.fallback).startswith("lexical")), n_tools=len(names))
    keep = set(res.picks) | set(res.core)
    chosen = [n for n in names if n in keep] if res.mode == "narrow" else (names if res.mode == "full" else list(core))
    return chosen, {**selector_recall(task, chosen), "variant": "hardened_embed", "k": res.k_eff, "elapsed_ms": sel.elapsed_ms,
                    "selector_in_tokens": 0, "selector_out_tokens": 0, "fallback": sel.fallback, "mode": res.mode, "rung": res.rung,
                    "plateau": res.plateau, "core": res.core, "picks": res.picks, "detail": res.detail}


def run_episode(task: Task, arm: str, world: World, llm: Any, seed: int, *, selectors: dict[str, Any] | None = None) -> dict[str, Any]:
    """Runs one arm on a freshly set-up world and returns the full unit record (no I/O besides the sandbox). The per-turn
    transcript is returned under ``_transcript`` (the runner writes it to ``transcripts/`` and removes the key)."""
    with arm_env(arm):
        return _run_episode(task, arm, world, llm, seed, selectors=selectors)


def _run_episode(task: Task, arm: str, world: World, llm: Any, seed: int, *, selectors: dict[str, Any] | None = None) -> dict[str, Any]:
    task.setup(world)
    world.audit.clear()
    sel_info: dict[str, Any] | None = None
    block = None
    bind: list[str] | None = None
    if arm in ("C1", "C2"):
        from src.prompt_builders.tool_selector import render_tools_block

        sel = (selectors or {})[arm].select(task.prompt, K_SELECTED)
        block = render_tools_block(world.tool_dicts(), sel.selected)
        sel_info = {**selector_recall(task, sel.selected), "variant": sel.variant, "k": sel.k, "elapsed_ms": sel.elapsed_ms,
                    "selector_in_tokens": sel.selector_in_tokens, "selector_out_tokens": sel.selector_out_tokens,
                    "fallback": sel.fallback, "detail": sel.detail}
    elif arm == "C3":
        from src.prompt_builders.tool_selector import render_tools_block

        bind, sel_info = pick_for_repl(task, world, selectors or {})
        block = render_tools_block(world.stub_tool_dicts(), bind)
    if arm == "A":
        rec = run_arm_a(task, world, llm, seed)
    elif arm == "A_pick":
        rec = run_arm_a_pick(task, world, llm, seed)
        called = rec["picker"]["called"]
        sent_first = set(rec["picker"]["sent_first"] or [])
        from src.prompt_builders import tool_picker as tp
        from src.prompt_builders import tool_selector as ts

        orow = tp.outcome_row("", "", sorted(sent_first), [], [c for c in called if c not in tp.SERVER_TOOLS])
        last = ts.LAST_NATIVE_SELECTION
        sel_info = {**selector_recall(task, sorted(sent_first)), "variant": "hardened_embed", "k": len(sent_first),
                    "elapsed_ms": float(last.get("elapsed_ms", 0.0)), "rung": last.get("rung"), "mode": last.get("mode"),
                    "selector_in_tokens": 0, "selector_out_tokens": 0, "fallback": None, "proxy_recall": orow["proxy_recall"],
                    "called_not_selected": orow["called_not_selected"], "recovery_fired": bool(rec["server_tool_calls"] or rec["repicks"]),
                    "server_tool_calls": rec["server_tool_calls"], "repicks": rec["repicks"], "breaker_opens": _breaker_opens()}
    else:
        rec = run_arm_repl(task, world, llm, seed, arm, block, bind)
    ok, detail = task.oracle(world)
    am = audit_metrics(world)
    turns = rec["turns"] or 1
    is_a = arm in ("A", "A_pick")
    nfail = rec.get("failed_calls", 0) if is_a else rec.get("failed_snippets", 0)
    transcript = rec.get("transcript", [])
    reasons: dict[str, int] = {}
    for t in transcript:
        r = t.get("failure_reason", "ok")
        reasons[r] = reasons.get(r, 0) + 1
    return {
        "arm": arm, "task_id": task.task_id, "family": task.family, "n": task.n, "seed": seed, "tier": task.tier,
        "stage": task.stage, "success": bool(ok), "oracle": detail,
        "in_tokens": rec["in_tokens"], "out_tokens": rec["out_tokens"], "in_cached_tokens": rec["in_cached"],
        "selector_in_tokens": (sel_info or {}).get("selector_in_tokens", 0),
        "selector_out_tokens": (sel_info or {}).get("selector_out_tokens", 0),
        "wall_s": round(rec["wall_s"], 3), "turns": rec["turns"], "end_reason": rec["end_reason"],
        "failed_snippets": rec.get("failed_snippets"), "failed_calls": rec.get("failed_calls"),
        "failed_snippet_rate": nfail / turns,  # arm A: failed tool calls per turn; B/C: failed snippets per turn (unit differs, see PREDECLARED)
        "unit_of_turn": "assistant_message" if is_a else "repl_turn",
        "turn_failure_reasons": reasons, "final_kind": final_kind(transcript, rec["end_reason"]), "transcript_turns": len(transcript),
        "max_turns": MAX_TURNS_A if is_a else (MAX_TURNS_V2 if arm in ARMS_V2 else MAX_TURNS_REPL),
        **am, "selector": sel_info, "tools_prompt_chars": rec["tools_prompt_chars"], "_transcript": transcript,
    }


def _breaker_opens() -> int:
    from src.prompt_builders import tool_picker as tp

    return tp.BREAKER.opened
