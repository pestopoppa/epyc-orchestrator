"""CM-2 / CM-4 / CM-5 probe arms ON TOP OF C3 (handoff harness-program.md: CM-2, CM-4, CM-5). Measurement only; production code is not touched.

Arms (all paired on (task_id, seed); every arm is the harness's own C3 code path, byte for byte, plus ONE feature):

  C3      the control: explicit FINAL + prompt v2 + typed stubs + structured_read_only + embed picker + tool-name repair + turn memory 4
          (ARM_ENV["C3"], which carries ORCHESTRATOR_REPL_TURN_MEMORY=4; repair_tool_names runs for arm "C3"; ORCHESTRATOR_REPL_TOOL_NAME_REPAIR=1
          is also set so the production flag, once merged, agrees with the harness repair). The three CM flags are explicitly UNSET.
  C3_cm2  C3 + ORCHESTRATOR_REPL_TOOL_SEARCH=1 and the SEARCH-FIRST prompt shape (see below).
  C3_cm4  C3 + ORCHESTRATOR_REPL_CLASSIFY=1 + a prompt addendum documenting classify / route_classify / risk_bands.
  C3_cm5  C3 + ORCHESTRATOR_REPL_HANDLES=1 + a prompt addendum documenting store / load / handles.

Three facts the arms are built around (each verified against the code, see PREDECLARED_cm.md):

  1. With the picker on (C3 has it), ``codemode_tools.install`` ALREADY installs search_tools / describe_tool and the C3 prompt already names them
     (``search_on = flag(ENV_SEARCH) or selector_variant() != "off"``), so ORCHESTRATOR_REPL_TOOL_SEARCH=1 alone is byte-identical to C3. The only thing CM-2
     can add is the shape it was written for: discovery by search instead of picked schemas. C3_cm2 therefore forces the picker's EXISTING ``core_search`` rung
     (core tools only, the rest by search_tools / describe_tool) and says so in one prompt line.
  2. No prompt names classify / route_classify / risk_bands / store / load / handles. A flag with no prompt text is invisible to the model, so the arms add
     the documentation line that production would have to ship with the flag; that line is part of the arm and is counted in its tokens.
  3. A REPL checkpoint already restores user variables, so CM-5's marginal value over plain restored variables is the open question; the two-phase
     H-PRICE task checkpoints after phase 1 and restores into a FRESH REPL (production ``restore``) for phase 2 in every arm.

Sub-calls: ``classify`` makes one model call per item through a counting wrapper of the probe's own LLM client; those tokens are recorded as
``sub_in_tokens`` / ``sub_out_tokens`` and included in ``total_tokens_all``, so a feature cannot win by moving cost off the root prompt.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
from collections.abc import Iterator
from typing import Any

from scripts.benchmark.codemode_ab import arms as A
from scripts.benchmark.codemode_ab.tasks import Task
from src.repl_environment import codemode_tools as CT

ARMS_CM = ("C3", "C3_cm2", "C3_cm4", "C3_cm5")
CM_FLAGS = (CT.ENV_SEARCH, CT.ENV_CLASSIFY, CT.ENV_HANDLES)
NAME_REPAIR_ENV = "ORCHESTRATOR_REPL_TOOL_NAME_REPAIR"
CM_ENV: dict[str, dict[str, str]] = {"C3": {}, "C3_cm2": {CT.ENV_SEARCH: "1"}, "C3_cm4": {CT.ENV_CLASSIFY: "1"}, "C3_cm5": {CT.ENV_HANDLES: "1"}}
CELL_TIMEOUT_S = 600  # the production REPL session timeout; the earlier probe rows used 60 (C3 cells take seconds, so this is inert for the control)
CM_GLOBALS = ("search_tools", "describe_tool", "classify", "route_classify", "risk_bands", "q_scores", "store", "load", "handles")

ADDENDUM: dict[str, str] = {
    "C3": "",
    "C3_cm2": ("- Only the core file tools are listed above. For ANY other tool call `search_tools(query)` (tool names, best first) and "
               "`describe_tool(name)` (its schema), then invoke it with `TOOL(name, **kw)` (Python object) or `CALL(name, **kw)`."),
    "C3_cm4": ("- `classify(items, labels, instruction='')`: labels every item with a model call WITHOUT putting the items in your prompt; returns one "
               "label (or None) per item, in order. `route_classify(items)` returns one dict per item with the system's routing labels "
               "(`role`, `strategy`, `confidence`, `matched_keywords`, `trinity_role`, `trinity_reason`); `risk_bands(items)` returns one dict per item "
               "(`risk_band`, `risk_score`, `difficulty_band`, `difficulty_score`). Items are strings, at most 200 per call."),
    "C3_cm5": ("- `store(name, value)` keeps a JSON value (at most 256 KiB) under a name and returns its handle `handle:<name>`; `load(handle_or_name)` "
               "returns it; `handles()` lists what is stored. A value is kept only if the cell that stored it succeeds. Stored values survive a session "
               "restore: pass the handle on instead of printing or re-reading a large value."),
}
_USE = {n: re.compile(rf"\b{n}\s*\(") for n in CM_GLOBALS}


class CountingPrimitives:
    """The typed-decisions seam (``llm_call``) over the probe's LLM client, counting every sub-call's tokens."""

    def __init__(self, llm: Any):
        self.llm = llm
        self.calls = self.in_tokens = self.out_tokens = self.errors = 0

    def llm_call(self, prompt: str, *, role: str = "worker", n_tokens: int = 512, json_schema: dict[str, Any] | None = None,
                 temperature: float = 0.0, seed: int = 0, **_: Any) -> str:
        rep = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=n_tokens, seed=seed, json_schema=json_schema)
        self.calls += 1
        self.in_tokens += rep.in_tokens
        self.out_tokens += rep.out_tokens
        if rep.error:
            self.errors += 1
            raise RuntimeError(rep.error)
        return rep.content


@dataclasses.dataclass
class _Ctx:
    arm: str
    llm: Any
    prims: CountingPrimitives
    env: Any = None
    ckpt: dict[str, Any] | None = None
    restored: bool = False
    phases_run: int = 0
    globals_present: set[str] = dataclasses.field(default_factory=set)
    forced_core: bool = False


@contextlib.contextmanager
def _patched(ctx: _Ctx) -> Iterator[None]:
    """Patch the harness module for ONE episode and restore it (every attribute) afterwards."""
    from src.prompt_builders import tool_picker as tp
    from src.prompt_builders import tool_selector as ts

    orig_make, orig_pick, orig_render, orig_run = A.make_repl, A.pick_for_repl, ts.render_tools_block, A.run_arm_repl

    def make_repl(world: Any, bind_names: Any = None, *, compact_list_tools: bool = False) -> Any:
        env = orig_make(world, bind_names, compact_list_tools=compact_list_tools)
        env.config.timeout_seconds = CELL_TIMEOUT_S
        if ctx.arm == "C3_cm4":  # the production install() adds `classify` only when the REPL has LLM primitives: give it the counting seam
            env.llm_primitives = ctx.prims
            CT.install(env, env._globals, environ={CT.ENV_CLASSIFY: "1"})  # noqa: SLF001 - the production installer, classify flag only
            env._builtin_global_keys = frozenset(env._globals.keys())  # noqa: SLF001 - installed functions are engine builtins, not user variables
        if ctx.ckpt is not None:  # phase 2: the production restore into this FRESH REPL, then re-pin the probe's sandboxed builtins (restore rebuilds globals)
            pinned = dict(env._globals)  # noqa: SLF001
            env.restore(ctx.ckpt)
            for k in env._builtin_global_keys:  # noqa: SLF001
                if k in pinned:
                    env._globals[k] = pinned[k]  # noqa: SLF001
            ctx.ckpt, ctx.restored = None, True
        ctx.env = env
        ctx.globals_present |= {n for n in CM_GLOBALS if n in env._globals}  # noqa: SLF001
        return env

    def pick_for_repl(task: Task, world: Any, selectors: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        chosen, info = orig_pick(task, world, selectors)
        if ctx.arm != "C3_cm2":
            return chosen, info
        core = tp.core_names([t["name"] for t in world.tool_dicts()])
        ctx.forced_core = True
        return list(core), {**info, **A.selector_recall(task, list(core)), "mode": "core_search_forced", "picks_unforced": info.get("picks"),
                            "chosen_unforced": chosen, "k": 0}

    def render_tools_block(tools: Any, selected: Any) -> str:
        block = orig_render(tools, selected)
        add = ADDENDUM[ctx.arm]
        return block + ("\n" + add if add else "")

    def run_arm_repl(task: Task, world: Any, llm: Any, seed: int, arm: str, selection_block: Any, bind_names: Any = None) -> dict[str, Any]:
        if not task.phases:
            ctx.phases_run = 1
            return orig_run(task, world, llm, seed, arm, selection_block, bind_names)
        merged: dict[str, Any] | None = None
        for i, prompt in enumerate(task.phases):
            if i:
                ctx.ckpt = ctx.env.checkpoint()  # a branched session: phase 2 starts from a checkpoint of phase 1
                world.blocked |= set(task.blocked_in_phase2)
            rec = orig_run(dataclasses.replace(task, prompt=prompt), world, llm, seed, arm, selection_block, bind_names)
            ctx.phases_run = i + 1
            for t in rec["transcript"]:
                t["phase"] = i + 1
            if merged is None:
                merged = rec
            else:
                for k in ("in_tokens", "out_tokens", "in_cached", "turns", "failed_snippets", "name_repairs", "wall_s"):
                    merged[k] = merged.get(k, 0) + rec.get(k, 0)
                merged["transcript"] = merged["transcript"] + rec["transcript"]
                merged["end_reason"] = rec["end_reason"]
            if str(rec["end_reason"]).startswith("llm_error"):
                merged["end_reason"] = rec["end_reason"]
                break
        assert merged is not None
        return merged

    A.make_repl, A.pick_for_repl, ts.render_tools_block, A.run_arm_repl = make_repl, pick_for_repl, render_tools_block, run_arm_repl
    try:
        yield
    finally:
        A.make_repl, A.pick_for_repl, ts.render_tools_block, A.run_arm_repl = orig_make, orig_pick, orig_render, orig_run


def code_uses(transcript: list[dict[str, Any]]) -> dict[str, int]:
    """How often the model's own code called each CM function (the proof an arm's feature was actually exercised)."""
    uses = {n: 0 for n in CM_GLOBALS}
    for t in transcript:
        code = str(t.get("code") or "")
        for n, rx in _USE.items():
            uses[n] += len(rx.findall(code))
    return {k: v for k, v in uses.items() if v}


def run_episode_cm(task: Task, arm: str, world: Any, llm: Any, seed: int, *, selectors: dict[str, Any] | None = None) -> dict[str, Any]:
    """One CM-probe episode: the harness's C3 path with the arm's flag, prompt line and (cm2) forced core_search pick. Returns the unit record
    (arm = the CM arm label) with ``cm`` provenance and the sub-call token totals; ``_transcript`` is left for the runner."""
    if arm not in ARMS_CM:
        raise ValueError(f"unknown CM arm {arm!r}; known {ARMS_CM}")
    assert A.ARM_ENV["C3"].get("ORCHESTRATOR_REPL_TURN_MEMORY") == str(A.V2_MEMORY_TURNS) == "4", "C3 must carry turn memory 4"
    ctx = _Ctx(arm=arm, llm=llm, prims=CountingPrimitives(llm))
    keys = CM_FLAGS + (NAME_REPAIR_ENV,)
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in CM_FLAGS:
            os.environ.pop(k, None)
        os.environ[NAME_REPAIR_ENV] = "1"
        os.environ.update(CM_ENV[arm])
        with _patched(ctx):
            rec = A.run_episode(task, "C3", world, llm, seed, selectors=selectors)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rec["arm"] = arm
    rec["turn_memory_n"] = A.V2_MEMORY_TURNS
    rec["sub_calls"], rec["sub_in_tokens"], rec["sub_out_tokens"] = ctx.prims.calls, ctx.prims.in_tokens, ctx.prims.out_tokens
    rec["total_tokens_all"] = int(rec.get("in_tokens") or 0) + int(rec.get("out_tokens") or 0) + ctx.prims.in_tokens + ctx.prims.out_tokens
    rec["cm"] = {"flags": dict(CM_ENV[arm]), "name_repair_env": "1", "globals_present": sorted(ctx.globals_present), "uses": code_uses(rec.get("_transcript") or []),
                 "addendum_chars": len(ADDENDUM[arm]), "forced_core_search": ctx.forced_core, "phases_run": ctx.phases_run, "restored_into_fresh_repl": ctx.restored,
                 "sub_call_errors": ctx.prims.errors, "cell_timeout_s": CELL_TIMEOUT_S}
    return rec
