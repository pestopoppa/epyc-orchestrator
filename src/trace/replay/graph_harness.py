"""Production ``Harness`` binding: the replay driver driving the REAL graph turn loop (HIL-16b).

``GraphHarness`` rebuilds a run's state from the recorded seed (``seed.py``) and then runs the real graph
nodes (``FrontdoorNode.run`` etc., hence the real ``_execute_turn`` and the real post-turn state
transitions) with a replay LLM in place of the backend. One node run is one driver turn:

* ``build_prompt()`` starts the node on a private event-loop thread and returns the prompt the real
  ``_execute_turn`` is about to send; the fake ``llm_call`` then parks until ``execute()`` hands it the
  model output (recorded output for turns < k; the live model's from turn k).
* Registry-tool calls are served from the CAS (never executed, so no side effects) and given back as the
  same kind of object that was recorded (str / JSON value / ``ToolOutput`` envelope / the same exception
  class name). Builtins (peek, grep, ...) run for real and their output is compared with the recording.
* Prompt-determinism hazards are neutralised ONLY here, from recorded values: the wall-clock reads (tape),
  the wall-time forced-answer decision, the random REPL session id, the prompt-affecting environment and
  feature flags, the per-call inference meta, and auxiliary model calls (session summary, consolidation,
  compaction), which are replayed from the recording. Anything that still differs is a ``PrefixDivergence``
  or an ``environment_drift`` report; it is never papered over.

Not neutralisable after the fact: ``repr()`` of a ``set`` of strings in REPL variables (order depends on the
recording process's hash seed, which is only recoverable when ``PYTHONHASHSEED`` was pinned at recording).
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
import queue
import threading
from types import SimpleNamespace
from typing import Any

from src.trace.replay.cas import CAS
from src.trace.replay.clock import ClockTape, activate as _activate_clock
from src.trace.replay.driver import Outcome, PrefixDivergence, ToolCall, ToolServer
from src.trace.replay.model import RecordedRun
from src.trace.step_events import COMPOSITE_BUILTINS, _text_of

_KEEP_ENV = ("ORCHESTRATOR_REPLAY_CAS_DIR",)
_TELEMETRY_ENV = ("ORCHESTRATOR_REPLAY_RECORDING", "ORCHESTRATOR_STEP_EVENT_LOGGING", "ORCHESTRATOR_OBSERVED_DEPENDENCY_LOGGING")
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_TURN_TIMEOUT_S = 900.0


class ServedToolRegistry:
    """Wraps the REPL's tool registry (everything but ``invoke`` delegates to it, so tool catalogues and
    prompt blocks are unchanged). While replaying a recorded turn ``invoke`` is answered from the CAS and
    the handler is never executed; the REPL's own bookkeeping around the call (research context, receipts,
    invocation records) still runs, so later prompts see the same state. After the restart it is live."""

    def __init__(self, serve: Any = None, real: Any = None) -> None:
        self._serve, self._real = serve, real

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or self._real is None:
            raise AttributeError(name)
        return getattr(self._real, name)

    def list_tools(self, role: Any = None, context: Any = None) -> list:
        return self._real.list_tools(role=role, context=context) if self._real is not None else []

    def get_read_only_tools(self) -> set:
        return self._real.get_read_only_tools() if self._real is not None else set()

    def get_chainable_tools(self) -> set:
        return self._real.get_chainable_tools() if self._real is not None else set()

    def invoke(self, tool_name: str, role: Any = None, caller_type: str = "direct", chain_id: Any = None,
               chain_index: int = 0, context: Any = None, **kwargs: Any) -> Any:
        if self._serve is not None:
            handled, value = self._serve(tool_name, kwargs)
            if handled:
                return value
        if self._real is None:
            raise RuntimeError(f"replay: tool {tool_name!r} is not recorded at this point of the run")
        return self._real.invoke(tool_name, role, caller_type=caller_type, chain_id=chain_id,
                                 chain_index=chain_index, context=context, **kwargs)


def default_repl_factory(seed: dict, context: str) -> Any:
    from src.repl_environment import REPLConfig, REPLEnvironment
    from src.repl_environment import environment as env_mod

    r = seed["repl"]
    cls = getattr(env_mod, r.get("class") or "REPLEnvironment", REPLEnvironment)
    return cls(context=context, artifacts=copy.deepcopy(r.get("artifacts")) or None, config=REPLConfig(**r.get("config", {})),
               role=r.get("role"), task_id=r["session_id"], tool_registry=ServedToolRegistry())   # no real handlers: replay only


class _Prims:
    """Minimal LLMPrimitives surface used by ``_execute_turn``; ``llm_call`` is the replay LLM."""

    def __init__(self, llm_call: Any) -> None:
        self.llm_call = llm_call
        self._early_stop_check = None
        self._last_inference_meta: dict = {}
        self._backends = True
        self.total_tokens_generated = 0
        self.total_prompt_eval_ms = 0.0
        self.total_generation_ms = 0.0
        self._last_predicted_tps = 0.0
        self.total_http_overhead_ms = 0.0

    def get_cache_stats(self) -> dict:
        return {"hits": 0, "misses": 0}


class _BuiltinTap:
    """Stands in for the REPL's step-event log: only builtin events matter for replay."""

    def __init__(self, on_builtin: Any) -> None:
        self._on = on_builtin
        self.ordinal = 0

    def in_call(self) -> bool:
        return False

    def observe_builtin(self, function: str, args: Any, result: Any) -> None:
        self._on(function, args, result)

    def observe_call(self, *a: Any, **k: Any) -> None:
        return None


class GraphHarness:
    def __init__(self, cas: CAS | None = None, *, repl_factory: Any = None) -> None:
        self.cas = cas or CAS()
        self.repl_factory = repl_factory or default_repl_factory
        self.run: RecordedRun | None = None
        self.warnings: list[str] = []
        self._saved_env: dict[str, str | None] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tools: ToolServer | None = None
        self._live_llm: Any = None
        self._div: PrefixDivergence | None = None
        self._fut: Any = None
        self._out_q: queue.Queue = queue.Queue()
        self._prompt_evt = threading.Event()
        self._prompt = ""
        self._n = 0
        self._llm_calls = 0

    # ----------------------------------------------------------------------------- lifecycle
    def bind_tools(self, tools: ToolServer) -> None:
        self._tools = tools

    def bind_llm(self, llm: Any) -> None:
        self._live_llm = llm

    def reset(self, run: RecordedRun) -> None:
        self.close()
        self.run, self._n, self._div, self.warnings = run, 0, None, []
        seed = run.seed
        if not seed:
            self._div = PrefixDivergence(0, "no_seed", "recording has no seed row (made before HIL-16b?)")
            return
        self._apply_env(seed)
        drift = self._drift(seed)
        if drift:
            self._div = PrefixDivergence(0, "environment_drift", "; ".join(drift))
        self._build(seed)
        self._start_loop()

    def close(self) -> None:
        self._out_q.put(None)   # release a parked llm_call
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
        self._restore_env()
        self._fut = None
        self._out_q = queue.Queue()

    # ----------------------------------------------------------------------------- environment
    def _apply_env(self, seed: dict) -> None:
        want = {k: v for k, v in (seed.get("env") or {}).items() if k != "PYTHONHASHSEED" and k not in _TELEMETRY_ENV}
        keys = set(want) | {k for k in os.environ if k.startswith("ORCHESTRATOR_") and k not in _KEEP_ENV
                            and not any(w in k.upper() for w in _SECRET_WORDS)} | set(_TELEMETRY_ENV)
        for k in keys:
            self._saved_env.setdefault(k, os.environ.get(k))
            if k in want:
                os.environ[k] = want[k]
            else:
                os.environ.pop(k, None)
        pinned = (seed.get("env") or {}).get("PYTHONHASHSEED")
        if pinned is None:
            self.warnings.append("recording process did not pin PYTHONHASHSEED: str-set reprs in REPL state may not replay")
        elif os.environ.get("PYTHONHASHSEED") != pinned:
            self.warnings.append(f"PYTHONHASHSEED differs (recorded {pinned!r}); relaunch with it set")
        from src.features import reset_features

        reset_features()

    def _restore_env(self) -> None:
        if not self._saved_env:
            return
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._saved_env = {}
        from src.features import reset_features

        reset_features()

    def _drift(self, seed: dict) -> list[str]:
        from src.trace.replay.seed import feature_snapshot, path_snapshot

        out: list[str] = []
        cur_f, rec_f = feature_snapshot(), seed.get("features") or {}
        for k in sorted(set(rec_f) | set(cur_f)):
            if k in ("replay_recording", "step_event_logging", "observed_dependency_logging"):
                continue
            if rec_f.get(k) != cur_f.get(k):
                out.append(f"feature {k}: recorded {rec_f.get(k)!r} now {cur_f.get(k)!r}")
        cur_p, rec_p = path_snapshot(), seed.get("paths") or {}
        for k in sorted(rec_p):
            if cur_p.get(k) != rec_p[k]:
                out.append(f"path {k}: recorded {rec_p[k]!r} now {cur_p.get(k)!r}")
        return out

    # ----------------------------------------------------------------------------- state
    def _build(self, seed: dict) -> None:
        from pydantic_graph import GraphRunContext

        from src.graph import nodes as graph_nodes
        from src.graph.state import TaskDeps, TaskState
        from src.roles import Role

        st = copy.deepcopy(seed["state"])   # nested lists/dicts must not leak between replays
        st["turns"] = max(0, int(st.get("turns", 1)) - 1)    # the seed was taken after the turn counter moved
        st["answer_force_at_s"] = None
        names = {f.name for f in dataclasses.fields(TaskState) if f.init}
        st = {k: v for k, v in st.items() if k in names}
        try:
            st["current_role"] = Role(st.get("current_role", seed.get("role")))
        except Exception:
            pass
        context = self.cas.get_text(seed["context_cas"]) if seed.get("context_cas") else ""
        self.state = TaskState(context=context, **st)
        self.prims = _Prims(self._llm)
        self.repl = self.repl_factory(seed, context)
        tap = _BuiltinTap(self._on_builtin)
        self.repl._exploration_log.step_events = tap
        self._wrap_tools()
        self.deps = TaskDeps(primitives=self.prims, repl=self.repl)
        self.ctx = GraphRunContext(state=self.state, deps=self.deps)
        name = seed.get("node") or "FrontdoorNode"
        cls = getattr(graph_nodes, name, None)
        if cls is None:
            self._div = self._div or PrefixDivergence(0, "unknown_entry_node", name)
            cls = graph_nodes.FrontdoorNode
        self.node: Any = cls()
        self.tape = ClockTape(replay=[t for turn in self.run.turns for t in turn.clock])  # type: ignore[union-attr]

    # ----------------------------------------------------------------------------- tool serving
    def _wrap_tools(self) -> None:
        repl = self.repl
        real = repl.tool_registry
        if isinstance(real, ServedToolRegistry):
            real._serve = self._serve_tool
        else:
            repl.tool_registry = ServedToolRegistry(self._serve_tool, real)

    def _serve_tool(self, tool_name: str, kwargs: dict) -> tuple[bool, Any]:
        tools = self._tools
        if tools is None or tools.live:
            if tools is not None:
                tools.names.append(tool_name)
            return False, None
        try:
            text = tools.call(tool_name, kwargs)
        except PrefixDivergence as e:
            self._stash(e)
            raise
        return True, self._rebuild(text, tools.last)

    def _rebuild(self, text: str, rec: Any) -> Any:
        extra = (rec.extra if rec is not None else {}) or {}
        exc = extra.get("exc")
        if exc:
            msg = self.cas.get_text(exc["message_cas"]) if exc.get("message_cas") else ""
            import builtins

            known = getattr(builtins, exc["type"], None)
            cls = known if isinstance(known, type) and issubclass(known, Exception) else type(exc["type"], (Exception,), {})
            raise cls(msg)
        kind = extra.get("output_kind", "str")
        payload: Any = None if kind == "none" else json.loads(text) if kind == "json" else text
        env = extra.get("envelope")
        if env:
            from src.registry.tool_registry import ToolOutput

            return ToolOutput(ok=env["ok"], status=env["status"], output=payload)
        return payload

    def _on_builtin(self, function: str, args: Any, result: Any) -> None:
        tools = self._tools
        if tools is None or function in COMPOSITE_BUILTINS:
            return
        if tools.live:
            tools.names.append(function)
            return
        try:
            want = tools.call(function, args)
        except PrefixDivergence as e:
            self._stash(e)
            return
        got = _text_of(result)
        if got != want:
            self._stash(PrefixDivergence(tools._turn, "builtin_output_differs", f"{function}: {len(want)} vs {len(got)} chars"))

    # ----------------------------------------------------------------------------- the replay LLM
    def _stash(self, e: PrefixDivergence) -> None:
        self._div = self._div or e

    def _turn_rec(self):
        return self.run.turns[self._n] if self.run is not None and self._n < len(self.run.turns) else None

    def _llm(self, prompt: str, role: Any = None, **kw: Any) -> str:
        t = self._turn_rec()
        turn_no = t.turn if t is not None else self._n
        aux = t.aux if t is not None else []
        i = self._llm_calls
        self._llm_calls += 1
        live = self._tools is not None and self._tools.live
        if i < len(aux):
            # An auxiliary call whose prompt is byte-identical to the recorded one is answered from the
            # recording even after the restart (it belongs to the unchanged state before the intervention);
            # a differing one is a divergence in replay and a live call after the restart.
            same = prompt.encode("utf-8") == self.cas.get(aux[i]["prompt_cas"])
            if same:
                return self.cas.get_text(aux[i]["output_cas"])
            if live and self._live_llm is not None:
                return self._live_llm(prompt, role=role, **kw)
            e = PrefixDivergence(turn_no, "aux_prompt_differs", f"auxiliary call {i}")
            self._stash(e)
            raise e
        if i > len(aux):
            if live and self._live_llm is not None:
                return self._live_llm(prompt, role=role, **kw)
            e = PrefixDivergence(turn_no, "unrecorded_llm_call", "second main call in one turn (reasoning retry?) is not recorded")
            self._stash(e)
            raise e
        self._prompt = prompt
        self.prims._last_inference_meta = dict(t.meta) if t is not None and t.meta else {}
        self._prompt_evt.set()
        out = self._out_q.get()
        if out is None:
            raise RuntimeError("replay harness closed")
        return out

    # ----------------------------------------------------------------------------- driver protocol
    def _start_loop(self) -> None:
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True, name="replay-graph-harness").start()
        self._loop = loop

    def _raise_stashed(self) -> None:
        if self._div is not None:
            raise self._div

    def build_prompt(self) -> str:
        self._raise_stashed()
        assert self._loop is not None and self.run is not None
        t = self._turn_rec()
        if t is not None and isinstance(t.force_answer, bool):
            self.state._replay_force_answer = t.force_answer    # type: ignore[attr-defined]
        else:
            self.state._replay_force_answer = None              # type: ignore[attr-defined]
        self._llm_calls = 0
        self._prompt_evt.clear()
        node, ctx, tape = self.node, self.ctx, self.tape

        async def _go() -> Any:
            _activate_clock(tape)
            return await node.run(ctx)

        self._fut = asyncio.run_coroutine_threadsafe(_go(), self._loop)
        while not self._prompt_evt.wait(0.02):
            if self._fut.done():
                break
        self._raise_stashed()
        if not self._prompt_evt.is_set():
            err = self._fut.exception()
            raise PrefixDivergence(t.turn if t else self._n, "turn_ended_before_llm", repr(err) if err else "node returned without a model call")
        return self._prompt

    def execute(self, model_output: str, tools: ToolServer) -> Outcome:
        from pydantic_graph import End

        self._tools = tools
        self._out_q.put(model_output)
        try:
            nxt = self._fut.result(timeout=_TURN_TIMEOUT_S)
        except Exception as e:
            self._raise_stashed()
            raise PrefixDivergence(self._n, "turn_failed", repr(e)) from e
        self._n += 1
        self._raise_stashed()
        calls = [ToolCall(n, "", "") for n in tools.names]
        if isinstance(nxt, End):
            data = nxt.data
            return Outcome(True, getattr(data, "answer", None), calls)
        self.node = nxt
        return Outcome(False, None, calls)


def make_harness() -> GraphHarness:
    """``--harness src.trace.replay.graph_harness:make_harness`` for ``scripts/trace/replay_flip_run.py``."""
    return GraphHarness(CAS())
