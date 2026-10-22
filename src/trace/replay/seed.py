"""Initial conditions of a recorded run (HIL-16b).

The per-turn rows hold prompts and outputs; they do NOT hold what a harness needs to rebuild those
prompts from its own state. ``capture_seed`` is called once, at the top of the first recorded turn
(before anything in the turn mutates state) and stores, in the first ``model_turn`` row:

* the JSON-safe declared ``TaskState`` fields (prompt, task id, limits, task IR, tool flags, ...) and the
  large ``context`` string by CAS address;
* the REPL's construction facts: class, session id (random when the caller gave no task id), the
  ``REPLConfig`` scalars and role; HIL-16c adds which collaborators it was built with (tool / script
  registry, LLM primitives, retriever, hybrid router, context bundle), its ``tool_context`` and a
  fingerprint of the tool catalogue it exposed (``tool_catalog_fingerprint``), so a replay harness can
  rebuild the same REPL and PROVE it did (a different registry is reported as drift, never hidden);
* the prompt-affecting environment (``ORCHESTRATOR_*``, ``TOOL_DEFINITION_VARIANT``, ``PYTHONHASHSEED``;
  never anything that looks like a secret), the resolved feature flags, and the config paths that end up
  inside prompts (session-log file, spill dir).

Flag-on only; read-only on live state.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

SKIP_STATE = {"context"}
REPL_CONFIG_KEYS = ("timeout_seconds", "output_cap", "spill_dir", "max_grep_results", "require_exploration_before_final",
                    "min_exploration_calls", "use_toon_encoding", "structured_mode")


def _json_safe(v: Any) -> tuple[bool, Any]:
    try:
        return True, json.loads(json.dumps(v, sort_keys=True))
    except Exception:
        return False, None


def feature_snapshot() -> dict[str, Any]:
    from src.features import features

    f = features()
    return {x.name: getattr(f, x.name) for x in dataclasses.fields(f) if isinstance(getattr(f, x.name), (bool, int, float, str))}


def path_snapshot() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from src.config import get_config

        out["tmp_dir"] = str(get_config().paths.tmp_dir)
    except Exception:
        pass
    from src.graph.file_artifacts import SPILL_DIR

    out["spill_dir"] = SPILL_DIR
    return out


def tool_catalog_fingerprint(repl: Any) -> dict[str, Any] | None:
    """What the REPL's tool / script registries expose to the model and to the dispatcher (HIL-16c).

    The role- and ``tool_context``-filtered ``list_tools()`` catalogue (exactly what the model's
    ``list_tools()`` returns), the read-only and chainable tool sets (parallel-dispatch decisions), the
    turn-1 tool grammar for the REPL's role, and the script catalogue; hashed, with the tool names kept for
    diagnosis. ``None`` when the REPL has no tool registry. Never raises (recording must never fail a turn).
    """
    reg = getattr(repl, "tool_registry", None)
    if reg is None:
        return None
    import hashlib

    role, tctx = getattr(repl, "role", None), getattr(repl, "tool_context", None)
    body: dict[str, Any] = {}

    def _try(key: str, fn: Any) -> None:
        try:
            body[key] = fn()
        except Exception as e:   # noqa: BLE001 - a missing surface is part of the fingerprint
            body[key] = f"<unavailable: {type(e).__name__}>"

    _try("tools", lambda: reg.list_tools(role=role, context=tctx))
    _try("read_only", lambda: sorted(reg.get_read_only_tools()))
    _try("chainable", lambda: sorted(reg.get_chainable_tools()))
    gbnf = getattr(reg, "generate_gbnf_grammar", None)
    _try("grammar_sha", lambda: hashlib.sha256(gbnf(str(role)).encode()).hexdigest() if callable(gbnf) else None)
    sreg = getattr(repl, "script_registry", None)
    _try("scripts", lambda: sreg.list_scripts() if sreg is not None else None)
    blob = json.dumps(body, sort_keys=True, default=str)
    tools = body.get("tools")
    names = [t.get("name") for t in tools] if isinstance(tools, list) else []
    return {"sha256": hashlib.sha256(blob.encode()).hexdigest(), "n_tools": len(names), "names": names}


def repl_construction(repl: Any) -> dict[str, Any]:
    """Which collaborators the REPL was constructed with, and its ``tool_context`` (HIL-16c)."""
    ok, tctx = _json_safe(getattr(repl, "tool_context", None))
    return {
        "has": {
            "tool_registry": getattr(repl, "tool_registry", None) is not None,
            "script_registry": getattr(repl, "script_registry", None) is not None,
            "llm_primitives": getattr(repl, "llm_primitives", None) is not None,
            "retriever": getattr(repl, "_retriever", None) is not None,
            "hybrid_router": getattr(repl, "_hybrid_router", None) is not None,
            "context_bundle": getattr(repl, "_context_bundle", None) is not None,
        },
        "tool_context": tctx if ok else repr(getattr(repl, "tool_context", None)),
        "tool_catalog": tool_catalog_fingerprint(repl),
    }


def entry_node_name() -> str | None:
    """Class name of the graph node whose ``run`` is executing this turn (walks the awaiting frames).

    A per-node LangGraph dispatch (``langgraph_<node>`` flags) runs inside the pydantic node's ``run``, so
    that node is still found. A turn driven by the full LangGraph graph (``run_task_lg``: a node FUNCTION in
    ``src/graph/langgraph/nodes.py`` with no pydantic node above it) is named ``"langgraph:<function>"`` so
    the replay harness refuses it explicitly instead of driving a pydantic node in its place (HIL-16c)."""
    import sys

    f = sys._getframe(1)
    lg: str | None = None
    for _ in range(16):
        if f is None:
            break
        me = f.f_locals.get("self")
        if me is not None and f.f_code.co_name == "run" and type(me).__name__.endswith("Node"):
            return type(me).__name__
        code = f.f_code
        if lg is None and code.co_name.endswith("_node") and code.co_filename.replace("\\", "/").endswith("graph/langgraph/nodes.py"):
            lg = f"langgraph:{code.co_name}"
        f = f.f_back
    return lg


def capture_seed(state: Any, deps: Any, role: Any, cas: Any) -> dict[str, Any]:
    from src.trace.replay.record import prompt_env_snapshot

    fields: dict[str, Any] = {}
    for f in dataclasses.fields(state):
        if f.name in SKIP_STATE:
            continue
        ok, val = _json_safe(getattr(state, f.name))
        if ok and (val is None or isinstance(val, (str, int, float, bool, list, dict))):
            fields[f.name] = val
    repl = deps.repl
    cfg = getattr(repl, "config", None)
    arts_ok, arts = _json_safe(dict(getattr(repl, "artifacts", {}) or {}))
    return {
        "state": fields,
        "context_cas": cas.put(str(getattr(state, "context", "") or "")),
        "role": str(role),
        "node": entry_node_name(),
        "repl": {
            "class": type(repl).__name__, "session_id": str(getattr(repl, "_session_id", "")),
            "role": getattr(repl, "role", None),
            "config": {k: getattr(cfg, k) for k in REPL_CONFIG_KEYS if cfg is not None and hasattr(cfg, k)},
            "artifacts": arts if arts_ok else {},
            **repl_construction(repl),     # HIL-16c
        },
        "env": prompt_env_snapshot(),
        "features": feature_snapshot(),
        "paths": path_snapshot(),
    }
