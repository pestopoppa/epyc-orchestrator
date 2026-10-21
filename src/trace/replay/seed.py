"""Initial conditions of a recorded run (HIL-16b).

The per-turn rows hold prompts and outputs; they do NOT hold what a harness needs to rebuild those
prompts from its own state. ``capture_seed`` is called once, at the top of the first recorded turn
(before anything in the turn mutates state) and stores, in the first ``model_turn`` row:

* the JSON-safe declared ``TaskState`` fields (prompt, task id, limits, task IR, tool flags, ...) and the
  large ``context`` string by CAS address;
* the REPL's construction facts: class, session id (random when the caller gave no task id), the
  ``REPLConfig`` scalars and role;
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


def entry_node_name() -> str | None:
    """Class name of the graph node whose ``run`` is executing this turn (walks the awaiting frames)."""
    import sys

    f = sys._getframe(1)
    for _ in range(12):
        if f is None:
            break
        me = f.f_locals.get("self")
        if me is not None and f.f_code.co_name == "run" and type(me).__name__.endswith("Node"):
            return type(me).__name__
        f = f.f_back
    return None


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
        },
        "env": prompt_env_snapshot(),
        "features": feature_snapshot(),
        "paths": path_snapshot(),
    }
