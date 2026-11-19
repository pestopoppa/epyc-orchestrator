"""UFH14-TOOLS-1: count and cap the model-visible tool surface per role.

Two independent, default-OFF feature flags:

* ``tool_surface_logging``: one JSONL record per REPL turn (cell) with the number of REPL-provided
  globals and registry tools the role can see, so the per-role surface cost is measured, not assumed.
  Records go to ``logs/tool_surface/tool_surface.jsonl`` (override ``ORCHESTRATOR_TOOL_SURFACE_LOG_DIR``).
* ``tool_surface_cap``: ``list_tools()`` returns only the per-role allowlist from
  ``orchestration/tool_surface_allowlist.yaml`` (deferred loading). Tools outside the allowlist stay
  callable and discoverable via ``list_tools(deferred=True)``; nothing is removed from the registry or
  from permission checks. A role with no allowlist entry sees everything (no cap).

Pure helpers; nothing here changes behaviour when both flags are off.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = "epyc.orchestrator.tool_surface.v1"
_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = _ROOT / "orchestration" / "tool_surface_allowlist.yaml"

_lock = threading.Lock()
_cache: dict[str, Any] = {"mtime": None, "path": None, "data": {}}


def load_allowlists(path: Path | None = None) -> dict[str, frozenset[str]]:
    """role -> allowed registry tool names. Missing/invalid file means no caps (fail open: a cap is
    an optimisation, never a reason to withhold tools)."""
    p = Path(os.environ.get("ORCHESTRATOR_TOOL_SURFACE_ALLOWLIST", "") or (path or ALLOWLIST_PATH))
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    with _lock:
        if _cache["path"] == str(p) and _cache["mtime"] == mtime:
            return _cache["data"]
    try:
        import yaml

        raw = yaml.safe_load(p.read_text()) or {}
        roles = raw.get("roles") or {}
        data = {str(r): frozenset(str(t) for t in (names or [])) for r, names in roles.items()}
    except Exception:
        data = {}
    with _lock:
        _cache.update(path=str(p), mtime=mtime, data=data)
    return data


def split_tools(role: str | None, tools: list[Any]) -> tuple[list[Any], list[Any]]:
    """(visible, deferred) for a registry tool-info list under the role's allowlist."""
    allow = load_allowlists().get(role or "")
    if allow is None:
        return list(tools), []
    visible, deferred = [], []
    for t in tools:
        name = t.get("name") if isinstance(t, dict) else getattr(t, "name", str(t))
        (visible if name in allow else deferred).append(t)
    return visible, deferred


def surface_counts(env: Any) -> dict[str, Any]:
    """Counts of what ``env``'s role can see right now (no side effects)."""
    from src.repl_environment.tool_broker import exposed_surface

    repl_globals = sorted(exposed_surface(env))
    registry: list[Any] = []
    if getattr(env, "tool_registry", None) is not None:
        try:
            registry = env.tool_registry.list_tools(role=env.role, context=getattr(env, "tool_context", None))
        except Exception:
            registry = []
    visible, deferred = split_tools(env.role, registry)
    return {
        "n_repl_globals": len(repl_globals),
        "n_registry_tools": len(registry),
        "n_registry_visible": len(visible),
        "n_registry_deferred": len(deferred),
        "n_surface_total": len(repl_globals) + len(registry),
    }


def _log_path() -> Path:
    base = os.environ.get("ORCHESTRATOR_TOOL_SURFACE_LOG_DIR") or str(_ROOT / "logs" / "tool_surface")
    return Path(base) / "tool_surface.jsonl"


def record_turn(env: Any, turn_index: int) -> dict[str, Any] | None:
    """Append one per-turn record. Never raises: logging must not fail a turn."""
    try:
        rec = {
            "schema": SCHEMA,
            "ts": time.time(),
            "role": env.role,
            "task_id": env.task_id,
            "session_id": str(getattr(env, "_session_id", "")),
            "turn_index": turn_index,
            "cap_enabled": bool(getattr(env, "_tool_surface_cap_on", False)),
            **surface_counts(env),
        }
        p = _log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with _lock, p.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec
    except Exception:
        return None
