"""UFH-TE-2: serving identity for the dispatch-time prepared-action recheck."""

from __future__ import annotations

import hashlib
import inspect
import subprocess
import time
from pathlib import Path
from typing import Any

from src.typed_decisions.prepared_action import fingerprint

_PROCESS_START_NS = time.time_ns()
_TREE_COMMIT: str | None = None


def _tree_commit() -> str | None:
    global _TREE_COMMIT
    if _TREE_COMMIT is None:
        try:
            root = Path(__file__).resolve().parents[2]
            _TREE_COMMIT = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip() or "unknown"
        except Exception:
            _TREE_COMMIT = "unknown"
    return _TREE_COMMIT


def _handler_artifact_digest(handler: Any) -> str | None:
    if handler is None:
        return None
    try:
        return hashlib.sha256(inspect.getsource(inspect.getmodule(handler)).encode()).hexdigest()
    except Exception:
        code = getattr(handler, "__code__", None)
        return hashlib.sha256(code.co_code).hexdigest() if code is not None else "unhashable"


def build_serving_identity(tool: Any) -> dict[str, Any]:
    """Identity of the live tool.  ``process_start_ns`` is audit-only (not compared)."""
    return {
        "descriptor_digest": fingerprint(
            {
                "parameters": getattr(tool, "parameters", None),
                "allowed_callers": getattr(tool, "allowed_callers", None),
            }
        ),
        "handler_artifact_digest": _handler_artifact_digest(getattr(tool, "handler", None)),
        "config_digest": getattr(tool, "code_hash", None),
        "serving_tree_commit": _tree_commit(),
        "process_start_ns": _PROCESS_START_NS,
    }
