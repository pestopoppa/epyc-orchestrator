"""Agent-file operating-point resolver (F2, operator decision 2026-11-12).

Maps a role's EXPLICIT per-role ``agent_file_compression_operating_point`` in
``orchestration/model_registry.yaml`` (``roles.<role>``) to an agent file under
``AGENT_FILES_ROOT``:

- ``none``                      -> ``ENGINEERING_STANDARDS.md`` (full text)
- ``mild|medium|aggressive``    -> ``ENGINEERING_STANDARDS.compressed-<level>.md``

The global ``runtime_defaults`` value is deliberately ignored: a role gets an
agent file only when the registry names one for it. Everything here is
serving-path safe: it never raises, warns once per missing file, and caches
both the registry read and the file text (no per-request disk I/O).

Injection is gated by the ``agent_file_injection`` feature flag (default off)
and the ``ORCHESTRATOR_AGENT_FILE_ROLES`` allowlist; see
``src/prompt_builders/agent_file.py``.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AGENT_FILES_ROOT_ENV = "AGENT_FILES_ROOT"
DEFAULT_AGENT_FILES_ROOT = "/workspace/agents/shared"
AGENT_FILE_STEM = "ENGINEERING_STANDARDS"
REGISTRY_FIELD = "agent_file_compression_operating_point"
VALID_LEVELS = ("none", "mild", "medium", "aggressive")

_LOCK = threading.Lock()
_ROLE_LEVELS: dict[str, str] | None = None
_TEXT_CACHE: dict[Path, str | None] = {}
_WARNED: set[str] = set()


def agent_files_root() -> Path:
    """Root directory holding the agent files (env ``AGENT_FILES_ROOT``)."""
    return Path(os.environ.get(AGENT_FILES_ROOT_ENV) or DEFAULT_AGENT_FILES_ROOT)


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(msg, *args)


def _registry_path() -> Path:
    return Path(
        os.environ.get("ORCHESTRATOR_PATHS_REGISTRY_PATH", "orchestration/model_registry.yaml")
    )


def _load_role_levels() -> dict[str, str]:
    """Explicit per-role operating points from ``roles.*`` (cached)."""
    global _ROLE_LEVELS
    if _ROLE_LEVELS is not None:
        return _ROLE_LEVELS
    with _LOCK:
        if _ROLE_LEVELS is not None:
            return _ROLE_LEVELS
        levels: dict[str, str] = {}
        try:
            import yaml

            path = _registry_path()
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    payload = yaml.safe_load(fh) or {}
                roles = payload.get("roles", {})
                if isinstance(roles, dict):
                    for name, row in roles.items():
                        if isinstance(row, dict) and REGISTRY_FIELD in row:
                            value = str(row[REGISTRY_FIELD]).strip().lower()
                            if value in VALID_LEVELS:
                                levels[str(name)] = value
                            else:
                                _warn_once(
                                    f"level:{name}",
                                    "agent file: role %s has invalid %s=%r; ignored",
                                    name, REGISTRY_FIELD, value,
                                )
        except Exception as exc:  # serving path: never raise
            _warn_once("registry", "agent file: registry unreadable (%s); no agent files", exc)
        _ROLE_LEVELS = levels
        return levels


def resolve_agent_file_level(role: str) -> str | None:
    """Explicit per-role operating point, or None when unset/global-only."""
    return _load_role_levels().get(str(role))


def resolve_agent_file(role: str) -> Path | None:
    """Agent file for ``role``, or None (unset, invalid, or missing file)."""
    level = resolve_agent_file_level(role)
    if level is None:
        return None
    name = f"{AGENT_FILE_STEM}.md" if level == "none" else f"{AGENT_FILE_STEM}.compressed-{level}.md"
    path = agent_files_root() / name
    if not path.is_file():
        _warn_once(f"missing:{path}", "agent file: %s missing for role %s (level %s)", path, role, level)
        return None
    return path


def read_agent_file_text(path: Path) -> str | None:
    """Cached file read; None (with one warning) on failure."""
    if path in _TEXT_CACHE:
        return _TEXT_CACHE[path]
    try:
        text: str | None = path.read_text(encoding="utf-8").strip() or None
    except Exception as exc:
        _warn_once(f"read:{path}", "agent file: cannot read %s (%s)", path, exc)
        text = None
    _TEXT_CACHE[path] = text
    return text


def reset_agent_file_caches() -> None:
    """Clear cached registry levels, file text and warn-once state (tests)."""
    global _ROLE_LEVELS
    with _LOCK:
        _ROLE_LEVELS = None
        _TEXT_CACHE.clear()
        _WARNED.clear()
