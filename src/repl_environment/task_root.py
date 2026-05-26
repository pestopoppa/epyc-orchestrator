"""Model task-root override for the BEP-2 / DCP-6 falsification harness.

`ORCHESTRATOR_EDIT_ROOT`, when set to an existing directory, redirects MODEL-FACING
filesystem surfaces — file write/read/peek/grep/list/file_info, ``run_shell`` cwd,
``code_search``/ColGREP root, the batch-edit repo root, and DCP file discovery — to a
scratch task repo, so a BEP/DCP A/B never mutates the orchestrator's own checkout.

CONTROL-PLANE paths stay on the real ``project_root`` (registry, model/tool config,
sessions, orchestration logs, patch ledgers, procedures/checkpoints, benchmarks) — those
keep calling the existing ``_get_project_root()`` copies, NOT these accessors.

Default (env unset / not a dir) == today's behavior exactly: ``get_task_root()`` returns
``project_root`` and ``resolve_task_path`` resolves relative paths against the process cwd,
so production is unchanged. See ``data/bep_sandbox/task_root_surface_audit.md`` (Phase 0) and
``handoffs/active/bep-dcp-falsification-harness.md``.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "ORCHESTRATOR_EDIT_ROOT"


def _project_root() -> Path:
    """The real orchestrator project root (control-plane anchor)."""
    try:
        from src.config import get_config

        return Path(get_config().paths.project_root)
    except Exception:
        # Mirror file_mutation/external_access fallback so this never hard-fails.
        return Path(os.getcwd())


def task_root_active() -> bool:
    """True iff ORCHESTRATOR_EDIT_ROOT is set to an existing directory."""
    v = os.environ.get(ENV_VAR, "").strip()
    return bool(v) and Path(v).is_dir()


def get_task_root() -> Path:
    """Model task-root: ``$ORCHESTRATOR_EDIT_ROOT`` if set + an existing dir, else
    ``project_root``. Read live (no caching) so an A/B can flip it via env across restarts
    and tests can monkeypatch it."""
    v = os.environ.get(ENV_VAR, "").strip()
    if v:
        p = Path(v)
        if p.is_dir():
            return p
    return _project_root()


def resolve_task_path(path: str) -> str:
    """Resolve a model-supplied path to a realpath string.

    When the task-root is active, a RELATIVE path resolves against the task-root (so the model
    inspecting/editing ``cart.py`` hits the scratch repo, not the orchestrator's cwd). Absolute
    paths pass through unchanged. When inactive, behaves exactly like ``os.path.realpath(path)``
    (today's behavior).
    """
    p = Path(path)
    if not p.is_absolute() and task_root_active():
        return os.path.realpath(str(get_task_root() / path))
    return os.path.realpath(path)
