"""Checkpoint helpers for the orchestrator stack.

Used by self-management procedures (`procedure_registry`) to snapshot the
orchestrator state before applying changes, and to restore the prior state
on failure. The functions are parameterized on `checkpoint_dir` and
`state_file` so this module has no coupling to orchestrator_stack constants;
`orchestrator_stack.py` wires them up via thin wrappers.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.server.stack_state import (
    ProcessInfo,
    load_state_file,
    save_state_file,
)


def checkpoint_create(
    name: str,
    checkpoint_dir: Path,
    state_file: Path,
    *,
    include_state: bool = True,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Create a checkpoint of the orchestrator stack state.

    Called by self-management procedures before making changes.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_id = f"{name}_{timestamp}"
    checkpoint_path = checkpoint_dir / f"{checkpoint_id}.json"

    checkpoint_data: dict[str, Any] = {
        "id": checkpoint_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "state": {},
        "registry_snapshot": None,
    }

    if include_state:
        state = load_state_file(state_file)
        checkpoint_data["state"] = {k: asdict(v) for k, v in state.items()}

    # Snapshot of registry (just metadata, not full file)
    if registry_path is not None and registry_path.exists():
        checkpoint_data["registry_snapshot"] = {
            "path": str(registry_path),
            "mtime": registry_path.stat().st_mtime,
            "size": registry_path.stat().st_size,
        }

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    return {
        "checkpoint_id": checkpoint_id,
        "path": str(checkpoint_path),
        "created_at": checkpoint_data["created_at"],
    }


def checkpoint_restore(
    checkpoint_id: str,
    checkpoint_dir: Path,
    state_file: Path,
) -> dict[str, Any]:
    """Restore orchestrator stack state from a checkpoint."""
    checkpoint_path = checkpoint_dir / f"{checkpoint_id}.json"

    if not checkpoint_path.exists():
        return {"success": False, "error": f"Checkpoint not found: {checkpoint_id}"}

    try:
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        if checkpoint_data.get("state"):
            saved_state = {
                k: ProcessInfo(**v)
                for k, v in checkpoint_data["state"].items()
            }
            save_state_file(state_file, saved_state)

        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "restored_at": datetime.now().isoformat(),
            "original_created_at": checkpoint_data.get("created_at"),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def checkpoint_list(checkpoint_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    """List available checkpoints (newest first)."""
    if not checkpoint_dir.exists():
        return []

    checkpoints: list[dict[str, Any]] = []
    for cp_path in sorted(checkpoint_dir.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(cp_path) as f:
                data = json.load(f)
            checkpoints.append({
                "id": data.get("id", cp_path.stem),
                "name": data.get("name"),
                "created_at": data.get("created_at"),
                "path": str(cp_path),
            })
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # Skip malformed or unreadable checkpoint files

    return checkpoints


def checkpoint_delete(checkpoint_id: str, checkpoint_dir: Path) -> bool:
    """Delete a checkpoint by id. Returns True when removed, False if missing."""
    checkpoint_path = checkpoint_dir / f"{checkpoint_id}.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        return True
    return False
