"""File-backed spill and solution-artifact helpers for graph execution."""

from __future__ import annotations

import logging
import os
import tempfile

from src.graph.state import TaskState

log = logging.getLogger(__name__)


def _solution_file_path(state: TaskState) -> str:
    """Return the path for the persisted solution file."""
    task_id = state.task_id or "scratch"
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:80]
    return f"/mnt/raid0/llm/tmp/{safe_id}_solution.py"


def _persist_solution_file(state: TaskState, code: str) -> None:
    """Write the model's current code to a file for incremental editing."""
    if not code or not code.strip():
        return
    stripped = code.strip()
    if stripped.startswith("FINAL(") and "\n" not in stripped:
        return
    try:
        path = _solution_file_path(state)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        log.debug("Failed to persist solution file: %s", e)


def _spill_if_truncated(text: str, max_chars: int, label: str, state: TaskState) -> str:
    """Return *text* with a retrieval pointer appended if it exceeds *max_chars*."""
    if len(text) <= max_chars:
        return text
    from src.features import features

    if not features().output_spill_to_file:
        return text
    task_id = state.task_id or "scratch"
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:80]
    turn = state.turns
    spill_path = f"/mnt/raid0/llm/tmp/{safe_id}_{label}_t{turn}.txt"
    try:
        os.makedirs(os.path.dirname(spill_path), exist_ok=True)
        with open(spill_path, "w", encoding="utf-8") as f:
            f.write(text)
        pointer_reserve = 150
        preview_end = max(200, max_chars - pointer_reserve)
        truncated = text[:preview_end]
        pointer = (
            f"\n[... {len(text) - preview_end} chars truncated; "
            f'full {label}: peek(99999, file_path="{spill_path}")]'
        )
        return truncated + pointer
    except Exception as e:
        log.debug("Failed to spill %s to file: %s", label, e)
        return text
