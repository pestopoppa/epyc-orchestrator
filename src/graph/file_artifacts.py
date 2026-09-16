"""File-backed spill and solution-artifact helpers for graph execution."""

from __future__ import annotations

import hashlib
import logging
import os

from src.graph.state import TaskState

log = logging.getLogger(__name__)


def _solution_file_path(state: TaskState) -> str:
    """Return the path for the persisted solution file.

    When a task-root is active, anchor the solution file INSIDE it. This path is shown to the
    model in its prompt (``solution_file=``); if it points outside the task-root (the old
    hardcoded ``/mnt/raid0/llm/tmp``), the model copies that prefix and writes its task edits
    there too — which file_write_safe rejects (outside the task-root) → silent no-op → the
    model re-emits the same write every turn until timeout. Keeping it task-root-relative makes
    the model's anchor match where its writes must land. (No-op in prod: task-root inactive.)
    """
    task_id = state.task_id or "scratch"
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:80]
    from src.repl_environment.task_root import get_task_root, task_root_active
    if task_root_active():
        return str(get_task_root() / f"{safe_id}_solution.py")
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


SPILL_DIR = "/mnt/raid0/llm/tmp"
_SPILL_HASH_CHARS = 16
_SPILL_MIN_BODY = 200


def _spill_file_path(text: str, label: str, state: TaskState) -> str:
    """Content-addressed spill path: ``{task}_{label}_{sha256[:16]}.txt``.

    The name is a function of the content, so a pointer already emitted into a
    transcript can never be silently re-pointed at different bytes by a later
    spill (the old ``{task}_{label}_t{turn}`` name was reopened with "w" on every
    re-run of the same task/turn). Identical output spills once. Files written
    under the old scheme are never touched, so their pointers keep resolving.
    """
    task_id = state.task_id or "scratch"
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:80]
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{SPILL_DIR}/{safe_id}_{safe_label}_{digest[:_SPILL_HASH_CHARS]}.txt"


def _write_spill_file(path: str, text: str) -> None:
    """Write *text* to *path* atomically; an existing file (same content by name) is kept."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _spill_if_truncated(text: str, max_chars: int, label: str, state: TaskState) -> str:
    """Return a head + tail excerpt of *text* with an exact-recall pointer if it exceeds *max_chars*.

    The full text is spilled to a content-addressed file. The excerpt keeps the
    head AND the tail (failure evidence usually lives at the end of a log), and
    the marker between them is an executable ``peek(n, file_path=..., offset=k)``
    call that returns exactly the omitted span. The whole result fits within
    *max_chars* whenever *max_chars* leaves room for the marker, so a downstream
    truncation cannot clip the pointer.
    """
    if len(text) <= max_chars:
        return text
    from src.features import features

    if not features().output_spill_to_file:
        return text
    spill_path = _spill_file_path(text, label, state)
    try:
        _write_spill_file(spill_path, text)

        def _marker(omitted: int, start: int) -> str:
            return (
                f"\n[... {omitted} of {len(text)} chars truncated; exact {label} span: "
                f'peek({omitted}, file_path="{spill_path}", offset={start})]\n'
            )

        # Size the marker with worst-case digit widths, then split the body 3:1 head:tail.
        reserve = len(_marker(len(text), len(text)))
        body = max(_SPILL_MIN_BODY, max_chars - reserve)
        body = min(body, len(text) - 1)
        tail_len = body // 4
        head_len = body - tail_len
        tail_start = len(text) - tail_len
        omitted = tail_start - head_len
        tail = text[tail_start:] if tail_len else ""
        return text[:head_len] + _marker(omitted, head_len) + tail
    except Exception as e:
        log.debug("Failed to spill %s to file: %s", label, e)
        return text
