"""Task correlation helpers extracted from dashboard.py (2026-05-21 refactor).

Per-task event mining (from the progress JSONL log), plain-text snapshot
rendering for chat-paste, and objective extraction. Orchestrator chat-XXX
task_ids don't appear in llama-server /slots state, so the dashboard correlates
by prompt content; _find_section_by_objective is the fallback that mines the
historical inference_tap.log when a task's slot is no longer alive.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.routes.dashboard_tap import (
    _INFERENCE_TAP_PATH,
    _parse_inference_sections,
    _read_tail,
)


# Keys to suppress from REPL-event payloads when rendering for chat-paste.
# stack_state in particular is ~4KB of registry dump per decision; valuable
# for offline debugging but pure noise in a conversation.
_NOISY_KEYS = {
    "stack_state", "similarity_topk", "q_topk", "selection_score_topk",
    "prior_term_topk", "posterior_score_topk", "learned_evidence_topk",
    "cost_term_topk",
}


def _task_events(task_id: str, path: Path, max_events: int = 200) -> list[dict[str, Any]]:
    """Return all progress-log events with a given task_id."""
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    try:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("task_id") != task_id:
                    continue
                events.append({
                    "event_type": e.get("event_type"),
                    "timestamp": e.get("timestamp"),
                    "data": e.get("data", {}),
                })
                if len(events) >= max_events:
                    break
    except Exception:
        pass
    return events


def _task_text_snapshot(
    task_id: str, events: list[dict[str, Any]], slot: dict | None,
) -> str:
    """Render a plain-text snapshot of a task suitable for pasting into chat."""
    lines: list[str] = []
    # Use timezone-aware UTC instead of the deprecated datetime.utcnow().
    # isoformat() on a tz-aware datetime ends with "+00:00"; the dashboard
    # snapshot format expects a trailing "Z" so we strip the offset.
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "")
    lines.append(f"=== Task {task_id} @ {now_utc}Z ===")
    lines.append("")
    prompt_text = ""
    stream_text = ""
    if slot:
        prompt_text = str(slot.get("prompt") or "")
        stream_text = str(slot.get("content") or "")
    if not prompt_text:
        for ev in events:
            if ev.get("event_type") == "task_started":
                prompt_text = ev.get("data", {}).get("objective", "") or ""
                break
    lines.append("PROMPT:")
    lines.append("-------")
    lines.append(prompt_text or "(not available)")
    lines.append("")
    lines.append("INFERENCE STREAM:")
    lines.append("-----------------")
    lines.append(stream_text or "(empty)")
    lines.append("")
    lines.append(f"REPL HISTORY ({len(events)} events):")
    lines.append("-----------------")
    for ev in events:
        ts = (ev.get("timestamp", "") or "").replace("T", " ")[11:19]
        ev_type = ev.get("event_type", "?")
        data = ev.get("data", {})
        if isinstance(data, dict):
            filtered = {k: v for k, v in data.items() if k not in _NOISY_KEYS}
            if len(filtered) < len(data):
                # Note when keys were elided so the reader knows.
                filtered["_elided_keys"] = sorted(set(data.keys()) - set(filtered.keys()))
        else:
            filtered = data
        try:
            data_str = json.dumps(filtered, separators=(",", ":"))
        except Exception:
            data_str = str(filtered)
        lines.append(f"[{ts}] {ev_type}: {data_str}")
    return "\n".join(lines)


def _objective_for_task(events: list[dict[str, Any]]) -> str:
    """Extract the original prompt/objective from a task's events."""
    for ev in events:
        if ev.get("event_type") == "task_started":
            return str(ev.get("data", {}).get("objective", "") or "")
    return ""


def _find_section_by_objective(objective: str) -> dict | None:
    """Search inference_tap.log for a completed section whose prompt contains `objective`.

    Returns the most recent match or None. Used as a fallback when the task's
    llama-server slot is no longer alive (orchestrator chat-XXX ids don't
    map to llama-server's internal numeric id_task).
    """
    if not objective or len(objective) < 8:
        return None
    needle = objective[:120].strip()
    tail_text = _read_tail(_INFERENCE_TAP_PATH, max_bytes=1024 * 1024)
    sections = _parse_inference_sections(tail_text, max_sections=80)
    for s in sections:  # already newest-first
        if needle and needle in (s.get("prompt") or ""):
            return s
    return None
