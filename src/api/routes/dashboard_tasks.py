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
    _INFERENCE_TAP_EVENTS_PATH,
    _INFERENCE_TAP_PATH,
    _parse_inference_sections,
    _parse_structured_tap_requests,
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
    task_id: str,
    events: list[dict[str, Any]],
    slot: dict | None,
    tap_section: dict | None = None,
) -> str:
    """Render a plain-text snapshot of a task suitable for pasting into chat.

    Inference-stream source priority:
        1. Live llama-server slot (highest fidelity — current cache state)
        2. inference_tap.log section matching the task's objective
           (covers completed tasks whose slot is gone — the JSON endpoint
           has always used this fallback; this function was missing it,
           producing "(empty)" INFERENCE STREAM for finished tasks even
           when the dashboard panel showed the tap content.)
        3. Empty placeholder
    """
    lines: list[str] = []
    # Use timezone-aware UTC instead of the deprecated datetime.utcnow().
    # isoformat() on a tz-aware datetime ends with "+00:00"; the dashboard
    # snapshot format expects a trailing "Z" so we strip the offset.
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "")
    lines.append(f"=== Task {task_id} @ {now_utc}Z ===")
    lines.append("")
    prompt_text = ""
    stream_text = ""
    source_note = ""
    if slot:
        prompt_text = str(slot.get("prompt") or "")
        stream_text = str(slot.get("content") or "")
        source_note = "(source: live llama-server slot)"
    elif tap_section:
        prompt_text = str(tap_section.get("prompt") or "")
        stream_text = str(tap_section.get("response") or "")
        ts = tap_section.get("timestamp") or tap_section.get("started_at") or "?"
        role = tap_section.get("role") or "?"
        if tap_section.get("source") == "structured_tap":
            req = tap_section.get("request_id") or "?"
            source_note = (
                f"(source: inference_tap_events.jsonl request {req} @ {ts} · "
                f"role={role})"
            )
        else:
            source_note = f"(source: inference_tap.log section @ {ts} · role={role})"
    if not prompt_text:
        for ev in events:
            if ev.get("event_type") == "task_started":
                prompt_text = ev.get("data", {}).get("objective", "") or ""
                break
    lines.append("PROMPT:")
    lines.append("-------")
    lines.append(prompt_text or "(not available)")
    lines.append("")
    lines.append("INFERENCE STREAM:" + (f"  {source_note}" if source_note else ""))
    lines.append("-----------------")
    lines.append(stream_text or "(empty — no live slot and no matching tap section)")
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


def _find_structured_request_by_id(task_id: str, max_requests: int = 400) -> dict | None:
    """Find a structured tap request by dashboard task id.

    The live dashboard exposes request rows as ``tap_<request_id>``. Those ids
    do not exist in progress JSONL or llama-server slot state, so text/detail
    endpoints need to resolve them directly from inference_tap_events.jsonl.
    """
    request_id = task_id[4:] if task_id.startswith("tap_") else task_id
    request_id = request_id.strip()
    if not request_id:
        return None
    tail_text = _read_tail(_INFERENCE_TAP_EVENTS_PATH, max_bytes=1024 * 1024)
    for request in _parse_structured_tap_requests(tail_text, max_requests=max_requests):
        if str(request.get("request_id") or "") == request_id:
            out = dict(request)
            out["source"] = "structured_tap"
            return out
    return None


def _find_structured_request_by_task_id(
    task_id: str, max_requests: int = 400
) -> dict | None:
    """Find the most-recent structured tap request whose ``task_id`` matches.

    Orchestrator chat task ids (e.g. ``chat-83123001``) appear in the structured
    event stream under the ``task_id`` field, while each inference call has its
    own derived ``request_id`` (e.g. ``chat-83123001:b763498c``). The dashboard's
    chat-* task-detail path used to skip the structured tap entirely and fall
    back to plaintext substring matching, which conflates concurrent sections
    when interleaved per-append writes produce syntactically-valid but
    cross-contaminated records. Resolving by ``task_id`` instead gives a
    deterministic mapping for any chat-* task that produced a streamed section.
    """
    task_id = (task_id or "").strip()
    if not task_id:
        return None
    tail_text = _read_tail(_INFERENCE_TAP_EVENTS_PATH, max_bytes=1024 * 1024)
    # _parse_structured_tap_requests returns most-recent-updated first.
    for request in _parse_structured_tap_requests(tail_text, max_requests=max_requests):
        if str(request.get("task_id") or "") == task_id:
            out = dict(request)
            out["source"] = "structured_tap"
            return out
    return None


def _find_section_by_objective(
    objective: str,
    expected_role: str | None = None,
) -> dict | None:
    """Search inference_tap.log for a completed section that matches the
    task's objective.

    Tries multiple strategies in order (most-specific first) — the single
    120-char substring search was too brittle for long prompts where the
    system-prompt + chat-template overhead pushed the user portion past
    the tap writer's truncation cap, breaking literal substring match.

    Match strategies:
        1. First 120c of objective in section prompt
        2. First 60c of objective in section prompt
        3. Middle 60c (objective[60:120]) — useful when chat template prefix
           varies but the middle of the user content is preserved
        4. Last 60c of objective in section prompt
        5. Role-filtered: same strategies (1)-(4) but limited to sections
           whose role matches `expected_role` (if provided) — improves
           precision when multiple roles processed the same user content

    Returns the most recent (newest-first) section that matches under the
    earliest-succeeding strategy, or None if every strategy fails.

    Used as a fallback when the task's llama-server slot is no longer
    alive (orchestrator chat-XXX ids don't map to llama-server's internal
    numeric id_task).
    """
    if not objective or len(objective) < 8:
        return None
    tail_text = _read_tail(_INFERENCE_TAP_PATH, max_bytes=1024 * 1024)
    sections = _parse_inference_sections(tail_text, max_sections=80)
    if not sections:
        return None

    # Strategy candidates, broadest match first within each pass.
    # For short objectives (< 60c) just use the whole thing; for longer ones
    # try a few overlapping windows so a single broken substring doesn't
    # disqualify the section.
    needles: list[str] = []
    obj = objective.strip()
    if len(obj) <= 120:
        needles.append(obj)
    else:
        needles.append(obj[:120].strip())
        needles.append(obj[:60].strip())
        if len(obj) >= 180:
            needles.append(obj[60:120].strip())
        needles.append(obj[-60:].strip())
    # De-dup while preserving order
    seen: set[str] = set()
    needles = [n for n in needles if n and not (n in seen or seen.add(n))]
    if not needles:
        return None

    def _match(sections_iter, needle: str) -> dict | None:
        for s in sections_iter:
            if needle in (s.get("prompt") or ""):
                return s
        return None

    # Pass 1 — role-filtered (if known). Higher precision; prefer over global.
    # When the caller knows the producer role, we deliberately do NOT fall back
    # to a global plaintext pass: under concurrent interleaved tap writes, a
    # syntactically-valid section from a different role can still contain the
    # objective substring while pairing it with the wrong response (observed
    # 2026-05-30: chat-83123001 routed to frontdoor but the global pass matched
    # a worker_explore record with an unrelated response). For chat-* tasks the
    # structured-event lookup is the deterministic path; this fallback is for
    # legacy callers that lack producer-role telemetry.
    if expected_role:
        role_sections = [s for s in sections if (s.get("role") or "") == expected_role]
        for n in needles:
            hit = _match(role_sections, n)
            if hit is not None:
                return hit
        return None

    # Pass 2 — global. Each needle in turn.
    for n in needles:
        hit = _match(sections, n)
        if hit is not None:
            return hit

    return None
