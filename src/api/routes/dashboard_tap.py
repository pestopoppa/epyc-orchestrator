"""Tap-file parsing helpers extracted from dashboard.py (2026-05-21 refactor).

The autopilot/seeding harness writes (ROLE, PROMPT, RESPONSE) sections to
/mnt/raid0/llm/tmp/inference_tap.log, current prompt to autopilot_prompt_tap.txt,
REPL history to repl_tap.log, and a sentinel file to .inference_tap_active. The
dashboard tails them to surface live inference state.

Includes _parse_trial_state for GEPA trial-state extraction from autopilot.log.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# Tap-file paths (autopilot/seeding writes; dashboard reads).
_INFERENCE_TAP_PATH = Path("/mnt/raid0/llm/tmp/inference_tap.log")
_INFERENCE_TAP_EVENTS_PATH = Path("/mnt/raid0/llm/tmp/inference_tap_events.jsonl")
_REPL_TAP_PATH = Path("/mnt/raid0/llm/tmp/repl_tap.log")
_PROMPT_TAP_PATH = Path("/mnt/raid0/llm/tmp/autopilot_prompt_tap.txt")
_TAP_SENTINEL_PATH = Path("/mnt/raid0/llm/tmp/.inference_tap_active")

_SECTION_SEP = "=" * 72
_SUBSECTION_SEP = "-" * 72


def _read_tail(path: Path, max_bytes: int = 256 * 1024) -> str:
    """Read the last ~max_bytes from a file, decoded as UTF-8."""
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
                # Discard partial first line
                _ = f.readline()
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _grep_lines_reverse(
    path: Path,
    needle: str,
    *,
    max_scan_bytes: int = 512 * 1024 * 1024,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> str:
    """Scan `path` backward and return the lines containing `needle`, oldest-first.

    A fixed tail window is useless on the inference tap once it grows to multi-GB
    under autopilot-eval load: at ~1 MB / 6 s, a 1 MB tail covers only seconds, so
    any task older than that renders "(empty)" in the dashboard even though its full
    stream is on disk (observed 2026-05-31: a 2.3 GB tap, chat task 10 min old,
    unrecoverable from the tail). This reads the file in reverse chunks, keeps only
    matching lines, and early-exits once it has passed the matched block (an older
    chunk with no matches), so a single request is recovered without loading the
    whole file. `max_scan_bytes` bounds the worst case for a never-matching needle.

    Chunk boundaries are stitched: the partial line at a chunk's low-offset edge is
    carried into the next (older) read so no event line is split across the seam.
    """
    if not path.exists() or not needle:
        return ""
    needle_b = needle.encode("utf-8")
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    pos = size
    scanned = 0
    found_any = False
    carry = b""  # tail fragment whose line-head lives in older (not-yet-read) bytes
    collected: list[bytes] = []  # matching-line blocks, newest-first
    try:
        with open(path, "rb") as f:
            while pos > 0 and scanned < max_scan_bytes:
                read = min(chunk_bytes, pos)
                pos -= read
                f.seek(pos)
                buf = f.read(read) + carry
                scanned += read
                if pos > 0:
                    nl = buf.find(b"\n")
                    if nl == -1:
                        carry = buf  # whole chunk is one partial line; keep going older
                        continue
                    carry = buf[:nl]  # fragment of an even-older line
                    body = buf[nl + 1:]
                else:
                    carry = b""
                    body = buf
                matches = [ln for ln in body.split(b"\n") if needle_b in ln]
                if matches:
                    found_any = True
                    collected.append(b"\n".join(matches))
                elif found_any:
                    # Already captured the request; this older chunk has none → done.
                    break
    except Exception:
        return ""
    collected.reverse()  # oldest-first, so the request's start event comes first
    return b"\n".join(collected).decode("utf-8", errors="ignore")


def _parse_inference_sections(tail_text: str, max_sections: int = 20) -> list[dict[str, Any]]:
    """Parse the last N (ROLE, PROMPT, RESPONSE) sections from inference_tap.log.

    The tap format used by autopilot/seeding:
        [2026-05-21 11:16:22] ROLE=worker_general
        ------------------------------------------------------------------------
        PROMPT: <prompt text>
        ------------------------------------------------------------------------
        RESPONSE:
        <response text>
        ========================================================================

    Returns the most-recent sections first (descending chronological).
    """
    sections: list[dict[str, Any]] = []
    if not tail_text:
        return sections
    # Split on the section terminator
    raw_sections = tail_text.split(_SECTION_SEP)
    # The first chunk is likely a partial section from before our tail window —
    # skip it unless it has both a PROMPT and RESPONSE
    candidates = raw_sections[-(max_sections + 1):]
    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Extract role + timestamp from "[YYYY-MM-DD HH:MM:SS] ROLE=xxx"
        role_match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*ROLE=(\S+)", chunk)
        # Extract PROMPT and RESPONSE blocks
        prompt_idx = chunk.find("PROMPT:")
        response_idx = chunk.find("RESPONSE:")
        if prompt_idx < 0 or response_idx < 0 or response_idx < prompt_idx:
            continue
        prompt_text = chunk[prompt_idx + len("PROMPT:"):response_idx].strip()
        # Strip the subsection separator before RESPONSE
        prompt_text = re.sub(r"-{20,}\s*$", "", prompt_text).strip()
        response_text = chunk[response_idx + len("RESPONSE:"):].strip()
        # Filter out llama-server's `TIMINGS:` probe/healthcheck responses —
        # those are llama.cpp's internal timing dumps emitted on empty
        # generations, not real inference output.
        response_stripped = response_text.lstrip("-").lstrip()
        if response_stripped.startswith("TIMINGS:") and len(response_text) < 400:
            continue
        sections.append({
            "timestamp": role_match.group(1) if role_match else None,
            "role": role_match.group(2) if role_match else None,
            "prompt": prompt_text,
            "response": response_text,
            "prompt_len": len(prompt_text),
            "response_len": len(response_text),
        })
    # Most recent first
    return list(reversed(sections))


def _fmt_structured_timings(event: dict[str, Any]) -> str:
    try:
        tokens = int(event.get("tokens") or 0)
        total_s = float(event.get("total_s") or 0.0)
        prompt_ms = float(event.get("prompt_ms") or 0.0)
        gen_ms = float(event.get("gen_ms") or 0.0)
        tps = float(event.get("tps") or 0.0)
    except (TypeError, ValueError):
        return ""
    return (
        f"{tokens} tokens in {total_s:.2f}s "
        f"(prompt={prompt_ms:.0f}ms, gen={gen_ms:.0f}ms, {tps:.1f} t/s)"
    )


def _parse_structured_tap_requests(
    tail_text: str,
    max_requests: int = 20,
    now_epoch: float | None = None,
    quiet_after_s: float = 15.0,
) -> list[dict[str, Any]]:
    """Parse structured JSONL tap events into request-grouped records.

    Unlike the legacy plaintext parser, this stream is safe under concurrent
    inference because every event carries request_id + instance metadata.
    """
    if not tail_text:
        return []
    requests: dict[str, dict[str, Any]] = {}
    order = 0

    for line in tail_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        request_id = str(event.get("request_id") or "").strip()
        if not request_id:
            continue
        order += 1
        rec = requests.get(request_id)
        if rec is None:
            rec = {
                "request_id": request_id,
                "parent_request_id": event.get("parent_request_id"),
                "role": event.get("role"),
                "task_id": event.get("task_id"),
                "trial_id": event.get("trial_id"),
                "batch_id": event.get("batch_id"),
                "instance_idx": event.get("instance_idx"),
                "concurrency_idx": event.get("concurrency_idx"),
                "instance_shape": event.get("instance_shape"),
                "instance_regions": event.get("instance_regions") or [],
                "topology_role": event.get("topology_role"),
                "lock_role": event.get("lock_role"),
                "port": event.get("port"),
                "pid": event.get("pid"),
                "topology_hash": event.get("topology_hash"),
                "backend_url": event.get("backend_url"),
                "started_at": event.get("ts"),
                "started_at_epoch": event.get("ts_epoch") or 0,
                "updated_at": event.get("ts"),
                "updated_at_epoch": event.get("ts_epoch") or 0,
                "status": "running",
                "prompt": "",
                "response": "",
                "timings": "",
                "timings_raw": None,
                "event_count": 0,
                "chunk_count": 0,
                "_order": order,
            }
            requests[request_id] = rec

        rec["event_count"] += 1
        rec["_order"] = order
        rec["updated_at"] = event.get("ts") or rec.get("updated_at")
        rec["updated_at_epoch"] = event.get("ts_epoch") or rec.get("updated_at_epoch") or 0
        for key in (
            "role",
            "parent_request_id",
            "task_id",
            "trial_id",
            "batch_id",
            "instance_idx",
            "concurrency_idx",
            "instance_shape",
            "instance_regions",
            "topology_role",
            "lock_role",
            "port",
            "pid",
            "topology_hash",
            "backend_url",
        ):
            value = event.get(key)
            if value not in (None, ""):
                rec[key] = value

        event_type = str(event.get("event") or "").lower()
        if event_type == "start":
            rec["prompt"] = str(event.get("prompt") or "")
            rec["started_at"] = event.get("ts") or rec.get("started_at")
            rec["started_at_epoch"] = event.get("ts_epoch") or rec.get("started_at_epoch") or 0
        elif event_type == "chunk":
            rec["response"] += str(event.get("text") or "")
            rec["chunk_count"] += 1
        elif event_type == "response":
            rec["response"] += str(event.get("text") or "")
        elif event_type == "timings":
            rec["timings_raw"] = event
            rec["timings"] = _fmt_structured_timings(event)
            rec["status"] = "complete"
        elif event_type == "end":
            rec["status"] = "complete"
            rec["ended_at"] = event.get("ts")
            rec["ended_at_epoch"] = event.get("ts_epoch") or 0

    out = []
    for rec in requests.values():
        public = {k: v for k, v in rec.items() if not k.startswith("_")}
        public["prompt_len"] = len(public.get("prompt") or "")
        public["response_len"] = len(public.get("response") or "")
        if now_epoch is not None:
            try:
                started = float(public.get("started_at_epoch") or 0)
            except (TypeError, ValueError):
                started = 0.0
            try:
                updated = float(public.get("updated_at_epoch") or 0)
            except (TypeError, ValueError):
                updated = 0.0
            public["age_s"] = max(0.0, now_epoch - started) if started else None
            public["quiet_s"] = max(0.0, now_epoch - updated) if updated else None
            if (
                public.get("status") == "running"
                and public["quiet_s"] is not None
                and public["quiet_s"] >= quiet_after_s
            ):
                public["status"] = "quiet"
                if public.get("response"):
                    public["status_reason"] = "tap stream quiet after response update"
                else:
                    public["status_reason"] = (
                        "no tap output captured since start; request may be queued, "
                        "pre/post-model, non-streaming, or orphaned"
                    )
        public["is_live"] = public.get("status") != "complete"
        out.append(public)
    out.sort(
        key=lambda r: (
            float(r.get("updated_at_epoch") or 0),
            int(requests[str(r["request_id"])].get("_order") or 0),
        ),
        reverse=True,
    )
    return out[:max_requests]


def _parse_trial_state(tail: str) -> dict[str, Any]:
    """Scan autopilot.log tail for the active GEPA trial's state."""
    state: dict[str, Any] = {
        "current_trial": None,
        "current_action": None,
        "current_file": None,
        "baseline_sentinels_total": None,
        "baseline_score": None,
        "last_event": None,
    }
    # Iterate in order — last match wins
    re_trial = re.compile(r"Trial (\d+):\s*({.*})")
    re_baseline = re.compile(r"GEPA: evaluating baseline for (\S+\.md) \((\d+) sentinels\)")
    re_score = re.compile(r"GEPA: baseline score = ([\d.]+)")
    re_dispatch = re.compile(r"Dispatching action: (\w+)")
    for line in tail.splitlines():
        m = re_trial.search(line)
        if m:
            state["current_trial"] = int(m.group(1))
            try:
                cfg = json.loads(m.group(2))
                state["current_action"] = cfg.get("type")
                state["current_file"] = cfg.get("file")
            except Exception:
                pass
        m = re_baseline.search(line)
        if m:
            state["current_file"] = m.group(1)
            state["baseline_sentinels_total"] = int(m.group(2))
            state["last_event"] = "evaluating_baseline"
        m = re_score.search(line)
        if m:
            state["baseline_score"] = float(m.group(1))
            state["last_event"] = "baseline_done"
        m = re_dispatch.search(line)
        if m and not state["current_action"]:
            state["current_action"] = m.group(1)
    return state
