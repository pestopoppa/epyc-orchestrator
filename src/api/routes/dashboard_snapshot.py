"""Snapshot scanners extracted from dashboard.py (2026-05-21 refactor).

Mines the orchestrator's progress JSONL log for routing decisions, in-flight /
recently-completed chat tasks, and pattern counts. All scanners are pure
file-reads; the dashboard route layer composes them into a single snapshot
JSON response.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def todays_progress_log(progress_log_dir: Path) -> Path:
    """Today's progress JSONL path under `progress_log_dir`."""
    return progress_log_dir / f"{date.today().isoformat()}.jsonl"


def scan_recent_decisions(
    path: Path, window_s: float = 600.0, max_items: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Tail today's progress JSONL, return recent decisions + counters.

    Returns (recent_list, source_counts_rolling, source_counts_cumulative).
    """
    recent: deque = deque(maxlen=max_items)
    source_rolling: dict[str, int] = {}
    source_cumulative: dict[str, int] = {}
    verifier_verdicts: dict[str, int] = {}
    now = time.time()
    if not path.exists():
        return list(recent), source_rolling, source_cumulative
    try:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event_type") != "routing_decision":
                    continue
                d = e.get("data", {})
                src = d.get("decision_source") or d.get("strategy") or "?"
                source_cumulative[src] = source_cumulative.get(src, 0) + 1
                ts_str = e.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = now
                age = now - ts
                if age <= window_s:
                    source_rolling[src] = source_rolling.get(src, 0) + 1
                    if d.get("verifier_verdict"):
                        v = d["verifier_verdict"]
                        verifier_verdicts[v] = verifier_verdicts.get(v, 0) + 1
                # Build a compact summary
                recent.append({
                    "task_id": e.get("task_id"),
                    "ts": ts_str,
                    "age_s": round(age, 1),
                    "source": src,
                    "chosen_action": d.get("chosen_action") or "",
                    "classifier_confidence": d.get("classifier_confidence"),
                    "verifier_p_success": d.get("verifier_p_success"),
                    "verifier_verdict": d.get("verifier_verdict"),
                    "verifier_shadow": d.get("verifier_shadow"),
                })
    except Exception as exc:
        logger.debug("scan_recent_decisions failed: %s", exc)
    source_rolling["_verifier_verdicts"] = verifier_verdicts  # type: ignore[assignment]
    return list(recent), source_rolling, source_cumulative


def scan_orchestrator_tasks(
    path: Path,
    in_flight_max_age_s: float = 90.0,
    completed_window_s: float = 600.0,
    max_items: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan today's progress JSONL for in-flight + recently-completed chat tasks.

    Uses the orchestrator's `chat-XXX` task_ids (not llama-server's internal
    numeric id_task). Returns (in_flight, recent_completed).

    NB: we DO NOT time-filter routing_decision events during the scan — we
    need them merged into every started/completed task even if the routing
    happened minutes before our window. Tasks that started > in_flight_max_age_s
    ago and never completed are treated as orphans (typically killed by an
    API restart) and excluded from in-flight.
    """
    if not path.exists():
        return [], []
    now = time.time()
    started: dict[str, dict[str, Any]] = {}
    terminal_events: dict[str, dict[str, Any]] = {}
    routing_meta: dict[str, dict[str, Any]] = {}
    try:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = e.get("task_id")
                if not tid or not isinstance(tid, str) or not tid.startswith("chat-"):
                    continue
                ev = e.get("event_type")
                ts_str = e.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    continue
                data = e.get("data", {})
                if ev == "task_started":
                    started[tid] = {
                        "task_id": tid,
                        "started_at": ts,
                        "age_s": now - ts,
                        "objective": (data.get("objective", "") or "")[:200],
                        "task_type": data.get("task_type"),
                        "priority": data.get("priority"),
                    }
                elif ev == "routing_decision":
                    routing_meta[tid] = {
                        "chosen_action": data.get("chosen_action") or "",
                        "decision_source": data.get("decision_source") or "",
                        "classifier_confidence": data.get("classifier_confidence"),
                        "verifier_p_success": data.get("verifier_p_success"),
                        "verifier_verdict": data.get("verifier_verdict"),
                        "difficulty_band": data.get("difficulty_band"),
                    }
                elif ev in ("task_completed", "task_failed", "escalation_triggered"):
                    terminal_events[tid] = {
                        "event_type": ev,
                        "ended_at": ts,
                        "age_s": now - ts,
                        "final_role": data.get("final_answer_role") or data.get("producer_role"),
                    }
    except Exception as exc:
        logger.debug("scan_orchestrator_tasks failed: %s", exc)

    in_flight: list[dict[str, Any]] = []
    recent_completed: list[dict[str, Any]] = []
    for tid, s in started.items():
        s.update(routing_meta.get(tid, {}))
        if tid not in terminal_events:
            if s["age_s"] <= in_flight_max_age_s:
                in_flight.append(s)
        else:
            t = terminal_events[tid]
            if t["age_s"] > completed_window_s:
                continue
            s["ended_at"] = t["ended_at"]
            s["end_age_s"] = t["age_s"]
            s["outcome"] = t["event_type"]
            s["duration_s"] = round(t["ended_at"] - s["started_at"], 2)
            s["final_role"] = t.get("final_role")
            recent_completed.append(s)
    in_flight.sort(key=lambda x: x["age_s"])
    recent_completed.sort(key=lambda x: x["end_age_s"])
    return in_flight[:max_items], recent_completed[:max_items]


def count_log_events(
    path: Path, patterns: dict[str, str], window_s: float = 600.0,
) -> dict[str, int]:
    """Tail the log at `path` and count occurrences of `patterns` regexes."""
    counts = {key: 0 for key in patterns}
    if not path.exists():
        return counts
    # Try a recent-tail-only optimization: read last 256KB if file is big
    try:
        size = path.stat().st_size
        if size > 256 * 1024:
            with open(path, "rb") as f:
                f.seek(-256 * 1024, 2)
                tail = f.read().decode("utf-8", errors="ignore")
        else:
            with open(path) as f:
                tail = f.read()
    except Exception:
        return counts
    compiled = {k: re.compile(v) for k, v in patterns.items()}
    for line in tail.splitlines():
        for key, regex in compiled.items():
            if regex.search(line):
                counts[key] += 1
    return counts
