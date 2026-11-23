"""AP-LH-WHOLE-TRAJECTORY-CAPTURE: a whole-trajectory trace for one autopilot trial (default off).

The contrastive trace bank is fed from the last 50 lines of the plaintext inference tap: LLM text only (REPL
results live in ``repl_tap.log``), cut at the tail, interleaved across concurrent requests. A trace like that is
never provably whole, so the AP-LH-DISTILL trace-replay pre-check (``harness_layers.replay_precheck``) can never
reject on it. This module builds the trace the pre-check needs, and proves or disproves its completeness.

Per-trial join (the only keys that exist; nothing is guessed):

* **trial -> batches.** The eval tower mints one ``eval_batch_id`` per ``_eval_batch`` call and stamps it on
  every request it dispatches; the tower records ``(batch ids, start epoch)`` under the current trial id
  (``EvalTower.note_trial_batches``). The tap events carry NO ``trial_id`` for eval traffic, so the batch id is
  the join key.
* **batch -> requests / tasks.** The structured inference-tap events (``inference_tap_events.jsonl``) carry
  ``batch_id``, ``request_id`` and ``task_id`` on every event. The plaintext tap is NOT used: its chunks are
  written without ids and interleave under concurrency.
* **task -> REPL turns.** ``repl_tap.log`` result blocks carry ``task_id`` only when this flag is on
  (``[turn N] RESULT | task_id=X | chars=L`` + exactly ``L`` characters of body; ``src/graph/repl_tap.py``),
  because the legacy result block has no id and results of concurrent tasks interleave.

Completeness is PROVEN, never assumed. The result is ``complete=True`` only when ALL of:

* both tap files cover the trial window (a binary search for the first event/block older than the batch start
  succeeds, or the file's start is reached and no rotated ``.1`` predecessor exists);
* every trial batch has events, and every request found has a ``start`` and an ``end`` event;
* every REPL exec header of a trial task has a matching result block, and every result block parses with its
  declared length (so a flag-off API, a crash mid-turn or an interleaved partial write reads as incomplete);
* no REPL result was cut by the tap (``... [N chars truncated]``);
* the assembled text fits the per-trial cap (``DEFAULT_MAX_CHARS``); a cap hit keeps the head, prefixes the
  bank's truncation marker and is ``complete=False``, never a cut trace marked complete.

Known limit, stated: an LLM call that never opens a ``tap_section`` (no ``backend_url``) is not in the tap and
so not in the trace; that cannot be detected from the tap alone.

Each REPL failure turn (the real guard's rule, ``src/graph/helpers.py::_note_turn_health``: a non-final result
with any non-empty error, else output starting ``[ERROR`` / ``[TOOL ERROR``) gets the fixed ``REPL_FAILURE_TAG``
line, which the failure-ratio predicate in ``harness_layers`` matches (tested).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import harness_layers

FLAG_ENV = "AUTOPILOT_LH_WHOLE_TRAJECTORY_CAPTURE"
SOURCE = "whole_trajectory"

#: Fixed tag, one line per failing REPL turn. Contains ``ERROR`` so the failure-ratio predicate matches it.
REPL_FAILURE_TAG = "[REPL TURN ERROR]"
#: Same text as ``eval_tower_trace_feedback.TRACE_TRUNCATED_MARKER`` (kept literal: this module stays pure).
TRUNCATED_MARKER = "[trace truncated]"

#: Per-trial cap on the assembled trajectory text (characters). A cap hit is ``complete=False``.
DEFAULT_MAX_CHARS = 262_144
#: Bytes scanned from the located window start to EOF, per file. Exceeding it is ``complete=False``.
DEFAULT_MAX_EVENTS_SCAN_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_REPL_SCAN_BYTES = 64 * 1024 * 1024
#: The window starts this long before the earliest batch (tap timestamps are written before the lock).
START_MARGIN_S = 2.0

_PROBE_BYTES = 1 << 20
_BINARY_SEARCH_STOP = 1 << 20

_REPL_BLOCK = re.compile(
    r"^\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] REPL turn (?P<xturn>\d+)(?: \| (?P<meta>[^\n]*))?$"
    r"|^\[turn (?P<rturn>\d+)\] RESULT \| task_id=(?P<rtask>\S+) \| chars=(?P<chars>\d+)$",
    re.M,
)
_TASK_META = re.compile(r"(?:^|\s)task_id=(\S+)")
_TAP_CUT = re.compile(r"\[\d+ chars truncated\]")
_EXEC_HEADER_TS = re.compile(rb"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] REPL turn \d+")


def flag_on() -> bool:
    """The default-off flag, honouring a live ``POST /config`` override (same reader as the other LH flags)."""
    return harness_layers.flag_on(FLAG_ENV)


@dataclass
class Capture:
    """One trial's whole-trajectory capture. ``complete`` is a proof result, ``reasons`` list what failed it."""

    text: str
    complete: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def tag_failure_turns(results: Iterable[dict[str, Any]]) -> list[str]:
    """Render parsed REPL results, one ``REPL_FAILURE_TAG`` line ahead of each failing turn's body."""
    out: list[str] = []
    for res in results:
        out.append(f"REPL RESULT task={res['task_id']} turn={res['turn']}")
        if result_is_failure(res["turn"], res["body"]):
            out.append(f"{REPL_FAILURE_TAG} task={res['task_id']} turn={res['turn']}")
        out.append(res["body"].rstrip("\n"))
    return out


def result_is_failure(turn: int, body: str) -> bool:
    """The real guard's failing-turn rule applied to a tap result body (final turns are never failures)."""
    if body.startswith(f"[turn {turn}] FINAL"):
        return False
    if re.search(rf"^\[turn {turn}\] ERROR:$", body, re.M):
        return True
    return body.lstrip().startswith(("[ERROR", "[TOOL ERROR"))


# ---------------------------------------------------------------------------------------------------------
# Window location: binary search over byte offsets for a time-ordered append-only file
# ---------------------------------------------------------------------------------------------------------
def _first_ts_after(f, offset: int, size: int, ts_of_line, max_lines: int = 4096) -> float | None:
    """Timestamp of the first FULL timestamped line at or after ``offset`` (None when there is none)."""
    f.seek(offset)
    if offset > 0:
        f.readline()  # discard the (possibly partial) line the probe landed in
    read = 0
    for _ in range(max_lines):
        line = f.readline()
        if not line:
            return None
        read += len(line)
        ts = ts_of_line(line)
        if ts is not None:
            return ts
        if read > _PROBE_BYTES:
            return None
    return None


def locate_window_start(path: Path, target_epoch: float, ts_of_line) -> tuple[int, bool]:
    """Largest probed offset whose next timestamped line is older than ``target_epoch``.

    Returns ``(offset, proven)``. ``proven`` means every line stamped at or after ``target_epoch`` is at or after
    ``offset``: the offset's following line is older than the target (invariant of the search), or ``offset`` is 0
    and the file has no rotated predecessor (``<file>.1``). Assumes the file is appended in time order.
    """
    size = path.stat().st_size
    lo, hi = 0, size
    with open(path, "rb") as f:
        while hi - lo > _BINARY_SEARCH_STOP:
            mid = (lo + hi) // 2
            ts = _first_ts_after(f, mid, size, ts_of_line)
            if ts is None or ts >= target_epoch:
                hi = mid
            else:
                lo = mid
    if lo > 0:
        return lo, True
    with open(path, "rb") as f:
        first = _first_ts_after(f, 0, size, ts_of_line)
    if first is not None and first < target_epoch:
        return 0, True  # the file's own first line predates the window: nothing older can matter
    return 0, not path.with_name(path.name + ".1").exists()


def _event_ts(line: bytes) -> float | None:
    try:
        value = json.loads(line).get("ts_epoch")
    except Exception:  # noqa: BLE001 - a torn/odd line is simply not a timestamp
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _repl_ts(line: bytes) -> float | None:
    m = _EXEC_HEADER_TS.match(line)
    if not m:
        return None
    try:
        from datetime import datetime

        return datetime.strptime(m.group(1).decode(), "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------------------------
# Inference-tap events
# ---------------------------------------------------------------------------------------------------------
def read_batch_events(
    events_path: Path,
    batch_ids: Sequence[str],
    start_epoch: float,
    *,
    max_scan_bytes: int = DEFAULT_MAX_EVENTS_SCAN_BYTES,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    """``(requests by request_id, reasons the read is not provably whole, stats)`` for the trial's batches."""
    reasons: list[str] = []
    stats: dict[str, Any] = {"events_scanned_bytes": 0}
    if not events_path.is_file():
        return {}, ["events_file_missing"], stats
    needles = [f'"batch_id":"{b}"'.encode() for b in batch_ids]
    offset, proven = locate_window_start(events_path, start_epoch - START_MARGIN_S, _event_ts)
    if not proven:
        reasons.append("events_window_start_not_proven")
    requests: dict[str, dict[str, Any]] = {}
    scanned = 0
    with open(events_path, "rb") as f:
        f.seek(offset)
        if offset > 0:
            f.readline()
        for line in f:
            scanned += len(line)
            if scanned > max_scan_bytes:
                reasons.append("events_scan_cap_hit")
                break
            if not any(n in line for n in needles):
                continue
            try:
                ev = json.loads(line)
            except Exception:  # noqa: BLE001
                reasons.append("events_unparseable_line")
                continue
            rid = str(ev.get("request_id") or "")
            if not rid:
                continue
            req = requests.setdefault(
                rid,
                {"request_id": rid, "task_id": "", "role": "", "batch_id": "", "ts": None,
                 "start": False, "end": False, "chunks": [], "response": None},
            )
            req["task_id"] = req["task_id"] or str(ev.get("task_id") or "")
            req["role"] = req["role"] or str(ev.get("role") or "")
            req["batch_id"] = req["batch_id"] or str(ev.get("batch_id") or "")
            ts = ev.get("ts_epoch")
            if isinstance(ts, (int, float)) and (req["ts"] is None or ts < req["ts"]):
                req["ts"] = float(ts)
            kind = ev.get("event")
            if kind == "start":
                req["start"] = True
            elif kind == "end":
                req["end"] = True
            elif kind == "chunk":
                req["chunks"].append(str(ev.get("text") or ""))
            elif kind == "response":
                req["response"] = str(ev.get("text") or "")
    stats["events_scanned_bytes"] = scanned
    seen_batches = {r["batch_id"] for r in requests.values()}
    for b in batch_ids:
        if b not in seen_batches:
            reasons.append("batch_without_events")
            break
    for req in requests.values():
        if not (req["start"] and req["end"]):
            reasons.append("request_without_start_or_end")
            break
    if not requests:
        reasons.append("no_requests_for_trial_batches")
    return requests, reasons, stats


# ---------------------------------------------------------------------------------------------------------
# REPL tap
# ---------------------------------------------------------------------------------------------------------
def parse_repl_tap(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """``(exec headers, result blocks, parse reasons)`` of a REPL tap text. Result blocks are length-delimited."""
    execs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    reasons: list[str] = []
    pos = 0
    while True:
        m = _REPL_BLOCK.search(text, pos)
        if m is None:
            break
        if m.group("rtask") is None:
            task = _TASK_META.search(m.group("meta") or "")
            execs.append({"turn": int(m.group("xturn")), "task_id": task.group(1) if task else ""})
            pos = m.end()
            continue
        length = int(m.group("chars"))
        body_start = m.end() + 1
        body_end = body_start + length
        body = text[body_start:body_end]
        rest = text[body_end:body_end + 80]
        if len(body) != length or not (rest == "" or rest.startswith("=" * 72) or _REPL_BLOCK.match(rest)):
            reasons.append("repl_result_block_length_mismatch")
            pos = m.end()
            continue
        results.append({"turn": int(m.group("rturn")), "task_id": m.group("rtask"), "body": body})
        pos = body_end
    return execs, results, reasons


def read_repl_turns(
    repl_path: Path,
    start_epoch: float,
    task_ids: set[str],
    *,
    max_scan_bytes: int = DEFAULT_MAX_REPL_SCAN_BYTES,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """``(result blocks of the trial's tasks in file order, reasons not provably whole, stats)``."""
    reasons: list[str] = []
    stats: dict[str, Any] = {"repl_scanned_bytes": 0}
    if not repl_path.is_file():
        return [], ["repl_tap_missing"], stats
    offset, proven = locate_window_start(repl_path, start_epoch - START_MARGIN_S, _repl_ts)
    if not proven:
        reasons.append("repl_window_start_not_proven")
    size = repl_path.stat().st_size
    if size - offset > max_scan_bytes:
        stats["repl_scanned_bytes"] = max_scan_bytes
        return [], reasons + ["repl_scan_cap_hit"], stats
    with open(repl_path, "rb") as f:
        f.seek(offset)
        if offset > 0:
            f.readline()
        raw = f.read()
    stats["repl_scanned_bytes"] = len(raw)
    execs, results, parse_reasons = parse_repl_tap(raw.decode("utf-8", errors="replace"))
    reasons.extend(parse_reasons)
    mine_exec = [e for e in execs if e["task_id"] in task_ids]
    mine = [r for r in results if r["task_id"] in task_ids]
    for task in {e["task_id"] for e in mine_exec}:
        if sum(1 for e in mine_exec if e["task_id"] == task) != sum(1 for r in mine if r["task_id"] == task):
            reasons.append("repl_exec_without_result")
            break
    if any(_TAP_CUT.search(r["body"]) for r in mine):
        reasons.append("repl_result_cut_by_tap")
    return mine, reasons, stats


# ---------------------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------------------
def build_capture(
    events_path: Path,
    repl_path: Path,
    batch_ids: Sequence[str],
    start_epoch: float,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_events_scan_bytes: int = DEFAULT_MAX_EVENTS_SCAN_BYTES,
    max_repl_scan_bytes: int = DEFAULT_MAX_REPL_SCAN_BYTES,
) -> Capture | None:
    """Join the trial's tap events and REPL tap into one trace; None when there is nothing to join on."""
    ids = [str(b) for b in batch_ids if str(b)]
    if not ids:
        return None
    requests, reasons, stats = read_batch_events(
        events_path, ids, start_epoch, max_scan_bytes=max_events_scan_bytes
    )
    task_ids = {r["task_id"] for r in requests.values() if r["task_id"]}
    repl_results, repl_reasons, repl_stats = read_repl_turns(
        repl_path, start_epoch, task_ids, max_scan_bytes=max_repl_scan_bytes
    )
    reasons = list(dict.fromkeys(reasons + repl_reasons))
    stats.update(repl_stats)

    by_task: dict[str, list[dict[str, Any]]] = {}
    for req in requests.values():
        by_task.setdefault(req["task_id"], []).append(req)
    pieces: list[str] = []
    total = 0
    capped = False
    budget = 2 * max_chars  # bound memory while assembling; the final cut is at max_chars

    def add(piece: str) -> None:
        nonlocal total, capped
        total += len(piece) + 1
        if total > max_chars:
            capped = True
        if total <= budget:
            pieces.append(piece)

    ordered = sorted(by_task.items(), key=lambda kv: min((r["ts"] or 0.0) for r in kv[1]))
    for task, reqs in ordered:
        add(f"=== task {task or '?'} ===")
        for req in sorted(reqs, key=lambda r: (r["ts"] or 0.0, r["request_id"])):
            response = req["response"] if req["response"] is not None else "".join(req["chunks"])
            add(f"ROLE={req['role'] or '?'} REQUEST={req['request_id']}\nRESPONSE:\n{response}")
        for line in tag_failure_turns(r for r in repl_results if r["task_id"] == task):
            add(line)
    text = "\n".join(pieces)
    if capped:
        reasons.append("trajectory_cap_hit")
        text = TRUNCATED_MARKER + "\n" + text[:max_chars]
    stats.update({"requests": len(requests), "tasks": len(task_ids), "repl_results": len(repl_results),
                  "chars": min(total, max_chars), "capped": capped})
    if not text.strip():
        return None
    return Capture(text=text, complete=not reasons, reasons=reasons, stats=stats)


def events_path_for(tap_path: Path) -> Path:
    """The structured-events sibling of the plaintext tap (same rule as ``inference_tap._structured_event_path``)."""
    override = os.environ.get("INFERENCE_TAP_EVENTS_FILE", "").strip()
    if override:
        return Path(override)
    if tap_path.name == "inference_tap.log":
        return tap_path.with_name("inference_tap_events.jsonl")
    return tap_path.with_suffix((tap_path.suffix or ".log") + ".events.jsonl")
