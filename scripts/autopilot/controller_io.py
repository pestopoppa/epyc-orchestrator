"""Controller invocation + action extraction + scope validation.

Extracted from autopilot.py during the 2026-05-22 Tranche-5 refactor. The
controller is the Claude CLI subprocess that proposes the next action;
this module handles invocation, JSON action extraction, and AP-9
single-variable scope validation.

2026-05-22 streaming overhaul:
- Switched `--output-format json` (single final JSON, fully buffered) to
  `stream-json` (line-delimited events emitted live). Caller can now tail
  the planner output as it streams instead of waiting up to 300s.
- Each line is teed to a per-call planner tap file at
  PLANNER_TAP_PATH so the dashboard can SSE-stream it. The tap file is
  appended across calls (with section separators) so the recent planning
  history survives across trials.

`autopilot.py` keeps the public function names as thin re-imports.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("autopilot")

# Tap file the planner subprocess streams into. Dashboard tails this via
# /dashboard/events/planner_tap. The path is fixed (not per-invocation) so
# a single SSE consumer can watch every planner session.
PLANNER_TAP_PATH = Path("/mnt/raid0/llm/tmp/planner_tap.log")

# Persistent JSONL archive of planner sessions — survives /tmp wipes and
# is queryable for reasoning-trace research. One line per session with
# the full event list. Lives alongside other autopilot logs so it's
# included in the same backup/retention scheme.
PLANNER_ARCHIVE_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/logs/planner_archive.jsonl"
)


def _open_planner_tap() -> Any:
    """Return an append-mode handle on the planner tap, creating dirs if needed."""
    try:
        PLANNER_TAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        return open(PLANNER_TAP_PATH, "a", buffering=1)  # line-buffered
    except Exception as exc:
        log.warning("Could not open planner tap %s: %s", PLANNER_TAP_PATH, exc)
        return None


def _append_planner_archive(record: dict) -> None:
    """Append one planner-session record to the persistent JSONL archive.

    Best-effort: silent on failure (the tap file is the live source of
    truth; archive is for after-the-fact analysis). One JSONL line per
    session with timestamp, duration, session_id, prompt hash + length,
    captured events, and final result.
    """
    try:
        PLANNER_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PLANNER_ARCHIVE_PATH, "a") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as exc:
        log.debug("planner archive append failed: %s", exc)


def _summarize_event(line: str) -> str:
    """Produce a human-readable one-line summary of a stream-json event.

    Stream-json events look like:
        {"type":"system","subtype":"init", ...}
        {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
        {"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read",...}]}}
        {"type":"user","message":{"content":[{"type":"tool_result","content":"..."}]}}
        {"type":"result","subtype":"success","total_cost_usd":...,"result":"..."}

    We summarize so the tap file is readable even at a glance.
    """
    try:
        evt = json.loads(line)
    except json.JSONDecodeError:
        return line.rstrip()

    t = evt.get("type", "?")
    if t == "system":
        sub = evt.get("subtype", "")
        return f"[system:{sub}] session={evt.get('session_id','')[:8]}…"
    if t == "assistant":
        msg = evt.get("message", {})
        parts = []
        for c in msg.get("content", []):
            ctype = c.get("type")
            if ctype == "text":
                txt = c.get("text", "").strip()
                if txt:
                    parts.append(txt[:300])
            elif ctype == "tool_use":
                name = c.get("name", "?")
                inp = c.get("input", {})
                arg_preview = json.dumps(inp)[:160]
                parts.append(f"TOOL_USE {name}({arg_preview})")
        return "[assistant] " + " | ".join(parts) if parts else "[assistant] (empty)"
    if t == "user":
        msg = evt.get("message", {})
        for c in msg.get("content", []):
            if c.get("type") == "tool_result":
                content = c.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        (b.get("text", "") if isinstance(b, dict) else str(b))
                        for b in content
                    )
                preview = str(content)[:240].replace("\n", " / ")
                return f"[tool_result] {preview}"
        return "[user] (no tool_result)"
    if t == "result":
        sub = evt.get("subtype", "")
        cost = evt.get("total_cost_usd")
        dur = evt.get("duration_ms")
        return (
            f"[result:{sub}] cost=${cost:.4f} duration={dur}ms"
            if isinstance(cost, (int, float))
            else f"[result:{sub}]"
        )
    return f"[{t}] {line.rstrip()[:200]}"


def invoke_controller(
    prompt: str,
    session_id: str | None = None,
    timeout: int = 300,
    *,
    cwd: Path | str | None = None,
) -> tuple[str, str | None]:
    """Invoke Claude CLI for meta-reasoning with live streaming to planner tap.

    Returns (response_text, session_id). `cwd` is the working directory
    Claude runs in; defaults to current process cwd if not provided.

    Streams each event to PLANNER_TAP_PATH as it's emitted so the dashboard
    can watch the planner reason in real time.
    """
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",  # required by claude CLI for stream-json output
        "--allowedTools", "Read,Grep,Glob",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])

    tap = _open_planner_tap()
    if tap is not None:
        try:
            tap.write(f"\n{'=' * 72}\n[{datetime.now().isoformat(timespec='seconds')}] PLANNER session start\n")
            if session_id:
                tap.write(f"resume_session: {session_id}\n")
            tap.write(f"prompt_chars: {len(prompt)}\n")
            tap.write(f"{'-' * 72}\n")
            tap.flush()
        except Exception:
            pass

    result_text = ""
    final_session_id = session_id
    proc: subprocess.Popen | None = None
    reader_thread: threading.Thread | None = None
    # Captured events for the archive write at end. Each entry is the
    # one-line summary; full raw JSON would bloat the JSONL too much for
    # routine grep, and the user can always reconstruct via the live tap.
    archive_events: list[str] = []
    archive_meta: dict[str, Any] = {}
    session_start_ts = time.time()

    def _drain_stdout(p: subprocess.Popen):
        """Read p.stdout line-by-line; tee each line to tap; capture result."""
        nonlocal result_text, final_session_id
        assert p.stdout is not None
        try:
            for raw_line in p.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                # Tee summarized + raw to tap, and also remember the
                # summary for the archive write.
                summary = _summarize_event(line)
                archive_events.append(summary)
                if tap is not None:
                    try:
                        tap.write(summary + "\n")
                        tap.flush()
                    except Exception:
                        pass
                # Capture the final result + session_id from the result event
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "result":
                    result_text = evt.get("result", "") or result_text
                    final_session_id = evt.get("session_id", final_session_id)
                    archive_meta["total_cost_usd"] = evt.get("total_cost_usd")
                    archive_meta["duration_ms"] = evt.get("duration_ms")
                    archive_meta["subtype"] = evt.get("subtype")
                elif evt.get("type") == "system" and evt.get("subtype") == "init":
                    final_session_id = evt.get("session_id", final_session_id)
        except Exception as exc:
            log.warning("Planner stdout drain failed: %s", exc)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            cwd=str(cwd) if cwd else None,
        )

        reader_thread = threading.Thread(target=_drain_stdout, args=(proc,), daemon=True)
        reader_thread.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            log.error("Controller timed out after %ds", timeout)
            if tap is not None:
                try:
                    tap.write(f"[TIMEOUT after {timeout}s]\n{'=' * 72}\n")
                    tap.flush()
                except Exception:
                    pass
            return "", session_id

        reader_thread.join(timeout=5)

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            log.error("Controller failed (rc=%d): %s", proc.returncode, stderr[:500])
            if tap is not None:
                try:
                    tap.write(f"[FAIL rc={proc.returncode}] {stderr[:400]}\n{'=' * 72}\n")
                    tap.flush()
                except Exception:
                    pass
            return "", session_id

        if tap is not None:
            try:
                tap.write(f"[END] result_chars={len(result_text)} session={(final_session_id or '')[:8]}…\n{'=' * 72}\n")
                tap.flush()
            except Exception:
                pass

        # Archive write (persistent JSONL, survives /tmp wipe)
        import hashlib
        _append_planner_archive({
            "ts": session_start_ts,
            "ts_iso": datetime.fromtimestamp(session_start_ts).isoformat(timespec="seconds"),
            "duration_s": time.time() - session_start_ts,
            "session_id": final_session_id,
            "resume_session_id": session_id,
            "prompt_chars": len(prompt),
            "prompt_sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "result_chars": len(result_text),
            "result_preview": (result_text or "")[:500],
            "n_events": len(archive_events),
            "events": archive_events[-200:],  # last 200 events, prevents megabyte lines
            **archive_meta,
        })

        return result_text, final_session_id

    except FileNotFoundError:
        log.error("Claude CLI not found")
        return "", session_id
    finally:
        if tap is not None:
            try:
                tap.close()
            except Exception:
                pass


def _unwrap_action(data: Any) -> dict[str, Any] | None:
    """Unwrap action from list or validate it's a dict with a 'type' field."""
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    if isinstance(data, dict) and "type" in data:
        return data
    return None


def extract_action(text: str) -> dict[str, Any] | None:
    """Extract structured action from controller response.

    Looks for ```json:autopilot_actions``` block first; falls back to any
    ```json``` block whose payload is a dict with a 'type' field.
    """
    marker = "```json:autopilot_actions"
    if marker in text:
        start = text.index(marker) + len(marker)
        end = text.index("```", start)
        try:
            data = json.loads(text[start:end].strip())
            return _unwrap_action(data)
        except json.JSONDecodeError as e:
            log.error("Failed to parse action JSON: %s", e)
            return None

    # Fallback: look for any JSON block
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        try:
            data = json.loads(text[start:end].strip())
            if isinstance(data, dict) and "type" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def validate_single_variable(action: dict[str, Any]) -> str | None:
    """AP-9: Validate that an action proposes a single-variable change.

    Returns an error message if the action violates the single-variable
    constraint, or None if it passes.
    """
    action_type = action.get("type", "")

    if action_type in ("prompt_mutation", "gepa_optimize"):
        target = action.get("file", "")
        if not target:
            return f"{action_type} must specify a single target file"
        if "," in target or ";" in target:
            return f"{action_type} targets multiple files: {target}"

    elif action_type == "code_mutation":
        target = action.get("file", "")
        if not target:
            return "code_mutation must specify a single target file"

    elif action_type == "structural_experiment":
        flags = action.get("flags", {})
        if len(flags) > 1:
            return (
                f"structural_experiment changes {len(flags)} flags at once "
                f"({list(flags.keys())}); limit to 1 for clean attribution"
            )

    elif action_type == "numeric_trial":
        params = action.get("params", {})
        # Optuna-suggested params are fine (controlled search), but explicit
        # multi-param overrides violate single-variable principle.
        if len(params) > 1:
            return (
                f"numeric_trial sets {len(params)} params explicitly; "
                "limit to 1 for clean attribution (Optuna suggestions exempt)"
            )

    return None
