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

import hashlib
import json
import logging
import os
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
PLANNER_SUBPROCESS_STATUS_PATH = Path(
    "/mnt/raid0/llm/tmp/autopilot_planner_subprocess.json"
)

# Do not inherit the operator's last interactive Claude model. Fable access is
# metered/temporary, and a stale global default can brick AutoPilot planning.
DEFAULT_CLAUDE_MODEL = "opus"
DEFAULT_CLAUDE_FALLBACK_MODEL = "sonnet"
PLANNER_ALLOWED_TOOLS = {"Read", "Grep", "Glob"}
PLANNER_DISALLOWED_TOOLS = {
    "Bash",
    "CronCreate",
    "CronDelete",
    "CronList",
    "DesignSync",
    "Edit",
    "EnterWorktree",
    "ExitPlanMode",
    "MultiEdit",
    "NotebookEdit",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
}
PLANNER_CLI_DISALLOWED_TOOLS = PLANNER_DISALLOWED_TOOLS - {
    # Current Claude Code rejects this legacy alias in --disallowedTools, but
    # keep it in PLANNER_DISALLOWED_TOOLS so stream-level detection still fails
    # closed if an older planner event emits it.
    "MultiEdit",
}

_FALLBACK_NUMERIC_SURFACES = {"memrl_retrieval", "think_harder", "monitor", "escalation"}


def _configured_numeric_surfaces() -> set[str]:
    try:
        from species.numeric_swarm import SURFACES as _NS_SURFACES
    except Exception:
        return set(_FALLBACK_NUMERIC_SURFACES)

    surfaces = {
        surface
        for surface in _NS_SURFACES
        if isinstance(surface, str) and surface.strip()
    }
    return surfaces or set(_FALLBACK_NUMERIC_SURFACES)


_RAW_NUMERIC_SURFACES = _configured_numeric_surfaces()
_SUPPRESSED_NUMERIC_SURFACES: set[str] = set()
_NUMERIC_SURFACES = set(_RAW_NUMERIC_SURFACES)
_PROMPT_MUTATIONS = {"targeted_fix", "compress", "few_shot_evolution"}
_CODE_MUTATIONS = {"targeted_fix"}
_SLOT_SCORERS = {"expected_attention", "knorm"}


def set_suppressed_numeric_surfaces(surfaces: set[str] | list[str] | tuple[str, ...]) -> None:
    """Update startup-scoped numeric surfaces hidden from planner validation."""
    global _NUMERIC_SURFACES
    _SUPPRESSED_NUMERIC_SURFACES.clear()
    _SUPPRESSED_NUMERIC_SURFACES.update(
        str(surface).strip()
        for surface in surfaces
        if str(surface).strip() in _RAW_NUMERIC_SURFACES
    )
    _NUMERIC_SURFACES = set(_RAW_NUMERIC_SURFACES) - _SUPPRESSED_NUMERIC_SURFACES
    _ACTION_SCHEMAS["numeric_trial"]["enums"]["surface"] = _NUMERIC_SURFACES


def _write_planner_subprocess_status(
    *,
    status: str,
    prompt: str,
    cmd: list[str],
    child_pid: int | None,
    started_at: float,
    returncode: int | None = None,
    error: str = "",
) -> None:
    """Best-effort heartbeat for planner subprocess lifetime diagnostics."""
    payload = {
        "status": status,
        "provider": "claude",
        "parent_pid": os.getpid(),
        "child_pid": child_pid,
        "started_at": started_at,
        "updated_at": time.time(),
        "duration_s": max(0.0, time.time() - started_at),
        "prompt_chars": len(prompt),
        "prompt_sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "cmd": [cmd[0], *["<prompt>" if item == prompt else item for item in cmd[1:]]],
        "returncode": returncode,
        "error": error[:1000],
    }
    try:
        PLANNER_SUBPROCESS_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLANNER_SUBPROCESS_STATUS_PATH.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        log.debug("planner subprocess heartbeat write failed", exc_info=True)

_ACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "seed_batch": {
        "allowed": {"type", "n_questions", "suites"},
    },
    "numeric_trial": {
        "allowed": {"type", "surface", "params"},
        "enums": {"surface": _NUMERIC_SURFACES},
    },
    "prompt_mutation": {
        "allowed": {"type", "file", "mutation", "description"},
        "required": {"file"},
        "enums": {"mutation": _PROMPT_MUTATIONS},
    },
    "gepa_optimize": {
        "allowed": {"type", "file", "max_evals", "description"},
        "required": {"file"},
    },
    "code_mutation": {
        "allowed": {"type", "file", "mutation", "description"},
        "required": {"file"},
        "enums": {"mutation": _CODE_MUTATIONS},
    },
    "structural_experiment": {
        "allowed": {"type", "flags"},
        "required": {"flags"},
    },
    "structural_prune": {
        "allowed": {"type", "file", "block", "description"},
        "required": {"file", "block"},
    },
    "slot_compact": {
        "allowed": {
            "type",
            "port",
            "slot_id",
            "keep_ratio",
            "scorer",
            "keep_first",
            "n_future",
            "use_covariance",
            "layer_weights",
            "threshold",
        },
        "enums": {"scorer": _SLOT_SCORERS},
    },
    "train_routing_models": {
        "allowed": {"type", "min_memories"},
    },
    "distill_skillbank": {
        "allowed": {"type", "teacher", "categories"},
    },
    "reset_memories": {
        "allowed": {"type", "keep_seen", "keep_skills"},
    },
    "deep_eval": {
        "allowed": {"type", "tier"},
        "required": {"tier"},
        "enums": {"tier": {0, 1, 2, 3}},
    },
    "rollback": {
        "allowed": {"type", "to_checkpoint"},
        "enums": {"to_checkpoint": {"production_best"}},
    },
    "distill_knowledge": {
        "allowed": {"type", "last_n"},
    },
}


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
        # For the init event surface model + tools + cwd so the operator
        # sees the planner's environment at a glance.
        sid = evt.get("session_id", "")[:8]
        extras = []
        if evt.get("model"):
            extras.append(f"model={evt['model']}")
        tools = evt.get("tools") or []
        if tools:
            extras.append(f"tools={','.join(tools[:8])}{'…' if len(tools) > 8 else ''}")
        if evt.get("cwd"):
            extras.append(f"cwd={evt['cwd']}")
        suffix = (" " + " ".join(extras)) if extras else ""
        return f"[system:{sub}] session={sid}…{suffix}"
    if t == "assistant":
        msg = evt.get("message", {})
        parts = []
        for c in msg.get("content", []):
            ctype = c.get("type")
            if ctype == "text":
                txt = c.get("text", "").strip()
                if txt:
                    # 1200c (was 300c) — assistant text is the planner's
                    # reasoning the operator wants to read mid-stream.
                    parts.append(txt[:1200])
            elif ctype == "tool_use":
                name = c.get("name", "?")
                inp = c.get("input", {})
                # 800c (was 160c) — args include file paths / search patterns
                # / shell commands that show WHAT the planner is doing.
                arg_preview = json.dumps(inp)[:800]
                parts.append(f"TOOL_USE {name}({arg_preview})")
            elif ctype == "thinking":
                # Extended-thinking content (when the planner model emits it)
                think = c.get("thinking", "").strip()
                if think:
                    parts.append(f"THINKING: {think[:800]}")
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
                is_error = c.get("is_error", False)
                # 1500c (was 240c) — tool results are the planner's evidence;
                # truncating them hides the inputs to its next decision.
                # Newlines preserved — the tap renderer wraps the content.
                preview = str(content)[:1500]
                tag = "[tool_error] " if is_error else "[tool_result] "
                return tag + preview
        return "[user] (no tool_result)"
    if t == "result":
        sub = evt.get("subtype", "")
        cost = evt.get("total_cost_usd")
        dur = evt.get("duration_ms")
        turns = evt.get("num_turns")
        usage = evt.get("usage") or {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        parts = [f"[result:{sub}]"]
        if isinstance(cost, (int, float)):
            parts.append(f"cost=${cost:.4f}")
        if isinstance(dur, (int, float)):
            parts.append(f"duration={dur}ms")
        if isinstance(turns, (int, float)):
            parts.append(f"turns={turns}")
        if isinstance(in_tok, (int, float)) and isinstance(out_tok, (int, float)):
            parts.append(f"tokens={in_tok}in/{out_tok}out")
        return " ".join(parts)
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
        # This is a JSON controller call, not Claude Code's interactive Plan
        # Mode. Plan mode can steer the CLI toward writing ~/.claude/plans/*
        # when it wants to escalate, which wastes a planner turn and returns an
        # empty action. Keep the available tool surface explicitly read-only.
        "--permission-mode", "default",
        "--safe-mode",
        "--tools", "Read,Grep,Glob",
        "--allowedTools", "Read,Grep,Glob",
        "--disallowedTools", ",".join(sorted(PLANNER_CLI_DISALLOWED_TOOLS)),
    ]
    planner_model = os.environ.get("AUTOPILOT_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL).strip()
    if planner_model:
        cmd.extend(["--model", planner_model])
    fallback_model = os.environ.get(
        "AUTOPILOT_CLAUDE_FALLBACK_MODEL",
        DEFAULT_CLAUDE_FALLBACK_MODEL,
    ).strip()
    if fallback_model:
        cmd.extend(["--fallback-model", fallback_model])
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
    disallowed_tool_uses: list[str] = []
    session_start_ts = time.time()

    def _archive_controller_call(
        *,
        status: str,
        ok: bool,
        error: str = "",
    ) -> None:
        import hashlib

        _append_planner_archive({
            "ts": session_start_ts,
            "ts_iso": datetime.fromtimestamp(session_start_ts).isoformat(
                timespec="seconds"
            ),
            "type": "planner_provider_call",
            "provider": "claude",
            "role": "draft",
            "status": status,
            "ok": ok,
            "error": error,
            "duration_s": time.time() - session_start_ts,
            "session_id": final_session_id,
            "resume_session_id": session_id,
            "prompt_chars": len(prompt),
            "prompt_sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "result_chars": len(result_text),
            "result_preview": (result_text or "")[:500],
            "n_events": len(archive_events),
            "events": archive_events[-200:],
            **archive_meta,
        })

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
                elif evt.get("type") == "assistant":
                    msg = evt.get("message") or {}
                    for content in msg.get("content", []):
                        if not isinstance(content, dict) or content.get("type") != "tool_use":
                            continue
                        name = str(content.get("name") or "")
                        if name and name not in PLANNER_ALLOWED_TOOLS:
                            disallowed_tool_uses.append(name)
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
            env={k: v for k, v in os.environ.items() if k != "CLAUDECODE"},
        )
        _write_planner_subprocess_status(
            status="running",
            prompt=prompt,
            cmd=cmd,
            child_pid=proc.pid,
            started_at=session_start_ts,
        )

        reader_thread = threading.Thread(target=_drain_stdout, args=(proc,), daemon=True)
        reader_thread.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            log.error("Controller timed out after %ds", timeout)
            _write_planner_subprocess_status(
                status="timeout",
                prompt=prompt,
                cmd=cmd,
                child_pid=proc.pid,
                started_at=session_start_ts,
                returncode=proc.returncode,
                error=f"timeout after {timeout}s",
            )
            if tap is not None:
                try:
                    tap.write(f"[TIMEOUT after {timeout}s]\n{'=' * 72}\n")
                    tap.flush()
                except Exception:
                    pass
            _archive_controller_call(
                status="timeout",
                ok=False,
                error=f"timeout after {timeout}s",
            )
            return "", session_id

        reader_thread.join(timeout=5)

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            log.error("Controller failed (rc=%d): %s", proc.returncode, stderr[:500])
            _write_planner_subprocess_status(
                status="process_failed",
                prompt=prompt,
                cmd=cmd,
                child_pid=proc.pid,
                started_at=session_start_ts,
                returncode=proc.returncode,
                error=stderr,
            )
            if tap is not None:
                try:
                    tap.write(f"[FAIL rc={proc.returncode}] {stderr[:400]}\n{'=' * 72}\n")
                    tap.flush()
                except Exception:
                    pass
            # 2026-05-23: detect stale --resume target. claude CLI emits
            # various stderr patterns when the resumed session has been
            # pruned / wasn't persisted; returning the same stale
            # session_id back to the caller causes it to save it again
            # into autopilot state, repeating the failure every trial.
            # Clear it so the next call starts fresh. Broadened pattern
            # set to catch CLI wording drift across versions.
            _stderr_low = (stderr or "").lower()
            _stale_session_phrases = (
                "no conversation found",
                "session expired",
                "session not found",
                "conversation not found",
                "unknown session",
                "invalid session",
                "could not resume",
                "session has been deleted",
            )
            _stale_id_combo = (
                "session id" in _stderr_low and (
                    "not found" in _stderr_low
                    or "expired" in _stderr_low
                    or "invalid" in _stderr_low
                    or "no such" in _stderr_low
                )
            )
            if _stale_id_combo or any(p in _stderr_low for p in _stale_session_phrases):
                log.warning(
                    "Clearing stale planner session_id=%s (CLI reports it no longer "
                    "exists / expired); next trial will start a fresh conversation",
                    (session_id or "")[:12],
                )
                _archive_controller_call(
                    status="stale_session",
                    ok=False,
                    error=stderr[:1000],
                )
                return "", None
            _archive_controller_call(
                status="process_failed",
                ok=False,
                error=stderr[:1000],
            )
            return "", session_id

        if disallowed_tool_uses:
            tools = sorted(set(disallowed_tool_uses))
            error = "planner used disallowed tool(s): " + ", ".join(tools)
            log.error(error)
            _write_planner_subprocess_status(
                status="disallowed_tool_use",
                prompt=prompt,
                cmd=cmd,
                child_pid=proc.pid,
                started_at=session_start_ts,
                returncode=proc.returncode,
                error=error,
            )
            if tap is not None:
                try:
                    tap.write(f"[FAIL disallowed_tool_use] {error}\n{'=' * 72}\n")
                    tap.flush()
                except Exception:
                    pass
            _archive_controller_call(
                status="disallowed_tool_use",
                ok=False,
                error=error,
            )
            return "", None

        if tap is not None:
            try:
                tap.write(f"[END] result_chars={len(result_text)} session={(final_session_id or '')[:8]}…\n{'=' * 72}\n")
                tap.flush()
            except Exception:
                pass

        # Archive write (persistent JSONL, survives /tmp wipe)
        _archive_controller_call(status="success", ok=True)
        _write_planner_subprocess_status(
            status="success",
            prompt=prompt,
            cmd=cmd,
            child_pid=proc.pid,
            started_at=session_start_ts,
            returncode=proc.returncode,
        )

        return result_text, final_session_id

    except FileNotFoundError:
        log.error("Claude CLI not found")
        _write_planner_subprocess_status(
            status="missing_cli",
            prompt=prompt,
            cmd=cmd,
            child_pid=None,
            started_at=session_start_ts,
            error="Claude CLI not found",
        )
        _archive_controller_call(
            status="missing_cli",
            ok=False,
            error="Claude CLI not found",
        )
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


def extract_rationale(text: str) -> dict[str, Any]:
    """Extract the optional rationale sidecar from the controller response.

    Looks for a ```json:autopilot_rationale``` fenced block. The block carries
    the chosen action's falsifier + self-scored rubric, e.g.

        ```json:autopilot_rationale
        {"falsifier": "...", "rubric_scores": {"info_gain": 4,
         "coherence": 5, "usefulness": 3, "synthesis_note": "..."}}
        ```

    Returns a dict with two keys — `falsifier` (str) and `rubric_scores` (dict)
    — defaulting both to empty when the block is missing or malformed. The
    contract is intentionally soft: rationale capture is observability, not a
    gate, so a missing block must not abort the trial.
    """
    empty: dict[str, Any] = {"falsifier": "", "rubric_scores": {}}
    marker = "```json:autopilot_rationale"
    if marker not in text:
        return empty
    start = text.index(marker) + len(marker)
    try:
        end = text.index("```", start)
    except ValueError:
        log.warning("autopilot_rationale block has no closing fence")
        return empty
    try:
        data = json.loads(text[start:end].strip())
    except json.JSONDecodeError as e:
        log.warning("Failed to parse autopilot_rationale JSON: %s", e)
        return empty
    if not isinstance(data, dict):
        return empty
    falsifier = data.get("falsifier", "")
    rubric = data.get("rubric_scores", {})
    if not isinstance(falsifier, str):
        falsifier = str(falsifier)
    if not isinstance(rubric, dict):
        rubric = {}
    return {"falsifier": falsifier, "rubric_scores": rubric}


def _validate_action_schema(action: dict[str, Any]) -> str | None:
    action_type = action.get("type", "")
    schema = _ACTION_SCHEMAS.get(action_type)
    if schema is None:
        return None

    allowed = schema.get("allowed", set())
    extra = sorted(set(action) - allowed)
    if extra:
        return (
            f"{action_type} unsupported keys: {extra}; "
            f"allowed keys: {sorted(allowed)}"
        )

    missing = sorted(schema.get("required", set()) - set(action))
    if missing:
        if missing == ["file"] and action_type in {
            "prompt_mutation",
            "gepa_optimize",
            "code_mutation",
        }:
            return f"{action_type} must specify a single target file"
        return f"{action_type} missing required keys: {missing}"

    for key, values in schema.get("enums", {}).items():
        if key not in action:
            continue
        value = action[key]
        if isinstance(value, bool) or value not in values:
            return (
                f"{action_type} {key} must be one of {sorted(values)}; "
                f"got {value!r}"
            )

    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_int_range(
    action: dict[str, Any],
    key: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> str | None:
    if key not in action:
        return None
    value = action[key]
    if not _is_int(value):
        return f"{action.get('type', 'action')} {key} must be an integer; got {value!r}"
    if min_value is not None and value < min_value:
        return (
            f"{action.get('type', 'action')} {key} must be >= {min_value}; "
            f"got {value!r}"
        )
    if max_value is not None and value > max_value:
        return (
            f"{action.get('type', 'action')} {key} must be <= {max_value}; "
            f"got {value!r}"
        )
    return None


def _validate_number_range(
    action: dict[str, Any],
    key: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_exclusive: bool = False,
) -> str | None:
    if key not in action:
        return None
    value = action[key]
    action_type = action.get("type", "action")
    if not _is_number(value):
        return f"{action_type} {key} must be numeric; got {value!r}"
    if min_value is not None:
        below = value <= min_value if min_exclusive else value < min_value
        if below:
            op = ">" if min_exclusive else ">="
            return f"{action_type} {key} must be {op} {min_value}; got {value!r}"
    if max_value is not None and value > max_value:
        return f"{action_type} {key} must be <= {max_value}; got {value!r}"
    return None


def _validate_bool(action: dict[str, Any], key: str) -> str | None:
    if key not in action:
        return None
    value = action[key]
    if not isinstance(value, bool):
        return f"{action.get('type', 'action')} {key} must be a boolean; got {value!r}"
    return None


def _validate_str_list(action: dict[str, Any], key: str) -> str | None:
    if key not in action:
        return None
    value = action[key]
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
    ):
        return f"{action.get('type', 'action')} {key} must be a list of strings"
    return None


def validate_single_variable(action: dict[str, Any]) -> str | None:
    """AP-9: Validate that an action proposes a single-variable change.

    Returns an error message if the action violates the single-variable
    constraint, or None if it passes.
    """
    action_type = action.get("type", "")

    schema_err = _validate_action_schema(action)
    if schema_err:
        return schema_err

    if action_type in ("prompt_mutation", "gepa_optimize"):
        target = action.get("file", "")
        if not target:
            return f"{action_type} must specify a single target file"
        if "," in target or ";" in target:
            return f"{action_type} targets multiple files: {target}"
        if action_type == "gepa_optimize":
            range_err = _validate_int_range(
                action, "max_evals", min_value=1, max_value=100
            )
            if range_err:
                return range_err

    elif action_type == "code_mutation":
        target = action.get("file", "")
        if not target:
            return "code_mutation must specify a single target file"
        if "," in target or ";" in target:
            return f"code_mutation targets multiple files: {target}"

    elif action_type == "structural_experiment":
        flags = action.get("flags", {})
        if not isinstance(flags, dict):
            return "structural_experiment flags must be an object"
        if len(flags) > 1:
            return (
                f"structural_experiment changes {len(flags)} flags at once "
                f"({list(flags.keys())}); limit to 1 for clean attribution"
            )
        for key, value in flags.items():
            if not isinstance(key, str) or not isinstance(value, bool):
                return "structural_experiment flags must map string names to booleans"

    elif action_type == "numeric_trial":
        params = action.get("params", {})
        if not isinstance(params, dict):
            return "numeric_trial params must be an object"
        # Optuna-suggested params are fine (controlled search), but explicit
        # multi-param overrides violate single-variable principle.
        if len(params) > 1:
            return (
                f"numeric_trial sets {len(params)} params explicitly; "
                "limit to 1 for clean attribution (Optuna suggestions exempt)"
            )

    elif action_type == "slot_compact":
        for key, min_value, max_value in (
            ("port", 1, 65535),
            ("slot_id", 0, None),
            ("keep_first", 0, None),
            ("n_future", 1, 8192),
        ):
            range_err = _validate_int_range(
                action, key, min_value=min_value, max_value=max_value
            )
            if range_err:
                return range_err
        for key in ("keep_ratio", "threshold"):
            range_err = _validate_number_range(
                action, key, min_value=0.0, max_value=1.0, min_exclusive=True
            )
            if range_err:
                return range_err
        bool_err = _validate_bool(action, "use_covariance")
        if bool_err:
            return bool_err
        if "layer_weights" in action:
            weights = action["layer_weights"]
            if (
                not isinstance(weights, list)
                or not weights
                or any(not _is_number(weight) for weight in weights)
            ):
                return "slot_compact layer_weights must be a non-empty numeric list"

    elif action_type == "seed_batch":
        range_err = _validate_int_range(
            action, "n_questions", min_value=1, max_value=50
        )
        if range_err:
            return range_err
        list_err = _validate_str_list(action, "suites")
        if list_err:
            return list_err

    elif action_type == "train_routing_models":
        range_err = _validate_int_range(
            action, "min_memories", min_value=1, max_value=100000
        )
        if range_err:
            return range_err

    elif action_type == "distill_skillbank":
        if "teacher" in action and not isinstance(action["teacher"], str):
            return "distill_skillbank teacher must be a string"
        list_err = _validate_str_list(action, "categories")
        if list_err:
            return list_err

    elif action_type == "reset_memories":
        for key in ("keep_seen", "keep_skills"):
            bool_err = _validate_bool(action, key)
            if bool_err:
                return bool_err

    elif action_type == "deep_eval":
        # Enum and required-key checks are covered by _ACTION_SCHEMAS.
        return None

    elif action_type == "distill_knowledge":
        range_err = _validate_int_range(action, "last_n", min_value=1, max_value=100)
        if range_err:
            return range_err

    return None
