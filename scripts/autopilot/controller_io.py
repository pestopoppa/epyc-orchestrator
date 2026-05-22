"""Controller invocation + action extraction + scope validation.

Extracted from autopilot.py during the 2026-05-22 Tranche-5 refactor. The
controller is the Claude CLI subprocess that proposes the next action;
this module handles invocation, JSON action extraction, and AP-9
single-variable scope validation.

`autopilot.py` keeps the public function names as thin re-imports.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("autopilot")


def invoke_controller(
    prompt: str,
    session_id: str | None = None,
    timeout: int = 300,
    *,
    cwd: Path | str | None = None,
) -> tuple[str, str | None]:
    """Invoke Claude CLI for meta-reasoning.

    Returns (response_text, session_id). `cwd` is the working directory
    Claude runs in; defaults to current process cwd if not provided.
    """
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--allowedTools", "Read,Grep,Glob",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
        stdout, stderr = proc.communicate(timeout=timeout)

        if proc.returncode != 0:
            log.error("Controller failed (rc=%d): %s", proc.returncode, stderr[:500])
            return "", session_id

        try:
            response = json.loads(stdout)
            new_session = response.get("session_id", session_id)
            return response.get("result", stdout), new_session
        except json.JSONDecodeError:
            return stdout, session_id

    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("Controller timed out after %ds", timeout)
        return "", session_id
    except FileNotFoundError:
        log.error("Claude CLI not found")
        return "", session_id


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
