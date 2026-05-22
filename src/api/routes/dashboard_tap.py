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
