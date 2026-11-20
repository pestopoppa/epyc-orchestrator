"""Codemode reliability fix 2: REPL turn memory (``ORCHESTRATOR_REPL_TURN_MEMORY=N``, DEFAULT 0 = off).

CODEMODE-AB found the code-mode root prompt STATELESS: each turn saw the task, 80-char variable previews, 1500 chars of
the last output and 500 chars of the last error, but never its own previous code. The model re-wrote the same refused
or exploratory snippet every turn. CodeAct / smolagents keep the full step memory; this carries the last N turns of
``(code, output-or-error)`` with a ~4000-char preview each.

What production carried before this flag (checked 2026-11-18): only ``session_log`` (``src/graph/session_log.py``): a
lossy summary of the first 2 and last 2 turns as one line each (code HASH, line count, 60 chars of output or error),
plus verbatim history only under the experimental ``context_arm_ab``/``session_log_verbatim`` flags. No default path
re-shows the model its own previous code.

The block is APPENDED to the end of the prompt (``RootLMPrompt.history``) so every section before it keeps its bytes
and the KV-cache prefix; across turns only the tail grows (and the window slides only once N turns exist).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from src.flag_env import flag_env

ENV_TURN_MEMORY = "ORCHESTRATOR_REPL_TURN_MEMORY"
DEFAULT_PREVIEW_CHARS = 4000
MAX_TURNS = 16


def turn_memory_n(environ: Mapping[str, str] | None = None) -> int:
    """Number of prior turns to carry; 0 (default, unset, non-integer or negative) means off."""
    env = flag_env() if environ is None else environ
    try:
        n = int(str(env.get(ENV_TURN_MEMORY, "0")).strip() or "0")
    except ValueError:
        return 0
    return max(0, min(n, MAX_TURNS))


# Prefix-cache layout (flags ``history_append_only`` / ``tool_output_bound``, default off).
APPEND_ONLY_CHAR_BUDGET = 60000  # hard limit on stored history chars before ONE truncation
APPEND_ONLY_TRUNCATE_TO = APPEND_ONLY_CHAR_BUDGET // 2  # the truncation lands here, so the next one is far away
TOOL_OUTPUT_BOUND_CHARS = 2000


def bound_text(text: str, limit: int = TOOL_OUTPUT_BOUND_CHARS) -> str:
    """Deterministic head:tail (3:1) cut of ``text`` to about ``limit`` chars with a fixed marker."""
    if len(text) <= limit:
        return text
    tail = limit // 4
    head = limit - tail
    return text[:head] + f"\n...[{len(text) - head - tail} chars omitted by tool_output_bound]...\n" + text[len(text) - tail:]


def _entry_chars(e: Mapping[str, Any]) -> int:
    return sum(min(len(str(e.get(k, "") or "")), DEFAULT_PREVIEW_CHARS) for k in ("code", "output", "error"))


class TurnMemory:
    """Bounded list of ``{"turn", "code", "output", "error"}`` entries.

    Default (``append_only`` False): a sliding window of the last ``n`` entries, exactly the C3 behaviour.
    ``append_only``: keep every entry (the ``n`` window is not applied) until the stored size passes
    ``APPEND_ONLY_CHAR_BUDGET``; then drop the oldest entries ONCE, down to ``APPEND_ONLY_TRUNCATE_TO``, so
    already-sent turns are never rewritten in between. ``bound_output`` cuts output/error text at store time.
    """

    def __init__(self, n: int, append_only: bool = False, bound_output: bool = False):
        self.n = max(0, int(n))
        self.append_only = bool(append_only)
        self.bound_output = bool(bound_output)
        self._d: list[dict[str, Any]] = []

    def add(self, turn: int, code: str, output: str = "", error: str = "") -> None:
        if not self.n:
            return
        output, error = output or "", error or ""
        if self.bound_output:
            output, error = bound_text(output), bound_text(error)
        self._d.append({"turn": int(turn), "code": code or "", "output": output, "error": error})
        if not self.append_only:
            del self._d[: max(0, len(self._d) - self.n)]
            return
        total = sum(_entry_chars(e) for e in self._d)
        if total > APPEND_ONLY_CHAR_BUDGET:
            while len(self._d) > 1 and total > APPEND_ONLY_TRUNCATE_TO:
                total -= _entry_chars(self._d.pop(0))

    def entries(self) -> list[dict[str, Any]]:
        return list(self._d)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n...[{len(text) - limit} more chars; print() less or write to a file]"


def render_turn_memory(entries: Sequence[Mapping[str, Any]], n: int, preview_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    """The ``## Previous Turns`` block for the last ``n`` entries ('' when off or empty). Oldest first, so a new turn
    only appends. An entry shows its code and then the error when it failed, else the output."""
    if n <= 0 or not entries:
        return ""
    out = ["## Previous Turns (your own earlier code and what it printed)"]  # fixed heading: appends never rewrite it
    for e in list(entries)[-n:]:
        out.append(f"### Turn {int(e.get('turn', 0)) + 1}")
        out.append("Code:\n```python\n" + _clip(str(e.get("code", "")), preview_chars) + "\n```")
        err = str(e.get("error", "") or "")
        if err:
            out.append("Error:\n```\n" + _clip(err, preview_chars) + "\n```")
        else:
            res = str(e.get("output", "") or "")
            out.append("Output:\n```\n" + (_clip(res, preview_chars) if res else "(no output)") + "\n```")
    return "\n".join(out)


def turn_memory_kwargs(state: Any) -> dict[str, Any]:
    """``{"turn_history": [...]}`` for ``build_root_lm_prompt`` when turn memory is on and the task has history, else
    ``{}`` (so the flag-off call is byte-identical to the legacy call)."""
    if not turn_memory_n():
        return {}
    mem = getattr(state, "turn_memory", None)
    entries = mem.entries() if isinstance(mem, TurnMemory) else []
    return {"turn_history": entries} if entries else {}
