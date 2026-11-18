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
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

ENV_TURN_MEMORY = "ORCHESTRATOR_REPL_TURN_MEMORY"
DEFAULT_PREVIEW_CHARS = 4000
MAX_TURNS = 16


def turn_memory_n(environ: Mapping[str, str] | None = None) -> int:
    """Number of prior turns to carry; 0 (default, unset, non-integer or negative) means off."""
    env = os.environ if environ is None else environ
    try:
        n = int(str(env.get(ENV_TURN_MEMORY, "0")).strip() or "0")
    except ValueError:
        return 0
    return max(0, min(n, MAX_TURNS))


class TurnMemory:
    """Bounded deque of ``{"turn", "code", "output", "error"}`` entries."""

    def __init__(self, n: int):
        self.n = max(0, int(n))
        self._d: deque[dict[str, Any]] = deque(maxlen=self.n or None)

    def add(self, turn: int, code: str, output: str = "", error: str = "") -> None:
        if self.n:
            self._d.append({"turn": int(turn), "code": code or "", "output": output or "", "error": error or ""})

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
