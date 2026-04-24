"""REPL tap file helpers for graph execution diagnostics."""

from __future__ import annotations

import os
import threading

_REPL_TAP_PATH: str | None = None
_repl_tap_lock = threading.Lock()
_IN_PYTEST = bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _get_repl_tap_path() -> str:
    global _REPL_TAP_PATH
    if _REPL_TAP_PATH is None:
        try:
            from src.config import get_config
            _REPL_TAP_PATH = str(get_config().paths.tmp_dir / "repl_tap.log")
        except Exception:
            _REPL_TAP_PATH = "/mnt/raid0/llm/tmp/repl_tap.log"
    return _REPL_TAP_PATH


def tap_write_repl_exec(code: str, turn: int) -> None:
    """Write REPL execution input to the REPL tap file."""
    if _IN_PYTEST:
        return
    try:
        preview = code[:4000]
        if len(code) > 4000:
            preview += f"\n... [{len(code) - 4000} chars truncated]"
        # Section marker so the TUI (section_aware=True) clears old content
        # and always shows the most recent REPL execution.
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"{'=' * 72}\n"
            f"[{ts}] REPL turn {turn}\n"
            f"{'-' * 72}\n"
            f"[turn {turn}] $ python3 <<'CODE'\n"
            f"{preview}\n"
            f"CODE\n"
        )
        with _repl_tap_lock:
            with open(_get_repl_tap_path(), "a") as f:
                f.write(text)
                f.flush()
    except Exception:
        pass


def tap_write_repl_result(
    output: str, error: str | None, is_final: bool, turn: int,
) -> None:
    """Write REPL execution result to the REPL tap file."""
    if _IN_PYTEST:
        return
    try:
        parts: list[str] = []
        if is_final:
            parts.append(f"[turn {turn}] FINAL")
        if output:
            out_preview = output[:4000]
            if len(output) > 4000:
                out_preview += f"\n... [{len(output) - 4000} chars truncated]"
            parts.append(out_preview)
        elif not error:
            parts.append(f"[turn {turn}] (no output)")
        if error:
            err_preview = error[:2000]
            if len(error) > 2000:
                err_preview += f"\n... [{len(error) - 2000} chars truncated]"
            parts.append(f"[turn {turn}] ERROR:\n{err_preview}")
        parts.append("")  # trailing newline
        text = "\n".join(parts)
        with _repl_tap_lock:
            with open(_get_repl_tap_path(), "a") as f:
                f.write(text)
                f.flush()
    except Exception:
        pass
