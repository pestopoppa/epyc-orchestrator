"""Server-log crash classification for dropped-connection failures (flag ``server_log_crash_class``).

A dropped connection, ``connect_error`` or an empty reply is usually the SYMPTOM of a
llama-server that died under the request; the server log names the cause. This module tails
the role's log and reports a crash marker so the failure can be classed ``candidate_crash``
instead of infra.

INFO-marker gap does not matter here: llama.cpp needs ``-lv 4`` to print INFO lines, but
GGML_ASSERT, SIGSEGV, "Segmentation fault" and "Aborted" are printed at error level / by the
shell or kernel into the same log, so they are present at the default verbosity.

Log location: ``LOG_DIR / llama-server-<port>.log`` (scripts/server/orchestrator_stack.py
``llama-server-{port}.log``); ports come from the role's NUMA instances.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Iterable

# SIGSEGV alternation mirrors orchestration/repl_memory/episodic_store.py (_SYMPTOM_PATTERN_STRINGS).
CRASH_MARKER = re.compile(
    r"GGML_ASSERT|sigsegv|segmentation fault|signal 11|\bAborted\b",
    re.IGNORECASE,
)

# Reasons that look like "the server went away" (the only ones worth a log read).
DROP_REASONS = frozenset({"connect_error", "empty_response", "connection_drop"})
_DROP_TEXT = re.compile(
    r"RemoteDisconnected|Remote end closed connection|Connection (?:refused|reset)|"
    r"ConnectionResetError|ConnectError|server died",
    re.IGNORECASE,
)

DEFAULT_TAIL_BYTES = 256 * 1024
DEFAULT_WINDOW_S = 300.0


def is_drop(reason: str | None, error_text: str = "") -> bool:
    return (reason or "") in DROP_REASONS or bool(_DROP_TEXT.search(error_text or ""))


def scan_log_for_crash(
    log_path: str | os.PathLike,
    *,
    failure_time: float | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
) -> str | None:
    """Return the first crash-marker line in the log tail, or None.

    When ``failure_time`` (epoch s) is given, a log last modified more than ``window_s``
    before it cannot hold the crash (a stale crash from an earlier run is not evidence).
    """
    p = Path(log_path)
    try:
        st = p.stat()
        if failure_time is not None and st.st_mtime < failure_time - window_s:
            return None
        with p.open("rb") as f:
            if st.st_size > tail_bytes:
                f.seek(-tail_bytes, os.SEEK_END)
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if CRASH_MARKER.search(line):
            return line.strip()[:300]
    return None


def server_log_crash(
    ports: Iterable[int],
    *,
    log_dir: str | os.PathLike,
    failure_time: float | None = None,
    window_s: float = DEFAULT_WINDOW_S,
) -> str | None:
    """First crash line across the role's per-port llama-server logs, else None."""
    if failure_time is None:
        failure_time = time.time()
    for port in ports:
        hit = scan_log_for_crash(
            Path(log_dir) / f"llama-server-{int(port)}.log",
            failure_time=failure_time,
            window_s=window_s,
        )
        if hit:
            return hit
    return None
