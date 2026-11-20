"""In-process counter for degenerate-generation stream aborts (flag ``stream_degeneration_guard``).

Same pattern as ``migration_counters``: a lock-guarded dict, a snapshot and a Prometheus text
renderer, no new dependency. Labels: reason (acceptance|repetition) and role. The tokens
generated at abort are accumulated per (reason, role) and the last events are kept for inspection.
"""

from __future__ import annotations

import threading
import time

ABORTS_TOTAL = "stream_degeneration_aborts_total"
ABORT_TOKENS_TOTAL = "stream_degeneration_abort_tokens_total"
_RECENT_LIMIT = 256

_lock = threading.Lock()
_aborts: dict[tuple[str, str], int] = {}
_tokens: dict[tuple[str, str], int] = {}
_recent: list[dict[str, object]] = []


def record_abort(reason: str, role: str, tokens_generated: int) -> None:
    key = (reason or "unknown", role or "unknown")
    n = max(0, int(tokens_generated or 0))
    with _lock:
        _aborts[key] = _aborts.get(key, 0) + 1
        _tokens[key] = _tokens.get(key, 0) + n
        _recent.append({"reason": key[0], "role": key[1], "tokens": n, "ts": time.time()})
        del _recent[:-_RECENT_LIMIT]


def abort_total(reason: str | None = None, role: str | None = None) -> int:
    with _lock:
        return sum(
            v for (r, ro), v in _aborts.items()
            if (reason is None or r == reason) and (role is None or ro == role)
        )


def snapshot() -> dict[str, object]:
    with _lock:
        return {
            ABORTS_TOTAL: {f"{r}/{ro}": v for (r, ro), v in _aborts.items()},
            ABORT_TOKENS_TOTAL: {f"{r}/{ro}": v for (r, ro), v in _tokens.items()},
            "recent": [dict(e) for e in _recent],
        }


def render_prometheus() -> str:
    with _lock:
        lines = [
            f"# HELP {ABORTS_TOTAL} Streams aborted as degenerate generation, by reason and role.",
            f"# TYPE {ABORTS_TOTAL} counter",
        ]
        for (r, ro), v in sorted(_aborts.items()):
            lines.append(f'{ABORTS_TOTAL}{{reason="{r}",role="{ro}"}} {v}')
        lines += [
            f"# HELP {ABORT_TOKENS_TOTAL} Tokens generated before degenerate-generation aborts.",
            f"# TYPE {ABORT_TOKENS_TOTAL} counter",
        ]
        for (r, ro), v in sorted(_tokens.items()):
            lines.append(f'{ABORT_TOKENS_TOTAL}{{reason="{r}",role="{ro}"}} {v}')
        return "\n".join(lines) + "\n"


def reset() -> None:
    """Test-only."""
    with _lock:
        _aborts.clear()
        _tokens.clear()
        _recent.clear()
