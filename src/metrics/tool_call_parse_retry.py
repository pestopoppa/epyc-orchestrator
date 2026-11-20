"""In-process counter for tool-call retries (flags ``tool_call_parse_error_retry`` and
``tool_call_arg_leak_guard``; both share one corrective re-prompt and this telemetry).

Same pattern as ``stream_degeneration``: lock-guarded dict, snapshot, Prometheus text renderer.
Labels: role, trigger (``parse_error`` = the server failed to parse a tool call: error.type
tool_call_parse_error or native parser loss; ``arg_leak`` = parsed arguments carried tool-call
markup) and outcome (``retried_ok`` = the single corrective retry came back clean, ``failed`` =
the retry failed too and the call failed as a format error). Each event also keeps a truncated
detail (the server's raw_excerpt, or the leaking tool/key/marker) for diagnosis.
"""

from __future__ import annotations

import threading
import time

RETRIES_TOTAL = "tool_call_parse_retry_total"
TRIGGERS = ("parse_error", "arg_leak")
OUTCOMES = ("retried_ok", "failed")
_RECENT_LIMIT = 256
_DETAIL_LIMIT = 400

_lock = threading.Lock()
_counts: dict[tuple[str, str, str], int] = {}
_recent: list[dict[str, object]] = []


def record(role: str, trigger: str, outcome: str, detail: str = "") -> None:
    key = (role or "unknown", trigger, outcome)
    with _lock:
        _counts[key] = _counts.get(key, 0) + 1
        _recent.append({"role": key[0], "trigger": trigger, "outcome": outcome,
                        "detail": detail[:_DETAIL_LIMIT], "ts": time.time()})
        del _recent[:-_RECENT_LIMIT]


def total(role: str | None = None, trigger: str | None = None, outcome: str | None = None) -> int:
    with _lock:
        return sum(
            v for (ro, tr, oc), v in _counts.items()
            if (role is None or ro == role) and (trigger is None or tr == trigger)
            and (outcome is None or oc == outcome)
        )


def snapshot() -> dict[str, object]:
    with _lock:
        return {
            RETRIES_TOTAL: {f"{ro}/{tr}/{oc}": v for (ro, tr, oc), v in _counts.items()},
            "recent": [dict(e) for e in _recent],
        }


def render_prometheus() -> str:
    with _lock:
        lines = [
            f"# HELP {RETRIES_TOTAL} Tool-call retries (server parse error or argument markup leak), by role, trigger and outcome.",
            f"# TYPE {RETRIES_TOTAL} counter",
        ]
        for (ro, tr, oc), v in sorted(_counts.items()):
            lines.append(f'{RETRIES_TOTAL}{{role="{ro}",trigger="{tr}",outcome="{oc}"}} {v}')
        return "\n".join(lines) + "\n"


def reset() -> None:
    """Test-only."""
    with _lock:
        _counts.clear()
        _recent.clear()
