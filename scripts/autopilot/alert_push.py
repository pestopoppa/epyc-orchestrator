"""Push alerts for AutoPilot failures (flag ``autopilot_alert_push``, default off).

AutoPilot surfaces failures only through pull-only state (consecutive_failures, phase health
fields, the journal). When the flag is on and ``alerts.target_session`` is set, a degraded
health transition, a failed smoke or a consecutive_failures threshold crossing pushes ONE line
to that session: a bus outbox row (``session_bus.py append``, by literal path) plus a flag file
touched on success AND failure. Dedupe per key, global rate limit. Every fault is swallowed:
an alert must never break the loop. Flag off, or empty target, is a strict no-op.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

SESSION_BUS = "/workspace/scripts/coordination/session_bus.py"
FLAG_DIR = Path(os.environ.get("AUTOPILOT_ALERT_FLAG_DIR", "/mnt/raid0/llm/tmp/autopilot-alerts"))
SENDER = os.environ.get("AUTOPILOT_ALERT_AGENT", "autopilot-backlog-session")
DEDUPE_WINDOW_S = 900.0
MIN_INTERVAL_S = 60.0
CONSECUTIVE_FAILURE_THRESHOLD = 3
LINE_MAX = 400

_sent: dict[str, float] = {}
_last_push = -1e18


def _enabled() -> bool:
    try:
        from src.features import features  # noqa: PLC0415
        return bool(features().autopilot_alert_push)
    except Exception:  # noqa: BLE001
        return False


def target_session() -> str:
    """Knob ``alerts.target_session`` (env ORCHESTRATOR_ALERTS_TARGET_SESSION); empty = no-op."""
    return (os.environ.get("ORCHESTRATOR_ALERTS_TARGET_SESSION") or "").strip()


def _default_transport(target: str, line: str) -> None:
    ok = False
    try:
        row = {"to": target, "kind": "nudge", "payload": {"alert": line}}
        r = subprocess.run([sys.executable, SESSION_BUS, "append", "--agent", SENDER, "--target", "outbox",
                            "--json", json.dumps(row)], capture_output=True, text=True, timeout=30)
        ok = r.returncode == 0
    except Exception:  # noqa: BLE001
        ok = False
    finally:
        try:
            FLAG_DIR.mkdir(parents=True, exist_ok=True)
            (FLAG_DIR / f"{target}.{'ok' if ok else 'failed'}").touch()
            with open(FLAG_DIR / f"{target}.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")
        except OSError:
            pass


def emit_alert(kind: str, key: str, detail: str = "", *, transport: Callable[[str, str], None] | None = None,
               now: float | None = None) -> bool:
    """Push one alert line. Returns True iff a push was sent."""
    global _last_push
    try:
        target = target_session()
        if not target or not _enabled():
            return False
        t = time.time() if now is None else now
        dk = f"{kind}:{key}"
        if t - _sent.get(dk, -1e18) < DEDUPE_WINDOW_S or t - _last_push < MIN_INTERVAL_S:
            return False
        _sent[dk] = t
        _last_push = t
        line = " ".join(f"autopilot {kind} key={key} {detail}".split())[:LINE_MAX]
        (transport or _default_transport)(target, line)
        return True
    except Exception:  # noqa: BLE001
        log.debug("alert push failed", exc_info=True)
        return False


def alert_degraded(reason: str, consecutive: int = 0, **kw) -> bool:
    return emit_alert("health-degraded", "health", f"cause={reason} consecutive={consecutive}", **kw)


def alert_smoke_failed(name: str, cause: str = "", **kw) -> bool:
    return emit_alert("smoke-fail", name, f"cause={cause}", **kw)


def alert_consecutive_failures(n: int, threshold: int = CONSECUTIVE_FAILURE_THRESHOLD, **kw) -> bool:
    """Fires when n crosses the threshold (n >= threshold)."""
    if n < threshold:
        return False
    return emit_alert("consecutive-failures", "gate", f"count={n} threshold={threshold}", **kw)


def reset() -> None:
    global _last_push
    _sent.clear()
    _last_push = -1e18
