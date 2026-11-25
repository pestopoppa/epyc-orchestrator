"""Push alerts for AutoPilot failures (flag ``autopilot_alert_push``, default off).

AutoPilot surfaces failures only through pull-only state (consecutive_failures, phase health
fields, the journal). When the flag is on and ``alerts.target_session`` is set, a degraded
health transition, a failed smoke or a consecutive_failures threshold crossing pushes ONE line
to that session: a bus outbox row from roster id ``autopilot`` (``session_bus.py append``, by
literal path) plus a flag file touched on success AND failure.

* Dedupe: a kind+key is pushed once while it stays active (until cleared).
* Rate limit: at most one push per MIN_INTERVAL_S across all keys.
* Escalation: still active after ESCALATE_S -> re-pushed once, marked ``escalated``.
* Auto-ack: ``clear()`` drops the dedupe/escalation state when the condition clears.

Every fault is swallowed: an alert must never break the loop. Flag off or empty target is a
strict no-op.
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
SENDER = os.environ.get("AUTOPILOT_ALERT_AGENT", "autopilot")  # roster id (config.yaml)
ESCALATE_S = 900.0
MIN_INTERVAL_S = 60.0
CONSECUTIVE_FAILURE_THRESHOLD = 3
LINE_MAX = 400

_active: dict[str, dict] = {}  # "kind:key" -> {"first": t, "escalated": bool, "kind", "key", "detail"}
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


def build_row(target: str, line: str) -> dict:
    """The outbox message row (id/ts/from/schema_version are filled by session_bus append)."""
    return {"to": target, "kind": "nudge", "payload": {"alert": line}}


def bus_append_cmd(row: dict, bus_root: str | None = None) -> list[str]:
    cmd = [sys.executable, SESSION_BUS]
    if bus_root:
        cmd += ["--bus-root", bus_root]
    return cmd + ["append", "--agent", SENDER, "--target", "outbox", "--json", json.dumps(row)]


def _default_transport(target: str, line: str) -> None:
    ok = False
    try:
        r = subprocess.run(bus_append_cmd(build_row(target, line), os.environ.get("AUTOPILOT_ALERT_BUS_ROOT")),
                           capture_output=True, text=True, timeout=30)
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


def _send(target: str, kind: str, key: str, detail: str, transport, t: float, tag: str = "") -> None:
    global _last_push
    _last_push = t
    line = " ".join(f"autopilot {tag}{kind} key={key} {detail}".split())[:LINE_MAX]
    (transport or _default_transport)(target, line)


def emit_alert(kind: str, key: str, detail: str = "", *, transport: Callable[[str, str], None] | None = None,
               now: float | None = None) -> bool:
    """Push one alert line. Returns True iff a push was sent."""
    try:
        target = target_session()
        if not target or not _enabled():
            return False
        t = time.time() if now is None else now
        dk = f"{kind}:{key}"
        if dk in _active or t - _last_push < MIN_INTERVAL_S:
            return False
        _active[dk] = {"first": t, "escalated": False, "kind": kind, "key": key, "detail": detail}
        _send(target, kind, key, detail, transport, t)
        return True
    except Exception:  # noqa: BLE001
        log.debug("alert push failed", exc_info=True)
        return False


def check_escalations(*, transport: Callable[[str, str], None] | None = None, now: float | None = None) -> int:
    """Re-push, once, every alert still active after ESCALATE_S. Returns the number pushed."""
    n = 0
    try:
        target = target_session()
        if not target or not _enabled():
            return 0
        t = time.time() if now is None else now
        for st in _active.values():
            if st["escalated"] or t - st["first"] < ESCALATE_S or t - _last_push < MIN_INTERVAL_S:
                continue
            st["escalated"] = True
            _send(target, st["kind"], st["key"], st["detail"] + f" unacked_s={int(t - st['first'])}",
                  transport, t, tag="escalated ")
            n += 1
    except Exception:  # noqa: BLE001
        log.debug("alert escalation failed", exc_info=True)
    return n


def clear(kind: str, key: str) -> bool:
    """Auto-ack: the condition cleared; forget dedupe/escalation state so a recurrence alerts afresh."""
    return _active.pop(f"{kind}:{key}", None) is not None


def alert_degraded(reason: str, consecutive: int = 0, **kw) -> bool:
    return emit_alert("health-degraded", "health", f"cause={reason} consecutive={consecutive}", **kw)


def clear_degraded() -> bool:
    return clear("health-degraded", "health")


def alert_smoke_failed(name: str, cause: str = "", **kw) -> bool:
    return emit_alert("smoke-fail", name, f"cause={cause}", **kw)


def alert_consecutive_failures(n: int, threshold: int = CONSECUTIVE_FAILURE_THRESHOLD, **kw) -> bool:
    """Fires when n crosses the threshold (n >= threshold); n below it auto-acks and re-checks escalation."""
    if n < threshold:
        clear("consecutive-failures", "gate")
        return False
    sent = emit_alert("consecutive-failures", "gate", f"count={n} threshold={threshold}", **kw)
    if not sent:
        check_escalations(**kw)
    return sent


def reset() -> None:
    global _last_push
    _active.clear()
    _last_push = -1e18
