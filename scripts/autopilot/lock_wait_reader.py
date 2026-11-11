"""INF-45 P1-8: read-only reader for per-holder region-lock wait telemetry.

Every ``dispatch.v1`` record carries ``region_lock_wait_s_by_holder``: a
PROCESS-CUMULATIVE snapshot ``{holder_role: {"wait_s": total, "waits": n}}``
(see ``src.runtime.cpu_region_lock.region_lock_wait_s_by_holder``). Summing
records would massively over-count, so this reader:

* groups records by process identity (``provenance.host``, ``pid``, ``run_id``),
* takes the LATEST snapshot per process (per holder the max, since counters are
  monotone within a process - robust to out-of-order lines),
* sums those per-process finals across processes.

Result: total wait seconds/count per holder across all processes seen in the
log. It is observation-only: no routing knob, no writes, never raises.
Autopilot surfaces it in the optimization brief as ``region_lock_wait`` so it
can be scored.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any

SCHEMA_KEY = "region_lock_wait_s_by_holder"
DEFAULT_MAX_LINES = 200_000


def default_dispatch_log_path() -> Path | None:
    """The dispatch.v1 log path as the writer resolves it (None when disabled)."""
    try:
        from src.backends import dispatch_log

        return dispatch_log.log_path()
    except Exception:  # noqa: BLE001 - observation must never raise
        return None


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f and f >= 0 else 0.0


def read_region_lock_wait(
    path: Path | str | None = None, *, max_lines: int = DEFAULT_MAX_LINES
) -> dict[str, Any]:
    """Per-holder lock-wait totals from the tail of the dispatch log (read-only)."""
    out: dict[str, Any] = {
        "read_only": True,
        "source": None,
        "basis": "latest cumulative snapshot per process, summed across processes",
        "processes": 0,
        "records_seen": 0,
        "by_holder": {},
        "total_wait_s": 0.0,
        "total_waits": 0,
    }
    try:
        p = Path(path) if path is not None else default_dispatch_log_path()
        if p is None or not p.is_file():
            return out
        out["source"] = str(p)
        per_proc: dict[tuple[str, str, str], dict[str, dict[str, float]]] = {}
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            lines = deque(fh, maxlen=max(1, int(max_lines)))
        for line in lines:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            snap = rec.get(SCHEMA_KEY) if isinstance(rec, dict) else None
            if not isinstance(snap, dict):
                continue
            out["records_seen"] += 1
            prov = rec.get("provenance") or {}
            key = (str(prov.get("host")), str(prov.get("pid")), str(prov.get("run_id")))
            cur = per_proc.setdefault(key, {})
            for holder, v in snap.items():
                if not isinstance(v, dict):
                    continue
                c = cur.setdefault(str(holder), {"wait_s": 0.0, "waits": 0.0})
                c["wait_s"] = max(c["wait_s"], _num(v.get("wait_s")))
                c["waits"] = max(c["waits"], _num(v.get("waits")))
        by_holder: dict[str, dict[str, float]] = {}
        for cur in per_proc.values():
            for holder, c in cur.items():
                agg = by_holder.setdefault(holder, {"wait_s": 0.0, "waits": 0})
                agg["wait_s"] = round(agg["wait_s"] + c["wait_s"], 6)
                agg["waits"] += int(c["waits"])
        out["processes"] = len(per_proc)
        out["by_holder"] = dict(
            sorted(by_holder.items(), key=lambda kv: kv[1]["wait_s"], reverse=True)
        )
        out["total_wait_s"] = round(sum(v["wait_s"] for v in by_holder.values()), 6)
        out["total_waits"] = sum(v["waits"] for v in by_holder.values())
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
