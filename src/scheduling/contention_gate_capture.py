"""SC19 (EVL-47) — producer-written capture of the `ChatResponse.contention_gate` echo.

The A14 residual (BRIDGE RESIDUAL 1) turns the contention verdict from an inferred proxy —
ROUTE-A1 reading a fail-closed 503 timeout as "queued" — into a directly measured field on
every chat response. This module is the WRITE side of that measurement for the vidya belief
kernel: it appends one request-keyed envelope per line to a durable JSONL capture file at the
point where the response is assembled, so an adapter can later project claim tuples from bytes
a producer actually wrote. Retrofitting the read side is impossible, which is why the hook
lands before the orchestrator next serves traffic.

Design rules, each load-bearing:

* **Feature-gated OFF by default.** The orchestrator is a production serving path; capture is
  opt-in via `ORCHESTRATOR_CONTENTION_GATE_CAPTURE`. A default-on hook would turn a
  measurement instrument into a permanent append load on every request.
* **One object per REQUEST, never per decision.** A request can pass the gate more than once
  — the `_dispatch` path records every candidate tried, not just the winner, so the probe can
  see the walk down the placement priority order. The envelope therefore carries the whole
  `gate_decisions` list under one `request_id`; a per-decision line would read one request as
  N witnesses. This is the locator trap the reader must not trip.
* **Never raises.** A capture failure must not break the serving path; the error is logged and
  the request proceeds. The write is a single-line append so a crash cannot tear an envelope
  in half.
* **No gate ran writes nothing.** A `None` payload (the echo is only set when the gate ran)
  is not a measurement, so it is not a row.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CAPTURE_SCHEMA = "contention_gate_capture.v1"
ENV_VAR = "ORCHESTRATOR_CONTENTION_GATE_CAPTURE"
# Off-tree bus runtime (shared clone of the runtime-plane writes). The alternative path
# (`/mnt/raid0/llm/contention_gate_capture.jsonl`) is used only if bus-runtime is absent.
DEFAULT_CAPTURE_PATH = "/mnt/raid0/llm/bus-runtime/contention_gate_capture.jsonl"


def capture(payload: dict[str, Any] | None, *, request_id: str) -> bool:
    """Append one request-keyed envelope to the capture file. Never raises.

    Returns ``True`` when a row was written, ``False`` when capture was skipped — feature
    off, no gate ran, or the write failed (logged, serving path unaffected).
    """
    if ENV_VAR not in os.environ or not payload:
        return False
    path = Path(os.environ[ENV_VAR] or DEFAULT_CAPTURE_PATH)
    decisions = list(payload.get("gate_decisions") or [])
    envelope = {
        "capture_schema": CAPTURE_SCHEMA,
        "request_id": request_id,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "gate_decisions": decisions,
        "decision_count": len(decisions),
        "admitted": payload.get("admitted"),
        "waited_s": payload.get("waited_s"),
        "candidate_topology_idx": payload.get("candidate_topology_idx"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode + a single write call keeps one envelope a single line, so a crash
        # cannot leave a half-object behind to corrupt the whole capture.
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(envelope, separators=(",", ":")) + "\n")
            handle.flush()
        return True
    except Exception:
        log.exception("contention-gate capture failed for request %s", request_id)
        return False
