"""Chaos test: the dashboard API dies mid-stream and must come back consumable.

AutoPilot restarts the :8000 API server at every trial boundary (~20-25 min),
tearing down every SSE connection and in-flight fetch. This test SIGKILLs a
real uvicorn serving the dashboard router mid-SSE, relaunches it on the same
port, and asserts a client can recover — fresh snapshot, live SSE event,
healthy serve_path — within the 15s budget the frontend watchdog assumes.

Server-side only: the browser-side recovery (snapshotTransportWatchdog) is
asserted structurally in test_dashboard_route_html.py and verified manually
per the deploy runbook.
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

RECOVERY_BUDGET_S = 15.0
STARTUP_BUDGET_S = 20.0

_LAUNCHER = """
import sys
import time
from pathlib import Path
from fastapi import FastAPI
from starlette.responses import JSONResponse
from src.api.routes import dashboard
import uvicorn

# Exercise real HTTP/SSE, snapshot wrapper and restart accounting against an
# owned producer. Never poll serving slots or tail/create the operator's logs.
private = Path(sys.argv[2])
for name in ("_INFERENCE_TAP_PATH", "AUTOPILOT_LOG", "_PLANNER_TAP_PATH", "_TAP_SENTINEL_PATH"):
    setattr(dashboard, name, private / name)
dashboard._PANELS = []
dashboard._latest_tap_events_mtime = lambda: None
dashboard._structured_tap_requests_for_dashboard = lambda **kwargs: []
dashboard.live_telemetry_sequence = lambda: ()
async def fixture_snapshot():
    return JSONResponse({"generated_at": time.time(),
                         "slots_poll_meta": {"fixture": "owned-offline"}})
dashboard._snapshot_impl = fixture_snapshot
app = FastAPI()
app.include_router(dashboard.router)
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(port: int, repo_root: Path, private: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _LAUNCHER, str(port), str(private)],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_healthy(base: str, deadline_s: float) -> dict:
    last_err: Exception | None = None
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/dashboard/api/health", timeout=3.0)
            if r.status_code == 200:
                return r.json()
        except Exception as exc:  # noqa: BLE001 — startup race is the point
            last_err = exc
        time.sleep(0.25)
    raise AssertionError(f"server never became healthy: {last_err}")


def _read_one_sse_event(base: str, timeout_s: float = 10.0) -> str:
    """Open the multiplex stream and return the first named event's data line."""
    with httpx.stream(
        "GET", f"{base}/dashboard/events/multiplex", timeout=timeout_s
    ) as resp:
        assert resp.status_code == 200
        current_event = ""
        for line in resp.iter_lines():
            if line.startswith("event: "):
                current_event = line[len("event: "):]
            elif line.startswith("data: ") and current_event:
                return current_event
    raise AssertionError("stream closed without a named event")


@pytest.mark.integration
def test_dashboard_survives_sigkill_restart(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    port = _free_port()
    base = f"http://127.0.0.1:{port}"

    proc = _launch(port, repo_root, tmp_path)
    try:
        _wait_healthy(base, STARTUP_BUDGET_S)

        pre = httpx.get(f"{base}/dashboard/api/snapshot", timeout=10.0).json()
        pre_ts = pre["generated_at"]
        assert _read_one_sse_event(base)  # stream delivers before the kill

        # The per-trial restart, worst case: no graceful shutdown at all.
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)

        proc = _launch(port, repo_root, tmp_path)
        recovered_at = time.time()
        health = _wait_healthy(base, RECOVERY_BUDGET_S)

        post = httpx.get(f"{base}/dashboard/api/snapshot", timeout=10.0).json()
        assert post["generated_at"] > pre_ts, "snapshot must be rebuilt fresh, not replayed"
        assert post["slots_poll_meta"] == {"fixture": "owned-offline"}

        event_name = _read_one_sse_event(base)
        assert event_name, "a reconnected SSE client must receive events again"

        health = httpx.get(f"{base}/dashboard/api/health", timeout=10.0).json()
        serve = health["serve_path"]
        assert serve["staleness_class"] == "fresh"
        assert serve["build_count"] >= 1
        assert time.time() - recovered_at < RECOVERY_BUDGET_S
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.integration
def test_dashboard_health_probe_runs_real_build(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    port = _free_port()
    base = f"http://127.0.0.1:{port}"

    proc = _launch(port, repo_root, tmp_path)
    try:
        _wait_healthy(base, STARTUP_BUDGET_S)
        r = httpx.get(f"{base}/dashboard/api/health?probe=snapshot", timeout=15.0)
        payload = r.json()
        assert payload["probe"]["ok"] is True
        assert payload["probe"]["duration_s"] < 10.0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
