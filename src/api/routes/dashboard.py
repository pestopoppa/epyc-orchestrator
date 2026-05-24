"""Interactive orchestrator dashboard — server topology, active tasks, live streams.

Routes:
    GET  /dashboard                         — single-page HTML UI
    GET  /dashboard/api/topology            — static topology (roles, ports, services)
    GET  /dashboard/api/snapshot            — current state of all slots + counters
    GET  /dashboard/events/stream           — 1Hz SSE: snapshot + recent decisions
    GET  /dashboard/api/task/{task_id}      — full task detail (prompt + REPL history)
    GET  /dashboard/events/task/{task_id}   — 5Hz SSE: live token stream for one task

Read-only observer. Polls existing llama-server /slots endpoints and tails the
progress JSONL log; never modifies routing or inference state.

SSH access: from your laptop, `ssh -L 8000:localhost:8000 daniele@<host>` then
open http://localhost:8000/dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from datetime import datetime, date
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.api.routes.dashboard_snapshot import (
    INFLIGHT_MAX_AGE_DEFAULT_S,
    count_log_events as _count_log_events_impl,
    scan_orchestrator_tasks as _scan_orchestrator_tasks_impl,
    scan_recent_decisions as _scan_recent_decisions_impl,
    todays_progress_log as _todays_progress_log_impl,
)
from src.api.routes.dashboard_tap import (
    _INFERENCE_TAP_PATH,
    _PROMPT_TAP_PATH,
    _REPL_TAP_PATH,
    _SECTION_SEP,
    _SUBSECTION_SEP,
    _TAP_SENTINEL_PATH,
    _parse_inference_sections,
    _parse_trial_state,
    _read_tail,
)
from src.api.routes.dashboard_tasks import (
    _find_section_by_objective,
    _objective_for_task,
    _task_events,
    _task_text_snapshot,
)
from src.api.routes.dashboard_topology import (
    _PORT_HINTS,
    role_aliases,
    _ROLE_COLORS,
    _discover_llama_ports,
    _process_info_by_match,
    _role_color,
    base_role,
)
from src.api.routes.dashboard_topology import _load_state_services as _load_state_services_impl

logger = logging.getLogger(__name__)
router = APIRouter()

ORCHESTRATOR_LOG_DIR = Path("/mnt/raid0/llm/epyc-orchestrator/logs")
PROGRESS_LOG_DIR = ORCHESTRATOR_LOG_DIR / "progress"
AUTOPILOT_LOG = ORCHESTRATOR_LOG_DIR / "autopilot.log"
ORCHESTRATOR_STATE_PATH = ORCHESTRATOR_LOG_DIR / "orchestrator_state.json"


def _load_state_services() -> list[dict[str, Any]]:
    """Wrapper supplying ORCHESTRATOR_STATE_PATH to dashboard_topology helper."""
    return _load_state_services_impl(ORCHESTRATOR_STATE_PATH)


# ---------------------------------------------------------------------------
# Inference taps — live prompt + response stream written by autopilot's
# seeding harness to /mnt/raid0/llm/tmp/*. These are the same files the
# autopilot --tui tails. Reading them is cheap (just stat + seek-to-end)
# and gives us the full token stream + REPL history for free.
# ---------------------------------------------------------------------------

@router.get("/dashboard/api/inference_tap")
async def inference_tap_snapshot(max_sections: int = 20) -> JSONResponse:
    """Return parsed inference sections + current prompt + REPL tail.

    Source: /mnt/raid0/llm/tmp/{inference_tap.log, autopilot_prompt_tap.txt,
    repl_tap.log} — the same files autopilot_tui.py tails.
    """
    tap_active = _TAP_SENTINEL_PATH.exists()
    inference_tail = _read_tail(_INFERENCE_TAP_PATH, max_bytes=512 * 1024)
    sections = _parse_inference_sections(inference_tail, max_sections=max_sections)
    current_prompt = ""
    if _PROMPT_TAP_PATH.exists():
        try:
            current_prompt = _PROMPT_TAP_PATH.read_text(errors="ignore")[-4000:]
        except Exception:
            pass
    repl_tail = _read_tail(_REPL_TAP_PATH, max_bytes=64 * 1024)
    # Just take the last ~3000 chars of REPL for compactness
    repl_tail = repl_tail[-3000:] if repl_tail else ""

    # File mtimes — surface staleness
    def mtime(p: Path) -> float | None:
        try:
            return p.stat().st_mtime
        except Exception:
            return None

    return JSONResponse({
        "tap_active": tap_active,
        "current_prompt": current_prompt,
        "current_prompt_mtime": mtime(_PROMPT_TAP_PATH),
        "inference_sections": sections,
        "inference_tap_mtime": mtime(_INFERENCE_TAP_PATH),
        "repl_tail": repl_tail,
        "repl_tap_mtime": mtime(_REPL_TAP_PATH),
        "now": time.time(),
    })


@router.get("/dashboard/api/contention")
async def contention_gate_snapshot(request: Request) -> JSONResponse:
    """Cross-role admission-gate metrics + per-role scheduling state.

    2026-05-24 cross-role-bw-aware-routing. Returns:
      - matrix_status: "ok"|"missing"|"stale"|"invalid"
      - active_decodes_by_role: {role: count} from region-lock holders
      - contention_blocked_count: {"roleA+roleB": int}
      - contention_wait_seconds: cumulative
      - contention_timeout_count, contention_admitted_count, etc.
      - per_role_scheduling: {role: {quarter_preference_order, migrations_started,
            migration_failures, kv_migration: {enabled, dispatch_path, ...}}}
          for every role backed by ConcurrencyAwareBackend
      - generated_at: time.time() for client cache-busting
    """
    import time as _time
    try:
        from src.scheduling.contention_gate import get_gate
        snap = get_gate().metrics_snapshot()
    except Exception as exc:  # noqa: BLE001
        snap = {"error": str(exc), "matrix_status": "unavailable"}

    # Per-role scheduling state (quarter preference + migration counts).
    # Fetched here because it lives on app.state.llm_primitives._backends,
    # which is per-request-injectable but not module-singleton.
    per_role: dict[str, Any] = {}
    try:
        primitives = getattr(request.app.state, "llm_primitives", None)
        if primitives is not None:
            for role, backend in getattr(primitives, "_backends", {}).items():
                if not hasattr(backend, "_quarter_preference_order"):
                    continue
                per_role[role] = {
                    "quarter_preference_order": list(getattr(backend, "_quarter_preference_order", [])),
                    "migrations_started": int(getattr(backend, "_migrations", 0)),
                    "migration_failures": int(getattr(backend, "_migration_failures", 0)),
                    "kv_migration": (
                        backend.kv_migration_status()
                        if hasattr(backend, "kv_migration_status")
                        else {}
                    ),
                }
    except Exception as exc:  # noqa: BLE001
        per_role = {"_error": str(exc)}
    snap["per_role_scheduling"] = per_role
    snap["generated_at"] = _time.time()
    return JSONResponse(snap)


@router.get("/dashboard/api/region_locks")
async def region_locks_snapshot() -> JSONResponse:
    """Per-CPU-region lock state — which (role, region) lock files are
    currently held, and by which PIDs.

    Built from /proc/locks scan of the orchestrator's lock files. Used
    by the dashboard to surface real-time concurrent dispatch — a glance
    at this endpoint shows whether the cross-process per-region lock
    layer (Phase 5 of 2026-05-22) is actually achieving multi-instance
    concurrency.

    Returns a list of {role, region, lock_path, holder_pids[]} entries
    for every region lock file that exists. Empty list when no lock
    files have been created yet (orchestrator was just started, no
    inference has flown through region-lock-aware path).
    """
    import os
    from pathlib import Path

    try:
        from src.runtime.cpu_region_lock import _tmp_dir, _current_lock_owner_pids
        tmp_dir = _tmp_dir()
    except Exception:
        tmp_dir = Path("/mnt/raid0/llm/tmp")
        from src.runtime.cpu_region_lock import _current_lock_owner_pids

    out: list[dict[str, Any]] = []
    try:
        for p in sorted(tmp_dir.glob("cpu_region.*.lock")):
            stem = p.stem  # "cpu_region.<role>.<region>"
            parts = stem.split(".", 2)
            if len(parts) < 3:
                continue
            _prefix, role, region = parts[0], parts[1], parts[2]
            holders = _current_lock_owner_pids(p)
            out.append({
                "role": role,
                "region": region,
                "lock_path": str(p),
                "holder_pids": holders,
                "held": bool(holders),
            })
    except Exception as exc:
        return JSONResponse({"error": str(exc), "entries": []}, status_code=200)

    # Group by role for easier dashboard rendering
    by_role: dict[str, list[dict[str, Any]]] = {}
    for entry in out:
        by_role.setdefault(entry["role"], []).append({
            "region": entry["region"],
            "held": entry["held"],
            "holder_pids": entry["holder_pids"],
        })

    feature_flag = os.environ.get("ORCHESTRATOR_PER_REGION_LOCKS", "0").strip()
    return JSONResponse({
        "per_region_locks_enabled": feature_flag in {"1", "true", "yes", "on"},
        "tmp_dir": str(tmp_dir),
        "entries": out,
        "by_role": by_role,
        "now": time.time(),
    })


@router.get("/dashboard/events/raw_tap")
async def raw_tap_stream(request: Request, tail_bytes: int = 8192) -> StreamingResponse:
    """True byte-level streaming SSE — mirrors what autopilot's TUI does.

    Tails inference_tap.log at 10Hz, pushing any new bytes since the last
    read directly to the client. The tap writer in eval_tower flushes
    per-token during generation, so this is genuine real-time streaming
    (same source the TUI uses).

    Anti-buffering headers ensure intermediate proxies and uvicorn itself
    don't batch small chunks — required for per-token feel in the browser.
    """

    async def event_gen():
        try:
            fh = open(_INFERENCE_TAP_PATH, "rb")
            fh.seek(0, 2)
            size = fh.tell()
            start = max(0, size - tail_bytes)
            fh.seek(start)
            initial = fh.read().decode("utf-8", errors="replace")
            yield "data: " + json.dumps({"chunk": initial, "initial": True}) + "\n\n"
            heartbeat_counter = 0
            while True:
                if await request.is_disconnected():
                    fh.close()
                    return
                pos = fh.tell()
                fh.seek(pos)
                raw = fh.read(8192)
                if raw:
                    chunk = raw.decode("utf-8", errors="replace")
                    yield "data: " + json.dumps({"chunk": chunk, "initial": False}) + "\n\n"
                    heartbeat_counter = 0
                else:
                    heartbeat_counter += 1
                    # Send an SSE comment every ~3s to keep the connection
                    # warm + flush any TCP buffer (Nagle / browser-side).
                    if heartbeat_counter >= 30:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                    await asyncio.sleep(0.1)
        except Exception as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
            try:
                fh.close()  # type: ignore[has-type]
            except Exception:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx / reverse-proxy buffering
            "Connection": "keep-alive",
        },
    )


@router.get("/dashboard/events/autopilot_log")
async def autopilot_log_stream(request: Request, tail_bytes: int = 16384) -> StreamingResponse:
    """Byte-level SSE tail of autopilot.log — surfaces every phase autopilot
    is in (eval / seed_batch / meta-planning via `claude -p` / rollback /
    safety violation / etc.).

    The inference_tap.log only sees /chat traffic. seed_specialist_routing,
    meta-planning, eval scoring, and rollback decisions all log to
    autopilot.log without ever hitting /chat. This panel is the single
    "what is autopilot doing right now" view.

    Initial payload: last `tail_bytes` of the log. Subsequent: incremental
    appends, pushed within ~100ms of being written.
    """
    log_path = AUTOPILOT_LOG

    async def event_gen():
        try:
            if not log_path.exists():
                yield "data: " + json.dumps({"error": f"{log_path} does not exist"}) + "\n\n"
                return
            fh = open(log_path, "rb")
            fh.seek(0, 2)
            size = fh.tell()
            start = max(0, size - tail_bytes)
            fh.seek(start)
            initial = fh.read().decode("utf-8", errors="replace")
            yield "data: " + json.dumps({"chunk": initial, "initial": True}) + "\n\n"
            heartbeat_counter = 0
            while True:
                if await request.is_disconnected():
                    fh.close()
                    return
                pos = fh.tell()
                fh.seek(pos)
                raw = fh.read(8192)
                if raw:
                    chunk = raw.decode("utf-8", errors="replace")
                    yield "data: " + json.dumps({"chunk": chunk, "initial": False}) + "\n\n"
                    heartbeat_counter = 0
                else:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 30:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                    await asyncio.sleep(0.1)
        except Exception as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
            try:
                fh.close()  # type: ignore[has-type]
            except Exception:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/dashboard/events/planner_tap")
async def planner_tap_stream(request: Request, tail_bytes: int = 16384) -> StreamingResponse:
    """Byte-level SSE tail of the planner tap file written by
    controller_io.invoke_controller. Streams each `claude -p` event
    (system init / assistant thinking / tool_use / tool_result / result)
    summarized as it's emitted — no more 60-120s black-box waits during
    meta-planning.

    The tap is APPENDED across planner sessions with section separators,
    so the panel shows recent planner history at a glance.
    """
    planner_tap_path = Path("/mnt/raid0/llm/tmp/planner_tap.log")

    async def event_gen():
        try:
            if not planner_tap_path.exists():
                # Create empty file so the SSE doesn't error before the
                # first planner session writes anything.
                planner_tap_path.parent.mkdir(parents=True, exist_ok=True)
                planner_tap_path.touch()
            fh = open(planner_tap_path, "rb")
            fh.seek(0, 2)
            size = fh.tell()
            start = max(0, size - tail_bytes)
            fh.seek(start)
            initial = fh.read().decode("utf-8", errors="replace")
            yield "data: " + json.dumps({"chunk": initial, "initial": True}) + "\n\n"
            heartbeat_counter = 0
            while True:
                if await request.is_disconnected():
                    fh.close()
                    return
                pos = fh.tell()
                fh.seek(pos)
                raw = fh.read(8192)
                if raw:
                    chunk = raw.decode("utf-8", errors="replace")
                    yield "data: " + json.dumps({"chunk": chunk, "initial": False}) + "\n\n"
                    heartbeat_counter = 0
                else:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 30:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                    await asyncio.sleep(0.1)
        except Exception as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
            try:
                fh.close()  # type: ignore[has-type]
            except Exception:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/dashboard/events/inference_tap")
async def inference_tap_stream(request: Request) -> StreamingResponse:
    """SSE stream — emit a new payload whenever any tap file mtime advances.

    Polls mtimes at 2Hz; pushes a full snapshot when any tap file has changed
    since the last push.
    """

    async def event_gen():
        last_mtimes = {"inference": 0.0, "prompt": 0.0, "repl": 0.0}
        while True:
            if await request.is_disconnected():
                return
            try:
                inf_m = _INFERENCE_TAP_PATH.stat().st_mtime if _INFERENCE_TAP_PATH.exists() else 0.0
                prm_m = _PROMPT_TAP_PATH.stat().st_mtime if _PROMPT_TAP_PATH.exists() else 0.0
                rpl_m = _REPL_TAP_PATH.stat().st_mtime if _REPL_TAP_PATH.exists() else 0.0
            except Exception:
                inf_m = prm_m = rpl_m = 0.0
            changed = (
                inf_m > last_mtimes["inference"]
                or prm_m > last_mtimes["prompt"]
                or rpl_m > last_mtimes["repl"]
            )
            if changed:
                last_mtimes = {"inference": inf_m, "prompt": prm_m, "repl": rpl_m}
                # Build the payload — same shape as the snapshot endpoint
                inference_tail = _read_tail(_INFERENCE_TAP_PATH, max_bytes=256 * 1024)
                sections = _parse_inference_sections(inference_tail, max_sections=10)
                current_prompt = ""
                if _PROMPT_TAP_PATH.exists():
                    try:
                        current_prompt = _PROMPT_TAP_PATH.read_text(errors="ignore")[-4000:]
                    except Exception:
                        pass
                payload = json.dumps({
                    "tap_active": _TAP_SENTINEL_PATH.exists(),
                    "current_prompt": current_prompt,
                    "inference_sections": sections,
                    "inference_tap_mtime": inf_m,
                    "prompt_tap_mtime": prm_m,
                    "repl_tap_mtime": rpl_m,
                })
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Process status — autopilot alive/dead + orchestrator uptime
# ---------------------------------------------------------------------------

@router.get("/dashboard/api/process_status")
async def process_status() -> JSONResponse:
    """Aliveness check for autopilot + count of GEPA worker subprocesses."""
    autopilot = _process_info_by_match("autopilot.py start")
    # Count spawn_main children (GEPA workers + reembed workers)
    try:
        import subprocess
        out = subprocess.run(
            ["pgrep", "-cf", "spawn_main"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        n_workers = int(out) if out else 0
    except Exception:
        n_workers = 0
    # Recent autopilot log tail (last 3 substantive lines)
    recent_lines: list[str] = []
    last_log_age_s: float | None = None
    if AUTOPILOT_LOG.exists():
        try:
            mtime = AUTOPILOT_LOG.stat().st_mtime
            last_log_age_s = time.time() - mtime
            size = AUTOPILOT_LOG.stat().st_size
            with open(AUTOPILOT_LOG, "rb") as f:
                if size > 16 * 1024:
                    f.seek(-16 * 1024, 2)
                tail = f.read().decode("utf-8", errors="ignore")
            recent_lines = [l for l in tail.splitlines() if l.strip()][-5:]
        except Exception:
            pass
    # Stream-source mtimes — drive the live-dot activity-state badges in the
    # left-column panels. The browser computes its own age from the timestamp
    # plus a freshness threshold; we just report mtimes (None when missing).
    def _age_s(p: Path) -> float | None:
        try:
            return time.time() - p.stat().st_mtime
        except Exception:
            return None

    return JSONResponse({
        "autopilot": autopilot,
        "gepa_worker_count": n_workers,
        "last_autopilot_log_age_s": last_log_age_s,
        "autopilot_recent_lines": recent_lines,
        "inference_tap_age_s": _age_s(_INFERENCE_TAP_PATH),
        "planner_tap_age_s": _age_s(Path("/mnt/raid0/llm/tmp/planner_tap.log")),
    })


# ---------------------------------------------------------------------------
# Per-node detail (for topology click)
# ---------------------------------------------------------------------------

@router.get("/dashboard/api/node/{port}")
async def node_detail(port: int) -> JSONResponse:
    """Full detail for one topology node: health, slots, recent decisions routed to it."""
    label = _PORT_HINTS.get(port, f"port_{port}")
    # PID + cmd from ps
    proc_info: dict[str, Any] = {}
    try:
        import subprocess
        out = subprocess.run(
            ["ps", "-eo", "pid,etime,pcpu,pmem,rss,cmd"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        port_arg = f"--port {port}"
        for line in out.splitlines()[1:]:
            if port_arg in line and "grep" not in line:
                parts = line.split(None, 5)
                if len(parts) >= 6:
                    proc_info = {
                        "pid": int(parts[0]),
                        "etime": parts[1],
                        "pcpu_cumulative": float(parts[2]),
                        "pmem": float(parts[3]),
                        "rss_kb": int(parts[4]),
                        "cmd": parts[5][:400],
                    }
                    break
    except Exception:
        pass

    # Live /slots state
    slots_data: list[dict[str, Any]] = []
    health_status: str | None = None
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            try:
                resp = await client.get(f"http://127.0.0.1:{port}/slots")
                if resp.status_code == 200:
                    raw = resp.json()
                    if isinstance(raw, list):
                        for s in raw:
                            prompt = s.get("prompt", "") or ""
                            if isinstance(prompt, list):
                                prompt = " ".join(str(x) for x in prompt)
                            content = s.get("content", "") or ""
                            if isinstance(content, list):
                                content = " ".join(str(x) for x in content)
                            slots_data.append({
                                "id": s.get("id"),
                                "id_task": s.get("id_task"),
                                "is_processing": s.get("is_processing"),
                                "n_decoded": s.get("n_decoded"),
                                "n_prompt_tokens": s.get("n_prompt_tokens"),
                                "prompt_preview": prompt[:180],
                                "content_preview": content[:200],
                                "content_len": len(content),
                            })
                    health_status = "ok"
                else:
                    health_status = f"http {resp.status_code}"
            except httpx.ConnectError:
                health_status = "connection_refused"
            except httpx.TimeoutException:
                health_status = "timeout"
            except Exception as exc:
                health_status = f"error: {type(exc).__name__}"
    except Exception:
        pass

    # Recent decisions routed to this node's role (from progress JSONL)
    role_base = label.split(".")[0]
    log_path = _todays_progress_log()
    recent_routed: list[dict[str, Any]] = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("event_type") != "routing_decision":
                        continue
                    data = e.get("data", {})
                    if data.get("chosen_action") == role_base:
                        recent_routed.append({
                            "task_id": e.get("task_id"),
                            "timestamp": e.get("timestamp", ""),
                            "decision_source": data.get("decision_source"),
                            "classifier_confidence": data.get("classifier_confidence"),
                            "verifier_p_success": data.get("verifier_p_success"),
                        })
            recent_routed = recent_routed[-15:]
        except Exception:
            pass

    return JSONResponse({
        "port": port,
        "label": label,
        "role": role_base,
        "color": _role_color(label),
        "process": proc_info,
        "health_status": health_status,
        "slots": slots_data,
        "n_slots": len(slots_data),
        "n_processing": sum(1 for s in slots_data if s.get("is_processing")),
        "recent_routed_count": len(recent_routed),
        "recent_routed": recent_routed,
    })


# ---------------------------------------------------------------------------
# Topology endpoint
# ---------------------------------------------------------------------------


# Dashboard source paths used by the /version endpoint below to surface
# build state to the browser, so users can tell when a hard-reload is needed.
_DASHBOARD_HTML_FOR_VERSION = Path(__file__).parent / "dashboard.html"
_DASHBOARD_PY_FOR_VERSION = Path(__file__)
_REPO_ROOT_FOR_VERSION = Path(__file__).resolve().parents[3]

# Fleet startup timestamp — read from the marker file written by
# scripts/server/orchestrator_stack.start_orchestrator() before uvicorn launches.
# Live measurement (2026-05-23) confirmed uvicorn forks workers BEFORE importing
# the app, so each worker imports dashboard.py independently. The per-worker
# `time.time()` fallback would produce different timestamps across workers
# (observed: ~15ms drift among the 6 workers), which would cause the autopilot
# watcher to detect spurious restarts when consecutive requests hit different
# workers. Reading the atomic-write marker file gives every worker the same
# value as long as the marker exists before uvicorn's Popen — which the launch
# script guarantees.
try:
    from scripts.server.fleet_markers import read_orchestrator_marker as _read_orch_marker
    _marker_val = _read_orch_marker()
    _SERVER_STARTED_AT = _marker_val if _marker_val is not None else time.time()
except Exception:
    _SERVER_STARTED_AT = time.time()


def _read_git_short_sha() -> str | None:
    """Read current HEAD short SHA from .git/ — no subprocess.

    Handles both branch checkouts (.git/HEAD = "ref: refs/heads/main\n") and
    detached HEAD (.git/HEAD = "<sha>\n"). Returns None on any error.
    """
    try:
        head_file = _REPO_ROOT_FOR_VERSION / ".git" / "HEAD"
        head = head_file.read_text().strip()
        if head.startswith("ref: "):
            ref_path = _REPO_ROOT_FOR_VERSION / ".git" / head.split(" ", 1)[1].strip()
            if ref_path.exists():
                return ref_path.read_text().strip()[:7]
            # Fall back to packed-refs
            packed = _REPO_ROOT_FOR_VERSION / ".git" / "packed-refs"
            if packed.exists():
                target_ref = head.split(" ", 1)[1].strip()
                for line in packed.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2 and parts[1] == target_ref:
                        return parts[0][:7]
            return None
        # Detached HEAD: file directly contains the SHA
        return head[:7] if len(head) >= 7 else None
    except Exception:
        return None


_AUTOPILOT_STATE_PATH = Path(__file__).resolve().parents[3] / "orchestration" / "autopilot_state.json"
_AUTOPILOT_JOURNAL_PATH = Path(__file__).resolve().parents[3] / "orchestration" / "autopilot_journal.jsonl"


@router.get("/dashboard/api/autopilot_progress")
async def autopilot_progress() -> JSONResponse:
    """Live progress estimate for the autopilot's currently-running trial.

    Reads `in_flight_trial` from autopilot_state.json (added by AP-39 exogenous-
    restart-resilience work) and estimates completion percentage as
    `elapsed_s / median(recent_trial_durations_s)`, clamped to [0, 1].

    Returns:
        in_flight: bool — is there a trial currently running?
        trial_id, action_type, started_at, elapsed_s
        expected_s: median of last 10 trustworthy trials' durations
        percent: 0..100 (clamped; > 100 caps at 99 to signal overrun)
        recent_avg_duration_s, recent_p50, recent_p90 — duration distribution stats
        autopilot_alive: bool — is the autopilot process even running?
    """
    out: dict[str, Any] = {
        "in_flight": False,
        "autopilot_alive": False,
        "trial_id": None,
        "action_type": None,
        "started_at": None,
        "elapsed_s": None,
        "expected_s": None,
        "percent": None,
        "recent_avg_duration_s": None,
        "recent_p50": None,
        "recent_p90": None,
    }
    # Is autopilot alive? Quick pgrep-equivalent
    try:
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                cmd = (p / "cmdline").read_bytes().decode(errors="replace")
            except Exception:
                continue
            if "autopilot.py" in cmd and "start" in cmd:
                out["autopilot_alive"] = True
                break
    except Exception:
        pass

    # Read state.json for in_flight_trial
    if _AUTOPILOT_STATE_PATH.exists():
        try:
            state = json.loads(_AUTOPILOT_STATE_PATH.read_text())
            in_flight = state.get("in_flight_trial") or {}
            if in_flight and in_flight.get("trial_id") is not None:
                started_at = float(in_flight.get("started_at", 0)) or None
                elapsed = (time.time() - started_at) if started_at else None
                out.update({
                    "in_flight": True,
                    "trial_id": in_flight.get("trial_id"),
                    "action_type": (in_flight.get("action") or {}).get("type"),
                    "started_at": started_at,
                    "elapsed_s": elapsed,
                })
        except Exception:
            pass

    # Compute expected duration from journal — median of last 10 trustworthy trials
    if _AUTOPILOT_JOURNAL_PATH.exists():
        try:
            durations: list[float] = []
            with open(_AUTOPILOT_JOURNAL_PATH) as f:
                # Tail-read: parse last ~50 lines and pull duration if present.
                # The journal is append-only; reading line-by-line is fine at typical sizes.
                lines = f.readlines()[-50:]
            for raw in lines:
                try:
                    e = json.loads(raw)
                except Exception:
                    continue
                # Skip bug-corrupted trials (their durations are also suspect)
                if e.get("bug_corrupted_by"):
                    continue
                ed = e.get("eval_details") or {}
                # Try several known duration fields
                dur = ed.get("trial_duration_s") or ed.get("wall_time_s") or e.get("duration_s")
                if dur and dur > 0:
                    durations.append(float(dur))
            if durations:
                durations.sort()
                n = len(durations)
                med = durations[n // 2]
                p90 = durations[min(n - 1, int(n * 0.9))]
                avg = sum(durations) / n
                out["recent_avg_duration_s"] = round(avg, 1)
                out["recent_p50"] = round(med, 1)
                out["recent_p90"] = round(p90, 1)
                out["expected_s"] = round(med, 1)
        except Exception:
            pass

    # Fallback expected_s if journal had no durations: 1200s (~20 min — typical recent cycle)
    if out["expected_s"] is None and out["in_flight"]:
        out["expected_s"] = 1200.0

    # Percent
    if out["in_flight"] and out["elapsed_s"] is not None and out["expected_s"]:
        pct = (out["elapsed_s"] / out["expected_s"]) * 100.0
        # Cap at 99 to signal "overrun" rather than "done"; the actual completion
        # event is the next state.json write.
        out["percent"] = round(min(99.0, max(0.0, pct)), 1)

    return JSONResponse(out)


@router.get("/dashboard/api/pareto")
async def pareto(max_dominated: int = 600) -> JSONResponse:
    """Return the autopilot's Pareto archive for visualization.

    Reads orchestration/autopilot_state.json on each call (file is small,
    a few hundred KB; mtime-based caching unnecessary at this scale).
    Returns enough data to draw a scatter of (quality, speed) with the
    frontier highlighted + the hypervolume timeline.

    Objectives in the archive are (quality, speed, -cost, reliability).
    The dashboard plots the first two by default since they're the most
    operationally meaningful; -cost and reliability ride along as
    per-point fields the client can surface in tooltips.
    """
    if not _AUTOPILOT_STATE_PATH.exists():
        return JSONResponse({
            "available": False,
            "reason": "autopilot_state.json not found",
            "frontier": [],
            "dominated": [],
            "hypervolume_history": [],
        })
    try:
        data = json.loads(_AUTOPILOT_STATE_PATH.read_text())
    except Exception as exc:
        return JSONResponse({
            "available": False,
            "reason": f"failed to parse autopilot_state.json: {exc}",
            "frontier": [],
            "dominated": [],
            "hypervolume_history": [],
        })

    archive = data.get("pareto_archive", {}) or {}
    frontier_raw = archive.get("frontier", []) or []
    all_raw = archive.get("all_entries", []) or []
    hv_history = archive.get("hypervolume_history", []) or []

    def _shape(entry: dict) -> dict:
        # Strip heavy config_snapshot for transport — plotted points only need
        # objectives + identity metadata. Caller can drill via trial_id if needed.
        obj = entry.get("objectives") or [0.0, 0.0, 0.0, 0.0]
        if len(obj) < 4:
            obj = list(obj) + [0.0] * (4 - len(obj))
        return {
            "trial_id": entry.get("trial_id"),
            "objectives": list(obj[:4]),
            "git_tag": entry.get("git_tag", ""),
            "species": entry.get("species", ""),
            "is_production_best": bool(entry.get("is_production_best", False)),
            "timestamp": entry.get("timestamp", ""),
            "reasoning": (entry.get("reasoning") or "")[:200],
        }

    frontier = [_shape(e) for e in frontier_raw]
    frontier_ids = {f["trial_id"] for f in frontier if f["trial_id"] is not None}

    # Dominated entries: newest first, trimmed to max_dominated to bound payload.
    dominated_only = [e for e in all_raw if e.get("trial_id") not in frontier_ids]
    dominated_only.sort(key=lambda e: (e.get("trial_id") or 0), reverse=True)
    dominated_shaped = [_shape(e) for e in dominated_only[:max_dominated]]

    hv_shaped: list[list[float]] = []
    for h in hv_history:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            try:
                hv_shaped.append([int(h[0]), float(h[1])])
            except (TypeError, ValueError):
                continue

    return JSONResponse({
        "available": True,
        "frontier": frontier,
        "dominated": dominated_shaped,
        "hypervolume_history": hv_shaped,
        "totals": {
            "frontier_size": len(frontier),
            "all_entries": len(all_raw),
            "hv_points": len(hv_shaped),
        },
        "objective_axes": [
            {"key": "quality", "index": 0, "direction": "max", "label": "quality"},
            {"key": "speed", "index": 1, "direction": "max", "label": "speed (t/s)"},
            {"key": "neg_cost", "index": 2, "direction": "max", "label": "-cost"},
            {"key": "reliability", "index": 3, "direction": "max", "label": "reliability"},
        ],
    })


@router.get("/dashboard/api/version")
async def version() -> JSONResponse:
    """Return current build state for the hard-reload-needed indicator.

    The browser polls this every ~30s. If the returned `dashboard_html_mtime`
    or `git_sha` differs from what was captured at page load, the dashboard
    shows a "new build — hard-reload" badge.

    Returns:
        git_sha: short SHA of the orchestrator repo HEAD (or None if read fails)
        dashboard_html_mtime: float epoch seconds; bumps on every save
        dashboard_py_mtime: float epoch seconds; bumps when route handlers
            change (would require an orchestrator API reload to take effect,
            but useful to surface in case the file changed without restart)
        server_started_at: float epoch seconds; bumps on orchestrator restart
    """
    def mtime(p: Path) -> float | None:
        try:
            return p.stat().st_mtime
        except Exception:
            return None

    return JSONResponse({
        "git_sha": _read_git_short_sha(),
        "dashboard_html_mtime": mtime(_DASHBOARD_HTML_FOR_VERSION),
        "dashboard_py_mtime": mtime(_DASHBOARD_PY_FOR_VERSION),
        "server_started_at": _SERVER_STARTED_AT,
    })


@router.get("/dashboard/api/llama_fleet_ids")
async def llama_fleet_ids() -> JSONResponse:
    """Aggregate per-port llama-server fleet-startup markers.

    Used by the autopilot's OrchestratorWatcher to:
      1. Resolve role→port lookups (each marker carries the canonical
         role list served by that llama-server process).
      2. Detect operator-initiated reloads (when a stored `started_at`
         differs from a fresh poll, that port restarted).
      3. Distinguish operator-initiated reloads (`source=stack_commands`)
         from external restarts (any other source value).

    See scripts/server/fleet_markers.py for the marker file format and
    handoffs/active/autopilot-exogenous-restart-resilience.md sections
    5.1 + 5.2 for the design.

    Returns:
        per_port: {port: {started_at: float, source: str, roles: list[str]}}
        now: float epoch seconds (for the client's freshness math)
    """
    try:
        from scripts.server.fleet_markers import discover_llama_markers
        per_port = discover_llama_markers()
    except Exception as exc:
        return JSONResponse({"error": str(exc), "per_port": {}, "now": time.time()})
    # JSON keys must be strings; convert int ports to strings for transport.
    return JSONResponse({
        "per_port": {str(p): m for p, m in per_port.items()},
        "now": time.time(),
    })


@router.get("/dashboard/api/topology_activity")
async def topology_activity(window_s: float = 600.0) -> JSONResponse:
    """Per-role recent activity stats for the topology strip.

    Aggregates from two cheap sources:
      - inference_tap.log sections (last ~80) — provides per-role recent
        request count, last activity timestamp, and TIMINGS (t/s).
      - recent_completed_tasks (last 10 min, from progress JSONL) —
        provides per-role per-task durations.

    Returns:
        {
          "<role>": {
            "n_recent": int,                  # tap sections matching role in window
            "n_completed": int,               # JSONL chat tasks matching role in window
            "last_activity_age_s": float,     # seconds since last tap section
            "avg_tps_recent": float | None,   # mean of TIMINGS t/s across recent sections
            "avg_duration_s": float | None,   # mean chat-task duration
          },
          ...
        }

    Cheap: tap parse is already what the live tap polling uses; we just
    aggregate it here. Cached header advised but not required at current
    request rates.
    """
    inference_tail = _read_tail(_INFERENCE_TAP_PATH, max_bytes=512 * 1024)
    sections = _parse_inference_sections(inference_tail, max_sections=80)
    now = time.time()

    # Per-role aggregation from tap sections. timestamp format is
    # "YYYY-MM-DD HH:MM:SS" in local time (writer uses datetime.now()).
    per_role: dict[str, dict[str, Any]] = {}
    for s in sections:
        role = base_role(s.get("role") or "")
        if not role:
            continue
        bucket = per_role.setdefault(role, {
            "n_recent": 0,
            "n_completed": 0,
            "last_activity_age_s": None,
            "_tps_samples": [],
            "_duration_samples": [],
        })
        ts = s.get("timestamp")
        if ts:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                age = max(0.0, now - dt.timestamp())
            except Exception:
                age = None
            if age is not None and age <= window_s:
                bucket["n_recent"] += 1
                if bucket["last_activity_age_s"] is None or age < bucket["last_activity_age_s"]:
                    bucket["last_activity_age_s"] = age
        # Parse the TIMINGS line for t/s — format "N tokens in Xs (prompt=..., gen=..., Y.Y t/s)"
        timings_str = (s.get("response") or "")
        m = re.search(r"([\d.]+)\s*t/s", timings_str)
        if m:
            try:
                bucket["_tps_samples"].append(float(m.group(1)))
            except ValueError:
                pass

    # Augment from recent_completed_tasks (gives chat-XXX-tracked durations).
    log_path = _todays_progress_log()
    completed_pairs, _rolling, _cum = _scan_recent_decisions(log_path)
    _, recent_completed = _scan_orchestrator_tasks(
        log_path,
        in_flight_max_age_s=90.0,
        completed_window_s=window_s,
        max_items=80,
    )
    for t in recent_completed:
        role = base_role(t.get("final_role") or t.get("chosen_action") or "")
        if not role:
            continue
        bucket = per_role.setdefault(role, {
            "n_recent": 0,
            "n_completed": 0,
            "last_activity_age_s": None,
            "_tps_samples": [],
            "_duration_samples": [],
        })
        bucket["n_completed"] += 1
        dur = t.get("duration_s")
        if isinstance(dur, (int, float)) and dur > 0:
            bucket["_duration_samples"].append(float(dur))

    # Collapse internal sample lists to aggregates.
    out: dict[str, dict[str, Any]] = {}
    for role, b in per_role.items():
        tps_samples = b.pop("_tps_samples")
        dur_samples = b.pop("_duration_samples")
        b["avg_tps_recent"] = (sum(tps_samples) / len(tps_samples)) if tps_samples else None
        b["avg_duration_s"] = (sum(dur_samples) / len(dur_samples)) if dur_samples else None
        out[role] = b
    return JSONResponse({"per_role": out, "window_s": window_s, "now": now})


@router.get("/dashboard/api/topology")
async def topology() -> JSONResponse:
    """Return the static topology: nodes with role + display color + port."""
    llama_ports = _discover_llama_ports()
    services = _load_state_services()
    seen_ports: set[int] = set()
    nodes: list[dict[str, Any]] = []

    # Orchestrator at the center.
    nodes.append({
        "id": "orchestrator",
        "label": "orchestrator",
        "role": "orchestrator",
        "port": 8000,
        "color": _role_color("orchestrator"),
        "kind": "orchestrator",
    })
    seen_ports.add(8000)

    # Llama-servers from /proc scan.
    for port, role in sorted(llama_ports.items()):
        if port in seen_ports:
            continue
        seen_ports.add(port)
        nodes.append({
            "id": f"port_{port}",
            "label": role,
            "role": role,
            "port": port,
            "color": _role_color(role),
            "kind": "llama-server",
            # Alias roles served by the same process (e.g. frontdoor port 8070
            # also serves coder_escalation + worker_summarize). Surfaced so the
            # dashboard can render them under the primary role label.
            "aliases": role_aliases(role),
        })

    # Auxiliary services not already covered.
    for svc in services:
        port = svc.get("port")
        if not port or port in seen_ports:
            continue
        seen_ports.add(port)
        nodes.append({
            "id": svc["name"],
            "label": svc["name"],
            "role": svc["role"],
            "port": port,
            "color": _role_color(svc["role"]),
            "kind": "service",
            "model": svc.get("model", ""),
        })

    return JSONResponse({"nodes": nodes, "generated_at": time.time()})


# ---------------------------------------------------------------------------
# Snapshot endpoint — point-in-time state of all slots + counters
# ---------------------------------------------------------------------------

async def _poll_slot(client: httpx.AsyncClient, port: int) -> list[dict[str, Any]]:
    """Fetch /slots from a single llama-server. Returns empty on failure."""
    try:
        resp = await client.get(f"http://127.0.0.1:{port}/slots", timeout=1.5)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []


async def _poll_all_slots() -> dict[int, list[dict[str, Any]]]:
    """Concurrently poll /slots on every llama-server port we discovered."""
    ports = list(_discover_llama_ports().keys())
    out: dict[int, list[dict[str, Any]]] = {}
    if not ports:
        return out
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_poll_slot(client, p) for p in ports], return_exceptions=True,
        )
    for port, result in zip(ports, results):
        if isinstance(result, Exception):
            out[port] = []
        else:
            out[port] = result
    return out


# Snapshot scanners moved to dashboard_snapshot.py — wrappers preserve in-file API
# (route handlers below call _todays_progress_log() / _scan_recent_decisions() / etc.).


def _todays_progress_log() -> Path:
    return _todays_progress_log_impl(PROGRESS_LOG_DIR)


def _scan_recent_decisions(
    path: Path, window_s: float = 600.0, max_items: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    return _scan_recent_decisions_impl(path, window_s=window_s, max_items=max_items)


def _scan_orchestrator_tasks(
    path: Path,
    in_flight_max_age_s: float = INFLIGHT_MAX_AGE_DEFAULT_S,
    completed_window_s: float = 600.0,
    max_items: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _scan_orchestrator_tasks_impl(
        path,
        in_flight_max_age_s=in_flight_max_age_s,
        completed_window_s=completed_window_s,
        max_items=max_items,
    )


# Always-show window for in-flight tasks (Fix 3). A task younger than this is
# kept regardless of live slot state: its llama-server slot may not have flipped
# to is_processing yet (the /slots poll is 2 Hz), or the poll briefly missed the
# server. Older tasks must be corroborated by a busy slot on their role's
# server, which is what drops restart-orphans.
_FRESH_INFLIGHT_S = 20.0


def _gate_inflight_by_live_slots(
    tasks: list[dict[str, Any]], role_busy: dict[str, int],
) -> list[dict[str, Any]]:
    """Reconcile JSONL-derived in-flight tasks with live slot occupancy.

    The progress JSONL alone cannot tell a genuinely-running task from a
    restart-orphan (a `task_started` whose terminal event was lost when the API
    restarted). The live `/slots` poll can: if a role's server reports zero
    busy slots, nothing for that role is actually in flight. So per base role we
    keep at most ``max(busy_slots, fresh_tasks)`` of the newest started-but-
    unterminated tasks. This makes the in-flight list, the active badge, and the
    slot dots agree, hides stale orphans, and keeps long-running tasks visible
    for as long as their slot stays busy.
    """
    by_role: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        by_role.setdefault(t.get("role") or "unknown", []).append(t)
    gated: list[dict[str, Any]] = []
    for role, group in by_role.items():
        group.sort(key=lambda x: x.get("age_s", 0.0))  # newest first
        busy = role_busy.get(role, 0)
        n_fresh = sum(1 for t in group if t.get("age_s", 0.0) <= _FRESH_INFLIGHT_S)
        gated.extend(group[: max(busy, n_fresh)])
    gated.sort(key=lambda x: x.get("age_s", 0.0))
    return gated


def _count_log_events(
    path: Path, patterns: dict[str, str], window_s: float = 600.0,
) -> dict[str, int]:
    return _count_log_events_impl(path, patterns, window_s=window_s)


@router.get("/dashboard/api/snapshot")
async def snapshot() -> JSONResponse:
    """Point-in-time state: all slots + recent decisions + counters."""
    slots_by_port = await _poll_all_slots()
    progress_log = _todays_progress_log()
    recent, rolling, cumulative = _scan_recent_decisions(progress_log)
    orch_log = ORCHESTRATOR_LOG_DIR / "orchestrator.log"
    log_counts = _count_log_events(orch_log, {
        "inference_aborted": r"Inference aborted",
        "inference_lock_timeout": r"Inference lock timeout",
        "slot_erase": r"erasing slots on holder port",
        "watchdog_force_release": r"Lock hold watchdog: force-releasing",
    })

    # Derive per-node activity from slot states + live busy slots per base role.
    port_roles = _discover_llama_ports()
    role_busy: dict[str, int] = {}
    activity: dict[int, dict[str, Any]] = {}
    for port, slots in slots_by_port.items():
        n_total = len(slots)
        n_active = sum(1 for s in slots if s.get("is_processing"))
        role = base_role(port_roles.get(port, ""))
        if role:
            role_busy[role] = role_busy.get(role, 0) + n_active
        active_slots: list[dict[str, Any]] = []
        for s in slots:
            if not s.get("is_processing"):
                continue
            prompt = s.get("prompt", "") or ""
            if isinstance(prompt, list):
                prompt = " ".join(str(x) for x in prompt)
            content = s.get("content", "") or ""
            if isinstance(content, list):
                content = " ".join(str(x) for x in content)
            active_slots.append({
                "slot_id": s.get("id"),
                "task_id": s.get("id_task") if s.get("id_task", -1) >= 0 else None,
                "prompt_preview": prompt[:160],
                "content_preview": content[:200],
                "content_len": len(content),
                "tokens_decoded": s.get("n_decoded"),
                "prompt_tokens": s.get("n_prompt_tokens"),
                "next_token": s.get("next_token"),
            })
        activity[port] = {
            "n_total": n_total,
            "n_active": n_active,
            "active_slots": active_slots,
        }

    in_flight_tasks, recent_completed_tasks = _scan_orchestrator_tasks(progress_log)
    # Gate in-flight tasks on live slot occupancy so the task list, the active
    # badge, and the slot dots can't disagree (drops restart-orphans).
    in_flight_tasks = _gate_inflight_by_live_slots(in_flight_tasks, role_busy)

    return JSONResponse({
        "generated_at": time.time(),
        "activity": activity,
        "in_flight_tasks": in_flight_tasks,
        "recent_completed_tasks": recent_completed_tasks,
        "live_busy_by_role": role_busy,
        "recent_decisions": recent,
        "source_counts_rolling": rolling,
        "source_counts_cumulative": cumulative,
        "log_counts": log_counts,
    })


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------

@router.get("/dashboard/events/stream")
async def stream(request: Request) -> StreamingResponse:
    """1Hz Server-Sent Events stream of full snapshots."""

    async def event_gen():
        while True:
            if await request.is_disconnected():
                return
            try:
                resp = await snapshot()
                payload = resp.body.decode("utf-8")  # type: ignore[union-attr]
            except Exception as exc:
                payload = json.dumps({"error": str(exc)})
            yield f"data: {payload}\n\n"
            # 2 Hz instead of 1 Hz: more responsive in-flight panel updates
            # so short-lived tasks (sub-second) are more likely to be caught.
            # Snapshot is cheap (file mtime + JSONL tail + /slots fan-out is
            # parallel + bounded), so doubling the rate is a small cost.
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Task detail
# ---------------------------------------------------------------------------

@router.get("/dashboard/api/task/{task_id}.txt")
async def task_text(task_id: str) -> Any:
    """Return a plain-text snapshot of a task — for clipboard, curl, downstream LLMs.

    Mirrors the source-priority logic of /dashboard/api/task/{task_id}
    (live slot → tap_section → empty) so the .txt output stays
    consistent with what the dashboard panel displays.
    """
    log_path = _todays_progress_log()
    events = _task_events(task_id, log_path)
    slots_by_port = await _poll_all_slots()
    found_slot = None
    for port, slots in slots_by_port.items():
        for s in slots:
            if str(s.get("id_task")) == task_id:
                found_slot = s
                break
        if found_slot:
            break

    # Tap-section fallback when no live slot — same matcher the JSON
    # endpoint uses, including the role-filtered pass for higher precision.
    tap_section = None
    if found_slot is None:
        objective = _objective_for_task(events)
        producer_role = None
        for ev in reversed(events):
            if ev.get("event_type") == "task_completed":
                producer_role = (ev.get("data") or {}).get("producer_role")
                break
        if not producer_role:
            for ev in events:
                if ev.get("event_type") == "routing_decision":
                    producer_role = (ev.get("data") or {}).get("chosen_action")
                    break
        tap_section = _find_section_by_objective(objective, expected_role=producer_role)

    text = _task_text_snapshot(task_id, events, found_slot, tap_section=tap_section)
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(text)


async def _find_slot_by_objective(
    objective: str, slots_by_port: dict[int, list[dict[str, Any]]] | None = None,
) -> tuple[int | None, dict | None]:
    """Find a slot whose prompt contains the task's objective text.

    Orchestrator chat-XXX task_ids do NOT appear in llama-server /slots state
    (slot.id_task is llama-server's internal numeric counter). So we correlate
    by prompt content: the task's objective will appear inside the slot's
    `prompt` field if that slot is currently serving the task.
    """
    if not objective or len(objective) < 8:
        return None, None
    needle = objective[:120].strip()
    if slots_by_port is None:
        slots_by_port = await _poll_all_slots()
    for port, slots in slots_by_port.items():
        for s in slots:
            if not s.get("is_processing"):
                continue
            prompt = s.get("prompt", "") or ""
            if isinstance(prompt, list):
                prompt = " ".join(str(x) for x in prompt)
            if needle and needle in prompt:
                return port, s
    return None, None


@router.get("/dashboard/api/task/{task_id}")
async def task_detail(task_id: str) -> JSONResponse:
    """Return all events for a task_id + active slot + tap fallback.

    Matches active slots by prompt content (orchestrator chat-XXX ids do not
    correspond to llama-server's internal numeric id_task). For completed
    tasks where the slot is gone, falls back to searching the inference_tap.log
    for a section whose prompt matches — letting the UI show the historical
    response text instead of "(empty)".
    """
    log_path = _todays_progress_log()
    events = _task_events(task_id, log_path)
    objective = _objective_for_task(events)
    slots_by_port = await _poll_all_slots()
    slot_port, active_slot = await _find_slot_by_objective(objective, slots_by_port)

    # Fallback: if no live slot but the task completed, mine inference_tap.log.
    # Pass the role the task actually completed under (task_completed.producer_role)
    # — when present, the matcher uses role-filtered passes first for higher
    # precision when multiple roles processed the same prompt (architect →
    # specialist, forced-route handoffs, etc.).
    tap_section = None
    if active_slot is None:
        producer_role = None
        for ev in reversed(events):  # task_completed is usually near the end
            if ev.get("event_type") == "task_completed":
                producer_role = (ev.get("data") or {}).get("producer_role")
                break
        if not producer_role:
            for ev in events:
                if ev.get("event_type") == "routing_decision":
                    producer_role = (ev.get("data") or {}).get("chosen_action")
                    break
        tap_section = _find_section_by_objective(objective, expected_role=producer_role)

    return JSONResponse({
        "task_id": task_id,
        "objective": objective,
        "events": events,
        "active_slot_port": slot_port,
        "active_slot_id": active_slot.get("id") if active_slot else None,
        "slot": active_slot,
        "tap_section": tap_section,
    })


@router.get("/dashboard/events/task/{task_id}")
async def task_stream(task_id: str, request: Request) -> StreamingResponse:
    """5Hz SSE stream of the slot serving this task — live token feed via polling.

    Matches by prompt content (objective from progress JSONL). Tolerant of brief
    gaps where the slot is between tokens — keeps polling for up to
    `idle_giveup_s` consecutive idle samples before signalling done.
    """

    async def event_gen():
        last_content = ""
        # Pre-fetch the task's objective for prompt matching
        log_path = _todays_progress_log()
        events = _task_events(task_id, log_path)
        objective = _objective_for_task(events)
        # Quick exit: if the task already has a terminal event, signal done immediately.
        terminal_seen = any(
            e.get("event_type") in ("task_completed", "task_failed", "escalation_triggered")
            for e in events
        )
        if terminal_seen:
            yield "data: " + json.dumps({"delta": "", "done": True, "reason": "task_completed_already"}) + "\n\n"
            return

        idle_ticks = 0
        IDLE_GIVEUP = 60  # 60 ticks * 0.2s = 12s of no slot before signalling done
        while True:
            if await request.is_disconnected():
                return
            slot_port, slot = await _find_slot_by_objective(objective)
            if slot is not None:
                idle_ticks = 0
                content = slot.get("content", "") or ""
                if isinstance(content, list):
                    content = " ".join(str(x) for x in content)
                delta = content[len(last_content):]
                last_content = content
                payload = json.dumps({
                    "delta": delta,
                    "content_len": len(content),
                    "tokens_decoded": slot.get("n_decoded"),
                    "matched_port": slot_port,
                    "done": False,
                })
                yield f"data: {payload}\n\n"
            else:
                idle_ticks += 1
                # Re-check terminal state every few ticks
                if idle_ticks % 20 == 0:
                    fresh_events = _task_events(task_id, log_path)
                    if any(e.get("event_type") in ("task_completed", "task_failed", "escalation_triggered")
                           for e in fresh_events):
                        yield "data: " + json.dumps({"delta": "", "done": True, "reason": "task_completed"}) + "\n\n"
                        return
                if idle_ticks >= IDLE_GIVEUP:
                    yield "data: " + json.dumps({"delta": "", "done": True, "reason": "idle_timeout"}) + "\n\n"
                    return
                # Heartbeat so the client knows we're still searching
                if idle_ticks % 5 == 0:
                    yield "data: " + json.dumps({"delta": "", "searching": True, "idle_ticks": idle_ticks}) + "\n\n"
            await asyncio.sleep(0.2)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# GEPA progress
# ---------------------------------------------------------------------------

_AUTOPILOT_JOURNAL = ORCHESTRATOR_LOG_DIR.parent / "orchestration/autopilot_journal.jsonl"


@router.get("/dashboard/api/gepa")
async def gepa_status() -> JSONResponse:
    """Recent GEPA progress lines + parsed trial state + sentinel completion count."""
    if not AUTOPILOT_LOG.exists():
        return JSONResponse({"active": False, "lines": [], "state": {}})
    try:
        size = AUTOPILOT_LOG.stat().st_size
        with open(AUTOPILOT_LOG, "rb") as f:
            if size > 256 * 1024:
                f.seek(-256 * 1024, 2)
            tail = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return JSONResponse({"active": False, "lines": [], "state": {}})

    lines = tail.splitlines()
    gepa_lines = [
        l for l in lines
        if "gepa" in l.lower() or "Trial" in l or "sentinel" in l.lower()
        or "Dispatching action" in l or "prompt_forge" in l
    ][-30:]
    trial_state = _parse_trial_state(tail)

    # Sentinel completion progress: count chat-XXX task_completed events since
    # the last "evaluating baseline" line in the autopilot log.
    sentinels_done = 0
    trial_start_ts: float | None = None
    for line in lines:
        if "GEPA: evaluating baseline" in line:
            # Parse the timestamp from the line (format: 2026-05-21 17:24:44,...)
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m:
                try:
                    trial_start_ts = datetime.strptime(
                        m.group(1), "%Y-%m-%d %H:%M:%S"
                    ).timestamp()
                except Exception:
                    pass
    if trial_start_ts is not None and trial_state.get("baseline_score") is None:
        # We're mid-baseline: count completed chat tasks since trial_start_ts
        log_path = _todays_progress_log()
        if log_path.exists():
            with open(log_path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("event_type") != "task_completed":
                        continue
                    tid = e.get("task_id")
                    if not tid or not tid.startswith("chat-"):
                        continue
                    try:
                        ts = datetime.fromisoformat(
                            e["timestamp"].replace("Z", "+00:00")
                        ).timestamp()
                    except Exception:
                        continue
                    # autopilot log times are local (CEST); JSONL is UTC. Add
                    # a generous timezone tolerance (7200s = 2h).
                    if ts >= trial_start_ts - 7200:
                        sentinels_done += 1
    trial_state["sentinels_completed"] = sentinels_done

    # Last 10 trials from autopilot_journal for trajectory
    recent_trials: list[dict[str, Any]] = []
    if _AUTOPILOT_JOURNAL.exists():
        try:
            with open(_AUTOPILOT_JOURNAL, "rb") as f:
                size = _AUTOPILOT_JOURNAL.stat().st_size
                if size > 128 * 1024:
                    f.seek(-128 * 1024, 2)
                    f.readline()  # discard partial line
                journal_tail = f.read().decode("utf-8", errors="ignore")
            for line in journal_tail.splitlines()[-15:]:
                try:
                    j = json.loads(line)
                    recent_trials.append({
                        "trial_id": j.get("trial_id"),
                        "timestamp": j.get("timestamp", ""),
                        "species": j.get("species"),
                        "quality": j.get("quality"),
                        "speed": j.get("speed"),
                        "cost": j.get("cost"),
                        "reliability": j.get("reliability"),
                        "pareto_status": j.get("pareto_status"),
                        "description": (j.get("config_snapshot", {}).get("description") or "")[:140],
                    })
                except Exception:
                    pass
        except Exception:
            pass

    return JSONResponse({
        "active": bool(gepa_lines),
        "lines": gepa_lines,
        "state": trial_state,
        "recent_trials": recent_trials,
    })


# ---------------------------------------------------------------------------
# Single-page HTML (extracted to dashboard.html; re-read on each request).
#
# The HTML is read fresh from disk on every /dashboard hit instead of cached
# at module load, so edits to dashboard.html take effect after a browser
# reload — no orchestrator API restart needed. Cost: one ~43 KB file read
# per page hit (Linux page cache makes repeat reads essentially free).
# Use this path for HTML/CSS/JS hotfixes; Python route handlers still
# require `orchestrator_stack reload orchestrator`.
# ---------------------------------------------------------------------------

_DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard.html"


def _read_dashboard_html() -> str:
    """Read the dashboard HTML file fresh from disk."""
    return _DASHBOARD_HTML_PATH.read_text()


# Backwards-compat: tests + external readers historically check
# `_DASHBOARD_HTML` as a module attribute. Module-level __getattr__ resolves
# it lazily on each access by re-reading the file.
def __getattr__(name: str) -> str:
    if name == "_DASHBOARD_HTML":
        return _read_dashboard_html()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@router.get("/dashboard")
async def dashboard_page() -> HTMLResponse:
    """Serve the single-page dashboard (HTML re-read from disk per request)."""
    return HTMLResponse(_read_dashboard_html())
