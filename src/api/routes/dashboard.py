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

logger = logging.getLogger(__name__)
router = APIRouter()

ORCHESTRATOR_LOG_DIR = Path("/mnt/raid0/llm/epyc-orchestrator/logs")
PROGRESS_LOG_DIR = ORCHESTRATOR_LOG_DIR / "progress"
AUTOPILOT_LOG = ORCHESTRATOR_LOG_DIR / "autopilot.log"
ORCHESTRATOR_STATE_PATH = ORCHESTRATOR_LOG_DIR / "orchestrator_state.json"

# Port-range hints used as fallback if registry doesn't resolve a port.
_PORT_HINTS: dict[int, str] = {
    8000: "orchestrator",
    8070: "frontdoor",
    8072: "worker_general",
    8083: "architect_general",
    8085: "ingest_long_context",
    8086: "worker_vision",
    8087: "vision_escalation",
    8088: "nextplaid-code",
    8089: "nextplaid-docs",
    8090: "embedder",
    8091: "embedder_1",
    8092: "embedder_2",
    8093: "embedder_3",
    8094: "embedder_4",
    8095: "embedder_5",
    8102: "worker_fast",
    8190: "sd_server",
    9000: "whisper",
    9001: "document_formalizer",
}
# NUMA quarters share the parent role.
for parent_base, parent_role in ((8080, "frontdoor"), (8082, "worker_general")):
    for q in range(4):
        _PORT_HINTS[parent_base + q * 100] = f"{parent_role}.q{q}"

# Per-role display colors (CSS hex).
_ROLE_COLORS: dict[str, str] = {
    "frontdoor": "#3b82f6",
    "worker_general": "#10b981",
    "worker_explore": "#10b981",
    "worker_math": "#10b981",
    "architect_general": "#a855f7",
    "ingest_long_context": "#f59e0b",
    "coder_escalation": "#ef4444",
    "worker_summarize": "#06b6d4",
    "worker_vision": "#ec4899",
    "vision_escalation": "#ec4899",
    "embedder": "#94a3b8",
    "orchestrator": "#475569",
}


def _role_color(role: str) -> str:
    """Resolve a role label to its display color, falling back to gray.

    Strips both `.qN` (NUMA quarter) and `_N` (numbered siblings like
    embedder_1) suffixes before lookup.
    """
    base = role.split(".")[0]
    # Strip trailing _<digits> if the prefix is a known role family.
    m = re.match(r"^(.+?)_\d+$", base)
    if m and m.group(1) in _ROLE_COLORS:
        base = m.group(1)
    return _ROLE_COLORS.get(base, "#64748b")


def _discover_llama_ports() -> dict[int, str]:
    """Scan /proc for running llama-server processes and extract port→role.

    Falls back to _PORT_HINTS for unmapped ports. Cheap (~5ms), runs once per
    snapshot poll.
    """
    ports: dict[int, str] = {}
    try:
        import subprocess
        out = subprocess.run(
            ["ps", "-eo", "pid,cmd"], capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception:
        out = ""
    pid_port_re = re.compile(r"--port\s+(\d+)")
    pid_model_re = re.compile(r"-m\s+(\S+)")
    for line in out.splitlines():
        if "llama-server" not in line:
            continue
        port_m = pid_port_re.search(line)
        if not port_m:
            continue
        port = int(port_m.group(1))
        role = _PORT_HINTS.get(port, f"port_{port}")
        # If the cmd has -m, prefer a model-derived label as a fallback role hint
        if role == f"port_{port}":
            model_m = pid_model_re.search(line)
            if model_m:
                stem = Path(model_m.group(1)).stem[:24]
                role = f"port_{port}({stem})"
        ports[port] = role
    return ports


def _load_state_services() -> list[dict[str, Any]]:
    """Load non-llama auxiliary services from orchestrator_state.json."""
    services: list[dict[str, Any]] = []
    try:
        with open(ORCHESTRATOR_STATE_PATH) as f:
            state = json.load(f)
        for key, info in state.items():
            if not isinstance(info, dict):
                continue
            services.append({
                "name": key,
                "role": info.get("role", key),
                "port": info.get("port"),
                "model": info.get("model_path", ""),
                "pid": info.get("pid", -1),
            })
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("Failed to load orchestrator_state.json: %s", exc)
    return services


# ---------------------------------------------------------------------------
# Inference taps — live prompt + response stream written by autopilot's
# seeding harness to /mnt/raid0/llm/tmp/*. These are the same files the
# autopilot --tui tails. Reading them is cheap (just stat + seek-to-end)
# and gives us the full token stream + REPL history for free.
# ---------------------------------------------------------------------------

_INFERENCE_TAP_PATH = Path("/mnt/raid0/llm/tmp/inference_tap.log")
_REPL_TAP_PATH = Path("/mnt/raid0/llm/tmp/repl_tap.log")
_PROMPT_TAP_PATH = Path("/mnt/raid0/llm/tmp/autopilot_prompt_tap.txt")
_TAP_SENTINEL_PATH = Path("/mnt/raid0/llm/tmp/.inference_tap_active")

_SECTION_SEP = "=" * 72
_SUBSECTION_SEP = "-" * 72


def _read_tail(path: Path, max_bytes: int = 256 * 1024) -> str:
    """Read the last ~max_bytes from a file, decoded as UTF-8."""
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
                # Discard partial first line
                _ = f.readline()
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _parse_inference_sections(tail_text: str, max_sections: int = 20) -> list[dict[str, Any]]:
    """Parse the last N (ROLE, PROMPT, RESPONSE) sections from inference_tap.log.

    The tap format used by autopilot/seeding:
        [2026-05-21 11:16:22] ROLE=worker_general
        ------------------------------------------------------------------------
        PROMPT: <prompt text>
        ------------------------------------------------------------------------
        RESPONSE:
        <response text>
        ========================================================================

    Returns the most-recent sections first (descending chronological).
    """
    sections: list[dict[str, Any]] = []
    if not tail_text:
        return sections
    # Split on the section terminator
    raw_sections = tail_text.split(_SECTION_SEP)
    # The first chunk is likely a partial section from before our tail window —
    # skip it unless it has both a PROMPT and RESPONSE
    candidates = raw_sections[-(max_sections + 1):]
    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Extract role + timestamp from "[YYYY-MM-DD HH:MM:SS] ROLE=xxx"
        role_match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*ROLE=(\S+)", chunk)
        # Extract PROMPT and RESPONSE blocks
        # PROMPT: ... RESPONSE: ... (RESPONSE may run to end of section)
        prompt_idx = chunk.find("PROMPT:")
        response_idx = chunk.find("RESPONSE:")
        if prompt_idx < 0 or response_idx < 0 or response_idx < prompt_idx:
            continue
        prompt_text = chunk[prompt_idx + len("PROMPT:"):response_idx].strip()
        # Strip the subsection separator before RESPONSE
        prompt_text = re.sub(r"-{20,}\s*$", "", prompt_text).strip()
        response_text = chunk[response_idx + len("RESPONSE:"):].strip()
        # Filter out llama-server's `TIMINGS:` probe/healthcheck responses —
        # those are llama.cpp's internal timing dumps emitted on empty
        # generations, not real inference output.
        response_stripped = response_text.lstrip("-").lstrip()
        if response_stripped.startswith("TIMINGS:") and len(response_text) < 400:
            continue
        sections.append({
            "timestamp": role_match.group(1) if role_match else None,
            "role": role_match.group(2) if role_match else None,
            "prompt": prompt_text,
            "response": response_text,
            "prompt_len": len(prompt_text),
            "response_len": len(response_text),
        })
    # Most recent first
    return list(reversed(sections))


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

def _process_info_by_match(needle: str) -> dict[str, Any]:
    """Find a long-running Python process by command-line substring."""
    try:
        import subprocess
        out = subprocess.run(
            ["ps", "-eo", "pid,etime,pcpu,cmd"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception:
        return {"running": False}
    for line in out.splitlines()[1:]:
        if needle in line and "grep" not in line:
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            return {
                "running": True,
                "pid": int(parts[0]),
                "etime": parts[1],
                "pcpu": float(parts[2]),
                "cmd": parts[3][:200],
            }
    return {"running": False}


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
    return JSONResponse({
        "autopilot": autopilot,
        "gepa_worker_count": n_workers,
        "last_autopilot_log_age_s": last_log_age_s,
        "autopilot_recent_lines": recent_lines,
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


def _todays_progress_log() -> Path:
    return PROGRESS_LOG_DIR / f"{date.today().isoformat()}.jsonl"


def _scan_recent_decisions(
    path: Path, window_s: float = 600.0, max_items: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Tail today's progress JSONL, return recent decisions + counters.

    Returns (recent_list, source_counts_rolling, source_counts_cumulative).
    """
    recent: deque = deque(maxlen=max_items)
    source_rolling: dict[str, int] = {}
    source_cumulative: dict[str, int] = {}
    verifier_verdicts: dict[str, int] = {}
    now = time.time()
    if not path.exists():
        return list(recent), source_rolling, source_cumulative
    try:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event_type") != "routing_decision":
                    continue
                d = e.get("data", {})
                src = d.get("decision_source") or d.get("strategy") or "?"
                source_cumulative[src] = source_cumulative.get(src, 0) + 1
                ts_str = e.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = now
                age = now - ts
                if age <= window_s:
                    source_rolling[src] = source_rolling.get(src, 0) + 1
                    if d.get("verifier_verdict"):
                        v = d["verifier_verdict"]
                        verifier_verdicts[v] = verifier_verdicts.get(v, 0) + 1
                # Build a compact summary
                recent.append({
                    "task_id": e.get("task_id"),
                    "ts": ts_str,
                    "age_s": round(age, 1),
                    "source": src,
                    "chosen_action": d.get("chosen_action") or "",
                    "classifier_confidence": d.get("classifier_confidence"),
                    "verifier_p_success": d.get("verifier_p_success"),
                    "verifier_verdict": d.get("verifier_verdict"),
                    "verifier_shadow": d.get("verifier_shadow"),
                })
    except Exception as exc:
        logger.debug("scan_recent_decisions failed: %s", exc)
    source_rolling["_verifier_verdicts"] = verifier_verdicts  # type: ignore[assignment]
    return list(recent), source_rolling, source_cumulative


def _scan_orchestrator_tasks(
    path: Path,
    in_flight_max_age_s: float = 300.0,   # only show in-flight if started < 5 min ago
    completed_window_s: float = 600.0,    # completed in last 10 min
    max_items: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan today's progress JSONL for in-flight + recently-completed chat tasks.

    Uses the orchestrator's `chat-XXX` task_ids (not llama-server's internal
    numeric id_task). Returns (in_flight, recent_completed).

    NB: we DO NOT time-filter routing_decision events during the scan — we
    need them merged into every started/completed task even if the routing
    happened minutes before our window. Tasks that started > in_flight_max_age_s
    ago and never completed are treated as orphans (typically killed by an
    API restart) and excluded from in-flight.
    """
    if not path.exists():
        return [], []
    now = time.time()
    started: dict[str, dict[str, Any]] = {}
    terminal_events: dict[str, dict[str, Any]] = {}
    routing_meta: dict[str, dict[str, Any]] = {}
    try:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = e.get("task_id")
                if not tid or not isinstance(tid, str) or not tid.startswith("chat-"):
                    continue
                ev = e.get("event_type")
                ts_str = e.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    continue
                data = e.get("data", {})
                if ev == "task_started":
                    started[tid] = {
                        "task_id": tid,
                        "started_at": ts,
                        "age_s": now - ts,
                        "objective": (data.get("objective", "") or "")[:200],
                        "task_type": data.get("task_type"),
                        "priority": data.get("priority"),
                    }
                elif ev == "routing_decision":
                    routing_meta[tid] = {
                        "chosen_action": data.get("chosen_action") or "",
                        "decision_source": data.get("decision_source") or "",
                        "classifier_confidence": data.get("classifier_confidence"),
                        "verifier_p_success": data.get("verifier_p_success"),
                        "verifier_verdict": data.get("verifier_verdict"),
                        "difficulty_band": data.get("difficulty_band"),
                    }
                elif ev in ("task_completed", "task_failed", "escalation_triggered"):
                    terminal_events[tid] = {
                        "event_type": ev,
                        "ended_at": ts,
                        "age_s": now - ts,
                        "final_role": data.get("final_answer_role") or data.get("producer_role"),
                    }
    except Exception as exc:
        logger.debug("scan_orchestrator_tasks failed: %s", exc)

    in_flight: list[dict[str, Any]] = []
    recent_completed: list[dict[str, Any]] = []
    for tid, s in started.items():
        s.update(routing_meta.get(tid, {}))
        if tid not in terminal_events:
            # Treat as in-flight only if young enough; older = orphan
            if s["age_s"] <= in_flight_max_age_s:
                in_flight.append(s)
        else:
            t = terminal_events[tid]
            # Time-filter completed: only show within the completed window
            if t["age_s"] > completed_window_s:
                continue
            s["ended_at"] = t["ended_at"]
            s["end_age_s"] = t["age_s"]
            s["outcome"] = t["event_type"]
            s["duration_s"] = round(t["ended_at"] - s["started_at"], 2)
            s["final_role"] = t.get("final_role")
            recent_completed.append(s)
    in_flight.sort(key=lambda x: x["age_s"])
    recent_completed.sort(key=lambda x: x["end_age_s"])
    return in_flight[:max_items], recent_completed[:max_items]


def _count_log_events(path: Path, patterns: dict[str, str], window_s: float = 600.0) -> dict[str, int]:
    """Tail the orchestrator log and count occurrences of regex patterns."""
    counts = {key: 0 for key in patterns}
    if not path.exists():
        return counts
    now = time.time()
    # Try a recent-tail-only optimization: read last 256KB if file is big
    try:
        size = path.stat().st_size
        if size > 256 * 1024:
            with open(path, "rb") as f:
                f.seek(-256 * 1024, 2)
                tail = f.read().decode("utf-8", errors="ignore")
        else:
            with open(path) as f:
                tail = f.read()
    except Exception:
        return counts
    compiled = {k: re.compile(v) for k, v in patterns.items()}
    for line in tail.splitlines():
        for key, regex in compiled.items():
            if regex.search(line):
                counts[key] += 1
    return counts


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

    # Derive per-node activity from slot states
    activity: dict[int, dict[str, Any]] = {}
    for port, slots in slots_by_port.items():
        n_total = len(slots)
        n_active = sum(1 for s in slots if s.get("is_processing"))
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

    return JSONResponse({
        "generated_at": time.time(),
        "activity": activity,
        "in_flight_tasks": in_flight_tasks,
        "recent_completed_tasks": recent_completed_tasks,
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
            await asyncio.sleep(1.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Task detail
# ---------------------------------------------------------------------------

def _task_events(task_id: str, path: Path, max_events: int = 200) -> list[dict[str, Any]]:
    """Return all progress-log events with a given task_id."""
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    try:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("task_id") != task_id:
                    continue
                events.append({
                    "event_type": e.get("event_type"),
                    "timestamp": e.get("timestamp"),
                    "data": e.get("data", {}),
                })
                if len(events) >= max_events:
                    break
    except Exception:
        pass
    return events


def _task_text_snapshot(task_id: str, events: list[dict[str, Any]], slot: dict | None) -> str:
    """Render a plain-text snapshot of a task suitable for pasting into chat."""
    lines: list[str] = []
    lines.append(f"=== Task {task_id} @ {datetime.utcnow().isoformat()}Z ===")
    lines.append("")
    prompt_text = ""
    stream_text = ""
    if slot:
        prompt_text = str(slot.get("prompt") or "")
        stream_text = str(slot.get("content") or "")
    if not prompt_text:
        for ev in events:
            if ev.get("event_type") == "task_started":
                prompt_text = ev.get("data", {}).get("objective", "") or ""
                break
    lines.append("PROMPT:")
    lines.append("-------")
    lines.append(prompt_text or "(not available)")
    lines.append("")
    lines.append("INFERENCE STREAM:")
    lines.append("-----------------")
    lines.append(stream_text or "(empty)")
    lines.append("")
    lines.append(f"REPL HISTORY ({len(events)} events):")
    lines.append("-----------------")
    # Keys to suppress from REPL-event payloads when rendering for chat-paste.
    # stack_state in particular is ~4KB of registry dump per decision; valuable
    # for offline debugging but pure noise in a conversation.
    _NOISY_KEYS = {"stack_state", "similarity_topk", "q_topk", "selection_score_topk",
                   "prior_term_topk", "posterior_score_topk", "learned_evidence_topk",
                   "cost_term_topk"}
    for ev in events:
        ts = (ev.get("timestamp", "") or "").replace("T", " ")[11:19]
        ev_type = ev.get("event_type", "?")
        data = ev.get("data", {})
        if isinstance(data, dict):
            filtered = {k: v for k, v in data.items() if k not in _NOISY_KEYS}
            if len(filtered) < len(data):
                # Note when keys were elided so the reader knows.
                filtered["_elided_keys"] = sorted(set(data.keys()) - set(filtered.keys()))
        else:
            filtered = data
        try:
            data_str = json.dumps(filtered, separators=(",", ":"))
        except Exception:
            data_str = str(filtered)
        lines.append(f"[{ts}] {ev_type}: {data_str}")
    return "\n".join(lines)


@router.get("/dashboard/api/task/{task_id}.txt")
async def task_text(task_id: str) -> Any:
    """Return a plain-text snapshot of a task — for clipboard, curl, downstream LLMs."""
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
    text = _task_text_snapshot(task_id, events, found_slot)
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(text)


def _objective_for_task(events: list[dict[str, Any]]) -> str:
    """Extract the original prompt/objective from a task's events."""
    for ev in events:
        if ev.get("event_type") == "task_started":
            return str(ev.get("data", {}).get("objective", "") or "")
    return ""


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
    """Return all events for a task_id + try to identify its current slot.

    Matches active slots by prompt content (orchestrator chat-XXX ids do not
    correspond to llama-server's internal numeric id_task).
    """
    log_path = _todays_progress_log()
    events = _task_events(task_id, log_path)
    objective = _objective_for_task(events)
    slots_by_port = await _poll_all_slots()
    slot_port, active_slot = await _find_slot_by_objective(objective, slots_by_port)

    return JSONResponse({
        "task_id": task_id,
        "objective": objective,
        "events": events,
        "active_slot_port": slot_port,
        "active_slot_id": active_slot.get("id") if active_slot else None,
        "slot": active_slot,
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


def _parse_trial_state(tail: str) -> dict[str, Any]:
    """Scan autopilot.log tail for the active trial's state."""
    state: dict[str, Any] = {
        "current_trial": None,
        "current_action": None,
        "current_file": None,
        "baseline_sentinels_total": None,
        "baseline_score": None,
        "last_event": None,
    }
    # Iterate in order — last match wins
    re_trial = re.compile(r"Trial (\d+):\s*({.*})")
    re_baseline = re.compile(r"GEPA: evaluating baseline for (\S+\.md) \((\d+) sentinels\)")
    re_score = re.compile(r"GEPA: baseline score = ([\d.]+)")
    re_dispatch = re.compile(r"Dispatching action: (\w+)")
    for line in tail.splitlines():
        m = re_trial.search(line)
        if m:
            state["current_trial"] = int(m.group(1))
            try:
                cfg = json.loads(m.group(2))
                state["current_action"] = cfg.get("type")
                state["current_file"] = cfg.get("file")
            except Exception:
                pass
        m = re_baseline.search(line)
        if m:
            state["current_file"] = m.group(1)
            state["baseline_sentinels_total"] = int(m.group(2))
            state["last_event"] = "evaluating_baseline"
        m = re_score.search(line)
        if m:
            state["baseline_score"] = float(m.group(1))
            state["last_event"] = "baseline_done"
        m = re_dispatch.search(line)
        if m and not state["current_action"]:
            state["current_action"] = m.group(1)
    return state


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
# Single-page HTML
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Orchestrator Dashboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked-highlight@2.1.1/lib/index.umd.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/atom-one-dark.min.css">
<style>
:root {
    --bg:#0f172a; --panel:#1e293b; --text:#e2e8f0; --muted:#94a3b8;
    --border:#334155; --accent:#3b82f6;
    --good:#10b981; --warn:#f59e0b; --bad:#ef4444;
}
* { box-sizing: border-box; }
body {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--bg); color: var(--text); margin:0; padding:12px;
    font-size: 12px; line-height: 1.4;
}
h1 { font-size: 16px; margin: 0 0 12px 0; color: var(--text); }
h2 { font-size: 13px; margin: 0 0 8px 0; color: var(--muted); letter-spacing: .05em; text-transform: uppercase; }
.grid {
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    grid-template-rows: auto auto;
    gap: 12px;
}
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    overflow: auto;
}
.panel.topology { grid-row: 1 / span 2; }
.counters {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
    margin-bottom: 12px;
}
.counter {
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 10px;
}
.counter .label { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .05em; }
.counter .value { font-size: 18px; font-weight: 600; color: var(--text); margin-top: 2px; }
.counter .sub { color: var(--muted); font-size: 10px; }
.task-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 8px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: background .1s;
}
.task-card:hover { background: rgba(255,255,255,0.08); }
.task-card .role { font-weight: 600; }
.task-card .meta { color: var(--muted); font-size: 10px; margin-top: 2px; }
.task-card .prompt { color: var(--text); font-size: 11px; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.decision-row {
    padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
    display: grid; grid-template-columns: 70px 100px 100px 60px 60px 60px;
    gap: 6px; font-size: 11px;
}
.decision-row .src { color: var(--accent); }
.decision-row .src.classifier { color: var(--good); }
.decision-row .src.rules { color: var(--warn); }
.tag {
    display: inline-block; padding: 1px 6px; border-radius: 2px;
    font-size: 10px; background: rgba(255,255,255,0.1);
}
.tag.accept { background: rgba(16,185,129,0.2); color: #34d399; }
.tag.reject { background: rgba(239,68,68,0.2); color: #f87171; }
.tag.shadow { background: rgba(245,158,11,0.15); color: #fcd34d; }
svg.topology { width: 100%; height: 500px; background: rgba(0,0,0,0.2); border-radius: 6px; }
svg .node-circle { transition: r .2s; cursor: pointer; }
svg .node-label { fill: var(--text); font-family: ui-monospace, monospace; font-size: 10px; pointer-events: none; }
svg .edge { stroke: var(--border); stroke-width: 1; fill: none; }
svg .edge.active { stroke: var(--accent); stroke-width: 2; }
svg .pulse {
    fill: var(--accent);
    animation: pulse 1.2s linear infinite;
}
@keyframes pulse { from { offset-distance: 0%; opacity: 1; } to { offset-distance: 100%; opacity: 0; } }
#detail-panel {
    position: fixed; top: 0; right: -640px; width: 620px; height: 100vh;
    background: var(--panel); border-left: 1px solid var(--border);
    transition: right .25s ease; overflow-y: auto;
    z-index: 100; padding: 14px;
}
#detail-panel.open { right: 0; }
#detail-close { float: right; cursor: pointer; color: var(--muted); font-size: 16px; padding: 0 6px; }
#detail-close:hover { color: var(--text); }
#detail-content h3 { margin: 6px 0 4px 0; color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
.prompt-text, .stream-text, .tap-content {
    background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px;
    max-height: 320px; overflow-y: auto; overflow-x: auto;
    font-size: 12px; line-height: 1.5;
    font-family: ui-sans-serif, system-ui, sans-serif;
    word-wrap: break-word; overflow-wrap: break-word;
}
.prompt-text pre, .stream-text pre, .tap-content pre {
    background: #0b1220; padding: 8px; border-radius: 3px;
    overflow-x: auto; font-family: ui-monospace, monospace; font-size: 11px;
    white-space: pre; word-wrap: normal;
}
.prompt-text code, .stream-text code, .tap-content code {
    background: rgba(255,255,255,0.07); padding: 1px 4px; border-radius: 2px;
    font-family: ui-monospace, monospace; font-size: 11px;
}
.prompt-text p, .stream-text p, .tap-content p { margin: 4px 0; }
details.tap-section { border-bottom: 1px solid var(--border); padding: 4px 0; }
details.tap-section summary { font-size: 11px; outline: none; }
details.tap-section summary::-webkit-details-marker { color: var(--muted); }
#detail-content .stream-text.live::after {
    content: "▌"; animation: blink 1s infinite;
    color: var(--accent);
}
.detail-actions {
    display: flex; gap: 6px; margin: 8px 0;
}
.detail-actions button {
    background: var(--accent); color: white; border: none; padding: 4px 10px;
    border-radius: 3px; cursor: pointer; font-size: 11px;
}
.detail-actions button.secondary {
    background: var(--border); color: var(--text);
}
.detail-actions button:hover { opacity: 0.85; }
.copy-feedback {
    color: var(--good); font-size: 11px; padding: 4px 8px;
    opacity: 0; transition: opacity .2s;
}
.copy-feedback.shown { opacity: 1; }
@keyframes blink { 50% { opacity: 0; } }
.repl-event {
    padding: 3px 6px; border-left: 2px solid var(--border);
    margin: 3px 0; font-size: 11px;
}
.repl-event .ts { color: var(--muted); font-size: 10px; }
.repl-event .ev { color: var(--accent); font-weight: 600; }
.gepa-line { font-size: 10px; color: var(--muted); padding: 1px 0; }
.gepa-line.bold { color: var(--text); font-weight: 600; }
</style>
</head><body>
<h1>orchestrator dashboard
    <span style="font-weight:normal;color:var(--muted)" id="status">connecting…</span>
    <span style="font-weight:normal;color:var(--muted);margin-left:12px;font-size:11px" id="process-status"></span>
</h1>

<div class="counters" id="counters"></div>

<div class="grid">
  <div class="panel topology">
    <h2>topology — click a node for details, edges glow when active</h2>
    <svg class="topology" id="topology-svg" viewBox="0 0 700 500"></svg>
  </div>
  <div class="panel">
    <h2>in-flight tasks <span style="color:var(--muted);font-weight:normal" id="inflight-count"></span></h2>
    <div id="inflight-tasks"></div>
    <h2 style="margin-top:14px">recently completed <span style="color:var(--muted);font-weight:normal" id="completed-count"></span></h2>
    <div id="completed-tasks"></div>
    <h2 style="margin-top:14px">live inference <span style="color:var(--muted);font-weight:normal" id="tap-status"></span></h2>
    <div id="inference-tap"></div>
  </div>
  <div class="panel">
    <h2>recent routing decisions (last 50)</h2>
    <div class="decision-row" style="font-weight:600;border-bottom:1px solid var(--border);color:var(--muted)">
      <div>age</div><div>task_id</div><div>action</div><div>src</div><div>conf</div><div>verifier</div>
    </div>
    <div id="decision-feed"></div>
  </div>
</div>

<div class="panel" style="margin-top:12px">
  <h2>gepa progress (autopilot.log tail)</h2>
  <div id="gepa-lines"></div>
</div>

<div id="detail-panel">
  <span id="detail-close" onclick="closeDetail()">×</span>
  <h2 id="detail-title">task detail</h2>
  <div id="detail-content"></div>
</div>

<script>
const NODE_RADIUS = 18;
let topology = null;
let activeStream = null;
let detailStream = null;

async function loadTopology() {
    const r = await fetch('/dashboard/api/topology');
    topology = await r.json();
    layoutTopology();
}

function layoutTopology() {
    const svg = document.getElementById('topology-svg');
    svg.innerHTML = '';
    const w = 700, h = 500;
    const cx = w / 2, cy = h / 2;
    const orchestrator = topology.nodes.find(n => n.kind === 'orchestrator');
    const others = topology.nodes.filter(n => n.kind !== 'orchestrator');
    const services = others.filter(n => n.kind === 'service');
    const llamas = others.filter(n => n.kind === 'llama-server');
    // Inner ring: llama-servers
    const radiusInner = 150;
    llamas.forEach((n, i) => {
        const a = (i / llamas.length) * 2 * Math.PI - Math.PI / 2;
        n._x = cx + radiusInner * Math.cos(a);
        n._y = cy + radiusInner * Math.sin(a);
    });
    // Outer ring: services
    const radiusOuter = 220;
    services.forEach((n, i) => {
        const a = (i / services.length) * 2 * Math.PI - Math.PI / 2 + 0.1;
        n._x = cx + radiusOuter * Math.cos(a);
        n._y = cy + radiusOuter * Math.sin(a);
    });
    if (orchestrator) { orchestrator._x = cx; orchestrator._y = cy; }

    // Draw edges from orchestrator to every other node (initially inactive)
    others.forEach(n => {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        path.setAttribute('id', `edge_${n.id}`);
        path.setAttribute('class', 'edge');
        path.setAttribute('x1', cx); path.setAttribute('y1', cy);
        path.setAttribute('x2', n._x); path.setAttribute('y2', n._y);
        svg.appendChild(path);
    });

    // Draw nodes
    topology.nodes.forEach(n => {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('transform', `translate(${n._x},${n._y})`);
        g.setAttribute('style', 'cursor:pointer');
        g.addEventListener('click', () => openNodeDetail(n.port, n.label));
        g.addEventListener('mouseenter', (ev) => showNodeTooltip(ev, n));
        g.addEventListener('mouseleave', hideNodeTooltip);
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('class', 'node-circle');
        c.setAttribute('id', `node_${n.id}`);
        c.setAttribute('r', NODE_RADIUS);
        c.setAttribute('fill', '#475569');
        c.setAttribute('stroke', n.color);
        c.setAttribute('stroke-width', 2);
        const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        t.setAttribute('class', 'node-label');
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('y', NODE_RADIUS + 12);
        t.textContent = n.label.length > 18 ? n.label.slice(0, 17) + '…' : n.label;
        const tp = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        tp.setAttribute('class', 'node-label');
        tp.setAttribute('text-anchor', 'middle');
        tp.setAttribute('y', 4);
        tp.setAttribute('id', `nodecount_${n.id}`);
        tp.textContent = '';
        g.appendChild(c); g.appendChild(t); g.appendChild(tp);
        svg.appendChild(g);
    });
}

let _tooltipEl = null;
function showNodeTooltip(ev, n) {
    if (!_tooltipEl) {
        _tooltipEl = document.createElement('div');
        _tooltipEl.id = 'node-tooltip';
        _tooltipEl.style.cssText = 'position:fixed; background:#1e293b; border:1px solid #334155; border-radius:4px; padding:6px 10px; font-size:11px; color:#e2e8f0; pointer-events:none; z-index:200; max-width:300px';
        document.body.appendChild(_tooltipEl);
    }
    _tooltipEl.innerHTML = `
        <div style="color:${n.color};font-weight:600">${escapeHTML(n.label)}</div>
        <div style="color:#94a3b8;font-size:10px;margin-top:2px">port ${n.port} · ${n.kind}</div>
        <div style="color:#94a3b8;font-size:10px;margin-top:2px">click for details</div>`;
    _tooltipEl.style.left = (ev.clientX + 12) + 'px';
    _tooltipEl.style.top = (ev.clientY + 12) + 'px';
    _tooltipEl.style.display = 'block';
}
function hideNodeTooltip() {
    if (_tooltipEl) _tooltipEl.style.display = 'none';
}

async function openNodeDetail(port, label) {
    const panel = document.getElementById('detail-panel');
    const content = document.getElementById('detail-content');
    document.getElementById('detail-title').textContent = `node ${label} (port ${port})`;
    content.innerHTML = '<div style="color:var(--muted)">loading…</div>';
    panel.classList.add('open');
    if (detailStream) { detailStream.close(); detailStream = null; }
    try {
        const r = await fetch(`/dashboard/api/node/${port}`);
        const d = await r.json();
        const proc = d.process || {};
        const slotRows = (d.slots || []).map(s => `
            <div class="repl-event">
                <span class="ev">slot ${s.id}</span>
                ${s.is_processing ? '<span class="tag accept" style="margin-left:6px">processing</span>' : '<span class="tag" style="margin-left:6px">idle</span>'}
                <div style="font-size:10px;color:var(--muted);margin-top:2px">
                    n_decoded=${s.n_decoded ?? '-'} · prompt_tokens=${s.n_prompt_tokens ?? '-'} · content=${s.content_len}c
                </div>
                ${s.prompt_preview ? `<div style="font-size:11px;margin-top:2px">prompt: ${escapeHTML(s.prompt_preview)}</div>` : ''}
            </div>
        `).join('') || '<div style="color:var(--muted)">no slot data</div>';
        const routedRows = (d.recent_routed || []).slice().reverse().map(r => `
            <div class="repl-event">
                <span class="ts">${(r.timestamp||'').replace('T',' ').slice(11,19)}</span>
                <span class="ev">${(r.task_id||'').slice(-12)}</span>
                <span style="font-size:10px;color:var(--muted)">src=${r.decision_source} · conf=${r.classifier_confidence!=null?r.classifier_confidence.toFixed(2):'-'}</span>
            </div>
        `).join('') || '<div style="color:var(--muted)">no recent decisions routed here</div>';
        const healthColor = d.health_status === 'ok' ? 'var(--good)' :
                            d.health_status === 'connection_refused' ? 'var(--bad)' :
                            'var(--warn)';
        content.innerHTML = `
            <h3>process</h3>
            <div style="font-size:11px;line-height:1.7">
                <div>pid: <code>${proc.pid||'—'}</code></div>
                <div>uptime: ${proc.etime||'—'}</div>
                <div>cum %CPU: ${proc.pcpu_cumulative!=null?proc.pcpu_cumulative.toFixed(1)+'%':'—'}</div>
                <div>RSS: ${proc.rss_kb!=null?(proc.rss_kb/1024/1024).toFixed(1)+' GB':'—'}</div>
                <div>health: <span style="color:${healthColor}">${d.health_status || '—'}</span></div>
                <div>slots: ${d.n_processing}/${d.n_slots} processing</div>
            </div>
            <h3>slots (${d.n_slots})</h3>
            <div>${slotRows}</div>
            <h3>recent decisions routed here (${d.recent_routed_count})</h3>
            <div>${routedRows}</div>
            <details style="margin-top:8px"><summary style="cursor:pointer;color:var(--muted);font-size:11px">cmd</summary>
                <div style="background:rgba(0,0,0,0.3);padding:6px;border-radius:3px;font-size:10px;word-break:break-all;margin-top:4px">${escapeHTML(proc.cmd||'')}</div>
            </details>
        `;
    } catch (e) {
        content.innerHTML = `<div style="color:var(--bad)">error: ${e.message}</div>`;
    }
}

function updateTopology(activity) {
    if (!topology) return;
    topology.nodes.forEach(n => {
        const port = n.port;
        const a = activity[port];
        const circle = document.getElementById(`node_${n.id}`);
        const countText = document.getElementById(`nodecount_${n.id}`);
        const edge = document.getElementById(`edge_${n.id}`);
        if (a && a.n_active > 0) {
            if (circle) {
                circle.setAttribute('fill', n.color);
                circle.setAttribute('r', NODE_RADIUS + Math.min(a.n_active * 2, 8));
            }
            if (countText) countText.textContent = `${a.n_active}/${a.n_total}`;
            if (edge) edge.setAttribute('class', 'edge active');
        } else {
            if (circle) {
                circle.setAttribute('fill', '#334155');
                circle.setAttribute('r', NODE_RADIUS);
            }
            if (countText) countText.textContent = '';
            if (edge) edge.setAttribute('class', 'edge');
        }
    });
}

function updateCounters(snap) {
    const c = document.getElementById('counters');
    const rolling = snap.source_counts_rolling || {};
    const cumulative = snap.source_counts_cumulative || {};
    const logCounts = snap.log_counts || {};
    const verdicts = rolling._verifier_verdicts || {};
    const rollingTotal = Object.entries(rolling).filter(([k]) => !k.startsWith('_')).reduce((s, [_, v]) => s + (typeof v === 'number' ? v : 0), 0);
    const cumTotal = Object.values(cumulative).reduce((a, b) => a + b, 0);
    const classifierShare = rolling.classifier || 0;
    c.innerHTML = `
        <div class="counter"><div class="label">decisions (10m / today)</div>
            <div class="value">${rollingTotal} <span class="sub">/ ${cumTotal}</span></div>
            <div class="sub">cls:${rolling.classifier||0} learn:${rolling.learned||0} rules:${rolling.rules||0}</div></div>
        <div class="counter"><div class="label">classifier share (rolling)</div>
            <div class="value">${rollingTotal>0 ? Math.round(100*classifierShare/rollingTotal) : 0}%</div>
            <div class="sub">${classifierShare} of ${rollingTotal}</div></div>
        <div class="counter"><div class="label">verifier verdicts (rolling)</div>
            <div class="value">${(verdicts.accept||0)}/${(verdicts.reject||0)}</div>
            <div class="sub">accept / reject</div></div>
        <div class="counter"><div class="label">lock events (today)</div>
            <div class="value">${logCounts.inference_aborted||0}</div>
            <div class="sub">aborts · ${logCounts.slot_erase||0} slot-erases</div></div>
    `;
}

function roleColor(role) {
    if (!topology) return '#64748b';
    const n = topology.nodes.find(n => n.role === role || n.label === role);
    return n ? n.color : '#64748b';
}

function renderTaskCard(t, kind) {
    // kind: 'in_flight' or 'completed'
    const role = t.chosen_action || t.final_role || '?';
    const color = roleColor(role);
    const conf = t.classifier_confidence != null ? ` · conf ${t.classifier_confidence.toFixed(2)}` : '';
    const src = t.decision_source ? ` · <span class="tag" style="background:rgba(255,255,255,0.08)">${t.decision_source}</span>` : '';
    let verifier = '';
    if (t.verifier_p_success != null) {
        verifier = ` · <span class="tag ${t.verifier_verdict||''}">v=${t.verifier_p_success.toFixed(2)} ${t.verifier_verdict||''}</span>`;
    }
    let outcome = '';
    if (kind === 'completed') {
        const okIcon = t.outcome === 'task_completed' ? '✅' : t.outcome === 'task_failed' ? '❌' : '↗';
        outcome = `<span style="float:right;color:var(--muted);font-size:10px">${okIcon} ${t.duration_s}s</span>`;
    } else {
        outcome = `<span style="float:right;color:var(--good);font-size:10px">running ${t.age_s.toFixed(0)}s</span>`;
    }
    return `
        <div class="task-card" onclick="openDetail('${t.task_id}', null, null)">
            <div class="role" style="color:${color}">${role}${outcome}</div>
            <div class="meta">${t.task_id}${conf}${src}${verifier}</div>
            <div class="prompt">${escapeHTML(t.objective || '')}</div>
        </div>
    `;
}

function updateTasks(snap) {
    const inflight = snap.in_flight_tasks || [];
    const completed = snap.recent_completed_tasks || [];
    // Compute slot summary from activity to show alongside chat-task count
    let slotsActive = 0, slotsTotal = 0;
    Object.values(snap.activity || {}).forEach(a => {
        slotsActive += a.n_active || 0;
        slotsTotal += a.n_total || 0;
    });
    document.getElementById('inflight-count').textContent =
        `(${inflight.length} chat · ${slotsActive}/${slotsTotal} slots active)`;
    document.getElementById('completed-count').textContent = `(${completed.length} in last 10m)`;
    const ifEl = document.getElementById('inflight-tasks');
    const cpEl = document.getElementById('completed-tasks');
    if (inflight.length) {
        ifEl.innerHTML = inflight.map(t => renderTaskCard(t, 'in_flight')).join('');
    } else if (slotsActive > 0) {
        ifEl.innerHTML = `<div style="color:var(--warn);font-size:11px">${slotsActive} slot(s) processing on llama-server but no chat-XXX task yet logged to the progress JSONL — this is the ~1-2s flush lag between request arrival and event log.</div>`;
    } else {
        ifEl.innerHTML = '<div style="color:var(--muted);font-size:11px">no tasks in flight — autopilot may be in a between-batch GEPA local-compute phase</div>';
    }
    cpEl.innerHTML = completed.length
        ? completed.slice(0, 12).map(t => renderTaskCard(t, 'completed')).join('')
        : '<div style="color:var(--muted);font-size:11px">no completed tasks in last 10 min</div>';
}

// ---- Inference tap panel (live prompt + response from /mnt/raid0/llm/tmp/) ----
let _tapStream = null;
let _expandedSectionIdx = 0; // 0 = most recent

function renderInferenceTap(data) {
    const status = document.getElementById('tap-status');
    const container = document.getElementById('inference-tap');
    const active = data.tap_active;
    const mtime = data.inference_tap_mtime;
    const ageS = mtime ? Math.max(0, Math.round(data.now ? (data.now - mtime) : (Date.now()/1000 - mtime))) : null;
    status.innerHTML = active
        ? `<span style="color:var(--good)">● tap active</span> · ${ageS!=null?ageS+'s since last write':''}`
        : '<span style="color:var(--bad)">● tap inactive</span>';
    const sections = data.inference_sections || [];
    if (!sections.length && !(data.current_prompt && data.current_prompt.trim())) {
        container.innerHTML = '<div style="color:var(--muted);font-size:11px">no inference activity in the tap file yet</div>';
        return;
    }
    const cur = data.current_prompt || '';
    const liveBlock = cur.trim() ? `
        <div style="margin-bottom:8px">
            <div style="color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em">current prompt</div>
            <div class="tap-content prompt-text" id="tap-current-prompt"></div>
        </div>` : '';
    const sectionsHtml = sections.map((s, i) => {
        const ts = s.timestamp || '';
        const role = s.role || '?';
        const color = roleColor(role) || 'var(--muted)';
        return `
            <details class="tap-section" ${i === 0 ? 'open' : ''}>
                <summary style="cursor:pointer;padding:4px 0;color:${color}">
                    <span style="font-weight:600">${escapeHTML(role)}</span>
                    <span style="color:var(--muted);font-size:10px;margin-left:6px">${escapeHTML(ts)} · ${s.prompt_len}c prompt · ${s.response_len}c resp</span>
                </summary>
                <div style="margin:4px 0">
                    <div style="color:var(--muted);font-size:10px">prompt</div>
                    <div class="tap-content prompt-text" data-md="prompt-${i}"></div>
                    <div style="color:var(--muted);font-size:10px;margin-top:4px">response</div>
                    <div class="tap-content stream-text" data-md="response-${i}"></div>
                </div>
            </details>`;
    }).join('');
    container.innerHTML = liveBlock + sectionsHtml;
    // Render markdown into each block
    if (cur.trim()) {
        renderMarkdownInto(document.getElementById('tap-current-prompt'), cur);
    }
    sections.forEach((s, i) => {
        const pEl = container.querySelector(`[data-md="prompt-${i}"]`);
        const rEl = container.querySelector(`[data-md="response-${i}"]`);
        if (pEl) renderMarkdownInto(pEl, s.prompt);
        if (rEl) renderMarkdownInto(rEl, s.response);
    });
}

function startTapStream() {
    if (_tapStream) _tapStream.close();
    _tapStream = new EventSource('/dashboard/events/inference_tap');
    _tapStream.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            data.now = Date.now() / 1000;
            renderInferenceTap(data);
        } catch {}
    };
    _tapStream.onerror = () => {
        setTimeout(startTapStream, 2000);
    };
    // Initial fetch for immediate content
    fetch('/dashboard/api/inference_tap').then(r => r.json()).then(data => {
        renderInferenceTap(data);
    }).catch(() => {});
}

function updateDecisions(snap) {
    const feed = document.getElementById('decision-feed');
    const rows = (snap.recent_decisions || []).slice(-20).reverse();
    feed.innerHTML = rows.map(d => {
        const conf = d.classifier_confidence != null ? d.classifier_confidence.toFixed(2) : '—';
        const verifier = d.verifier_p_success != null
            ? `<span class="tag ${d.verifier_verdict||''}">${d.verifier_verdict||'?'} ${d.verifier_p_success.toFixed(2)}${d.verifier_shadow?' s':''}</span>`
            : '—';
        return `<div class="decision-row">
            <div>${d.age_s}s</div>
            <div title="${d.task_id||''}">${(d.task_id||'').slice(-12)}</div>
            <div>${d.chosen_action||'—'}</div>
            <div class="src ${d.source}">${d.source}</div>
            <div>${conf}</div>
            <div>${verifier}</div>
        </div>`;
    }).join('') || '<div style="color:var(--muted);padding:6px 0">no decisions yet</div>';
}

async function updateGepa() {
    try {
        const r = await fetch('/dashboard/api/gepa');
        const d = await r.json();
        const el = document.getElementById('gepa-lines');
        const st = d.state || {};
        let header = '';
        if (st.current_trial) {
            const prog = st.baseline_sentinels_total
                ? `<span style="color:var(--accent)">${st.sentinels_completed||0}/${st.baseline_sentinels_total}</span>`
                : '';
            const score = st.baseline_score != null
                ? ` · baseline=<span style="color:var(--good)">${st.baseline_score.toFixed(3)}</span>`
                : ' · evaluating…';
            header = `
                <div style="background:rgba(59,130,246,0.08); padding:6px 10px; border-radius:4px; margin-bottom:8px; border-left:3px solid var(--accent)">
                    <div style="font-weight:600">Trial ${st.current_trial} · ${st.current_action || '?'} · ${st.current_file || '?'}</div>
                    <div style="font-size:11px;color:var(--muted);margin-top:2px">sentinels ${prog}${score} · last: ${st.last_event || 'n/a'}</div>
                </div>`;
        }
        let trialsTable = '';
        if (d.recent_trials && d.recent_trials.length) {
            trialsTable = '<div style="margin:8px 0 4px 0;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em">last 10 trials (pareto trajectory)</div>';
            trialsTable += '<div style="font-size:10px">';
            trialsTable += '<div style="display:grid;grid-template-columns:48px 80px 60px 70px 60px 60px 1fr;gap:6px;color:var(--muted);padding:2px 0;border-bottom:1px solid var(--border)">' +
                '<div>trial</div><div>species</div><div>quality</div><div>speed</div><div>cost</div><div>pareto</div><div>desc</div></div>';
            d.recent_trials.slice(-10).reverse().forEach(t => {
                const paretoColor = t.pareto_status === 'pareto_frontier' ? '#34d399' :
                                     t.pareto_status === 'dominated' ? '#94a3b8' : '#fcd34d';
                trialsTable += `<div style="display:grid;grid-template-columns:48px 80px 60px 70px 60px 60px 1fr;gap:6px;padding:1px 0">
                    <div>${t.trial_id||'-'}</div>
                    <div>${(t.species||'').slice(0,12)}</div>
                    <div>${t.quality!=null?t.quality.toFixed(2):'-'}</div>
                    <div>${t.speed!=null?t.speed.toFixed(1):'-'}</div>
                    <div>${t.cost!=null?t.cost.toFixed(2):'-'}</div>
                    <div style="color:${paretoColor}">${(t.pareto_status||'').replace('pareto_','')}</div>
                    <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escapeHTML(t.description||'')}">${escapeHTML((t.description||'').slice(0,80))}</div>
                </div>`;
            });
            trialsTable += '</div>';
        }
        let recentLines = '';
        if (d.lines && d.lines.length) {
            recentLines = '<div style="margin-top:8px;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em">log tail</div>';
            recentLines += d.lines.slice(-8).map(l => {
                const bold = /Trial \d+|baseline|sentinel/.test(l) ? 'bold' : '';
                return `<div class="gepa-line ${bold}">${escapeHTML(l.slice(-180))}</div>`;
            }).join('');
        }
        const all = header + trialsTable + recentLines;
        el.innerHTML = all || '<div style="color:var(--muted)">no recent GEPA activity</div>';
    } catch (e) {}
}

function escapeHTML(s) {
    return (s || '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

// Configure marked to use highlight.js for code blocks (once on first call).
let _markedConfigured = false;
function configureMarked() {
    if (_markedConfigured || !window.marked) return;
    try {
        if (window.markedHighlight && window.hljs) {
            marked.use(markedHighlight.markedHighlight({
                langPrefix: 'hljs language-',
                highlight(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language, ignoreIllegals: true }).value;
                },
            }));
        }
        marked.use({ breaks: true, gfm: true });
        _markedConfigured = true;
    } catch (e) {}
}

function renderMarkdownInto(el, rawText) {
    configureMarked();
    if (window.marked) {
        try {
            el.innerHTML = marked.parse(rawText || '');
        } catch {
            el.textContent = rawText || '';
        }
    } else {
        el.textContent = rawText || '';
    }
    if (window.renderMathInElement) {
        try {
            renderMathInElement(el, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "\\[", right: "\\]", display: true},
                    {left: "\\(", right: "\\)", display: false},
                    {left: "$", right: "$", display: false},
                ],
                throwOnError: false,
            });
        } catch (e) {}
    }
}

let _currentTaskRawText = { taskId: null, prompt: '', stream: '', repl: [] };

function buildTextSnapshot() {
    const c = _currentTaskRawText;
    const lines = [];
    lines.push(`=== Task ${c.taskId} @ ${new Date().toISOString()} ===`);
    lines.push('');
    lines.push('PROMPT:');
    lines.push('-------');
    lines.push(c.prompt || '(not available)');
    lines.push('');
    lines.push('INFERENCE STREAM:');
    lines.push('-----------------');
    lines.push(c.stream || '(empty)');
    lines.push('');
    lines.push(`REPL HISTORY (${c.repl.length} events):`);
    lines.push('-----------------');
    c.repl.forEach(ev => {
        const ts = (ev.timestamp || '').replace('T', ' ').slice(11, 19);
        lines.push(`[${ts}] ${ev.event_type}: ${JSON.stringify(ev.data)}`);
    });
    return lines.join('\n');
}

async function copySnapshotToClipboard() {
    // Prefer the server-side filtered .txt endpoint (it strips stack_state
    // and other noisy keys). Fall back to the local builder if the fetch fails.
    let text;
    try {
        const r = await fetch(`/dashboard/api/task/${encodeURIComponent(_currentTaskRawText.taskId)}.txt`);
        if (r.ok) {
            text = await r.text();
        } else {
            text = buildTextSnapshot();
        }
    } catch {
        text = buildTextSnapshot();
    }
    try {
        await navigator.clipboard.writeText(text);
        const fb = document.getElementById('copy-feedback');
        fb.textContent = `copied ${text.length}c`;
        fb.classList.add('shown');
        setTimeout(() => fb.classList.remove('shown'), 1500);
    } catch (e) {
        alert('clipboard write failed: ' + e.message);
    }
}

function downloadSnapshot() {
    const text = buildTextSnapshot();
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `task_${_currentTaskRawText.taskId}_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

async function openDetail(taskId, port, slotId) {
    if (taskId === null || taskId === 'null' || taskId === undefined) {
        return;
    }
    const panel = document.getElementById('detail-panel');
    const content = document.getElementById('detail-content');
    document.getElementById('detail-title').textContent = `task ${taskId}`;
    content.innerHTML = '<div style="color:var(--muted)">loading…</div>';
    panel.classList.add('open');
    try {
        const r = await fetch(`/dashboard/api/task/${encodeURIComponent(taskId)}`);
        const d = await r.json();
        let promptText = '';
        let initialContent = '';
        if (d.slot) {
            promptText = (d.slot.prompt || '').toString();
            initialContent = (d.slot.content || '').toString();
        } else {
            const startedEv = d.events.find(e => e.event_type === 'task_started');
            if (startedEv) {
                promptText = startedEv.data.objective || '';
            }
        }
        _currentTaskRawText = {
            taskId,
            prompt: promptText,
            stream: initialContent,
            repl: d.events,
        };
        content.innerHTML = `
            <div class="detail-actions">
                <button onclick="copySnapshotToClipboard()">📋 copy snapshot</button>
                <button class="secondary" onclick="downloadSnapshot()">⬇ download .txt</button>
                <button class="secondary" onclick="window.open('/dashboard/api/task/${encodeURIComponent(taskId)}.txt','_blank')">↗ open as text</button>
                <span id="copy-feedback" class="copy-feedback"></span>
            </div>
            <h3>prompt</h3>
            <div class="prompt-text" id="prompt-text"></div>
            <h3>live inference</h3>
            <div class="stream-text live" id="stream-text"></div>
            <h3>task events (${d.events.length}) <span style="font-weight:normal;color:var(--muted);font-size:10px">orchestrator-side; not REPL turns</span></h3>
            <div id="repl-history"></div>
        `;
        renderMarkdownInto(document.getElementById('prompt-text'), promptText || '*(prompt not available)*');
        renderMarkdownInto(document.getElementById('stream-text'), initialContent);
        const replEl = document.getElementById('repl-history');
        replEl.innerHTML = d.events.map(ev => `
            <div class="repl-event">
                <span class="ts">${(ev.timestamp || '').replace('T', ' ').slice(11, 19)}</span>
                <span class="ev">${ev.event_type}</span>
                <span>${escapeHTML(JSON.stringify(ev.data).slice(0, 180))}</span>
            </div>
        `).join('');
        // Start live stream
        if (detailStream) { detailStream.close(); detailStream = null; }
        detailStream = new EventSource(`/dashboard/events/task/${encodeURIComponent(taskId)}`);
        detailStream.onmessage = (e) => {
            const u = JSON.parse(e.data);
            const sel = document.getElementById('stream-text');
            if (!sel) return;
            if (u.done) {
                sel.classList.remove('live');
                detailStream.close();
                detailStream = null;
                return;
            }
            if (u.delta) {
                _currentTaskRawText.stream += u.delta;
                // Re-render after every batch — cheap for short streams
                renderMarkdownInto(sel, _currentTaskRawText.stream);
                sel.classList.add('live');
            }
        };
    } catch (e) {
        content.innerHTML = `<div style="color:var(--bad)">error: ${e.message}</div>`;
    }
}

function closeDetail() {
    document.getElementById('detail-panel').classList.remove('open');
    if (detailStream) { detailStream.close(); detailStream = null; }
}

function startStream() {
    if (activeStream) activeStream.close();
    activeStream = new EventSource('/dashboard/events/stream');
    document.getElementById('status').textContent = 'connecting…';
    activeStream.onopen = () => { document.getElementById('status').textContent = 'live'; };
    activeStream.onmessage = (e) => {
        try {
            const snap = JSON.parse(e.data);
            if (snap.error) { document.getElementById('status').textContent = `error: ${snap.error}`; return; }
            updateTopology(snap.activity || {});
            updateCounters(snap);
            updateTasks(snap);
            updateDecisions(snap);
        } catch (err) {
            console.error(err);
        }
    };
    activeStream.onerror = () => {
        document.getElementById('status').textContent = 'reconnecting…';
        setTimeout(() => startStream(), 2000);
    };
}

async function updateProcessStatus() {
    try {
        const r = await fetch('/dashboard/api/process_status');
        const d = await r.json();
        const ap = d.autopilot || {};
        const el = document.getElementById('process-status');
        if (ap.running) {
            const workers = d.gepa_worker_count ? ` · ${d.gepa_worker_count} gepa workers` : '';
            const ageLabel = d.last_autopilot_log_age_s != null
                ? ` · last log ${Math.round(d.last_autopilot_log_age_s)}s ago`
                : '';
            el.innerHTML = `<span style="color:var(--good)">● autopilot up</span>
                <span style="color:var(--muted)">pid ${ap.pid} · ${ap.etime}${workers}${ageLabel}</span>`;
        } else {
            el.innerHTML = `<span style="color:var(--bad)">● autopilot DOWN</span>
                <span style="color:var(--muted)">no process matching "autopilot.py start"</span>`;
        }
    } catch (e) {}
}

(async function init() {
    await loadTopology();
    startStream();
    startTapStream();
    updateGepa();
    updateProcessStatus();
    setInterval(updateGepa, 5000);
    setInterval(updateProcessStatus, 3000);
})();
</script>
</body></html>
"""


@router.get("/dashboard")
async def dashboard_page() -> HTMLResponse:
    """Serve the single-page dashboard."""
    return HTMLResponse(_DASHBOARD_HTML)
