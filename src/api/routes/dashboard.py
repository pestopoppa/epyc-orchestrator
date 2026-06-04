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
import statistics
import time
from collections import deque
from datetime import datetime, date
from itertools import combinations
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
    _INFERENCE_TAP_EVENTS_PATH,
    _INFERENCE_TAP_PATH,
    _PROMPT_TAP_PATH,
    _REPL_TAP_PATH,
    _SECTION_SEP,
    _SUBSECTION_SEP,
    _TAP_SENTINEL_PATH,
    _parse_inference_sections,
    _parse_structured_tap_requests,
    _parse_trial_state,
    _read_tail,
)
from src.api.routes.dashboard_tasks import (
    _find_section_by_objective,
    _find_structured_request_by_id,
    _find_structured_request_by_task_id,
    _objective_for_task,
    _task_events,
    _task_text_snapshot,
)
from src.api.routes.dashboard_topology import (
    _PORT_HINTS,
    role_aliases,
    _ROLE_COLORS,
    _clean_model_name,
    _discover_llama_models,
    _discover_llama_ports,
    _process_info_by_match,
    _role_color,
    base_role,
)
from src.api.routes.dashboard_topology import _load_state_services as _load_state_services_impl
from src.autopilot_core.tier_specs import DEFAULT_FRONTIER_TIER, MIN_FRONTIER_EVAL_TIER, spec_for

logger = logging.getLogger(__name__)
router = APIRouter()

ORCHESTRATOR_LOG_DIR = Path("/mnt/raid0/llm/epyc-orchestrator/logs")
PROGRESS_LOG_DIR = ORCHESTRATOR_LOG_DIR / "progress"
AUTOPILOT_LOG = ORCHESTRATOR_LOG_DIR / "autopilot.log"
AUTOPILOT_PHASE_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.json")
ORCHESTRATOR_STATE_PATH = ORCHESTRATOR_LOG_DIR / "orchestrator_state.json"


def _load_state_services() -> list[dict[str, Any]]:
    """Wrapper supplying ORCHESTRATOR_STATE_PATH to dashboard_topology helper."""
    return _load_state_services_impl(ORCHESTRATOR_STATE_PATH)


def _read_autopilot_phase() -> dict[str, Any]:
    try:
        data = json.loads(AUTOPILOT_PHASE_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    structured_tail = _read_tail(_INFERENCE_TAP_EVENTS_PATH, max_bytes=1024 * 1024)
    now_epoch = time.time()
    structured_requests = _parse_structured_tap_requests(
        structured_tail,
        max_requests=max_sections,
        now_epoch=now_epoch,
    )
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
        "structured_requests": structured_requests,
        "inference_tap_mtime": mtime(_INFERENCE_TAP_PATH),
        "structured_tap_mtime": mtime(_INFERENCE_TAP_EVENTS_PATH),
        "repl_tail": repl_tail,
        "repl_tap_mtime": mtime(_REPL_TAP_PATH),
        "now": now_epoch,
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
        # The ConcurrencyAwareBackend (which holds migration counters / placement state) lives on
        # the REAL-inference primitives (state._real_primitives, lazily built by _init_primitives
        # with the full:-prefixed server_urls), NOT the startup state.llm_primitives (constructed
        # without server_urls → no backends). Prefer the real one so per_role_scheduling actually
        # reflects live CAB placement/migrations (J2/J3 observability fix, 2026-05-27).
        primitives = (getattr(request.app.state, "_real_primitives", None)
                      or getattr(request.app.state, "llm_primitives", None))
        if primitives is not None:
            for role, backend in getattr(primitives, "_backends", {}).items():
                if not hasattr(backend, "_quarter_preference_order"):
                    continue
                per_role[role] = {
                    "quarter_preference_order": list(getattr(backend, "_quarter_preference_order", [])),
                    "migrations_started": int(getattr(backend, "_migrations", 0)),
                    "migration_failures": int(getattr(backend, "_migration_failures", 0)),
                    # WP-4 reverse migration (quarter→full on load drop) — surfaced for J2/J3
                    # observation (forward = migrations_started above; this is the reverse leg).
                    "reverse_migrations": int(
                        sum((getattr(backend, "_reverse_migration_counts", {}) or {}).values())
                    ),
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


# ── region_locks helpers (module-scope so tests can import) ─────────────

def _shape_for_regions(regs: "frozenset[str] | set[str] | list[str]") -> str:
    """Canonical column-bucket name for an instance covering `regs`.

    Returns one of: "full" (all 4 quarters), "half0" (q0+q1), "half1"
    (q2+q3), "q0".."q3" (single quarter), or "+".join(sorted(regs)) as
    a fallback for exotic shapes not yet in NUMA_CONFIG.
    """
    rs = set(regs)
    if rs == {"q0", "q1", "q2", "q3"}:
        return "full"
    if rs == {"q0", "q1"}:
        return "half0"
    if rs == {"q2", "q3"}:
        return "half1"
    if len(rs) == 1:
        return next(iter(rs))
    return "+".join(sorted(rs))


def _panel_shapes_from_matrix(sr, primary_shape: str) -> set[str]:
    """Canonical shapes a role's region-locks-panel row should display, per
    the contention matrix `same_role` entry (the operator-blessed source of
    truth for which within-role shapes are usable).

    Strict interpretation:
      - sr is None                       → empty set (role not in matrix → hide)
      - sr has no instance_pairs         → {primary_shape} (single-instance or
                                            'primary only' roles like ingest)
      - sr has instance_pairs            → union of all a/b labels, with the
                                            matrix's "full" alias translated to
                                            the role's primary shape

    `primary_shape` is the canonical shape of the role's idx=0 NUMA_CONFIG
    instance ("full", "half0", "half1", or "qN") — i.e. what the matrix means
    when it says "full" for that specific role.
    """
    if sr is None:
        return set()
    if not sr.instance_pairs:
        return {primary_shape}
    out: set[str] = set()
    for ip in sr.instance_pairs:
        for label in (ip.a, ip.b):
            out.add(primary_shape if label == "full" else label)
    return out


def _resolve_pid_to_instance_idx(
    role_pid_regions: dict[str, dict[str, set[str]]],
    role_instances_by_regions: dict[str, dict["frozenset[str]", int]],
) -> dict[tuple[str, str], int]:
    """Map each (role, pid) to its NUMA_CONFIG instance idx by matching the
    set of regions the PID currently holds against the role's instance
    region-sets.

    Each NUMA_CONFIG instance has a unique region set within a role, so
    holding {q0,q1} unambiguously identifies the half0 instance, etc.
    Locks are acquired by orchestrator uvicorn workers (not llama-server
    processes), so this region-set-based lookup is the only reliable
    way to attribute holders to instances.
    """
    out: dict[tuple[str, str], int] = {}
    for role, pid_regs in role_pid_regions.items():
        region_map = role_instances_by_regions.get(role) or {}
        for pid, regs in pid_regs.items():
            idx = region_map.get(frozenset(regs))
            if idx is not None:
                out[(role, pid)] = idx
    return out


def _region_lock_blocked_by_roles(
    row_role: str,
    active_roles: set[str],
    matrix: Any | None,
) -> list[str]:
    """Roles whose active locks should render as wait/blocking for row_role.

    Same-role overlaps are always real lock waits because the lock files are
    keyed by (role, region). Cross-role overlaps follow the contention matrix:
    allow means concurrent dispatch is expected, queue/block means the row role
    would wait behind that holder for background/autopilot traffic.
    """
    blocked: set[str] = set()
    for holder_role in active_roles:
        if not holder_role:
            continue
        if holder_role == row_role:
            blocked.add(holder_role)
            continue
        if matrix is None:
            blocked.add(holder_role)
            continue
        try:
            from src.scheduling.contention import PairDecision, TrafficClass, pair_policy

            decision = pair_policy(
                row_role,
                holder_role,
                TrafficClass.BACKGROUND,
                matrix=matrix,
            )
        except Exception:
            blocked.add(holder_role)
            continue
        if decision != PairDecision.ALLOW:
            blocked.add(holder_role)
    return sorted(blocked)


@router.get("/dashboard/api/region_locks")
async def region_locks_snapshot() -> JSONResponse:
    """Per-CPU-region lock state — which (role, region) lock files are
    currently held, and by which PIDs.

    Built from /proc/locks scan of the orchestrator's lock files, plus
    the static topology from `instance_topology.get_instance_regions()`
    so the panel surfaces ALL roles that *have* quartered instances —
    not just the ones whose dispatch path created a lock file. Roles
    with quarter instances but no lock files are tagged
    ``wired=False`` so the dashboard can render them greyed out with
    an explanatory badge (they were quartered at launch but their
    backend never acquires `cpu_region_lock`, so cross-process
    concurrency isn't actually enforced for them yet).
    """
    import os
    from pathlib import Path

    try:
        from src.runtime.cpu_region_lock import _tmp_dir, _current_lock_owner_pids
        tmp_dir = _tmp_dir()
    except Exception:
        tmp_dir = Path("/mnt/raid0/llm/tmp")
        from src.runtime.cpu_region_lock import _current_lock_owner_pids

    # Pull the live (role, idx) → {regions} map. Anything with >1 instance OR
    # an instance pinned to a strict subset of {q0..q3} is a candidate for
    # region-locking — those are the rows the panel should always show.
    try:
        from src.runtime.instance_topology import get_instance_regions, ATOMIC_REGIONS
        topology = get_instance_regions()
        all_regions = list(ATOMIC_REGIONS)
    except Exception:
        topology = {}
        all_regions = ["q0", "q1", "q2", "q3"]

    # Source of truth for which roles/shapes appear in the panel: the
    # operator-curated contention matrix (`orchestration/contention_matrix.yaml`).
    # A role appears iff it has a `same_role` entry. Its visible shapes come
    # from `_panel_shapes_from_matrix()` (strict: union of a/b labels in
    # instance_pairs, with "full" → role's primary idx=0 shape; n/a or no
    # pairs → only the primary shape). Lock-file existence is NO LONGER used
    # to gate "wiring" — it's just a runtime hot/cold signal (cells stay
    # ✅ Ready until the backend first acquires a lock).
    matrix = None
    try:
        from src.scheduling.contention import load_contention_matrix
        matrix = load_contention_matrix()
    except Exception:
        matrix = None  # fail-open: panel still renders with NUMA-only fallback below

    # Build instance_topology from NUMA_CONFIG, filtered to matrix-allowed shapes.
    instance_topology_all: dict[str, list[dict[str, Any]]] = {}
    for (role, idx), regs in topology.items():
        if not regs:
            continue
        instance_topology_all.setdefault(role, []).append({
            "idx": idx,
            "regions": sorted(regs),
            "span": len(regs),
            "shape": _shape_for_regions(regs),
            "is_full": len(regs) >= 2,  # deprecated
        })
    for role in instance_topology_all:
        instance_topology_all[role].sort(key=lambda x: (-x["span"], x["regions"]))

    # Determine the panel rows + per-role allowed shapes.
    panel_roles: set[str] = set()
    role_allowed_shapes: dict[str, set[str]] = {}
    if matrix is not None and matrix.same_role:
        for role, sr in matrix.same_role.items():
            insts = instance_topology_all.get(role) or []
            # primary = idx=0 (NUMA_CONFIG convention); fall back to span-sort first.
            primary = next((i for i in insts if i["idx"] == 0), insts[0] if insts else None)
            if primary is None:
                continue  # matrix mentions a role we have no NUMA_CONFIG for
            allowed = _panel_shapes_from_matrix(sr, primary["shape"])
            if not allowed:
                continue
            role_allowed_shapes[role] = allowed
            panel_roles.add(role)
    else:
        # Fallback: matrix missing/unreadable → fall back to the prior
        # NUMA_CONFIG-based behavior (every multi-instance + multi-region role).
        for role, insts in instance_topology_all.items():
            if len(insts) >= 2:
                panel_roles.add(role)
                role_allowed_shapes[role] = {i["shape"] for i in insts}

    # Filter the per-role instance list to just the matrix-allowed shapes.
    instance_topology: dict[str, list[dict[str, Any]]] = {}
    for role in panel_roles:
        allowed = role_allowed_shapes[role]
        instance_topology[role] = [
            i for i in (instance_topology_all.get(role) or [])
            if i["shape"] in allowed
        ]

    # Per-role: held-region-set → instance idx, used to attribute lock holders
    # to a specific instance after we scan /proc/locks. Resolve against the
    # full NUMA topology, not only matrix-visible panel shapes: a runtime holder
    # can be on a shape the matrix omits for idle display, and unresolved holders
    # make the panel report "no locks held" while regions are visibly occupied.
    role_instances_by_regions: dict[str, dict[frozenset[str], int]] = {
        role: {frozenset(inst["regions"]): inst["idx"] for inst in (instance_topology_all.get(role) or [])}
        for role in panel_roles
    }

    # Pass 1a: collect raw lock-holder PIDs per (role, region) for matrix-included
    # roles only. Locks are acquired by orchestrator uvicorn workers (not by
    # llama-server processes) so the only reliable identification is the SET of
    # regions a worker is currently holding.
    out: list[dict[str, Any]] = []
    role_pid_regions: dict[str, dict[str, set[str]]] = {}  # role → pid → held regions
    try:
        for p in sorted(tmp_dir.glob("cpu_region.*.lock")):
            stem = p.stem  # "cpu_region.<role>.<region>"
            parts = stem.split(".", 2)
            if len(parts) < 3:
                continue
            _prefix, role, region = parts[0], parts[1], parts[2]
            if role not in panel_roles:
                continue  # role isn't in the matrix → don't surface lock state for it
            holders = _current_lock_owner_pids(p)
            out.append({
                "role": role,
                "region": region,
                "lock_path": str(p),
                "holder_pids": holders,
                "holder_instance_idxs": [],  # populated in Pass 1b
                "held": bool(holders),
                "wired": True,
            })
            if holders:
                role_pid_regions.setdefault(role, {})
                for pid in holders:
                    role_pid_regions[role].setdefault(pid, set()).add(region)
    except Exception as exc:
        return JSONResponse({"error": str(exc), "entries": []}, status_code=200)

    # Pass 1b: resolve held-region SET per PID to a NUMA_CONFIG instance idx.
    pid_to_instance = _resolve_pid_to_instance_idx(role_pid_regions, role_instances_by_regions)
    for entry in out:
        if not entry["holder_pids"]:
            continue
        idxs = {pid_to_instance[(entry["role"], pid)]
                for pid in entry["holder_pids"]
                if (entry["role"], pid) in pid_to_instance}
        entry["holder_instance_idxs"] = sorted(idxs)

    visible_instances: dict[str, list[dict[str, Any]]] = {
        role: list(insts) for role, insts in instance_topology.items()
    }
    for (role, _pid), idx in pid_to_instance.items():
        insts = visible_instances.setdefault(role, [])
        if any(inst["idx"] == idx for inst in insts):
            continue
        runtime_inst = next(
            (inst for inst in (instance_topology_all.get(role) or []) if inst["idx"] == idx),
            None,
        )
        if runtime_inst is not None:
            insts.append({**runtime_inst, "runtime_only": True})
    for insts in visible_instances.values():
        insts.sort(key=lambda x: (-x["span"], x["regions"]))

    # Pass 2: ensure every panel role has a row even when no lock file exists
    # yet on disk (a wired-but-never-dispatched role like vision_escalation
    # before its first request). Synthesize empty (free) region entries for
    # all 4 quarters so the frontend can render ✅ Ready cells.
    seen_role_regions = {(e["role"], e["region"]) for e in out}
    for role in sorted(panel_roles):
        for region in all_regions:
            if (role, region) in seen_role_regions:
                continue
            out.append({
                "role": role,
                "region": region,
                "lock_path": str(tmp_dir / f"cpu_region.{role}.{region}.lock"),
                "holder_pids": [],
                "holder_instance_idxs": [],
                "held": False,
                "wired": True,   # matrix-membership = wired; lock files appear lazily
            })

    # Group by role for easier dashboard rendering.
    by_role: dict[str, dict[str, Any]] = {}
    for entry in out:
        bucket = by_role.setdefault(entry["role"], {
            "wired": True,
            "regions": [],
            "instances": visible_instances.get(entry["role"], []),
            "active_instance_idxs": set(),
            "blocked_by_roles": [],
        })
        bucket["regions"].append({
            "region": entry["region"],
            "held": entry["held"],
            "holder_pids": entry["holder_pids"],
            "holder_instance_idxs": entry["holder_instance_idxs"],
        })
        for idx in entry["holder_instance_idxs"]:
            bucket["active_instance_idxs"].add(idx)

    active_lock_roles = {
        role
        for role, bucket in by_role.items()
        if bucket["active_instance_idxs"]
        or any(region.get("held") for region in bucket.get("regions", []))
    }
    for role, bucket in by_role.items():
        bucket["active_instance_idxs"] = sorted(bucket["active_instance_idxs"])
        bucket["blocked_by_roles"] = _region_lock_blocked_by_roles(
            role,
            active_lock_roles,
            matrix,
        )

    feature_flag = os.environ.get("ORCHESTRATOR_PER_REGION_LOCKS", "0").strip()
    return JSONResponse({
        "per_region_locks_enabled": feature_flag in {"1", "true", "yes", "on"},
        "matrix_loaded": matrix is not None,
        "tmp_dir": str(tmp_dir),
        "entries": out,
        "by_role": by_role,
        "topology_quartered_roles": sorted(panel_roles),  # back-compat field
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


@router.get("/dashboard/events/structured_tap")
async def structured_tap_stream(request: Request) -> StreamingResponse:
    """SSE stream of request-grouped structured tap snapshots."""

    async def event_gen():
        last_mtime = -1.0
        last_emit = 0.0
        while True:
            if await request.is_disconnected():
                return
            try:
                mtime = (
                    _INFERENCE_TAP_EVENTS_PATH.stat().st_mtime
                    if _INFERENCE_TAP_EVENTS_PATH.exists()
                    else 0.0
                )
            except Exception:
                mtime = 0.0
            now_epoch = time.time()
            # Also repaint open requests as they age into "quiet"; waiting
            # only for mtime changes leaves a silent running request looking
            # green forever until another tap event arrives.
            if mtime != last_mtime or (now_epoch - last_emit) >= 2.0:
                last_mtime = mtime
                last_emit = now_epoch
                tail = _read_tail(_INFERENCE_TAP_EVENTS_PATH, max_bytes=1024 * 1024)
                payload = json.dumps({
                    "tap_active": _TAP_SENTINEL_PATH.exists(),
                    "structured_requests": _parse_structured_tap_requests(
                        tail,
                        max_requests=40,
                        now_epoch=now_epoch,
                    ),
                    "structured_tap_mtime": mtime or None,
                    "now": now_epoch,
                })
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
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
        last_mtimes = {"inference": 0.0, "structured": 0.0, "prompt": 0.0, "repl": 0.0}
        while True:
            if await request.is_disconnected():
                return
            try:
                inf_m = _INFERENCE_TAP_PATH.stat().st_mtime if _INFERENCE_TAP_PATH.exists() else 0.0
                structured_m = (
                    _INFERENCE_TAP_EVENTS_PATH.stat().st_mtime
                    if _INFERENCE_TAP_EVENTS_PATH.exists()
                    else 0.0
                )
                prm_m = _PROMPT_TAP_PATH.stat().st_mtime if _PROMPT_TAP_PATH.exists() else 0.0
                rpl_m = _REPL_TAP_PATH.stat().st_mtime if _REPL_TAP_PATH.exists() else 0.0
            except Exception:
                inf_m = structured_m = prm_m = rpl_m = 0.0
            changed = (
                inf_m > last_mtimes["inference"]
                or structured_m > last_mtimes["structured"]
                or prm_m > last_mtimes["prompt"]
                or rpl_m > last_mtimes["repl"]
            )
            if changed:
                last_mtimes = {
                    "inference": inf_m,
                    "structured": structured_m,
                    "prompt": prm_m,
                    "repl": rpl_m,
                }
                # Build the payload — same shape as the snapshot endpoint
                inference_tail = _read_tail(_INFERENCE_TAP_PATH, max_bytes=256 * 1024)
                sections = _parse_inference_sections(inference_tail, max_sections=10)
                structured_tail = _read_tail(_INFERENCE_TAP_EVENTS_PATH, max_bytes=1024 * 1024)
                now_epoch = time.time()
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
                    "structured_requests": _parse_structured_tap_requests(
                        structured_tail,
                        max_requests=10,
                        now_epoch=now_epoch,
                    ),
                    "inference_tap_mtime": inf_m,
                    "structured_tap_mtime": structured_m,
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
    phase = _read_autopilot_phase()
    phase_age_s = _age_s(AUTOPILOT_PHASE_PATH)
    if phase and not (autopilot or {}).get("running"):
        phase = dict(phase)
        phase.setdefault("idle_reason", "autopilot process not running")

    return JSONResponse({
        "autopilot": autopilot,
        "gepa_worker_count": n_workers,
        "last_autopilot_log_age_s": last_log_age_s,
        "autopilot_recent_lines": recent_lines,
        "autopilot_phase": phase,
        "autopilot_phase_age_s": phase_age_s,
        "inference_tap_age_s": _age_s(_INFERENCE_TAP_PATH),
        "planner_tap_age_s": _age_s(Path("/mnt/raid0/llm/tmp/planner_tap.log")),
    }, headers=_NO_STORE_HEADERS)


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
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}

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
_AUTOPILOT_LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
_PARETO_REFERENCE_POINT = (0.0, 0.0, -1.0, 0.0)


def _parse_journal_ts(value: Any) -> float | None:
    """Parse a journal `timestamp` value (ISO-8601 string or unix float) to unix seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            from datetime import datetime
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    return None


def _pareto_objectives_from_journal(entry: dict[str, Any]) -> list[float] | None:
    try:
        tier = int(entry.get("tier", DEFAULT_FRONTIER_TIER))
    except (TypeError, ValueError):
        tier = DEFAULT_FRONTIER_TIER
    objectives = spec_for(tier).objectives_from_row(entry)
    if objectives is None:
        return None
    return list(objectives[:4])


# Trusted within-noise exclusion reasons (recorded in eval_details.learning_exclusion.by).
# These are NOT corruption — they are Pareto-admissible as robust-median representatives,
# mirroring autopilot.BENIGN_LEARNING_EXCLUSIONS / ParetoArchive.upsert_representative.
_WITHIN_NOISE_EXCL = {"mad_noise", "reproduction_confirmed"}


def _config_fingerprint_from_row(row: dict[str, Any]) -> str:
    """Mirror autopilot._config_fingerprint for a journal row: keyed on the FULL action
    signature (the journaled config_snapshot, falling back to the reasoning string which also
    stores the action JSON). The action — not just the flag set — determines the outcome, so
    genuine reproductions (same action+params) cluster into ONE robust-median representative
    while distinct interventions stay distinct."""
    cfg = row.get("config_snapshot")
    if not cfg:
        try:
            cfg = json.loads(row.get("reasoning") or "{}")
        except Exception:
            cfg = {}
    try:
        return json.dumps(cfg, sort_keys=True, default=str)
    except Exception:
        return str(cfg)


def _pareto_dominates(a: list[float], b: list[float]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _pareto_hypervolume(points: list[list[float]]) -> float:
    """Compute the same 4D max-objective hypervolume used by autopilot.

    Dashboard sessions normally have a small frontier. If it grows beyond the
    exact inclusion-exclusion range, return a deterministic coarse estimate so
    the trend still moves without making the dashboard endpoint expensive.
    """
    valid = [
        p for p in points
        if len(p) >= 4 and all(pi > ri for pi, ri in zip(p[:4], _PARETO_REFERENCE_POINT))
    ]
    if not valid:
        return 0.0
    if len(valid) > 100:
        # Deterministic approximation over a small lattice. This branch should
        # be rare for a current autopilot session, but prevents pathological
        # request latency if the frontier becomes very large.
        maxes = [max(p[d] for p in valid) for d in range(4)]
        spans = [max(0.0, maxes[d] - _PARETO_REFERENCE_POINT[d]) for d in range(4)]
        total_box = 1.0
        for span in spans:
            total_box *= span
        if total_box <= 0.0:
            return 0.0
        dominated = 0
        samples = 4096
        for i in range(samples):
            coords = []
            n = i
            for d in range(4):
                n = (1103515245 * n + 12345 + d) & 0x7fffffff
                frac = (n % 1024) / 1023.0
                coords.append(_PARETO_REFERENCE_POINT[d] + frac * spans[d])
            if any(all(p[d] >= coords[d] for d in range(4)) for p in valid):
                dominated += 1
        return total_box * (dominated / samples)

    total = 0.0
    for size in range(1, len(valid) + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(valid, size):
            volume = 1.0
            for dim in range(4):
                volume *= max(
                    0.0,
                    min(point[dim] for point in subset) - _PARETO_REFERENCE_POINT[dim],
                )
            total += sign * volume
    return total


def _shape_pareto_entry(entry: dict[str, Any]) -> dict[str, Any]:
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
        "eval_tier": entry.get("eval_tier", entry.get("tier", DEFAULT_FRONTIER_TIER)),
        "speed_deinflated": bool(entry.get("speed_deinflated", False)),
    }


def _latest_journal_run_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the latest contiguous journal segment after a trial-id reset.

    `autopilot_journal.jsonl` is append-only across stack/campaign restarts.
    The current run is identifiable by the last point where trial_id decreases
    (for example old trial 569 followed by new trial 0). This keeps dashboard
    progress plots scoped to the current orchestration stack/run while still
    surviving API/autopilot reloads inside that run.
    """
    start_idx = 0
    prev_trial_id: int | None = None
    for i, row in enumerate(rows):
        try:
            trial_id = int(row.get("trial_id"))
        except (TypeError, ValueError):
            continue
        if prev_trial_id is not None and trial_id < prev_trial_id:
            start_idx = i
        prev_trial_id = trial_id

    selected = rows[start_idx:]
    meta = {
        "journal_run_start_index": start_idx,
        "journal_run_start_trial_id": selected[0].get("trial_id") if selected else None,
        "journal_run_start_ts": selected[0].get("timestamp") if selected else None,
    }
    return selected, meta


def _pareto_from_journal(
    session_start_ts: float | None,
    *,
    current_run_only: bool = False,
    max_trial_id: int | None = None,
    deinflate_before_ts: float | None = None,
    deinflate_factor: float = 1.0,
) -> dict[str, Any] | None:
    """Reconstruct Pareto data from the append-only journal.

    By default this can reconstruct any timestamp-filtered slice. For the
    dashboard's main plots, use `current_run_only=True` so old incompatible
    stack/campaign rows do not pollute the current frontier.
    """
    if not _AUTOPILOT_JOURNAL_PATH.exists():
        return None

    all_entries: list[dict[str, Any]] = []
    frontiers_by_tier: dict[int, list[dict[str, Any]]] = {}
    hv_history_by_tier: dict[int, list[list[float]]] = {}
    t0_audit: list[dict[str, Any]] = []
    run_meta: dict[str, Any] = {}
    try:
        with open(_AUTOPILOT_JOURNAL_PATH) as f:
            rows = []
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return None

    if current_run_only:
        rows, run_meta = _latest_journal_run_rows(rows)

    # Policy parity (2026-06-04): mirror ParetoArchive.upsert_representative. A TRUSTED
    # within-noise row (mad_noise / reproduction_confirmed) is NOT corruption — it is a
    # representative candidate. Legacy rows carry bug_corrupted_by="mad_noise"; post-2026-06-04
    # rows carry the reason only in eval_details.learning_exclusion.by. Cluster such rows by
    # stable config fingerprint and admit ONE robust-MEDIAN representative per config, so the
    # panel matches the live archive instead of hiding the points (old skip) or admitting N
    # noisy per-trial samples. Genuine corruption (kills / reloads / commit-SHA) still excluded.
    processed: list[tuple[int, dict[str, Any]]] = []
    repr_clusters: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        bug = row.get("bug_corrupted_by") or ""
        excl_by = (row.get("eval_details") or {}).get("learning_exclusion", {}).get("by", "")
        if bug and bug != "mad_noise":
            continue  # genuine corruption: kill / exogenous reload / commit-invalidation
        trusted_within_noise = bug == "mad_noise" or excl_by in _WITHIN_NOISE_EXCL
        try:
            tier = int(row.get("tier", DEFAULT_FRONTIER_TIER))
        except (TypeError, ValueError):
            tier = DEFAULT_FRONTIER_TIER
        # T0 is a fast-reject sentinel tier (10q, saturates ~2.4 = 8/10). It is
        # AUDIT-ONLY: surfaced on the scatter as dim points for operator visibility
        # but it must NEVER enter a frontier or the hypervolume (mirroring the real
        # ParetoArchive, which excludes it). Tiers >=1 are segregated below so T2
        # validation points cannot dominate the canonical T1 production frontier.
        audit_only = tier < MIN_FRONTIER_EVAL_TIER
        ts = _parse_journal_ts(row.get("timestamp"))
        if session_start_ts is not None and (ts is None or ts < session_start_ts):
            continue
        try:
            trial_id = int(row.get("trial_id"))
        except (TypeError, ValueError):
            continue
        if max_trial_id is not None and trial_id > max_trial_id:
            continue
        objectives = _pareto_objectives_from_journal(row)
        if objectives is None:
            continue
        # Option (iii) speed rebase: trials recorded BEFORE the metric-fix epoch had
        # double-counted generated tokens (~1.5-2x), so their speed objective is
        # overstated. De-inflate it in place (display only — the journal stays as the
        # audit record) so honest post-fix points are compared on the same scale
        # instead of being dominated by stale, inflated speed.
        deinflated = False
        if (
            deinflate_before_ts is not None
            and deinflate_factor != 1.0
            and ts is not None
            and ts < deinflate_before_ts
            and len(objectives) >= 2
        ):
            objectives = (objectives[0], objectives[1] * deinflate_factor) + tuple(objectives[2:])
            deinflated = True
        objectives = list(objectives)
        shaped = {
            "trial_id": row.get("trial_id"),
            "objectives": objectives,
            "git_tag": row.get("git_tag", ""),
            "species": row.get("species", ""),
            "is_production_best": False,
            "timestamp": row.get("timestamp", ""),
            "reasoning": row.get("reasoning", ""),
            "eval_tier": tier,
            "speed_deinflated": deinflated,
        }
        if audit_only:
            # Audit-only point: visible on the scatter, excluded from every
            # frontier, hypervolume, and the dominated/all-entries accounting.
            t0_audit.append(shaped)
            continue
        if trusted_within_noise:
            # Defer: fold into a per-(tier, fingerprint) cluster; emit ONE median below.
            key = (tier, _config_fingerprint_from_row(row))
            cl = repr_clusters.setdefault(key, {"objs": [], "last_tid": -1, "shaped": shaped})
            cl["objs"].append(objectives)
            if trial_id >= cl["last_tid"]:
                cl["last_tid"], cl["shaped"] = trial_id, shaped
            continue
        processed.append((trial_id, shaped))

    # One robust-median representative per (tier, fingerprint); dominance is tested on the
    # MEDIAN, never a lucky single-trial speed sample (mirrors the live archive's upsert).
    for (rep_tier, fp), cl in repr_clusters.items():
        rep = dict(cl["shaped"])
        rep["objectives"] = [statistics.median(axis) for axis in zip(*cl["objs"])]
        rep["config_fingerprint"] = fp
        rep["n_reproductions"] = len(cl["objs"])
        rep["is_representative"] = True
        processed.append((cl["last_tid"], rep))

    # Replay in trial order through the same per-tier domination + hypervolume logic.
    processed.sort(key=lambda item: item[0])
    for trial_id, shaped in processed:
        tier = shaped["eval_tier"]
        objectives = shaped["objectives"]
        all_entries.append(shaped)
        frontier = frontiers_by_tier.setdefault(tier, [])
        if not any(_pareto_dominates(f["objectives"], objectives) for f in frontier):
            frontiers_by_tier[tier] = [
                f for f in frontier
                if not _pareto_dominates(objectives, f["objectives"])
            ]
            frontiers_by_tier[tier].append(shaped)
        tier_frontier = frontiers_by_tier[tier]
        tier_hv_history = hv_history_by_tier.setdefault(tier, [])
        hv = round(_pareto_hypervolume([f["objectives"] for f in tier_frontier]), 4)
        if tier_hv_history:
            hv = max(tier_hv_history[-1][1], hv)
        tier_hv_history.append([
            trial_id,
            hv,
        ])

    if not all_entries and not t0_audit:
        return None
    canonical_frontier = frontiers_by_tier.get(DEFAULT_FRONTIER_TIER, [])
    canonical_hv_history = hv_history_by_tier.get(DEFAULT_FRONTIER_TIER, [])
    archive = {
        "frontier": canonical_frontier,
        "frontiers_by_tier": {
            str(tier): front for tier, front in sorted(frontiers_by_tier.items())
        },
        "all_entries": all_entries,
        "t0_audit": t0_audit,
        "hypervolume_history": canonical_hv_history,
        "hv_history_by_tier": {
            str(tier): hist for tier, hist in sorted(hv_history_by_tier.items())
        },
        "session_start_ts": session_start_ts,
        "canonical_tier": DEFAULT_FRONTIER_TIER,
    }
    archive.update(run_meta)
    return archive


def _newest_autopilot_log() -> Path | None:
    """Return the most-recently-modified autopilot stdout log, if any."""
    try:
        candidates = sorted(
            _AUTOPILOT_LOG_DIR.glob("autopilot_restart_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None
    except Exception:
        return None


def _tail_deep_eval_progress(log_path: Path) -> tuple[int, int] | None:
    """Scan the autopilot log for the most-recent `T2 progress: X/Y` line.

    The eval tower emits these lines as each question completes, so this gives
    the *true* completion fraction for a deep_eval trial — far more accurate
    than the historical-median estimate. Returns (completed, total) or None.
    """
    if not log_path.exists():
        return None
    pat = re.compile(r"T[12] progress: (\d+)/(\d+)")
    try:
        # Tail-read: tier-2 evals can run hours and the log can be large; read
        # only the last 64 KB to find the most recent progress marker.
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            chunk = fh.read().decode("utf-8", errors="replace")
        last_match = None
        for m in pat.finditer(chunk):
            last_match = m
        if last_match:
            return int(last_match.group(1)), int(last_match.group(2))
    except Exception:
        pass
    return None


@router.get("/dashboard/api/autopilot_progress")
async def autopilot_progress() -> JSONResponse:
    """Live progress estimate for the autopilot's currently-running trial.

    Sources, in priority order:
      1. For deep_eval/structural_experiment: tails the autopilot stdout log
         for `T2 progress: X/Y` lines — these are the authoritative per-question
         completion markers, so percent = X/Y exactly (no estimation).
      2. Otherwise: full quantile distribution (p25/p50/p75/p90) of historical
         durations for the same `action_type`, derived from successive
         `timestamp` deltas in the journal (autopilot runs trials serially,
         so gap ≈ runtime).
      3. Fallback: aggregate distribution across all action types.

    The journal schema has NO numeric duration field — only an ISO `timestamp`
    written when the trial completes. Earlier versions of this endpoint looked
    for `trial_duration_s`/`wall_time_s`/`duration_s` which don't exist.

    Bar denominator is **p90, not p50**: trial-runtime distributions are
    heavily right-skewed (rollback has p90 ≈ 6× p50), so a p50 bar saturates
    to 99% on the majority of trials and conveys nothing. With p90 as the
    "expected ceiling," ~10% of trials cross into the slow tail (`slow_tail=True`),
    making that signal actually meaningful. The frontend draws ticks at
    p25/p50/p75 inside the bar so the operator can see where elapsed sits
    in the distribution at a glance; `now_percentile` is the precise CDF rank.
    """
    out: dict[str, Any] = {
        "in_flight": False,
        "autopilot_alive": False,
        "trial_id": None,
        "action_type": None,
        "started_at": None,
        "elapsed_s": None,
        "expected_s": None,                # log_tail: extrapolated projected total; others: p50 (legacy meta label)
        "percent": None,                   # elapsed/p90 * 100 (action_p50/aggregate_p50); X/Y * 100 (log_tail)
        "recent_avg_duration_s": None,
        "recent_p25": None,                # distribution quantiles used by the tick-style bar
        "recent_p50": None,
        "recent_p75": None,
        "recent_p90": None,                # bar denominator for action_p50/aggregate_p50
        "now_percentile": None,            # CDF rank of elapsed within same-action history, 0.0..1.0
        "slow_tail": False,                # True when elapsed > p90 — bar saturates, signals genuine tail
        "percent_source": None,            # "log_tail" | "action_p50" | "aggregate_p50" | "fallback"
        "n_action_type_samples": None,
        "log_tail_progress": None,         # {"completed": X, "total": Y} when percent_source=log_tail
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

    # Per-action_type durations from journal: successive-timestamp deltas
    # (autopilot is serial, so entry[i].timestamp − entry[i-1].timestamp
    # approximates the runtime of entry[i] modulo ~seconds of dispatch overhead).
    by_action: dict[str, list[float]] = {}
    all_durations: list[float] = []
    if _AUTOPILOT_JOURNAL_PATH.exists():
        try:
            with open(_AUTOPILOT_JOURNAL_PATH) as f:
                # Read the whole file (~1 MB at typical scale; cheap). For
                # multi-MB futures, switch to a tail-bytes seek.
                entries = []
                for raw in f:
                    try:
                        entries.append(json.loads(raw))
                    except Exception:
                        continue
            # Sort by timestamp ascending so deltas make sense even if the
            # journal was rewritten out-of-order (shouldn't happen, but cheap).
            entries.sort(key=lambda e: _parse_journal_ts(e.get("timestamp")) or 0)
            for i in range(1, len(entries)):
                a = _parse_journal_ts(entries[i].get("timestamp"))
                b = _parse_journal_ts(entries[i-1].get("timestamp"))
                if a is None or b is None:
                    continue
                dt = a - b
                # Drop: negative gaps, sub-30s (likely back-to-back fast failures
                # or quarantined trials), and >6h (likely overnight pauses or
                # exogenous-pause windows that don't reflect trial runtime).
                if not (30 < dt < 6 * 3600):
                    continue
                # Trust filter: skip if THIS entry was bug-quarantined
                if entries[i].get("bug_corrupted_by"):
                    continue
                at = entries[i].get("action_type") or "unknown"
                by_action.setdefault(at, []).append(dt)
                all_durations.append(dt)
        except Exception:
            pass

    def _stats(durs: list[float]) -> dict[str, float] | None:
        if not durs:
            return None
        ss = sorted(durs)
        n = len(ss)
        def q(p: float) -> float:
            return ss[min(n - 1, int(n * p))]
        return {"p25": q(0.25), "p50": q(0.5), "p75": q(0.75),
                "p90": q(0.9), "avg": sum(ss) / n, "n": n}

    def _cdf_rank(value: float, durs: list[float]) -> float | None:
        # Fraction of past durations <= value. Used so the bar can answer
        # "where does *now* sit in the historical distribution?" instead of
        # the meaningless elapsed/p50 ratio. With small n, this is granular
        # (n=18 → 5.5pp steps) but still strictly more informative than a
        # saturating bar capped at 99%.
        if not durs:
            return None
        return sum(1 for d in durs if d <= value) / len(durs)

    # 1. Log-tail (authoritative for deep_eval-style multi-question evals)
    if out["in_flight"] and out["action_type"] in ("deep_eval", "structural_experiment"):
        log = _newest_autopilot_log()
        if log:
            progress = _tail_deep_eval_progress(log)
            if progress:
                completed, total = progress
                if total > 0:
                    pct = (completed / total) * 100.0
                    out["percent"] = round(min(99.0, max(0.0, pct)), 1)
                    out["percent_source"] = "log_tail"
                    out["log_tail_progress"] = {"completed": completed, "total": total}
                    # Also extrapolate an expected_s for the meta line:
                    # if X/Y done in elapsed_s, projected total ≈ elapsed_s × Y/X
                    if completed > 0 and out["elapsed_s"]:
                        out["expected_s"] = round(out["elapsed_s"] * total / completed, 1)

    # 2. Action-type-stratified distribution (when log-tail didn't apply)
    if out["expected_s"] is None and out["in_flight"]:
        at = out["action_type"] or "unknown"
        durs = by_action.get(at, [])
        s = _stats(durs)
        if s and s["n"] >= 2:
            out["expected_s"] = round(s["p50"], 1)
            out["recent_p25"] = round(s["p25"], 1)
            out["recent_p50"] = round(s["p50"], 1)
            out["recent_p75"] = round(s["p75"], 1)
            out["recent_p90"] = round(s["p90"], 1)
            out["recent_avg_duration_s"] = round(s["avg"], 1)
            out["n_action_type_samples"] = s["n"]
            out["percent_source"] = "action_p50"
            if out["elapsed_s"] is not None:
                rank = _cdf_rank(out["elapsed_s"], durs)
                if rank is not None:
                    out["now_percentile"] = round(rank, 3)

    # 3. Aggregate distribution across all action types
    if out["expected_s"] is None and out["in_flight"]:
        s = _stats(all_durations)
        if s:
            out["expected_s"] = round(s["p50"], 1)
            out["recent_p25"] = round(s["p25"], 1)
            out["recent_p50"] = round(s["p50"], 1)
            out["recent_p75"] = round(s["p75"], 1)
            out["recent_p90"] = round(s["p90"], 1)
            out["recent_avg_duration_s"] = round(s["avg"], 1)
            out["n_action_type_samples"] = s["n"]
            out["percent_source"] = "aggregate_p50"
            if out["elapsed_s"] is not None:
                rank = _cdf_rank(out["elapsed_s"], all_durations)
                if rank is not None:
                    out["now_percentile"] = round(rank, 3)

    # 4. Hard fallback only if we have literally no journal data
    if out["expected_s"] is None and out["in_flight"]:
        out["expected_s"] = 1200.0
        out["percent_source"] = "fallback"

    # Percent: bar fill ratio.
    #   - log_tail: already set above to X/Y (true progress).
    #   - action_p50/aggregate_p50: elapsed/p90. p90 is "the slow-tail edge" —
    #     ~90% of past trials of this action_type finish by then, so the bar
    #     saturates only when we're genuinely in the tail. Allowed to exceed
    #     100 (slow_tail=True) so the JS can render "p93+" without lying.
    #   - fallback: elapsed/1200, clamped at 100.
    if (
        out["percent"] is None
        and out["in_flight"]
        and out["elapsed_s"] is not None
    ):
        denom = out["recent_p90"] if out["recent_p90"] else out["expected_s"]
        if denom:
            pct = (out["elapsed_s"] / denom) * 100.0
            out["slow_tail"] = pct > 100.0
            # Cap at 150 — anything beyond is visually identical "deep tail";
            # the precise number is in elapsed_s if anyone needs it.
            out["percent"] = round(min(150.0, max(0.0, pct)), 1)

    return JSONResponse(out, headers=_NO_STORE_HEADERS)


@router.get("/dashboard/api/pareto")
async def pareto(max_dominated: int = 600) -> JSONResponse:
    """Return the autopilot's Pareto archive for visualization.

    Prefer reconstructing all trusted progress from the append-only journal. This
    keeps the dashboard useful when autopilot is down and protects the panel
    from a stale `pareto_archive` cache inside autopilot_state.json.

    Objectives in the archive are (quality, speed, -cost, reliability).
    The dashboard plots the first two by default since they're the most
    operationally meaningful; -cost and reliability ride along as
    per-point fields the client can surface in tooltips.
    """
    data: dict[str, Any] = {}
    state_error: str | None = None
    if _AUTOPILOT_STATE_PATH.exists():
        try:
            data = json.loads(_AUTOPILOT_STATE_PATH.read_text())
        except Exception as exc:
            state_error = f"failed to parse autopilot_state.json: {exc}"

    session_start_ts = None
    try:
        session_start_ts = float(data.get("autopilot_fleet_started_at") or 0.0) or None
    except (TypeError, ValueError):
        session_start_ts = None

    state_trial_counter = None
    try:
        state_trial_counter = int(data.get("trial_counter"))
    except (TypeError, ValueError):
        state_trial_counter = None

    # The operator-facing Pareto/GEPA plots are current-run progress charts,
    # not "this API process since reload" and not all historical campaigns.
    # Use the latest journal segment after a trial-id reset so restarts inside
    # the current run survive while pre-current-stack rows stay out.
    # A deliberate Pareto rebase (e.g. the 2026-06-01 speed double-count correction)
    # records a stable `pareto_epoch_ts`: trials before it were measured under the old
    # metric and must NOT be reconstructed onto the frontier (they would dominate the
    # honest post-fix points on the changed axis). The trial counter is NOT reset on a
    # rebase, so `current_run_only` alone can't exclude them — scope by timestamp too.
    pareto_epoch_ts = None
    try:
        pareto_epoch_ts = float(data.get("pareto_epoch_ts") or 0.0) or None
    except (TypeError, ValueError):
        pareto_epoch_ts = None
    # Option (iii): rather than EXCLUDING pre-fix trials, keep them but de-inflate their
    # double-counted speed (factor tunable via state; default 0.5 ≈ the ~2x inflation of
    # no-thinking eval trials) so the metric correction is visible on the plot.
    deinflate_factor = 0.5
    try:
        deinflate_factor = float(data.get("pareto_pre_epoch_speed_factor", 0.5))
    except (TypeError, ValueError):
        deinflate_factor = 0.5
    journal_archive = _pareto_from_journal(
        None,
        current_run_only=True,
        max_trial_id=state_trial_counter,
        deinflate_before_ts=pareto_epoch_ts,
        deinflate_factor=deinflate_factor,
    )
    source = "journal_current_run"
    source_reason = "reconstructed from latest trial-id reset segment in autopilot_journal.jsonl"

    if journal_archive:
        archive = journal_archive
    elif data:
        archive = data.get("pareto_archive", {}) or {}
        source = "state_archive"
        source_reason = "journal unavailable or no trusted current-run entries"
    else:
        return JSONResponse({
            "available": False,
            "reason": state_error or "autopilot_state.json and autopilot_journal.jsonl not found",
            "frontier": [],
            "dominated": [],
            "hypervolume_history": [],
        })

    canonical_tier = int(archive.get("canonical_tier", DEFAULT_FRONTIER_TIER))
    frontiers_by_tier_raw = archive.get("frontiers_by_tier", {}) or {}
    hv_history_by_tier_raw = archive.get("hv_history_by_tier", {}) or {}
    frontier_raw = (
        frontiers_by_tier_raw.get(str(canonical_tier))
        or frontiers_by_tier_raw.get(canonical_tier)
        or archive.get("frontier", [])
        or []
    )
    all_raw = archive.get("all_entries", []) or []
    t0_audit_raw = archive.get("t0_audit", []) or []
    t0_audit_shaped = [_shape_pareto_entry(e) for e in t0_audit_raw]
    hv_history = (
        hv_history_by_tier_raw.get(str(canonical_tier))
        or hv_history_by_tier_raw.get(canonical_tier)
        or archive.get("hypervolume_history", [])
        or []
    )
    frontier = [_shape_pareto_entry(e) for e in frontier_raw]
    frontiers_by_tier = {
        str(tier): [_shape_pareto_entry(e) for e in entries]
        for tier, entries in frontiers_by_tier_raw.items()
    }
    frontier_ids = {f["trial_id"] for f in frontier if f["trial_id"] is not None}

    # Dominated entries: newest first, trimmed to max_dominated to bound payload.
    dominated_only = [e for e in all_raw if e.get("trial_id") not in frontier_ids]
    dominated_only.sort(key=lambda e: (e.get("trial_id") or 0), reverse=True)
    dominated_shaped = [_shape_pareto_entry(e) for e in dominated_only[:max_dominated]]

    hv_shaped: list[list[float]] = []
    for h in hv_history:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            try:
                hv_shaped.append([int(h[0]), float(h[1])])
            except (TypeError, ValueError):
                continue

    return JSONResponse({
        "available": True,
        "source": source,
        "source_reason": source_reason,
        "frontier": frontier,
        "frontiers_by_tier": frontiers_by_tier,
        "dominated": dominated_shaped,
        "t0_audit": t0_audit_shaped,
        "hypervolume_history": hv_shaped,
        "hv_history_by_tier": hv_history_by_tier_raw,
        "totals": {
            "frontier_size": len(frontier),
            "all_entries": len(all_raw),
            "hv_points": len(hv_shaped),
        },
        "canonical_tier": canonical_tier,
        "session_start_ts": archive.get("session_start_ts"),
        "journal_run_start_index": archive.get("journal_run_start_index"),
        "journal_run_start_trial_id": archive.get("journal_run_start_trial_id"),
        "journal_run_start_ts": archive.get("journal_run_start_ts"),
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
    llama_models = _discover_llama_models()
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
            # Model actually loaded by this llama-server (-m GGUF basename,
            # vendor-prefix + shard-suffix stripped). Surfaced so the topology
            # strip can label each role with its backing model + quant.
            "model": llama_models.get(port, ""),
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
            "model": _clean_model_name(svc.get("model", "")),
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
    path: Path, window_s: float = 600.0, max_items: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    # 2026-05-25: window widened 50 → 200 — at typical chat traffic the
    # last-50 cut was overwhelmingly frontdoor (~93% of all chat hits today),
    # making the panel look like routing was frontdoor-only when in reality
    # 6+ distinct base roles get hit per day. 200 gives a more honest mix
    # without paying the cost of scanning the whole day's JSONL.
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
    tasks: list[dict[str, Any]],
    role_busy: dict[str, int],
    alias_to_topology_role: dict[str, str] | None = None,
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
    alias_to_topology_role = alias_to_topology_role or {}

    def _role_key(role: Any) -> str:
        return str(role or "unknown").split(":", 1)[0]

    by_role: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        logical_role = _role_key(t.get("role") or "unknown")
        topology_role = alias_to_topology_role.get(logical_role, logical_role)
        by_role.setdefault(topology_role, []).append(t)
    gated: list[dict[str, Any]] = []
    for role, group in by_role.items():
        group.sort(key=lambda x: x.get("age_s", 0.0))  # newest first
        busy = role_busy.get(role, 0)
        n_fresh = sum(1 for t in group if t.get("age_s", 0.0) <= _FRESH_INFLIGHT_S)
        for idx, task in enumerate(group[: max(busy, n_fresh)]):
            annotated = dict(task)
            raw_role = annotated.get("role") or "unknown"
            logical_role = _role_key(raw_role)
            topology_role = alias_to_topology_role.get(logical_role, logical_role)
            if topology_role != str(raw_role):
                annotated["topology_role"] = topology_role
            if idx < busy:
                annotated["live_state"] = "decoding"
                annotated["live_state_reason"] = "live slot busy for topology role"
            else:
                annotated["live_state"] = "pending"
                annotated["live_state_reason"] = (
                    "fresh task_started with no busy slot yet; likely queued, "
                    "pre-inference, or local post-processing"
                )
            gated.append(annotated)
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
    alias_to_topology_role: dict[str, str] = {}
    for role_label in set(port_roles.values()):
        role = base_role(role_label)
        if not role:
            continue
        try:
            for alias in role_aliases(role):
                alias_base = base_role(alias)
                if alias_base:
                    alias_to_topology_role[alias_base] = role
        except Exception:
            continue
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
    in_flight_tasks = _gate_inflight_by_live_slots(
        in_flight_tasks,
        role_busy,
        alias_to_topology_role=alias_to_topology_role,
    )

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
    structured_tap = _find_structured_request_by_id(task_id)
    if structured_tap is not None and task_id.startswith("tap_"):
        text = _task_text_snapshot(
            task_id, events, None, tap_section=structured_tap
        )
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(text)

    slots_by_port = await _poll_all_slots()
    found_slot = None
    for port, slots in slots_by_port.items():
        for s in slots:
            if str(s.get("id_task")) == task_id:
                found_slot = s
                break
        if found_slot:
            break

    # Tap-section fallback when no live slot. Prefer the structured event
    # stream by task_id (deterministic mapping) over the plaintext substring
    # matcher (vulnerable to interleaved per-append writes producing
    # syntactically-valid but cross-contaminated records — see 2026-05-30
    # chat-83123001/chat-c7bf9580 incident).
    tap_section = None
    if found_slot is None:
        if not task_id.startswith("tap_"):
            tap_section = _find_structured_request_by_task_id(task_id)
        if tap_section is None:
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
            tap_section = _find_section_by_objective(
                objective, expected_role=producer_role
            )

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
    structured_tap = _find_structured_request_by_id(task_id)
    if structured_tap is not None and task_id.startswith("tap_"):
        return JSONResponse({
            "task_id": task_id,
            "objective": structured_tap.get("prompt") or "",
            "events": events,
            "active_slot_port": None,
            "active_slot_id": None,
            "slot": None,
            "tap_section": structured_tap,
        })

    objective = _objective_for_task(events)
    slots_by_port = await _poll_all_slots()
    slot_port, active_slot = await _find_slot_by_objective(objective, slots_by_port)

    # Fallback when no live slot. Prefer the structured event stream by
    # task_id (deterministic mapping by request metadata) over the plaintext
    # substring matcher, which is vulnerable to interleaved per-append writes
    # (chat-83123001/chat-c7bf9580 cross-contamination, 2026-05-30). If the
    # structured stream has nothing for this task id, fall back to plaintext;
    # producer_role from task_completed constrains the role-filtered pass and
    # also blocks the unsafe global fallback in _find_section_by_objective.
    tap_section = None
    if active_slot is None:
        if not task_id.startswith("tap_"):
            tap_section = _find_structured_request_by_task_id(task_id)
        if tap_section is None:
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
            tap_section = _find_section_by_objective(
                objective, expected_role=producer_role
            )

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
    return HTMLResponse(
        _read_dashboard_html(),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )
