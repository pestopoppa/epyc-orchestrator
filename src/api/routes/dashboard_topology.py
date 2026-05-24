"""Dashboard topology helpers — port → role discovery, color resolution, process info.

Pure-data helpers extracted from src/api/routes/dashboard.py during the 2026-05-21
refactor. Route handlers in dashboard.py re-import these so signatures stay
unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
# NUMA quarters share the parent role. Extended 2026-05-24 to cover
# ingest_long_context (8185/8285/8385/8485) and vision_escalation (8187/8287/8387/8487)
# that were added by Phase 1b of the cross-role-bw-aware-routing handoff.
for _parent_base, _parent_role in (
    (8080, "frontdoor"),
    (8082, "worker_general"),
    (8185, "ingest_long_context"),
    (8187, "vision_escalation"),
):
    for _q in range(4):
        _PORT_HINTS[_parent_base + _q * 100] = f"{_parent_role}.q{_q}"

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


def base_role(role: str) -> str:
    """Collapse an instance/quarter label to its canonical base role.

    Mirrors the grouping the dashboard front-end (`renderTopologyStrip`) applies
    so that every surface — topology rows, slot-dot aggregation, in-flight task
    grouping, and the recent-activity headline — keys off the same string:

        "frontdoor.q2"  -> "frontdoor"   (NUMA quarter)
        "embedder_3"    -> "embedder"    (numbered sibling)
        "architect_general" -> unchanged

    Only a trailing `_<digits>` is stripped, so multi-word roles like
    `architect_general` / `ingest_long_context` are left intact.
    """
    if not role:
        return ""
    base = role.split(".")[0]
    return re.sub(r"_\d+$", "", base)


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


def role_aliases(role: str) -> list[str]:
    """Return the list of alias role names served by the same llama-server.

    Reads `shared_with_first_n` from stack_manifest.ROLE_LAUNCH_META. e.g.
    `frontdoor` returns `["coder_escalation", "worker_summarize"]`. Returns []
    when the role has no aliases or when the manifest cannot be imported
    (test contexts, scripts run outside the stack tree).
    """
    base = base_role(role)
    try:
        # Lazy import — keeps dashboard_topology importable without the scripts
        # tree on sys.path (e.g. unit tests).
        import sys
        scripts_dir = Path(__file__).resolve().parents[3] / "scripts" / "server"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from stack_manifest import ROLE_LAUNCH_META  # type: ignore
        meta = ROLE_LAUNCH_META.get(base, {})
        aliases = meta.get("shared_with_first_n") or []
        return list(aliases)
    except Exception:
        return []


def _discover_llama_ports() -> dict[int, str]:
    """Scan /proc for running llama-server processes and extract port→role.

    Falls back to _PORT_HINTS for unmapped ports. Cheap (~5ms), runs once per
    snapshot poll.
    """
    ports: dict[int, str] = {}
    try:
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


def _load_state_services(state_path: Path) -> list[dict[str, Any]]:
    """Load non-llama auxiliary services from orchestrator_state.json at `state_path`."""
    services: list[dict[str, Any]] = []
    try:
        with open(state_path) as f:
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


def _process_info_by_match(needle: str) -> dict[str, Any]:
    """Find a long-running Python process by command-line substring."""
    try:
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
