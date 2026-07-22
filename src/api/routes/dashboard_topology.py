"""Dashboard topology helpers — port → role discovery, color resolution, process info.

Pure-data helpers extracted from src/api/routes/dashboard.py during the 2026-05-21
refactor. Route handlers in dashboard.py re-import these so signatures stay
unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from scripts.server.realized_fleet import derive_realized_numa_mode
from scripts.server.runtime_facts_manifest import (
    read_runtime_stack_numa_mode,
    read_runtime_stack_selected_servers,
    runtime_facts_manifest_path,
)
from scripts.server.stack_numa_mode import (
    DASHBOARD_RUNTIME_FALLBACK_NUMA_MODE,
    VALID_STACK_NUMA_MODES,
)
from src.roles import Role
from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_serving,
    stack_prior_serving_ports,
)

logger = logging.getLogger(__name__)

_DEFAULT_STACK_PRIORS_PATH = (
    Path(__file__).resolve().parents[3] / "orchestration" / "derived" / "stack_priors.yaml"
)

# Service-only hints. Model-serving ports are projected from generated stack
# priors below so dashboard labels follow the same launch contract as the stack.
_BASE_SERVICE_PORT_HINTS: dict[int, str] = {
    8000: "orchestrator",
    8088: "nextplaid-code",
    8089: "nextplaid-docs",
    8090: "embedder",
    8091: "embedder_1",
    8092: "embedder_2",
    8093: "embedder_3",
    8094: "embedder_4",
    8095: "embedder_5",
    8190: "sd_server",
    9000: "whisper",
    9001: "document_formalizer",
}


def _service_port_hints() -> dict[int, str]:
    return dict(_BASE_SERVICE_PORT_HINTS)


def _label_for_stack_prior_entry(role: str, entry: dict[str, Any]) -> tuple[int, str] | None:
    if entry.get("alias"):
        return None
    primary_role = entry.get("primary_role")
    if isinstance(primary_role, str) and primary_role and primary_role != role:
        return None
    port = entry.get("port")
    if not isinstance(port, int):
        return None
    numa_instance = entry.get("numa_instance")
    if isinstance(numa_instance, int) and numa_instance > 0:
        return port, f"{role}.q{numa_instance - 1}"
    return port, role


def _stack_prior_port_hints(
    stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH,
) -> dict[int, str]:
    """Project live model-serving port labels from generated stack priors."""
    roles = live_stack_role_records(stack_priors_path)
    if not roles:
        return {}

    hints: dict[int, str] = {}
    for role, record in sorted(roles.items()):
        serving = stack_prior_serving(record)
        launch = serving.get("launch")
        launch = launch if isinstance(launch, dict) else {}
        primary_roles = launch.get("primary_roles")
        if isinstance(primary_roles, list) and primary_roles and role not in primary_roles:
            continue

        mapped = False
        entries = launch.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                label = _label_for_stack_prior_entry(role, entry)
                if label is None:
                    continue
                port, name = label
                hints[port] = name
                mapped = True

        if mapped:
            continue
        ports = stack_prior_serving_ports(serving)
        for index, port in enumerate(ports):
            hints[port] = role if index == 0 else f"{role}.q{index - 1}"

    return hints


def _build_port_hints() -> dict[int, str]:
    hints = _service_port_hints()
    hints.update(_stack_prior_port_hints())
    return hints


# Public compatibility map used by tests and dashboard callers.
_PORT_HINTS: dict[int, str] = _build_port_hints()


# Realized-fleet probe cache. The dashboard must render the REALIZED live fleet,
# not launch-time intent (ESC-8 / audit finding C1): a uvicorn worker that
# inherited ORCHESTRATOR_STACK_NUMA_MODE=full must never paint a quarters-only
# fleet as rogue. The probe is a handful of bare-TCP connect_ex() calls against
# localhost (scripts.server.realized_fleet.derive_realized_numa_mode); a short
# TTL collapses a 2 Hz × 6-worker poll burst into a single probe so dashboard
# polling does not storm the loopback port universe.
_REALIZED_NUMA_CACHE: dict[str, Any] = {"ts": 0.0, "value": None, "probed": False}
_REALIZED_NUMA_TTL_S = 5.0


def _probe_realized_numa_mode() -> str | None:
    """Probe the live fleet and classify its realized NUMA mode (``full`` /
    ``quarter`` / ``both``), or ``None`` when nothing in the quarterable port
    universe is listening / the probe fails.

    Isolated seam: production opens localhost TCP connections here; tests
    neutralize or substitute this without touching sockets.
    """
    try:
        return derive_realized_numa_mode()
    except Exception:
        logger.debug("realized-fleet NUMA probe failed; ignoring", exc_info=True)
        return None


def _cached_realized_numa_mode() -> str | None:
    """TTL-cached ``_probe_realized_numa_mode()`` (see the cache note above)."""
    now = time.monotonic()
    cache = _REALIZED_NUMA_CACHE
    if cache.get("probed") and (now - cache["ts"]) < _REALIZED_NUMA_TTL_S:
        return cache["value"]
    value = _probe_realized_numa_mode()
    cache["ts"] = now
    cache["value"] = value
    cache["probed"] = True
    return value


def _env_declared_numa_mode() -> str | None:
    """The env-declared stack NUMA mode, or ``None`` when unset/blank/unrecognized.

    Unlike ``env_stack_numa_mode`` (which coerces anything to a default), this
    returns ``None`` for absent or non-canonical values so the resolver can treat
    the env purely as a spawn-time hint that carries intent only when it is a real
    mode string.
    """
    raw = os.environ.get("ORCHESTRATOR_STACK_NUMA_MODE")
    if raw is None:
        return None
    normalized = raw.strip().lower()
    return normalized if normalized in VALID_STACK_NUMA_MODES else None


def active_stack_numa_mode_resolution() -> dict[str, Any]:
    """Resolve the stack NUMA mode the dashboard should render, realized-fleet first.

    Precedence (audit finding C1 — the dashboard renders the REALIZED fleet, not
    launch-time intent)::

        realized live fleet  >  hardened runtime-facts manifest  >  env hint  >  default

    ``ORCHESTRATOR_STACK_NUMA_MODE`` is demoted to a LAST-resort spawn-time hint.
    A uvicorn worker that inherited ``full`` must not flag a quarters-only fleet as
    rogue (the proven C1 inversion) — the realized probe overrides it. When a
    lower-precedence source contradicts the resolved mode, the contradiction is
    recorded in ``disagreements`` so surfaces can badge it ("env disagrees: full")
    instead of silently trusting the lying source.

    Returns a provenance dict:
      * ``mode``          — the string the surfaces render.
      * ``source``        — ``realized_fleet`` / ``runtime_manifest`` / ``env`` / ``default``.
      * ``realized`` / ``manifest`` / ``env`` — each source's value (str | None).
      * ``disagreements`` — human-readable lower-precedence contradictions.
    """
    realized = _cached_realized_numa_mode()
    manifest = _fail_closed_runtime_stack_numa_mode()
    env = _env_declared_numa_mode()

    if realized is not None:
        mode, source = realized, "realized_fleet"
        lower: list[tuple[str, str | None]] = [("manifest", manifest), ("env", env)]
    elif manifest is not None:
        mode, source = manifest, "runtime_manifest"
        lower = [("env", env)]
    elif env is not None:
        mode, source = env, "env"
        lower = []
    else:
        mode, source = DASHBOARD_RUNTIME_FALLBACK_NUMA_MODE, "default"
        lower = []

    disagreements = [
        f"{label} disagrees: {value}"
        for label, value in lower
        if value is not None and value != mode
    ]
    return {
        "mode": mode,
        "source": source,
        "realized": realized,
        "manifest": manifest,
        "env": env,
        "disagreements": disagreements,
    }


def active_stack_numa_mode() -> str:
    """Return the stack NUMA mode string the dashboard surfaces should render.

    Thin wrapper over :func:`active_stack_numa_mode_resolution` for the many call
    sites that need only the mode string. See that function for the
    realized-fleet-first precedence and provenance semantics. This drives only the
    dashboard/health family; the config compiler and launcher own their own
    spawn-time planning.
    """
    return active_stack_numa_mode_resolution()["mode"]


def _fail_closed_runtime_stack_numa_mode() -> str | None:
    """Return the runtime-facts stack NUMA mode ONLY when the manifest passes the
    same fail-closed contract the URL reader (read_runtime_stack_selected_servers)
    enforces: a concrete expected mode string AND a non-empty selected-server
    lineup consistent with the declared ports.

    WP-14: the launcher can leave a phantom full-era lineup behind (the real
    current shape is stack_numa_mode=None, selected_ports=[], full-era
    selected_servers). read_runtime_stack_numa_mode() alone would either accept a
    stale mode or return None while the topology port hints still projected the
    phantom lineup, so the dashboard could render a NUMA mode that no live
    process backs. Mirror the URL reader's rejection here and fall back to the
    dashboard's historical env/NUMA_CONFIG default with one loud log line.
    """
    mode = read_runtime_stack_numa_mode()
    servers = read_runtime_stack_selected_servers()
    lineup_ok = isinstance(servers, list) and bool(servers)
    if isinstance(mode, str) and mode and lineup_ok:
        return mode.strip().lower()

    try:
        manifest_present = runtime_facts_manifest_path().exists()
    except Exception:
        manifest_present = False
    if manifest_present:
        logger.warning(
            "runtime-facts manifest rejected (fail-closed: stack_numa_mode=%r, "
            "selected lineup %s); falling back to dashboard NUMA_CONFIG default",
            mode,
            "present" if lineup_ok else "empty/inconsistent",
        )
    return None


def _manifest_server_label(server: dict[str, Any]) -> str:
    roles = server.get("roles") or []
    role = str(roles[0]) if isinstance(roles, list) and roles else ""
    if not role:
        return ""
    numa_instance = server.get("numa_instance")
    if isinstance(numa_instance, int) and numa_instance > 0:
        return f"{role}.q{numa_instance - 1}"
    return role


def _manifest_port_hints(numa_mode: str | None = None) -> dict[int, str]:
    if numa_mode is None and os.environ.get("ORCHESTRATOR_STACK_NUMA_MODE") is None:
        runtime_servers = read_runtime_stack_selected_servers()
        if runtime_servers is not None:
            hints: dict[int, str] = {}
            for server in runtime_servers:
                port = server.get("port")
                label = _manifest_server_label(server)
                if isinstance(port, int) and label:
                    hints[port] = label
            return hints

    try:
        from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS, _filter_by_numa_mode
    except Exception:
        return {}
    mode = numa_mode or active_stack_numa_mode()
    try:
        servers = _filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, mode)
    except Exception:
        servers = HOT_SERVERS + WARM_SERVERS
    hints: dict[int, str] = {}
    for server in servers:
        if not isinstance(server, dict):
            continue
        port = server.get("port")
        label = _manifest_server_label(server)
        if isinstance(port, int) and label:
            hints[port] = label
    return hints


def _configured_numa_port_hints() -> dict[int, str]:
    """Labels for every statically configured NUMA llama-server port.

    This is deliberately independent of the active launch mode. A quarter
    listener that is already running must be labeled as `role.qN` in topology
    and lock surfaces even if the current manifest mode says that quarter was
    not expected for this run.
    """
    try:
        from scripts.server.stack_numa import NUMA_CONFIG
    except Exception:
        return {}

    hints: dict[int, str] = {}
    for role, cfg in (NUMA_CONFIG or {}).items():
        if not isinstance(cfg, dict):
            continue
        instances = cfg.get("instances")
        if not isinstance(instances, list):
            continue
        full_idx = cfg.get("full_instance_idx")
        for idx, entry in enumerate(instances):
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            port = entry[1]
            if not isinstance(port, int):
                continue
            label = role
            if isinstance(full_idx, int) and idx != full_idx:
                label = f"{role}.q{idx - 1 if idx > full_idx else idx}"
            hints[port] = label
    return hints


def _port_hint(port: int) -> str:
    return (
        _PORT_HINTS.get(port)
        or _manifest_port_hints().get(port)
        or _configured_numa_port_hints().get(port)
        or f"port_{port}"
    )

# Per-role display colors (CSS hex).
_ROLE_COLORS: dict[str, str] = {
    "mi210_gpu": "#f97316",
    "frontdoor": "#3b82f6",
    "worker_general": "#10b981",
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
    canonical = Role.from_string(base)
    if canonical is not None:
        base = canonical.value
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
    for line in out.splitlines():
        if "llama-server" not in line:
            continue
        port_m = pid_port_re.search(line)
        if not port_m:
            continue
        port = int(port_m.group(1))
        role = _port_hint(port)
        if role == f"port_{port}":
            # Unmapped port: name it by what it IS. Mangling the model stem
            # into the role (`port_8802(Qwen3.6-…)`) leaked garbled keys into
            # live_busy_by_role and made the activity view unreadable.
            # MI210 HIP builds are the GPU testbed — operator-decided
            # (2026-07-05) to render first-class ahead of stack integration.
            role = "mi210_gpu" if "mi210" in line.lower() else f"extern_{port}"
        ports[port] = role
    return ports


_VENDOR_PREFIX_RE = re.compile(
    r"^(Qwen|Meta|Google|Mistral|DeepSeek|unsloth|bartowski|lmstudio[-_]community)[-_]",
    re.IGNORECASE,
)
_SHARD_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")


def _clean_model_name(model_path: str) -> str:
    """Human-friendly model label from a GGUF path.

    basename → drop `.gguf` → drop multi-file shard suffix (`-00001-of-00003`)
    → drop a redundant leading vendor prefix (`Qwen_Qwen3.6…` → `Qwen3.6…`).
    Returns '' for empty input so callers can omit the field.
    """
    if not model_path:
        return ""
    stem = Path(model_path).name
    stem = re.sub(r"\.gguf$", "", stem, flags=re.IGNORECASE)
    stem = _SHARD_SUFFIX_RE.sub("", stem)
    stem = _VENDOR_PREFIX_RE.sub("", stem)
    return stem


def _discover_llama_models() -> dict[int, str]:
    """Scan /proc for running llama-server processes → {port: cleaned model name}.

    Mirrors `_discover_llama_ports` but extracts the `-m <model>` GGUF path so the
    topology endpoint can label each role with the model it is actually serving.
    """
    models: dict[int, str] = {}
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
        model_m = pid_model_re.search(line)
        if not port_m or not model_m:
            continue
        models[int(port_m.group(1))] = _clean_model_name(model_m.group(1))
    return models


def _pid_is_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    return pid_int > 0 and Path(f"/proc/{pid_int}").exists()


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
                "running": _pid_is_running(info.get("pid")),
            })
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("Failed to load orchestrator_state.json: %s", exc)
    return services


def expected_stack_services(numa_mode: str | None = None) -> list[dict[str, Any]]:
    """Expected stack servers from the launch manifest, including unloaded ports."""
    if numa_mode is None and "ORCHESTRATOR_STACK_NUMA_MODE" not in os.environ:
        runtime_servers = read_runtime_stack_selected_servers()
        if runtime_servers is not None:
            return _expected_services_from_manifest_servers(runtime_servers)

    try:
        from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS, _filter_by_numa_mode
    except Exception as exc:
        logger.debug("Failed to load stack manifest services: %s", exc)
        return []

    mode = numa_mode or active_stack_numa_mode()
    try:
        servers = _filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, mode)
    except Exception as exc:
        logger.debug("Failed to filter stack manifest services by NUMA mode %s: %s", mode, exc)
        servers = HOT_SERVERS + WARM_SERVERS
    return _expected_services_from_manifest_servers(servers)


def _expected_services_from_manifest_servers(servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    services: list[dict[str, Any]] = []
    for server in servers:
        if not isinstance(server, dict):
            continue
        port = server.get("port")
        roles = server.get("roles") or []
        if not isinstance(port, int) or not isinstance(roles, list) or not roles:
            continue
        role = _manifest_server_label(server) or _port_hint(port)
        if role == f"port_{port}":
            role = str(roles[0])
        services.append({
            "name": role,
            "role": role,
            "port": port,
            "roles": [str(r) for r in roles],
            "embedding": bool(server.get("embedding")),
            "vision": bool(server.get("vision")),
            "worker_pool": bool(server.get("worker_pool")),
            "numa_instance": server.get("numa_instance"),
        })
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
