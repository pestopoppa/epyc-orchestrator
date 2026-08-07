"""Runtime-only stack facts manifest.

This module writes a derived JSON cache under the runtime tmp directory. It is
not a source of truth: launch/state facts still come from stack priors, the
launcher state file, and fleet marker files.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.server.fleet_markers import (
    discover_llama_markers,
    read_orchestrator_marker_metadata,
)
from scripts.server.realized_fleet import classify_numa_mode_from_ports
from scripts.server.stack_numa_mode import VALID_STACK_NUMA_MODES, normalize_stack_numa_mode
from scripts.server.stack_paths import LLAMA_SERVER, LOG_DIR, STATE_FILE, _PATHS
from scripts.server.stack_processes import pid_alive as _default_pid_alive
from src.registry.stack_priors import (
    load_stack_priors_artifact,
    live_stack_role_records,
)

PidAliveFn = Callable[[int], bool]


RUNTIME_FACTS_MANIFEST_NAME = "orchestrator_runtime_facts.json"
RUNTIME_FACTS_SCHEMA = "epyc.orchestrator.runtime_facts"
RUNTIME_FACTS_MANIFEST_VERSION = 1


def runtime_facts_manifest_path(tmp_dir: Path | None = None) -> Path:
    """Return the runtime facts manifest path."""
    return (tmp_dir or _PATHS["tmp_dir"]) / RUNTIME_FACTS_MANIFEST_NAME


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _stack_priors_summary(stack_priors_path: Path) -> dict[str, Any]:
    artifact = load_stack_priors_artifact(stack_priors_path)
    if artifact is None:
        return {
            "path": str(stack_priors_path),
            "available": False,
            "live_role_count": 0,
            "live_roles": [],
        }

    live_roles = sorted(live_stack_role_records(stack_priors_path))
    source_artifacts = artifact.get("source_artifacts")
    return {
        "path": str(stack_priors_path),
        "available": True,
        "stack_priors_version": artifact.get("stack_priors_version"),
        "compiled_at": artifact.get("compiled_at"),
        "status": artifact.get("status"),
        "source_artifacts": source_artifacts if isinstance(source_artifacts, dict) else {},
        "live_role_count": len(live_roles),
        "live_roles": live_roles,
    }


def _effective_paths_summary(*, stack_priors_path: Path, tmp_dir: Path | None) -> dict[str, str]:
    marker_dir = tmp_dir or _PATHS["tmp_dir"]
    return {
        "tmp_dir": str(marker_dir),
        "state_file": str(STATE_FILE),
        "stack_priors_path": str(stack_priors_path),
        "log_dir": str(LOG_DIR),
        "llama_server": str(LLAMA_SERVER),
    }


def _known_llama_roles() -> set[str]:
    """Role names that correspond to launched llama-servers (declarative view).

    Used to exclude non-llama state rows (orchestrator API, document_formalizer,
    docker services, handoff dashboard) from the realized serving-fleet
    derivation so ``selected_servers`` mirrors the declarative HOT/WARM shape.
    """
    try:
        from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS
    except Exception:
        return set()
    roles: set[str] = set()
    for server in list(HOT_SERVERS) + list(WARM_SERVERS):
        if not isinstance(server, dict):
            continue
        for role in server.get("roles", []) or []:
            if isinstance(role, str) and role:
                roles.add(role)
    return roles


def _state_rows(state: Mapping[str, Any]):
    """Yield (key, role, port, pid) tuples from ProcessInfo-like state entries."""
    for key, value in state.items():
        if is_dataclass(value) and not isinstance(value, type):
            yield (
                key,
                getattr(value, "role", None),
                getattr(value, "port", None),
                getattr(value, "pid", None),
            )
        elif isinstance(value, Mapping):
            yield (key, value.get("role"), value.get("port"), value.get("pid"))


def _int_port(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _realized_selected_servers(
    state: Mapping[str, Any],
    *,
    is_alive: PidAliveFn,
) -> list[dict[str, Any]]:
    """Derive the realized serving fleet from the launcher ``state``.

    Groups live llama-server rows by port, resolves each endpoint's TRUE
    topology index by port (DISPATCH-A2 ``topology_idx_for_port``), and filters
    out dead rows (stale server_8096/8097/8098 embedder entries, docker rows,
    unknown pids). This replaces the previous static
    ``_filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, mode)`` view, which
    recorded declarative ports (dead full 8070/8072/8085) rather than what the
    launcher actually realized.
    """
    from src.runtime.instance_topology import (
        topology_idx_for_port,
        topology_instance_for_port,
    )

    known = _known_llama_roles()
    roles_by_port: dict[int, set[str]] = {}
    primary_by_port: dict[int, str] = {}
    for key, role, port, pid in _state_rows(state):
        p = _int_port(port)
        if p is None:
            continue
        # Liveness filter: skip dead/unknown pids and docker rows (pid == -1).
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            continue
        if not is_alive(pid):
            continue
        roles_by_port.setdefault(p, set())
        if isinstance(role, str) and role:
            roles_by_port[p].add(role)
            primary_by_port.setdefault(p, role)
        # Role-keyed alias rows (state[alias] = same ProcessInfo) carry the
        # alias name in the KEY, not the ProcessInfo.role field.
        if isinstance(key, str) and key and not key.startswith("server_"):
            roles_by_port[p].add(key)

    servers: list[dict[str, Any]] = []
    for port in sorted(roles_by_port):
        roles = sorted(role for role in roles_by_port[port] if role in known)
        if not roles:
            continue  # non-llama row (orchestrator/document_formalizer/docker)
        primary = primary_by_port.get(port)
        if primary not in roles:
            primary = roles[0]
        server: dict[str, Any] = {"port": port, "roles": roles}
        # ProcessInfo.role is a logical serving alias for several shared fleets
        # (the worker processes are launched as ``worker_explore`` while their
        # physical NUMA topology is keyed by ``worker_general``). Resolving only
        # the ProcessInfo primary silently dropped ``numa_instance`` from runtime
        # facts, which in turn degraded config to plain round-robin URLs and
        # bypassed split-aware placement. Try every realized alias and retain the
        # first role whose topology actually owns this port.
        physical_owner = topology_instance_for_port(port)
        if physical_owner is not None:
            topology_role, idx = physical_owner
            server["numa_instance"] = idx
            server["topology_role"] = topology_role
        else:
            for topology_role in [primary, *roles]:
                idx = topology_idx_for_port(topology_role, port)
                if idx is not None:
                    server["numa_instance"] = idx
                    server["topology_role"] = topology_role
                    break
        servers.append(server)
    return servers


def realized_stack_numa_mode_from_state(
    state: Mapping[str, Any],
    *,
    pid_alive: PidAliveFn | None = None,
) -> str | None:
    """Classify the realized NUMA mode from the launcher ``state``.

    Returns ``"full"`` / ``"quarter"`` / ``"both"`` derived from which
    quarterable-role ports are actually live, or ``None`` when the state carries
    no full/quarter discriminator (empty state, or only single-instance roles).
    Never coerces the unknown case to ``"full"``. Fix-2 writer call sites use
    this to pass an explicit realized mode.
    """
    is_alive = pid_alive or _default_pid_alive
    servers = _realized_selected_servers(state, is_alive=is_alive)
    return classify_numa_mode_from_ports({server["port"] for server in servers})


def _runtime_stack_summary(
    *,
    state: Mapping[str, Any],
    stack_numa_mode: str | None,
    stack_priors_path: Path,
    tmp_dir: Path | None,
    is_alive: PidAliveFn,
) -> dict[str, Any]:
    selected_servers = _realized_selected_servers(state, is_alive=is_alive)
    selected_ports = sorted({server["port"] for server in selected_servers})

    # Mode: honor an explicit caller-supplied mode; otherwise DERIVE from the
    # realized ports. Never coerce an unknown mode to "full" — a genuinely
    # undetermined mode is serialized as null, which every reader already treats
    # fail-safe (rejects the manifest → falls back to stack priors).
    if stack_numa_mode is not None:
        mode: str | None = normalize_stack_numa_mode(stack_numa_mode)
    else:
        mode = classify_numa_mode_from_ports(selected_ports)

    return {
        "stack_numa_mode": mode,
        "selected_servers": selected_servers,
        "selected_ports": selected_ports,
        "paths": _effective_paths_summary(
            stack_priors_path=stack_priors_path,
            tmp_dir=tmp_dir,
        ),
    }


def build_runtime_facts_manifest(
    *,
    state: Mapping[str, Any],
    launch_contracts: Mapping[str, Any],
    stack_priors_path: Path,
    stack_numa_mode: str | None = None,
    tmp_dir: Path | None = None,
    repo_short_sha: str | None = None,
    source: str,
    pid_alive: PidAliveFn | None = None,
) -> dict[str, Any]:
    """Build a serializable runtime-facts payload from existing sources.

    ``pid_alive`` is injectable so the realized-fleet derivation is testable
    without live processes; it defaults to a real ``os.kill(pid, 0)`` probe.
    """
    marker_dir = tmp_dir or _PATHS["tmp_dir"]
    is_alive = pid_alive or _default_pid_alive
    llama_markers = {
        str(port): metadata
        for port, metadata in sorted(discover_llama_markers(tmp_dir=marker_dir).items())
    }
    return {
        "schema": RUNTIME_FACTS_SCHEMA,
        "schema_version": RUNTIME_FACTS_MANIFEST_VERSION,
        "generated_at": _timestamp(),
        "source": str(source),
        "repo": {
            "short_sha": repo_short_sha,
        },
        "runtime_stack": _runtime_stack_summary(
            state=state,
            stack_numa_mode=stack_numa_mode,
            stack_priors_path=stack_priors_path,
            tmp_dir=tmp_dir,
            is_alive=is_alive,
        ),
        "stack_priors": _stack_priors_summary(stack_priors_path),
        "launch_contracts": _jsonable(launch_contracts),
        "state": _jsonable(state),
        "fleet_markers": {
            "orchestrator": read_orchestrator_marker_metadata(tmp_dir=marker_dir),
            "llama": llama_markers,
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_runtime_facts_manifest(
    *,
    state: Mapping[str, Any],
    launch_contracts: Mapping[str, Any],
    stack_priors_path: Path,
    stack_numa_mode: str | None = None,
    tmp_dir: Path | None = None,
    repo_short_sha: str | None = None,
    source: str,
    pid_alive: PidAliveFn | None = None,
) -> Path:
    """Write the runtime facts manifest atomically and return its path.

    ESC-8: the previous ``stack_numa_mode or read_runtime_stack_numa_mode(...)``
    carry-forward re-read the PREVIOUS manifest's mode when a caller passed no
    mode — the A1 self-perpetuation that carried a poisoned ``full`` forward
    across every stop/reload. It is removed: when no explicit mode is supplied,
    the mode is DERIVED from the realized ``state`` inside ``build_...`` (never
    coerced to ``full``).
    """
    path = runtime_facts_manifest_path(tmp_dir)
    payload = build_runtime_facts_manifest(
        state=state,
        launch_contracts=launch_contracts,
        stack_priors_path=stack_priors_path,
        stack_numa_mode=stack_numa_mode,
        tmp_dir=tmp_dir,
        repo_short_sha=repo_short_sha,
        source=source,
        pid_alive=pid_alive,
    )
    _atomic_write_json(path, payload)
    return path


def _load_manifest_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_header_is_valid(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema") == RUNTIME_FACTS_SCHEMA
        and payload.get("schema_version") == RUNTIME_FACTS_MANIFEST_VERSION
    )


def _manifest_is_stale(manifest_path: Path, state_file: Path | None) -> bool:
    if state_file is None or not state_file.exists():
        return False
    try:
        return manifest_path.stat().st_mtime < state_file.stat().st_mtime
    except OSError:
        return True


def read_runtime_stack_numa_mode(
    *,
    manifest_path: Path | None = None,
    state_file: Path | None = STATE_FILE,
) -> str | None:
    """Return the manifest's recorded stack NUMA mode when structurally valid."""
    path = manifest_path or runtime_facts_manifest_path()
    payload = _load_manifest_payload(path)
    if not payload or not _manifest_header_is_valid(payload):
        return None
    if _manifest_is_stale(path, state_file):
        return None
    runtime_stack = payload.get("runtime_stack")
    if not isinstance(runtime_stack, Mapping):
        return None
    mode = runtime_stack.get("stack_numa_mode")
    if not isinstance(mode, str):
        return None
    normalized = mode.strip().lower()
    return normalized if normalized in VALID_STACK_NUMA_MODES else None


def _normalized_selected_servers(runtime_stack: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    selected_servers = runtime_stack.get("selected_servers")
    selected_ports = runtime_stack.get("selected_ports")
    if not isinstance(selected_servers, list) or not isinstance(selected_ports, list):
        return None

    declared_ports: list[int] = []
    for port in selected_ports:
        if isinstance(port, bool) or not isinstance(port, int):
            return None
        declared_ports.append(port)

    normalized: list[dict[str, Any]] = []
    observed_ports: list[int] = []
    for server in selected_servers:
        if not isinstance(server, dict):
            return None
        port = server.get("port")
        roles = server.get("roles")
        if isinstance(port, bool) or not isinstance(port, int):
            return None
        if not isinstance(roles, list) or not roles:
            return None
        normalized_roles = [str(role) for role in roles if isinstance(role, str) and role]
        if len(normalized_roles) != len(roles):
            return None
        observed_ports.append(port)
        item = dict(server)
        item["port"] = port
        item["roles"] = normalized_roles
        normalized.append(item)

    if len(set(observed_ports)) != len(observed_ports):
        return None
    if sorted(observed_ports) != sorted(declared_ports):
        return None
    return normalized


def read_runtime_stack_selected_servers(
    *,
    manifest_path: Path | None = None,
    state_file: Path | None = STATE_FILE,
) -> list[dict[str, Any]] | None:
    """Read validated runtime-selected stack servers from the manifest.

    Returns None when the manifest is absent, malformed, stale relative to the
    launcher state file, or internally inconsistent. Callers should then fall
    back to their static source of truth instead of inferring runtime facts.
    """
    path = manifest_path or runtime_facts_manifest_path()
    payload = _load_manifest_payload(path)
    if not payload or not _manifest_header_is_valid(payload):
        return None
    if _manifest_is_stale(path, state_file):
        return None
    runtime_stack = payload.get("runtime_stack")
    if not isinstance(runtime_stack, Mapping):
        return None
    mode = runtime_stack.get("stack_numa_mode")
    if (
        not isinstance(mode, str)
        or read_runtime_stack_numa_mode(
            manifest_path=path,
            state_file=state_file,
        )
        is None
    ):
        return None
    return _normalized_selected_servers(runtime_stack)
