"""Fleet startup markers for orchestrator + per-port llama-server processes.

Each process that gets started by stack_commands writes a small marker file
BEFORE invoking subprocess.Popen. The marker file lives in /mnt/raid0/llm/tmp
(RAID0, no system tmpfs cleanup) and is read by the autopilot's
OrchestratorWatcher to detect operator-initiated restarts versus genuine
production crashes.

Marker file format (text, atomic-rename writes):

  Orchestrator marker (/mnt/raid0/llm/tmp/orchestrator_fleet_started_at):
    line 1: <float_epoch_seconds>

  Llama-server marker (/mnt/raid0/llm/tmp/llama_<port>_started_at):
    line 1: <float_epoch_seconds>
    line 2: <launch_source>             ∈ {stack_commands, external}
    line 3: <role>[,<role>,...]         canonical role names served by this process

The role list comes from the launch path; the watcher uses it to do
role→port lookups without consulting any other in-process mapping.

Atomic write semantics (temp + os.replace) are required: workers fork after
the marker is written; each worker reads the file independently at module
import. A partial read of a half-written file would yield inconsistent
server_started_at values across workers.

See handoffs/active/autopilot-exogenous-restart-resilience.md sections
5.1, 5.2, 6a for the design.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


_TMP_DIR_DEFAULT = Path("/mnt/raid0/llm/tmp")
ORCHESTRATOR_MARKER_NAME = "orchestrator_fleet_started_at"
LLAMA_MARKER_PATTERN = "llama_{port}_started_at"

# Valid launch_source tags. "external" is for future processes that start a
# llama-server outside stack_commands and want to declare themselves as such.
LAUNCH_SOURCE_STACK_COMMANDS = "stack_commands"
LAUNCH_SOURCE_EXTERNAL = "external"
_VALID_SOURCES = {LAUNCH_SOURCE_STACK_COMMANDS, LAUNCH_SOURCE_EXTERNAL}


def _tmp_dir() -> Path:
    """Return the configured tmp dir; falls back to /mnt/raid0/llm/tmp."""
    try:
        from src.config import get_config  # type: ignore[import-not-found]

        return get_config().paths.tmp_dir
    except Exception:
        return _TMP_DIR_DEFAULT


def orchestrator_marker_path(tmp_dir: Path | None = None) -> Path:
    """Path to the orchestrator fleet-startup marker."""
    return (tmp_dir or _tmp_dir()) / ORCHESTRATOR_MARKER_NAME


def llama_marker_path(port: int, tmp_dir: Path | None = None) -> Path:
    """Path to a llama-server's per-port fleet-startup marker."""
    return (tmp_dir or _tmp_dir()) / LLAMA_MARKER_PATTERN.format(port=port)


def _atomic_write(path: Path, content: str) -> None:
    """Write content atomically via temp file + os.replace.

    Required so that uvicorn workers (which fork BEFORE importing the
    app and then read this file independently) never see a partial
    write. Tested: workers without atomic writes can read 0-byte or
    half-written files and crash at import or carry inconsistent state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_orchestrator_marker(tmp_dir: Path | None = None) -> Path:
    """Write the orchestrator fleet-startup marker.

    Called by scripts/server/orchestrator_stack.start_orchestrator()
    BEFORE invoking uvicorn. All uvicorn workers will read this file
    during their independent module-import of dashboard.py.

    Returns the marker path so callers can log it.
    """
    path = orchestrator_marker_path(tmp_dir)
    _atomic_write(path, f"{time.time()}\n")
    return path


def write_llama_marker(
    port: int,
    roles: list[str],
    source: str = LAUNCH_SOURCE_STACK_COMMANDS,
    tmp_dir: Path | None = None,
) -> Path:
    """Write a per-port llama-server fleet-startup marker.

    Called by scripts/server/orchestrator_stack.start_server() in each of
    its four launch paths (vision / embedding / worker_pool / standard)
    BEFORE invoking subprocess.Popen.

    Returns the marker path.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"launch source must be one of {sorted(_VALID_SOURCES)}, got {source!r}"
        )
    path = llama_marker_path(port, tmp_dir)
    roles_line = ",".join(roles) if roles else ""
    _atomic_write(path, f"{time.time()}\n{source}\n{roles_line}\n")
    return path


def read_orchestrator_marker(tmp_dir: Path | None = None) -> float | None:
    """Read the orchestrator marker's startup timestamp.

    Returns None when the marker is missing or malformed. The dashboard
    /version endpoint uses this; on missing/malformed it falls back to
    the module-load time.time() for backward compatibility.
    """
    path = orchestrator_marker_path(tmp_dir)
    try:
        line = path.read_text().splitlines()[0].strip()
        return float(line)
    except Exception:
        return None


def read_llama_marker(
    port: int, tmp_dir: Path | None = None
) -> dict | None:
    """Read a llama-server marker.

    Returns {started_at: float, source: str, roles: list[str]} or None
    when missing/malformed. The /dashboard/api/llama_fleet_ids endpoint
    aggregates these into a single response.
    """
    path = llama_marker_path(port, tmp_dir)
    try:
        lines = path.read_text().splitlines()
        started_at = float(lines[0].strip())
        source = lines[1].strip() if len(lines) >= 2 else LAUNCH_SOURCE_STACK_COMMANDS
        roles_line = lines[2].strip() if len(lines) >= 3 else ""
        roles = [r for r in (roles_line.split(",") if roles_line else []) if r]
        return {
            "started_at": started_at,
            "source": source,
            "roles": roles,
        }
    except Exception:
        return None


def discover_llama_markers(tmp_dir: Path | None = None) -> dict[int, dict]:
    """Scan the tmp dir for all llama-server markers.

    Returns {port: {started_at, source, roles}}. Used by the dashboard's
    /llama_fleet_ids endpoint to surface fleet state to the watcher.
    """
    base = tmp_dir or _tmp_dir()
    out: dict[int, dict] = {}
    if not base.exists():
        return out
    prefix = "llama_"
    suffix = "_started_at"
    for entry in base.iterdir():
        name = entry.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        port_str = name[len(prefix):-len(suffix)]
        try:
            port = int(port_str)
        except ValueError:
            continue
        m = read_llama_marker(port, tmp_dir=base)
        if m is not None:
            out[port] = m
    return out
