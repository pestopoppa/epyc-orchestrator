"""AP-55: per-trial infrastructure fingerprint and explicit comparability marking.

WHY
---
A trial's quality/speed delta is only meaningful against a reference measured in
the SAME infrastructure regime. A public 28-candidate harness search
(``intake-1362#record``) saw its only clear pooled shift coincide with a provider
regime change, and same-commit re-runs moved 12.1% -> 8.7%. Nothing in an
AutoPilot journal row said which binary, recipe, model file, evaluator or host
configuration produced it, so a cross-regime comparison was silent.

WHAT
----
:func:`collect_infra_fingerprint` captures, without running inference or touching
any process, six components:

* ``orchestrator`` — git HEAD of the orchestrator tree plus a tracked-files dirty flag.
* ``evaluator``    — SHA-256 over the evaluator/scoring sources and eval registry.
* ``kernel``       — the llama-server executable and the ggml/llama shared
  libraries actually MAPPED by the running servers (``/proc/<pid>/exe`` and
  ``/proc/<pid>/maps`` for the PIDs recorded in the stack state file — never a
  process-name scan), each identified by SHA-256. ``ldd`` cannot prove the ggml
  generation because llama.cpp dlopens its backends; the live map can.
* ``recipe``       — SHA-256 over the launch recipe files (registry, launch
  manifest, stack topology).
* ``models``       — per role: GGUF path, size, and SHA-256 of the first MiB
  (header + metadata), which changes on any requantisation.
* ``host``         — kernel release, CPU model, logical CPUs, THP, NUMA
  balancing, cpufreq governor.

Each component gets its own digest, and ``digest`` is the digest over all of
them, so :func:`compare_infra_fingerprints` can name WHICH component moved.
Volatile facts (server PIDs, start times) are recorded under ``observed`` and
excluded from every digest: a restart of an identical server is the same regime.

A component that cannot be read is recorded as ``{"status": "unavailable"}``
rather than omitted or guessed; two unavailable components compare as unknown,
never as equal, so a missing read can never manufacture comparability.

The collector never raises: provenance must not fail a trial.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

INFRA_FINGERPRINT_SCHEMA_VERSION = 1

COMPARABLE = "COMPARABLE"
NON_COMPARABLE = "NON_COMPARABLE"
#: One side carries no fingerprint (legacy row, capture error, or a component
#: that could not be read on either side). Not evidence of comparability.
UNVERIFIED = "UNVERIFIED"

COMPONENTS: tuple[str, ...] = (
    "orchestrator",
    "evaluator",
    "kernel",
    "recipe",
    "models",
    "host",
)

_ORCH_ROOT = Path(__file__).resolve().parents[2]
_HEADER_BYTES = 1 << 20
_KERNEL_LIB_MARKERS = ("libggml", "libllama", "libmtmd")
_PID_START_TOLERANCE_S = 300.0

DEFAULT_EVALUATOR_PATHS: tuple[str, ...] = (
    "scripts/autopilot/eval_tower.py",
    "scripts/autopilot/rubric_scoring.py",
    "src/autopilot_core/tier_specs.py",
    "orchestration/eval_registry.yaml",
)
DEFAULT_RECIPE_PATHS: tuple[str, ...] = (
    "orchestration/model_registry.yaml",
    "orchestration/launch_manifest.yaml",
    "orchestration/stack_topology.yaml",
)

# (path, size, mtime_ns) -> sha256; binaries/libraries are tens of MB and a
# trial re-reads them every time, so hash each file version once per process.
_FILE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path, *, limit: int | None = None) -> str | None:
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), int(st.st_size), int(st.st_mtime_ns))
    if limit is None and key in _FILE_DIGEST_CACHE:
        return _FILE_DIGEST_CACHE[key]
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            if limit is not None:
                h.update(f.read(limit))
            else:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    except OSError:
        return None
    out = h.hexdigest()
    if limit is None:
        _FILE_DIGEST_CACHE[key] = out
    return out


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": str(reason)[:200]}


def _sealed(component: dict[str, Any]) -> dict[str, Any]:
    """Attach the component digest (over everything except volatile ``observed``)."""
    if component.get("status") == "unavailable":
        return component
    stable = {k: v for k, v in component.items() if k not in {"observed", "digest"}}
    component["digest"] = _digest(stable)
    return component


# ── components ────────────────────────────────────────────────────────────────


def _orchestrator_component(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty_out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable(f"git: {type(exc).__name__}")
    dirty_paths = sorted(line[3:] for line in dirty_out.splitlines() if line.strip())
    component: dict[str, Any] = {"commit": head, "dirty": bool(dirty_paths)}
    if dirty_paths:
        # A dirty tree is its own regime: bind the tracked modifications' content.
        component["dirty_digest"] = _digest(
            {p: _file_sha256(root / p) for p in dirty_paths}
        )
    return _sealed(component)


def _files_component(root: Path, rel_paths: Iterable[str]) -> dict[str, Any]:
    files: dict[str, str | None] = {}
    for rel in rel_paths:
        path = root / rel
        files[rel] = _file_sha256(path) if path.exists() else None
    if not any(files.values()):
        return _unavailable("no listed file readable")
    return _sealed({"files": files})


def _load_stack_state(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _server_rows(stack_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, row in stack_state.items():
        if not isinstance(row, Mapping):
            continue
        if "pid" not in row and "model_path" not in row:
            continue
        rows.append({"key": str(key), **dict(row)})
    return rows


def _proc_exe_and_libs(pid: int, proc_root: Path) -> tuple[str | None, list[str]]:
    exe: str | None = None
    try:
        exe = os.readlink(proc_root / str(pid) / "exe")
    except OSError:
        exe = None
    libs: set[str] = set()
    try:
        with open(proc_root / str(pid) / "maps") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) < 6:
                    continue
                mapped = parts[5].strip()
                name = Path(mapped.replace(" (deleted)", "")).name
                if any(name.startswith(marker) for marker in _KERNEL_LIB_MARKERS):
                    libs.add(mapped)
    except OSError:
        pass
    return exe, sorted(libs)


def _boot_time(proc_root: Path) -> float | None:
    try:
        with open(proc_root / "stat") as f:
            for line in f:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _pid_start_epoch(pid: int, proc_root: Path) -> float | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text()
        fields = raw.rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])  # field 22 overall; 20th after "pid (comm)"
    except (OSError, ValueError, IndexError):
        return None
    btime = _boot_time(proc_root)
    if btime is None:
        return None
    return btime + start_ticks / os.sysconf("SC_CLK_TCK")


def _pid_matches_row(pid: int, row: Mapping[str, Any], proc_root: Path) -> bool:
    """Guard against PID reuse: the process must have started when the row says.

    The stack state file outlives its servers; a recorded PID can belong to an
    unrelated process now. ``started_at`` is a naive local ISO time.
    """
    started = row.get("started_at")
    if not started:
        return True
    try:
        declared = datetime.fromisoformat(str(started))
    except ValueError:
        return True
    if declared.tzinfo is None:
        declared = declared.astimezone()  # naive -> local
    actual = _pid_start_epoch(pid, proc_root)
    if actual is None:
        return False
    return abs(actual - declared.timestamp()) <= _PID_START_TOLERANCE_S


def _kernel_component(
    stack_state: Mapping[str, Any] | None,
    *,
    fallback_binary: Path | None,
    proc_root: Path,
) -> dict[str, Any]:
    declared: dict[str, str | None] = {}
    if fallback_binary is not None and Path(fallback_binary).exists():
        declared[str(fallback_binary)] = _file_sha256(Path(fallback_binary))
    binaries: dict[str, str | None] = {}
    libraries: dict[str, str | None] = {}
    live_pids: list[int] = []
    stale_pids: list[int] = []
    seen: set[int] = set()
    for row in _server_rows(stack_state or {}):
        try:
            pid = int(row.get("pid"))
        except (TypeError, ValueError):
            continue
        if pid <= 0 or pid in seen:
            continue
        seen.add(pid)
        exe, libs = _proc_exe_and_libs(pid, proc_root)
        if exe is None or not _pid_matches_row(pid, row, proc_root):
            stale_pids.append(pid)
            continue
        if Path(exe.replace(" (deleted)", "")).name.startswith("python"):
            continue  # control-plane service, identified by the orchestrator component
        live_pids.append(pid)
        # A "(deleted)" executable means the on-disk binary was replaced under a
        # live server: the path no longer identifies what runs. Record it as such.
        binaries[exe] = None if exe.endswith(" (deleted)") else _file_sha256(Path(exe))
        for lib in libs:
            libraries[lib] = None if lib.endswith(" (deleted)") else _file_sha256(Path(lib))
    if not declared and not binaries:
        return _unavailable("no readable server process and no declared binary")
    return _sealed({
        "declared_binaries": declared,
        "live_binaries": dict(sorted(binaries.items())),
        "live_libraries": dict(sorted(libraries.items())),
        "observed": {"live_pids": sorted(live_pids), "stale_or_invisible_pids": sorted(stale_pids)},
    })


def _models_component(stack_state: Mapping[str, Any] | None) -> dict[str, Any]:
    roles: dict[str, dict[str, Any]] = {}
    started: dict[str, Any] = {}
    for row in _server_rows(stack_state or {}):
        model_path = row.get("model_path")
        if not model_path:
            continue
        # Key by port when present: two roles sharing one server are one identity.
        port = row.get("port")
        slot = f"{row.get('role') or row['key']}@{port}" if port is not None else row["key"]
        path = Path(str(model_path))
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = None
        roles[slot] = {
            "model_path": str(path),
            "size": size,
            "header_sha256": _file_sha256(path, limit=_HEADER_BYTES) if size else None,
        }
        started[slot] = row.get("started_at")
    if not roles:
        return _unavailable("stack state lists no model paths")
    return _sealed({"roles": dict(sorted(roles.items())), "observed": {"started_at": started}})


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu_model(proc_root: Path) -> str | None:
    try:
        with open(proc_root / "cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _host_component(proc_root: Path, sys_root: Path) -> dict[str, Any]:
    return _sealed({
        "hostname": socket.gethostname(),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(proc_root),
        "logical_cpus": os.cpu_count(),
        "thp_enabled": _read_text(str(sys_root / "kernel/mm/transparent_hugepage/enabled")),
        "thp_defrag": _read_text(str(sys_root / "kernel/mm/transparent_hugepage/defrag")),
        "numa_balancing": _read_text(str(proc_root / "sys/kernel/numa_balancing")),
        "cpufreq_governor": _read_text(
            str(sys_root / "devices/system/cpu/cpu0/cpufreq/scaling_governor")
        ),
    })


def _default_stack_state_path() -> Path | None:
    env = os.environ.get("AUTOPILOT_STACK_STATE_FILE")
    if env:
        return Path(env)
    log_dir = os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR")
    return (Path(log_dir) if log_dir else _ORCH_ROOT / "logs") / "orchestrator_state.json"


def _default_llama_server() -> Path | None:
    bin_dir = os.environ.get("ORCHESTRATOR_PATHS_LLAMA_CPP_BIN")
    base = Path(bin_dir) if bin_dir else Path("/mnt/raid0/llm/llama.cpp/build/bin")
    return base / "llama-server"


# ── public API ────────────────────────────────────────────────────────────────


def collect_infra_fingerprint(
    *,
    orchestrator_root: Path | None = None,
    stack_state_path: Path | None = None,
    stack_state: Mapping[str, Any] | None = None,
    evaluator_paths: Iterable[str] = DEFAULT_EVALUATOR_PATHS,
    recipe_paths: Iterable[str] = DEFAULT_RECIPE_PATHS,
    fallback_binary: Path | None = None,
    proc_root: Path = Path("/proc"),
    sys_root: Path = Path("/sys"),
) -> dict[str, Any]:
    """Capture the infra regime a trial ran in. Never raises; zero inference."""
    try:
        root = Path(orchestrator_root) if orchestrator_root else _ORCH_ROOT
        if stack_state is None:
            stack_state = _load_stack_state(
                stack_state_path if stack_state_path is not None else _default_stack_state_path()
            )
        binary = fallback_binary if fallback_binary is not None else _default_llama_server()
        components: dict[str, Any] = {}
        builders = {
            "orchestrator": lambda: _orchestrator_component(root),
            "evaluator": lambda: _files_component(root, evaluator_paths),
            "kernel": lambda: _kernel_component(
                stack_state, fallback_binary=binary, proc_root=proc_root
            ),
            "recipe": lambda: _files_component(root, recipe_paths),
            "models": lambda: _models_component(stack_state),
            "host": lambda: _host_component(proc_root, sys_root),
        }
        for name in COMPONENTS:
            try:
                components[name] = builders[name]()
            except Exception as exc:  # noqa: BLE001 - a component failure is data
                components[name] = _unavailable(f"{type(exc).__name__}: {exc}")
        return {
            "schema_version": INFRA_FINGERPRINT_SCHEMA_VERSION,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "components": components,
            "component_digests": {
                name: comp.get("digest") for name, comp in components.items()
            },
            "digest": _digest(
                {name: comp.get("digest") for name, comp in components.items()}
            ),
            "unavailable": sorted(
                name for name, comp in components.items()
                if comp.get("status") == "unavailable"
            ),
        }
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a trial
        return {
            "schema_version": INFRA_FINGERPRINT_SCHEMA_VERSION,
            "status": "capture_error",
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


# Components that define the serving REGIME for reproduction identity (gate-frontier,
# 2026-09-16). The orchestrator component (git HEAD + dirty digest) is excluded: every
# autopilot auto-commit, merge or doc commit moves HEAD, so including it would start a
# fresh reproduction cluster on each commit and the 3-reproduction bar could never be
# met. Orchestrator CODE and PROMPTS that a trial changed are identified instead by the
# served-file sha on the row (``served_content``). The evaluator component stays: it is
# the scoring instrument, and a scorer change is a different measurement.
REGIME_COMPONENTS: tuple[str, ...] = ("evaluator", "kernel", "recipe", "models", "host")


def regime_digest(fingerprint: Mapping[str, Any] | None) -> str:
    """Digest over the non-orchestrator components; "" when there is no fingerprint.

    An unreadable component is recorded by name ("unavailable"), so two rows that could
    not read the same component agree on it; the AP-55 comparability verdict (which
    reports such rows as UNVERIFIED) is the separate place where that is judged.
    """
    if not isinstance(fingerprint, Mapping):
        return ""
    digests = fingerprint.get("component_digests")
    if not isinstance(digests, Mapping) or not digests:
        return ""
    return _digest(
        {name: (digests.get(name) or "unavailable") for name in REGIME_COMPONENTS}
    )


def fingerprint_digest(fingerprint: Mapping[str, Any] | None) -> str:
    if not isinstance(fingerprint, Mapping):
        return ""
    value = fingerprint.get("digest")
    return str(value) if value else ""


def compare_infra_fingerprints(
    candidate: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    *,
    reference_label: str = "",
) -> dict[str, Any]:
    """Decide whether two measurements share one infra regime.

    ``NON_COMPARABLE`` when any component readable on BOTH sides differs.
    ``UNVERIFIED`` when either side has no fingerprint, or when no difference was
    found but some component could not be read on at least one side — absence
    of evidence is not comparability. ``COMPARABLE`` only when every component
    was read on both sides and all digests match.
    """
    out: dict[str, Any] = {
        "schema_version": INFRA_FINGERPRINT_SCHEMA_VERSION,
        "reference": reference_label,
        "candidate_digest": fingerprint_digest(candidate),
        "reference_digest": fingerprint_digest(reference),
        "differing_components": [],
        "unverified_components": [],
    }
    if not out["candidate_digest"] or not out["reference_digest"]:
        missing = []
        if not out["candidate_digest"]:
            missing.append("candidate")
        if not out["reference_digest"]:
            missing.append("reference")
        out["status"] = UNVERIFIED
        out["reason"] = "no infra fingerprint on " + " and ".join(missing)
        return out
    cand = candidate.get("component_digests") or {}
    ref = reference.get("component_digests") or {}
    for name in COMPONENTS:
        a, b = cand.get(name), ref.get(name)
        if not a or not b:
            out["unverified_components"].append(name)
        elif a != b:
            out["differing_components"].append(name)
    if out["differing_components"]:
        out["status"] = NON_COMPARABLE
        out["reason"] = "infra differs: " + ",".join(out["differing_components"])
    elif out["unverified_components"]:
        out["status"] = UNVERIFIED
        out["reason"] = "unreadable components: " + ",".join(out["unverified_components"])
    else:
        out["status"] = COMPARABLE
        out["reason"] = "all components match"
    return out


def group_by_infra_regime(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[Any]]:
    """Group journal rows (dicts) by fingerprint digest; ``""`` = unfingerprinted.

    A candidate batch must be compared within one key; a batch spanning keys is
    NON_COMPARABLE as a batch (AP-55 (c): no winner across regimes).
    """
    groups: dict[str, list[Any]] = {}
    for row in rows:
        key = fingerprint_digest(row.get("infra_fingerprint"))
        groups.setdefault(key, []).append(row.get("trial_id"))
    return groups
