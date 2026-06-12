#!/usr/bin/env python3
"""Generate a read-only running-state attestation artifact.

W1 scope: process inventory + binary/RUNPATH checks. Later waypoints append
feature-flag, serving-config, eval-instrument, drift, and backup sections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY = Path("orchestration/model_registry.yaml")
DEFAULT_OUT_DIR = Path("orchestration/attestation")
DEFAULT_PROC_ROOT = Path("/proc")
DEFAULT_CONFIG_URL = "http://127.0.0.1:8000"
DEFAULT_SENTINEL_FILES = (
    Path("orchestration/instrument_eras.yaml"),
    Path("scripts/autopilot/sentinel_questions.yaml"),
    Path("scripts/autopilot/tool_sentinels.yaml"),
    Path("orchestration/deep_research_sentinel.yaml"),
)
DEFAULT_GITNEXUS_REPOS = (
    Path("/mnt/raid0/llm/epyc-root"),
    Path("/mnt/raid0/llm/epyc-orchestrator"),
    Path("/mnt/raid0/llm/epyc-inference-research"),
    Path("/mnt/raid0/llm/llama.cpp"),
)
LLAMA_LIB_NAMES = ("libllama", "libggml", "libmtmd")
PROCESS_MARKERS = (
    "llama-server",
    "uvicorn",
    "lightonocr_llama_server.py",
    "whisper_server.py",
    "mcp_server.py",
    "autopilot.py",
)


def _arg_basename(value: str) -> str:
    return Path(value).name


def _has_arg_basename(cmdline: list[str], name: str) -> bool:
    return any(_arg_basename(arg) == name for arg in cmdline)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def file_attestation(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        return {
            "path": str(path),
            "exists": False,
            "error": str(exc),
            "sha256": None,
        }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "sha256": sha256_file(path),
    }


def read_cmdline(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> list[str]:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def read_exe(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> str | None:
    try:
        return os.readlink(proc_root / str(pid) / "exe")
    except OSError:
        return None


def read_status(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> dict[str, str]:
    try:
        lines = (proc_root / str(pid) / "status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    status: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        status[key.strip()] = value.strip()
    return status


def read_task_cpu_masks(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> dict[str, int]:
    task_root = proc_root / str(pid) / "task"
    masks: dict[str, int] = {}
    try:
        task_dirs = list(task_root.iterdir())
    except OSError:
        return masks
    for task_dir in task_dirs:
        if not task_dir.name.isdigit():
            continue
        try:
            status = read_status(int(task_dir.name), proc_root / str(pid) / "task")
        except OSError:
            continue
        mask = status.get("Cpus_allowed_list")
        if mask:
            masks[mask] = masks.get(mask, 0) + 1
    return dict(sorted(masks.items(), key=lambda item: (-item[1], item[0])))


def read_environ(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> dict[str, str]:
    try:
        raw = (proc_root / str(pid) / "environ").read_bytes()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for part in raw.split(b"\0"):
        if not part or b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def read_proc_start_time(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> str | None:
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        btime = None
        for line in (proc_root / "stat").read_text(encoding="utf-8").splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
                break
        if btime is None:
            return None
        # Field 2 may contain spaces inside parentheses. Everything after ") "
        # starts at field 3; process start ticks are field 22.
        after_comm = stat.rsplit(") ", 1)[1].split()
        start_ticks = int(after_comm[19])
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        ts = datetime.fromtimestamp(btime + (start_ticks / hz), tz=timezone.utc)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, ValueError, IndexError, KeyError):
        return None


def _flag_value(args: list[str], *names: str) -> str | None:
    for idx, item in enumerate(args):
        for name in names:
            if item == name and idx + 1 < len(args):
                return args[idx + 1]
            if item.startswith(f"{name}="):
                return item.split("=", 1)[1]
    return None


def _has_flag(args: list[str], name: str) -> bool:
    return name in args


def _env_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_cpu_list(value: str | None) -> set[int]:
    cpus: set[int] = set()
    if not value:
        return cpus
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if start.isdigit() and end.isdigit():
                cpus.update(range(int(start), int(end) + 1))
        elif part.isdigit():
            cpus.add(int(part))
    return cpus


def _format_cpu_set(cpus: set[int]) -> str:
    if not cpus:
        return ""
    ordered = sorted(cpus)
    ranges: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def classify_process(cmdline: list[str]) -> str | None:
    joined = " ".join(cmdline)
    if not joined:
        return None
    if _has_arg_basename(cmdline[:1], "llama-server"):
        return "llama_server"
    if "uvicorn" in joined and "src.api:app" in joined:
        return "orchestrator_api"
    if _has_arg_basename(cmdline, "lightonocr_llama_server.py"):
        return "lightonocr"
    if _has_arg_basename(cmdline, "whisper_server.py"):
        return "whisper"
    if _has_arg_basename(cmdline, "mcp_server.py"):
        return "mcp_server"
    if _has_arg_basename(cmdline, "autopilot.py"):
        return "autopilot"
    return None


def parse_process_args(cmdline: list[str]) -> dict[str, Any]:
    port = _flag_value(cmdline, "--port", "-p")
    info: dict[str, Any] = {
        "port": int(port) if port and port.isdigit() else None,
    }
    if classify_process(cmdline) == "llama_server":
        info.update(
            {
                "model_path": _flag_value(cmdline, "-m", "--model"),
                "draft_model_path": _flag_value(cmdline, "-md", "--model-draft"),
                "mmproj_path": _flag_value(cmdline, "--mmproj"),
                "parallel_slots": _flag_value(cmdline, "-np", "--parallel"),
                "context_length": _flag_value(cmdline, "-c", "--ctx-size"),
                "threads": _flag_value(cmdline, "-t", "--threads"),
                "ubatch_size": _flag_value(cmdline, "-ub", "--ubatch-size"),
                "kv_cache_type_k": _flag_value(cmdline, "-ctk", "--cache-type-k"),
                "kv_cache_type_v": _flag_value(cmdline, "-ctv", "--cache-type-v"),
                "spec_type": _flag_value(cmdline, "--spec-type"),
                "draft_max": _flag_value(cmdline, "--draft-max"),
                "flash_attention": _flag_value(cmdline, "--flash-attn"),
                "mlock": _has_flag(cmdline, "--mlock"),
                "no_mmap": _has_flag(cmdline, "--no-mmap"),
            }
        )
    return info


def _registry_entry(section: str, cfg: dict[str, Any], port_kind: str) -> dict[str, Any]:
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    model_name = model_cfg.get("name") if model_cfg else cfg.get("model")
    return {
        "role": section.rsplit(".", 1)[-1],
        "model_name": model_name,
        "model_path": model_cfg.get("path") if model_cfg else cfg.get("model_path"),
        "registry_section": section,
        "port_kind": port_kind,
    }


def _append_registry_port(
    ports: dict[int, list[dict[str, Any]]],
    port: int,
    entry: dict[str, Any],
) -> None:
    entries = ports.setdefault(port, [])
    if entry not in entries:
        entries.append(entry)


def load_registry_ports(path: Path) -> dict[int, list[dict[str, Any]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ports: dict[int, list[dict[str, Any]]] = {}

    def walk(node: Any, path_parts: list[str]) -> None:
        if not isinstance(node, dict):
            return
        section = ".".join(path_parts)
        if section:
            port = node.get("port")
            if isinstance(port, int):
                _append_registry_port(ports, port, _registry_entry(section, node, "primary"))
            for numa_port in node.get("numa_ports") or []:
                if isinstance(numa_port, int):
                    _append_registry_port(
                        ports,
                        numa_port,
                        _registry_entry(section, node, "numa_replica"),
                    )
        for key, value in node.items():
            walk(value, [*path_parts, str(key)])

    walk(data, [])
    return ports


def load_numa_ports() -> dict[int, dict[str, Any]]:
    try:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
    except Exception:
        return {}
    ports: dict[int, dict[str, Any]] = {}
    for role, cfg in NUMA_CONFIG.items():
        if not isinstance(cfg, dict):
            continue
        policy = cfg.get("numactl_policy")
        for idx, instance in enumerate(cfg.get("instances") or []):
            if len(instance) < 3:
                continue
            cpu_list, port, threads = instance[:3]
            if isinstance(port, int):
                ports[port] = {
                    "role": role,
                    "instance_idx": idx,
                    "cpu_list": str(cpu_list),
                    "threads": int(threads),
                    "numactl_policy": policy,
                }
    return ports


def load_declared_feature_env() -> dict[str, Any]:
    try:
        from scripts.server.orchestrator_stack import _production_feature_env
        from src.features import _FEATURE_REGISTRY
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": str(exc),
            "env": {},
            "flag_env_names": {},
            "flags": {},
        }

    env = _production_feature_env()
    flag_env_names = {
        spec.name: f"ORCHESTRATOR_FEATURE_{spec.env_var}" for spec in _FEATURE_REGISTRY
    }
    flags = {
        name: _env_bool(env.get(env_name))
        for name, env_name in flag_env_names.items()
        if env_name in env
    }
    return {
        "status": "ok",
        "env": env,
        "flag_env_names": flag_env_names,
        "flags": flags,
    }


def _fetch_json(url: str, timeout_s: float = 2.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def _flag_heterogeneity(workers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    all_flags = sorted(
        {name for data in workers.values() for name in (data.get("flags", {}) or {}).keys()}
    )
    hetero: dict[str, dict[str, Any]] = {}
    for name in all_flags:
        values = {
            pid: (data.get("flags", {}) or {}).get(name)
            for pid, data in workers.items()
            if not data.get("error")
        }
        if values and len(set(values.values())) > 1:
            hetero[name] = values
    return hetero


def _expected_flag_diffs(
    workers: dict[str, dict[str, Any]],
    expected: dict[str, bool | None],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for pid, data in workers.items():
        if data.get("error"):
            continue
        flags = data.get("flags", {}) or {}
        sources = data.get("sources", {}) or {}
        for name, expected_value in expected.items():
            if expected_value is None:
                continue
            actual = flags.get(name)
            if actual != expected_value:
                diffs.append(
                    {
                        "pid": pid,
                        "flag": name,
                        "expected": expected_value,
                        "actual": actual,
                        "source": sources.get(name),
                    }
                )
    return diffs


def _worker_env_diffs(
    worker_env: dict[str, dict[str, str]],
    expected_env: dict[str, str],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for pid, env in worker_env.items():
        for name, expected in expected_env.items():
            actual = env.get(name)
            if actual != expected:
                diffs.append(
                    {
                        "pid": pid,
                        "env": name,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return diffs


def collect_feature_flags(
    *,
    config_url: str = DEFAULT_CONFIG_URL,
    polls: int = 0,
    delay_s: float = 0.05,
    min_workers: int = 1,
    proc_root: Path = DEFAULT_PROC_ROOT,
) -> dict[str, Any]:
    declared = load_declared_feature_env()
    endpoint = f"{config_url.rstrip('/')}/config/attest"
    workers: dict[str, dict[str, Any]] = {}
    if polls <= 0:
        return {
            "status": "disabled",
            "endpoint": endpoint,
            "polls": polls,
            "min_workers": min_workers,
            "declared": declared,
            "workers_seen": 0,
            "workers": {},
            "worker_env": {},
            "heterogeneous": {},
            "intent_diffs": [],
            "env_diffs": [],
            "errors": {},
            "too_few_workers": False,
        }

    for idx in range(polls):
        try:
            data = _fetch_json(endpoint)
        except Exception as exc:
            data = {"pid": f"error-{idx}", "error": str(exc), "flags": {}, "sources": {}}
        pid = str(data.get("pid") or f"unknown-{idx}")
        workers[pid] = data
        if delay_s > 0 and idx + 1 < polls:
            time.sleep(delay_s)

    worker_env: dict[str, dict[str, str]] = {}
    expected_env = declared.get("env", {}) or {}
    selected_env_names = set(expected_env)
    selected_env_names.add("ORCHESTRATOR_RUNTIME_FLAGS_PATH")
    for pid in workers:
        if not pid.isdigit():
            continue
        env = read_environ(int(pid), proc_root)
        worker_env[pid] = {
            key: value
            for key, value in sorted(env.items())
            if key in selected_env_names or key.startswith("ORCHESTRATOR_FEATURE_")
        }

    errors = {pid: data.get("error") for pid, data in workers.items() if data.get("error")}
    hetero = _flag_heterogeneity(workers)
    expected_flags = declared.get("flags", {}) or {}
    intent_diffs = _expected_flag_diffs(workers, expected_flags)
    env_diffs = _worker_env_diffs(worker_env, expected_env)
    too_few_workers = len([pid for pid in workers if pid.isdigit()]) < min_workers
    status = "ok"
    if errors or hetero or intent_diffs or env_diffs or too_few_workers:
        status = "warn"
    return {
        "status": status,
        "endpoint": endpoint,
        "polls": polls,
        "min_workers": min_workers,
        "declared": declared,
        "workers_seen": len(workers),
        "workers": workers,
        "worker_env": worker_env,
        "heterogeneous": hetero,
        "intent_diffs": intent_diffs,
        "env_diffs": env_diffs,
        "errors": errors,
        "too_few_workers": too_few_workers,
    }


def collect_eval_instrument(
    processes: list[dict[str, Any]],
    *,
    proc_root: Path = DEFAULT_PROC_ROOT,
    sentinel_paths: tuple[Path, ...] = DEFAULT_SENTINEL_FILES,
) -> dict[str, Any]:
    files = [file_attestation(path) for path in sentinel_paths]
    process_env: list[dict[str, Any]] = []
    missing_tool_sentinel_env: list[dict[str, Any]] = []
    for proc in processes:
        if proc["kind"] not in {"autopilot", "orchestrator_api"}:
            continue
        env = read_environ(proc["pid"], proc_root)
        value = env.get("AUTOPILOT_TOOL_SENTINELS")
        row = {
            "pid": proc["pid"],
            "kind": proc["kind"],
            "has_AUTOPILOT_TOOL_SENTINELS": value is not None,
            "AUTOPILOT_TOOL_SENTINELS": value,
        }
        process_env.append(row)
        if value is None:
            missing_tool_sentinel_env.append(
                {
                    "pid": proc["pid"],
                    "kind": proc["kind"],
                }
            )
    missing_files = [item for item in files if not item.get("exists")]
    status = "ok" if not missing_files and not missing_tool_sentinel_env else "warn"
    return {
        "status": status,
        "files": files,
        "process_env": process_env,
        "missing_files": missing_files,
        "missing_tool_sentinel_env": missing_tool_sentinel_env,
    }


def _parse_gitnexus_status(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"raw": output}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        parsed[key] = value.strip()
    indexed = parsed.get("indexed_commit")
    current = parsed.get("current_commit")
    parsed["stale"] = bool(indexed and current and indexed != current)
    return parsed


def collect_drift_checks(
    *,
    gitnexus_repos: tuple[Path, ...] = DEFAULT_GITNEXUS_REPOS,
) -> dict[str, Any]:
    gitnexus: list[dict[str, Any]] = []
    for repo in gitnexus_repos:
        entry: dict[str, Any] = {"repo": str(repo)}
        try:
            result = subprocess.run(
                ["gitnexus", "status"],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            entry["returncode"] = result.returncode
            entry.update(_parse_gitnexus_status(result.stdout.strip()))
            if result.stderr.strip():
                entry["stderr"] = result.stderr.strip()
        except Exception as exc:
            entry.update({"returncode": None, "error": str(exc), "stale": True})
        gitnexus.append(entry)
    stale = [
        item
        for item in gitnexus
        if item.get("stale") or item.get("returncode") not in (0, None) or item.get("error")
    ]
    return {
        "status": "ok" if not stale else "warn",
        "gitnexus": gitnexus,
        "stale_or_error": stale,
    }


def parse_readelf_dynamic(output: str) -> dict[str, Any]:
    needed: list[str] = []
    rpaths: list[str] = []
    runpaths: list[str] = []
    for line in output.splitlines():
        bracketed = re.search(r"\[(.*?)\]", line)
        value = bracketed.group(1) if bracketed else ""
        if "(NEEDED)" in line and value:
            needed.append(value)
        elif "(RPATH)" in line and value:
            rpaths.extend(part for part in value.split(":") if part)
        elif "(RUNPATH)" in line and value:
            runpaths.extend(part for part in value.split(":") if part)
    return {"needed": needed, "rpath": rpaths, "runpath": runpaths}


def parse_ldd(output: str) -> dict[str, str | None]:
    libs: dict[str, str | None] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=>" in stripped:
            name, rest = stripped.split("=>", 1)
            name = name.strip()
            target = rest.strip().split()[0]
            libs[name] = None if target == "not" else target
        else:
            parts = stripped.split()
            if parts:
                libs[parts[0]] = parts[0] if parts[0].startswith("/") else None
    return libs


def llama_tree(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        if part.endswith("llama.cpp"):
            return str(Path(*parts[: idx + 1]))
    return None


def run_dynamic_checks(exe_path: str | None) -> dict[str, Any]:
    if not exe_path:
        return {
            "status": "unknown",
            "issues": ["exe_path_unreadable"],
            "readelf": {},
            "ldd": {},
        }
    path = Path(exe_path)
    checks: dict[str, Any] = {"status": "ok", "issues": []}
    try:
        readelf = subprocess.run(
            ["readelf", "-d", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        checks["readelf_returncode"] = readelf.returncode
        checks["readelf"] = parse_readelf_dynamic(readelf.stdout)
        if readelf.returncode != 0:
            checks["issues"].append("readelf_failed_or_not_elf")
    except Exception as exc:
        checks["readelf"] = {}
        checks["issues"].append(f"readelf_error:{exc}")

    try:
        ldd = subprocess.run(
            ["ldd", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        checks["ldd_returncode"] = ldd.returncode
        libs = parse_ldd(ldd.stdout)
        checks["ldd"] = libs
        checks["llama_resolution"] = llama_resolution_status(str(path), libs)
        if checks["llama_resolution"]["issues"]:
            checks["issues"].extend(checks["llama_resolution"]["issues"])
    except Exception as exc:
        checks["ldd"] = {}
        checks["issues"].append(f"ldd_error:{exc}")

    if checks["issues"]:
        checks["status"] = "warn"
    return checks


def llama_resolution_status(exe_path: str, libs: dict[str, str | None]) -> dict[str, Any]:
    expected_tree = llama_tree(exe_path)
    resolved = {
        name: target
        for name, target in libs.items()
        if any(name.startswith(prefix) for prefix in LLAMA_LIB_NAMES)
    }
    issues: list[str] = []
    for name, target in resolved.items():
        if target is None:
            issues.append(f"{name}_not_found")
            continue
        target_tree = llama_tree(target)
        if expected_tree and target_tree and target_tree != expected_tree:
            issues.append(f"{name}_tree_mismatch:{target_tree}")
    return {
        "expected_tree": expected_tree,
        "resolved": resolved,
        "issues": issues,
    }


def collect_processes(
    *,
    proc_root: Path = DEFAULT_PROC_ROOT,
    registry_ports: dict[int, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    registry_ports = registry_ports or {}
    processes: list[dict[str, Any]] = []
    for pid_dir in sorted(
        proc_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1
    ):
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        cmdline = read_cmdline(pid, proc_root)
        kind = classify_process(cmdline)
        if not kind:
            continue
        exe = read_exe(pid, proc_root)
        status = read_status(pid, proc_root)
        args = parse_process_args(cmdline)
        registry_matches = (
            registry_ports.get(args["port"], []) if args.get("port") is not None else []
        )
        binary_sha = sha256_file(Path(exe)) if exe else None
        processes.append(
            {
                "pid": pid,
                "kind": kind,
                "start_time": read_proc_start_time(pid, proc_root),
                "exe": exe,
                "binary_sha256": binary_sha,
                "cmdline": cmdline,
                "port": args.get("port"),
                "registry_matches": registry_matches,
                "args": args,
                "cpus_allowed_list": status.get("Cpus_allowed_list"),
                "task_cpu_masks": read_task_cpu_masks(pid, proc_root),
                "state": status.get("State"),
                "dynamic_linking": run_dynamic_checks(exe),
            }
        )
    return processes


def build_serving_config(
    processes: list[dict[str, Any]],
    *,
    numa_ports: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    numa_ports = numa_ports or {}
    rows: list[dict[str, Any]] = []
    for proc in processes:
        if proc["kind"] != "llama_server":
            continue
        port = proc.get("port")
        args = proc.get("args") or {}
        numa_intent = numa_ports.get(port) if isinstance(port, int) else None
        cpus_allowed = proc.get("cpus_allowed_list")
        task_cpu_masks = proc.get("task_cpu_masks") or {}
        task_cpu_union: set[int] = set()
        for mask in task_cpu_masks:
            task_cpu_union.update(_parse_cpu_list(mask))
        task_cpu_union_text = _format_cpu_set(task_cpu_union)
        expected_cpus = _parse_cpu_list(numa_intent.get("cpu_list") if numa_intent else None)
        process_cpus = _parse_cpu_list(cpus_allowed)
        numa_match = None
        if numa_intent and cpus_allowed:
            numa_match = (
                process_cpus == expected_cpus
                or task_cpu_union == expected_cpus
                or numa_intent.get("cpu_list") in task_cpu_masks
            )
        rows.append(
            {
                "pid": proc["pid"],
                "port": port,
                "registry_matches": proc.get("registry_matches") or [],
                "model_path": args.get("model_path"),
                "draft_model_path": args.get("draft_model_path"),
                "mmproj_path": args.get("mmproj_path"),
                "parallel_slots": args.get("parallel_slots"),
                "context_length": args.get("context_length"),
                "threads": args.get("threads"),
                "ubatch_size": args.get("ubatch_size"),
                "kv_cache_type_k": args.get("kv_cache_type_k"),
                "kv_cache_type_v": args.get("kv_cache_type_v"),
                "spec_type": args.get("spec_type"),
                "draft_max": args.get("draft_max"),
                "flash_attention": args.get("flash_attention"),
                "mlock": args.get("mlock"),
                "no_mmap": args.get("no_mmap"),
                "cpus_allowed_list": cpus_allowed,
                "task_cpu_masks": task_cpu_masks,
                "task_cpu_union": task_cpu_union_text,
                "numa_intent": numa_intent,
                "numa_match": numa_match,
            }
        )
    return rows


def summarize(
    processes: list[dict[str, Any]],
    *,
    feature_flags: dict[str, Any] | None = None,
    serving_config: list[dict[str, Any]] | None = None,
    eval_instrument: dict[str, Any] | None = None,
    drift: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    by_kind: dict[str, int] = {}
    for proc in processes:
        by_kind[proc["kind"]] = by_kind.get(proc["kind"], 0) + 1
        dyn = proc.get("dynamic_linking") or {}
        if dyn.get("issues"):
            issues.append(
                {
                    "pid": proc["pid"],
                    "kind": proc["kind"],
                    "port": proc.get("port"),
                    "issues": dyn["issues"],
                }
            )
        if proc["kind"] == "llama_server" and proc.get("port") and not proc.get("registry_matches"):
            issues.append(
                {
                    "pid": proc["pid"],
                    "kind": proc["kind"],
                    "port": proc.get("port"),
                    "issues": ["port_not_found_in_registry"],
                }
            )
    for row in serving_config or []:
        if row.get("numa_match") is False:
            issues.append(
                {
                    "pid": row["pid"],
                    "kind": "llama_server",
                    "port": row.get("port"),
                    "issues": [
                        "cpu_affinity_mismatch:"
                        f"actual={row.get('cpus_allowed_list')}:"
                        f"task_union={row.get('task_cpu_union')}:"
                        f"expected={row.get('numa_intent', {}).get('cpu_list')}"
                    ],
                }
            )
    if feature_flags and feature_flags.get("status") == "warn":
        flag_issues: list[str] = []
        if feature_flags.get("errors"):
            flag_issues.append(f"flag_endpoint_errors={len(feature_flags['errors'])}")
        if feature_flags.get("too_few_workers"):
            flag_issues.append(
                f"flag_workers_seen={feature_flags.get('workers_seen')}"
                f"<{feature_flags.get('min_workers')}"
            )
        if feature_flags.get("heterogeneous"):
            flag_issues.append(f"heterogeneous_flags={len(feature_flags['heterogeneous'])}")
        if feature_flags.get("intent_diffs"):
            flag_issues.append(f"flag_intent_diffs={len(feature_flags['intent_diffs'])}")
        if feature_flags.get("env_diffs"):
            flag_issues.append(f"feature_env_diffs={len(feature_flags['env_diffs'])}")
        issues.append(
            {
                "pid": "",
                "kind": "feature_flags",
                "port": "",
                "issues": flag_issues or ["feature_flag_attestation_warn"],
            }
        )
    if eval_instrument and eval_instrument.get("status") == "warn":
        eval_issues: list[str] = []
        if eval_instrument.get("missing_files"):
            eval_issues.append(f"missing_instrument_files={len(eval_instrument['missing_files'])}")
        if eval_instrument.get("missing_tool_sentinel_env"):
            eval_issues.append(
                "missing_AUTOPILOT_TOOL_SENTINELS="
                f"{len(eval_instrument['missing_tool_sentinel_env'])}"
            )
        issues.append(
            {
                "pid": "",
                "kind": "eval_instrument",
                "port": "",
                "issues": eval_issues or ["eval_instrument_warn"],
            }
        )
    if drift and drift.get("status") == "warn":
        issues.append(
            {
                "pid": "",
                "kind": "drift",
                "port": "",
                "issues": [f"gitnexus_stale_or_error={len(drift.get('stale_or_error') or [])}"],
            }
        )
    return {
        "process_count": len(processes),
        "by_kind": dict(sorted(by_kind.items())),
        "issue_count": len(issues),
        "issues": issues,
    }


def build_report(
    *,
    registry: Path = DEFAULT_REGISTRY,
    proc_root: Path = DEFAULT_PROC_ROOT,
    config_url: str = DEFAULT_CONFIG_URL,
    flag_polls: int = 0,
    flag_delay_s: float = 0.05,
    flag_min_workers: int = 1,
    gitnexus_repos: tuple[Path, ...] = (),
    generated_at: str | None = None,
) -> dict[str, Any]:
    registry_ports = load_registry_ports(registry)
    processes = collect_processes(proc_root=proc_root, registry_ports=registry_ports)
    numa_ports = load_numa_ports()
    serving_config = build_serving_config(processes, numa_ports=numa_ports)
    feature_flags = collect_feature_flags(
        config_url=config_url,
        polls=flag_polls,
        delay_s=flag_delay_s,
        min_workers=flag_min_workers,
        proc_root=proc_root,
    )
    eval_instrument = collect_eval_instrument(processes, proc_root=proc_root)
    drift = collect_drift_checks(gitnexus_repos=gitnexus_repos)
    report = {
        "schema_version": 3,
        "generated_at": generated_at or utc_now(),
        "scope": "W1_W2_W3_process_flags_serving_eval_drift",
        "sources": {
            "proc_root": str(proc_root),
            "registry": str(registry),
            "config_url": config_url,
        },
        "registry_ports": {str(port): entries for port, entries in sorted(registry_ports.items())},
        "numa_ports": {str(port): entry for port, entry in sorted(numa_ports.items())},
        "sections": {
            "processes": processes,
            "feature_flags": feature_flags,
            "serving_config": serving_config,
            "eval_instrument": eval_instrument,
            "drift": drift,
        },
        "summary": summarize(
            processes,
            feature_flags=feature_flags,
            serving_config=serving_config,
            eval_instrument=eval_instrument,
            drift=drift,
        ),
        "pending_sections": [
            "backup_w3",
            "cadence_consumers_w4",
        ],
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Running-State Attestation",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Scope: `{report['scope']}`",
        f"Processes: `{summary['process_count']}`",
        f"Issues: `{summary['issue_count']}`",
        "",
        "## Process Summary",
        "",
        "| kind | count |",
        "|---|---:|",
    ]
    for kind, count in summary["by_kind"].items():
        lines.append(f"| {kind} | {count} |")
    lines.extend(
        [
            "",
            "## Processes",
            "",
            "| pid | kind | port | registry | exe | sha256[:12] | link status |",
        ]
    )
    lines.append("|---:|---|---:|---|---|---|---|")
    for proc in report["sections"]["processes"]:
        matches = proc.get("registry_matches") or []
        registry = ", ".join(dict.fromkeys(match["registry_section"] for match in matches))
        sha = (proc.get("binary_sha256") or "")[:12]
        dyn = proc.get("dynamic_linking") or {}
        link_status = dyn.get("status", "unknown")
        lines.append(
            "| {pid} | {kind} | {port} | {registry} | `{exe}` | {sha} | {status} |".format(
                pid=proc["pid"],
                kind=proc["kind"],
                port=proc.get("port") if proc.get("port") is not None else "",
                registry=registry,
                exe=proc.get("exe") or "",
                sha=sha,
                status=link_status,
            )
        )
    feature_flags = report["sections"].get("feature_flags") or {}
    lines.extend(
        [
            "",
            "## Feature Flags",
            "",
            f"Status: `{feature_flags.get('status', 'unknown')}`",
            f"Endpoint: `{feature_flags.get('endpoint', '')}`",
            f"Workers seen: `{feature_flags.get('workers_seen', 0)}`",
            f"Heterogeneous flags: `{len(feature_flags.get('heterogeneous') or {})}`",
            f"Intent diffs: `{len(feature_flags.get('intent_diffs') or [])}`",
            f"Env diffs: `{len(feature_flags.get('env_diffs') or [])}`",
            "",
        ]
    )
    workers = feature_flags.get("workers") or {}
    if workers:
        lines.extend(["| pid | error | enabled flags |", "|---:|---|---:|"])
        for pid, worker in sorted(workers.items()):
            enabled = sum(1 for value in (worker.get("flags") or {}).values() if value is True)
            lines.append(
                "| {pid} | {error} | {enabled} |".format(
                    pid=pid,
                    error=worker.get("error") or "",
                    enabled=enabled,
                )
            )
        lines.append("")
    serving = report["sections"].get("serving_config") or []
    lines.extend(
        [
            "## Serving Config",
            "",
            "| pid | port | registry | model | draft | ctx | threads | proc cpus | task union | cpu intent | numa |",
            "|---:|---:|---|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for row in serving:
        matches = row.get("registry_matches") or []
        registry = ", ".join(dict.fromkeys(match["registry_section"] for match in matches))
        intent = row.get("numa_intent") or {}
        lines.append(
            "| {pid} | {port} | {registry} | `{model}` | `{draft}` | {ctx} | {threads} | {cpus} | {task_union} | {intent} | {numa} |".format(
                pid=row["pid"],
                port=row.get("port") if row.get("port") is not None else "",
                registry=registry,
                model=row.get("model_path") or "",
                draft=row.get("draft_model_path") or "",
                ctx=row.get("context_length") or "",
                threads=row.get("threads") or "",
                cpus=row.get("cpus_allowed_list") or "",
                task_union=row.get("task_cpu_union") or "",
                intent=intent.get("cpu_list") or "",
                numa=(
                    "n/a"
                    if row.get("numa_match") is None
                    else ("ok" if row.get("numa_match") else "mismatch")
                ),
            )
        )
    eval_instrument = report["sections"].get("eval_instrument") or {}
    lines.extend(
        [
            "",
            "## Eval Instrument",
            "",
            f"Status: `{eval_instrument.get('status', 'unknown')}`",
            "",
            "| file | exists | sha256[:12] | mtime |",
            "|---|---|---|---|",
        ]
    )
    for item in eval_instrument.get("files") or []:
        lines.append(
            "| `{path}` | {exists} | {sha} | {mtime} |".format(
                path=item.get("path", ""),
                exists=item.get("exists"),
                sha=(item.get("sha256") or "")[:12],
                mtime=item.get("mtime") or "",
            )
        )
    process_env = eval_instrument.get("process_env") or []
    if process_env:
        lines.extend(
            [
                "",
                "| pid | kind | AUTOPILOT_TOOL_SENTINELS |",
                "|---:|---|---|",
            ]
        )
        for item in process_env:
            lines.append(
                "| {pid} | {kind} | `{value}` |".format(
                    pid=item.get("pid", ""),
                    kind=item.get("kind", ""),
                    value=item.get("AUTOPILOT_TOOL_SENTINELS") or "",
                )
            )
    drift = report["sections"].get("drift") or {}
    lines.extend(
        [
            "",
            "## Drift",
            "",
            f"Status: `{drift.get('status', 'unknown')}`",
            "",
            "| repo | indexed | current | stale | status |",
            "|---|---|---|---|---|",
        ]
    )
    for item in drift.get("gitnexus") or []:
        lines.append(
            "| `{repo}` | {indexed} | {current} | {stale} | {status} |".format(
                repo=item.get("repo", ""),
                indexed=item.get("indexed_commit", ""),
                current=item.get("current_commit", ""),
                stale=item.get("stale", ""),
                status=item.get("status", ""),
            )
        )
    if summary["issues"]:
        lines.extend(["", "## Issues", "", "| pid | kind | port | issues |", "|---:|---|---:|---|"])
        for issue in summary["issues"]:
            lines.append(
                "| {pid} | {kind} | {port} | {issues} |".format(
                    pid=issue["pid"],
                    kind=issue["kind"],
                    port=issue.get("port") if issue.get("port") is not None else "",
                    issues=", ".join(issue["issues"]),
                )
            )
    lines.extend(
        [
            "",
            "## Pending Sections",
            "",
            ", ".join(report["pending_sections"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latest.json"
    md_path = out_dir / "latest.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--proc-root", type=Path, default=DEFAULT_PROC_ROOT)
    parser.add_argument("--config-url", default=DEFAULT_CONFIG_URL)
    parser.add_argument("--flag-polls", type=int, default=120)
    parser.add_argument("--flag-delay-s", type=float, default=0.05)
    parser.add_argument("--flag-min-workers", type=int, default=1)
    parser.add_argument(
        "--gitnexus-repo",
        action="append",
        type=Path,
        default=None,
        help="Repository path to include in GitNexus freshness checks; repeatable.",
    )
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--print-md", action="store_true")
    args = parser.parse_args(argv)
    gitnexus_repos = tuple(args.gitnexus_repo or DEFAULT_GITNEXUS_REPOS)

    report = build_report(
        registry=args.registry,
        proc_root=args.proc_root,
        config_url=args.config_url,
        flag_polls=args.flag_polls,
        flag_delay_s=args.flag_delay_s,
        flag_min_workers=args.flag_min_workers,
        gitnexus_repos=gitnexus_repos,
        generated_at=args.generated_at,
    )
    json_path, md_path = write_report(report, args.out_dir)
    if args.print_md:
        print(render_markdown(report), end="")
    else:
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
    return 1 if report["summary"]["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
