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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY = Path("orchestration/model_registry.yaml")
DEFAULT_OUT_DIR = Path("orchestration/attestation")
DEFAULT_PROC_ROOT = Path("/proc")
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
                "state": status.get("State"),
                "dynamic_linking": run_dynamic_checks(exe),
            }
        )
    return processes


def summarize(processes: list[dict[str, Any]]) -> dict[str, Any]:
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
    generated_at: str | None = None,
) -> dict[str, Any]:
    registry_ports = load_registry_ports(registry)
    processes = collect_processes(proc_root=proc_root, registry_ports=registry_ports)
    report = {
        "schema_version": 1,
        "generated_at": generated_at or utc_now(),
        "scope": "W1_process_inventory",
        "sources": {
            "proc_root": str(proc_root),
            "registry": str(registry),
        },
        "registry_ports": {str(port): entries for port, entries in sorted(registry_ports.items())},
        "sections": {
            "processes": processes,
        },
        "summary": summarize(processes),
        "pending_sections": [
            "feature_flags_w2",
            "serving_config_w2",
            "eval_instrument_w3",
            "drift_w3",
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
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--print-md", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        registry=args.registry,
        proc_root=args.proc_root,
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
