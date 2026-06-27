#!/usr/bin/env python3
"""Live-affinity preflight + artifact (bulk-inference campaign hard gate).

`topology_hash` certifies the *intended* NUMA_CONFIG; it does NOT certify that the live
llama-server processes are actually pinned to those CPUs. A launcher bug (2026-05-26) pinned
worker_general/vision quarters to the wrong (overlapping) cores while -t was correct, invalidating
any concurrency measurement. This tool closes that gap: for every NUMA_CONFIG (role, instance),
it finds the live port's process, computes the UNION of its threads' CPU affinity (the real
footprint — the main thread alone is misleading because idle threads cluster on the first core),
and checks it EXACTLY matches the configured cpuset.

It also records `/proc/<pid>/numa_maps` placement for model-weight pages. Shared-mmap model pages
are reported for diagnosis; strict locality is opt-in because most production quarter roles are
currently deliberately shared-mmap. Use `--require-memory-locality` when validating a private
`--no-mmap` role flip: single-quarter private copies must have at least
`--memory-locality-threshold` of observed weight/proxy pages on the bound NUMA node.

Emits `data/contention_matrix/affinity_preflight_<ts>.json` with per-port {expected, observed,
match} + a top-level `live_affinity_verified` bool. J4/J5/J6 + any concurrent run must require
`live_affinity_verified=true` against a fresh artifact for the current launch.

Exit 0 iff all live instances match. Usage: python3 scripts/server/affinity_preflight.py [--roles a b]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ORCH = Path(__file__).resolve().parents[2]
NODE_CPUSETS = {
    0: "0-23,96-119",
    1: "24-47,120-143",
    2: "48-71,144-167",
    3: "72-95,168-191",
}
NODE_RE = re.compile(r"\bN(?P<node>\d+)=(?P<pages>\d+)\b")


def _parse_cpulist(s: str) -> set[int]:
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _pid_on_port(port: int) -> str | None:
    r = subprocess.run(
        ["bash", "-c",
         f"ps -eo pid,args | grep -E 'llama-server|ik_llama' | grep -- '--port {port}' "
         f"| grep -v grep | awk '{{print $1}}' | head -1"],
        capture_output=True, text=True)
    return r.stdout.strip() or None


def _thread_union(pid: str) -> set[int]:
    cpus: set[int] = set()
    for status in glob.glob(f"/proc/{pid}/task/*/status"):
        try:
            for line in open(status):
                if line.startswith("Cpus_allowed_list:"):
                    cpus |= _parse_cpulist(line.split(":", 1)[1])
                    break
        except Exception:
            pass
    return cpus


def _cmdline(pid: str) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _pages_by_node_from_line(line: str) -> dict[int, int]:
    return {int(m.group("node")): int(m.group("pages")) for m in NODE_RE.finditer(line)}


def _add_pages(dst: dict[int, int], src: dict[int, int]) -> None:
    for node, pages in src.items():
        dst[node] = dst.get(node, 0) + pages


def _node_page_total(pages: dict[int, int]) -> int:
    return sum(pages.values())


def _expected_nodes(cpus: set[int]) -> set[int]:
    nodes: set[int] = set()
    for node, cpu_list in NODE_CPUSETS.items():
        if cpus & _parse_cpulist(cpu_list):
            nodes.add(node)
    return nodes


def _fmt_nodes(nodes: set[int] | list[int]) -> str:
    if not nodes:
        return "(none)"
    return ",".join(f"N{node}" for node in sorted(nodes))


def _summarize_numa_maps(
    lines: list[str],
    *,
    no_mmap: bool,
    expected_nodes: set[int],
    threshold: float,
) -> dict:
    mmap_pages: dict[int, int] = {}
    anon_pages: dict[int, int] = {}
    largest_anon_pages: dict[int, int] = {}
    largest_anon_total = 0
    model_files: set[str] = set()

    for line in lines:
        by_node = _pages_by_node_from_line(line)
        if not by_node:
            continue
        if "file=" in line and ".gguf" in line:
            _add_pages(mmap_pages, by_node)
            for token in line.split():
                if token.startswith("file=") and ".gguf" in token:
                    model_files.add(token.removeprefix("file="))
        if "anon=" in line:
            _add_pages(anon_pages, by_node)
            total = _node_page_total(by_node)
            if total > largest_anon_total:
                largest_anon_total = total
                largest_anon_pages = dict(by_node)

    signal = anon_pages if no_mmap else mmap_pages
    signal_kind = "anon_pages" if no_mmap else "mmap_gguf_pages"
    total = _node_page_total(signal)
    expected_pages = sum(signal.get(node, 0) for node in expected_nodes)
    local_fraction = (expected_pages / total) if total else None
    required = no_mmap and len(expected_nodes) == 1
    match = (local_fraction is not None and local_fraction >= threshold) if required else None
    note = "not required"
    if required:
        note = "ok" if match else "MEMORY LOCALITY MISMATCH"
    elif no_mmap and len(expected_nodes) != 1:
        note = "multi-node no_mmap placement observed"
    elif not no_mmap:
        note = "shared mmap placement observed"

    return {
        "checked": True,
        "required": required,
        "match": match,
        "threshold": threshold,
        "no_mmap": no_mmap,
        "signal_kind": signal_kind,
        "expected_nodes": _fmt_nodes(expected_nodes),
        "pages_by_node": {f"N{k}": v for k, v in sorted(signal.items())},
        "total_pages": total,
        "expected_node_pages": expected_pages,
        "local_fraction": local_fraction,
        "model_files": sorted(model_files),
        "largest_anon_pages_by_node": {f"N{k}": v for k, v in sorted(largest_anon_pages.items())},
        "largest_anon_total_pages": largest_anon_total,
        "note": note,
    }


def _memory_placement(pid: str | None, expected_nodes: set[int], threshold: float) -> dict:
    """Summarize model-weight memory placement from `/proc/<pid>/numa_maps`.

    For mmap-backed roles, `.gguf` file mappings are the model-weight signal.
    For `--no-mmap`, weights are anonymous after load; the best live proxy is
    the aggregate anonymous placement, with the largest anonymous segment also
    reported to make accidental KV/heap skew visible.
    """
    if not pid:
        return {
            "checked": False,
            "required": False,
            "match": None,
            "note": "no live process",
        }
    cmdline = _cmdline(pid)
    no_mmap = "--no-mmap" in cmdline
    try:
        lines = Path(f"/proc/{pid}/numa_maps").read_text(errors="replace").splitlines()
    except Exception as exc:
        return {
            "checked": False,
            "required": False,
            "match": None,
            "no_mmap": no_mmap,
            "note": f"could not read numa_maps: {exc}",
        }
    return _summarize_numa_maps(
        lines,
        no_mmap=no_mmap,
        expected_nodes=expected_nodes,
        threshold=threshold,
    )


def _fmt(cpus: set[int]) -> str:
    """Compact range string for readability."""
    if not cpus:
        return "(none)"
    s = sorted(cpus)
    parts, lo, prev = [], s[0], s[0]
    for c in s[1:]:
        if c == prev + 1:
            prev = c
        else:
            parts.append(f"{lo}-{prev}" if lo != prev else f"{lo}")
            lo = prev = c
    parts.append(f"{lo}-{prev}" if lo != prev else f"{lo}")
    return ",".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="*", default=None, help="subset of roles (default: all)")
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--require-memory-locality",
        action="store_true",
        help="fail if any single-node --no-mmap instance has off-node weight/proxy pages",
    )
    ap.add_argument(
        "--memory-locality-threshold",
        type=float,
        default=0.85,
        help="minimum fraction of --no-mmap weight/proxy pages on the expected NUMA node",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(ORCH))
    from scripts.server.stack_numa import NUMA_CONFIG

    roles = args.roles or list(NUMA_CONFIG.keys())
    entries: list[dict] = []
    all_match = True
    memory_required_entries = 0
    memory_mismatches = 0
    for role in roles:
        cfg = NUMA_CONFIG.get(role)
        if not cfg:
            continue
        for idx, inst in enumerate(cfg["instances"]):
            cpuset, port, _threads = inst[0], inst[1], inst[2]
            expected = _parse_cpulist(cpuset)
            pid = _pid_on_port(port)
            observed = _thread_union(pid) if pid else set()
            match = bool(pid) and observed == expected
            expected_nodes = _expected_nodes(expected)
            memory = _memory_placement(pid, expected_nodes, args.memory_locality_threshold)
            if memory.get("required"):
                memory_required_entries += 1
                if memory.get("match") is False:
                    memory_mismatches += 1
            all_match &= match
            entries.append({
                "role": role, "instance_idx": idx, "port": port, "pid": pid,
                "expected_cpus": _fmt(expected), "observed_thread_union": _fmt(observed),
                "expected_numa_nodes": _fmt_nodes(expected_nodes),
                "n_expected": len(expected), "n_observed": len(observed),
                "match": match,
                "memory_placement": memory,
                "note": ("ok" if match else ("no live process" if not pid else "AFFINITY MISMATCH")),
            })

    memory_verified = memory_mismatches == 0

    artifact = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "live_affinity_verified": all_match,
        "live_memory_placement_verified": memory_verified,
        "memory_locality_required": args.require_memory_locality,
        "memory_locality_threshold": args.memory_locality_threshold,
        "memory_required_entries": memory_required_entries,
        "memory_mismatches": memory_mismatches,
        "roles_checked": roles,
        "instances": entries,
    }
    out = Path(args.output) if args.output else (
        ORCH / "data" / "contention_matrix" / f"affinity_preflight_{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))

    for e in entries:
        flag = "OK " if e["match"] else "XX "
        mem = e["memory_placement"]
        mem_flag = ""
        if mem.get("required"):
            mem_flag = f" mem={'OK' if mem.get('match') else 'XX'}"
        elif mem.get("checked"):
            mem_flag = " mem=obs"
        print(f"  {flag}{e['role']:22} idx{e['instance_idx']} :{e['port']} "
              f"expected={e['expected_cpus']:18} observed={e['observed_thread_union']:18} {e['note']}")
        print(
            f"     memory nodes={mem.get('expected_nodes', '(none)'):11} "
            f"signal={mem.get('signal_kind', 'n/a'):16} "
            f"local={mem.get('local_fraction')} {mem.get('note')}{mem_flag}"
        )
    print(f"\nlive_affinity_verified = {all_match}")
    print(
        f"live_memory_placement_verified = {memory_verified} "
        f"(required={memory_required_entries}, mismatches={memory_mismatches})  → {out}"
    )
    if not all_match:
        return 1
    if args.require_memory_locality and not memory_verified:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
