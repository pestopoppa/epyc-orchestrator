#!/usr/bin/env python3
"""Serving proof for the embedder pool's DECLARED placement (UFH-12 Phase 0).

Reads the truth from the kernel, never from the declaration. For every port in
stack_manifest.EMBEDDING_PLACEMENT whose server is running, it checks:

  1. exe      /proc/<pid>/exe resolves into the CPU kernel store
               (/mnt/raid0/llm/kernels/production/cpu).
  2. argv     -c / -np / -t match the recipe, and --no-mmap is present when the
               recipe asks for it.
  3. threads  EVERY task's Cpus_allowed_list is inside the declared cpuset (the
               taskset prefix is inherited by all threads), and the tasks pinned
               to a sub-mask -- the OMP compute team -- hold `threads`
               DISJOINT places. The 2026-09-26 defect was invisible to a
               main-thread-only check: /proc/<pid>/status read "0,96" for all six.
  4. pool     no cpu is allowed to two different ports' tasks.
  5. memory   >= --min-local of the process's resident pages are on the
               declared cpuset's NPS4 node (from /proc/<pid>/numa_maps), so the
               membind is real and not silently defeated by shared mmap.

Read-only and PID-safe: a process is matched on the exact `--port <n>` token
pair of its argv, never on a name pattern, and nothing is signalled. Exit 0 when
every running declared port passes; 1 on any violation; 2 when a declared POOL
port has no process (unless --allow-missing).

    embedder_affinity_check.py
    embedder_affinity_check.py --json
    embedder_affinity_check.py --allow-missing     # e.g. 8096 is warm/down
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.server.stack_manifest import (  # noqa: E402
    EMBEDDER_PORTS,
    EMBEDDING_PLACEMENT,
    EMBEDDING_SERVER_RECIPES,
)
from scripts.server.stack_numa import _parse_cpus  # noqa: E402

KERNEL_STORE_CPU = Path("/mnt/raid0/llm/kernels/production/cpu")
PROC = Path("/proc")


def _argv(pid: str) -> list[str]:
    try:
        raw = (PROC / pid / "cmdline").read_bytes()
    except OSError:
        return []
    return [tok.decode(errors="replace") for tok in raw.split(b"\0") if tok]


def _pid_for_port(port: int) -> str | None:
    """The one llama-server whose argv carries the exact tokens `--port <port>`."""
    hits = []
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        argv = _argv(entry.name)
        if not argv or not argv[0].endswith("llama-server"):
            continue
        for i, tok in enumerate(argv[:-1]):
            if tok == "--port" and argv[i + 1] == str(port):
                hits.append(entry.name)
                break
    return hits[0] if len(hits) == 1 else (None if not hits else "AMBIGUOUS:" + ",".join(hits))


def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def _task_masks(pid: str) -> list[set[int]]:
    masks = []
    for task in (PROC / pid / "task").iterdir():
        try:
            for line in (task / "status").read_text().splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    masks.append(_parse_cpus(line.split(":", 1)[1].strip()))
        except OSError:
            continue
    return masks


def _pages_by_node(pid: str) -> dict[int, int]:
    out: dict[int, int] = {}
    try:
        text = (PROC / pid / "numa_maps").read_text()
    except OSError:
        return out
    for line in text.splitlines():
        for tok in line.split():
            if tok.startswith("N") and "=" in tok and tok[1:].split("=", 1)[0].isdigit():
                node, pages = tok[1:].split("=", 1)
                out[int(node)] = out.get(int(node), 0) + int(pages)
    return out


def _fmt(cpus: set[int]) -> str:
    return ",".join(str(c) for c in sorted(cpus))


def check(min_local: float, allow_missing: bool) -> tuple[int, list[dict]]:
    rows: list[dict] = []
    rc = 0
    owner: dict[int, int] = {}
    store = os.path.realpath(KERNEL_STORE_CPU)
    for port in sorted(EMBEDDING_PLACEMENT):
        placement = EMBEDDING_PLACEMENT[port]
        recipe = EMBEDDING_SERVER_RECIPES[port]
        declared = _parse_cpus(placement.cpuset)
        row: dict = {"port": port, "cpuset": placement.cpuset, "node": placement.numa_node, "problems": []}
        pid = _pid_for_port(port)
        if pid is None:
            row["status"] = "absent"
            if port in EMBEDDER_PORTS and not allow_missing:
                row["problems"].append("declared pool port has no running llama-server")
                rc = max(rc, 2)
            rows.append(row)
            continue
        if pid.startswith("AMBIGUOUS"):
            row["problems"].append(f"more than one process claims --port {port}: {pid}")
            rows.append(row)
            rc = max(rc, 1)
            continue
        row["pid"] = int(pid)
        argv = _argv(pid)
        exe = os.path.realpath(PROC / pid / "exe")
        if not exe.startswith(store + os.sep):
            row["problems"].append(f"exe {exe} is not under the kernel store {store}")
        for flag, want in (("-c", recipe["context_tokens"]), ("-np", recipe["slots"]), ("-t", recipe["threads"])):
            got = _flag(argv, flag)
            if got != str(want):
                row["problems"].append(f"argv {flag} {got} != recipe {want}")
        if recipe.get("no_mmap") and "--no-mmap" not in argv:
            row["problems"].append("recipe asks for --no-mmap; argv lacks it")
        masks = _task_masks(pid)
        outside = [sorted(m - declared) for m in masks if m - declared]
        if outside:
            row["problems"].append(f"{len(outside)} task(s) allowed outside {placement.cpuset}, e.g. {outside[0][:8]}")
        # The OMP compute team is every task whose mask differs from the modal
        # one (the modal mask is what the process was spawned with and what its
        # idle/HTTP threads inherit). Its places must be `threads` DISJOINT sets.
        modal = max(masks, key=lambda m: sum(1 for x in masks if x == m)) if masks else set()
        team = [m for m in masks if m != modal]
        row["tasks"] = len(masks)
        row["omp_places"] = sorted(_fmt(m) for m in team)
        disjoint = all(not (a & b) for i, a in enumerate(team) for b in team[i + 1:])
        if len(team) != int(recipe["threads"]) or not disjoint:
            row["problems"].append(
                f"OMP team places {row['omp_places']} — want {recipe['threads']} disjoint places"
            )
        clashes: dict[int, set[int]] = {}
        for cpu in set().union(*masks) if masks else set():
            other = owner.setdefault(cpu, port)
            if other != port:
                clashes.setdefault(other, set()).add(cpu)
        for other, cpus in sorted(clashes.items()):
            row["problems"].append(
                f"{len(cpus)} cpu(s) also allowed to :{other}'s tasks, e.g. {sorted(cpus)[:8]}"
            )
        pages = _pages_by_node(pid)
        total = sum(pages.values()) or 1
        local = pages.get(placement.numa_node, 0) / total
        row["local_page_fraction"] = round(local, 4)
        if local < min_local:
            row["problems"].append(
                f"only {local:.1%} of resident pages on node {placement.numa_node} (want >= {min_local:.0%})"
            )
        row["status"] = "ok" if not row["problems"] else "FAIL"
        if row["problems"]:
            rc = max(rc, 1)
        rows.append(row)
    return rc, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--min-local", type=float, default=0.95)
    args = ap.parse_args(argv)
    rc, rows = check(args.min_local, args.allow_missing)
    if args.json:
        print(json.dumps({"rc": rc, "ports": rows}, indent=2))
    else:
        for r in rows:
            head = f":{r['port']:<5} {r['cpuset']:<9} node{r['node']} {r.get('status', '?'):<6}"
            extra = f" pid={r.get('pid', '-')} omp={r.get('omp_places', '-')} local={r.get('local_page_fraction', '-')}"
            print(head + extra)
            for p in r["problems"]:
                print(f"    - {p}")
        print(f"rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
