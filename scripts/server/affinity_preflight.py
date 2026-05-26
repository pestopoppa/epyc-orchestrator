#!/usr/bin/env python3
"""Live-affinity preflight + artifact (bulk-inference campaign hard gate).

`topology_hash` certifies the *intended* NUMA_CONFIG; it does NOT certify that the live
llama-server processes are actually pinned to those CPUs. A launcher bug (2026-05-26) pinned
worker_general/vision quarters to the wrong (overlapping) cores while -t was correct, invalidating
any concurrency measurement. This tool closes that gap: for every NUMA_CONFIG (role, instance),
it finds the live port's process, computes the UNION of its threads' CPU affinity (the real
footprint — the main thread alone is misleading because idle threads cluster on the first core),
and checks it EXACTLY matches the configured cpuset.

Emits `data/contention_matrix/affinity_preflight_<ts>.json` with per-port {expected, observed,
match} + a top-level `live_affinity_verified` bool. J4/J5/J6 + any concurrent run must require
`live_affinity_verified=true` against a fresh artifact for the current launch.

Exit 0 iff all live instances match. Usage: python3 scripts/server/affinity_preflight.py [--roles a b]
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

ORCH = Path(__file__).resolve().parents[2]


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
    args = ap.parse_args()

    sys.path.insert(0, str(ORCH))
    from scripts.server.stack_numa import NUMA_CONFIG

    roles = args.roles or list(NUMA_CONFIG.keys())
    entries: list[dict] = []
    all_match = True
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
            all_match &= match
            entries.append({
                "role": role, "instance_idx": idx, "port": port, "pid": pid,
                "expected_cpus": _fmt(expected), "observed_thread_union": _fmt(observed),
                "n_expected": len(expected), "n_observed": len(observed),
                "match": match,
                "note": ("ok" if match else ("no live process" if not pid else "AFFINITY MISMATCH")),
            })

    artifact = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "live_affinity_verified": all_match,
        "roles_checked": roles,
        "instances": entries,
    }
    out = Path(args.output) if args.output else (
        ORCH / "data" / "contention_matrix" / f"affinity_preflight_{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))

    for e in entries:
        flag = "OK " if e["match"] else "XX "
        print(f"  {flag}{e['role']:22} idx{e['instance_idx']} :{e['port']} "
              f"expected={e['expected_cpus']:18} observed={e['observed_thread_union']:18} {e['note']}")
    print(f"\nlive_affinity_verified = {all_match}  → {out}")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
