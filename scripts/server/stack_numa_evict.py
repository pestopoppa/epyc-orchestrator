"""Pre-launch NUMA eviction and post-launch placement observation (INF-70/C7).

PREPARED, NOT ENABLED. Every entry point here is a no-op unless a role declares
`numa_pre_evict_gib: <N>` in `orchestration/stack_topology.yaml`; the default is
0 (off) and no role declares it today.

WHY
---
`numactl --interleave=all` is a per-allocation POLICY HINT, not a guarantee.
When the round robin reaches a node with no free pages the kernel silently
falls back to a node that has some, rather than reclaiming on the intended one.
Page cache counts as "not free", so on a box that has been serving for hours
the striping quietly collapses — same argv, same prefix, no warning, no error,
nothing in the log.

Measured 2026-09-02 (INF-70): a 98 GB model loaded under the canonical prefix
landed 57.7 / 10.7 / 8.0 / 17.7 GB across nodes 0-3 (61% on node 0 against an
even 25%) and decode measured 7.65 t/s against 10.09 t/s with clean placement —
**-25%**, entirely remote-node traffic on a bandwidth-bound decode. The same
audit found a live `sd-server` skewed 8/40/32/19%.

`stack_prewarm` is a different lever and does not cover this: it populates the
PAGE CACHE interleaved so a later mlock finds pages already spread. This module
does the opposite — it FREES pages on every node so a fresh anonymous
allocation can be spread at all.

THE MECHANISM
-------------
Per node: allocate and TOUCH N GiB under `numactl --membind=<node>`. The hard
binding leaves the kernel no fallback, so it reclaims that node's page cache to
satisfy the fault; freeing the allocation then leaves N GiB genuinely free on
that node. Do it on every node and the subsequent `--interleave=all` load
stripes evenly because every node can honour its share.

The canonical implementation is `scripts/utils/numa_evict.py` in
epyc-inference-research (wired into `bench_canonical.sh --pre-evict-gib`); this
module reuses it when present and falls back to an inline touch loop otherwise,
so the stack and the bench recipe cannot drift into two different fixes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

# The research repo's productized helper. Reused verbatim when present so the
# stack and the canonical bench recipe run the SAME eviction.
RESEARCH_EVICT = "/mnt/raid0/llm/epyc-inference-research/scripts/utils/numa_evict.py"

# Refuse absurd declarations: one node on this host is ~256 GiB.
MAX_PRE_EVICT_GIB = 200

# Per-node eviction is memory-bandwidth work, not compute; 40 GiB is seconds
# with the numpy path and ~40 s with the fallback. The cap stops a
# misconfiguration from stalling a stack start indefinitely.
EVICT_TIMEOUT_S = 900


def pre_evict_gib_for_role(numa_cfg: dict | None) -> int:
    """Resolve `numa_pre_evict_gib` from a role's stack_topology entry.

    Absent, malformed or non-positive ⇒ 0 ⇒ the step does not run. A value is
    never inferred: this is opt-in per role, exactly like `mlock`.
    """
    if not isinstance(numa_cfg, dict):
        return 0
    raw = numa_cfg.get("numa_pre_evict_gib", 0)
    try:
        gib = int(raw)
    except (TypeError, ValueError):
        return 0
    if gib <= 0:
        return 0
    return min(gib, MAX_PRE_EVICT_GIB)


def _node_ids() -> list[int]:
    """NUMA node ids from /sys, or [] when the host is not NUMA."""
    base = "/sys/devices/system/node"
    try:
        entries = os.listdir(base)
    except OSError:
        return []
    ids = []
    for name in entries:
        if name.startswith("node") and name[4:].isdigit():
            ids.append(int(name[4:]))
    return sorted(ids)


def pre_evict_nodes(target_gib: int, *, timeout_s: int = EVICT_TIMEOUT_S) -> tuple[bool, str]:
    """Force >= `target_gib` free on every NUMA node. Returns (ok, message).

    Never raises and never blocks a launch: a failure here degrades placement,
    it does not prevent serving. The caller logs the message either way.
    """
    if target_gib <= 0:
        return True, "disabled (numa_pre_evict_gib=0)"
    numactl = shutil.which("numactl")
    if numactl is None:
        return False, "numactl binary not found on PATH"
    nodes = _node_ids()
    if len(nodes) < 2:
        return True, "single-node host; nothing to evict"

    t0 = time.monotonic()
    if os.path.isfile(RESEARCH_EVICT):
        cmd = ["python3", RESEARCH_EVICT, "--target-gib", str(target_gib)]
        source = "epyc-inference-research/scripts/utils/numa_evict.py"
    else:
        cmd = [
            numactl,
            "--interleave=all",
            "python3",
            "-c",
            _INLINE_EVICT,
            str(target_gib),
        ]
        source = "inline fallback"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_s}s ({source})"
    except Exception as exc:  # never block a launch: this improves placement, it does not serve
        return False, f"{type(exc).__name__}: {exc} ({source})"
    elapsed = time.monotonic() - t0
    tail = (proc.stdout or "").strip().splitlines()
    detail = tail[-1] if tail else ""
    if proc.returncode != 0:
        return False, f"exit {proc.returncode} after {elapsed:.1f}s via {source}: {detail}"
    return True, f"{target_gib} GiB/node in {elapsed:.1f}s via {source}: {detail}"


# Fallback used only when the research helper is absent. Deliberately tiny and
# dependency-free; the research helper is the maintained implementation.
_INLINE_EVICT = """
import ctypes, mmap, subprocess, sys
target = int(sys.argv[1])
out = subprocess.run(['numactl','-H'], capture_output=True, text=True).stdout
free = {}
for line in out.splitlines():
    f = line.split()
    if len(f) >= 4 and f[0] == 'node' and f[2] == 'free:':
        free[int(f[1])] = int(f[3])
for node in sorted(free):
    need = target - free[node] // 1024 + 1
    if need <= 0:
        continue
    subprocess.run(['numactl', '--membind=%d' % node, 'python3', '-c',
                    'import mmap,sys;n=int(sys.argv[1])<<30;'
                    'm=mmap.mmap(-1,n,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS);'
                    '[m.__setitem__(o,1) for o in range(0,n,4096)];m.close()',
                    str(need)])
print('inline eviction done for nodes', sorted(free))
"""


def placement_summary(pid: int) -> str:
    """One-line per-node fold of a live process's resident pages.

    Reads `/proc/<pid>/numa_maps` rather than shelling out to `numastat -p`.
    Same quantity — numastat aggregates exactly these per-node page counts —
    but with no subprocess on the launch path, no dependency on the separate
    `numastat` binary, and no perturbation of the callers' process bookkeeping.
    `scripts/server/affinity_preflight.py` already reads numa_maps for the same
    reason. Read-only observation: it gates nothing and never raises.
    """
    try:
        text = open(f"/proc/{pid}/numa_maps", errors="replace").read()
    except OSError as exc:
        return f"numa_maps unreadable: {exc}"
    return summarize_numa_maps(text)


def summarize_numa_maps(text: str) -> str:
    """Fold numa_maps text into 'n0=… n1=… total=… max=nodeX@YY% (even=ZZ%)'.

    Each line's `N<node>=<pages>` counts pages of THAT mapping's page size, so
    `kernelpagesize_kB` is applied per line — a THP-backed model mapping counts
    2 MiB per page, and ignoring it understates the very mappings that matter.
    """
    per_node: dict[int, float] = {}
    for line in text.splitlines():
        fields = line.split()
        page_kb = 4.0
        for tok in fields:
            if tok.startswith("kernelpagesize_kB="):
                try:
                    page_kb = float(tok.split("=", 1)[1])
                except ValueError:
                    page_kb = 4.0
        for tok in fields:
            if tok.startswith("N") and "=" in tok and tok[1:2].isdigit():
                key, _, val = tok.partition("=")
                try:
                    node = int(key[1:])
                    pages = float(val)
                except ValueError:
                    continue
                per_node[node] = per_node.get(node, 0.0) + pages * page_kb / 1024.0
    if not per_node:
        return "no NUMA-resident mappings"
    total = sum(per_node.values())
    if total <= 0:
        return "no resident pages yet"
    nodes = sorted(per_node)
    parts = " ".join(f"n{n}={per_node[n]:.0f}MB" for n in nodes)
    max_node = max(nodes, key=lambda n: per_node[n])
    max_share = 100.0 * per_node[max_node] / total
    even = 100.0 / len(nodes)
    return (
        f"{parts} total={total:.0f}MB "
        f"max=node{max_node}@{max_share:.1f}% (even={even:.0f}%)"
    )


__all__ = [
    "MAX_PRE_EVICT_GIB",
    "EVICT_TIMEOUT_S",
    "RESEARCH_EVICT",
    "pre_evict_gib_for_role",
    "pre_evict_nodes",
    "placement_summary",
    "summarize_numa_maps",
]
