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

Cell-manifest mode (E5 batched-decode bench, 2026-07-23; ADDITIVE — the default
role-keyed invocation above is unchanged): `--cell-manifest <e5-cell-manifest JSON>`
or repeatable `--cell '{"cpuset":"48-95,144-191","port":19011[,"pid":123]}'` verifies
LIVE thread-union affinity for arbitrary {cpuset, port, pid?} cells that have no
NUMA_CONFIG role (synthesized half1, bench 19xxx ports). Fail-closed: no live process
on a port = FAIL; supplied pid disagreeing with the pid discovered on the port = FAIL;
any OTHER llama-family process (llama-server/llama-bench/llama-cli/ik_llama) whose
thread union overlaps the declared cpusets = FAIL (no-pre-existing-llama precondition). Cell mode refuses ports outside 19000-19999
unless --allow-any-port. Exit codes: 0 all cells matched, 1 any failure,
2 usage/parse error. The artifact JSON is also printed to stdout for the harness.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ORCH = Path(__file__).resolve().parents[2]

# The single implementation of "do these two cpusets contend?" — imported, never
# copied, per its own docstring in gpu_shadow_lane_lease.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from gpu_shadow_lane_lease import fold_cpus_to_physical as _fold_cpus_to_physical
except Exception:  # pragma: no cover - keeps the preflight usable if the lane tool moves
    def _fold_cpus_to_physical(cpus: set[int]) -> set[int]:
        raise RuntimeError(
            "gpu_shadow_lane_lease.fold_cpus_to_physical is unavailable; refusing to "
            "answer an SMT-contention question with a second, possibly divergent fold"
        )
CELL_MANIFEST_SCHEMA_VERSION = "e5-cell-manifest/1"
BENCH_PORT_RANGE = (19000, 19999)
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


# Cell-mode foreign-process scan: same llama family the research harness's
# find_llama_processes matches — a llama-bench/llama-cli squatting on a
# declared cpuset is contention exactly like a foreign llama-server (review
# F8). \b anchors avoid substring false positives; also a valid Python regex
# so tests can exercise the pattern offline.
LLAMA_PROC_PATTERN = r"\b(llama-server|llama-bench|llama-cli|ik_llama)\b"


def _llama_processes() -> list[tuple[str, str]]:
    """Live llama-family processes as (pid, args) — realized state via ps, never config.

    The pattern is matched against the EXECUTABLE (argv[0] basename), never
    the full args line: other processes legitimately carry llama names inside
    their ARGUMENTS (live incident 2026-07-23: earlyoom's --ignore/--prefer
    regexes contain "llama-server"/"llama-bench" and earlyoom spans all CPUs —
    a full-cmdline grep made every E5 cell fail its foreign-overlap gate).
    """
    r = subprocess.run(
        ["ps", "-eo", "pid,args"],
        capture_output=True, text=True)
    pattern = re.compile(LLAMA_PROC_PATTERN)
    out: list[tuple[str, str]] = []
    for line in r.stdout.splitlines():
        pid, _, procargs = line.strip().partition(" ")
        if not pid.isdigit():
            continue
        argv0 = procargs.strip().split(" ", 1)[0] if procargs.strip() else ""
        if pattern.search(argv0.rsplit("/", 1)[-1]):
            out.append((pid, procargs.strip()))
    return out



# ── GPU / training tenants, and SMT-aware contention (E5 protection defect) ────
# Two blind spots, filed 2026-07-29, both of which let a real contender look absent.
#
# (1) DISCOVERY WAS LLAMA-ONLY. `_llama_processes` matches argv[0] against
#     LLAMA_PROC_PATTERN, so a `python` ROCm/PyTorch/TRL trainer holding the MI210
#     is simply not a process this tool can see. Pattern-matching argv is also the
#     wrong instrument for the question — the definitive test for "is this process
#     on the GPU" is whether it holds an AMD GPU device open, which is checked here
#     against /dev/kfd and /dev/dri/render*.
#
# (2) OVERLAP WAS SMT-BLIND. The gate intersects raw LOGICAL cpu ids, so the GPU
#     host lane on 184-191 and an E5 cell on 0-95 intersect to the EMPTY SET even
#     though 184-191 are the siblings of physical 88-95, which the cell owns. The
#     fold is imported from gpu_shadow_lane_lease rather than rewritten: its own
#     docstring requires that "do these two cpusets contend?" have exactly one
#     implementation and not answer differently in two places.
#
# This records the facts. It deliberately does NOT change what gates a cell — see
# the note in the artifact and the decision package, because "which overlap is
# RELEVANT" decides decision_grade and the GPU host lane is a permanent, declared
# co-tenant of every 0-95 shape. A gate that fails every cell is the throttle-gate
# incident again.
_GPU_DEVICE_HINTS = ("/dev/kfd", "/dev/dri/render")


def _holds_gpu_device(pid: str) -> bool:
    """True when the process has an AMD GPU device node open.

    Realized state, not a name guess: a trainer, a HIP bench and a rocm-smi all
    show up, and a process merely CALLED something gpu-ish does not.
    """
    for link in glob.glob(f"/proc/{pid}/fd/*"):
        try:
            target = os.readlink(link)
        except OSError:
            continue
        if any(target.startswith(hint) for hint in _GPU_DEVICE_HINTS):
            return True
    return False


def _gpu_processes() -> list[tuple[str, str]]:
    """Live processes holding an AMD GPU device, as (pid, args)."""
    r = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True)
    out: list[tuple[str, str]] = []
    for line in r.stdout.splitlines():
        pid, _, procargs = line.strip().partition(" ")
        if not pid.isdigit():
            continue
        if _holds_gpu_device(pid):
            out.append((pid, procargs.strip()))
    return out


def _physical_overlap(cpus_a: set[int], cpus_b: set[int]) -> set[int]:
    """Physical cores both sets occupy, siblings folded. Empty means genuinely disjoint."""
    return _fold_cpus_to_physical(cpus_a) & _fold_cpus_to_physical(cpus_b)


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


class _UsageError(Exception):
    """CLI usage / parse error → exit 2 (bad JSON, unknown schema_version, port refusal)."""


def _load_cells(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Parse cell-mode inputs into [{cpuset, port, pid}] + artifact metadata.

    Reads ONLY instances[].{cpu_list, port} from a manifest (JSON is the cross-repo
    contract; no research-repo Python is imported). Raises _UsageError on any
    malformed input, unknown schema_version, or out-of-range port.
    """
    if args.cell_manifest and args.cell:
        raise _UsageError("--cell-manifest and --cell are mutually exclusive")
    if args.roles is not None:
        raise _UsageError("--roles cannot be combined with cell mode (--cell-manifest/--cell)")

    pid_map: dict[str, str] = {}
    if args.pid_map:
        try:
            raw = json.loads(args.pid_map)
        except json.JSONDecodeError as exc:
            raise _UsageError(f"--pid-map is not valid JSON: {exc}")
        if not isinstance(raw, dict):
            raise _UsageError('--pid-map must be a JSON object {"<port>": pid}')
        pid_map = {str(k): str(v) for k, v in raw.items()}

    cells: list[dict] = []
    if args.cell_manifest:
        path = Path(args.cell_manifest)
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise _UsageError(f"could not read cell manifest {path}: {exc}")
        if not isinstance(manifest, dict):
            raise _UsageError(f"cell manifest {path} is not a JSON object")
        version = manifest.get("schema_version")
        if version != CELL_MANIFEST_SCHEMA_VERSION:
            raise _UsageError(
                f"unknown schema_version {version!r} "
                f"(expected {CELL_MANIFEST_SCHEMA_VERSION!r})")
        instances = manifest.get("instances")
        if not isinstance(instances, list) or not instances:
            raise _UsageError(f"cell manifest {path} has no instances[]")
        meta = {
            "source": "cell-manifest",
            "manifest_path": str(path),
            "cell_id": manifest.get("cell_id"),
        }
        for inst in instances:
            if not isinstance(inst, dict) or "cpu_list" not in inst or "port" not in inst:
                raise _UsageError(f"manifest instance missing cpu_list/port: {inst!r}")
            cells.append({"cpuset": str(inst["cpu_list"]), "port": int(inst["port"]), "pid": None})
    else:
        meta = {"source": "cell", "manifest_path": None, "cell_id": None}
        for cell_json in args.cell:
            try:
                obj = json.loads(cell_json)
            except json.JSONDecodeError as exc:
                raise _UsageError(f"--cell is not valid JSON: {exc}")
            if not isinstance(obj, dict):
                raise _UsageError(f"--cell must be a JSON object: {cell_json}")
            cpuset = obj.get("cpuset", obj.get("cpu_list"))
            if not cpuset or "port" not in obj:
                raise _UsageError(f"--cell requires cpuset and port: {cell_json}")
            pid = obj.get("pid")
            cells.append({
                "cpuset": str(cpuset), "port": int(obj["port"]),
                "pid": str(pid) if pid is not None else None,
            })

    for cell in cells:
        mapped = pid_map.get(str(cell["port"]))
        if mapped is not None:
            if cell["pid"] is not None and cell["pid"] != mapped:
                raise _UsageError(
                    f"conflicting pids for port {cell['port']}: "
                    f"cell.pid={cell['pid']} pid-map={mapped}")
            cell["pid"] = mapped
        try:
            parsed = _parse_cpulist(cell["cpuset"])
        except ValueError as exc:
            raise _UsageError(f"bad cpuset {cell['cpuset']!r}: {exc}")
        if not parsed:
            raise _UsageError(f"empty cpuset for port {cell['port']}")
        if not args.allow_any_port and not (
                BENCH_PORT_RANGE[0] <= cell["port"] <= BENCH_PORT_RANGE[1]):
            raise _UsageError(
                f"port {cell['port']} outside bench range "
                f"{BENCH_PORT_RANGE[0]}-{BENCH_PORT_RANGE[1]} "
                f"(prod ports are off-limits; use --allow-any-port to override)")
    return cells, meta


def _run_cell_mode(cells: list[dict], meta: dict, args: argparse.Namespace) -> int:
    """Verify LIVE thread-union affinity for arbitrary cells. Fail closed everywhere."""
    entries: list[dict] = []
    all_match = True
    memory_required_entries = 0
    memory_mismatches = 0
    declared_union: set[int] = set()
    cell_pids: set[str] = set()

    for idx, cell in enumerate(cells):
        expected = _parse_cpulist(cell["cpuset"])
        declared_union |= expected
        supplied = cell["pid"]
        discovered = _pid_on_port(cell["port"])
        pid = supplied or discovered
        if pid:
            cell_pids.add(pid)
        if discovered:
            cell_pids.add(discovered)
        observed = _thread_union(pid) if pid else set()

        if discovered is None:
            match = False
            note = "no live process on port"
        elif supplied is not None and supplied != discovered:
            match = False
            note = f"PID CROSS-CHECK MISMATCH (supplied {supplied}, on-port {discovered})"
        elif observed == expected:
            match = True
            note = "ok"
        else:
            match = False
            note = "AFFINITY MISMATCH"

        expected_nodes = _expected_nodes(expected)
        memory = _memory_placement(pid, expected_nodes, args.memory_locality_threshold)
        if memory.get("required"):
            memory_required_entries += 1
            if memory.get("match") is False:
                memory_mismatches += 1
        all_match &= match
        entries.append({
            "source": meta["source"],
            "cell_index": idx,
            "port": cell["port"],
            "pid": pid,
            "pid_on_port": discovered,
            "expected_cpus": _fmt(expected),
            "observed_thread_union": _fmt(observed),
            "expected_numa_nodes": _fmt_nodes(expected_nodes),
            "n_expected": len(expected), "n_observed": len(observed),
            "match": match,
            "memory_placement": memory,
            "note": note,
        })

    # No-pre-existing-llama precondition: any OTHER llama process whose live thread
    # union overlaps the declared cpusets invalidates the cell (contention hazard).
    # --foreign-allow-pattern (operator-sanctioned coexistence, 2026-07-23: the
    # GPU session cycles short-lived build-hip bench servers on the shared
    # host): matching processes are RECORDED as allowed overlaps — attested,
    # never silently dropped — but do not gate the cell.
    allow_re = re.compile(args.foreign_allow_pattern) if getattr(
        args, "foreign_allow_pattern", None
    ) else None
    foreign: list[dict] = []
    foreign_allowed: list[dict] = []
    for fpid, fargs in _llama_processes():
        if fpid in cell_pids:
            continue
        overlap = _thread_union(fpid) & declared_union
        if overlap:
            row = {"pid": fpid, "args": fargs, "overlap_cpus": _fmt(overlap)}
            if allow_re is not None and allow_re.search(fargs):
                foreign_allowed.append(row)
            else:
                foreign.append(row)
    if foreign:
        all_match = False

    # E5 protection defect, recorded half. Two facts the llama-only + logical-id
    # gate above cannot see, now attested on every cell:
    #   * GPU/training tenants, discovered by an open AMD device node rather than
    #     by an argv pattern, so a `python` ROCm trainer is no longer invisible.
    #   * SIBLING-FOLDED overlap, so a GPU host lane on 184-191 stops reading as
    #     disjoint from a cell on 0-95 when it physically shares cores 88-95.
    # Recorded, NOT gating: the GPU host lane is a permanent declared co-tenant of
    # every full-machine shape, so gating on physical overlap alone would fail
    # every 0-95 cell forever — the shape of the throttle-gate incident, where the
    # gate could never pass. Which overlap is RELEVANT decides decision_grade and
    # is therefore an operator ruling, filed separately.
    gpu_tenants: list[dict] = []
    for gpid, gargs in _gpu_processes():
        if gpid in cell_pids:
            continue
        gcpus = _thread_union(gpid)
        logical = gcpus & declared_union
        physical = _physical_overlap(gcpus, declared_union)
        if not (logical or physical):
            continue
        gpu_tenants.append({
            "pid": gpid,
            "args": gargs,
            "logical_overlap_cpus": _fmt(logical),
            "physical_overlap_cores": _fmt(physical),
            "smt_only": bool(physical and not logical),
            "allowed_by_pattern": bool(allow_re is not None and allow_re.search(gargs)),
        })
    smt_only_contention = [row for row in gpu_tenants if row["smt_only"]]

    # T5, partial: "verified" must not be satisfiable by checking NOTHING.
    #
    # `memory_verified` is `mismatches == 0`, and the locality predicate is
    # `no_mmap and len(expected_nodes) == 1` — which after the topology change is
    # false for EVERY instance, because every instance is now multi-node. So the
    # artifact asserted `live_memory_placement_verified: true` on a run where zero
    # entries were examined, and `--require-memory-locality` — an explicit operator
    # request for the guarantee — was satisfied vacuously.
    #
    # Absence read as a pass, the A11 shape again. Arming the predicate itself is
    # NOT done here: T5 is blocked on ratifying INTERLEAVE_TOLERANCE, and mmap
    # placement is shared across instances (a single-node mmap role can show a low
    # local_fraction because another instance first-touched the pages), so arming it
    # tonight could hard-fail correctly-configured roles. What needs no ratification
    # is making "not checked" distinguishable from "checked and passed".
    memory_checked = memory_required_entries > 0
    memory_verified = memory_mismatches == 0
    artifact = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "cell",
        "schema_version": CELL_MANIFEST_SCHEMA_VERSION,
        "manifest_path": meta["manifest_path"],
        "cell_id": meta["cell_id"],
        "live_affinity_verified": all_match,
        "live_memory_placement_verified": memory_verified,
        "memory_locality_required": args.require_memory_locality,
        "memory_locality_threshold": args.memory_locality_threshold,
        "live_memory_placement_checked": memory_checked,
        "memory_locality_vacuous": bool(args.require_memory_locality and not memory_checked),
        "memory_required_entries": memory_required_entries,
        "memory_mismatches": memory_mismatches,
        "foreign_llama_overlaps": foreign,
        # Recorded 2026-08-11 (E5 protection defect). `gpu_tenant_overlaps` closes the
        # discovery blind spot; `smt_only_contention` is the subset that the previous
        # logical-id intersection reported as ZERO overlap while the processes shared
        # physical cores. A non-empty smt_only list on a cell that passed is the exact
        # evidence the old gate could not produce.
        "gpu_tenant_overlaps": gpu_tenants,
        "smt_only_contention": smt_only_contention,
        "contention_gate_semantics": (
            "llama-family processes on a LOGICAL cpu intersection gate the cell; "
            "GPU/training tenants and SMT-sibling-only overlap are RECORDED and do not "
            "gate — see batched-decode-measurement.md, E5 protection defect"
        ),
        "foreign_allowed_overlaps": foreign_allowed,
        "foreign_allow_pattern": getattr(args, "foreign_allow_pattern", None),
        "instances": entries,
    }
    out = Path(args.output) if args.output else (
        ORCH / "data" / "contention_matrix" / f"affinity_preflight_{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))

    # Machine-parseable per-cell reasons for the invoking harness.
    print(json.dumps(artifact, indent=2))
    print(f"live_affinity_verified = {all_match}  → {out}", file=sys.stderr)
    if not all_match:
        return 1
    if args.require_memory_locality and not memory_checked:
        print(
            "REFUSING: --require-memory-locality was requested but ZERO entries were "
            "eligible for the locality check (predicate: no_mmap AND single-node; "
            "every instance is multi-node after the topology change). A guarantee "
            "satisfied by checking nothing is not a guarantee — see T5 in "
            "handoffs/active/numa-placement-defect-20260730.md, blocked on ratifying "
            "INTERLEAVE_TOLERANCE.",
            file=sys.stderr,
        )
        return 1
    if args.require_memory_locality and not memory_verified:
        return 1
    return 0


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
    ap.add_argument(
        "--foreign-allow-pattern",
        default=None,
        help="cell mode only: regex; foreign llama processes whose args match are "
             "RECORDED as foreign_allowed_overlaps (attested) but do not gate the "
             "cell — operator-sanctioned coexistence with e.g. build-hip GPU bench "
             "servers on the shared host",
    )
    ap.add_argument(
        "--cell-manifest",
        default=None,
        help="e5-cell-manifest JSON path; verify its instances[].{cpu_list,port} cells "
             "instead of NUMA_CONFIG roles (mutually exclusive with --roles/--cell)",
    )
    ap.add_argument(
        "--cell",
        action="append",
        default=None,
        help='repeatable explicit cell JSON, e.g. \'{"cpuset":"48-95,144-191","port":19011}\' '
             '(optional "pid"); mutually exclusive with --cell-manifest/--roles',
    )
    ap.add_argument(
        "--pid-map",
        default=None,
        help='optional JSON {"<port>": pid} from the launching harness; supplied pids are '
             "cross-checked against the pid discovered on the port (disagreement = fail)",
    )
    ap.add_argument(
        "--allow-any-port",
        action="store_true",
        help=f"cell mode refuses ports outside {BENCH_PORT_RANGE[0]}-{BENCH_PORT_RANGE[1]} "
             "by default (prevents gating against prod servers); this opts out",
    )
    args = ap.parse_args()

    if args.cell_manifest or args.cell:
        try:
            cells, meta = _load_cells(args)
        except _UsageError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return _run_cell_mode(cells, meta, args)
    if args.pid_map or args.allow_any_port:
        print("error: --pid-map/--allow-any-port require cell mode "
              "(--cell-manifest or --cell)", file=sys.stderr)
        return 2

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
