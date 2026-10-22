#!/usr/bin/env python3
"""Every thread of a CPU-pinned launch stays on the placement's CPUs.

The finding (workspace-58, 2026-10-22). A HIP llama-server started as
``taskset -c 184-191 numactl --membind=3 llama-server ... --device ROCm0`` had
196 threads; 195 of them carried ``Cpus_allowed_list: 184-191`` and ONE, an early
ROCm-runtime (ROCr) thread, carried ``0-191``. That thread can run on any core of
the host, including the cores CPU inference and timing windows own. ``taskset -c``
only sets the mask of the first task; every later task inherits the mask of the
task that creates it -- unless the creator hands ``pthread_create`` an explicit
affinity, which is what ROCr does.

The mechanism, read from the installed ROCr (``/opt/rocm/lib/
libhsa-runtime64.so.1.14.60200``, ROCm 6.2.0): ``os::CreateThread`` checks a
runtime flag and, when it is set, fills a cpu_set_t with CPUs
``0..get_nprocs_conf()-1`` and passes it to ``pthread_attr_setaffinity_np``
before ``pthread_create``. The flag is parsed from
``HSA_OVERRIDE_CPU_AFFINITY_DEBUG`` and is TRUE unless the value is exactly
``"0"``. So by default every thread ROCr creates -- at init, and any it creates
later -- is widened to the whole machine, whatever the process was pinned to.
(The HIP runtime's own affinity code, ``AMD_CPU_AFFINITY``, defaults off.)

So pinning is made deterministic in two layers, used by every launcher:

1. ``apply_pinned_launch_env(env, prefix)``: a launch whose spawn prefix pins
   (``taskset -c <cpus>``) gets ``HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0``. ROCr then
   creates its threads with the default attribute, i.e. they inherit the
   creating thread's (pinned) mask. This fixes the source, including threads ROCr
   creates after start-up. The variable is read only by ROCr; for a CPU binary
   it is inert.
2. ``pin_all_tasks(pid, cpus)`` after health: every task whose mask is NOT a
   subset of the placement is re-pinned to the placement (a task already on a
   narrower sub-mask -- an OpenMP place -- is left alone), repeated until a pass
   finds nothing new. Then ``affinity_escapes`` must return [] or the launch is a
   failure. This catches anything layer 1 does not cover (another library, a
   future ROCr that ignores the knob).

Neither layer is kernel-enforced: a thread that calls ``sched_setaffinity`` on
itself can still widen its own mask. That is why the third piece exists:
``affinity_attestation_warnings`` is called from ``orchestrator_stack.py status``
and the runtime attestation, so a regression is REPORTED on every status read
instead of being silently tolerated. (A cgroup-v2 cpuset would be the
kernel-enforced form; see the stack-change package for why it is not the
primary mechanism here.)

CLI (read-only unless --repin):

    launch_affinity.py --check PID CPUS        # exit 1 on any escaping task
    launch_affinity.py --check PID CPUS --json
    launch_affinity.py --repin PID CPUS        # pin_all_tasks, then re-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROCR_AFFINITY_ENV_KEY = "HSA_OVERRIDE_CPU_AFFINITY_DEBUG"
ROCR_AFFINITY_ENV_VALUE = "0"
DEFAULT_PROC_ROOT = Path("/proc")
DEFAULT_PIN_PASSES = 4


class LaunchAffinityError(ValueError):
    """A cpu list or spawn prefix that cannot be interpreted."""


def parse_cpu_list(spec: str) -> frozenset[int]:
    """Strict taskset-style list ('0-23,96-119') -> cpu ids. Raises on garbage."""
    if not isinstance(spec, str) or not spec.strip():
        raise LaunchAffinityError(f"empty cpu list: {spec!r}")
    cpus: set[int] = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            raise LaunchAffinityError(f"empty segment in cpu list {spec!r}")
        lo_s, sep, hi_s = part.partition("-")
        if not lo_s.isdigit() or (sep and not hi_s.isdigit()):
            raise LaunchAffinityError(f"malformed segment {part!r} in cpu list {spec!r}")
        lo, hi = int(lo_s), int(hi_s) if sep else int(lo_s)
        if hi < lo:
            raise LaunchAffinityError(f"descending range {part!r} in cpu list {spec!r}")
        cpus.update(range(lo, hi + 1))
    return frozenset(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """cpu ids -> compact taskset-style list ('184-191')."""
    ordered = sorted(set(cpus))
    if not ordered:
        return ""
    runs: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        runs.append(f"{start}-{prev}" if prev != start else str(start))
        start = prev = cpu
    runs.append(f"{start}-{prev}" if prev != start else str(start))
    return ",".join(runs)


def prefix_cpuset(prefix: Sequence[str]) -> str | None:
    """The cpu list a spawn prefix pins with ``taskset -c``, or None if it does not pin."""
    if "taskset" not in prefix:
        return None
    idx = list(prefix).index("taskset")
    if len(prefix) > idx + 2 and prefix[idx + 1] == "-c":
        return prefix[idx + 2]
    raise LaunchAffinityError(f"taskset prefix without a `-c <cpu_list>` argument: {list(prefix)!r}")


def apply_pinned_launch_env(env: MutableMapping[str, str], prefix: Sequence[str]) -> bool:
    """Set ``HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0`` on a pinned launch's env.

    Unconditional on a pinned launch (it overrides an inherited value): a declared
    placement is the contract, and an ambient ``1`` would silently re-open the
    escape. An unpinned launch is left untouched. Returns True when applied.
    """
    if prefix_cpuset(prefix) is None:
        return False
    env[ROCR_AFFINITY_ENV_KEY] = ROCR_AFFINITY_ENV_VALUE
    return True


@dataclass(frozen=True)
class TaskAffinity:
    tid: int
    comm: str
    cpus: frozenset[int]

    @property
    def cpu_list(self) -> str:
        return format_cpu_list(self.cpus)


def read_task_affinities(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> list[TaskAffinity]:
    """Every task of ``pid`` with its Cpus_allowed_list. [] if the pid is gone.

    A task that exits mid-walk is skipped (it no longer runs anywhere).
    """
    task_root = Path(proc_root) / str(pid) / "task"
    try:
        entries = sorted((e for e in task_root.iterdir() if e.name.isdigit()), key=lambda e: int(e.name))
    except OSError:
        return []
    out: list[TaskAffinity] = []
    for entry in entries:
        try:
            status = (entry / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        mask = None
        for line in status.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                mask = line.split(":", 1)[1].strip()
                break
        if mask is None:
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            comm = "?"
        out.append(TaskAffinity(tid=int(entry.name), comm=comm, cpus=parse_cpu_list(mask)))
    return out


def affinity_escapes(
    pid: int, cpus: frozenset[int] | str, proc_root: Path = DEFAULT_PROC_ROOT
) -> list[TaskAffinity]:
    """Tasks of ``pid`` allowed to run on any CPU outside ``cpus``."""
    allowed = parse_cpu_list(cpus) if isinstance(cpus, str) else frozenset(cpus)
    return [task for task in read_task_affinities(pid, proc_root) if not task.cpus <= allowed]


@dataclass
class PinResult:
    pid: int
    cpus: str
    tasks_seen: int = 0
    repinned: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    escapes_after: list[dict] = field(default_factory=list)
    passes: int = 0

    @property
    def ok(self) -> bool:
        return self.tasks_seen > 0 and not self.escapes_after

    def as_dict(self) -> dict:
        return {
            "pid": self.pid,
            "cpus": self.cpus,
            "tasks_seen": self.tasks_seen,
            "repinned": self.repinned,
            "errors": self.errors,
            "escapes_after": self.escapes_after,
            "passes": self.passes,
            "ok": self.ok,
        }


def _task_row(task: TaskAffinity) -> dict:
    return {"tid": task.tid, "comm": task.comm, "cpus": task.cpu_list}


def pin_all_tasks(
    pid: int,
    cpus: frozenset[int] | str,
    *,
    proc_root: Path = DEFAULT_PROC_ROOT,
    setter: Callable[[int, Iterable[int]], None] = os.sched_setaffinity,
    passes: int = DEFAULT_PIN_PASSES,
) -> PinResult:
    """Re-pin every task of ``pid`` whose mask is not inside ``cpus``; then re-check.

    ``os.sched_setaffinity`` on a TID sets that one thread's mask (Linux affinity
    is per task), so this is ``taskset -a -p -c`` done in-process -- but it leaves
    a task already on a NARROWER sub-mask alone, where ``taskset -a`` would widen
    an OpenMP-bound thread back to the whole placement. Repeats until a pass
    re-pins nothing (a thread created mid-walk is caught by the next pass), at
    most ``passes`` times. Only ever pass a pid this launcher spawned itself.
    """
    allowed = parse_cpu_list(cpus) if isinstance(cpus, str) else frozenset(cpus)
    result = PinResult(pid=pid, cpus=format_cpu_list(allowed))
    seen: set[int] = set()
    for _ in range(max(1, passes)):
        result.passes += 1
        tasks = read_task_affinities(pid, proc_root)
        seen.update(task.tid for task in tasks)
        moved = 0
        for task in tasks:
            if task.cpus <= allowed:
                continue
            try:
                setter(task.tid, allowed)
            except OSError as exc:
                # ESRCH: the thread exited between the read and the set -- not an escape.
                if getattr(exc, "errno", None) == 3:
                    continue
                result.errors.append({**_task_row(task), "error": str(exc)})
                continue
            result.repinned.append(_task_row(task))
            moved += 1
        if moved == 0:
            break
    result.tasks_seen = len(seen)
    result.escapes_after = [_task_row(t) for t in affinity_escapes(pid, allowed, proc_root)]
    return result


def parent_pid(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> int | None:
    """PPid of ``pid`` from /proc/<pid>/stat, or None when it cannot be read."""
    try:
        stat = (Path(proc_root) / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
        # Field 2 (comm) may contain spaces/parens; everything after the LAST ") " is fields 3..
        return int(stat.rsplit(") ", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def enforce_launch_affinity(
    pid: int,
    prefix: Sequence[str],
    *,
    label: str,
    proc_root: Path = DEFAULT_PROC_ROOT,
    setter: Callable[[int, Iterable[int]], None] = os.sched_setaffinity,
    emit: Callable[[str], None] = print,
    launcher_pid: int | None = None,
) -> PinResult | None:
    """Launcher hook: after health, pin every task of OUR pid to the prefix's cpus.

    Returns None for an unpinned prefix (nothing is asserted about a launch that
    declared no placement) and for a pid that is not a direct child of this
    launcher (it is never touched: prove OUR pid before acting on it -- a recycled
    pid or a test double must not get another process re-pinned). Otherwise the
    PinResult; the caller treats ``not result.ok`` as a failed launch.

    The spawn is ``taskset -c ... [numactl ...] llama-server``: both wrappers
    exec, so the Popen pid IS the server and its parent is this launcher.
    """
    cpuset = prefix_cpuset(prefix)
    if cpuset is None:
        return None
    me = os.getpid() if launcher_pid is None else launcher_pid
    ppid = parent_pid(pid, proc_root)
    if ppid != me:
        emit(
            f"    [WARN] [affinity] {label}: pid={pid} is not a child of this launcher "
            f"(ppid={ppid}, launcher={me}); not touching it"
        )
        return None
    result = pin_all_tasks(pid, cpuset, proc_root=proc_root, setter=setter)
    if result.ok:
        moved = f", re-pinned {len(result.repinned)}" if result.repinned else ""
        emit(f"    [affinity] {label}: pid={pid} {result.tasks_seen} tasks inside {cpuset}{moved}")
    else:
        emit(
            f"    [FAIL] [affinity] {label}: pid={pid} tasks escaping {cpuset} after re-pin: "
            f"{result.escapes_after or 'no tasks readable'}"
            + (f"; errors {result.errors}" if result.errors else "")
        )
    return result


def declared_cpuset_for_port(port: int) -> str | None:
    """The placement the stack declares for a port, or None when it declares none.

    Sources, in the order the launcher consults them: the embedder pool's
    per-port placement, then NUMA_CONFIG instances (incl. the GPU host lane),
    then aux services' declared ``cpuset``.
    """
    from scripts.server.stack_manifest import AUX_SERVICES, EMBEDDING_PLACEMENT
    from scripts.server.stack_numa import NUMA_CONFIG

    placement = EMBEDDING_PLACEMENT.get(port)
    if placement is not None:
        return placement.cpuset
    for cfg in NUMA_CONFIG.values():
        for instance in cfg.get("instances") or []:
            if len(instance) >= 2 and instance[1] == port:
                return str(instance[0])
    for service in AUX_SERVICES.values():
        if getattr(service, "port", None) == port and getattr(service, "cpuset", None):
            return str(service.cpuset)
    return None


def affinity_attestation_warnings(
    name: str,
    pid: int,
    port: int,
    *,
    proc_root: Path = DEFAULT_PROC_ROOT,
    declared: str | None = None,
) -> list[str]:
    """`status` / runtime-attestation check: every task of the live pid inside its placement.

    The "prove OUR pid" form: it reads the kernel's per-task masks for the pid the
    state file names, never the declaration or the main thread alone (the main
    thread read 184-191 while one task ran on 0-191).
    """
    if pid is None or pid <= 0:
        return []
    cpuset = declared if declared is not None else declared_cpuset_for_port(port)
    if not cpuset:
        return []
    tasks = read_task_affinities(pid, proc_root)
    if not tasks:
        return []
    allowed = parse_cpu_list(cpuset)
    escapes = [t for t in tasks if not t.cpus <= allowed]
    if not escapes:
        return []
    shown = ", ".join(f"{t.tid}({t.comm})={t.cpu_list}" for t in escapes[:4])
    more = f" (+{len(escapes) - 4} more)" if len(escapes) > 4 else ""
    return [
        f"{name} pid {pid} cpu_affinity_escape: {len(escapes)}/{len(tasks)} tasks "
        f"outside declared {cpuset}: {shown}{more} "
        f"(remedy: scripts/server/launch_affinity.py --repin {pid} {cpuset})"
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="read-only: list escaping tasks")
    mode.add_argument("--repin", action="store_true", help="pin_all_tasks, then re-check")
    parser.add_argument("pid", type=int)
    parser.add_argument("cpus", help="taskset-style cpu list, e.g. 184-191")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--proc-root", type=Path, default=DEFAULT_PROC_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repin:
        result = pin_all_tasks(args.pid, args.cpus, proc_root=args.proc_root)
        payload = result.as_dict()
        ok = result.ok
    else:
        tasks = read_task_affinities(args.pid, args.proc_root)
        escapes = [_task_row(t) for t in affinity_escapes(args.pid, args.cpus, args.proc_root)]
        payload = {"pid": args.pid, "cpus": args.cpus, "tasks_seen": len(tasks), "escapes": escapes}
        ok = bool(tasks) and not escapes
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(("OK " if ok else "FAIL ") + json.dumps(payload, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
