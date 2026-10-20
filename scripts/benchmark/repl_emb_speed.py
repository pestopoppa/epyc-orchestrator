#!/usr/bin/env python3
"""REPL-EMB-2.7 EPYC speed-block harness (code only; run later under an exclusive ``region-lock``).

Measures, per embedding arm, under a held CPU claim: encode throughput (chunks/s, tokens/s; batch
sweep 1/8/32/128 x threads 1/4, optional full-socket), single-query latency (p50/p95/max over >= 200
DISTINCT queries x 3 repeats, in-process and as-served legs), CPU-seconds per query, index-build time
for 4,096 chunks at the pool-like config (batch 32, threads 4) with the translation into the shipped
``context_search`` index budget, RSS and on-disk size. Timing claims only; no retrieval-quality claim.

Method (the reasons are the point of the file):

* ABA ordering. Within each leg the arms run forward, then reversed (``A B C C B A``), so linear drift
  cancels and every arm is measured twice at mirrored positions. ``aba_order`` is the single source.
* A/A block. One arm (default: the incumbent) is also measured at an EARLY and a LATE slot of the
  run. The A/A spread = max-min of that arm's ok measurements of the same leg across
  {early, the two ABA visits, late}, carried in the metric's own unit (chunks/s, ms, cpu-s) plus the
  relative figure, written to ``noise_floor.json``. A cross-arm difference not larger than the floor
  is labelled ``within_noise`` by ``summarize``; the harness never reports "faster" inside it.
* Safety. The runner refuses to start without ``--claim-tag`` AND a passing claim-environment check
  (affinity mask == ``--cpu-list``; no foreign llama*/python process running on, or pinned inside,
  those cpus; a ``region-lock`` holder among its own ancestors; caller-owned servers inside the
  claim). It refuses any ``--arm-url`` on a production or stack-registered port or a non-loopback
  host. It never starts, signals or kills a process: servers are caller-owned (``--arm-url`` +
  ``--arm-server-pid``). A leg whose cpus see foreign busy time above a stated fraction (from
  ``/proc/stat`` less this harness's own CPU time), or whose 1-minute loadavg rose by more than the
  stated excess, is aborted and recorded ``contended`` (kept in the log, excluded from the medians).
* Pinning. ``os.sched_setaffinity`` is applied to EVERY thread of this process (it is per-thread on
  Linux) and read back; a thread outside the leg's cpus marks the leg ``contended``
  (``affinity_violation``). ``OMP_PROC_BIND`` defaults to ``false`` here because a GOMP team re-pinning
  to the startup places would silently defeat the narrowing.
* Sampling DURING the window. CPU frequency and contention are sampled inside the timed window
  (every ``--check-interval-s``), never after.
* Resumable. One JSON row per measurement unit in ``<out>/speed.jsonl`` (fsync per row); a unit that
  finished ``ok``/``not_applicable`` is skipped on re-run; ``contended``/``error`` units are retried.
  ``run_manifest.json`` carries the plan digest; a changed plan refuses to resume into the same dir.

Commands (all paths literal; the runner is meant for the isolated venv, ``python -I``)::

    repl_emb_speed.py prepare   --corpus C.jsonl --queries Q.json --out DIR
    region-lock run --cpu-list 24-71 --role bench-cpu --tag TAG -- taskset -c 24-71 \\
        VENV/bin/python -I repl_emb_speed.py run --out DIR --cpu-list 24-71 --claim-tag TAG \\
        --require-claim-env --arm-url bge-large-gguf-f16=http://127.0.0.1:18751 \\
        --arm-server-pid bge-large-gguf-f16=PID --arm-server-threads bge-large-gguf-f16=16 ... \\
        --arm-model-path eg2-st-fp32=SNAPSHOT_DIR ...
    repl_emb_speed.py summarize --out DIR
    repl_emb_speed.py run --dry-run --out DIR        # whole pipeline on fake encoders, 32 chunks

``rows_for_capture(speed_jsonl, noise_floor)`` turns the log into the dicts that
``repl_retrieval_capture.SpeedRun.add_speed`` takes (that module is NOT imported here).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repl_emb_arms as arms  # noqa: E402

SCHEMA = "repl-emb-speed/1"
ROW_SCHEMA = "repl-emb-speed-row/1"
SUMMARY_SCHEMA = "repl-emb-speed-summary/1"
NOISE_SCHEMA = "repl-emb-speed-noise-floor/1"
INPUTS_SCHEMA = "repl-emb-speed-inputs/1"

# Ports a caller-supplied server may NEVER use: the production set from the REPL-EMB-2.7 brief,
# plus every other port the stack registers in orchestration/launch_manifest.yaml (a harness-owned
# server belongs on a free high port, like the 187xx servers of REPL-EMB-2.6).
PRODUCTION_PORTS = frozenset(
    {8000, 8083, 8090, 8091, 8092, 8093, 8094, 8095, 9000, 9002, 8088, 8089}
)
STACK_PORTS = frozenset(
    {8070, 8074, 8080, 8084, 8086, 8087, 8096, 8097, 8098, 8102, 8180, 8190, 9001, 18070}
)
REFUSED_PORTS = PRODUCTION_PORTS | STACK_PORTS
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

FOREIGN_COMM_PREFIXES = ("llama", "python", "ik_llama", "whisper", "qwentts")
DEFAULT_REGION_LOCK_BIN = "/workspace/repos/epyc-orchestrator/scripts/region-lock"

BGE_GGUF_SHA256 = "3379a0e9cea28fc6d7136df8ea7a88ef99ccce5963b9a6f7af9609997be762e3"
BGE_ST_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"

ROSTER_ORDER = (
    "bge-large-gguf-f16",
    "bge-large-st-fp32",
    "granite-97m-r2-gguf-f16",
    "eg2-gguf-bf16",
    "eg2-gguf-q8_0",
    "eg2-st-bf16",
    "eg2-st-fp32",
    "potion-code-16m-v2",
    "potion-base-8m",
)
ARM_KINDS = {
    "bge-large-gguf-f16": "gguf",
    "bge-large-st-fp32": "st",
    "granite-97m-r2-gguf-f16": "gguf",
    "eg2-gguf-bf16": "gguf",
    "eg2-gguf-q8_0": "gguf",
    "eg2-st-bf16": "st",
    "eg2-st-fp32": "st",
    "potion-code-16m-v2": "static",
    "potion-base-8m": "static",
}
FAMILIES = ("rss", "disk", "throughput", "query", "index")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".npy", ".onnx", ".pt", ".pth")
FULL_SOCKET_BATCHES = (32, 128)
UNRESOLVED_RUNTIME = "unresolved: llama-server build_info read from /props at run time"
INDEX_TARGET_CHUNKS = 4096  # the shipped MAX_DENSE_CHUNKS bundle the budget is judged at
MB = 1_000_000  # decimal megabytes, the capture module's unit "MB" (RSS kB is KiB, converted)


class Refusal(RuntimeError):
    """A safety refusal: the runner exits 64 (region-lock's usage/preflight code) and starts nothing."""


class LegContended(Exception):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


# ───────────────────────────────────────────────────────────────────────────── small pure helpers


def parse_cpu_list(text: str) -> list[int]:
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"bad cpu range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"empty cpu list {text!r}")
    return sorted(out)


def format_cpu_list(cpus: Iterable[int]) -> str:
    xs = sorted(set(int(c) for c in cpus))
    parts: list[str] = []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        parts.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(parts)


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), pure Python."""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("percentile of an empty sample")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, Any]:
    if not latencies_ms:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None, "mean_ms": None}
    return {
        "n": len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "max_ms": float(max(latencies_ms)),
        "mean_ms": float(sum(latencies_ms) / len(latencies_ms)),
    }


def aba_order(arm_ids: Sequence[str], passes: int = 2) -> list[tuple[int, int, str]]:
    """``(pass, position, arm)`` in measurement order: forward, then reversed, alternating.

    Two passes over A,B,C give ``A B C C B A``: each arm is measured twice at mirrored positions,
    so a linear drift adds the same amount to both and cancels in the arm-vs-arm comparison."""
    order: list[tuple[int, int, str]] = []
    pos = 0
    for p in range(int(passes)):
        seq = list(arm_ids) if p % 2 == 0 else list(arm_ids)[::-1]
        for a in seq:
            order.append((p, pos, a))
            pos += 1
    return order


def spread_of(values: Sequence[float]) -> dict[str, Any]:
    """Range of repeated measurements: absolute (in the metric's unit) and relative to the median."""
    xs = [float(v) for v in values]
    if len(xs) < 2:
        return {"n": len(xs), "abs": None, "rel": None, "median": xs[0] if xs else None}
    med = statistics.median(xs)
    rng = max(xs) - min(xs)
    return {"n": len(xs), "abs": rng, "rel": (rng / abs(med)) if med else None, "median": med}


def stride_subset(items: Sequence[Any], n: int) -> list[Any]:
    """Deterministic evenly-strided subset (a prefix would be all-Python in the pilot corpus)."""
    n = int(n)
    if n >= len(items):
        return list(items)
    return [items[int(i * len(items) / n)] for i in range(n)]


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def dumps(o: Any, **kw: Any) -> str:
    return json.dumps(_clean(o), default=_jsonable, sort_keys=True, **kw)


def sha256_json(o: Any) -> str:
    return hashlib.sha256(dumps(o).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_kv(items: Sequence[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise Refusal(f"{what}: expected ARM=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        if not k or not v:
            raise Refusal(f"{what}: expected ARM=VALUE, got {it!r}")
        out[k] = v
    return out


def parse_server_threads(cmdline: Sequence[str]) -> int | None:
    for i, a in enumerate(cmdline):
        if a in ("-t", "--threads") and i + 1 < len(cmdline):
            try:
                return int(cmdline[i + 1])
            except ValueError:
                return None
        if a.startswith("--threads="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return None
    return None


def parse_server_model(cmdline: Sequence[str]) -> str | None:
    for i, a in enumerate(cmdline):
        if a in ("-m", "--model") and i + 1 < len(cmdline):
            return cmdline[i + 1]
        if a.startswith("--model="):
            return a.split("=", 1)[1]
    return None


def disk_usage(path: Path) -> dict[str, Any]:
    """Bytes of a model artifact: a file, or a directory with symlinks resolved and deduplicated
    by inode (a HF snapshot is a tree of links into ``blobs/``)."""
    path = Path(path)
    seen: set[tuple[int, int]] = set()
    total = weights = files = 0
    candidates = [path] if path.is_file() else [p for p in sorted(path.rglob("*"))]
    for p in candidates:
        try:
            if not p.is_file():
                continue
            st = p.stat()
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            continue
        seen.add(key)
        files += 1
        total += st.st_size
        if p.suffix in WEIGHT_SUFFIXES:
            weights += st.st_size
    return {"path": str(path), "bytes": total, "weights_bytes": weights, "files": files}


def self_status_kb(key: str) -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


# ───────────────────────────────────────────────────────────────────────────── /proc reader


class Proc:
    """Every /proc and /sys read goes through four primitives so tests can subclass them."""

    def __init__(
        self,
        proc_root: str = "/proc",
        sys_cpu_root: str = "/sys/devices/system/cpu",
        clk_tck: int | None = None,
    ) -> None:
        self.proc_root = Path(proc_root)
        self.sys_cpu_root = Path(sys_cpu_root)
        self.clk_tck = int(clk_tck or os.sysconf("SC_CLK_TCK"))

    # primitives -------------------------------------------------------------------------
    def _read(self, rel: str) -> str:
        return (self.proc_root / rel).read_text()

    def _read_bytes(self, rel: str) -> bytes:
        return (self.proc_root / rel).read_bytes()

    def _listdir(self, rel: str) -> list[str]:
        return os.listdir(self.proc_root / rel) if rel else os.listdir(self.proc_root)

    def _read_sys(self, rel: str) -> str:
        return (self.sys_cpu_root / rel).read_text()

    # derived ----------------------------------------------------------------------------
    def loadavg(self) -> tuple[float, float, float]:
        a = self._read("loadavg").split()
        return float(a[0]), float(a[1]), float(a[2])

    def loadavg_safe(self) -> tuple[float, float, float] | None:
        try:
            return self.loadavg()
        except (OSError, ValueError, IndexError):
            return None

    def pids(self) -> list[int]:
        return sorted(int(n) for n in self._listdir("") if n.isdigit())

    def tids(self, pid: int) -> list[int]:
        return sorted(int(n) for n in self._listdir(f"{pid}/task") if n.isdigit())

    @staticmethod
    def parse_stat(text: str) -> dict[str, Any]:
        lp, rp = text.index("("), text.rindex(")")
        rest = text[rp + 2 :].split()
        return {
            "comm": text[lp + 1 : rp],
            "state": rest[0],
            "ppid": int(rest[1]),
            "utime": int(rest[11]),
            "stime": int(rest[12]),
            "processor": int(rest[36]) if len(rest) > 36 else None,
        }

    def stat(self, pid: int, tid: int | None = None) -> dict[str, Any]:
        rel = f"{pid}/stat" if tid is None else f"{pid}/task/{tid}/stat"
        return self.parse_stat(self._read(rel))

    def cpu_ticks(self, pid: int) -> int:
        """utime + stime of the whole process (all threads, exited ones included), in clock ticks."""
        st = self.stat(pid)
        return st["utime"] + st["stime"]

    def cpu_clock_s(self, pid: int) -> float | None:
        """Nanosecond-resolution process CPU time (CLOCK_PROCESS_CPUTIME of ``pid``)."""
        # Linux dynamic clock id for a process: MAKE_PROCESS_CPUCLOCK(pid, CPUCLOCK_SCHED).
        try:
            return float(time.clock_gettime(((~int(pid)) << 3) | 2))
        except (OSError, ValueError, OverflowError):
            return None

    def status_value(self, pid: int, key: str) -> str | None:
        for line in self._read(f"{pid}/status").splitlines():
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip()
        return None

    def status_kb(self, pid: int, key: str) -> int | None:
        v = self.status_value(pid, key)
        return int(v.split()[0]) if v else None

    def cmdline(self, pid: int) -> list[str]:
        raw = self._read_bytes(f"{pid}/cmdline")
        return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]

    def per_cpu_ticks(self) -> dict[int, tuple[int, int]]:
        """``{cpu: (busy, total)}`` in ticks. busy = user+nice+system+steal (irq/softirq excluded:
        ambient kernel work is not a competing job)."""
        out: dict[int, tuple[int, int]] = {}
        for line in self._read("stat").splitlines():
            m = re.match(r"cpu(\d+)\s+(.*)", line)
            if not m:
                continue
            f = [int(x) for x in m.group(2).split()]
            f += [0] * (8 - len(f))
            user, nice, system, idle, iowait, irq, softirq, steal = f[:8]
            out[int(m.group(1))] = (user + nice + system + steal, sum(f[:8]))
        return out

    def cpu_freq_khz(self, cpu: int) -> int | None:
        try:
            return int(self._read_sys(f"cpu{cpu}/cpufreq/scaling_cur_freq").strip())
        except (OSError, ValueError):
            return None

    def thread_siblings(self, cpu: int) -> list[int] | None:
        try:
            return parse_cpu_list(self._read_sys(f"cpu{cpu}/topology/thread_siblings_list").strip())
        except (OSError, ValueError):
            return None

    def ancestors(self, pid: int) -> list[int]:
        out: list[int] = []
        cur = pid
        for _ in range(64):
            try:
                cur = self.stat(cur)["ppid"]
            except (OSError, ValueError, IndexError):
                break
            if cur <= 0 or cur in out:
                break
            out.append(cur)
        return out


def pick_cpus(cpu_list: Sequence[int], n: int, proc: Proc) -> list[int]:
    """First ``n`` cpus of the claim, one per physical core before any SMT sibling."""
    cpus = sorted(cpu_list)
    n = max(1, min(int(n), len(cpus)))
    if n >= len(cpus):
        return cpus
    chosen: list[int] = []
    cores: set[int] = set()
    for c in cpus:
        sib = proc.thread_siblings(c) or [c]
        key = min(sib)
        if key in cores:
            continue
        cores.add(key)
        chosen.append(c)
        if len(chosen) == n:
            return chosen
    for c in cpus:  # fewer physical cores than requested threads: fill with siblings
        if c not in chosen:
            chosen.append(c)
            if len(chosen) == n:
                break
    return sorted(chosen)


# ───────────────────────────────────────────────────────────────────────────── affinity


class OsAffinity:
    """Applies a cpu set to EVERY thread of this process and reads it back.

    ``os.sched_setaffinity(0, ...)`` changes only the calling thread; threads created earlier (the
    OpenMP/torch pool of a previous leg) would keep their old mask."""

    def __init__(
        self,
        *,
        set_fn: Callable[[int, set[int]], None] | None = None,
        get_fn: Callable[[int], set[int]] | None = None,
        tids_fn: Callable[[], list[int]] | None = None,
    ) -> None:
        self._set = set_fn or os.sched_setaffinity
        self._get = get_fn or os.sched_getaffinity
        self._tids = tids_fn or (lambda: sorted(int(n) for n in os.listdir("/proc/self/task")))

    def current(self) -> set[int]:
        return set(self._get(0))

    def apply(self, cpus: Sequence[int]) -> dict[str, Any]:
        want = set(int(c) for c in cpus)
        for tid in self._tids():
            try:
                self._set(tid, want)
            except ProcessLookupError:
                continue
        return self.readback(want)

    def readback(self, want: set[int] | None = None) -> dict[str, Any]:
        per: dict[int, set[int]] = {}
        for tid in self._tids():
            try:
                per[tid] = set(self._get(tid))
            except ProcessLookupError:
                continue
        main = set(self._get(0))
        violations = sorted(t for t, s in per.items() if want and not s <= want)
        return {
            "applied": sorted(main),
            "n_threads": len(per),
            "uniform": len({frozenset(s) for s in per.values()}) <= 1,
            "violations": violations,
        }


class DryAffinity:
    """Dry runs and unit tests must not re-pin the host process."""

    def __init__(self, initial: Sequence[int] | None = None) -> None:
        self._cur = set(initial) if initial else set(os.sched_getaffinity(0))
        self.applied: list[list[int]] = []

    def current(self) -> set[int]:
        return set(self._cur)

    def apply(self, cpus: Sequence[int]) -> dict[str, Any]:
        self._cur = set(int(c) for c in cpus)
        self.applied.append(sorted(self._cur))
        return self.readback(self._cur)

    def readback(self, want: set[int] | None = None) -> dict[str, Any]:
        return {"applied": sorted(self._cur), "n_threads": 1, "uniform": True, "violations": []}


# ───────────────────────────────────────────────────────────────────────────── safety


def check_arm_url(arm_id: str, url: str) -> dict[str, Any]:
    """Refuse a server URL the harness must never touch (production/stack port, remote host)."""
    p = urllib.parse.urlparse(url)
    if p.scheme != "http" or not p.hostname:
        raise Refusal(f"--arm-url {arm_id}: need http://127.0.0.1:PORT, got {url!r}")
    if p.hostname not in LOOPBACK_HOSTS:
        raise Refusal(f"--arm-url {arm_id}: host {p.hostname!r} is not loopback; servers are local")
    if p.port is None:
        raise Refusal(f"--arm-url {arm_id}: an explicit port is required ({url!r})")
    if p.port in REFUSED_PORTS:
        raise Refusal(
            f"--arm-url {arm_id}: port {p.port} is a production/stack port; start a harness-owned "
            "server on a free port and never point the speed block at the serving stack"
        )
    host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname
    return {"host": p.hostname, "port": p.port, "base_url": f"http://{host}:{p.port}"}


def scan_foreign(
    proc: Proc,
    cpus: Sequence[int],
    exempt_pids: Iterable[int],
    *,
    comm_prefixes: Sequence[str] = FOREIGN_COMM_PREFIXES,
    samples: int = 3,
    interval_s: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
    max_report: int = 50,
) -> dict[str, Any]:
    """Foreign llama*/python processes that run on, or are pinned inside, ``cpus``. Read-only.

    Offender = a matching process (a) with a thread in state R on a claimed cpu, or (b) whose
    ``Cpus_allowed_list`` lies wholly inside the claim. Unpinned idle processes that merely MAY run
    there are counted, not refused (hundreds of idle pythons exist on this host); the per-window
    foreign-busy measurement in ``ContentionWatch`` covers them."""
    cpuset = set(cpus)
    exempt = set(exempt_pids)
    prefixes = tuple(comm_prefixes)
    found: dict[int, dict[str, Any]] = {}
    overlap_idle: set[int] = set()
    for s in range(max(1, samples)):
        if s:
            sleep(interval_s)
        for pid in proc.pids():
            if pid in exempt:
                continue
            try:
                st = proc.stat(pid)
            except (OSError, ValueError, IndexError):
                continue
            comm = st["comm"]
            if not comm.startswith(prefixes):
                continue
            try:
                allowed = set(parse_cpu_list(proc.status_value(pid, "Cpus_allowed_list") or ""))
            except (OSError, ValueError):
                allowed = set()
            reasons: list[str] = []
            if allowed and allowed <= cpuset:
                reasons.append("pinned_inside_claim")
            try:
                tids = proc.tids(pid)
            except OSError:
                tids = [pid]
            for tid in tids:
                try:
                    ts = proc.stat(pid, tid)
                except (OSError, ValueError, IndexError):
                    continue
                if ts["state"] == "R" and ts["processor"] in cpuset:
                    reasons.append(f"running_on_cpu{ts['processor']}")
                    break
            if reasons:
                rec = found.setdefault(
                    pid,
                    {
                        "pid": pid,
                        "comm": comm,
                        "reasons": [],
                        "cpus_allowed": format_cpu_list(allowed) if allowed else None,
                        "cmdline": _short_cmdline(proc, pid),
                    },
                )
                for r in reasons:
                    if r not in rec["reasons"]:
                        rec["reasons"].append(r)
            elif allowed & cpuset:
                overlap_idle.add(pid)
    offenders = sorted(found.values(), key=lambda r: r["pid"])
    return {
        "offenders": offenders[:max_report],
        "n_offenders": len(offenders),
        "n_unpinned_overlap_idle": len(overlap_idle),
        "samples": max(1, samples),
    }


def _short_cmdline(proc: Proc, pid: int) -> str:
    try:
        return " ".join(proc.cmdline(pid))[:160]
    except (OSError, ValueError):
        return ""


def region_lock_holders(
    bin_path: str, timeout_s: float = 60.0, runner: Callable[..., Any] = subprocess.run
) -> list[dict[str, Any]]:
    """Flattened holders from ``region-lock status --json`` (read-only)."""
    cp = runner(
        [bin_path, "status", "--json"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"region-lock status rc={cp.returncode}: {(cp.stderr or '')[-200:]}")
    rows = json.loads(cp.stdout)
    out: list[dict[str, Any]] = []
    for r in rows:
        for h in r.get("holders", []):
            out.append({"region": r.get("region"), **h})
    return out


def verify_claim_env(
    *,
    cpu_list: Sequence[int],
    claim_tag: str,
    proc: Proc,
    affinity_now: set[int],
    exempt_pids: Iterable[int],
    server_pids: dict[str, int] | None = None,
    holders_fn: Callable[[], list[dict[str, Any]]] | None = None,
    require_region_lock_holder: bool = True,
    scan_samples: int = 3,
    scan_interval_s: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """The cheap, verifiable evidence that this process owns ``cpu_list`` right now. Signals nothing."""
    cpuset = set(cpu_list)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})

    add("claim_tag", bool((claim_tag or "").strip()), f"claim_tag={claim_tag!r}")
    add(
        "affinity_equals_cpu_list",
        set(affinity_now) == cpuset,
        f"affinity={format_cpu_list(affinity_now)} cpu_list={format_cpu_list(cpuset)} "
        "(start the runner under `taskset -c <cpu-list>`)",
    )
    me = os.getpid()
    ancestors = proc.ancestors(me)
    scan = scan_foreign(
        proc,
        cpu_list,
        set(exempt_pids) | {me} | set(ancestors),
        samples=scan_samples,
        interval_s=scan_interval_s,
        sleep=sleep,
    )
    add(
        "no_foreign_llama_python_on_cpus",
        scan["n_offenders"] == 0,
        f"{scan['n_offenders']} offender(s) over {scan['samples']} samples; "
        f"{scan['n_unpinned_overlap_idle']} unpinned idle process(es) merely overlap (not refused)",
    )
    for arm, pid in sorted((server_pids or {}).items()):
        try:
            allowed = set(parse_cpu_list(proc.status_value(pid, "Cpus_allowed_list") or ""))
        except (OSError, ValueError):
            allowed = set()
        add(
            f"server_{arm}_inside_claim",
            bool(allowed) and allowed <= cpuset,
            f"pid {pid} Cpus_allowed_list={format_cpu_list(allowed) if allowed else 'unreadable'}",
        )
    holder_detail = "no holders query"
    holder_ok = False
    matched: list[dict[str, Any]] = []
    if holders_fn is not None:
        try:
            holders = holders_fn()
            matched = [h for h in holders if h.get("pid") in set(ancestors)]
            holder_ok = bool(matched)
            holder_detail = (
                "ancestor holds "
                + ",".join(f"{h.get('region')}/{h.get('request_tag')}" for h in matched)
                if matched
                else "no region-lock holder among this process's ancestors "
                "(run under `region-lock run --cpu-list ... --tag ...`)"
            )
        except Exception as exc:  # noqa: BLE001 -- fail closed, say why
            holder_detail = f"region-lock status unavailable: {type(exc).__name__}: {exc}"
    add(
        "region_lock_holder_is_ancestor",
        holder_ok,
        holder_detail,
        required=require_region_lock_holder,
    )
    warnings = []
    for h in matched:
        if h.get("request_tag") and h["request_tag"] != claim_tag:
            warnings.append(
                f"claim tag {claim_tag!r} differs from region-lock tag {h['request_tag']!r}"
            )
    return {
        "ok": all(c["ok"] for c in checks if c["required"]),
        "checks": checks,
        "offenders": scan["offenders"],
        "n_offenders": scan["n_offenders"],
        "holders": matched,
        "warnings": warnings,
        "ancestors": ancestors,
    }


# ───────────────────────────────────────────────────────────────────────────── contention / CPU


class ContentionWatch:
    """Foreign load on THIS leg's cpus, sampled during the window.

    ``foreign_busy_frac`` = (busy ticks of the leg cpus - this harness's own ticks, client plus the
    caller-owned server) / (wall x n_cpus). ``loadavg_excess`` = rise of the system 1-minute loadavg
    minus the threads this leg itself runs; it is a host-wide proxy (per-cpu loadavg does not exist),
    kept because peers on other cores also steal memory bandwidth."""

    def __init__(
        self,
        proc: Proc,
        cpus: Sequence[int],
        own_pids: Sequence[int],
        *,
        own_threads: int,
        max_foreign_busy_frac: float,
        max_loadavg_excess: float,
        interval_s: float,
        min_window_s: float,
        abort: bool = True,
        perf: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.proc, self.cpus, self.own_pids = proc, list(cpus), list(own_pids)
        self.own_threads = int(own_threads)
        self.max_frac = max_foreign_busy_frac
        self.max_load = max_loadavg_excess
        self.interval_s, self.min_window_s, self.abort, self.perf = (
            interval_s,
            min_window_s,
            abort,
            perf,
        )
        self.reasons: list[str] = []
        self.freqs: list[int] = []
        self.n_checks = 0
        self.cum_frac: float | None = None
        self.max_interval_frac = 0.0
        self.load_excess: float | None = None
        self.available = False

    def _busy(self) -> int | None:
        try:
            ticks = self.proc.per_cpu_ticks()
            return sum(ticks[c][0] for c in self.cpus if c in ticks)
        except (OSError, ValueError):
            return None

    def _own(self) -> int | None:
        try:
            return sum(self.proc.cpu_ticks(p) for p in self.own_pids)
        except (OSError, ValueError, IndexError):
            return None

    def _sample_freq(self) -> None:
        for c in self.cpus:
            v = self.proc.cpu_freq_khz(c)
            if v is not None:
                self.freqs.append(v)

    def _frac(self, b0: int, o0: int, b1: int, o1: int, dt: float) -> float:
        foreign = max(0, (b1 - b0) - (o1 - o0)) / self.proc.clk_tck
        return foreign / dt / max(1, len(self.cpus))

    def _flag(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def start(self) -> None:
        self.t0 = self.last_t = self.perf()
        self.busy0 = self.last_busy = self._busy()
        self.own0 = self.last_own = self._own()
        self.available = self.busy0 is not None and self.own0 is not None
        load = self.proc.loadavg_safe()
        self.load0 = load[0] if load else None
        self._sample_freq()

    def check(self, force: bool = False, final: bool = False) -> None:
        now = self.perf()
        if not force and now - self.last_t < self.interval_s:
            return
        self.n_checks += 1
        self._sample_freq()
        if self.available:
            busy, own = self._busy(), self._own()
            if busy is not None and own is not None:
                dt = now - self.last_t
                if dt > 0 and (dt >= self.min_window_s or final):
                    f_i = self._frac(self.last_busy, self.last_own, busy, own, dt)
                    self.max_interval_frac = max(self.max_interval_frac, f_i)
                    if f_i > self.max_frac:
                        self._flag("foreign_busy_interval")
                    self.last_t, self.last_busy, self.last_own = now, busy, own
                total_t = now - self.t0
                if total_t >= self.min_window_s or (final and total_t > 0):
                    self.cum_frac = self._frac(self.busy0, self.own0, busy, own, total_t)
                    if self.cum_frac > self.max_frac:
                        self._flag("foreign_busy_cumulative")
        load = self.proc.loadavg_safe()
        if load and self.load0 is not None:
            self.load_excess = load[0] - self.load0 - self.own_threads
            if self.max_load and self.load_excess > self.max_load:
                self._flag("loadavg_excess")
        if self.reasons and self.abort and not final:
            raise LegContended(self.reasons)

    def finish(self) -> dict[str, Any]:
        self.check(force=True, final=True)
        freq = None
        if self.freqs:
            freq = {
                "min": min(self.freqs),
                "median": statistics.median(self.freqs),
                "max": max(self.freqs),
                "n": len(self.freqs),
            }
        return {
            "contended": bool(self.reasons),
            "reasons": list(self.reasons),
            "monitor_available": self.available,
            "foreign_busy_frac": self.cum_frac,
            "max_interval_frac": self.max_interval_frac if self.available else None,
            "loadavg_excess": self.load_excess,
            "n_checks": self.n_checks,
            "thresholds": {
                "max_foreign_busy_frac": self.max_frac,
                "max_loadavg_excess": self.max_load,
            },
            "freq_khz": freq,
        }


class CpuMeter:
    """Process CPU time of the client (this process) and, for served arms, the server pid.

    Canonical value = utime+stime from /proc/<pid>/stat (clock-tick resolution). When the window is
    shorter than one second of CPU (a static-embedding query window), the nanosecond process-CPU
    clock replaces it, and ``cpu_source`` says so."""

    LOW_RES_S = 1.0

    def __init__(self, proc: Proc, pids: dict[str, int]) -> None:
        self.proc, self.pids = proc, dict(pids)
        self.base: dict[str, dict[str, Any]] = {}

    def read(self) -> dict[str, dict[str, Any]]:
        return {
            role: {"ticks": self.proc.cpu_ticks(pid), "clock_s": self.proc.cpu_clock_s(pid)}
            for role, pid in self.pids.items()
        }

    def start(self) -> dict[str, dict[str, Any]]:
        self.base = self.read()
        return self.base

    def delta(self, a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, Any]:
        tck = self.proc.clk_tck
        by_role: dict[str, dict[str, Any]] = {}
        clocks: list[float | None] = []
        for role in a:
            st = (b[role]["ticks"] - a[role]["ticks"]) / tck
            cl = (
                b[role]["clock_s"] - a[role]["clock_s"]
                if b[role]["clock_s"] is not None and a[role]["clock_s"] is not None
                else None
            )
            clocks.append(cl)
            by_role[role] = {"cpu_s_stat": st, "cpu_s_clock": cl}
        stat_s = sum(r["cpu_s_stat"] for r in by_role.values())
        clock_s = sum(c for c in clocks if c is not None) if clocks and None not in clocks else None
        if clock_s is not None and stat_s < self.LOW_RES_S:
            value, source = clock_s, "clock"
        else:
            value, source = stat_s, "stat"
        return {
            "cpu_s": value,
            "cpu_source": source,
            "cpu_s_stat": stat_s,
            "cpu_s_clock": clock_s,
            "by_role": by_role,
        }

    def since(self, snap: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return self.delta(snap, self.read())

    def total(self) -> dict[str, Any]:
        return self.since(self.base)


def index_budget_report(
    chunks_per_s: float, n_target: int, cs: Any, n_measured: int | None = None
) -> dict[str, Any]:
    """Translate a measured encode rate into the shipped index budget.

    ``implied_build_s`` = n_target / chunks_per_s; the budget is the shipped
    ``index_build_timeout_s(n_target)`` = clamp(BASE + n/INDEX_ASSUMED_TEXTS_PER_S, MIN, MAX);
    ``exposes_index_timeout`` = implied build does not fit one search's budget, i.e. the first
    search returns ``lexical_fallback:index_timeout`` and the build resumes on the next search."""
    implied = (n_target / chunks_per_s) if chunks_per_s and chunks_per_s > 0 else math.inf
    budget = float(cs.index_build_timeout_s(n_target))
    rate = float(cs.INDEX_ASSUMED_TEXTS_PER_S)
    return {
        "n_target": n_target,
        "n_measured": n_measured,
        "chunks_per_s": chunks_per_s,
        "implied_build_s": implied,
        "budget_s": budget,
        "budget_formula": (
            f"clamp({cs.INDEX_TIMEOUT_BASE_S}+n/{rate}, {cs.INDEX_TIMEOUT_MIN_S}, {cs.INDEX_TIMEOUT_MAX_S})"
        ),
        "exposes_index_timeout": bool(implied > budget),
        "searches_to_complete": (math.ceil(implied / budget) if math.isfinite(implied) else None),
        "assumed_texts_per_s": rate,
        "assumed_build_s": n_target / rate,
        "measured_over_assumed": (chunks_per_s / rate) if chunks_per_s else None,
        "max_dense_chunks": int(cs.MAX_DENSE_CHUNKS),
    }


# ───────────────────────────────────────────────────────────────────────────── arm roster


@dataclass(frozen=True)
class RosterEntry:
    arm_id: str
    kind: str  # "gguf" (served) | "st" (in-process) | "static" (in-process)
    spec: arms.ArmSpec
    text_only: bool = False
    recall_arm_ids: tuple[str, ...] = ()
    shares_forward_pass: tuple[dict[str, str], ...] = ()
    note: str = ""


def speed_roster(
    cs: Any,
    st_runtime: str,
    llama_runtime: str,
    m2v_runtime: str,
    *,
    runtime_overrides: dict[str, str] | None = None,
) -> dict[str, RosterEntry]:
    """The nine speed arms. EmbeddingGemma 2 and potion specs come from ``arms.arm_specs`` so their
    identity digests equal the frozen 2.5 manifest's; BGE-large and Granite are defined here."""
    ov = runtime_overrides or {}
    by_id = {s.arm_id: s for s in arms.arm_specs(st_runtime, llama_runtime, m2v_runtime)}
    bge_prefix = cs.query_text("bge-large-en-v1.5", "")
    if not bge_prefix:
        raise RuntimeError("shipped _QUERY_PREFIXES no longer carries a bge-large-en-v1.5 prefix")
    out: dict[str, RosterEntry] = {}

    def spec_for(arm_id: str, src: str) -> arms.ArmSpec:
        s = by_id[src]
        return dataclasses.replace(s, runtime=ov[arm_id]) if arm_id in ov else s

    def shares(arm_id: str, dtype_tag: str, family: str) -> tuple[dict[str, str], ...]:
        s256 = by_id[f"eg2-{family}-{dtype_tag}-256"]
        ident = s256.index_identity()
        return ({"arm_id": s256.arm_id, "identity_digest": ident["digest"], "identity": ident},)

    out["bge-large-gguf-f16"] = RosterEntry(
        "bge-large-gguf-f16",
        "gguf",
        arms.ArmSpec(
            "bge-large-gguf-f16",
            "gguf",
            "BAAI/bge-large-en-v1.5 (incumbent GGUF, as served)",
            f"sha256:{BGE_GGUF_SHA256}",
            1024,
            "f16",
            ov.get("bge-large-gguf-f16", UNRESOLVED_RUNTIME),
            pooling="cls",
            query_prompt=bge_prefix,
            normalise="l2",
            native_dim=1024,
            notes="incumbent; query prefix = shipped _QUERY_PREFIXES BGE instruction",
        ),
        note="incumbent as served",
    )
    out["bge-large-st-fp32"] = RosterEntry(
        "bge-large-st-fp32",
        "st",
        arms.ArmSpec(
            "bge-large-st-fp32",
            "st",
            "BAAI/bge-large-en-v1.5",
            BGE_ST_REVISION,
            1024,
            "fp32",
            ov.get("bge-large-st-fp32", st_runtime),
            pooling="cls",
            query_prompt=bge_prefix,
            normalise="l2",
            native_dim=1024,
        ),
        note="same-runtime control for the incumbent",
    )
    out["granite-97m-r2-gguf-f16"] = RosterEntry(
        "granite-97m-r2-gguf-f16",
        "gguf",
        arms.ArmSpec(
            "granite-97m-r2-gguf-f16",
            "gguf",
            "ibm-granite/granite-embedding-97m-multilingual-r2 (GGUF f16, as served)",
            "unresolved: sha256 of the served GGUF file is recorded at run time",
            0,
            "f16",
            ov.get("granite-97m-r2-gguf-f16", UNRESOLVED_RUNTIME),
            pooling="cls",
            normalise="l2",
            native_dim=0,
            notes="no instruction prefix applied; dim observed from the server",
        ),
        note="dim and file sha256 resolved at run time",
    )
    for q, tag in (("bf16", "BF16"), ("q8_0", "Q8_0")):
        sid = f"eg2-gguf-{q}"
        out[sid] = RosterEntry(
            sid,
            "gguf",
            spec_for(sid, f"eg2-gguf-{q}-768"),
            recall_arm_ids=(f"eg2-gguf-{q}-768", f"eg2-gguf-{q}-256"),
            shares_forward_pass=shares(sid, q, "gguf"),
            note="768-d requests; @256 is slice + L2 on the same forward pass (not a second leg)",
        )
    out["eg2-st-bf16"] = RosterEntry(
        "eg2-st-bf16",
        "st",
        spec_for("eg2-st-bf16", "eg2-st-bf16-768"),
        text_only=True,
        recall_arm_ids=("eg2-st-bf16-768",),
        note="BF16 reference lane; no @256 recall arm",
    )
    out["eg2-st-fp32"] = RosterEntry(
        "eg2-st-fp32",
        "st",
        spec_for("eg2-st-fp32", "eg2-st-fp32-768"),
        text_only=True,
        recall_arm_ids=("eg2-st-fp32-768", "eg2-st-fp32-256"),
        shares_forward_pass=shares("eg2-st-fp32", "fp32", "st"),
        note="FP32 is speed-only here (dropped from RECALL arms only); @256 shares the forward pass",
    )
    for sid in ("potion-code-16m-v2", "potion-base-8m"):
        out[sid] = RosterEntry(
            sid, "static", spec_for(sid, sid), recall_arm_ids=(sid,), note="single-threaded lookup"
        )
    return {k: out[k] for k in ROSTER_ORDER}


# ───────────────────────────────────────────────────────────────────────────── arm runtimes

_PROBE_TEXT = "dimension probe"


class ArmRuntime:
    """What the legs call. Real subclasses wrap the 2.5 adapters; ``FakeRuntime`` serves tests."""

    kind = ""

    def __init__(self, entry: RosterEntry) -> None:
        self.entry = entry
        self.arm_id = entry.arm_id
        self.spec = entry.spec
        self.server_pid: int | None = None
        self.server_threads: int | None = None
        self.server_meta: dict[str, Any] = {}
        self.warnings: list[str] = []

    def open(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def set_threads(self, n: int | None) -> None:
        return None

    def encode_docs(self, texts: Sequence[str], batch_size: int) -> Any:
        raise NotImplementedError

    def encode_query(self, query: str) -> Any:
        raise NotImplementedError

    def count_tokens(self, texts: Sequence[str]) -> int | None:
        return None

    def take_tokens(self) -> int | None:
        return None

    @property
    def nonfinite_rows(self) -> int:
        return 0

    def _resolve_native_dim(self, observed: int) -> None:
        if observed <= 0:
            raise RuntimeError(f"{self.arm_id}: probe returned a {observed}-d vector")
        if self.spec.native_dim in (0, None):
            self.spec = dataclasses.replace(self.spec, dim=observed, native_dim=observed)
        elif self.spec.native_dim != observed:
            raise RuntimeError(
                f"{self.arm_id}: native dim {observed} != declared {self.spec.native_dim}"
            )


class SpeedST(arms.SentenceTransformersAdapter):
    """The 2.5 ST adapter with the client batch size under the leg's control."""

    batch_size = 32

    def _encode(self, texts: list[str]) -> Any:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype("float32")


class SpeedM2V(arms.Model2VecAdapter):
    batch_size = 1024

    def __init__(self, spec: arms.ArmSpec, model_path: str) -> None:
        super().__init__(spec, model_path)
        params = inspect.signature(self.model.encode).parameters
        self._kw: dict[str, Any] = {"show_progress_bar": False}
        if "use_multiprocessing" in params:  # a pool would make "single-threaded" untrue
            self._kw["use_multiprocessing"] = False
        self._has_bs = "batch_size" in params

    def _encode(self, texts: list[str]) -> Any:
        kw = dict(self._kw)
        if self._has_bs:
            kw["batch_size"] = self.batch_size
        return self.model.encode(texts, **kw)


class SpeedLlamaServer(arms.LlamaServerAdapter):
    """The 2.5 llama-server adapter that also keeps ``usage.prompt_tokens`` (tokens/s for GGUF arms)."""

    def __init__(self, spec: arms.ArmSpec, base_url: str, *, timeout_s: float = 120.0) -> None:
        super().__init__(spec, base_url, timeout_s=timeout_s)
        self.tokens_accum = 0
        self.usage_seen = False

    def _encode(self, texts: list[str]) -> Any:
        body = json.dumps({"input": texts, "encoding_format": "float"}).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:  # noqa: S310 -- loopback harness server
            payload = json.loads(r.read())
        rows = sorted(payload["data"], key=lambda d: d["index"])
        if len(rows) != len(texts):
            raise ValueError(f"server returned {len(rows)} rows for {len(texts)} texts")
        pt = (payload.get("usage") or {}).get("prompt_tokens")
        if isinstance(pt, int):
            self.tokens_accum += pt
            self.usage_seen = True
        return np.asarray([d["embedding"] for d in rows], dtype=np.float32)


class STRuntime(ArmRuntime):
    kind = "st"

    def __init__(self, entry: RosterEntry, model_path: str) -> None:
        super().__init__(entry)
        self.model_path = model_path
        self.adapter: SpeedST | None = None

    def open(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.adapter = SpeedST(self.spec, self.model_path, text_only=self.entry.text_only)
        load_s = time.perf_counter() - t0
        raw = self.adapter._encode(self.adapter._prefixed([_PROBE_TEXT], "document"))
        self._resolve_native_dim(int(raw.shape[1]))
        self.adapter.spec = self.spec
        return {"load_s": load_s, "native_dim": int(raw.shape[1]), "model_path": self.model_path}

    def set_threads(self, n: int | None) -> None:
        if n:
            import torch

            torch.set_num_threads(int(n))

    def encode_docs(self, texts: Sequence[str], batch_size: int) -> Any:
        assert self.adapter is not None
        self.adapter.batch_size = int(batch_size)
        return self.adapter.encode_side(list(texts), "document")

    def encode_query(self, query: str) -> Any:
        assert self.adapter is not None
        self.adapter.batch_size = 1
        return self.adapter.encode_side([query], "query")

    def count_tokens(self, texts: Sequence[str]) -> int | None:
        assert self.adapter is not None
        enc = self.adapter.model.tokenizer(
            self.adapter._prefixed(list(texts), "document"), add_special_tokens=True
        )
        return int(sum(len(x) for x in enc["input_ids"]))

    @property
    def nonfinite_rows(self) -> int:
        return self.adapter.nonfinite_rows if self.adapter else 0


class StaticRuntime(ArmRuntime):
    kind = "static"

    def __init__(self, entry: RosterEntry, model_path: str) -> None:
        super().__init__(entry)
        self.model_path = model_path
        self.adapter: SpeedM2V | None = None

    def open(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.adapter = SpeedM2V(self.spec, self.model_path)
        load_s = time.perf_counter() - t0
        raw = self.adapter._encode([_PROBE_TEXT])
        self._resolve_native_dim(int(np.asarray(raw).shape[1]))
        return {"load_s": load_s, "native_dim": int(np.asarray(raw).shape[1])}

    def encode_docs(self, texts: Sequence[str], batch_size: int) -> Any:
        assert self.adapter is not None
        self.adapter.batch_size = int(batch_size)
        return self.adapter.encode_side(list(texts), "document")

    def encode_query(self, query: str) -> Any:
        assert self.adapter is not None
        self.adapter.batch_size = 1
        return self.adapter.encode_side([query], "query")

    def count_tokens(self, texts: Sequence[str]) -> int | None:
        assert self.adapter is not None
        return int(sum(len(e.ids) for e in self.adapter.model.tokenizer.encode_batch(list(texts))))

    @property
    def nonfinite_rows(self) -> int:
        return self.adapter.nonfinite_rows if self.adapter else 0


class ServedRuntime(ArmRuntime):
    """A caller-owned llama-server. This class only speaks HTTP and READS /proc/<pid>."""

    kind = "gguf"

    def __init__(
        self,
        entry: RosterEntry,
        base_url: str,
        pid: int,
        proc: Proc,
        *,
        server_threads: int | None = None,
        server_flags: str | None = None,
        model_file: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(entry)
        self.base_url, self.proc, self.timeout_s = base_url.rstrip("/"), proc, timeout_s
        self.server_pid = pid
        self.server_threads = server_threads
        self.server_flags = server_flags
        self.model_file = model_file
        self.adapter: SpeedLlamaServer | None = None

    def _props(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(self.base_url + "/props", timeout=10) as r:  # noqa: S310
                p = json.loads(r.read())
        except Exception as exc:  # noqa: BLE001 -- advisory metadata only
            return {"error": f"{type(exc).__name__}: {exc}"}
        dgs = p.get("default_generation_settings") or {}
        return {
            k: v
            for k, v in {
                "total_slots": p.get("total_slots"),
                "build_info": p.get("build_info"),
                "model_path": p.get("model_path"),
                "n_ctx": dgs.get("n_ctx"),
            }.items()
            if v is not None
        }

    def open(self) -> dict[str, Any]:
        assert self.server_pid is not None
        cmdline = self.proc.cmdline(self.server_pid)
        if not cmdline:
            raise RuntimeError(f"server pid {self.server_pid} has no readable cmdline (dead?)")
        threads_cmd = parse_server_threads(cmdline)
        if self.server_threads is None:
            self.server_threads = threads_cmd
        elif threads_cmd is not None and threads_cmd != self.server_threads:
            self.warnings.append(
                f"--arm-server-threads {self.server_threads} != cmdline --threads {threads_cmd}"
            )
        model_file = self.model_file or parse_server_model(cmdline)
        if (
            self.spec.revision.startswith("unresolved")
            and model_file
            and Path(model_file).is_file()
        ):
            # no frozen revision exists for this arm: the served file's sha256 IS its identity
            self.spec = dataclasses.replace(
                self.spec, revision=f"sha256:{sha256_file(Path(model_file))}"
            )
        props = self._props()
        if self.spec.runtime.startswith("unresolved") and props.get("build_info"):
            self.spec = dataclasses.replace(
                self.spec, runtime=f"llama-server {props['build_info']}"
            )
        self.adapter = SpeedLlamaServer(self.spec, self.base_url, timeout_s=self.timeout_s)
        raw = self.adapter._encode([_PROBE_TEXT])
        self._resolve_native_dim(int(raw.shape[1]))
        self.adapter.spec = self.spec
        self.adapter.tokens_accum, self.adapter.usage_seen = 0, False
        build = props.get("build_info")
        if build and build not in self.spec.runtime and self.entry.recall_arm_ids:
            self.warnings.append(
                f"server build_info {build!r} not in the frozen runtime string {self.spec.runtime!r}"
            )
        self.server_meta = {
            "pid": self.server_pid,
            "threads": self.server_threads,
            "cmdline": cmdline,
            "flags_echo": self.server_flags,
            "props": props,
            "model_file_from_cmdline": parse_server_model(cmdline),
        }
        return {"native_dim": int(raw.shape[1]), "server": self.server_meta}

    def encode_docs(self, texts: Sequence[str], batch_size: int) -> Any:
        assert self.adapter is not None
        return self.adapter.encode_side(list(texts), "document")

    def encode_query(self, query: str) -> Any:
        assert self.adapter is not None
        return self.adapter.encode_side([query], "query")

    def take_tokens(self) -> int | None:
        assert self.adapter is not None
        seen, n = self.adapter.usage_seen, self.adapter.tokens_accum
        self.adapter.tokens_accum, self.adapter.usage_seen = 0, False
        return n if seen else None

    @property
    def nonfinite_rows(self) -> int:
        return self.adapter.nonfinite_rows if self.adapter else 0


class FakeRuntime(ArmRuntime):
    """Deterministic, sleep-free stand-in for any kind of arm (``--dry-run`` and the unit tests)."""

    def __init__(
        self,
        entry: RosterEntry,
        *,
        dim: int = 16,
        fail_open: str | None = None,
        nonfinite: bool = False,
        server_threads: int | None = 8,
    ) -> None:
        super().__init__(entry)
        self.kind = entry.kind
        self.dim, self.fail_open, self.nonfinite = dim, fail_open, nonfinite
        self.server_threads = server_threads if entry.kind == "gguf" else None
        self.thread_calls: list[int | None] = []
        self.doc_calls: list[int] = []
        self.query_calls = 0
        self._tokens = 0
        self._nf = 0

    def _vec(self, text: str) -> np.ndarray:
        d = np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8).astype(np.float32)
        v = np.resize(d, self.dim) + 1.0
        return v / np.linalg.norm(v)

    def open(self) -> dict[str, Any]:
        if self.fail_open:
            raise RuntimeError(self.fail_open)
        if self.spec.native_dim in (0, None):
            self.spec = dataclasses.replace(self.spec, dim=self.dim, native_dim=self.dim)
        return {"load_s": 0.0, "native_dim": self.dim, "fake": True}

    def set_threads(self, n: int | None) -> None:
        self.thread_calls.append(n)

    def encode_docs(self, texts: Sequence[str], batch_size: int) -> Any:
        self.doc_calls.append(len(texts))
        self._tokens += sum(len(t.split()) for t in texts)
        out = np.stack([self._vec(t) for t in texts])
        if self.nonfinite:
            out[0, 0] = np.nan
            self._nf += 1
        return out

    def encode_query(self, query: str) -> Any:
        self.query_calls += 1
        return self._vec(query)[None, :]

    def count_tokens(self, texts: Sequence[str]) -> int | None:
        return None if self.kind == "gguf" else int(sum(len(t.split()) for t in texts))

    def take_tokens(self) -> int | None:
        if self.kind != "gguf":
            return None
        n, self._tokens = self._tokens, 0
        return n

    @property
    def nonfinite_rows(self) -> int:
        return self._nf


# ───────────────────────────────────────────────────────────────────────────── legs, config, plan


@dataclass(frozen=True)
class Leg:
    leg_id: str
    kind: str  # throughput | query | index_build | rss | disk | rss_end
    batch: int | None = None
    threads: Any = None  # int | "full" | "server" | "single" | None
    side: str | None = None  # query legs: inproc | served
    arm_kinds: tuple[str, ...] = ("st", "static", "gguf")

    def applies(self, entry: RosterEntry) -> bool:
        return entry.kind in self.arm_kinds


@dataclass
class RunConfig:
    out_dir: Path
    cpu_list: list[int]
    claim_tag: str
    dry_run: bool = False
    run_id: str = ""
    passes: int = 2
    batches: tuple[int, ...] = (1, 8, 32, 128)
    threads: tuple[int, ...] = (1, 4)
    full_socket: bool = False
    subset_n: int = 256
    subset_n_b1: int = 64
    subset_n_t1: int | None = None
    static_subset_n: int = 4096
    query_threads: int = 4
    repeats: int = 3
    index_batch: int = 32
    index_threads: int = 4
    n_index: int = 4096
    warmup_chunks: int = 32
    warmup_queries: int = 20
    aa_arm: str | None = None
    only: tuple[str, ...] = FAMILIES
    retry_contended: bool = True
    max_foreign_busy_frac: float = 0.10
    max_loadavg_excess: float = 6.0
    check_interval_s: float = 2.0
    min_window_s: float = 1.0
    rss_child_timeout_s: float = 1800.0

    def plan_params(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in ("out_dir", "cpu_list", "claim_tag", "dry_run", "run_id", "retry_contended"):
            d.pop(k, None)
        for k in (
            "max_foreign_busy_frac",
            "max_loadavg_excess",
            "check_interval_s",
            "min_window_s",
        ):
            d.pop(k, None)
        d.pop("rss_child_timeout_s", None)
        return d


def build_legs(cfg: RunConfig) -> list[Leg]:
    """Decision-relevant legs first (index build, query latency, pool-like threads 4), the slow
    single-thread sweep last, so a claim that ends early still yields the numbers 2.1 needs."""
    batches = sorted(set(cfg.batches), key=lambda b: (b != 32, b))
    legs: list[Leg] = []
    if "index" in cfg.only:
        legs.append(
            Leg(
                f"idx_b{cfg.index_batch}",
                "index_build",
                batch=cfg.index_batch,
                threads=cfg.index_threads,
            )
        )
    if "query" in cfg.only:
        legs.append(
            Leg(
                "q_inproc",
                "query",
                threads=cfg.query_threads,
                side="inproc",
                arm_kinds=("st", "static"),
            )
        )
        legs.append(Leg("q_served", "query", threads="server", side="served", arm_kinds=("gguf",)))
    if "throughput" in cfg.only:
        for t in sorted(set(cfg.threads), reverse=True):
            if t != 1:
                legs += [
                    Leg(f"tp_b{b}_t{t}", "throughput", batch=b, threads=t, arm_kinds=("st",))
                    for b in batches
                ]
        legs += [
            Leg(f"tp_b{b}_srv", "throughput", batch=b, threads="server", arm_kinds=("gguf",))
            for b in batches
        ]
        legs += [
            Leg(f"tp_b{b}_single", "throughput", batch=b, threads="single", arm_kinds=("static",))
            for b in batches
        ]
        if 1 in cfg.threads:
            legs += [
                Leg(f"tp_b{b}_t1", "throughput", batch=b, threads=1, arm_kinds=("st",))
                for b in batches
            ]
        if cfg.full_socket:
            legs += [
                Leg(f"tp_b{b}_tfull", "throughput", batch=b, threads="full", arm_kinds=("st",))
                for b in FULL_SOCKET_BATCHES
                if b in cfg.batches
            ]
    return legs


def aa_leg_ids(cfg: RunConfig, aa_kind: str) -> list[str]:
    """The A/A reference legs for an arm kind: pool-like throughput, query, index build."""
    tp = {"st": "tp_b32_t4", "gguf": "tp_b32_srv", "static": "tp_b32_single"}[aa_kind]
    q = "q_served" if aa_kind == "gguf" else "q_inproc"
    out = []
    if "throughput" in cfg.only:
        out.append(tp)
    if "query" in cfg.only:
        out.append(q)
    if "index" in cfg.only:
        out.append(f"idx_b{cfg.index_batch}")
    return out


@dataclass(frozen=True)
class Unit:
    key: str
    phase: str  # static | aa_early | main | aa_late
    leg_id: str
    arm_id: str
    rep: int | None = None
    order_pos: int | None = None
    order_dir: str | None = None  # fwd | rev | na


STATIC_LEGS = {
    "rss": Leg("rss", "rss"),
    "disk": Leg("disk", "disk"),
    "rss_end": Leg("rss_end", "rss_end", arm_kinds=("gguf",)),
}


class RowStore:
    """Append-only JSONL log with last-row-wins resume semantics."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.latest: dict[str, dict[str, Any]] = {}
        self.bad_lines = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    self.latest[row["unit_key"]] = row
                except (json.JSONDecodeError, KeyError):
                    self.bad_lines += 1

    def append(self, row: dict[str, Any]) -> None:
        line = dumps(row)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.latest[row["unit_key"]] = json.loads(line)

    def is_final(self, unit_key: str, retry_contended: bool = True) -> bool:
        row = self.latest.get(unit_key)
        if row is None:
            return False
        st = row.get("status")
        if st in ("ok", "not_applicable"):
            return True
        if st == "contended":
            return not retry_contended
        return False  # error / anything else: retry

    def rows(self) -> list[dict[str, Any]]:
        return list(self.latest.values())


def load_rows(path: Path | str) -> list[dict[str, Any]]:
    return RowStore(Path(path)).rows()


# ───────────────────────────────────────────────────────────────────────────── the runner


class SpeedRunner:
    def __init__(
        self,
        cfg: RunConfig,
        roster: dict[str, RosterEntry],
        runtimes: dict[str, ArmRuntime],
        corpus: list[str],
        queries: list[str],
        *,
        cs: Any,
        proc: Proc | None = None,
        affinity: Any = None,
        claim_report: dict[str, Any] | None = None,
        rss_child: Callable[[RosterEntry], dict[str, Any]] | None = None,
        model_files: dict[str, str] | None = None,
        model_paths: dict[str, str] | None = None,
        now: Callable[[], float] = time.time,
        perf: Callable[[], float] = time.perf_counter,
        after_unit: Callable[[Unit, dict[str, Any]], None] | None = None,
        inputs_meta: dict[str, Any] | None = None,
        extra_manifest: dict[str, Any] | None = None,
    ) -> None:
        self.cfg, self.roster, self.runtimes = cfg, roster, runtimes
        self.corpus, self.queries, self.cs = corpus, queries, cs
        self.proc = proc or Proc()
        self.affinity = affinity or (DryAffinity(cfg.cpu_list) if cfg.dry_run else OsAffinity())
        self.claim_report = claim_report or {}
        self.rss_child, self.after_unit = rss_child, after_unit
        self.model_files, self.model_paths = model_files or {}, model_paths or {}
        self.now, self.perf = now, perf
        self.inputs_meta, self.extra_manifest = inputs_meta or {}, extra_manifest or {}
        self.arm_order = [a for a in ROSTER_ORDER if a in runtimes]
        self.legs = build_legs(cfg)
        self.legs_by_id = {leg.leg_id: leg for leg in self.legs} | STATIC_LEGS
        self.store = RowStore(cfg.out_dir / "speed.jsonl")
        self.open_errors: dict[str, str] = {}
        self.open_meta: dict[str, dict[str, Any]] = {}
        self._opened: set[str] = set()
        self._warmed: set[tuple[Any, ...]] = set()
        self._subsets: dict[int, list[str]] = {}
        self.aa_arm = self._pick_aa_arm()

    # ----- plan --------------------------------------------------------------------------
    def _pick_aa_arm(self) -> str | None:
        if self.cfg.aa_arm:
            if self.cfg.aa_arm not in self.runtimes:
                raise Refusal(f"--aa-arm {self.cfg.aa_arm!r} is not a selected arm")
            return self.cfg.aa_arm
        for pref in ("bge-large-gguf-f16", "bge-large-st-fp32"):
            if pref in self.runtimes:
                return pref
        return self.arm_order[0] if self.arm_order else None

    def build_plan(self) -> list[Unit]:
        cfg, units = self.cfg, []
        if (
            "rss" in cfg.only
        ):  # isolated child per in-process arm, twice: RSS is allocator-dependent
            for a in self.arm_order:
                n_rss = 1 if self.roster[a].kind == "gguf" else 2
                units += [Unit(f"static|rss|{a}|p{i}", "static", "rss", a, i) for i in range(n_rss)]
        if "disk" in cfg.only:
            units += [Unit(f"static|disk|{a}", "static", "disk", a, 0) for a in self.arm_order]
        aa_legs: list[str] = []
        if self.aa_arm:
            aa_legs = [
                lid
                for lid in aa_leg_ids(cfg, self.roster[self.aa_arm].kind)
                if lid in self.legs_by_id and self.legs_by_id[lid].applies(self.roster[self.aa_arm])
            ]
        units += [
            Unit(f"aa_early|{lid}|{self.aa_arm}|p0", "aa_early", lid, self.aa_arm, 0)
            for lid in aa_legs
        ]
        for leg in self.legs:
            applicable = [a for a in self.arm_order if leg.applies(self.roster[a])]
            for p, pos, a in aba_order(applicable, cfg.passes):
                units.append(
                    Unit(
                        f"main|{leg.leg_id}|{a}|p{p}",
                        "main",
                        leg.leg_id,
                        a,
                        p,
                        pos,
                        "fwd" if p % 2 == 0 else "rev",
                    )
                )
            if leg.kind == "query":  # explicit not_applicable rows for the other side
                for a in self.arm_order:
                    if not leg.applies(self.roster[a]):
                        units.append(
                            Unit(
                                f"main|{leg.leg_id}|{a}|na", "main", leg.leg_id, a, None, None, "na"
                            )
                        )
        units += [
            Unit(f"aa_late|{lid}|{self.aa_arm}|p0", "aa_late", lid, self.aa_arm, 0)
            for lid in reversed(aa_legs)
        ]
        if "rss" in cfg.only:
            units += [
                Unit(f"static|rss_end|{a}", "static", "rss_end", a, 0)
                for a in self.arm_order
                if self.roster[a].kind == "gguf"
            ]
        return units

    def plan_sha256(self) -> str:
        return sha256_json(
            {
                "inputs": {
                    "corpus_sha256": self.inputs_meta.get("corpus_sha256"),
                    "queries_sha256": self.inputs_meta.get("queries_sha256"),
                    "n_chunks": len(self.corpus),
                    "n_queries": len(self.queries),
                },
                "arms": [
                    (a, self.roster[a].spec.index_identity()["digest"]) for a in self.arm_order
                ],
                "legs": [dataclasses.asdict(leg) for leg in self.legs],
                "params": self.cfg.plan_params(),
                "aa_arm": self.aa_arm,
                "units": [u.key for u in self.build_plan()],
            }
        )

    # ----- helpers -----------------------------------------------------------------------
    def _subset(self, n: int) -> list[str]:
        if n not in self._subsets:
            self._subsets[n] = stride_subset(self.corpus, n)
        return self._subsets[n]

    def _subset_size(self, rt: ArmRuntime, leg: Leg) -> int:
        cfg = self.cfg
        if rt.kind == "static":
            return cfg.static_subset_n
        if leg.batch == 1:
            n = cfg.subset_n_b1
            if leg.threads == 1 and cfg.subset_n_t1:
                n = min(n, cfg.subset_n_t1)
            return n
        if leg.threads == 1 and cfg.subset_n_t1:
            return cfg.subset_n_t1
        return cfg.subset_n

    def _cpus_for(self, rt: ArmRuntime, leg: Leg) -> tuple[list[int], int | None]:
        cl = self.cfg.cpu_list
        if rt.kind == "gguf":
            return list(cl), None
        if rt.kind == "static":
            n = 1
        else:
            n = len(cl) if leg.threads == "full" else int(leg.threads)
        n = max(1, min(n, len(cl)))
        return pick_cpus(cl, n, self.proc), n

    def _ensure_open(self, arm_id: str) -> ArmRuntime:
        rt = self.runtimes[arm_id]
        if arm_id in self.open_errors:
            raise RuntimeError(f"arm failed to open: {self.open_errors[arm_id]}")
        if arm_id not in self._opened:
            try:
                self.open_meta[arm_id] = rt.open()
            except Exception as exc:  # noqa: BLE001 -- recorded on every unit of this arm
                self.open_errors[arm_id] = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(f"arm failed to open: {self.open_errors[arm_id]}") from exc
            self._opened.add(arm_id)
        return rt

    def _prepare(self, rt: ArmRuntime, leg: Leg, cpus: list[int], nthreads: int | None) -> None:
        self.affinity.apply(cpus)
        rt.set_threads(nthreads)
        key = (rt.arm_id, leg.kind, leg.batch, leg.threads)
        if key in self._warmed:
            return
        if leg.kind == "query":
            for q in self.queries[: self.cfg.warmup_queries]:
                rt.encode_query(q)
        else:
            b = leg.batch or 1
            n_warm = min(len(self.corpus), max(self.cfg.warmup_chunks, b))
            warm = self.corpus[:n_warm]
            for i in range(0, len(warm), b):
                rt.encode_docs(warm[i : i + b], b)
        self._warmed.add(key)

    def _base_row(self, u: Unit, rt: ArmRuntime | None, leg: Leg) -> dict[str, Any]:
        entry = self.roster[u.arm_id]
        spec = rt.spec if rt is not None else entry.spec
        ident = spec.index_identity()
        return {
            "schema": ROW_SCHEMA,
            "run_id": self.cfg.run_id,
            "unit_key": u.key,
            "phase": u.phase,
            "arm_id": u.arm_id,
            "arm_kind": entry.kind,
            "arm_identity": ident,
            "arm_identity_digest": ident["digest"],
            "recall_arm_ids": list(entry.recall_arm_ids),
            "shares_forward_pass": [dict(x) for x in entry.shares_forward_pass],
            "leg": leg.leg_id,
            "leg_kind": leg.kind,
            "rep": u.rep,
            "order_pos": u.order_pos,
            "order_dir": u.order_dir,
            "claim_tag": self.cfg.claim_tag,
            "dry_run": self.cfg.dry_run,
            "cpu_list_claimed": format_cpu_list(self.cfg.cpu_list),
        }

    def _window(
        self,
        rt: ArmRuntime,
        cpus: list[int],
        nthreads: int | None,
        body: Callable[[ContentionWatch, CpuMeter, dict[str, Any]], None],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg = self.cfg
        pids = {"client": os.getpid()}
        if rt.server_pid:
            pids["server"] = rt.server_pid
        watch = ContentionWatch(
            self.proc,
            cpus,
            list(pids.values()),
            own_threads=nthreads or (rt.server_threads or 1),
            max_foreign_busy_frac=cfg.max_foreign_busy_frac,
            max_loadavg_excess=cfg.max_loadavg_excess,
            interval_s=cfg.check_interval_s,
            min_window_s=cfg.min_window_s,
            perf=self.perf,
        )
        meter = CpuMeter(self.proc, pids)
        res: dict[str, Any] = {}
        load_before = self.proc.loadavg_safe()
        rss_before = self._server_rss_kb(rt)
        t_start = self.now()
        status, reasons, aborted = "ok", [], False
        meter.start()
        watch.start()
        t0 = self.perf()
        try:
            body(watch, meter, res)
        except LegContended as exc:
            status, reasons, aborted = "contended", list(exc.reasons), True
        wall = self.perf() - t0
        cpu = meter.total()
        cont = watch.finish()
        t_end = self.now()
        load_after = self.proc.loadavg_safe()
        rss_after = self._server_rss_kb(rt)
        aff = self.affinity.readback(set(cpus))
        if aff["violations"]:
            status = "contended"
            reasons.append("affinity_violation")
        elif cont["contended"] and status == "ok":
            status, reasons = "contended", list(cont["reasons"])
        env = {
            "status": status,
            "aborted": aborted,
            "wall_s": wall,
            "t_start": t_start,
            "t_end": t_end,
            "loadavg_before": load_before,
            "loadavg_after": load_after,
            "cpus_requested": format_cpu_list(cpus),
            "cpus_applied": format_cpu_list(aff["applied"]),
            "affinity_uniform": aff["uniform"],
            "affinity_violations": aff["violations"],
            "freq_khz": cont["freq_khz"],
            "contention": {**cont, "reasons": reasons or cont["reasons"]},
            "cpu": cpu,
            "server_rss_kb": [x for x in (rss_before, rss_after) if x is not None],
        }
        return env, res

    def _server_rss_kb(self, rt: ArmRuntime) -> int | None:
        if not rt.server_pid:
            return None
        try:
            return self.proc.status_kb(rt.server_pid, "VmRSS")
        except (OSError, ValueError):
            return None

    def _server_block(self, rt: ArmRuntime) -> dict[str, Any] | None:
        if rt.kind != "gguf":
            return None
        block = {k: v for k, v in rt.server_meta.items() if k != "props"}
        block["props"] = rt.server_meta.get("props")
        block["threads"] = rt.server_threads
        if rt.server_pid:
            try:
                block["cpus_allowed"] = self.proc.status_value(rt.server_pid, "Cpus_allowed_list")
            except (OSError, ValueError):
                block["cpus_allowed"] = None
        return block

    # ----- units -------------------------------------------------------------------------
    def _unit_throughput(self, u: Unit, rt: ArmRuntime, leg: Leg, row: dict[str, Any]) -> None:
        texts = self._subset(self._subset_size(rt, leg))
        B = int(leg.batch or 1)
        cpus, nthreads = self._cpus_for(rt, leg)
        self._prepare(rt, leg, cpus, nthreads)
        rt.take_tokens()
        nf0 = rt.nonfinite_rows

        def body(watch: ContentionWatch, meter: CpuMeter, res: dict[str, Any]) -> None:
            res["n_done"] = 0
            for i in range(0, len(texts), B):
                sl = texts[i : i + B]
                rt.encode_docs(sl, B)
                res["n_done"] += len(sl)
                watch.check()

        env, res = self._window(rt, cpus, nthreads, body)
        n_done = int(res.get("n_done", 0))
        tokens, source = rt.take_tokens(), "server_usage"
        if tokens is None:
            tokens, source = rt.count_tokens(texts[:n_done]) if n_done else 0, "tokenizer"
        wall = env["wall_s"]
        row.update(env)
        row.update(
            config={
                "batch": B,
                "threads": nthreads,
                "threads_kind": {"gguf": "server", "static": "static_single", "st": "torch"}[
                    rt.kind
                ],
                "server_threads": rt.server_threads,
                "n_chunks_planned": len(texts),
                "subset_sha256": sha256_text("\n".join(texts)),
            },
            server=self._server_block(rt),
            n_chunks=n_done,
            tokens_total=tokens,
            tokens_source=source if tokens is not None else None,
            chunks_per_s=(n_done / wall) if wall > 0 and n_done else None,
            tokens_per_s=(tokens / wall) if tokens is not None and wall > 0 and n_done else None,
            nonfinite_rows=rt.nonfinite_rows - nf0,
            cpu_s_window=env["cpu"]["cpu_s"],
        )

    def _unit_query(self, u: Unit, rt: ArmRuntime, leg: Leg, row: dict[str, Any]) -> None:
        cpus, nthreads = self._cpus_for(rt, leg)
        self._prepare(rt, leg, cpus, nthreads)
        queries, reps, seed0 = self.queries, self.cfg.repeats, 1000 * (u.rep or 0)

        def body(watch: ContentionWatch, meter: CpuMeter, res: dict[str, Any]) -> None:
            res.update(latencies_ms=[], repeats=[], n_done=0)
            for r in range(reps):
                order = list(queries)
                random.Random(seed0 + r).shuffle(order)
                snap, lat = meter.read(), []
                for q in order:
                    t = self.perf()
                    rt.encode_query(q)
                    dt = (self.perf() - t) * 1000.0
                    lat.append(dt)
                    res["latencies_ms"].append(dt)
                    res["n_done"] += 1
                    watch.check()
                cpu_r = meter.since(snap)
                res["repeats"].append(
                    {
                        "repeat": r,
                        **latency_stats(lat),
                        "cpu_s": cpu_r["cpu_s"],
                        "cpu_source": cpu_r["cpu_source"],
                    }
                )

        env, res = self._window(rt, cpus, nthreads, body)
        lat = res.get("latencies_ms", [])
        n = int(res.get("n_done", 0))
        row.update(env)
        row.update(
            config={
                "threads": nthreads,
                "threads_kind": {"gguf": "server", "static": "static_single", "st": "torch"}[
                    rt.kind
                ],
                "server_threads": rt.server_threads,
                "repeats": reps,
                "n_distinct_queries": len(set(queries)),
                "query_side": leg.side,
            },
            server=self._server_block(rt),
            n_queries=n,
            latencies_ms=[round(x, 4) for x in lat],
            repeats=res.get("repeats", []),
            cpu_s_window=env["cpu"]["cpu_s"],
            cpu_s_per_query=(env["cpu"]["cpu_s"] / n) if n else None,
            **{k: v for k, v in latency_stats(lat).items() if k != "n"},
        )

    def _unit_index(self, u: Unit, rt: ArmRuntime, leg: Leg, row: dict[str, Any]) -> None:
        texts = self.corpus[: self.cfg.n_index]
        B = int(leg.batch or 32)
        cpus, nthreads = self._cpus_for(rt, leg)
        self._prepare(rt, leg, cpus, nthreads)
        rt.take_tokens()
        nf0 = rt.nonfinite_rows

        def body(watch: ContentionWatch, meter: CpuMeter, res: dict[str, Any]) -> None:
            res.update(n_done=0, finite=True, dim=None)
            for i in range(0, len(texts), B):
                sl = texts[i : i + B]
                v = rt.encode_docs(sl, B)
                if not np.isfinite(v).all():
                    res["finite"] = False
                res["dim"] = int(v.shape[1])
                res["n_done"] += len(sl)
                watch.check()

        env, res = self._window(rt, cpus, nthreads, body)
        n_done = int(res.get("n_done", 0))
        wall = env["wall_s"]
        cps = (n_done / wall) if wall > 0 and n_done else None
        report = (
            index_budget_report(cps, INDEX_TARGET_CHUNKS, self.cs, n_measured=n_done)
            if cps
            else None
        )
        row.update(env)
        row.update(
            config={
                "batch": B,
                "threads": nthreads,
                "threads_kind": {"gguf": "server", "static": "static_single", "st": "torch"}[
                    rt.kind
                ],
                "server_threads": rt.server_threads,
                "n_index_target": INDEX_TARGET_CHUNKS,
            },
            server=self._server_block(rt),
            n_index_chunks=n_done,
            build_wall_s=wall,
            chunks_per_s=cps,
            implied_build_s_4096=(report or {}).get("implied_build_s"),
            budget=report,
            finite=bool(res.get("finite", True)),
            nonfinite_rows=rt.nonfinite_rows - nf0,
            dim=res.get("dim"),
            cpu_s_window=env["cpu"]["cpu_s"],
        )

    def _rss_of_server_or_self(self, rt: ArmRuntime, row: dict[str, Any], note: str) -> None:
        pid = rt.server_pid
        if pid:
            row.update(
                rss_source="server_proc_status",
                server_pid=pid,
                rss_after_load_kb=self.proc.status_kb(pid, "VmRSS"),
                hwm_kb=self.proc.status_kb(pid, "VmHWM"),
                note=note,
            )
        elif self.cfg.dry_run:
            row.update(
                rss_source="dry_run_self",
                rss_after_load_kb=self_status_kb("VmRSS"),
                hwm_kb=self_status_kb("VmHWM"),
            )
        else:
            raise RuntimeError(f"{rt.arm_id}: no server pid to read RSS from")

    def _unit_rss(self, u: Unit, rt: ArmRuntime, entry: RosterEntry, row: dict[str, Any]) -> None:
        row.update(status="ok")
        if entry.kind == "gguf":
            self._rss_of_server_or_self(
                rt,
                row,
                "as found at harness start; includes file-backed GGUF pages mapped by the server",
            )
        elif self.cfg.dry_run or self.rss_child is None:
            row.update(
                rss_source="dry_run_self" if self.cfg.dry_run else "in_process_unisolated",
                rss_after_load_kb=self_status_kb("VmRSS"),
                hwm_kb=self_status_kb("VmHWM"),
            )
        else:
            row.update(rss_source="isolated_child_process", **self.rss_child(entry))

    def _unit_rss_end(self, u: Unit, rt: ArmRuntime, row: dict[str, Any]) -> None:
        row.update(status="ok")
        self._rss_of_server_or_self(
            rt, row, "end of run: VmHWM is the server's peak over every leg of this run"
        )

    def _unit_disk(self, u: Unit, rt: ArmRuntime, entry: RosterEntry, row: dict[str, Any]) -> None:
        path: str | None
        if entry.kind == "gguf":
            path = self.model_files.get(entry.arm_id)
            if not path and rt.server_pid:
                path = parse_server_model(self.proc.cmdline(rt.server_pid))
        else:
            path = self.model_paths.get(entry.arm_id)
        if not path or not Path(path).exists():
            if self.cfg.dry_run:
                row.update(
                    status="ok",
                    disk={"path": None, "bytes": 0, "weights_bytes": 0, "files": 0},
                    note="dry run",
                )
                return
            raise RuntimeError(f"no readable model artifact path for {entry.arm_id} (got {path!r})")
        du = disk_usage(Path(path))
        du["repeat_bytes"] = disk_usage(Path(path))[
            "bytes"
        ]  # a second stat: the A/A of a file size
        if entry.kind == "gguf" and Path(path).is_file() and du["bytes"] < (4 << 30):
            du["sha256"] = sha256_file(Path(path))
            if entry.arm_id == "bge-large-gguf-f16" and du["sha256"] != BGE_GGUF_SHA256:
                du["identity_warning"] = (
                    f"served sha256 differs from the frozen incumbent {BGE_GGUF_SHA256}"
                )
        row.update(status="ok", disk=du)

    def _execute(self, u: Unit) -> dict[str, Any]:
        leg = self.legs_by_id[u.leg_id]
        entry = self.roster[u.arm_id]
        if u.order_dir == "na":
            row = self._base_row(u, None, leg)
            row.update(
                status="not_applicable",
                reason=f"{leg.side} leg does not apply to a {entry.kind} arm",
            )
            return row
        if leg.kind in ("rss", "rss_end", "disk"):
            rt = self.runtimes[
                u.arm_id
            ]  # static units never open the arm: they read files and /proc
            row = self._base_row(u, rt, leg)
            if leg.kind == "rss":
                self._unit_rss(u, rt, entry, row)
            elif leg.kind == "rss_end":
                self._unit_rss_end(u, rt, row)
            else:
                self._unit_disk(u, rt, entry, row)
            return row
        rt = self._ensure_open(u.arm_id)
        row = self._base_row(u, rt, leg)
        row["warnings"] = list(rt.warnings)
        if leg.kind == "throughput":
            self._unit_throughput(u, rt, leg, row)
        elif leg.kind == "query":
            self._unit_query(u, rt, leg, row)
        else:
            self._unit_index(u, rt, leg, row)
        return row

    def _run_unit(self, u: Unit) -> dict[str, Any]:
        try:
            row = self._execute(u)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 -- a failing arm is recorded, never dropped
            leg = self.legs_by_id[u.leg_id]
            row = self._base_row(u, None, leg)
            row.update(status="error", error=f"{type(exc).__name__}: {exc}", t_start=self.now())
        self.store.append(row)
        if self.after_unit:
            self.after_unit(u, row)
        return row

    # ----- run ---------------------------------------------------------------------------
    def manifest_path(self) -> Path:
        return self.cfg.out_dir / "run_manifest.json"

    def _write_manifest(self, plan: list[Unit], plan_sha: str, finished: bool) -> None:
        cfg = self.cfg
        man = {
            "schema": SCHEMA,
            "run_id": cfg.run_id,
            "plan_sha256": plan_sha,
            "dry_run": cfg.dry_run,
            "claim_tag": cfg.claim_tag,
            "cpu_list": format_cpu_list(cfg.cpu_list),
            "claim_report": self.claim_report,
            "inputs": self.inputs_meta,
            "units_planned": len(plan),
            "legs": [dataclasses.asdict(leg) for leg in self.legs],
            "aa_arm": self.aa_arm,
            "thresholds": {
                "max_foreign_busy_frac": cfg.max_foreign_busy_frac,
                "max_loadavg_excess": cfg.max_loadavg_excess,
                "check_interval_s": cfg.check_interval_s,
                "min_window_s": cfg.min_window_s,
            },
            "config": cfg.plan_params(),
            "arms": {
                a: {
                    "kind": self.roster[a].kind,
                    "identity": self.runtimes[a].spec.index_identity(),
                    "recall_arm_ids": list(self.roster[a].recall_arm_ids),
                    "shares_forward_pass": [dict(x) for x in self.roster[a].shares_forward_pass],
                    "note": self.roster[a].note,
                    "model_path": self.model_paths.get(a),
                    "model_file": self.model_files.get(a),
                    "open_meta": self.open_meta.get(a),
                    "open_error": self.open_errors.get(a),
                    "warnings": list(self.runtimes[a].warnings),
                }
                for a in self.arm_order
            },
            "host": host_facts(self.proc),
            "env": {
                k: v
                for k, v in sorted(os.environ.items())
                if k.startswith(("OMP", "MKL", "KMP", "TORCH", "HF_", "TOKENIZERS", "GOMP"))
            },
            "finished": finished,
            "script_sha256": sha256_file(Path(__file__)),
            **self.extra_manifest,
        }
        self.manifest_path().write_text(dumps(man, indent=1) + "\n")

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        plan = self.build_plan()
        plan_sha = self.plan_sha256()
        if self.manifest_path().exists():
            prev = json.loads(self.manifest_path().read_text())
            if prev.get("plan_sha256") != plan_sha:
                raise Refusal(
                    f"plan changed since {self.manifest_path()} was written "
                    f"({prev.get('plan_sha256')} != {plan_sha}); use a new --out"
                )
            cfg.run_id = prev.get("run_id") or cfg.run_id
        self._write_manifest(plan, plan_sha, finished=False)
        skipped = executed = 0
        counts: Counter[str] = Counter()
        for u in plan:
            if self.store.is_final(u.key, cfg.retry_contended):
                skipped += 1
                continue
            row = self._run_unit(u)
            executed += 1
            counts[row.get("status", "?")] += 1
        nf = compute_noise_floor(self.store.rows(), self.aa_arm)
        (cfg.out_dir / "noise_floor.json").write_text(dumps(nf, indent=1) + "\n")
        self._write_manifest(plan, plan_sha, finished=True)
        return {
            "units_planned": len(plan),
            "executed": executed,
            "skipped_done": skipped,
            "status_counts": dict(counts),
            "plan_sha256": plan_sha,
        }


def host_facts(proc: Proc) -> dict[str, Any]:
    facts: dict[str, Any] = {"loadavg": proc.loadavg_safe()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                facts["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        facts["kernel"] = os.uname().release
    except AttributeError:
        pass
    try:
        facts["governor"] = proc._read_sys("cpu0/cpufreq/scaling_governor").strip()
    except OSError:
        facts["governor"] = None
    facts["python"] = sys.version.split()[0]
    facts["numpy"] = np.__version__
    for mod in ("torch", "sentence_transformers", "transformers", "model2vec"):
        m = sys.modules.get(mod)
        if m is not None:
            facts[mod] = getattr(m, "__version__", "?")
    return facts


# ───────────────────────────────────────────────────────────────────────────── noise floor

NOISE_METRICS: dict[str, tuple[str, str]] = {
    "chunks_per_s": ("chunks/s", "throughput"),
    "tokens_per_s": ("tokens/s", "throughput"),
    "query_p50_ms": ("ms", "query"),
    "query_p95_ms": ("ms", "query"),
    "query_max_ms": ("ms", "query"),
    "cpu_s_per_query": ("cpu-s/query", "query"),
    "index_build_s_4096": ("s", "index"),
}
_ROW_FIELD = {
    "chunks_per_s": "chunks_per_s",
    "tokens_per_s": "tokens_per_s",
    "query_p50_ms": "p50_ms",
    "query_p95_ms": "p95_ms",
    "query_max_ms": "max_ms",
    "cpu_s_per_query": "cpu_s_per_query",
    "index_build_s_4096": "implied_build_s_4096",
}


def _aa_ref_leg(kind: str, family: str) -> str:
    if family == "throughput":
        return {"st": "tp_b32_t4", "gguf": "tp_b32_srv", "static": "tp_b32_single"}[kind]
    if family == "query":
        return "q_served" if kind == "gguf" else "q_inproc"
    return "idx_b32"


def compute_noise_floor(
    rows: Sequence[dict[str, Any]], aa_arm: str | None = None
) -> dict[str, Any]:
    """The A/A spread, in each metric's own unit, from the A/A arm at its reference legs."""
    if aa_arm is None:
        aa_arm = next(
            (r["arm_id"] for r in rows if str(r.get("phase", "")).startswith("aa_")), None
        )
    out: dict[str, Any] = {
        "schema": NOISE_SCHEMA,
        "aa_arm": aa_arm,
        "definition": (
            "A/A spread = max-min of the SAME arm's ok measurements of the same leg at different times "
            "in the run (early block, the two ABA visits, late block), in the metric's own unit; "
            "rel = abs / median. A cross-arm difference <= max(abs, rel x mean of the two) is within_noise."
        ),
        "metrics": {},
    }
    if aa_arm is None:
        out["reason"] = "no A/A rows"
        return out
    mine = [r for r in rows if r.get("arm_id") == aa_arm]
    kind = next((r.get("arm_kind") for r in mine if r.get("arm_kind")), None)
    if kind is None:
        out["reason"] = f"no rows for A/A arm {aa_arm}"
        return out
    out["aa_arm_identity_digest"] = next(
        (r.get("arm_identity_digest") for r in mine if r.get("arm_identity_digest")), None
    )
    for metric, (unit, family) in NOISE_METRICS.items():
        leg = _aa_ref_leg(kind, family)
        field = _ROW_FIELD[metric]
        samples = []
        for r in mine:
            if r.get("leg") != leg or r.get("status") != "ok" or r.get("leg_kind") == "rss":
                continue
            v = r.get(field)
            if v is None or not isinstance(v, (int, float)):
                continue
            slot = r["phase"] if str(r["phase"]).startswith("aa_") else f"aba_p{r.get('rep')}"
            samples.append({"slot": slot, "t": r.get("t_start"), "value": float(v)})
        samples.sort(key=lambda s: (s["t"] is None, s["t"]))
        sp = spread_of([s["value"] for s in samples])
        out["metrics"][metric] = {
            "unit": unit,
            "leg": leg,
            "samples": samples,
            "n": sp["n"],
            "abs": sp["abs"],
            "rel": sp["rel"],
            "median": sp["median"],
            **({} if sp["n"] >= 2 else {"reason": "fewer than 2 ok A/A samples"}),
        }
    # rep-to-rep (the two ABA visits of every arm): secondary evidence of the floor's scale
    rep: dict[str, Any] = {}
    for metric in ("chunks_per_s", "query_p50_ms", "cpu_s_per_query", "index_build_s_4096"):
        field = _ROW_FIELD[metric]
        by: dict[tuple[str, str], dict[int, float]] = {}
        for r in rows:
            if r.get("phase") != "main" or r.get("status") != "ok" or r.get("rep") is None:
                continue
            v = r.get(field)
            if isinstance(v, (int, float)):
                by.setdefault((r["arm_id"], r["leg"]), {})[int(r["rep"])] = float(v)
        rels = []
        for d in by.values():
            if 0 in d and 1 in d and (d[0] + d[1]):
                rels.append(abs(d[1] - d[0]) / ((d[0] + d[1]) / 2))
        rep[metric] = {
            "n_pairs": len(rels),
            "median_rel": statistics.median(rels) if rels else None,
            "max_rel": max(rels) if rels else None,
        }
    out["rep_to_rep"] = rep
    return out


def is_within_noise(
    a: float | None, b: float | None, floor: dict[str, Any] | None, *, leg: str | None = None
) -> bool | None:
    """True if |a-b| <= the noise floor; None if there is no floor to judge by (never guess)."""
    if a is None or b is None or not floor or floor.get("abs") is None:
        return None
    mean = (abs(a) + abs(b)) / 2
    thr = (floor["rel"] or 0.0) * mean
    if leg is None or leg == floor.get("leg"):
        thr = max(thr, floor["abs"])
    return abs(a - b) <= thr


# ───────────────────────────────────────────────────────────────────────────── aggregation


def _median(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm aggregates from the MAIN-phase (and static) rows. A/A rows feed only the noise floor;
    contended/error rows are counted and excluded from every median."""
    arms_out: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    for r in rows:
        if r.get("phase") not in ("main", "static"):
            continue
        arm = r["arm_id"]
        a = arms_out.setdefault(
            arm,
            {
                "arm_id": arm,
                "kind": r.get("arm_kind"),
                "identity": r.get("arm_identity"),
                "identity_digest": r.get("arm_identity_digest"),
                "recall_arm_ids": r.get("recall_arm_ids") or [],
                "shares_forward_pass": r.get("shares_forward_pass") or [],
                "throughput": {},
                "query": {},
                "index_build": None,
                "rss": None,
                "disk": None,
                "status_counts": Counter(),
            },
        )
        st = r.get("status", "?")
        a["status_counts"][st] += 1
        counts[st] += 1
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("phase") in ("main", "static"):
            by_group.setdefault((r["arm_id"], r.get("leg_kind", ""), r.get("leg", "")), []).append(
                r
            )
    for (arm, lk, leg), grp in by_group.items():
        a = arms_out[arm]
        ok = [r for r in grp if r.get("status") == "ok"]
        nc = sum(1 for r in grp if r.get("status") == "contended")
        ne = sum(1 for r in grp if r.get("status") == "error")
        if lk == "throughput":
            cps = [r["chunks_per_s"] for r in ok if r.get("chunks_per_s") is not None]
            tps = [r["tokens_per_s"] for r in ok if r.get("tokens_per_s") is not None]
            cfg = (ok[0] if ok else grp[0]).get("config") or {}
            a["throughput"][leg] = {
                "leg": leg,
                "batch": cfg.get("batch"),
                "threads": cfg.get("threads"),
                "threads_kind": cfg.get("threads_kind"),
                "server_threads": cfg.get("server_threads"),
                "n_chunks": (ok[0].get("n_chunks") if ok else None),
                "chunks_per_s": _median(cps),
                "chunks_per_s_reps": cps,
                "tokens_per_s": _median(tps),
                "tokens_per_s_reps": tps,
                "rep_spread": spread_of(cps) if cps else None,
                "n_ok": len(ok),
                "n_contended": nc,
                "n_error": ne,
            }
        elif lk == "query":
            if grp and all(r.get("status") == "not_applicable" for r in grp):
                a["query"][leg] = {"leg": leg, "status": "not_applicable"}
                continue
            lat = [x for r in ok for x in (r.get("latencies_ms") or [])]
            n_q = sum(int(r.get("n_queries") or 0) for r in ok)
            cpu = sum(float(r.get("cpu_s_window") or 0.0) for r in ok)
            stats = latency_stats(lat)
            a["query"][leg] = {
                "leg": leg,
                "status": "ok" if ok else "no_ok_rows",
                "n_samples": stats["n"],
                "p50_ms": stats["p50_ms"],
                "p95_ms": stats["p95_ms"],
                "max_ms": stats["max_ms"],
                "per_visit": [
                    {
                        "p50_ms": r.get("p50_ms"),
                        "p95_ms": r.get("p95_ms"),
                        "max_ms": r.get("max_ms"),
                        "cpu_s_per_query": r.get("cpu_s_per_query"),
                    }
                    for r in ok
                ],
                "cpu_s_per_query": (cpu / n_q) if n_q else None,
                "n_queries_total": n_q,
                "n_distinct_queries": (
                    (ok[0].get("config") or {}).get("n_distinct_queries") if ok else None
                ),
                "repeats_per_visit": ((ok[0].get("config") or {}).get("repeats") if ok else None),
                "threads": ((ok[0].get("config") or {}).get("threads") if ok else None),
                "server_threads": (
                    (ok[0].get("config") or {}).get("server_threads") if ok else None
                ),
                "cpu_source": sorted(
                    {
                        c
                        for c in (r.get("cpu", {}).get("cpu_source") for r in ok if r.get("cpu"))
                        if c
                    }
                ),
                "n_ok": len(ok),
                "n_contended": nc,
                "n_error": ne,
            }
        elif lk == "index_build":
            imp = [
                r["implied_build_s_4096"] for r in ok if r.get("implied_build_s_4096") is not None
            ]
            cps = [r["chunks_per_s"] for r in ok if r.get("chunks_per_s") is not None]
            bud = next((r["budget"] for r in ok if r.get("budget")), None)
            med = _median(imp)
            icfg = (ok[0] if ok else grp[0]).get("config") or {}
            a["index_build"] = {
                "leg": leg,
                "batch": icfg.get("batch"),
                "threads": icfg.get("threads"),
                "threads_kind": icfg.get("threads_kind"),
                "implied_build_s_4096": med,
                "implied_build_s_reps": imp,
                "build_wall_s_reps": [r.get("build_wall_s") for r in ok],
                "chunks_per_s": _median(cps),
                "n_index_chunks": (ok[0].get("n_index_chunks") if ok else None),
                "finite": all(r.get("finite", True) for r in ok) if ok else None,
                "nonfinite_rows": sum(int(r.get("nonfinite_rows") or 0) for r in ok),
                "dim": (ok[0].get("dim") if ok else None),
                "budget_s": bud["budget_s"] if bud else None,
                "budget_formula": bud["budget_formula"] if bud else None,
                "exposes_index_timeout": (med > bud["budget_s"])
                if (bud and med is not None)
                else None,
                "searches_to_complete": (
                    math.ceil(med / bud["budget_s"]) if (bud and med) else None
                ),
                "measured_over_assumed": (
                    (_median(cps) / bud["assumed_texts_per_s"]) if (bud and cps) else None
                ),
                "assumed_texts_per_s": bud["assumed_texts_per_s"] if bud else None,
                "n_ok": len(ok),
                "n_contended": nc,
                "n_error": ne,
            }
        elif lk in ("rss", "rss_end"):
            cur = a["rss"] or {
                "rss_mb": None,
                "peak_hwm_mb": None,
                "source": None,
                "_after": [],
                "_peaks": [],
                "_delta": [],
                "_load": [],
            }
            for r0 in ok:
                if lk == "rss":
                    if r0.get("rss_after_load_kb") is not None:
                        cur["_after"].append(r0["rss_after_load_kb"])
                    base_kb = r0.get("rss_baseline_kb")  # process RSS before the model, imports in
                    if isinstance(base_kb, (int, float)) and r0.get("rss_after_load_kb"):
                        cur["_delta"].append(r0["rss_after_load_kb"] - base_kb)
                    if r0.get("load_s") is not None:
                        cur["_load"].append(r0["load_s"])
                    cur["source"] = r0.get("rss_source")
                for k in ("hwm_kb", "hwm_peak_kb", "hwm_after_load_kb"):
                    if isinstance(r0.get(k), (int, float)):
                        cur["_peaks"].append(r0[k])
            a["rss"] = cur
        elif lk == "disk":
            if ok:
                d = ok[0].get("disk") or {}
                a["disk"] = {
                    "disk_mb": (d.get("bytes", 0) / MB) if d else None,
                    "repeat_spread_mb": (
                        abs(d["repeat_bytes"] - d["bytes"]) / MB
                        if d.get("repeat_bytes") is not None
                        else None
                    ),
                    "weights_mb": (d.get("weights_bytes", 0) / MB) if d else None,
                    "files": d.get("files"),
                    "path": d.get("path"),
                    "sha256": d.get("sha256"),
                    "identity_warning": d.get("identity_warning"),
                }
    # identity as RESOLVED by a measuring (main, ok) row; static rows can predate open()
    for r in rows:
        a = arms_out.get(r.get("arm_id"))
        if a is None or r.get("status") != "ok":
            continue
        if r.get("phase") == "main" and not a.get("_resolved"):
            a["identity"], a["identity_digest"], a["_resolved"] = (
                r.get("arm_identity"),
                r.get("arm_identity_digest"),
                True,
            )
        if r.get("phase") == "main" and r.get("server_rss_kb") and a["rss"] is not None:
            a["rss"]["_after"].extend(r["server_rss_kb"])  # a served arm's RSS across the run
    for a in arms_out.values():
        a.pop("_resolved", None)
        a["status_counts"] = dict(a["status_counts"])
        rs = a["rss"]
        if rs is not None:
            after, peaks = rs.pop("_after"), rs.pop("_peaks")
            delta, load = rs.pop("_delta"), rs.pop("_load")
            vals = [_kb_to_mb(x) for x in after]
            rs.update(
                rss_mb=_median(vals),
                rss_mb_values=vals,
                rss_spread_mb=(max(vals) - min(vals)) if len(vals) >= 2 else None,
                n_rss=len(vals),
                peak_hwm_mb=_kb_to_mb(max(peaks)) if peaks else None,
                rss_delta_mb=_kb_to_mb(_median(delta)),
                load_s=_median(load),
            )
    return {"arms": arms_out, "status_counts": dict(counts)}


def _kb_to_mb(kb: Any) -> float | None:
    """/proc reports kB = KiB; the capture unit is decimal MB (1e6 bytes)."""
    return (float(kb) * 1024.0 / MB) if isinstance(kb, (int, float)) else None


def selection_costs(agg: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """``{arm_id: {cpu_s_per_query, p95_ms}}`` for the 2.1 selection rule, and which leg each came
    from: the AS-SERVED leg for GGUF arms, the IN-PROCESS leg for ST/static arms."""
    costs: dict[str, dict[str, Any]] = {}
    legs: dict[str, str] = {}
    for arm, a in agg["arms"].items():
        leg = "q_served" if a["kind"] == "gguf" else "q_inproc"
        q = a["query"].get(leg) or {}
        ok = q.get("status") == "ok"
        costs[arm] = {
            "cpu_s_per_query": q.get("cpu_s_per_query") if ok else None,
            "p95_ms": q.get("p95_ms") if ok else None,
        }
        legs[arm] = "as_served" if a["kind"] == "gguf" else "in_process"
    return costs, legs


def build_summary(
    rows: Sequence[dict[str, Any]],
    noise: dict[str, Any] | None = None,
    baseline_arm: str | None = None,
) -> dict[str, Any]:
    noise = noise if noise is not None else compute_noise_floor(rows)
    agg = aggregate(rows)
    arms_ = agg["arms"]
    order = [a for a in ROSTER_ORDER if a in arms_] + [a for a in arms_ if a not in ROSTER_ORDER]
    base = (
        baseline_arm
        if baseline_arm in arms_
        else (
            "bge-large-gguf-f16" if "bge-large-gguf-f16" in arms_ else (order[0] if order else None)
        )
    )
    nm = noise.get("metrics", {})

    def floor(metric: str) -> dict[str, Any] | None:
        m = nm.get(metric)
        return m if m and m.get("abs") is not None else None

    vs: dict[str, list[dict[str, Any]]] = {}
    if base:
        b = arms_[base]
        for arm in order:
            if arm == base:
                continue
            a, items = arms_[arm], []

            def add(metric: str, label: str, noise_leg: str | None, av: Any, bv: Any) -> None:
                # noise_leg: the throughput leg the abs floor may be applied to; None for metrics
                # whose unit is comparable across legs (ms, cpu-s, s), where the abs floor applies.
                if av is None or bv is None:
                    return
                items.append(
                    {
                        "metric": metric,
                        "leg": label,
                        "value": av,
                        "baseline_value": bv,
                        "ratio": (av / bv) if bv else None,
                        "diff": av - bv,
                        "within_noise": is_within_noise(av, bv, floor(metric), leg=noise_leg),
                    }
                )

            for leg, t in a["throughput"].items():
                bt = b["throughput"].get(leg)  # same leg id: same batch and threads config
                if (
                    bt is None
                ):  # else the baseline's pool-like leg at the same batch (threads 4 / server)
                    bt = next(
                        (
                            x
                            for x in b["throughput"].values()
                            if x["batch"] == t["batch"]
                            and (x["threads"] == 4 or x["threads_kind"] == "server")
                        ),
                        None,
                    )
                if bt:
                    add("chunks_per_s", leg, leg, t["chunks_per_s"], bt["chunks_per_s"])
            sel_a = "q_served" if a["kind"] == "gguf" else "q_inproc"
            sel_b = "q_served" if b["kind"] == "gguf" else "q_inproc"
            qa, qb = a["query"].get(sel_a) or {}, b["query"].get(sel_b) or {}
            for metric, fld in (
                ("query_p50_ms", "p50_ms"),
                ("query_p95_ms", "p95_ms"),
                ("query_max_ms", "max_ms"),
                ("cpu_s_per_query", "cpu_s_per_query"),
            ):
                add(metric, sel_a, None, qa.get(fld), qb.get(fld))
            ia, ib = a["index_build"] or {}, b["index_build"] or {}
            add(
                "index_build_s_4096",
                ia.get("leg", "idx"),
                None,
                ia.get("implied_build_s_4096"),
                ib.get("implied_build_s_4096"),
            )
            vs[arm] = items
    costs, legs = selection_costs(agg)
    ranking = []
    ranked = sorted(
        (a for a in costs if costs[a]["cpu_s_per_query"] is not None),
        key=lambda a: (
            costs[a]["cpu_s_per_query"],
            costs[a]["p95_ms"] if costs[a]["p95_ms"] is not None else math.inf,
        ),
    )
    prev = None
    for arm in ranked:
        ranking.append(
            {
                "arm_id": arm,
                "leg": legs[arm],
                **costs[arm],
                "cpu_within_noise_of_prev": (
                    is_within_noise(
                        costs[arm]["cpu_s_per_query"],
                        costs[prev]["cpu_s_per_query"],
                        floor("cpu_s_per_query"),
                        leg=None,
                    )
                    if prev
                    else None
                ),
                "p95_within_noise_of_prev": (
                    is_within_noise(
                        costs[arm]["p95_ms"], costs[prev]["p95_ms"], floor("query_p95_ms"), leg=None
                    )
                    if prev
                    else None
                ),
            }
        )
        prev = arm
    aliases = {rid: arm for arm, a in arms_.items() for rid in a["recall_arm_ids"]}
    dry = any(r.get("dry_run") for r in rows)
    return {
        "schema": SUMMARY_SCHEMA,
        "dry_run": dry,
        "run_id": next((r.get("run_id") for r in rows if r.get("run_id")), None),
        "claim_tag": next((r.get("claim_tag") for r in rows if r.get("claim_tag")), None),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "row_status_counts": agg["status_counts"],
        "baseline_arm": base,
        "noise_floor": noise,
        "arms": arms_,
        "arm_order": order,
        "vs_baseline": vs,
        "selection": {
            "costs": costs,
            "legs": legs,
            "ranking_by_cpu_s_per_query": ranking,
            "rule": "2.1: cheapest by CPU-s per query, latency (p95) as tie-break; as-served leg for GGUF arms, in-process leg for ST/static arms",
        },
        "arm_aliases": aliases,
        "notes": [
            "EmbeddingGemma 2 @256 shares the @768 forward pass (slice + L2 renorm only): one leg per quant/runtime, no duplicate @256 legs; arm_aliases maps recall arm ids to speed arms.",
            "eg2-st-fp32 is speed-only here; the operator dropped FP32 from the RECALL arms only.",
            "Static (model2vec) arms run a larger chunk subset so the timed window is long enough to resolve; chunks/s is per chunk and comparable.",
            "cpu_s_per_query sums client and server CPU over the query window; ST legs include OpenMP spin if OMP_WAIT_POLICY=active.",
            "Served-arm RSS includes file-backed GGUF pages mapped by the server.",
            "Contended and error rows are excluded from every figure here and stay in speed.jsonl.",
        ],
    }


def _f(v: Any, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(summary: dict[str, Any]) -> str:
    s = summary
    L: list[str] = []
    P = L.append
    if s["dry_run"]:
        P("> **DRY RUN: fake encoders, 32-chunk corpus. NOTHING below is a measurement.**\n")
    P("# REPL-EMB-2.7 EPYC speed block\n")
    P(
        f"run `{s['run_id']}`, claim tag `{s['claim_tag']}`, generated {s['generated_utc']}. Timing only; no retrieval-quality claim."
    )
    P(
        f"Row statuses: {s['row_status_counts']}. Contended/error rows are excluded from every figure below.\n"
    )
    nf = s["noise_floor"]
    P("## Noise floor (A/A spread, in each metric's unit)\n")
    P(f"A/A arm: `{nf.get('aa_arm')}`. {nf.get('definition', '')}\n")
    P("| metric | unit | leg | n | abs | rel | median |\n|---|---|---|---|---|---|---|")
    for m, v in nf.get("metrics", {}).items():
        P(
            f"| {m} | {v['unit']} | {v['leg']} | {v['n']} | {_f(v['abs'], 4)} | {_f(None if v['rel'] is None else 100 * v['rel'], 1)}% | {_f(v['median'], 3)} |"
        )
    P(
        "\nA cross-arm difference not larger than the floor is marked `within_noise` and must not be reported as a speed difference.\n"
    )
    P("## Encode throughput (median over ok visits)\n")
    P(
        "| arm | leg | batch | threads | n chunks | chunks/s | tokens/s | reps | contended |\n|---|---|---|---|---|---|---|---|---|"
    )
    for arm in s["arm_order"]:
        for leg, t in sorted(
            s["arms"][arm]["throughput"].items(),
            key=lambda kv: (
                kv[1]["threads_kind"] or "",
                str(kv[1]["threads"]),
                kv[1]["batch"] or 0,
            ),
        ):
            thr = t["threads"] if t["threads"] is not None else f"server={t['server_threads']}"
            P(
                f"| {arm} | {leg} | {t['batch']} | {thr} | {t['n_chunks']} | {_f(t['chunks_per_s'])} | {_f(t['tokens_per_s'], 0)} | {[round(x, 2) for x in t['chunks_per_s_reps']]} | {t['n_contended']} |"
            )
    P("\n## Single-query latency and CPU per query\n")
    P(
        "| arm | leg | samples | p50 ms | p95 ms | max ms | cpu-s/query | cpu source |\n|---|---|---|---|---|---|---|---|"
    )
    for arm in s["arm_order"]:
        for leg, q in sorted(s["arms"][arm]["query"].items()):
            if q.get("status") == "not_applicable":
                P(f"| {arm} | {leg} | - | not_applicable | | | | |")
                continue
            P(
                f"| {arm} | {leg} | {q['n_samples']} | {_f(q['p50_ms'])} | {_f(q['p95_ms'])} | {_f(q['max_ms'])} | {_f(q['cpu_s_per_query'], 4)} | {','.join(q['cpu_source'])} |"
            )
    P(
        "\n## Index build, 4,096 chunks (batch 32, threads 4 / static single / server) vs the shipped budget\n"
    )
    P(
        "| arm | implied build s | chunks/s | budget s | exposes index_timeout | searches to complete | measured / assumed 25 | finite | dim |\n|---|---|---|---|---|---|---|---|---|"
    )
    for arm in s["arm_order"]:
        ib = s["arms"][arm]["index_build"]
        if ib:
            P(
                f"| {arm} | {_f(ib['implied_build_s_4096'], 1)} | {_f(ib['chunks_per_s'])} | {_f(ib['budget_s'], 1)} | {ib['exposes_index_timeout']} | {ib['searches_to_complete']} | {_f(ib['measured_over_assumed'], 2)}x | {ib['finite']} | {ib['dim']} |"
            )
    P("\n## RSS and on-disk size\n")
    P(
        "| arm | RSS after load MB | model delta MB | peak VmHWM MB | rss source | disk MB | weights MB |\n|---|---|---|---|---|---|---|"
    )
    for arm in s["arm_order"]:
        a = s["arms"][arm]
        r, d = a["rss"] or {}, a["disk"] or {}
        P(
            f"| {arm} | {_f(r.get('rss_mb'), 1)} | {_f(r.get('rss_delta_mb'), 1)} | {_f(r.get('peak_hwm_mb'), 1)} | {r.get('source')} | {_f(d.get('disk_mb'), 1)} | {_f(d.get('weights_mb'), 1)} |"
        )
    P(f"\n## Versus baseline `{s['baseline_arm']}`\n")
    P(
        "| arm | metric | leg | value | baseline | ratio | within_noise |\n|---|---|---|---|---|---|---|"
    )
    for arm, items in s["vs_baseline"].items():
        for it in items:
            wn = {True: "**within_noise**", False: "no", None: "n/a (no floor)"}[it["within_noise"]]
            P(
                f"| {arm} | {it['metric']} | {it['leg']} | {_f(it['value'], 3)} | {_f(it['baseline_value'], 3)} | {_f(it['ratio'], 2)} | {wn} |"
            )
    P("\n## Selection costs for the 2.1 rule\n")
    P(f"{s['selection']['rule']}.\n")
    P(
        "| arm | leg used | cpu-s/query | p95 ms | cpu vs prev within_noise | p95 vs prev within_noise |\n|---|---|---|---|---|---|"
    )
    for it in s["selection"]["ranking_by_cpu_s_per_query"]:
        P(
            f"| {it['arm_id']} | {it['leg']} | {_f(it['cpu_s_per_query'], 4)} | {_f(it['p95_ms'])} | {it['cpu_within_noise_of_prev']} | {it['p95_within_noise_of_prev']} |"
        )
    P("\n## Recall-arm aliases and notes\n")
    P(", ".join(f"`{k}` -> `{v}`" for k, v in sorted(s["arm_aliases"].items())) or "-")
    P("")
    for n in s["notes"]:
        P(f"- {n}")
    return "\n".join(L) + "\n"


def summarize_dir(out_dir: Path, baseline_arm: str | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    rows = load_rows(out_dir / "speed.jsonl")
    if not rows:
        raise Refusal(f"{out_dir}/speed.jsonl has no rows")
    aa = None
    mp = out_dir / "run_manifest.json"
    if mp.exists():
        aa = json.loads(mp.read_text()).get("aa_arm")
    noise = compute_noise_floor(rows, aa)
    summary = build_summary(rows, noise, baseline_arm)
    (out_dir / "noise_floor.json").write_text(dumps(noise, indent=1) + "\n")
    (out_dir / "speed_summary.json").write_text(dumps(summary, indent=1) + "\n")
    (out_dir / "speed_summary.md").write_text(render_markdown(summary))
    (out_dir / "selection_costs.json").write_text(
        dumps(summary["selection"]["costs"], indent=1) + "\n"
    )
    return summary


# ───────────────────────────────────────────────────────────────────────────── capture adapter
# Shaped for ``repl_retrieval_capture.SpeedRun`` (read, not imported): ``add_speed`` takes exactly
# CAPTURE_KEYS, wants a positive value, an int n >= 1, a spread that is a number >= 0 (never None),
# an arm identity with exactly ARM_FIELDS, and its metric in the capture module's own unit
# ("MB", decimal); ``finalize`` then demands an A/A floor per (arm, metric), which
# ``noise_floors_for_capture`` supplies for ``add_noise_floor``.

CAPTURE_KEYS = (
    "arm_identity",
    "metric",
    "value",
    "unit",
    "n",
    "spread",
    "spread_unit",
    "config_echo",
)
FLOOR_KEYS = ("arm_identity", "metric", "aa_spread", "unit", "replication_unit", "n", "config_echo")
ARM_FIELDS = (
    "arm_id",
    "model",
    "revision",
    "dim",
    "dtype_or_quant",
    "runtime",
    "prompt_hash",
    "index_identity_digest",
    "role",
)
CAPTURE_UNITS = {
    "chunks_per_s": "chunks/s",
    "tokens_per_s": "tokens/s",
    "query_p50_ms": "ms",
    "query_p95_ms": "ms",
    "query_max_ms": "ms",
    "cpu_s_per_query": "cpu-s/query",
    "index_build_s_4096": "s",
    "rss_mb": "MB",
    "disk_mb": "MB",
}
# ClaimTuple category of each arm: the incumbent and its same-runtime control are the baseline.
BASELINE_ARMS = frozenset({"bge-large-gguf-f16", "bge-large-st-fp32"})


def _capture_identity(ident: dict[str, Any], arm_id: str, role: str) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "model": ident["model"],
        "revision": ident["revision"],
        "dim": int(ident["dim"]),
        "dtype_or_quant": ident["dtype"],
        "runtime": ident["runtime"],
        "prompt_hash": ident["prompt_hash"],
        "index_identity_digest": ident["digest"],
        "role": role,
    }


def _capture_arms(
    a: dict[str, Any], roles: dict[str, str], aliases: bool
) -> list[tuple[dict[str, Any], str | None]]:
    """``[(arm_identity, alias_of)]``: the measured arm first, then (if ``aliases``) the @256 arms
    that share its forward pass, reported under their own manifest ids and flagged."""
    role = roles.get(a["arm_id"]) or ("BASELINE" if a["arm_id"] in BASELINE_ARMS else "CANDIDATE")
    ids = a.get("recall_arm_ids") or []
    measured_id = ids[0] if ids else a["arm_id"]
    out = [(_capture_identity(a["identity"], measured_id, role), None)]
    if aliases:
        for sh in a.get("shares_forward_pass") or []:
            if sh.get("identity"):
                out.append((_capture_identity(sh["identity"], sh["arm_id"], role), measured_id))
    return out


def _range(vals: Sequence[Any]) -> float | None:
    xs = [float(v) for v in vals if isinstance(v, (int, float))]
    return (max(xs) - min(xs)) if len(xs) >= 2 else None


def _capture_build(
    speed_jsonl: Path | str | Sequence[dict[str, Any]],
    noise_floor: dict[str, Any] | Path | str | None,
    *,
    allow_dry_run: bool,
    roles: dict[str, str] | None,
    include_aliases: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(speed_jsonl) if isinstance(speed_jsonl, (list, tuple)) else load_rows(speed_jsonl)
    if isinstance(noise_floor, (str, Path)):
        noise_floor = json.loads(Path(noise_floor).read_text())
    nm = (noise_floor or {}).get("metrics", {})
    aa_arm = (noise_floor or {}).get("aa_arm")
    if any(r.get("dry_run") for r in rows) and not allow_dry_run:
        raise ValueError(
            "speed rows come from a --dry-run (fake encoders); refusing to emit capture rows"
        )
    agg = aggregate(rows)["arms"]
    claim_cpus = next((r.get("cpu_list_claimed") for r in rows if r.get("cpu_list_claimed")), None)
    claim_tag = next((r.get("claim_tag") for r in rows if r.get("claim_tag")), None)
    cand: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    floors: dict[tuple[str, str], dict[str, Any]] = {}
    roles = roles or {}

    for arm, a in agg.items():
        targets = _capture_arms(a, roles, include_aliases)

        def emit(
            metric: str,
            value: Any,
            n: Any,
            spread: Any,
            spread_kind: str,
            echo: dict[str, Any],
            own_floor: dict[str, Any] | None = None,
            _a: dict[str, Any] = a,
            _arm: str = arm,
            _targets: list[tuple[dict[str, Any], str | None]] = targets,
        ) -> None:
            where = echo.get("leg") or echo.get("kind")
            why = None
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                why = f"value {value!r} is not a positive finite number"
            elif not (isinstance(n, int) and n >= 1):
                why = f"n {n!r} < 1"
            elif spread is None:
                why = "fewer than 2 ok repeats: no spread (the writer refuses a missing spread)"
            if why:
                gaps.append({"arm_id": _arm, "metric": metric, "config": where, "reason": why})
                return
            if own_floor is None:  # the run-level A/A floor, scaled to this arm through its rel
                fl = nm.get(metric) or {}
                if fl.get("abs") is None or (fl.get("n") or 0) < 2:
                    own_floor = None
                else:
                    scaled = (fl["rel"] or 0.0) * float(value) if fl.get("rel") else None
                    if _arm == aa_arm and echo.get("leg") == fl.get("leg"):
                        scaled = max(float(fl["abs"]), scaled or 0.0)  # the A/A arm: its own floor
                    own_floor = (
                        {"aa_spread": scaled, "replication_unit": "arm", "n": int(fl["n"])}
                        if scaled is not None
                        else None
                    )
            for ident, alias_of in _targets:
                cfg = {
                    **echo,
                    "speed_arm_id": _arm,
                    "spread_kind": spread_kind,
                    "alias_of": alias_of,
                    "recall_arm_ids": _a["recall_arm_ids"],
                    "claim_cpu_list": claim_cpus,
                    "claim_tag": claim_tag,
                }
                cand.append(
                    {
                        "_key": (ident["arm_id"], metric),
                        "_floor": own_floor,
                        "row": {
                            "arm_identity": ident,
                            "metric": metric,
                            "value": float(value),
                            "unit": CAPTURE_UNITS[metric],
                            "n": int(n),
                            "spread": float(spread),
                            "spread_unit": CAPTURE_UNITS[metric],
                            "config_echo": _clean(cfg),
                        },
                    }
                )

        for leg, t in a["throughput"].items():
            echo = {
                "leg": leg,
                "batch": t["batch"],
                "threads": t["threads"],
                "threads_kind": t["threads_kind"],
                "server_threads": t["server_threads"],
                "n_chunks": t["n_chunks"],
            }
            emit(
                "chunks_per_s",
                t["chunks_per_s"],
                t["n_ok"],
                _range(t["chunks_per_s_reps"]),
                "rep_range",
                echo,
            )
            emit(
                "tokens_per_s",
                t["tokens_per_s"],
                t["n_ok"],
                _range(t["tokens_per_s_reps"]),
                "rep_range",
                echo,
            )
        for leg, q in a["query"].items():
            if q.get("status") != "ok":
                continue
            echo = {
                "leg": leg,
                "served": "as_served" if leg == "q_served" else "in_process",
                "threads": q["threads"],
                "server_threads": q["server_threads"],
                "n_distinct_queries": q["n_distinct_queries"],
                "repeats_per_visit": q["repeats_per_visit"],
                "cpu_source": q["cpu_source"],
            }
            pv = q["per_visit"]
            emit(
                "query_p50_ms",
                q["p50_ms"],
                q["n_samples"],
                _range([v["p50_ms"] for v in pv]),
                "visit_range",
                echo,
            )
            emit(
                "query_p95_ms",
                q["p95_ms"],
                q["n_samples"],
                _range([v["p95_ms"] for v in pv]),
                "visit_range",
                echo,
            )
            emit(
                "query_max_ms",
                q["max_ms"],
                q["n_samples"],
                _range([v["max_ms"] for v in pv]),
                "visit_range",
                echo,
            )
            emit(
                "cpu_s_per_query",
                q["cpu_s_per_query"],
                q["n_queries_total"],
                _range([v["cpu_s_per_query"] for v in pv]),
                "visit_range",
                echo,
            )
        ib = a["index_build"]
        if ib and ib["implied_build_s_4096"] is not None:
            echo = {
                "leg": ib["leg"],
                "batch": ib["batch"],
                "threads": ib["threads"],
                "n_index_chunks": ib["n_index_chunks"],
                "budget_s": ib["budget_s"],
                "budget_formula": ib["budget_formula"],
                "exposes_index_timeout": ib["exposes_index_timeout"],
                "searches_to_complete": ib["searches_to_complete"],
                "measured_over_assumed": ib["measured_over_assumed"],
                "chunks_per_s": ib["chunks_per_s"],
                "finite": ib["finite"],
            }
            emit(
                "index_build_s_4096",
                ib["implied_build_s_4096"],
                ib["n_ok"],
                _range(ib["implied_build_s_reps"]),
                "rep_range",
                echo,
            )
        r, d = a["rss"] or {}, a["disk"] or {}
        if r.get("rss_mb") is not None:
            # the A/A for RSS is the arm's own repeats: two fresh processes (in-process arms) or
            # the same server sampled across the run (served arms)
            f_rss = (
                {
                    "aa_spread": r["rss_spread_mb"],
                    "replication_unit": "arm" if a["kind"] == "gguf" else "process",
                    "n": r["n_rss"],
                }
                if r.get("rss_spread_mb") is not None and r["n_rss"] >= 2
                else None
            )
            emit("rss_mb", r["rss_mb"], r["n_rss"], r.get("rss_spread_mb"), "rss_sample_range",
                 {"kind": "after_load", "source": r.get("source"), "rss_delta_mb": r.get("rss_delta_mb"), "load_s": r.get("load_s")},
                 own_floor=f_rss)  # fmt: skip
            if r.get("peak_hwm_mb") is not None:
                emit("rss_mb", r["peak_hwm_mb"], max(1, r["n_rss"]), 0.0, "peak_counter",
                     {"kind": "peak_hwm", "source": r.get("source")}, own_floor=f_rss)  # fmt: skip
        if d.get("disk_mb") is not None:
            f_disk = (
                {"aa_spread": d["repeat_spread_mb"], "replication_unit": "arm", "n": 2}
                if d.get("repeat_spread_mb") is not None
                else None
            )
            emit("disk_mb", d["disk_mb"], 1, 0.0, "exact_file_size",
                 {"kind": "model_artifact", "weights_mb": d.get("weights_mb"), "files": d.get("files"), "path": d.get("path")},
                 own_floor=f_disk)  # fmt: skip

    # a speed number is never emitted without its floor (the writer's finalize() would refuse it)
    for c in cand:
        fl = c["_floor"]
        if fl is None:
            gaps.append(
                {
                    "arm_id": c["row"]["arm_identity"]["arm_id"],
                    "metric": c["_key"][1],
                    "config": c["row"]["config_echo"].get("leg")
                    or c["row"]["config_echo"].get("kind"),
                    "reason": "no A/A noise floor for this metric (fewer than 2 ok A/A samples)",
                }
            )
            continue
        cur = floors.get(c["_key"])
        if cur is None or fl["aa_spread"] > cur["aa_spread"]:
            floors[c["_key"]] = {
                "arm_identity": c["row"]["arm_identity"],
                "metric": c["_key"][1],
                "aa_spread": float(fl["aa_spread"]),
                "unit": CAPTURE_UNITS[c["_key"][1]],
                "replication_unit": fl["replication_unit"],
                "n": int(fl["n"]),
                "config_echo": None,
            }
    out = [c["row"] for c in cand if c["_floor"] is not None]
    return out, list(floors.values()), gaps


def rows_for_capture(
    speed_jsonl: Path | str | Sequence[dict[str, Any]],
    noise_floor: dict[str, Any] | Path | str | None,
    *,
    allow_dry_run: bool = False,
    roles: dict[str, str] | None = None,
    include_aliases: bool = True,
) -> list[dict[str, Any]]:
    """Dicts with exactly ``SpeedRun.add_speed``'s eight arguments, one per (arm, metric, config).

    Values are medians over ok (non-contended) visits; ``spread`` is the dispersion of the
    repeats in the metric's unit (range of the per-visit values; ``config_echo['spread_kind']``
    says which), NOT the A/A floor, which ``noise_floors_for_capture`` supplies for
    ``add_noise_floor``. A row the writer would refuse (non-positive value, fewer than 2 repeats so
    no spread) is omitted and listed by ``capture_gaps``, never defaulted. Dry-run rows are refused.

    EmbeddingGemma 2 rows are emitted under the frozen manifest's ``-768`` arm id; with
    ``include_aliases`` the ``-256`` arm that shares the forward pass gets the same measured numbers
    under its own identity, flagged ``config_echo['alias_of']`` (one measurement, two index identities)."""
    return _capture_build(
        speed_jsonl,
        noise_floor,
        allow_dry_run=allow_dry_run,
        roles=roles,
        include_aliases=include_aliases,
    )[0]


def noise_floors_for_capture(
    speed_jsonl: Path | str | Sequence[dict[str, Any]],
    noise_floor: dict[str, Any] | Path | str | None,
    *,
    allow_dry_run: bool = False,
    roles: dict[str, str] | None = None,
    include_aliases: bool = True,
) -> list[dict[str, Any]]:
    """Dicts with ``SpeedRun.add_noise_floor``'s arguments (``config_echo`` None: bound to every leg
    of the arm and metric), one per (arm, metric) that has a speed row, so ``finalize()`` can pass.

    ``aa_spread`` is the run's A/A spread in the metric's unit: absolute on the A/A arm, the
    relative A/A spread times the arm's largest value elsewhere (an arm's scale is not the A/A
    arm's). ``replication_unit`` is ``arm``: the same arm re-measured at other times in one run."""
    return _capture_build(
        speed_jsonl,
        noise_floor,
        allow_dry_run=allow_dry_run,
        roles=roles,
        include_aliases=include_aliases,
    )[1]


def capture_gaps(
    speed_jsonl: Path | str | Sequence[dict[str, Any]],
    noise_floor: dict[str, Any] | Path | str | None,
    *,
    allow_dry_run: bool = False,
) -> list[dict[str, Any]]:
    """What ``rows_for_capture`` / ``noise_floors_for_capture`` left out, and why."""
    return _capture_build(
        speed_jsonl, noise_floor, allow_dry_run=allow_dry_run, roles=None, include_aliases=False
    )[2]


# ───────────────────────────────────────────────────────────────────────────── inputs / CLI


def load_corpus(path: Path) -> list[str]:
    out: list[str] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out.append(o["text"] if isinstance(o, dict) else str(o))
    return out


def load_queries(path: Path) -> list[str]:
    raw = json.loads(Path(path).read_text())
    return [x["query"] if isinstance(x, dict) else str(x) for x in raw]


def shipped_facts(cs: Any) -> dict[str, Any]:
    p = Path(cs.__file__)
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "constants": {
            "INDEX_ASSUMED_TEXTS_PER_S": cs.INDEX_ASSUMED_TEXTS_PER_S,
            "INDEX_TIMEOUT_BASE_S": cs.INDEX_TIMEOUT_BASE_S,
            "INDEX_TIMEOUT_MIN_S": cs.INDEX_TIMEOUT_MIN_S,
            "INDEX_TIMEOUT_MAX_S": cs.INDEX_TIMEOUT_MAX_S,
            "MAX_DENSE_CHUNKS": cs.MAX_DENSE_CHUNKS,
        },
        "budget_s_at_4096": cs.index_build_timeout_s(4096),
        "assumed_build_s_4096": 4096 / cs.INDEX_ASSUMED_TEXTS_PER_S,
        "bge_query_prefix": cs.query_text("bge-large-en-v1.5", ""),
    }


def cmd_prepare(args: argparse.Namespace) -> int:
    out = Path(args.out)
    inp = out / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(Path(args.corpus))
    queries = list(dict.fromkeys(load_queries(Path(args.queries))))
    if len(corpus) < args.n_chunks:
        raise Refusal(f"corpus has {len(corpus)} chunks, need >= {args.n_chunks}")
    if len(queries) < args.min_queries:
        raise Refusal(f"{len(queries)} DISTINCT queries, need >= {args.min_queries}")
    corpus = corpus[: args.n_chunks]
    (inp / "corpus.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in corpus))
    (inp / "queries.json").write_text(json.dumps(queries, indent=0) + "\n")
    cs = arms.load_context_search()
    meta = {
        "schema": INPUTS_SCHEMA,
        "corpus": {
            "source": str(args.corpus),
            "source_sha256": sha256_file(Path(args.corpus)),
            "file": "inputs/corpus.jsonl",
            "sha256": sha256_file(inp / "corpus.jsonl"),
            "n_chunks": len(corpus),
            "mean_chars": sum(len(t) for t in corpus) / len(corpus),
        },
        "queries": {
            "source": str(args.queries),
            "source_sha256": sha256_file(Path(args.queries)),
            "file": "inputs/queries.json",
            "sha256": sha256_file(inp / "queries.json"),
            "n": len(queries),
            "n_distinct": len(set(queries)),
        },
        "subset_rule": "stride_subset(corpus, n): items[int(i*N/n)] for i in range(n)",
        "subsets": {
            str(n): sha256_text("\n".join(stride_subset(corpus, n))) for n in (64, 256, 4096)
        },
        "shipped_context_search": shipped_facts(cs),
        "roster": list(ROSTER_ORDER),
    }
    (out / "speed_inputs.json").write_text(dumps(meta, indent=1) + "\n")
    print(
        f"prepared {out}/speed_inputs.json  corpus={meta['corpus']['sha256'][:12]}  queries={meta['queries']['sha256'][:12]}"
    )
    return 0


def load_inputs(out: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    meta_p = out / "speed_inputs.json"
    if not meta_p.exists():
        raise Refusal(f"{meta_p} missing: run `prepare` first")
    meta = json.loads(meta_p.read_text())
    for key in ("corpus", "queries"):
        f = out / meta[key]["file"]
        if sha256_file(f) != meta[key]["sha256"]:
            raise Refusal(f"{f} changed since `prepare` (sha256 mismatch)")
    return (
        load_corpus(out / meta["corpus"]["file"]),
        load_queries(out / meta["queries"]["file"]),
        {
            "corpus_sha256": meta["corpus"]["sha256"],
            "queries_sha256": meta["queries"]["sha256"],
            **{"speed_inputs": meta},
        },
    )


def _dry_inputs() -> tuple[list[str], list[str], dict[str, Any]]:
    corpus = [
        f"chunk {i} " + " ".join(f"tok{(i * 7 + j) % 53}" for j in range(40)) for i in range(32)
    ]
    queries = [f"dry query number {i} about topic {i % 5}" for i in range(12)]
    return (
        corpus,
        queries,
        {
            "corpus_sha256": sha256_text("\n".join(corpus)),
            "queries_sha256": sha256_text("\n".join(queries)),
            "dry_run": True,
        },
    )


def _roster_runtimes(args: argparse.Namespace, cs: Any) -> tuple[str, str, str, str | None]:
    """Runtime strings for the specs: from the frozen manifest if given (so digests equal it), else
    detected from the installed packages / the --*-runtime args."""
    if args.arm_manifest:
        m = json.loads(Path(args.arm_manifest).read_text())
        by = {a["arm_id"]: a for a in m["arms"]}
        return (
            by["eg2-st-fp32-768"]["runtime"],
            by["eg2-gguf-bf16-768"]["runtime"],
            by["potion-code-16m-v2"]["runtime"],
            m.get("manifest_sha256"),
        )
    if args.dry_run:
        return "dry-run/st", "dry-run/llama", "dry-run/m2v", None
    import importlib.metadata as im

    st = args.st_runtime or (
        f"sentence-transformers=={im.version('sentence-transformers')}"
        f"/transformers=={im.version('transformers')}/torch=={im.version('torch')}"
    )
    m2v = args.m2v_runtime or f"model2vec=={im.version('model2vec')}/numpy=={np.__version__}"
    return st, args.llama_runtime or "llama-server (build unresolved)", m2v, None


def _rss_child_runner(
    args: argparse.Namespace,
    runtimes: tuple[str, str, str, str | None],
    model_paths: dict[str, str],
    corpus_file: Path,
    timeout_s: float,
) -> Callable[[RosterEntry], dict[str, Any]]:
    def run(entry: RosterEntry) -> dict[str, Any]:
        cmd = [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "_rss-child",
            "--arm-id", entry.arm_id,
            "--model-path", model_paths[entry.arm_id],
            "--corpus", str(corpus_file),
            "--st-runtime", runtimes[0],
            "--llama-runtime", runtimes[1],
            "--m2v-runtime", runtimes[2],
        ]  # fmt: skip
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        if cp.returncode != 0:
            raise RuntimeError(f"rss child rc={cp.returncode}: {(cp.stderr or '')[-300:]}")
        line = next((ln for ln in reversed(cp.stdout.splitlines()) if ln.startswith("{")), None)
        if line is None:
            raise RuntimeError("rss child printed no JSON")
        return json.loads(line)

    return run


def cmd_rss_child(args: argparse.Namespace) -> int:
    cs = arms.load_context_search()
    roster = speed_roster(cs, args.st_runtime, args.llama_runtime, args.m2v_runtime)
    entry = roster[args.arm_id]
    if entry.kind == "st":
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    elif entry.kind == "static":
        import model2vec  # noqa: F401
    corpus = load_corpus(Path(args.corpus))[:32]
    out: dict[str, Any] = {"arm_id": args.arm_id, "rss_baseline_kb": self_status_kb("VmRSS")}
    rt: ArmRuntime = (STRuntime if entry.kind == "st" else StaticRuntime)(entry, args.model_path)
    t0 = time.perf_counter()
    rt.open()
    out["load_s"] = time.perf_counter() - t0
    out["rss_after_load_kb"] = self_status_kb("VmRSS")
    out["hwm_after_load_kb"] = self_status_kb("VmHWM")
    rt.encode_docs(corpus, 32)
    rt.encode_query(corpus[0][:80])
    out["rss_after_encode_kb"] = self_status_kb("VmRSS")
    out["hwm_peak_kb"] = self_status_kb("VmHWM")
    print(json.dumps(out))
    return 0


def build_config(args: argparse.Namespace) -> RunConfig:
    ints = lambda s: tuple(int(x) for x in str(s).split(",") if x.strip())  # noqa: E731
    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) if args.only else FAMILIES
    bad = [x for x in only if x not in FAMILIES]
    if bad:
        raise Refusal(f"--only: unknown {bad}; choose from {FAMILIES}")
    try:
        cpu_list = (
            parse_cpu_list(args.cpu_list) if args.cpu_list else sorted(os.sched_getaffinity(0))
        )
    except ValueError as exc:
        raise Refusal(f"--cpu-list {args.cpu_list!r}: {exc}") from exc
    cfg = RunConfig(
        out_dir=Path(args.out),
        cpu_list=cpu_list,
        claim_tag=args.claim_tag or "",
        dry_run=args.dry_run,
        run_id=args.run_id or f"speed-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        passes=args.passes,
        batches=ints(args.batches),
        threads=ints(args.threads),
        full_socket=args.full_socket,
        subset_n=args.subset_n,
        subset_n_b1=args.subset_n_b1,
        subset_n_t1=args.subset_n_t1,
        static_subset_n=args.static_subset_n,
        query_threads=args.query_threads,
        repeats=args.repeats,
        index_batch=args.index_batch,
        index_threads=args.index_threads,
        n_index=args.n_index,
        warmup_chunks=args.warmup_chunks,
        warmup_queries=args.warmup_queries,
        aa_arm=args.aa_arm,
        only=only,
        retry_contended=not args.no_retry_contended,
        max_foreign_busy_frac=args.max_foreign_busy_frac,
        max_loadavg_excess=args.max_loadavg_excess,
        check_interval_s=args.check_interval_s,
        min_window_s=args.min_window_s,
    )
    if args.dry_run:
        cfg.max_foreign_busy_frac, cfg.max_loadavg_excess = math.inf, 0.0
        cfg.claim_tag = cfg.claim_tag or "DRY-RUN"
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    # --- SAFETY, before anything is loaded or started --------------------------------------
    if not args.dry_run:
        if not (args.claim_tag or "").strip():
            raise Refusal(
                "--claim-tag is required (the region-lock claim this run is measured under)"
            )
        if not args.require_claim_env:
            raise Refusal("--no-require-claim-env is only allowed with --dry-run")
        if not args.cpu_list:
            raise Refusal("--cpu-list is required")
    urls = {k: check_arm_url(k, v) for k, v in parse_kv(args.arm_url, "--arm-url").items()}
    pids = {k: int(v) for k, v in parse_kv(args.arm_server_pid, "--arm-server-pid").items()}
    s_threads = {
        k: int(v) for k, v in parse_kv(args.arm_server_threads, "--arm-server-threads").items()
    }
    s_flags = parse_kv(args.arm_server_flags, "--arm-server-flags")
    model_files = parse_kv(args.arm_model_file, "--arm-model-file")
    model_paths = parse_kv(args.arm_model_path, "--arm-model-path")
    rt_over = parse_kv(args.arm_runtime, "--arm-runtime")
    selected = (
        [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else list(ROSTER_ORDER)
    )
    unknown = [a for a in selected if a not in ROSTER_ORDER]
    if unknown:
        raise Refusal(f"unknown arm(s) {unknown}; roster is {list(ROSTER_ORDER)}")
    selected = [a for a in ROSTER_ORDER if a in selected]
    if not args.dry_run:
        missing = []
        for a in selected:
            if ARM_KINDS[a] == "gguf" and (a not in urls or a not in pids):
                missing.append(f"{a}: needs --arm-url and --arm-server-pid")
            if ARM_KINDS[a] != "gguf" and a not in model_paths:
                missing.append(f"{a}: needs --arm-model-path")
        if missing:
            raise Refusal("missing caller inputs:\n  " + "\n  ".join(missing))
    cs = arms.load_context_search()
    st_rt, llama_rt, m2v_rt, manifest_sha = _roster_runtimes(args, cs)
    roster = speed_roster(cs, st_rt, llama_rt, m2v_rt, runtime_overrides=rt_over)
    runtimes: dict[str, ArmRuntime] = {}
    proc = Proc()
    claim_report: dict[str, Any] = {"dry_run": True, "verified": False}
    if args.dry_run:
        corpus, queries, inputs_meta = _dry_inputs()
        cfg.n_index = len(corpus)
        for a in selected:
            runtimes[a] = FakeRuntime(roster[a])
    else:
        corpus, queries, inputs_meta = load_inputs(out)
        if len(set(queries)) < 200:
            raise Refusal(f"{len(set(queries))} distinct queries < 200")
        if len(corpus) < cfg.n_index:
            raise Refusal(f"corpus {len(corpus)} < n_index {cfg.n_index}")
        # OMP/tokenizer threading must not undo the per-leg pinning (see module docstring).
        os.environ.setdefault("OMP_PROC_BIND", "false")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        aff = OsAffinity()
        holders_fn = lambda: region_lock_holders(args.region_lock_bin)  # noqa: E731
        report = verify_claim_env(
            cpu_list=cfg.cpu_list,
            claim_tag=cfg.claim_tag,
            proc=proc,
            affinity_now=aff.current(),
            exempt_pids=set(pids.values()),
            server_pids={a: pids[a] for a in selected if a in pids},
            holders_fn=holders_fn,
            require_region_lock_holder=not args.no_require_region_lock_holder,
            scan_samples=args.scan_samples,
        )
        (out / "claim_report.json").write_text(dumps(report, indent=1) + "\n")
        if not report["ok"]:
            failed = [c for c in report["checks"] if c["required"] and not c["ok"]]
            raise Refusal(
                "claim-environment verification FAILED:\n  "
                + "\n  ".join(f"{c['name']}: {c['detail']}" for c in failed)
                + f"\n  offenders: {report['offenders'][:10]}"
            )
        claim_report = {**report, "verified": True}
    for a in selected:
        if a in runtimes:
            continue
        e = roster[a]
        if e.kind == "gguf":
            runtimes[a] = ServedRuntime(
                e,
                urls[a]["base_url"],
                pids[a],
                proc,
                server_threads=s_threads.get(a),
                server_flags=s_flags.get(a),
                model_file=model_files.get(a),
            )
        elif e.kind == "st":
            runtimes[a] = STRuntime(e, model_paths[a])
        else:
            runtimes[a] = StaticRuntime(e, model_paths[a])
    rss_child = None
    if not args.dry_run:
        rss_child = _rss_child_runner(
            args,
            (st_rt, llama_rt, m2v_rt, manifest_sha),
            model_paths,
            out / inputs_meta["speed_inputs"]["corpus"]["file"],
            cfg.rss_child_timeout_s,
        )
    runner = SpeedRunner(
        cfg, roster, runtimes, corpus, queries, cs=cs, proc=proc,
        affinity=DryAffinity(cfg.cpu_list) if args.dry_run else OsAffinity(),
        claim_report=claim_report, rss_child=rss_child, model_files=model_files, model_paths=model_paths,
        inputs_meta=inputs_meta,
        extra_manifest={"argv": list(sys.argv), "arm_manifest_sha256": manifest_sha, "urls": {k: v["base_url"] for k, v in urls.items()}, "server_pids": pids},
    )  # fmt: skip
    res = runner.run()
    print(json.dumps(res))
    summarize_dir(out, args.baseline_arm)
    print(
        f"wrote {out}/speed_summary.md, speed_summary.json, selection_costs.json, noise_floor.json"
    )
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    summarize_dir(Path(args.out), args.baseline_arm)
    print(
        f"wrote {args.out}/speed_summary.md, speed_summary.json, selection_costs.json, noise_floor.json"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="repl_emb_speed", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser(
        "prepare", help="freeze the corpus + queries and record the shipped budget constants"
    )
    pp.add_argument("--corpus", required=True)
    pp.add_argument("--queries", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--n-chunks", type=int, default=4096)
    pp.add_argument("--min-queries", type=int, default=200)
    pp.set_defaults(fn=cmd_prepare)

    r = sub.add_parser(
        "run", help="measure (resumable); needs a held region-lock claim unless --dry-run"
    )
    r.add_argument("--out", required=True)
    r.add_argument("--cpu-list", help="taskset-style list of the claimed cpus, e.g. 24-71")
    r.add_argument(
        "--claim-tag", help="the region-lock claim tag this run is measured under (required)"
    )
    r.add_argument("--require-claim-env", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--region-lock-bin", default=DEFAULT_REGION_LOCK_BIN)
    r.add_argument(
        "--no-require-region-lock-holder",
        action="store_true",
        help="downgrade the ancestor-holder check to a warning",
    )
    r.add_argument("--scan-samples", type=int, default=3)
    r.add_argument(
        "--dry-run",
        action="store_true",
        help="whole pipeline on fake encoders and a 32-chunk corpus",
    )
    r.add_argument("--arms", help=f"comma list; default all of {','.join(ROSTER_ORDER)}")
    r.add_argument(
        "--arm-url",
        action="append",
        metavar="ARM=URL",
        help="caller-owned llama-server (loopback, non-production port)",
    )
    r.add_argument("--arm-server-pid", action="append", metavar="ARM=PID")
    r.add_argument("--arm-server-threads", action="append", metavar="ARM=N")
    r.add_argument(
        "--arm-server-flags",
        action="append",
        metavar="ARM=STR",
        help="launch flags to echo into the manifest",
    )
    r.add_argument(
        "--arm-model-file",
        action="append",
        metavar="ARM=PATH",
        help="served GGUF file (else parsed from the server cmdline)",
    )
    r.add_argument(
        "--arm-model-path",
        action="append",
        metavar="ARM=PATH",
        help="in-process arms: HF snapshot dir",
    )
    r.add_argument(
        "--arm-runtime",
        action="append",
        metavar="ARM=STR",
        help="override the runtime string in an arm's identity",
    )
    r.add_argument(
        "--arm-manifest",
        help="frozen 2.5 arm_manifest.json: runtime strings (and thus identity digests) are taken from it",
    )
    r.add_argument("--st-runtime")
    r.add_argument("--llama-runtime")
    r.add_argument("--m2v-runtime")
    r.add_argument("--run-id")
    r.add_argument("--passes", type=int, default=2)
    r.add_argument("--batches", default="1,8,32,128")
    r.add_argument("--threads", default="1,4")
    r.add_argument(
        "--full-socket",
        action="store_true",
        help="also run threads = len(cpu-list) at batch 32,128 (ST arms)",
    )
    r.add_argument("--subset-n", type=int, default=256)
    r.add_argument("--subset-n-b1", type=int, default=64)
    r.add_argument(
        "--subset-n-t1",
        type=int,
        default=None,
        help="optional smaller subset for threads=1 legs (the pilot used 16)",
    )
    r.add_argument("--static-subset-n", type=int, default=4096)
    r.add_argument("--query-threads", type=int, default=4)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--index-batch", type=int, default=32)
    r.add_argument("--index-threads", type=int, default=4)
    r.add_argument("--n-index", type=int, default=4096)
    r.add_argument("--warmup-chunks", type=int, default=32)
    r.add_argument("--warmup-queries", type=int, default=20)
    r.add_argument("--aa-arm")
    r.add_argument("--baseline-arm")
    r.add_argument("--only", help=f"comma list of {','.join(FAMILIES)}")
    r.add_argument("--no-retry-contended", action="store_true")
    r.add_argument(
        "--max-foreign-busy-frac",
        type=float,
        default=0.10,
        help="contended if foreign busy time on the leg's cpus exceeds this fraction of their capacity",
    )
    r.add_argument(
        "--max-loadavg-excess",
        type=float,
        default=6.0,
        help="contended if the 1-min loadavg rises by more than this beyond the leg's own threads (0 disables)",
    )
    r.add_argument("--check-interval-s", type=float, default=2.0)
    r.add_argument("--min-window-s", type=float, default=1.0)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser(
        "summarize", help="write speed_summary.{json,md}, selection_costs.json, noise_floor.json"
    )
    s.add_argument("--out", required=True)
    s.add_argument("--baseline-arm")
    s.set_defaults(fn=cmd_summarize)

    c = sub.add_parser("_rss-child", help=argparse.SUPPRESS)
    c.add_argument("--arm-id", required=True)
    c.add_argument("--model-path", required=True)
    c.add_argument("--corpus", required=True)
    c.add_argument("--st-runtime", required=True)
    c.add_argument("--llama-runtime", required=True)
    c.add_argument("--m2v-runtime", required=True)
    c.set_defaults(fn=cmd_rss_child)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except Refusal as exc:
        print(f"[repl_emb_speed] REFUSING: {exc}", file=sys.stderr)
        out = getattr(args, "out", None)
        if out and args.cmd == "run":
            try:
                Path(out).mkdir(parents=True, exist_ok=True)
                (Path(out) / "refusal.json").write_text(
                    dumps({"refused": str(exc), "t": time.time()}, indent=1) + "\n"
                )
            except OSError:
                pass
        return 64


if __name__ == "__main__":
    sys.exit(main())
