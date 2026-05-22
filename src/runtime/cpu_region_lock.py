"""Cross-process per-CPU-region locking — replacement for the single
global heavy_model.lock.

Motivation: the orchestrator has multiple llama-server instances per role
(e.g. frontdoor: 1×full + 4×quarter via `stack_numa.NUMA_CONFIG`). The
"full" instance uses all of CPUs 0-47 (NUMA_NODE0); a "quarter" uses
24 cores. Multiple non-overlapping instances *should* be able to run
concurrently, but the existing `inference_lock` takes a single
exclusive flock on `heavy_model.lock`, serializing every heavy
inference globally. With multi-worker uvicorn, in-process state in
`ConcurrencyAwareBackend._select` cannot coordinate across processes,
so the global file lock was the only safe primitive.

This module adds per-atomic-region file locks. A given (role,
instance_idx) maps to a set of atomic regions via
`src.runtime.instance_topology`. To dispatch through an instance, the
caller acquires LOCK_EX on the region file for *each* region the
instance occupies — in lexicographic order to prevent deadlock.

The "full" instance acquires multiple region locks; a "quarter" acquires
exactly one. Two quarter instances on disjoint regions can run
concurrently because their lock sets don't intersect.

Lock files live under `tmp_dir/cpu_region.{role}.{region}.lock` (the
orchestrator's configured tmp_dir, defaulting to /mnt/raid0/llm/tmp).
Files are created on-demand on first acquisition. They're zero-content;
the flock state is what matters.

Behavior modeled on `src.runtime.inference_lock`:
- Honors a `deadline_s` absolute deadline + per-acquire `timeout_s`.
- Honors a `cancel_check` callable for early abort.
- Polls at `ORCHESTRATOR_INFERENCE_LOCK_POLL_MS` (default 50ms) — same
  knob as the existing lock module.
- Logs periodic "still waiting" diagnostics when blocked.

Concurrency safety:
- All-or-nothing acquisition: if any region lock times out, all
  previously-acquired locks are released before raising. No partial
  state.
- LIFO release on context exit: ensures fcntl's per-fd state stays
  consistent.

2026-05-22: introduced for the full+quarter concurrency project; lives
alongside the legacy `inference_lock` until phase-3 migration completes.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, IO, Optional

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _tmp_dir() -> Path:
    """Resolve the orchestrator's configured tmp_dir.

    Resolution order (env override wins so tests can redirect):
      1. ORCHESTRATOR_TMP_DIR env var (explicit override)
      2. src.config.get_config().paths.tmp_dir
      3. /mnt/raid0/llm/tmp hard-coded fallback
    """
    env_override = os.environ.get("ORCHESTRATOR_TMP_DIR")
    if env_override:
        return Path(env_override)
    try:
        from src.config import get_config  # type: ignore[import-not-found]
        return Path(get_config().paths.tmp_dir)
    except Exception:
        return Path("/mnt/raid0/llm/tmp")


def _lock_poll_s() -> float:
    return max(0.005, _env_float("ORCHESTRATOR_INFERENCE_LOCK_POLL_MS", 50.0) / 1000.0)


def _lock_log_every_s() -> float:
    return max(1.0, _env_float("ORCHESTRATOR_INFERENCE_LOCK_LOG_EVERY_S", 15.0))


def _default_timeout_s() -> float:
    return _env_float("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S", 180.0)


def region_lock_path(role: str, region: str) -> Path:
    """Return the lock-file path for one atomic region of a role.

    Path format: {tmp_dir}/cpu_region.{role}.{region}.lock
    Both `role` and `region` are sanitized to remove path separators —
    callers should never pass attacker-controlled input but defense is
    cheap.
    """
    safe_role = role.replace("/", "_").replace("\\", "_")
    safe_region = region.replace("/", "_").replace("\\", "_")
    return _tmp_dir() / f"cpu_region.{safe_role}.{safe_region}.lock"


class CpuRegionLockTimeout(RuntimeError):
    """Raised when one of the region locks could not be acquired within
    the time budget. All locks acquired before the timeout are released
    before this exception propagates.
    """


def _try_flock(fd: int, lock_type: int) -> bool:
    """Non-blocking fcntl.flock — returns True on success, False if the
    lock is held by another process. Any other OSError propagates.
    """
    try:
        fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
        return True
    except OSError as e:
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return False
        raise


def _acquire_one_with_timeout(
    fh: IO[bytes],
    *,
    region: str,
    role: str,
    timeout_s: float,
    deadline_s: Optional[float],
    cancel_check: Optional[Callable[[], bool]],
    request_tag: Optional[str],
) -> float:
    """Block until LOCK_EX is acquired on `fh`, or raise on timeout/cancel.

    Returns elapsed wait seconds. Logs every ~15s while waiting.
    """
    start = time.perf_counter()
    poll_s = _lock_poll_s()
    log_every_s = _lock_log_every_s()
    last_log = start
    last_holders_logged: tuple[str, ...] | None = None
    abs_deadline = None if timeout_s <= 0 else (start + timeout_s)

    while True:
        if _try_flock(fh.fileno(), fcntl.LOCK_EX):
            return time.perf_counter() - start
        if cancel_check is not None and cancel_check():
            raise CpuRegionLockTimeout(
                f"region lock cancelled before acquire (role={role}, region={region}, "
                f"tag={request_tag})"
            )
        if deadline_s is not None and time.perf_counter() >= deadline_s:
            raise CpuRegionLockTimeout(
                f"region lock deadline exceeded (role={role}, region={region}, "
                f"tag={request_tag})"
            )
        now = time.perf_counter()
        if abs_deadline is not None and now >= abs_deadline:
            raise CpuRegionLockTimeout(
                f"region lock timeout after {timeout_s:.1f}s "
                f"(role={role}, region={region}, tag={request_tag})"
            )
        if now - last_log >= log_every_s:
            holders = _current_lock_owner_pids(Path(fh.name))
            holder_tuple = tuple(holders)
            if holder_tuple != last_holders_logged:
                logger.info(
                    "still waiting for region lock role=%s region=%s "
                    "elapsed=%.1fs holders=%s",
                    role, region, now - start, ",".join(holders) or "(unknown)",
                )
                last_holders_logged = holder_tuple
            last_log = now
        time.sleep(poll_s)


def _current_lock_owner_pids(lock_file: Path) -> list[str]:
    """Best-effort lock owners from /proc/locks for the target inode."""
    try:
        inode = str(lock_file.stat().st_ino)
    except Exception:
        return []
    owners: set[str] = set()
    try:
        with open("/proc/locks", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 6:
                    continue
                pid = parts[4]
                dev_inode = parts[5]
                if not pid.isdigit():
                    continue
                if dev_inode.rsplit(":", 1)[-1] == inode:
                    owners.add(pid)
    except Exception:
        return []
    return sorted(owners)


@contextmanager
def cpu_region_lock(
    role: str,
    regions: frozenset[str] | set[str] | list[str] | tuple[str, ...],
    *,
    timeout_s: Optional[float] = None,
    deadline_s: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    request_tag: Optional[str] = None,
):
    """Acquire LOCK_EX on each region's file lock, in lexicographic order.

    Yields a dict {region: Path} so callers can inspect / log which lock
    files were taken. Releases all locks (LIFO order) on context exit,
    including on exceptions.

    Args:
        role: Role name (e.g. "frontdoor", "worker_general"). Used only
            for the lock filename and diagnostic logs.
        regions: Set of atomic region identifiers (e.g. {"q0", "q1"}).
            Iteration order is sorted internally; callers can pass any
            iterable.
        timeout_s: Per-acquire timeout in seconds. None → use the env
            default (`ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S` or 180).
            0 or negative disables timeout (blocks indefinitely).
        deadline_s: Absolute deadline as `time.perf_counter()` value.
            Both `timeout_s` and `deadline_s` may be supplied; whichever
            is more restrictive applies per-region. If you have a budget
            for the whole multi-lock acquisition, pass `deadline_s`.
        cancel_check: Optional zero-arg callable that returns True to
            cancel an in-progress acquire (e.g. client disconnected).
        request_tag: Opaque tag included in diagnostic logs to help
            identify which inference triggered the lock attempt.

    Raises:
        CpuRegionLockTimeout: if any region cannot be acquired within
            the budget. All previously-acquired locks released first.

    Example:
        from src.runtime.instance_topology import get_instance_regions
        regions = get_instance_regions().get(("frontdoor", 0), frozenset())
        with cpu_region_lock("frontdoor", regions, deadline_s=time.perf_counter() + 60):
            # safe to dispatch to frontdoor's full instance — no other
            # instance can be running on q0..q3 right now
            run_inference()
    """
    if not regions:
        # Empty region set → no-op context. Treat as no-CPU-conflict
        # (some embedder paths route through this for code simplicity).
        yield {}
        return

    if timeout_s is None:
        timeout_s = _default_timeout_s()

    sorted_regions = sorted(regions)
    handles: list[tuple[str, Path, IO[bytes]]] = []
    acquired_paths: dict[str, Path] = {}

    try:
        for region in sorted_regions:
            path = region_lock_path(role, region)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Open in 'a+b' so the file is created if missing and we have a
            # binary fd suitable for fcntl.flock.
            fh = open(path, "a+b")
            try:
                _acquire_one_with_timeout(
                    fh,
                    region=region,
                    role=role,
                    timeout_s=timeout_s,
                    deadline_s=deadline_s,
                    cancel_check=cancel_check,
                    request_tag=request_tag,
                )
            except BaseException:
                fh.close()
                raise
            handles.append((region, path, fh))
            acquired_paths[region] = path
        yield acquired_paths
    finally:
        # LIFO release: close in reverse acquire order. Closing the file
        # descriptor releases the fcntl lock automatically.
        for _region, _path, fh in reversed(handles):
            try:
                fh.close()
            except Exception:
                pass


@contextmanager
def cpu_region_lock_for_instance(
    role: str,
    instance_idx: int,
    *,
    timeout_s: Optional[float] = None,
    deadline_s: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    request_tag: Optional[str] = None,
):
    """Convenience: look up regions for (role, instance_idx) via
    `instance_topology.get_instance_regions()` and acquire them.

    If the (role, instance_idx) pair is unknown to the topology table
    (e.g. an embedder or a role not declared in NUMA_CONFIG), yields
    an empty path-set and runs the body without any locking. Callers
    that need strict locking should use `cpu_region_lock(role, regions)`
    directly and assemble `regions` themselves.
    """
    from src.runtime.instance_topology import get_instance_regions
    regions = get_instance_regions().get((role, instance_idx), frozenset())
    with cpu_region_lock(
        role, regions,
        timeout_s=timeout_s, deadline_s=deadline_s,
        cancel_check=cancel_check, request_tag=request_tag,
    ) as paths:
        yield paths
