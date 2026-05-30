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


# Reserved pseudo-role for the cross-role global region mutex (A-1). Never a
# real role, so `active_region_holders` (which iterates the topology table)
# never surfaces it as a holder — attribution stays role-only.
_GLOBAL_MUTEX_ROLE = "GLOBAL"


def global_region_lock_path(region: str) -> Path:
    """Lock-file path for the role-AGNOSTIC global mutex on one atomic region.

    Path format: {tmp_dir}/cpu_region.GLOBAL.{region}.lock

    Acquired (when ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT is enabled)
    before the per-role region lock, so two DIFFERENT roles cannot hold the
    same physical region concurrently. The per-role locks remain the
    attribution layer (`active_region_holders`); this is a separate exclusion
    layer that does not affect attribution.
    """
    safe_region = region.replace("/", "_").replace("\\", "_")
    return _tmp_dir() / f"cpu_region.{_GLOBAL_MUTEX_ROLE}.{safe_region}.lock"


def _cross_role_mutex_enabled() -> bool:
    """A-1: gate the global cross-role region mutex behind the same flag as the
    placement-side change. Off by default → byte-identical legacy behavior
    (per-role locks only; cross-role same-region overlap remains possible)."""
    return os.environ.get(
        "ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "0"
    ).strip() in {"1", "true", "yes", "on"}


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


def active_region_holders(
    instance_regions: dict[tuple[str, int], frozenset[str]] | None = None,
) -> dict[str, list[int]]:
    """Return {role: [instance_idx, ...]} for instances currently holding
    any CPU region lock (i.e. actively dispatched under PER_REGION_LOCKS=1).

    This is the canonical cross-process source of "which role is decoding"
    used by the cross-role admission gate (Phase B of the cross-role-bw-aware
    routing handoff). Returns empty dict if PER_REGION_LOCKS is disabled or
    no instances are dispatching.

    `instance_regions`: optional override (e.g. for tests). Defaults to the
    live mapping from `src.runtime.instance_topology.get_instance_regions()`.

    The mapping is cheap to call (one /proc/locks scan + per-region stat),
    typically <2 ms even on a busy host. Safe to call from request-handling
    threads.
    """
    if instance_regions is None:
        try:
            from src.runtime.instance_topology import get_instance_regions
            instance_regions = get_instance_regions()
        except Exception:
            return {}

    if not instance_regions:
        return {}

    # Cache lock-file holder lookup per region — many instances share regions.
    region_held: dict[str, bool] = {}

    def _region_has_holder(role: str, region: str) -> bool:
        key = f"{role}.{region}"
        if key in region_held:
            return region_held[key]
        lock_path = region_lock_path(role, region)
        if not lock_path.exists():
            region_held[key] = False
            return False
        held = bool(_current_lock_owner_pids(lock_path))
        region_held[key] = held
        return held

    out: dict[str, list[int]] = {}
    for (role, idx), regions in instance_regions.items():
        if not regions:
            continue
        # Instance is "active" if ANY of its regions has a current holder.
        # That matches how cpu_region_lock acquires the union of region locks
        # for an instance — if even one is held, the instance is dispatched.
        for region in regions:
            if _region_has_holder(role, region):
                out.setdefault(role, []).append(idx)
                break
    # Sort instance indices for deterministic output (helps tests + logs).
    for role in out:
        out[role] = sorted(out[role])
    return out


def held_regions_by_role(
    instance_regions: dict[tuple[str, int], frozenset[str]] | None = None,
) -> dict[str, frozenset[str]]:
    """EXACT cross-role region view: {role: frozenset(regions with a held lock)}.

    Unlike `active_region_holders` (an ATTRIBUTION view that marks an *instance*
    active when ANY of its regions is held — and therefore over-reports, since a
    held quarter q0 also flags the role's `full` instance that contains q0),
    this reports ONLY the physical regions whose per-role lock file is actually
    held. It is the correct input for shape-aware contention/placement decisions
    (audit P1) — placement disjointness must be computed from the precise held
    region set, not from over-reported instance membership.

    `instance_regions` defaults to the live topology map; pass an override for
    tests. Roles with no held region are omitted (no empty entries). Reuses the
    same `/proc/locks` detection as `active_region_holders`; that function is
    UNCHANGED — this is a separate, additive helper.
    """
    if instance_regions is None:
        try:
            from src.runtime.instance_topology import get_instance_regions
            instance_regions = get_instance_regions()
        except Exception:
            return {}

    if not instance_regions:
        return {}

    # The atomic regions any role could occupy, keyed by role → set(regions),
    # so we probe each (role, region) lock file exactly once.
    role_regions: dict[str, set[str]] = {}
    for (role, _idx), regions in instance_regions.items():
        if regions:
            role_regions.setdefault(role, set()).update(regions)

    out: dict[str, frozenset[str]] = {}
    for role, regions in role_regions.items():
        held = {
            region
            for region in regions
            if (lp := region_lock_path(role, region)).exists()
            and _current_lock_owner_pids(lp)
        }
        if held:
            out[role] = frozenset(held)
    return out


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

    def _acquire(lock_role: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Open in 'a+b' so the file is created if missing and we have a
        # binary fd suitable for fcntl.flock.
        fh = open(path, "a+b")
        try:
            _acquire_one_with_timeout(
                fh,
                region=region,
                role=lock_role,
                timeout_s=timeout_s,
                deadline_s=deadline_s,
                cancel_check=cancel_check,
                request_tag=request_tag,
            )
        except BaseException:
            fh.close()
            raise
        handles.append((region, path, fh))

    cross_role_mutex = _cross_role_mutex_enabled()

    try:
        # A-1: when enabled, acquire the role-AGNOSTIC global region mutex for
        # ALL regions first (sorted) — true cross-role exclusion — then the
        # per-role attribution locks (also sorted). Consistent global ordering
        # (GLOBAL-all-then-role-all, each region-sorted) across every caller
        # prevents deadlock. Flag off → skip the GLOBAL layer entirely →
        # legacy behavior unchanged.
        if cross_role_mutex:
            for region in sorted_regions:
                _acquire(_GLOBAL_MUTEX_ROLE, global_region_lock_path(region))
        for region in sorted_regions:
            path = region_lock_path(role, region)
            _acquire(role, path)
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
