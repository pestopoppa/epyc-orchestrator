"""Region-lock + lease integration for the GPU shadow lane (P2-1).

PROGRAM CONSTRAINT (operator-accepted 2026-07-28): **an idle MI210 does not
imply a startable lane.** The lane's host threads pin to SMT siblings 184-191,
whose physical cores are 88-95 = atomic region ``q3``. Starting the lane means
holding q3's flock, the same lock the production CPU roles take.

Measured, not inferred (``scripts/region-lock status``, 2026-07-28T16:2xZ):
``q0`` HELD frontdoor[1], ``q1`` HELD frontdoor[2], ``q2`` free, ``q3`` **HELD**
by ``bench-e8-quality`` (``e8-v5-r2-cadencefix-20260728T160917Z``) — while
``rocm-smi`` simultaneously reported VRAM 0% and no KFD PIDs. The GPU was
entirely idle and the lane was still not startable. "The GPU looks free" is
therefore not a precondition, not exclusion, and not evidence.

Two physical resources, two facts, one rule each:

1. **Host CPU slice.** The lane's host threads run on SMT siblings 184-191.
   Those siblings sit on PHYSICAL cores 88-95, which live in atomic region
   ``q3``. Occupying them means holding ``q3``'s flock via
   ``src.runtime.cpu_region_lock`` — the same lock every production role uses,
   so the lane cannot be invisible to them or they to it.

2. **The GPU device.** ROCm0 is not a CPU region and must never be modelled as
   one. It gets its own flock file. Modelling the device as a pseudo-region
   would put two different physical resources behind one lock and make
   "who holds the GPU" unanswerable.

Fabric axioms this file implements literally:

- **Axiom 1 — one fact per physical resource.** The flock IS the claim. The
  advisory lease layer here sits ABOVE the flock and never claims to be it: a
  lease record with no live flock is stale metadata, not a claim.
- **Axiom 4 / BUS_PROTOCOL rule 8 — reclaim is always quiesce-and-drain.**
  Revocation marks the holder ``revoking``; the holder stops accepting new work
  and releases at its NEXT BOUNDARY. There is no forcible path in this module:
  no kill, no signal, no lock-breaking. A revocation the holder ignores must
  surface as a defect, never as a silent inconsistency — hence
  ``LaneLease.overdue``.
- **BUS_PROTOCOL rule 7 — claims are acquired, not observed.** ``probe_*``
  helpers exist for scheduling hints and are explicitly NOT exclusion. Nothing
  in this module lets an observation stand in for a claim.

Everything is behind the default-off ``ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE``
flag, and acquiring requires an explicit call — importing this module claims
nothing.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import errno
import fcntl
import json
import os
from pathlib import Path
from typing import Callable, Iterator

from scripts.server.gpu_shadow_lane import (
    GpuShadowLaneDisabled,
    LANE_DEVICE,
    LANE_HOST_CPUSET,
    LANE_NAME,
    lane_enabled,
)
from src.features import Features
from src.runtime.cpu_region_lock import _tmp_dir, cpu_region_lock
from src.runtime.instance_topology import REGION_CORE_RANGE

# EPYC 9655: logical CPUs 96-191 are the SMT siblings of physical cores 0-95.
SMT_SIBLING_OFFSET = 96
MAX_PHYSICAL_CORE = 95

# Lease states. These mirror BUS_PROTOCOL rule 8 exactly so a lane lease and a
# bus queue row cannot drift into disagreeing vocabularies.
LEASE_HELD = "held"
LEASE_REVOKING = "revoking"
LEASE_DRAINING = "draining"
LEASE_RELEASED = "released"


class LaneLeaseError(RuntimeError):
    """Raised when a lease operation violates the drain-never-force contract."""


# ── SMT folding: the step raw string overlap gets wrong ──────────────────────


def parse_cpu_spec(spec: str) -> set[int]:
    """Parse a taskset-style list into logical CPU ids (no folding)."""
    cpus: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            cpus.update(range(int(lo_s), int(hi_s) + 1))
        else:
            cpus.add(int(chunk))
    return cpus


def fold_smt_to_physical(spec: str) -> set[int]:
    """Fold a logical-CPU spec onto the PHYSICAL cores it actually occupies.

    This is the correction for P2-4 finding P1-2. Two facts collide here:

    - ``src.runtime.instance_topology.parse_cpu_list`` DROPS logical CPUs
      96-191 rather than folding them, so the lane's "184-191" maps to the
      empty set and therefore to no region at all.
    - The preflight probe's own parser expands ranges without folding, so
      "184-191" vs ``architect_general``'s "0-95" shows zero overlap — even
      though 184-191 are the siblings of 88-95, which architect_general owns.

    Either way the lane looks free of a role that is physically sharing its
    cores. Folding first is what makes both questions answerable.
    """
    folded: set[int] = set()
    for cpu in parse_cpu_spec(spec):
        core = cpu - SMT_SIBLING_OFFSET if cpu > MAX_PHYSICAL_CORE else cpu
        if 0 <= core <= MAX_PHYSICAL_CORE:
            folded.add(core)
    return folded


def physical_cores_to_regions(cores: set[int]) -> frozenset[str]:
    """Map physical cores onto the atomic regions that contain them."""
    touched: set[str] = set()
    for core in cores:
        for region, (lo, hi) in REGION_CORE_RANGE.items():
            if lo <= core <= hi:
                touched.add(region)
                break
    return frozenset(touched)


def lane_host_regions(host_cpuset: str = LANE_HOST_CPUSET) -> frozenset[str]:
    """Atomic CPU regions the lane's host threads occupy. For 184-191: {q3}."""
    return physical_cores_to_regions(fold_smt_to_physical(host_cpuset))


def cpuset_shares_physical_cores(a: str, b: str) -> set[int]:
    """Physical cores two logical-CPU specs share (SMT-aware). 0 = disjoint."""
    return fold_smt_to_physical(a) & fold_smt_to_physical(b)


# ── GPU device claim (its own flock — never a CPU pseudo-region) ─────────────


def device_lock_path(device: str = LANE_DEVICE) -> Path:
    safe = device.replace("/", "_").replace("\\", "_")
    return _tmp_dir() / f"gpu_device.{safe}.lock"


@contextmanager
def gpu_device_lock(
    device: str = LANE_DEVICE, *, holder: str = LANE_NAME
) -> Iterator[Path]:
    """Hold an exclusive, NON-BLOCKING flock on the GPU device.

    Non-blocking on purpose: a GPU claim that queues would sit behind an
    unrelated bench window for an unbounded time while the caller believed it
    was making progress. Failing fast lets the caller take other work — the bus
    contract's "never block" rule.

    The payload is attribution only. The flock is the fact (axiom 1); if the
    payload and the flock ever disagree, the flock wins.
    """
    path = device_lock_path(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")  # noqa: SIM115 — released in the finally below
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LaneLeaseError(
                    f"GPU device {device} is already claimed (lock {path}). A claim is "
                    "acquired, not observed — do not proceed on the assumption it is free."
                ) from exc
            raise
        payload = {
            "holder": holder,
            "device": device,
            "pid": os.getpid(),
            "acquired_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(payload).encode("utf-8"))
        handle.flush()
        yield path
    finally:
        try:
            handle.seek(0)
            handle.truncate()
            handle.flush()
        except OSError:
            pass
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def read_device_lock_payload(device: str = LANE_DEVICE) -> dict | None:
    """Read the device lock's attribution payload.

    SCHEDULING HINT ONLY. This is an observation and therefore TOCTOU: the
    holder may release, or another claimant may acquire, between this read and
    any action taken on it (BUS_PROTOCOL rule 7). Never gate occupancy on it.
    """
    path = device_lock_path(device)
    try:
        raw = path.read_bytes().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


# ── The advisory lease layer (sits ABOVE the flocks) ─────────────────────────


@dataclass
class LaneLease:
    """Advisory lease over a held lane claim. Drain-at-boundary, never forcible.

    Lifecycle: ``held`` -> (revoke requested) ``revoking`` -> (holder reaches a
    boundary) ``draining`` -> ``released``. The transitions are driven by the
    HOLDER calling ``at_boundary()``; a revoker can only ask.
    """

    lane: str
    regions: frozenset[str]
    device: str | None
    state: str = LEASE_HELD
    revoke_reason: str | None = None
    boundaries_since_revoke: int = 0

    @property
    def accepting_work(self) -> bool:
        """New work is admitted only while the lease is fully held."""
        return self.state == LEASE_HELD

    @property
    def overdue(self) -> bool:
        """Revoked, but the holder passed a boundary without draining.

        BUS_PROTOCOL rule 8: this must surface as a defect. It is deliberately a
        property rather than an exception — the revoker cannot force the holder,
        so the only correct response is to REPORT, not to seize.
        """
        return self.state == LEASE_REVOKING and self.boundaries_since_revoke > 0

    def request_revoke(self, reason: str) -> None:
        """Ask the holder to drain. Never stops work in flight (axiom 4)."""
        if self.state in (LEASE_DRAINING, LEASE_RELEASED):
            return
        self.state = LEASE_REVOKING
        self.revoke_reason = reason

    def at_boundary(self) -> bool:
        """Call at every request/task boundary. True = release now.

        This is the ONLY place a revocation takes effect. Mid-decode preemption
        has no code path here, by construction rather than by discipline.
        """
        if self.state == LEASE_REVOKING:
            if self.boundaries_since_revoke == 0:
                self.state = LEASE_DRAINING
                self.boundaries_since_revoke += 1
                return True
            self.boundaries_since_revoke += 1
            return True
        return False

    def force_release(self) -> None:
        raise LaneLeaseError(
            "forcible release is not implementable here by design (fabric axiom 4): "
            "reclaim is quiesce-and-drain at a boundary. If a holder will not drain, "
            "file a defect — do not seize."
        )


@contextmanager
def lane_claim(
    *,
    host_cpuset: str = LANE_HOST_CPUSET,
    device: str | None = LANE_DEVICE,
    lock_role: str = LANE_NAME,
    timeout_s: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    feats: Features | None = None,
) -> Iterator[LaneLease]:
    """Acquire the lane's CPU regions and (optionally) its GPU device.

    Acquisition order is CPU regions then device, and release is the reverse.
    Fixed ordering is what keeps a lane claim from deadlocking against a
    production role that takes CPU regions for its own reasons.

    Yields a LaneLease. The caller MUST poll ``lease.at_boundary()`` at its
    request boundaries; that is the drain contract, and nothing else honours it.
    """
    if not lane_enabled(feats):
        raise GpuShadowLaneDisabled(
            "gpu_shadow_lane feature flag is off — refusing to claim host regions "
            "or the GPU device"
        )
    regions = lane_host_regions(host_cpuset)
    if not regions:
        raise LaneLeaseError(
            f"host cpuset {host_cpuset!r} folds to no physical cores — refusing to "
            "run unpinned. An empty region set would take NO lock and look like a "
            "successful claim."
        )
    with cpu_region_lock(
        lock_role,
        regions,
        timeout_s=timeout_s,
        cancel_check=cancel_check,
        request_tag=f"{LANE_NAME}:host",
    ):
        if device is None:
            yield LaneLease(lane=LANE_NAME, regions=regions, device=None)
            return
        with gpu_device_lock(device, holder=lock_role):
            yield LaneLease(lane=LANE_NAME, regions=regions, device=device)
