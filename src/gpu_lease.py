"""Process-local GPU lease for orchestrated MI210 cutover experiments."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable


CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class GpuLeaseStatus:
    acquired: bool
    owner: str | None = None
    reason: str | None = None
    age_s: float = 0.0


@dataclass(frozen=True)
class GpuLeaseAcquireResult:
    acquired: bool
    reason: str
    owner: str | None = None
    handle: "GpuLeaseHandle | None" = None


class GpuLeaseHandle:
    """Lease handle returned by :class:`GpuLeaseManager`.

    Release is idempotent so cleanup paths can safely call it from ``finally``
    blocks after partial failures.
    """

    def __init__(self, manager: "GpuLeaseManager", owner: str, token: int):
        self._manager = manager
        self.owner = owner
        self._token = token
        self._released = False

    def release(self) -> bool:
        if self._released:
            return False
        self._released = True
        return self._manager.release(self.owner, self._token)

    def __enter__(self) -> "GpuLeaseHandle":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


class GpuLeaseManager:
    """Single-owner process-local GPU lease.

    This does not start, stop, or inspect model servers. It only coordinates
    orchestrator-side ownership for features that need exclusive GPU cutover.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None):
        self._clock = clock or time.monotonic
        self._condition = threading.Condition()
        self._owner: str | None = None
        self._reason: str | None = None
        self._acquired_at: float = 0.0
        self._token = 0

    def status(self) -> GpuLeaseStatus:
        with self._condition:
            if self._owner is None:
                return GpuLeaseStatus(acquired=False)
            return GpuLeaseStatus(
                acquired=True,
                owner=self._owner,
                reason=self._reason,
                age_s=max(0.0, self._clock() - self._acquired_at),
            )

    def acquire(
        self,
        owner: str,
        *,
        reason: str | None = None,
        wait: bool = False,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> GpuLeaseAcquireResult:
        if not owner:
            raise ValueError("owner must be non-empty")
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")

        start = self._clock()
        with self._condition:
            while self._owner is not None:
                if cancel_check is not None and cancel_check():
                    return GpuLeaseAcquireResult(False, "cancelled", owner=self._owner)
                now = self._clock()
                if deadline_s is not None and now >= deadline_s:
                    return GpuLeaseAcquireResult(False, "deadline", owner=self._owner)
                if not wait:
                    return GpuLeaseAcquireResult(False, "occupied", owner=self._owner)

                remaining: float | None = None
                if timeout_s is not None:
                    remaining = max(0.0, timeout_s - (now - start))
                    if remaining <= 0:
                        return GpuLeaseAcquireResult(False, "timeout", owner=self._owner)
                if deadline_s is not None:
                    deadline_remaining = max(0.0, deadline_s - now)
                    remaining = (
                        deadline_remaining
                        if remaining is None
                        else min(remaining, deadline_remaining)
                    )
                self._condition.wait(timeout=remaining)

            self._token += 1
            self._owner = owner
            self._reason = reason
            self._acquired_at = self._clock()
            handle = GpuLeaseHandle(self, owner, self._token)
            return GpuLeaseAcquireResult(True, "acquired", owner=owner, handle=handle)

    def release(self, owner: str, token: int) -> bool:
        with self._condition:
            if self._owner != owner or self._token != token:
                return False
            self._owner = None
            self._reason = None
            self._acquired_at = 0.0
            self._condition.notify_all()
            return True
