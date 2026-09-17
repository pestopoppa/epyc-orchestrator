"""Cross-process session ownership lease with fencing tokens (D-f).

Why this exists
---------------
The orchestrator API runs under uvicorn with several worker PROCESSES. A
``threading.Lock`` (or ``asyncio.Lock``) only serializes callers inside one
process, so two requests for the same ``session_id`` that land on different
workers would both restore the same checkpoint, both run, and both write a
checkpoint back: the later write silently erases the earlier turn. Ownership
therefore has to live in the one thing every worker shares, the SQLite session
database, and be decided inside a write transaction.

Contract
--------
* One row per ``session_id`` in ``session_leases``. The row is never deleted,
  so ``fencing_token`` increases monotonically across every acquisition of that
  session for the lifetime of the database.
* :meth:`SessionLeaseManager.acquire` runs under ``BEGIN IMMEDIATE``, which
  takes SQLite's RESERVED lock, so two simultaneous acquirers (in any process)
  are serialized and exactly one of them wins.
* A lease held by someone else is reclaimable only when
  (a) it has expired (``expires_at <= now``), or
  (b) liveness reconciliation proves the owner is dead: same host and boot, and
      the owner PID is gone or now names a DIFFERENT process (PID reuse,
      detected through the ``/proc/<pid>/stat`` start time), or the host has
      rebooted since the lease was written.
  An owner on another host is never declared dead; it can only expire.
* Every mutating checkpoint/session write passes through :func:`check_fence`
  inside its own write transaction. A write that presents a token must present
  the CURRENT, unreleased, unexpired one. A write that presents no token is
  refused while any live lease exists on that session. With no live lease an
  unfenced write is allowed, which keeps lease-unaware callers (CLI, the
  sessions API outside a chat turn) working.
* :meth:`SessionLeaseManager.release` is idempotent: releasing twice, or
  releasing a lease somebody else has since reclaimed, is a no-op that returns
  ``False`` and never disturbs the current owner.
* Reads never take the lease. Concurrent read-only inspection is unaffected.

Clock: wall-clock ``time.time()``. All workers of one API share a host clock;
an owner on another host with a skewed clock is the residual risk, which is why
the TTL is generous and a foreign owner is never judged by PID.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_LEASE_TTL_S = 60.0
DEFAULT_POLL_S = 0.05
_MAX_POLL_S = 0.5
_BUSY_TIMEOUT_S = 30.0

LEASE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS session_leases (
    session_id TEXT PRIMARY KEY,
    holder_id TEXT NOT NULL,
    owner_label TEXT NOT NULL DEFAULT '',
    owner_host TEXT NOT NULL,
    owner_boot_id TEXT NOT NULL,
    owner_pid INTEGER NOT NULL,
    owner_start_ticks TEXT NOT NULL,
    fencing_token INTEGER NOT NULL,
    acquired_at REAL NOT NULL,
    heartbeat_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    released_at REAL
)
"""


class SessionLeaseError(RuntimeError):
    """Base class for session-lease failures."""


class SessionLeaseHeld(SessionLeaseError):
    """Another live owner holds the lease."""

    def __init__(self, session_id: str, holder: "LeaseRecord"):
        self.session_id = session_id
        self.holder = holder
        super().__init__(
            f"session {session_id!r} is leased by pid {holder.owner_pid} on "
            f"{holder.owner_host} (token {holder.fencing_token}, expires in "
            f"{max(0.0, holder.expires_at - time.time()):.1f}s)"
        )


class SessionLeaseLost(SessionLeaseError):
    """The caller's lease is no longer current (expired, released or reclaimed)."""


class StaleFencingToken(SessionLeaseError):
    """A mutating write presented a token that is not the live one."""


# ---------------------------------------------------------------------------
# Process identity
# ---------------------------------------------------------------------------


def _read_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def process_start_ticks(pid: int) -> str | None:
    """Start time of ``pid`` in clock ticks since boot, or None if it does not exist.

    Field 22 of ``/proc/<pid>/stat``. The comm field (2) may contain spaces and
    parentheses, so split after the LAST ``)``.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError:
        return ""  # exists but unreadable: identity unknown, not dead
    try:
        rest = raw[raw.rindex(")") + 2 :].split()
        return rest[19]  # fields start at 3 (state); 22 - 3 = 19
    except (ValueError, IndexError):
        return ""


@dataclass(frozen=True)
class OwnerIdentity:
    host: str
    boot_id: str
    pid: int
    start_ticks: str

    @classmethod
    def current(cls) -> "OwnerIdentity":
        pid = os.getpid()
        return cls(
            host=socket.gethostname(),
            boot_id=_read_boot_id(),
            pid=pid,
            start_ticks=process_start_ticks(pid) or "",
        )


@dataclass(frozen=True)
class LeaseRecord:
    session_id: str
    holder_id: str
    owner_label: str
    owner_host: str
    owner_boot_id: str
    owner_pid: int
    owner_start_ticks: str
    fencing_token: int
    acquired_at: float
    heartbeat_at: float
    expires_at: float
    released_at: float | None

    @classmethod
    def from_row(cls, row: sqlite3.Row | tuple) -> "LeaseRecord":
        return cls(*tuple(row))

    def is_live(self, now: float) -> bool:
        return self.released_at is None and self.expires_at > now


@dataclass(frozen=True)
class SessionLease:
    """A lease this caller holds. ``fencing_token`` must accompany every write."""

    session_id: str
    holder_id: str
    fencing_token: int
    expires_at: float
    ttl_s: float


_SELECT = (
    "SELECT session_id, holder_id, owner_label, owner_host, owner_boot_id, owner_pid, "
    "owner_start_ticks, fencing_token, acquired_at, heartbeat_at, expires_at, released_at "
    "FROM session_leases WHERE session_id = ?"
)


def _load(conn: sqlite3.Connection, session_id: str) -> LeaseRecord | None:
    row = conn.execute(_SELECT, (session_id,)).fetchone()
    return LeaseRecord.from_row(row) if row is not None else None


def check_fence(
    conn: sqlite3.Connection,
    session_id: str,
    fencing_token: int | None,
    *,
    now: float | None = None,
) -> None:
    """Refuse a mutating write that does not hold the live lease.

    Call inside the write's own transaction (after ``BEGIN IMMEDIATE``) so the
    check and the write are atomic with respect to lease changes.
    """
    now = time.time() if now is None else now
    record = _load(conn, session_id)
    if fencing_token is None:
        if record is not None and record.is_live(now):
            raise StaleFencingToken(
                f"session {session_id!r} is leased (token {record.fencing_token}); "
                "an unfenced write is refused"
            )
        return
    if record is None:
        raise StaleFencingToken(
            f"session {session_id!r} has no lease; token {fencing_token} is not current"
        )
    if record.fencing_token != fencing_token:
        raise StaleFencingToken(
            f"session {session_id!r}: token {fencing_token} is stale "
            f"(current {record.fencing_token})"
        )
    if record.released_at is not None:
        raise StaleFencingToken(
            f"session {session_id!r}: token {fencing_token} was released"
        )
    if record.expires_at <= now:
        raise StaleFencingToken(
            f"session {session_id!r}: token {fencing_token} expired"
        )


class SessionLeaseManager:
    """Acquire/heartbeat/release session leases in a shared SQLite database."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        identity: OwnerIdentity | None = None,
        clock: Callable[[], float] = time.time,
        liveness: Callable[[int], str | None] = process_start_ticks,
    ):
        self.db_path = Path(db_path)
        self._identity = identity
        self._clock = clock
        self._liveness = liveness
        with self._connect() as conn:
            conn.execute(LEASE_TABLE_DDL)

    # -- plumbing ---------------------------------------------------------

    @property
    def identity(self) -> OwnerIdentity:
        # Resolved lazily: a manager built before fork must report the child.
        return self._identity or OwnerIdentity.current()

    def _connect(self) -> "_Conn":
        conn = sqlite3.connect(
            self.db_path, timeout=_BUSY_TIMEOUT_S, isolation_level=None, factory=_Conn
        )
        return conn

    # -- liveness ---------------------------------------------------------

    def owner_is_dead(self, record: LeaseRecord) -> bool:
        """True only when we can PROVE the owning process no longer exists."""
        me = self.identity
        if record.owner_host != me.host:
            return False  # foreign host: expiry is the only evidence
        if record.owner_boot_id and me.boot_id and record.owner_boot_id != me.boot_id:
            return True  # host rebooted since the lease was written
        ticks = self._liveness(record.owner_pid)
        if ticks is None:
            return True  # PID gone
        if ticks and record.owner_start_ticks and ticks != record.owner_start_ticks:
            return True  # PID reused by a different process
        return False

    # -- operations -------------------------------------------------------

    def try_acquire(
        self, session_id: str, *, ttl_s: float = DEFAULT_LEASE_TTL_S, label: str = ""
    ) -> SessionLease:
        """Acquire once or raise :class:`SessionLeaseHeld`."""
        me = self.identity
        holder_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = self._clock()
            current = _load(conn, session_id)
            if current is not None and current.is_live(now) and not self.owner_is_dead(current):
                raise SessionLeaseHeld(session_id, current)
            token = (current.fencing_token if current is not None else 0) + 1
            expires = now + ttl_s
            conn.execute(
                "INSERT INTO session_leases (session_id, holder_id, owner_label, owner_host, "
                "owner_boot_id, owner_pid, owner_start_ticks, fencing_token, acquired_at, "
                "heartbeat_at, expires_at, released_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL) "
                "ON CONFLICT(session_id) DO UPDATE SET holder_id=excluded.holder_id, "
                "owner_label=excluded.owner_label, owner_host=excluded.owner_host, "
                "owner_boot_id=excluded.owner_boot_id, owner_pid=excluded.owner_pid, "
                "owner_start_ticks=excluded.owner_start_ticks, "
                "fencing_token=excluded.fencing_token, acquired_at=excluded.acquired_at, "
                "heartbeat_at=excluded.heartbeat_at, expires_at=excluded.expires_at, "
                "released_at=NULL",
                (
                    session_id, holder_id, label, me.host, me.boot_id, me.pid,
                    me.start_ticks, token, now, now, expires,
                ),
            )
            conn.execute("COMMIT")
        return SessionLease(session_id, holder_id, token, expires, ttl_s)

    def acquire(
        self,
        session_id: str,
        *,
        ttl_s: float = DEFAULT_LEASE_TTL_S,
        wait_s: float = 0.0,
        label: str = "",
    ) -> SessionLease:
        """Acquire, polling for up to ``wait_s`` while another owner holds it."""
        deadline = time.monotonic() + max(0.0, wait_s)
        poll = DEFAULT_POLL_S
        while True:
            try:
                return self.try_acquire(session_id, ttl_s=ttl_s, label=label)
            except SessionLeaseHeld:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(poll, remaining))
                poll = min(poll * 2, _MAX_POLL_S)

    def heartbeat(self, lease: SessionLease) -> SessionLease:
        """Extend a live lease; raise :class:`SessionLeaseLost` if it is not current."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = self._clock()
            expires = now + lease.ttl_s
            cur = conn.execute(
                "UPDATE session_leases SET heartbeat_at = ?, expires_at = ? "
                "WHERE session_id = ? AND holder_id = ? AND fencing_token = ? "
                "AND released_at IS NULL AND expires_at > ?",
                (now, expires, lease.session_id, lease.holder_id, lease.fencing_token, now),
            )
            conn.execute("COMMIT")
        if cur.rowcount != 1:
            raise SessionLeaseLost(
                f"session {lease.session_id!r}: lease token {lease.fencing_token} is no "
                "longer current"
            )
        return SessionLease(
            lease.session_id, lease.holder_id, lease.fencing_token, expires, lease.ttl_s
        )

    def release(self, lease: SessionLease) -> bool:
        """Release ``lease``. Idempotent; never touches a lease someone else now holds."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE session_leases SET released_at = ? "
                "WHERE session_id = ? AND holder_id = ? AND fencing_token = ? "
                "AND released_at IS NULL",
                (self._clock(), lease.session_id, lease.holder_id, lease.fencing_token),
            )
            conn.execute("COMMIT")
        return cur.rowcount == 1

    def get(self, session_id: str) -> LeaseRecord | None:
        """Read-only inspection; takes no lock beyond a snapshot read."""
        with self._connect() as conn:
            return _load(conn, session_id)


class _Conn(sqlite3.Connection):
    """Autocommit connection that rolls back an open transaction and closes on exit."""

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[override]
        try:
            if self.in_transaction:
                self.execute("ROLLBACK")
        finally:
            self.close()
        return False


# ---------------------------------------------------------------------------
# Async holder for request-scoped ownership (chat pipeline)
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class HeldSessionLease:
    """Hold a session lease for the duration of an ``async with`` block.

    Never raises on entry: a lease that cannot be obtained is reported through
    :attr:`error` so the caller decides how to degrade. While held, a background
    task renews the lease every ``ttl_s / 4`` seconds; if renewal finds the lease
    gone, :attr:`lost` is set and the caller's next fenced write is refused by
    :func:`check_fence`, which is the point of the token.
    """

    def __init__(
        self,
        manager: SessionLeaseManager,
        session_id: str,
        *,
        ttl_s: float | None = None,
        wait_s: float | None = None,
        label: str = "",
    ):
        self.manager = manager
        self.session_id = session_id
        self.ttl_s = ttl_s if ttl_s is not None else _env_float(
            "ORCHESTRATOR_SESSION_LEASE_TTL_S", DEFAULT_LEASE_TTL_S
        )
        self.wait_s = wait_s if wait_s is not None else _env_float(
            "ORCHESTRATOR_SESSION_LEASE_WAIT_S", 30.0
        )
        self.label = label
        self.lease: SessionLease | None = None
        self.error: str | None = None
        self.lost = False
        self._task = None

    @property
    def token(self) -> int | None:
        return self.lease.fencing_token if self.lease is not None else None

    async def __aenter__(self) -> "HeldSessionLease":
        import asyncio

        try:
            self.lease = await asyncio.to_thread(
                self.manager.acquire,
                self.session_id,
                ttl_s=self.ttl_s,
                wait_s=self.wait_s,
                label=self.label,
            )
        except SessionLeaseHeld as exc:
            self.error = f"held_by_other_owner: {exc}"
        except Exception as exc:  # noqa: BLE001 - reported, caller degrades
            self.error = f"lease_unavailable: {type(exc).__name__}: {exc}"
        if self.lease is not None:
            self._task = asyncio.create_task(self._renew())
        return self

    async def _renew(self) -> None:
        import asyncio
        import logging

        interval = max(0.05, self.ttl_s / 4.0)
        while self.lease is not None:
            await asyncio.sleep(interval)
            try:
                self.lease = await asyncio.to_thread(self.manager.heartbeat, self.lease)
            except SessionLeaseLost:
                self.lost = True
                return
            except Exception:  # noqa: BLE001 - transient; expiry is the backstop
                logging.getLogger(__name__).warning(
                    "session lease heartbeat failed for %s", self.session_id[:8], exc_info=True
                )

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        import asyncio
        import contextlib

        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        if self.lease is not None:
            try:
                await asyncio.to_thread(self.manager.release, self.lease)
            except Exception:  # noqa: BLE001 - expiry reclaims it anyway
                import logging

                logging.getLogger(__name__).warning(
                    "session lease release failed for %s", self.session_id[:8], exc_info=True
                )
        return False
