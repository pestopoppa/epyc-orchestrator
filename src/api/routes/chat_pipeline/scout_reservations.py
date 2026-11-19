"""Cross-process scout slot reservations (OAB-17 F8, multi-worker).

Production runs ``uvicorn --workers 6``: each worker is its own process, so an in-process
counter lets six workers read the same free /slots and together over-admit scouts past
``reserve_slots``. Reservations therefore live in one small JSON file per server URL under a
run dir, every read-modify-write guarded by ``fcntl.flock`` (``LOCK_EX`` on the file itself).

* Each reservation records ``(id, pid, start, t, n)``. Entries whose owner is dead (``kill 0``
  fails, or the ``/proc/<pid>/stat`` start time differs, i.e. pid reuse) or older than the TTL
  are reaped under the lock before every count, so a crashed worker cannot leak slots.
* FAIL SAFE: if the dir, file or lock is unavailable, `ReservationUnavailable` is raised and
  the caller admits 0 scouts. It never over-admits and never lets the error reach the request.

Unlike ``scripts/autopilot/state_lock.py`` (which fails OPEN on timeout), admission must fail
CLOSED, so this module has its own lock helper.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

DIR_ENV = "EPYC_SCOUT_RESERVE_DIR"
TTL_ENV = "EPYC_SCOUT_RESERVE_TTL_S"
DEFAULT_TTL_S = 900.0
LOCK_TIMEOUT_S = 2.0
_POLL_S = 0.01
_TRANSIENT = frozenset({errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK})

#: flock is per open-file-description, so threads of one process also exclude each other
#: through their own fds; this lock just avoids needless polling between threads.
_THREAD_LOCK = threading.Lock()


class ReservationUnavailable(RuntimeError):
    """The shared reservation store cannot be used; the caller must admit nothing."""


def reserve_dir() -> Path:
    raw = os.environ.get(DIR_ENV)
    if raw:
        return Path(raw)
    return Path(tempfile.gettempdir()) / f"epyc-scout-reserve-{os.getuid()}"


def _ttl_s() -> float:
    try:
        return float(os.environ.get(TTL_ENV, DEFAULT_TTL_S))
    except ValueError:
        return DEFAULT_TTL_S


def proc_start_ticks(pid: int) -> str | None:
    """Field 22 of ``/proc/<pid>/stat`` (process start time); None if unreadable."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            raw = fh.read().decode("ascii", "replace")
        return raw[raw.rindex(")") + 2:].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def _pid_alive(pid: int, start: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                                   # exists, owned by someone else
    except OSError:
        return False
    if start is not None:
        now = proc_start_ticks(pid)
        if now is not None and now != start:   # same pid, different process
            return False
    return True


class _Slate:
    """The locked view of one URL's reservations."""

    def __init__(self, entries: list[dict[str, Any]]):
        self.entries = entries
        self.dirty = False

    def inflight(self) -> int:
        return sum(int(e.get("n", 0)) for e in self.entries)

    def add(self, n: int) -> str:
        pid = os.getpid()
        token = uuid.uuid4().hex
        self.entries.append({"id": token, "pid": pid, "start": proc_start_ticks(pid),
                             "t": time.time(), "n": int(n)})
        self.dirty = True
        return token

    def remove(self, token: str) -> None:
        kept = [e for e in self.entries if e.get("id") != token]
        if len(kept) != len(self.entries):
            self.entries = kept
            self.dirty = True


def _reap(entries: list[dict[str, Any]], now: float, ttl: float) -> list[dict[str, Any]]:
    live = []
    for e in entries:
        try:
            pid = int(e["pid"])
            age = now - float(e["t"])
            n = int(e["n"])
        except (KeyError, TypeError, ValueError):
            continue                            # malformed -> drop
        if n <= 0 or age > ttl or not _pid_alive(pid, e.get("start")):
            continue
        live.append(e)
    return live


@contextlib.contextmanager
def locked(url: str, *, timeout: float = LOCK_TIMEOUT_S) -> Iterator[_Slate]:
    """Exclusive, reaped view of ``url``'s reservations. Raises `ReservationUnavailable`."""
    path = None
    fd = -1
    got_thread = False
    try:
        d = reserve_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / (hashlib.sha1(url.encode("utf-8")).hexdigest()[:20] + ".json")
        deadline = time.monotonic() + timeout
        got_thread = _THREAD_LOCK.acquire(timeout=timeout)
        if not got_thread:
            raise ReservationUnavailable("thread lock timeout")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in _TRANSIENT:
                    raise
                if time.monotonic() >= deadline:
                    raise ReservationUnavailable(f"flock timeout on {path}") from exc
                time.sleep(_POLL_S)
        try:
            raw = os.read(fd, 1 << 20)
            entries: list[dict[str, Any]] = []
            if raw.strip():
                loaded = json.loads(raw.decode("utf-8"))
                entries = list(loaded.get("entries", [])) if isinstance(loaded, dict) else []
        except (ValueError, UnicodeDecodeError):
            raise ReservationUnavailable(f"corrupt reservation file {path}") from None
        slate = _Slate(_reap(entries, time.time(), _ttl_s()))
        slate.dirty = len(slate.entries) != len(entries)
        yield slate
        if slate.dirty:
            payload = json.dumps({"url": url, "entries": slate.entries}).encode("utf-8")
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
    except ReservationUnavailable:
        raise
    except OSError as exc:
        raise ReservationUnavailable(f"{type(exc).__name__}: {exc}") from exc
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)                    # closing drops the flock
        if got_thread:
            _THREAD_LOCK.release()


def release(url: str | None, token: str | None) -> bool:
    """Drop one reservation by token. Never raises; False if the store was unavailable."""
    if not url or not token:
        return True
    try:
        with locked(url) as slate:
            slate.remove(token)
        return True
    except ReservationUnavailable as exc:
        # The owner pid dies with the worker or the TTL expires, so a failed release
        # is bounded; it can only under-admit meanwhile.
        log.warning("scouts: could not release reservation for %s: %s", url, exc)
        return False
