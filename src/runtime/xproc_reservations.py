"""Reusable cross-process reservation store (host-wide admission state).

Production runs ``uvicorn --workers 6``: six processes. Any admission counter kept in a
Python object is therefore per process, and six workers together over-commit whatever the
counter guards. This module is the shared primitive: one small JSON file per key (a server
URL) under a run dir, every read-modify-write guarded by ``fcntl.flock`` (``LOCK_EX`` on the
file itself).

* Each entry records ``(id, pid, start, t, n)``. Entries whose owner is dead (``kill 0``
  fails, or the ``/proc/<pid>/stat`` start time differs, i.e. pid reuse) or older than the
  TTL are reaped under the lock before every view, so a crashed worker cannot leak capacity.
* FAIL CLOSED: if the dir, file or lock is unavailable (or the file is corrupt),
  `ReservationUnavailable` is raised and the caller must admit nothing (or queue). It never
  over-admits. Unlike ``scripts/autopilot/state_lock.py`` (fails OPEN on timeout), admission
  needs this, hence a dedicated lock helper.
* Lock waits poll with exponential backoff (a few ms up to ``_POLL_MAX_S``) and a hard
  timeout; never a busy spin.

Two usage shapes, both under one `ReservationStore.locked(key)` view:

* per-reservation tokens: ``slate.add(n)`` / ``slate.remove(token)`` (scout slots);
* one aggregate entry per (process, tag): ``slate.set_own(tag, n)`` and
  ``slate.foreign_total()`` (the other processes' sum), for gates that keep rich per-process
  state locally and only need the host-wide *sum* to decide admission (KV-pool tokens, queue
  depth). ``foreign_total`` excludes this pid, so several in-process instances (tests) never
  see each other while distinct processes always do.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import logging
import os
import random
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

LOCK_TIMEOUT_S = 2.0
_POLL_MIN_S = 0.002
_POLL_MAX_S = 0.05
_TRANSIENT = frozenset({errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK})


class ReservationUnavailable(RuntimeError):
    """The shared reservation store cannot be used; the caller must admit nothing."""


def proc_start_ticks(pid: int) -> str | None:
    """Field 22 of ``/proc/<pid>/stat`` (process start time); None if unreadable."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            raw = fh.read().decode("ascii", "replace")
        return raw[raw.rindex(")") + 2:].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def pid_alive(pid: int, start: str | None) -> bool:
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


class Slate:
    """The locked, reaped view of one key's entries."""

    def __init__(self, entries: list[dict[str, Any]]):
        self.entries = entries
        self.dirty = False

    def inflight(self) -> int:
        """Sum of ``n`` over every live entry (all processes)."""
        return sum(int(e.get("n", 0)) for e in self.entries)

    def foreign_total(self, tag: str | None = None) -> int:
        """Sum of ``n`` over live entries owned by OTHER pids (optionally one ``tag``)."""
        me = os.getpid()
        return sum(int(e.get("n", 0)) for e in self.entries
                   if e.get("pid") != me and (tag is None or e.get("tag") == tag))

    def add(self, n: int) -> str:
        pid = os.getpid()
        token = uuid.uuid4().hex
        self.entries.append({"id": token, "pid": pid, "start": proc_start_ticks(pid),
                             "t": time.time(), "n": int(n)})
        self.dirty = True
        return token

    def set_own(self, tag: str, n: int) -> None:
        """Upsert this process's aggregate entry for ``tag`` (``n <= 0`` removes it)."""
        pid = os.getpid()
        eid = f"own-{pid}-{tag}"
        kept = [e for e in self.entries if e.get("id") != eid]
        if n > 0:
            kept.append({"id": eid, "pid": pid, "start": proc_start_ticks(pid),
                         "t": time.time(), "n": int(n), "tag": tag})
        if kept != self.entries:
            self.entries = kept
            self.dirty = True

    def remove(self, token: str) -> None:
        kept = [e for e in self.entries if e.get("id") != token]
        if len(kept) != len(self.entries):
            self.entries = kept
            self.dirty = True


def reap(entries: list[dict[str, Any]], now: float, ttl: float) -> list[dict[str, Any]]:
    live = []
    for e in entries:
        try:
            pid = int(e["pid"])
            age = now - float(e["t"])
            n = int(e["n"])
        except (KeyError, TypeError, ValueError):
            continue                            # malformed -> drop
        if n <= 0 or age > ttl or not pid_alive(pid, e.get("start")):
            continue
        live.append(e)
    return live


class ReservationStore:
    """A namespace of per-key reservation files with its own dir/TTL environment knobs."""

    def __init__(self, namespace: str, *, dir_env: str, ttl_env: str,
                 default_ttl_s: float = 900.0, lock_timeout_s: float = LOCK_TIMEOUT_S):
        self.namespace = namespace
        self.dir_env = dir_env
        self.ttl_env = ttl_env
        self.default_ttl_s = float(default_ttl_s)
        self.lock_timeout_s = float(lock_timeout_s)
        #: flock is per open-file-description, so threads of one process also exclude each
        #: other through their own fds; this lock just avoids needless polling.
        self._thread_lock = threading.Lock()

    def reserve_dir(self) -> Path:
        raw = os.environ.get(self.dir_env)
        if raw:
            return Path(raw)
        return Path(tempfile.gettempdir()) / f"epyc-{self.namespace}-{os.getuid()}"

    def ttl_s(self) -> float:
        try:
            return float(os.environ.get(self.ttl_env, self.default_ttl_s))
        except ValueError:
            return self.default_ttl_s

    @contextlib.contextmanager
    def locked(self, key: str, *, timeout: float | None = None) -> Iterator[Slate]:
        """Exclusive, reaped view of ``key``. Raises `ReservationUnavailable`."""
        timeout = self.lock_timeout_s if timeout is None else timeout
        path = None
        fd = -1
        got_thread = False
        try:
            d = self.reserve_dir()
            d.mkdir(parents=True, exist_ok=True)
            path = d / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:20] + ".json")
            deadline = time.monotonic() + timeout
            got_thread = self._thread_lock.acquire(timeout=timeout)
            if not got_thread:
                raise ReservationUnavailable("thread lock timeout")
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            delay = _POLL_MIN_S
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in _TRANSIENT:
                        raise
                    if time.monotonic() >= deadline:
                        raise ReservationUnavailable(f"flock timeout on {path}") from exc
                    time.sleep(delay * (0.5 + random.random()))
                    delay = min(delay * 2, _POLL_MAX_S)
            try:
                raw = os.read(fd, 1 << 20)
                entries: list[dict[str, Any]] = []
                if raw.strip():
                    loaded = json.loads(raw.decode("utf-8"))
                    entries = list(loaded.get("entries", [])) if isinstance(loaded, dict) else []
            except (ValueError, UnicodeDecodeError):
                raise ReservationUnavailable(f"corrupt reservation file {path}") from None
            slate = Slate(reap(entries, time.time(), self.ttl_s()))
            slate.dirty = len(slate.entries) != len(entries)
            yield slate
            if slate.dirty:
                payload = json.dumps({"url": key, "entries": slate.entries}).encode("utf-8")
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
                    os.close(fd)                # closing drops the flock
            if got_thread:
                self._thread_lock.release()

    def release(self, key: str | None, token: str | None) -> bool:
        """Drop one reservation by token. Never raises; False if the store was unavailable."""
        if not key or not token:
            return True
        try:
            with self.locked(key) as slate:
                slate.remove(token)
            return True
        except ReservationUnavailable as exc:
            # The owner pid dies with the worker or the TTL expires, so a failed release
            # is bounded; it can only under-admit meanwhile.
            log.warning("%s: could not release reservation for %s: %s", self.namespace, key, exc)
            return False
