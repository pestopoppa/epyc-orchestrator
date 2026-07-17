"""Cross-process advisory lock serializing read-modify-write on autopilot_state.json.

Background (2026-07-17 incoherence root-cause, hypothesis H4): autopilot_state.json
is a whole-file JSON rewritten by 5+ independent processes — the autopilot daemon
(per-trial save), the dashboard API under ``uvicorn --workers 6`` (a pause/resume
click does a full read-modify-write), ``host_health.flush_cache_with_pause``,
``config_applicator``, and the archiver. Writes are ATOMIC (tmp + os.replace) but
had ZERO mutual exclusion, so concurrent read-modify-write cycles LOST updates:
writer B's whole-file write, based on a stale read, clobbers writer A's committed
change -> state.json silently diverges from the append-only journal -> the
dashboard's autopilot panels disagree. Atomicity prevents *torn reads*; it does
NOT prevent *lost updates*. This ``LOCK_EX`` flock serializes the ENTIRE
read-modify-write critical section across processes on the host.

USAGE — wrap the whole read -> modify -> write, not just the write::

    with state_write_lock(state_path):
        state = load_state(state_path)     # READ
        state["paused"] = True             # MODIFY
        save_state(state, state_path)      # WRITE (atomic tmp+replace)

CRITICAL: keep the critical section SHORT. NEVER hold the lock across a sleep,
inference call, drop_caches, or NUMA rewarm — that stalls every other writer
(e.g. the autopilot daemon's per-trial save). ``host_health``'s cache-flush must
take the lock briefly to flip the pause flag, RELEASE across the ~11s sleep, then
re-take it briefly to resume — two short locked RMWs bracketing an UNLOCKED sleep.

On acquire timeout the manager FAILS OPEN (logs a loud warning and proceeds
without the lock) rather than deadlocking/crashing a writer: a single lost update
is less harmful than stalling autopilot, and a wait beyond ``timeout`` signals a
distinct bug (a writer wrongly holding the lock across a slow op) to fix at that
call site rather than here. The context manager yields ``True`` when the lock was
actually held, ``False`` when it failed open.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import time
from typing import Iterator, Union

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0
_POLL_S = 0.05
_TRANSIENT = frozenset({errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK})


@contextlib.contextmanager
def state_write_lock(
    state_path: Union[str, "os.PathLike[str]"],
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Iterator[bool]:
    """Serialize a read-modify-write of ``state_path`` across processes.

    Acquires an exclusive advisory ``flock`` on ``<state_path>.lock``. Yields
    ``True`` if the lock was acquired, ``False`` if it failed open after
    ``timeout`` seconds. Always releases the lock and closes the fd on exit,
    including on exception.
    """
    lock_path = f"{os.fspath(state_path)}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    acquired = False
    deadline = time.monotonic() + max(0.0, timeout)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in _TRANSIENT:
                    raise
                if time.monotonic() >= deadline:
                    logger.warning(
                        "state_write_lock: could not acquire %s within %.1fs; "
                        "proceeding WITHOUT the lock (fail-open) — a writer is "
                        "likely holding it across a slow op; fix that call site.",
                        lock_path,
                        timeout,
                    )
                    break
                time.sleep(_POLL_S)
        yield acquired
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
