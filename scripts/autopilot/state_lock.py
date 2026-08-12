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
import uuid
from typing import Any, Dict, Iterator, Optional, Union

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


# ───────────────────────── pause lease (2026-08-12) ─────────────────────────
#
# The write lock above serializes each read-modify-write, but ``paused`` is a
# BOOLEAN with several owners, so serialization alone cannot say WHOSE pause is
# on disk. Two out-of-band pausers — ``config_applicator._pause_autopilot_dispatch``
# (around a role/API restart) and ``host_health.flush_cache_with_pause`` (around
# drop_caches + NUMA rewarm) — take the shape "remember paused_pre, set
# paused=True, do slow work, restore paused=False if paused_pre was False".
#
# THE HOLE: between the set and the restore (tens of seconds to minutes — a whole
# stack reload, or an 11 s grace + flush + serial GGUF rewarm) the operator can
# run ``autopilot.py pause``. That write is honoured — and then silently UNDONE by
# the restore, because an operator pause and the applicator's own pause are the
# same byte on disk (``"paused": true``). The operator believes AutoPilot is
# stopped; it is dispatching. Live outage 2026-08-03: the apply left the API down,
# the operator paused, the applicator's ``finally:`` resumed AutoPilot, and the
# loop retried ``Connection refused`` forever behind a pause the operator had set.
#
# THE INTERLOCK: a *lease*. A pauser stamps ``pause_owner`` + a unique
# ``pause_token`` alongside ``paused=True``, and may only clear the pause while
# its own token is still on disk. A stricter pauser (the operator) SUPERSEDES the
# lease — it takes the pause over rather than being refused, which keeps the
# project's quiesce-and-drain semantics: the in-flight apply is never aborted
# mid-operation, it just loses the right to resume. The supersession is recorded
# in ``pause_collision`` so the collision is legible after the fact instead of
# being a silent no-op.
#
# These fields are out-of-band control state: they must appear in the autopilot
# daemon's ``_EXTERNAL_CONTROL_FIELDS`` so a trial-end whole-file save merges
# rather than drops them. They are written as explicit ``None`` (never popped) so
# that merge — which compares keys PRESENT on disk — always sees a release.

PAUSE_FIELD = "paused"
PAUSE_OWNER_FIELD = "pause_owner"
PAUSE_TOKEN_FIELD = "pause_token"
PAUSE_COLLISION_FIELD = "pause_collision"

#: Reserved owner for the operator CLI (``autopilot.py pause``). The operator
#: outranks every automated pauser: an automated lease can never clear a pause
#: this owner holds.
OPERATOR_PAUSE_OWNER = "operator"

PAUSE_LEASE_FIELDS = (PAUSE_OWNER_FIELD, PAUSE_TOKEN_FIELD, PAUSE_COLLISION_FIELD)


def claim_pause_lease(state: Dict[str, Any], owner: str) -> str:
    """Set ``paused=True`` on ``state`` and stamp a fresh lease. Returns the token.

    Mutates ``state`` in place; the caller must already hold ``state_write_lock``
    for the surrounding read-modify-write.
    """
    token = f"{owner}:{uuid.uuid4().hex}"
    state[PAUSE_FIELD] = True
    state[PAUSE_OWNER_FIELD] = owner
    state[PAUSE_TOKEN_FIELD] = token
    return token


def pause_lease_held(state: Dict[str, Any], token: Optional[str]) -> bool:
    """True when ``token`` is still the lease recorded in ``state``."""
    if not token:
        return False
    return state.get(PAUSE_TOKEN_FIELD) == token


def release_pause_lease(state: Dict[str, Any], token: Optional[str]) -> bool:
    """Clear the pause only if ``token`` still holds the lease.

    Returns True when the pause was cleared, False when the lease was superseded
    (or never held) — in which case ``state`` is left untouched and the caller
    MUST report the refusal rather than swallowing it.
    """
    if not pause_lease_held(state, token):
        return False
    state[PAUSE_FIELD] = False
    state[PAUSE_OWNER_FIELD] = None
    state[PAUSE_TOKEN_FIELD] = None
    return True


def supersede_pause_lease(state: Dict[str, Any], owner: str) -> Optional[Dict[str, Any]]:
    """Take the pause over for ``owner``, displacing any other live lease.

    Returns a collision record when an in-flight lease held by a DIFFERENT owner
    was displaced (also stored on ``state`` under ``pause_collision`` so it stays
    legible to ``autopilot status`` and the dashboard), else ``None``.
    """
    prior_owner = state.get(PAUSE_OWNER_FIELD)
    prior_token = state.get(PAUSE_TOKEN_FIELD)
    was_paused = bool(state.get(PAUSE_FIELD, False))
    collision: Optional[Dict[str, Any]] = None
    if prior_token and prior_owner != owner:
        collision = {
            "superseded_owner": prior_owner,
            "superseded_token": prior_token,
            "new_owner": owner,
            "was_paused": was_paused,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    claim_pause_lease(state, owner)
    state[PAUSE_COLLISION_FIELD] = collision
    return collision


def clear_pause_lease(state: Dict[str, Any]) -> None:
    """Unconditionally drop lease bookkeeping (resume path)."""
    state[PAUSE_OWNER_FIELD] = None
    state[PAUSE_TOKEN_FIELD] = None
    state[PAUSE_COLLISION_FIELD] = None
