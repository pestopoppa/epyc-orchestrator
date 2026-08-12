"""Ownership of ``autopilot_state.json`` fields — declared as data, enforced at runtime.

WHAT IS OWNED BY WHOM
---------------------
``autopilot_state.json`` is a whole-file JSON document rewritten by several
independent processes. Its fields fall into two disjoint classes:

* **out-of-band control fields** (``paused``, ``pause_reason``, ``_in_cache_flush``
  and the pause-lease keys) — written by the dashboard, ``host_health``,
  ``config_applicator`` and the operator CLI *while the daemon is running*, and
  merged back by the daemon under the write lock. See
  ``autopilot._EXTERNAL_CONTROL_FIELDS``.
* **daemon-owned fields** (``DAEMON_OWNED_STATE_FIELDS`` below) — the counters,
  baselines, quality histories and Pareto cache that the AutoPilot daemon holds in
  a long-lived in-memory dict across a whole trial and rewrites wholesale at every
  save. **An external write to one of these is silently destroyed at the daemon's
  next save.**

ORIGIN (2026-08-03)
-------------------
``operator_seed_e8_operational_baseline.py`` wrote a fresh ``baseline_state`` from
a 59-minute T1 calibration, printed ``APPLIED``, and the daemon's next
``save_state(merge_control=True)`` rewrote the field from its own memory. The
measurement was gone. The cross-process write lock did **not** help: it serialises
writes, it does not stop a later writer from persisting a stale in-memory
snapshot. Only daemon *absence* makes an external write durable.

Until this module existed the prohibition was PROSE — a comment above
``_EXTERNAL_CONTROL_FIELDS`` and a hand-rolled ``pgrep`` gate inside exactly one
operator script. Prose binds nothing, and a gate living inside the file being
copied is patched out by copying the file (INC 2026-08-05, see
``scripts/hooks/check_operator_apply_copy.sh`` in epyc-root).

TWO LAYERS, DELIBERATELY DIFFERENT IN REMOVABILITY
--------------------------------------------------
**Layer 1 — writer-side refusal** (:func:`assert_external_write_allowed`).
Keyed on DERIVATION, never on a self-declared role: the AutoPilot daemon is, by
construction, the process holding the exclusive ``flock`` on
``orchestration/.autopilot.lock`` (``autopilot.cmd_start``). ``/proc/locks`` names
that holder's PID, so "am I the owner?" is answered by the kernel rather than by
anything the caller asserts about itself. A caller that is not the holder, while
somebody else is, and that is changing a daemon-owned field, is *definitionally*
about to lose that write — so it is refused. This layer lives in the caller's
path and a determined violator can route around it; hence layer 2.

**Layer 2 — victim-side detection** (:func:`quarantine_clobbered_fields`).
Runs inside the daemon's own save. Its witness is the divergence itself: a
daemon-owned field whose on-disk value differs from what *this process last wrote*
was changed by somebody else, whatever wrote it and however it wrote. That
comparison lives in the victim, not the violator, so there is nothing for a
violator to patch out. It cannot prevent the overwrite (refusing the daemon's own
save would strand the daemon's trial state and is a worse failure), but it
QUARANTINES the on-disk values to a sidecar before they are destroyed and logs at
ERROR — turning an unrecoverable silent loss into a recoverable, attributed one.

WHY LAYER 2 NEEDS A LAST-WRITTEN BASELINE. "disk differs from memory" is NOT
evidence of an external write: the daemon's own in-memory dict legitimately
differs from disk on every save (it just incremented ``trial_counter``). Only
"disk differs from what *I* last wrote" isolates a third-party write, because
every daemon write goes through :func:`record_write`. Digests are kept, not
values, so the baseline is O(fields) regardless of state size.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not read, scan or pattern-match any source text, so documentation, tests,
handoffs and comments naming these fields are unaffected by construction — the
mechanism is a runtime ownership question, not a spelling. And the daemon writing
its own state is the first branch of layer 1, never a violation.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("autopilot")


#: Fields the AutoPilot daemon owns: it holds them in memory across a whole trial
#: and rewrites them wholesale at every save. An external write to any of these
#: while the daemon lives is destroyed at the daemon's next save.
DAEMON_OWNED_STATE_FIELDS: tuple[str, ...] = (
    "baseline_state",
    "quality_history",
    "quality_history_by_tier",
    "quality_history_provenance_by_tier",
    "pareto_archive",
    "trial_counter",
    "in_flight_trial",
    "consecutive_failures",
    "consecutive_meta_actions",
    "consecutive_skip_actions",
    "last_invalid_action",
    "last_invalid_reason",
    "last_invalid_status",
    "species_budget",
    "seeder_state",
)

#: Every field an external process is allowed to write while the daemon runs.
#: Mirrors ``autopilot._EXTERNAL_CONTROL_FIELDS`` (+ ``state_lock.PAUSE_LEASE_FIELDS``);
#: the two sets MUST stay disjoint from ``DAEMON_OWNED_STATE_FIELDS`` or the guard
#: would forbid the very idiom it exists to protect. Pinned by the test suite.
KNOWN_EXTERNAL_CONTROL_FIELDS: tuple[str, ...] = (
    "paused",
    "pause_reason",
    "_in_cache_flush",
    "pause_owner",
    "pause_token",
    "pause_collision",
)

#: ``_dispatch_deficiency`` value the skip-action circuit-breaker latches.
SKIP_LOOP_DEFICIENCY = "skip_action_loop"

#: Halt annotations written next to the latched counters. Not daemon-OWNED in the
#: registry sense (the daemon pops rather than rewrites them), but they live in the
#: same in-memory dict, so they are cleared together with the counters or not at all.
HALT_LATCH_MARKER_FIELDS: tuple[str, ...] = ("_dispatch_deficiency", "_meta_halt_reason")

#: What "clear the halt latch" means, per breaker. Both sets are daemon-owned:
#: they are streak counters and planner-feedback the daemon derives from its own
#: trial stream, so only a writer that owns the daemon's memory can clear them
#: durably. See :func:`clear_halt_latch`.
_SKIP_LATCH_CLEARED: dict[str, Any] = {
    "consecutive_skip_actions": 0,
    "last_invalid_action": None,
    "last_invalid_reason": None,
    "last_invalid_status": None,
}
_META_LATCH_CLEARED: dict[str, Any] = {"consecutive_meta_actions": 0}

#: Name of the AutoPilot singleton lock, resolved next to the state file exactly as
#: ``autopilot.LOCK_PATH`` sits next to ``autopilot.STATE_PATH``.
DAEMON_LOCK_NAME = ".autopilot.lock"

#: Key stamped into the state the daemon is about to write when layer 2 fires.
CONFLICT_MARKER_FIELD = "_daemon_owned_external_write_conflict"

_PROC_LOCKS = Path("/proc/locks")
_DEVINO_RE = re.compile(r"^[0-9a-f]+:[0-9a-f]+:\d+$")
_MISSING = object()

# Per-path digests of the daemon-owned subset THIS PROCESS last wrote. See the
# module docstring: without it, "disk differs from memory" cannot distinguish the
# daemon's own pending change from a third party's committed one.
_LAST_WRITTEN_DIGESTS: dict[str, dict[str, str]] = {}


class DaemonOwnedStateWriteError(RuntimeError):
    """An external process tried to write daemon-owned state while the daemon lives."""


# ── plumbing ────────────────────────────────────────────────────────────────


def daemon_lock_path_for(state_path: str | os.PathLike[str]) -> Path:
    """Return the AutoPilot singleton lock that governs ``state_path``."""
    return Path(state_path).parent / DAEMON_LOCK_NAME


def read_state_file(state_path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return the on-disk state mapping, or ``None`` when absent/unreadable/corrupt.

    ``None`` means "there is nothing on disk to lose", which is why every caller
    treats it as permission to proceed. Corrupt-state refusal is
    ``state_store.load_state``'s job and is deliberately not duplicated here.
    """
    try:
        loaded = json.loads(Path(state_path).read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _digest(value: Any) -> str:
    if value is _MISSING:
        return "\x00missing"
    try:
        payload = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def _digests(state: dict[str, Any], fields: Iterable[str]) -> dict[str, str]:
    return {field: _digest(state.get(field, _MISSING)) for field in fields}


# ── who owns the daemon lock (kernel-attested, not self-declared) ────────────


def daemon_lock_holder_pids(lock_path: str | os.PathLike[str]) -> set[int] | None:
    """PIDs holding an advisory lock on ``lock_path``, or ``None`` if undeterminable.

    Reads ``/proc/locks``, which names the holding PID — the only way a process can
    ask "is the lock mine?" about an ``flock``, since a second fd on the same file
    conflicts with its own process just as it would with any other.

    An empty set means the lock file exists but nobody holds it (daemon absent).
    ``None`` means ``/proc/locks`` could not be read or the lock file does not
    exist; callers degrade rather than guess.
    """
    path = Path(lock_path)
    try:
        stat = path.stat()
    except OSError:
        return None
    want = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    try:
        raw = _PROC_LOCKS.read_text()
    except OSError:
        return None

    pids: set[int] = set()
    for line in raw.splitlines():
        parts = line.split()
        for index, token in enumerate(parts):
            if index == 0 or not _DEVINO_RE.match(token):
                continue
            if token != want:
                break
            try:
                pid = int(parts[index - 1])
            except (IndexError, ValueError):
                break
            if pid > 0:
                pids.add(pid)
            break
    return pids


def _flock_probe_held(lock_path: str | os.PathLike[str]) -> bool | None:
    """Fallback liveness probe: can a shared lock be taken right now?

    Used only when ``/proc/locks`` is unavailable. Answers "is somebody holding
    it", never "is it me" — a shared probe is granted unless an exclusive holder
    exists, and is released within microseconds.
    """
    try:
        fd = os.open(os.fspath(lock_path), os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
            return True
        return None
    else:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        return False
    finally:
        os.close(fd)


def process_is_daemon(lock_path: str | os.PathLike[str]) -> bool | None:
    """True when THIS process holds the AutoPilot singleton lock. ``None`` if unknown."""
    pids = daemon_lock_holder_pids(lock_path)
    if pids is None:
        return None
    return os.getpid() in pids


# ── the ownership question ──────────────────────────────────────────────────


def daemon_owned_fields_changed(
    new_state: dict[str, Any],
    disk_state: dict[str, Any],
) -> list[str]:
    """Daemon-owned fields whose value ``new_state`` would change on disk.

    Presence matters as much as value: adding or dropping a daemon-owned key is a
    change. A writer that read the file under the lock and touched only control
    fields produces an empty list and is never refused.
    """
    if not isinstance(disk_state, dict) or not isinstance(new_state, dict):
        return []
    return [
        field
        for field in DAEMON_OWNED_STATE_FIELDS
        if new_state.get(field, _MISSING) != disk_state.get(field, _MISSING)
    ]


def externally_written_fields(
    state_path: str | os.PathLike[str],
    disk_state: dict[str, Any],
) -> list[str]:
    """Daemon-owned fields on disk that changed since THIS process last wrote them.

    Empty when this process has not written yet (no baseline to compare against) —
    a fresh daemon has just loaded from disk, so its first save has nothing to
    detect. Every later save is covered.
    """
    baseline = _LAST_WRITTEN_DIGESTS.get(str(Path(state_path).resolve()))
    if baseline is None or not isinstance(disk_state, dict):
        return []
    return [
        field
        for field in DAEMON_OWNED_STATE_FIELDS
        if _digest(disk_state.get(field, _MISSING)) != baseline.get(field)
    ]


def record_write(state_path: str | os.PathLike[str], state: dict[str, Any]) -> None:
    """Remember the daemon-owned subset just written, as digests. Called after a write."""
    _LAST_WRITTEN_DIGESTS[str(Path(state_path).resolve())] = _digests(
        state, DAEMON_OWNED_STATE_FIELDS
    )


def forget_writes(state_path: str | os.PathLike[str] | None = None) -> None:
    """Drop the last-written baseline (all paths when ``state_path`` is None)."""
    if state_path is None:
        _LAST_WRITTEN_DIGESTS.clear()
    else:
        _LAST_WRITTEN_DIGESTS.pop(str(Path(state_path).resolve()), None)


# ── layer 1: writer-side refusal ────────────────────────────────────────────


def assert_external_write_allowed(
    state_path: str | os.PathLike[str],
    new_state: dict[str, Any],
    *,
    disk_state: dict[str, Any] | None = None,
) -> None:
    """Refuse a non-daemon write of daemon-owned fields while the daemon holds the lock.

    Allows, in order: the daemon itself; a state file with nothing on disk to lose;
    a write that changes no daemon-owned field; a host where the daemon is absent.
    Raises :class:`DaemonOwnedStateWriteError` otherwise.
    """
    if disk_state is None:
        disk_state = read_state_file(state_path)
    if disk_state is None:
        return

    changed = daemon_owned_fields_changed(new_state, disk_state)
    if not changed:
        return

    lock_path = daemon_lock_path_for(state_path)
    holders = daemon_lock_holder_pids(lock_path)
    if holders is None:
        held = _flock_probe_held(lock_path)
        if held:
            # The lock is held but /proc/locks could not attribute it, so we cannot
            # tell the daemon apart from a violator. Refusing here would strand the
            # daemon's own save; layer 2 still catches the loss and names it.
            log.warning(
                "state ownership: %s is held but /proc/locks is unreadable — cannot "
                "attribute the holder; allowing the write to %s and relying on the "
                "daemon-side detector. Fields at risk: %s",
                lock_path,
                state_path,
                ", ".join(changed),
            )
        return

    if os.getpid() in holders:
        return
    if not holders:
        return

    raise DaemonOwnedStateWriteError(
        "refusing to write daemon-owned AutoPilot state while the daemon is running.\n"
        f"  state file    : {state_path}\n"
        f"  daemon lock   : {lock_path} (held by pid {sorted(holders)})\n"
        f"  fields at risk: {', '.join(changed)}\n"
        "These fields live in the daemon's in-memory dict for the whole of a trial "
        "and are rewritten wholesale at its next save, so this write would be "
        "silently destroyed — that is how a 59-minute calibration was lost on "
        "2026-08-03 after printing APPLIED. The cross-process write lock does not "
        "help: it orders writes, it does not stop a stale snapshot landing later.\n"
        "Stop the daemon (supervisor first, then the child), verify both are gone, "
        "write, then restart it — a fresh daemon loads this file from disk."
    )


# ── layer 2: victim-side detection and quarantine ───────────────────────────


def quarantine_clobbered_fields(
    state_path: str | os.PathLike[str],
    new_state: dict[str, Any],
    disk_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Preserve daemon-owned values this save is about to destroy. Returns the record.

    Fires only for fields that changed on disk since this process last wrote them,
    i.e. by somebody else. Writes a sidecar next to the state file, logs at ERROR,
    and stamps ``CONFLICT_MARKER_FIELD`` into ``new_state`` so the loss is visible
    to whatever reads the state next. Returns ``None`` when there is nothing to
    preserve.
    """
    if disk_state is None:
        return None
    fields = externally_written_fields(state_path, disk_state)
    if not fields:
        return None

    detected_at = datetime.now(timezone.utc).isoformat()
    path = Path(state_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine_path = path.with_name(f"{path.name}.external-write-{stamp}.json")
    record = {
        "detected_at": detected_at,
        "state_path": str(path),
        "detected_by_pid": os.getpid(),
        "fields": {
            field: {
                "on_disk_destroyed": disk_state.get(field),
                "written_instead": new_state.get(field),
            }
            for field in fields
        },
    }
    try:
        quarantine_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        saved_to: str | None = str(quarantine_path)
    except OSError as exc:
        saved_to = None
        log.error("state ownership: could not write quarantine sidecar: %s", exc)

    log.error(
        "state ownership: another process wrote daemon-owned field(s) %s in %s since "
        "this process last wrote them; this save destroys that write. The destroyed "
        "values are preserved at %s. Daemon-owned state must only be written with the "
        "AutoPilot daemon stopped.",
        ", ".join(fields),
        path,
        saved_to or "<sidecar write failed>",
    )
    new_state[CONFLICT_MARKER_FIELD] = {
        "detected_at": detected_at,
        "fields": fields,
        "quarantine_path": saved_to,
    }
    return record


# ── single entry point used by the writer ───────────────────────────────────


def enforce_state_write(
    state_path: str | os.PathLike[str],
    new_state: dict[str, Any],
) -> dict[str, Any] | None:
    """Apply both layers immediately before a whole-file state write.

    Raises :class:`DaemonOwnedStateWriteError` for a refused external write.
    Returns the quarantine record when layer 2 fired, else ``None``.
    """
    disk_state = read_state_file(state_path)
    assert_external_write_allowed(state_path, new_state, disk_state=disk_state)
    return quarantine_clobbered_fields(state_path, new_state, disk_state)


# ── the halt latch: clearing it is DELEGATED, never faked ───────────────────
#
# A ``skip_action_loop`` / ``meta_action_loop`` circuit-breaker latches by setting
# ``paused=True`` plus a streak counter and a deficiency marker. Resuming has to
# clear the latch too, or the very next non-executing action re-trips the breaker
# (``skip_streak >= MAX_CONSECUTIVE_SKIP``) and the halt "comes back".
#
# WHY THE COUNTERS ARE NOT CONTROL STATE. ``paused`` is safe for an external
# process to write because the daemon's copy is not authoritative — the daemon
# READS it as a command at every iteration top. ``consecutive_skip_actions`` is the
# opposite: the daemon derives it from its own trial stream, holds the authoritative
# copy in memory for the whole run, and rewrites it wholesale. Putting it in the
# merge set would invert that authority — ``_merge_external_control_fields`` copies
# disk→memory before every merged save, so the daemon's freshly incremented streak
# would be reverted to the value it last persisted and the breaker would never fire
# again. The circuit-breaker cannot be armed by a field any of 5+ writers may rewind.
#
# So the clearing is daemon work. An external resume may perform it only when the
# daemon is ABSENT (nothing in memory to overwrite it) — exactly the rule layer 1
# already enforces — and must otherwise say it delegated, never claim success.


def halt_latch_updates(state: dict[str, Any]) -> dict[str, Any]:
    """Field values a resume must install to clear whatever halt latch is set.

    The skip-loop counters are cleared only under their own deficiency marker (the
    streak is real evidence about the planner and a resume for an unrelated reason
    must not silently rewind a 3-of-4 streak); the meta counter is cleared on any
    resume, matching the daemon's behaviour since 2026-05-31.
    """
    updates: dict[str, Any] = {}
    if state.get("_dispatch_deficiency") == SKIP_LOOP_DEFICIENCY:
        updates.update(_SKIP_LATCH_CLEARED)
    updates.update(_META_LATCH_CLEARED)
    return updates


def halt_latch_pending(state: dict[str, Any]) -> list[str]:
    """Keys a resume would actually have to change. Empty means nothing is latched."""
    pending = [
        field
        for field, cleared in halt_latch_updates(state).items()
        if state.get(field, _MISSING) != cleared
    ]
    pending.extend(marker for marker in HALT_LATCH_MARKER_FIELDS if marker in state)
    return pending


def clear_halt_latch(
    state_path: str | os.PathLike[str],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Clear the halt latch in ``state`` — but only where that write would survive.

    Mutates ``state`` in place and returns ``outcome="cleared"`` when this process
    owns the write (it is the daemon, or the daemon is absent). When the daemon
    holds the lock the latch fields are left ALONE and the outcome is
    ``"delegated_to_daemon"``: the daemon clears them from its own memory on the
    paused True→False edge, which is the only clearing that lasts. ``"noop"`` when
    no latch was set.

    The caller must report the outcome. Mutating nothing and reporting success is
    the defect this function exists to remove.
    """
    deficiency = state.get("_dispatch_deficiency")
    pending = halt_latch_pending(state)
    if not pending:
        return {
            "outcome": "noop",
            "deficiency": deficiency,
            "fields": [],
            "detail": "no halt latch was set",
        }

    updates = halt_latch_updates(state)
    candidate = dict(state)
    candidate.update(updates)
    for marker in HALT_LATCH_MARKER_FIELDS:
        candidate.pop(marker, None)

    try:
        assert_external_write_allowed(state_path, candidate)
    except DaemonOwnedStateWriteError:
        return {
            "outcome": "delegated_to_daemon",
            "deficiency": deficiency,
            "fields": pending,
            "detail": (
                "the AutoPilot daemon holds these counters in memory, so clearing "
                "them from here would be destroyed at its next save; the daemon "
                "clears the latch itself when it observes the resume"
            ),
        }

    state.update(updates)
    for marker in HALT_LATCH_MARKER_FIELDS:
        state.pop(marker, None)
    return {
        "outcome": "cleared",
        "deficiency": deficiency,
        "fields": pending,
        "detail": "halt latch cleared by this writer",
    }


def halt_latch_message(latch: dict[str, Any] | None) -> str:
    """One operator-facing sentence for a :func:`clear_halt_latch` outcome."""
    if not latch or latch.get("outcome") == "noop":
        return ""
    deficiency = latch.get("deficiency") or "halt"
    if latch.get("outcome") == "cleared":
        return f"{deficiency} latch cleared"
    return (
        f"{deficiency} latch NOT cleared here — the AutoPilot daemon owns "
        f"{', '.join(latch.get('fields') or [])} and clears them when it picks up "
        "the resume. Re-check the dispatch deficiency; if it is still set after the "
        "daemon's next poll, stop the daemon and resume again."
    )
