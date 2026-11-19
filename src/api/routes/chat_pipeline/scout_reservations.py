"""Cross-process scout slot reservations (OAB-17 F8, multi-worker).

Production runs ``uvicorn --workers 6``: each worker is its own process, so an in-process
counter lets six workers read the same free /slots and together over-admit scouts past
``reserve_slots``. Reservations therefore live in one small JSON file per server URL under a
run dir, via the shared primitive `src.runtime.xproc_reservations` (flock per URL, owner
pid + start-ticks and TTL reaping, fail CLOSED with `ReservationUnavailable`: the caller
admits 0 scouts).

This module only binds the primitive to the scout namespace and keeps its historical API.
"""
from __future__ import annotations

import hashlib  # noqa: F401  (tests locate the per-URL file by the same hash)
from pathlib import Path
from typing import ContextManager

from src.runtime.xproc_reservations import (  # noqa: F401  (re-exported API)
    LOCK_TIMEOUT_S,
    ReservationStore,
    ReservationUnavailable,
    Slate as _Slate,
    proc_start_ticks,
)

DIR_ENV = "EPYC_SCOUT_RESERVE_DIR"
TTL_ENV = "EPYC_SCOUT_RESERVE_TTL_S"
DEFAULT_TTL_S = 900.0

_STORE = ReservationStore("scout-reserve", dir_env=DIR_ENV, ttl_env=TTL_ENV,
                          default_ttl_s=DEFAULT_TTL_S)


def reserve_dir() -> Path:
    return _STORE.reserve_dir()


def locked(url: str, *, timeout: float = LOCK_TIMEOUT_S) -> ContextManager[_Slate]:
    """Exclusive, reaped view of ``url``'s reservations. Raises `ReservationUnavailable`."""
    return _STORE.locked(url, timeout=timeout)


def release(url: str | None, token: str | None) -> bool:
    """Drop one reservation by token. Never raises; False if the store was unavailable."""
    return _STORE.release(url, token)
