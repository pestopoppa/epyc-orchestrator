"""KPF-21: the prefill-only trunk request of a trunk-first fan-out.

The gate (``kv_pool_admission``, ``ORCHESTRATOR_PREFIX_INDEX_FORK``) already
orders a fan-out trunk-first: the earliest sibling is the trunk owner and the
others are held until its ``prefill_done``. Without this module the owner's
trunk prefill is its own full request, so the held siblings are released only
at the owner's FIRST OUTPUT chunk and the server's fork source is whatever the
owner's slot holds then.

With the RTG-58 P1 server (``llama.cpp`` branch ``experimental/kv-prefix-fork``)
the orchestrator can do better: issue the trunk ONCE as a prefill-only request
(``max_tokens: 0``) that pins a checkpoint at the junction
(``checkpoint_at: [{"message": k, "at": "end"}]`` on chat endpoints, or
``[len(trunk)]`` token positions elsewhere), and release the children when that
prefill has completed (INTERFACE.md section 2, "Trunk-first recipe").

This module owns the neutral parts: the capability test and a once-per-trunk
registry (one issuer, the other siblings wait for it or skip). The transport
that actually sends the request lives with the fan-out site
(``scout_stage.ChatCompletionsTransport.prefill_trunk``).

Everything here is behind ``ORCHESTRATOR_PREFIX_INDEX`` AND
``ORCHESTRATOR_PREFIX_INDEX_FORK`` and runs only when the server's
``/props.slot_fork`` advertises the fork. Flag off: nothing in this module is
imported by a fan-out site's hot path beyond one ``prefix_index.fork_requested``
check, and no request is added or changed.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: How long a finished trunk prefill suppresses a repeat for the same trunk.
DONE_TTL_S = 60.0
#: How long a sibling waits for the issuer's prefill before going on without it.
WAIT_S = 120.0

_DONE, _FAILED, _RUNNING = "done", "failed", "running"


def trunk_key(url: str, trunk_text: str) -> str:
    """Identity of a trunk on a server: the url and the trunk's exact bytes."""
    h = hashlib.sha256(trunk_text.encode("utf-8", "surrogatepass")).hexdigest()[:24]
    return f"{url.rstrip('/')}#{h}"


def capability(caps: dict[str, Any] | None, trunk_tokens_est: float) -> str | None:
    """How to send the prefill-only trunk request on this server, or None.

    ``"checkpoint"``: the hybrid/SWA path, send ``checkpoint_at`` (needs the
    server to advertise it). ``"kv"``: a pure-attention server, the plain
    ``max_tokens: 0`` request suffices (INTERFACE.md section 2: "On kv-mode
    models step 1 needs no checkpoint_at"). None: flag off, no capability
    (v10 server, fork configured off, ``/props`` unreadable), or a trunk below
    the server's ``min_tokens`` (no fork would happen)."""
    from src.inference import prefix_index

    if not prefix_index.fork_requested() or not isinstance(caps, dict):
        return None
    if not prefix_index.fork_enabled(caps):
        return None
    if trunk_tokens_est < max(1, int(caps.get("min_tokens") or 0)):
        return None
    mode = caps.get("mode")
    if mode == "checkpoint":
        return "checkpoint" if caps.get("checkpoint_at") is True else None
    return "kv" if mode == "kv" else None


class _Slot:
    __slots__ = ("state", "event", "ts", "info")

    def __init__(self) -> None:
        self.state = _RUNNING
        self.event = threading.Event()
        self.ts = time.monotonic()
        self.info: dict[str, Any] = {}


class TrunkPrefillRegistry:
    """One prefill-only trunk request per (server, trunk), however many
    siblings ask. Thread-safe; the lock is a leaf (``issue`` runs outside it)."""

    def __init__(self, *, done_ttl_s: float = DONE_TTL_S, wait_s: float = WAIT_S,
                 clock: Callable[[], float] = time.monotonic):
        self._lock = threading.Lock()
        self._slots: dict[str, _Slot] = {}
        self._done_ttl_s = done_ttl_s
        self._wait_s = wait_s
        self._clock = clock
        self.stats = {"issued": 0, "failed": 0, "reused": 0, "waited": 0, "wait_timeouts": 0}

    def reset(self) -> None:
        with self._lock:
            self._slots.clear()
            for k in self.stats:
                self.stats[k] = 0

    def ensure(self, key: str, issue: Callable[[], dict[str, Any] | None],
               *, wait_s: float | None = None) -> dict[str, Any]:
        """Run ``issue`` (the prefill-only request) unless this trunk was just
        issued. Returns ``{"outcome": issued|failed|reused|waited|wait_timeout, ...}``.
        The issuer's ``issue`` must return once the server's prefill completed;
        it may raise (the outcome is ``failed``; siblings proceed without it and
        are NOT retried within the TTL, so a broken endpoint costs one attempt)."""
        now = self._clock()
        with self._lock:
            for k in [k for k, s in self._slots.items()
                      if s.state != _RUNNING and now - s.ts > self._done_ttl_s]:
                self._slots.pop(k, None)
            slot = self._slots.get(key)
            mine = slot is None
            if mine:
                slot = self._slots[key] = _Slot()
        if not mine:
            if slot.state == _RUNNING:
                with self._lock:
                    self.stats["waited"] += 1
                if not slot.event.wait(self._wait_s if wait_s is None else wait_s):
                    with self._lock:
                        self.stats["wait_timeouts"] += 1
                    return {"outcome": "wait_timeout"}
                return {"outcome": "waited", "state": slot.state, **slot.info}
            with self._lock:
                self.stats["reused"] += 1
            return {"outcome": "reused", "state": slot.state, **slot.info}
        try:
            info = issue() or {}
            slot.info = dict(info)
            slot.state = _DONE
            with self._lock:
                self.stats["issued"] += 1
            return {"outcome": "issued", "state": _DONE, **slot.info}
        except Exception as exc:  # noqa: BLE001 -- a trunk request never fails a fan-out
            logger.warning("trunk prefill failed (%s): %s", key, exc)
            slot.info = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            slot.state = _FAILED
            with self._lock:
                self.stats["failed"] += 1
            return {"outcome": "failed", "state": _FAILED, **slot.info}
        finally:
            slot.ts = self._clock()
            slot.event.set()


_registry = TrunkPrefillRegistry()


def get_registry() -> TrunkPrefillRegistry:
    return _registry
