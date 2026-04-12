"""Concurrency-aware backend for pre-warm NUMA deployments.

Routes single sessions to the full-speed (1×96t) instance for maximum
per-request throughput. When concurrent requests arrive, migrates KV state
from the full instance to a quarter (48t) instance and routes new requests
to idle quarters.

Pre-warm architecture:
    - 1 full-speed instance (96t, node-pinned) — best single-session speed
    - 4 quarter instances (48t each, NUMA-quarter-pinned) — concurrent slots

The full instance is ALWAYS running (weights in RAM, mlocked). Quarter
instances are ALWAYS running too. The only dynamic operation is KV state
save/restore on transition, using llama.cpp's slot save/restore API.

KV migration flow (Phase D):
    1. Session A starts → routes to full (96t) instance
    2. Session B arrives while Session A between turns (full idle)
       → Save A's KV from full (POST /slots/0?action=save)
       → Restore A's KV on quarter 0 (POST /slots/0?action=restore)
       → Route A's next turn to quarter 0
       → Route B to full instance (fresh, max speed)
    3. Session A completes → quarter 0 freed
    4. Only one session left → next turn goes back to full (max speed)

Usage:
    full_backend = CachingBackend(srv_96t, ...)
    quarter_backends = [CachingBackend(srv_48t_0, ...), ...]
    ca = ConcurrencyAwareBackend(full_backend, quarter_backends, role="frontdoor")
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Slot save/restore timeout — KV state can be 2-16 GB for production conversations
_SLOT_SAVE_TIMEOUT = 30.0  # seconds
_SLOT_RESTORE_TIMEOUT = 30.0


def _get_base_url(backend: Any) -> str | None:
    """Extract the base URL from a CachingBackend or LlamaServerBackend."""
    # CachingBackend wraps LlamaServerBackend
    inner = getattr(backend, "_backend", backend)
    config = getattr(inner, "config", None)
    if config:
        return getattr(config, "base_url", None)
    return None


def _slot_save(base_url: str, slot_id: int = 0) -> bool:
    """Save KV state from a llama-server slot. Returns True on success."""
    if not _HTTPX_AVAILABLE:
        return False
    try:
        url = f"{base_url}/slots/{slot_id}?action=save"
        resp = httpx.post(url, timeout=_SLOT_SAVE_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(
                "KV save: slot %d, %d tokens, %.1fms",
                slot_id, data.get("n_saved", 0), data.get("timings", {}).get("save_ms", 0),
            )
            return True
        logger.warning("KV save failed: HTTP %d from %s", resp.status_code, url)
        return False
    except Exception as exc:
        logger.debug("KV save failed: %s", exc)
        return False


def _slot_restore(base_url: str, slot_id: int = 0) -> bool:
    """Restore KV state to a llama-server slot. Returns True on success."""
    if not _HTTPX_AVAILABLE:
        return False
    try:
        url = f"{base_url}/slots/{slot_id}?action=restore"
        resp = httpx.post(url, timeout=_SLOT_RESTORE_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(
                "KV restore: slot %d, %d tokens, %.1fms",
                slot_id, data.get("n_restored", 0), data.get("timings", {}).get("restore_ms", 0),
            )
            return True
        logger.warning("KV restore failed: HTTP %d from %s", resp.status_code, url)
        return False
    except Exception as exc:
        logger.debug("KV restore failed: %s", exc)
        return False


def _slot_erase(base_url: str, slot_id: int = 0) -> bool:
    """Erase KV state from a llama-server slot."""
    if not _HTTPX_AVAILABLE:
        return False
    try:
        resp = httpx.post(f"{base_url}/slots/{slot_id}?action=erase", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


class ConcurrencyAwareBackend:
    """Routes requests between full-speed and quarter instances based on load.

    Single active request  → full-speed instance (max per-request throughput)
    Multiple active requests → quarter instances (max concurrency)

    KV state migration (Phase D): When the full instance is idle between turns
    and a new session arrives, the existing session's KV state is migrated from
    the full instance to a quarter instance. This is best-effort — if migration
    fails, the quarter starts cold (no KV state), which is functionally correct
    but loses prefix cache benefit.
    """

    def __init__(
        self,
        full_backend: Any,
        quarter_backends: list[Any],
        role: str = "",
        full_port: int = 0,
    ):
        if not quarter_backends:
            raise ValueError("ConcurrencyAwareBackend requires at least one quarter backend")
        self._full = full_backend
        self._quarters = quarter_backends
        self._role = role
        self._full_port = full_port
        self._lock = threading.Lock()

        # Extract base URLs for slot API calls
        self._full_url = _get_base_url(full_backend)
        self._quarter_urls = [_get_base_url(q) for q in quarter_backends]

        # Tracking state
        self._full_active = False
        self._quarter_active: list[bool] = [False] * len(quarter_backends)
        self._total_requests = 0
        self._full_requests = 0
        self._quarter_requests = 0
        self._migrations = 0
        self._migration_failures = 0

        # Session affinity: track which session was last on the full instance
        # so we can migrate its KV state to a quarter on concurrent arrival.
        self._full_last_session: str | None = None

        # Migration tracking: session_id → quarter_idx
        # When a session migrates from full to quarter, record it here.
        # Next request from this session goes to its assigned quarter.
        self._session_quarter: dict[str, int] = {}

        logger.info(
            "ConcurrencyAwareBackend[%s]: 1 full (%s) + %d quarters, KV migration %s",
            role or "unknown",
            self._full_url or "?",
            len(quarter_backends),
            "enabled" if _HTTPX_AVAILABLE else "disabled (no httpx)",
        )

    def _select(self, session_id: str = "") -> tuple[Any, int, bool]:
        """Select the best backend for the next request.

        Args:
            session_id: Optional session identifier for affinity routing.

        Returns (backend, index, is_full) where index is:
            -1 for full instance
            0..N for quarter instances
        """
        with self._lock:
            self._total_requests += 1

            # Check session affinity: if this session was migrated to a quarter,
            # route it back to that quarter (preserves KV state from migration).
            if session_id and session_id in self._session_quarter:
                q_idx = self._session_quarter[session_id]
                if 0 <= q_idx < len(self._quarters):
                    self._quarter_active[q_idx] = True
                    self._quarter_requests += 1
                    return self._quarters[q_idx], q_idx, False

            # If full instance is idle, use it (best per-request speed)
            if not self._full_active:
                # If there was a previous session on full and we're a NEW session,
                # we need to migrate the previous session's KV to a quarter first.
                if (
                    self._full_last_session
                    and session_id
                    and session_id != self._full_last_session
                    and self._full_last_session not in self._session_quarter
                ):
                    # Find an idle quarter for the migration target
                    migrate_target = None
                    for i, active in enumerate(self._quarter_active):
                        if not active:
                            migrate_target = i
                            break

                    if migrate_target is not None:
                        # Schedule migration (non-blocking — happens before we
                        # return the full instance to the new session)
                        old_session = self._full_last_session
                        self._session_quarter[old_session] = migrate_target
                        self._migrations += 1
                        # Release lock for I/O, then do migration
                        self._full_active = True
                        self._full_requests += 1
                        self._full_last_session = session_id

                        # Do KV migration outside lock
                        threading.Thread(
                            target=self._migrate_kv,
                            args=(migrate_target,),
                            daemon=True,
                            name=f"kv-migrate-{self._role}-{old_session[:8]}",
                        ).start()

                        return self._full, -1, True

                self._full_active = True
                self._full_requests += 1
                if session_id:
                    self._full_last_session = session_id
                return self._full, -1, True

            # Full is busy — find an idle quarter
            for i, active in enumerate(self._quarter_active):
                if not active:
                    self._quarter_active[i] = True
                    self._quarter_requests += 1
                    return self._quarters[i], i, False

            # All quarters busy — overflow to least-recently-used quarter
            idx = self._quarter_requests % len(self._quarters)
            self._quarter_active[idx] = True
            self._quarter_requests += 1
            logger.warning(
                "All %s instances busy (%d quarters), overflow to quarter %d",
                self._role, len(self._quarters), idx,
            )
            return self._quarters[idx], idx, False

    def _migrate_kv(self, target_quarter: int) -> None:
        """Migrate KV state from full instance to a quarter (background thread).

        Best-effort: if save or restore fails, the quarter starts cold.
        """
        if not self._full_url:
            return
        target_url = self._quarter_urls[target_quarter] if target_quarter < len(self._quarter_urls) else None
        if not target_url:
            return

        t0 = time.monotonic()
        saved = _slot_save(self._full_url)
        if not saved:
            self._migration_failures += 1
            logger.warning("KV migration save failed for %s, quarter %d starts cold", self._role, target_quarter)
            return

        restored = _slot_restore(target_url)
        if not restored:
            self._migration_failures += 1
            logger.warning("KV migration restore failed for %s quarter %d", self._role, target_quarter)
            return

        # Erase KV from full instance (it now belongs to the quarter)
        _slot_erase(self._full_url)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "KV migration complete: %s full → quarter %d (%.0fms)",
            self._role, target_quarter, elapsed_ms,
        )

    def _release(self, idx: int, is_full: bool) -> None:
        with self._lock:
            if is_full:
                self._full_active = False
            elif 0 <= idx < len(self._quarter_active):
                self._quarter_active[idx] = False

    def clear_session(self, session_id: str) -> None:
        """Remove session affinity (call when session completes)."""
        with self._lock:
            self._session_quarter.pop(session_id, None)
            if self._full_last_session == session_id:
                self._full_last_session = None

    # === Forward all backend interface methods ===
    # Note: session_id extraction from request is best-effort.
    # If request has no session_id, routing falls back to load-based selection.

    def _extract_session_id(self, request: Any) -> str:
        """Try to extract session_id from request for affinity routing."""
        if hasattr(request, "session_id"):
            return str(request.session_id or "")
        if hasattr(request, "task_id"):
            return str(request.task_id or "")
        return ""

    def infer(self, role_config: Any, request: Any) -> Any:
        sid = self._extract_session_id(request)
        backend, idx, is_full = self._select(session_id=sid)
        try:
            return backend.infer(role_config, request)
        finally:
            self._release(idx, is_full)

    def infer_streaming(self, role_config: Any, request: Any) -> Any:
        sid = self._extract_session_id(request)
        backend, idx, is_full = self._select(session_id=sid)
        try:
            return backend.infer_streaming(role_config, request)
        finally:
            self._release(idx, is_full)

    def infer_stream_text(self, role_config: Any, request: Any, on_chunk: Any = None) -> Any:
        sid = self._extract_session_id(request)
        backend, idx, is_full = self._select(session_id=sid)
        try:
            return backend.infer_stream_text(role_config, request, on_chunk=on_chunk)
        finally:
            self._release(idx, is_full)

    def health_check(self, pid: int = 0) -> bool:
        """Check health of full instance + all quarters."""
        full_ok = self._full.health_check(pid)
        quarters_ok = all(q.health_check(pid) for q in self._quarters)
        return full_ok and quarters_ok

    def get_stats(self) -> dict[str, Any]:
        """Telemetry for observability (DS-1 compatible)."""
        with self._lock:
            quarter_active = list(self._quarter_active)
            full_active = self._full_active
            session_map = dict(self._session_quarter)

        return {
            "role": self._role,
            "backend_type": "concurrency_aware",
            "full_instance": {
                "port": self._full_port,
                "active": full_active,
                "total_served": self._full_requests,
                "current_session": self._full_last_session,
            },
            "quarter_instances": len(self._quarters),
            "quarter_active": quarter_active,
            "total_active": (1 if full_active else 0) + sum(quarter_active),
            "idle_quarters": sum(1 for a in quarter_active if not a),
            "total_requests": self._total_requests,
            "full_requests": self._full_requests,
            "quarter_requests": self._quarter_requests,
            "migrations": self._migrations,
            "migration_failures": self._migration_failures,
            "session_affinity": session_map,
            "kv_migration_enabled": _HTTPX_AVAILABLE and bool(self._full_url),
        }

    # DS-6: Dynamic quarter management for QuarterScheduler

    def add_quarter(self, backend: Any) -> int:
        """Add a quarter backend instance. Returns the new quarter index.

        Thread-safe. The new quarter starts receiving requests immediately.
        """
        with self._lock:
            idx = len(self._quarters)
            self._quarters.append(backend)
            self._quarter_active.append(False)
            self._quarter_urls.append(_get_base_url(backend))
        logger.info(
            "ConcurrencyAware[%s]: added quarter %d (now %d quarters)",
            self._role, idx, len(self._quarters),
        )
        return idx

    def remove_quarter(self, idx: int) -> bool:
        """Remove a quarter backend by index. Returns False if index invalid.

        Thread-safe. Refuses removal if the quarter has active requests.
        Caller must drain traffic before calling this.
        Also cleans up any session affinity pointing to this quarter.
        """
        with self._lock:
            if idx < 0 or idx >= len(self._quarters):
                return False
            if self._quarter_active[idx]:
                logger.warning(
                    "ConcurrencyAware[%s]: refusing to remove active quarter %d",
                    self._role, idx,
                )
                return False
            self._quarters.pop(idx)
            self._quarter_active.pop(idx)
            self._quarter_urls.pop(idx)
            # Fix up session affinity: remove stale references, shift indices
            stale = [sid for sid, qidx in self._session_quarter.items() if qidx == idx]
            for sid in stale:
                del self._session_quarter[sid]
            for sid in list(self._session_quarter):
                if self._session_quarter[sid] > idx:
                    self._session_quarter[sid] -= 1
        logger.info(
            "ConcurrencyAware[%s]: removed quarter %d (now %d quarters)",
            self._role, idx, len(self._quarters),
        )
        return True

    def quarter_count(self) -> int:
        """Return current number of quarter instances."""
        with self._lock:
            return len(self._quarters)

    def is_quarter_active(self, idx: int) -> bool:
        """Check if a specific quarter has active requests."""
        with self._lock:
            if 0 <= idx < len(self._quarter_active):
                return self._quarter_active[idx]
            return False

    def __len__(self) -> int:
        return 1 + len(self._quarters)

    def __repr__(self) -> str:
        return (
            f"ConcurrencyAwareBackend(role={self._role!r}, "
            f"full=1, quarters={len(self._quarters)}, "
            f"migrations={self._migrations})"
        )
