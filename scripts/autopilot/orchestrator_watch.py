"""OrchestratorWatcher — fleet-marker-aware restart detection for autopilot.

Polls /dashboard/api/version (orchestrator startup id) and
/dashboard/api/llama_fleet_ids (per-llama-server startup ids + roles + source)
to classify why a /chat call failed:

  - operator_reload    — startup id changed AND source=stack_commands.
                         The failure is exogenous; safe to wait+retry.
  - external_restart   — startup id changed AND source!=stack_commands.
                         Treat as recoverable (retry once) but DO journal
                         as real if it doesn't recover; planner should see it.
  - unreachable        — endpoint failed or marker missing. Could be a
                         legit production crash (most likely) or a brief
                         network blip. Caller decides; the watcher just
                         reports.

Stateful — caches the last good ids for cache_ttl_s seconds to avoid
hammering /version on every /chat call.

See handoffs/active/autopilot-exogenous-restart-resilience.md sections
5.2 (watcher), 5.3 (resilient_post that consumes this), and 5.4
(propagation through result types).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# wait_for_health lives in scripts/server/stack_health; expose via a small
# import shim so we don't add another sys.path manipulation chain inside
# autopilot's import graph. Direct relative import would fail because
# scripts/server isn't a package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from scripts.server.stack_health import wait_for_health as _wait_for_health
except Exception:  # pragma: no cover — defensive; tests stub this out
    _wait_for_health = None  # type: ignore[assignment]


log = logging.getLogger("autopilot.watcher")


# Classification constants surfaced in meta dicts and tests.
CLASS_OPERATOR_RELOAD = "operator_reload"
CLASS_EXTERNAL_RESTART = "external_restart"
CLASS_UNREACHABLE = "unreachable"


# Sentinel surfaced when no marker has ever been observed (first poll).
# Distinguishable from None (unreachable) and from a real float.
NEVER_SEEN: float = -1.0


class OrchestratorWatcher:
    """Watches the orchestrator + llama-server fleet for restarts.

    Construction:
        watcher = OrchestratorWatcher()
        watcher = OrchestratorWatcher(disabled=True)  # no-op for tests/dev

    The disabled mode allows unit tests and dev environments without
    marker files to opt out cleanly — all methods become no-ops returning
    safe defaults so call sites don't have to branch on watcher being None.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        health_timeout_s: float = 120.0,
        cache_ttl_s: float = 2.0,
        http_timeout_s: float = 3.0,
        disabled: bool | None = None,
    ) -> None:
        if disabled is None:
            disabled = os.environ.get("AUTOPILOT_WATCHER_DISABLED", "") == "1"
        self.api_url = api_url.rstrip("/")
        self.health_timeout_s = health_timeout_s
        self.cache_ttl_s = cache_ttl_s
        self.http_timeout_s = http_timeout_s
        self.disabled = disabled

        # Per-call caches. Values: (timestamp_of_last_fetch, value).
        # `value` for orchestrator is float or None; for llama is dict
        # {port:int → {started_at, source, roles}} or None.
        self._orch_cache: tuple[float, float | None] | None = None
        self._llama_cache: tuple[float, dict[int, dict] | None] | None = None
        # Rate-limit "missing role" log so we don't spam.
        self._missing_role_warned: set[str] = set()

    # ── public surface ─────────────────────────────────────────────

    def current_orchestrator_id(self) -> float | None:
        """Fetch the orchestrator's `server_started_at` from /version.

        Cached for self.cache_ttl_s. Returns None when the endpoint is
        unreachable (network error, 5xx, etc.).
        """
        if self.disabled:
            return None
        now = time.time()
        if self._orch_cache and (now - self._orch_cache[0]) < self.cache_ttl_s:
            return self._orch_cache[1]
        val: float | None = None
        try:
            r = httpx.get(
                f"{self.api_url}/dashboard/api/version",
                timeout=self.http_timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            v = data.get("server_started_at")
            if isinstance(v, (int, float)):
                val = float(v)
        except Exception as exc:
            log.debug("orchestrator id fetch failed: %s", exc)
            val = None
        self._orch_cache = (now, val)
        return val

    def current_llama_fleet(self) -> dict[int, dict] | None:
        """Fetch the per-port llama-server marker dict.

        Cached for self.cache_ttl_s. Returns None when the endpoint is
        unreachable, OR a (possibly empty) dict on success.
        """
        if self.disabled:
            return {}
        now = time.time()
        if self._llama_cache and (now - self._llama_cache[0]) < self.cache_ttl_s:
            return self._llama_cache[1]
        val: dict[int, dict] | None = None
        try:
            r = httpx.get(
                f"{self.api_url}/dashboard/api/llama_fleet_ids",
                timeout=self.http_timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            raw = data.get("per_port") or {}
            # JSON keys are strings — convert to int ports
            val = {}
            for port_str, info in raw.items():
                try:
                    val[int(port_str)] = info
                except (TypeError, ValueError):
                    continue
        except Exception as exc:
            log.debug("llama fleet ids fetch failed: %s", exc)
            val = None
        self._llama_cache = (now, val)
        return val

    def current_llama_id(self, port: int) -> tuple[float, str] | None:
        """Get (started_at, source) for a specific llama-server port.

        Returns None if the port has no marker or the fleet endpoint is
        unreachable.
        """
        if self.disabled:
            return None
        fleet = self.current_llama_fleet()
        if not fleet:
            return None
        info = fleet.get(port)
        if not info:
            return None
        try:
            return float(info["started_at"]), str(info.get("source", ""))
        except (KeyError, TypeError, ValueError):
            return None

    def port_for_role(self, role: str) -> int | None:
        """Resolve role→port via the live llama_fleet_ids endpoint.

        First match wins (a given role currently maps to a single
        process). Caches via current_llama_fleet's cache. Emits a
        rate-limited warning (once per role per process) when the role
        isn't found in any marker.
        """
        if self.disabled or not role:
            return None
        fleet = self.current_llama_fleet()
        if not fleet:
            return None
        for port, info in fleet.items():
            roles = info.get("roles") or []
            if role in roles:
                return port
        if role not in self._missing_role_warned:
            self._missing_role_warned.add(role)
            log.warning(
                "OrchestratorWatcher: no llama marker carries role=%r "
                "(known fleet ports: %s). Restart detection for this role "
                "will degrade to orchestrator-only.",
                role,
                sorted(fleet.keys()),
            )
        return None

    def reference_for_role(self, role: str | None) -> dict[str, float]:
        """Snapshot of fleet identifiers relevant to a /chat call.

        Returns {"orchestrator": <id>, "llama_<port>": <id>, ...}.
        When role is None or no marker matches the role, returns
        orchestrator-only. Used by resilient_post to capture state
        BEFORE the POST so it can detect changes AFTER a failure.

        Missing/unreachable identifiers are present with value
        NEVER_SEEN (a distinguishable sentinel) so the comparator in
        was_restarted_since can tell them apart from a real float.
        """
        ref: dict[str, float] = {}
        orch = self.current_orchestrator_id()
        ref["orchestrator"] = orch if orch is not None else NEVER_SEEN
        if role:
            port = self.port_for_role(role)
            if port is not None:
                pair = self.current_llama_id(port)
                if pair is not None:
                    ref[f"llama_{port}"] = pair[0]
                else:
                    ref[f"llama_{port}"] = NEVER_SEEN
        return ref

    def was_restarted_since(
        self, reference_ids: dict[str, float]
    ) -> dict[str, str]:
        """Classify each identifier as operator_reload / external_restart / unreachable.

        Only identifiers whose current value differs from the reference
        are included in the output. Identifiers that match (same id) are
        absent — caller treats absence as "no restart."

        Reference values of NEVER_SEEN (sentinel for "no prior reading")
        do NOT trigger a restart classification; you can't have been
        restarted if you were never observed.
        """
        out: dict[str, str] = {}
        if self.disabled:
            return out
        for key, ref_val in reference_ids.items():
            if key == "orchestrator":
                current = self.current_orchestrator_id()
                if current is None:
                    out[key] = CLASS_UNREACHABLE
                elif ref_val != NEVER_SEEN and current != ref_val:
                    # Orchestrator has no `source` field — treat as
                    # operator-initiated by convention (only stack_commands
                    # ever restarts uvicorn in this stack).
                    out[key] = CLASS_OPERATOR_RELOAD
                continue
            if key.startswith("llama_"):
                try:
                    port = int(key[len("llama_"):])
                except ValueError:
                    continue
                pair = self.current_llama_id(port)
                if pair is None:
                    out[key] = CLASS_UNREACHABLE
                    continue
                current_id, source = pair
                if ref_val != NEVER_SEEN and current_id != ref_val:
                    if source == "stack_commands":
                        out[key] = CLASS_OPERATOR_RELOAD
                    else:
                        out[key] = CLASS_EXTERNAL_RESTART
        return out

    # ── recovery polls (wrapped around existing wait_for_health) ──

    def wait_for_orchestrator(self, timeout_s: float | None = None) -> bool:
        """Block until the orchestrator's /health returns 200 or timeout."""
        if self.disabled or _wait_for_health is None:
            return True
        return _wait_for_health(8000, timeout=int(timeout_s or self.health_timeout_s))

    def wait_for_llama(self, port: int, timeout_s: float | None = None) -> bool:
        """Block until a llama-server's /health returns 200 or timeout."""
        if self.disabled or _wait_for_health is None:
            return True
        return _wait_for_health(port, timeout=int(timeout_s or self.health_timeout_s))

    def wait_for_recovery(
        self,
        changes: dict[str, str],
        timeout_s: float | None = None,
    ) -> bool:
        """Wait for every changed identifier to come back to /health 200.

        Convenience for resilient_post: pass the dict returned by
        was_restarted_since(); waits in order. Returns True only when
        all wait_for_* calls succeeded.
        """
        if self.disabled:
            return True
        for key in changes:
            if key == "orchestrator":
                if not self.wait_for_orchestrator(timeout_s):
                    return False
            elif key.startswith("llama_"):
                try:
                    port = int(key[len("llama_"):])
                except ValueError:
                    continue
                if not self.wait_for_llama(port, timeout_s):
                    return False
        return True

    def invalidate_cache(self) -> None:
        """Force the next call to re-fetch /version + /llama_fleet_ids."""
        self._orch_cache = None
        self._llama_cache = None
