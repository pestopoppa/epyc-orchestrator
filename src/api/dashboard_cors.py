"""Narrowly-scoped CORS for the dashboard data plane.

RTG-47 Phase 1a (epyc-root ``handoffs/active/dashboard-architecture-restructure.md``).
The epyc-root dashboard hub on ``:8100`` is growing pages (``/machine``, ``/autopilot``)
whose *browsers* fetch this orchestrator's dashboard JSON and SSE endpoints directly,
cross-origin — no proxy, the hub stays a dumb static server.

Scope rationale — why this is not ``CORSMiddleware`` with a wider allowlist:
    The cross-origin need is confined to the **dashboard data plane**
    (``/dashboard/api/*`` and ``/dashboard/events/*``), which is read-mostly operator
    telemetry plus one control latch. The inference and session surfaces (``/v1/*``,
    ``/sessions/*``, …) must stay same-origin-only. Keeping this middleware
    path-scoped means the CORS surface can never widen to those APIs as a side effect
    of somebody editing a config list: every other path is passed through byte-identical,
    with no response-header mutation at all.

Policy:
    * Allowed origins: ``^https?://[^/]+:8100$`` — the hub on any host, http or https.
      Deliberately host-agnostic (the hub is reached by hostname, IP, or SSH-forwarded
      localhost depending on the operator's client) and deliberately *not* a wildcard.
      Safe only because credentials are never allowed: the browser sends no cookies or
      auth headers, so a matching origin buys nothing an unauthenticated LAN client
      could not fetch itself.
    * Credentials: never. ``Access-Control-Allow-Credentials`` is never emitted — and
      an inner one is *stripped* when we grant an origin (see ``send_with_cors``).
    * No ``Origin`` header, or a non-matching one: pass through untouched, so the
      legacy same-origin ``:8000/dashboard`` page sees byte-identical responses.
    * ``OPTIONS`` preflight on a dashboard path from a matching origin is answered
      directly with 204 — the app is never called. ``POST`` is in the allowed methods
      for ``/dashboard/api/autopilot_control`` (the hub autopilot page's pause/resume).
    * Everything else with a matching origin is forwarded, with the CORS headers
      appended to the ``http.response.start`` message only. Bodies are never buffered,
      so SSE/streaming endpoints keep streaming.

Implemented as a pure-ASGI middleware (not ``BaseHTTPMiddleware``) matching
``src/api/rate_limit.py``: BaseHTTPMiddleware buffers, which would break the SSE
endpoints this exists to serve.

Must be the OUTERMOST middleware (added last in ``create_app``): Starlette's global
``CORSMiddleware`` intercepts any ``OPTIONS`` carrying ``access-control-request-method``
and answers a non-allowlisted origin with ``400 Disallowed CORS origin``, so an inner
placement would never see the hub's preflight.

Usage:
    from src.api.dashboard_cors import DashboardCORSMiddleware

    app.add_middleware(DashboardCORSMiddleware)
"""

from __future__ import annotations

import re

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Dashboard data plane only. Prefixes, not exact paths: both trees are
# parameterised (``/dashboard/api/task/{id}``, ``/dashboard/events/task/{id}``).
# Note the trailing slash — the HTML page itself (``/dashboard``) is same-origin
# and stays outside this middleware.
DASHBOARD_PATH_PREFIXES: tuple[str, ...] = ("/dashboard/api/", "/dashboard/events/")

# The epyc-root hub, any host, http or https. No wildcard, no credentials.
HUB_ORIGIN_PATTERN = r"^https?://[^/]+:8100$"

_ALLOW_METHODS = b"GET, POST, OPTIONS"
_ALLOW_HEADERS = b"content-type"
_MAX_AGE = b"3600"


class DashboardCORSMiddleware:
    """Pure-ASGI CORS middleware scoped to the dashboard data plane.

    Args:
        app: The ASGI application.
        origin_pattern: Regex an ``Origin`` header must fully satisfy to be allowed.
        path_prefixes: Request-path prefixes this middleware applies to. Any other
            path is passed through with no response mutation whatsoever.
    """

    def __init__(
        self,
        app: ASGIApp,
        origin_pattern: str = HUB_ORIGIN_PATTERN,
        path_prefixes: tuple[str, ...] = DASHBOARD_PATH_PREFIXES,
    ):
        self.app = app
        self._origin_re = re.compile(origin_pattern)
        self._path_prefixes = tuple(path_prefixes)

    def _matching_origin(self, scope: Scope) -> bytes | None:
        """Return the raw ``Origin`` header iff it is an allowed hub origin."""
        for name, value in scope.get("headers", []):
            if name.lower() == b"origin":
                if self._origin_re.match(value.decode("latin-1")):
                    return bytes(value)
                return None
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not scope.get("path", "").startswith(self._path_prefixes):
            await self.app(scope, receive, send)
            return

        origin = self._matching_origin(scope)
        if origin is None:
            # No Origin, or an origin we do not serve: byte-identical passthrough.
            await self.app(scope, receive, send)
            return

        if scope.get("method", "").upper() == "OPTIONS":
            await self._send_preflight(origin, send)
            return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Strip any inner Access-Control-Allow-Credentials. Starlette's global
                # CORSMiddleware stamps `true` onto EVERY response that carries an
                # Origin header — including origins it does not allow, where the header
                # is inert because no Allow-Origin accompanies it. Ours does accompany
                # it, so leaving it would silently upgrade this uncredentialed grant
                # into a credentialed one.
                headers = [
                    tuple(h)
                    for h in message.get("headers") or []
                    if h[0].lower() != b"access-control-allow-credentials"
                ]
                # Defensive: if an inner middleware already answered CORS for this
                # origin, a second Access-Control-Allow-Origin would make the browser
                # reject the response outright.
                if not any(name.lower() == b"access-control-allow-origin" for name, _ in headers):
                    headers.append((b"access-control-allow-origin", origin))
                headers.append((b"vary", b"Origin"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_cors)

    async def _send_preflight(self, origin: bytes, send: Send) -> None:
        """Answer a preflight with 204 directly; the app is never invoked."""
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [
                    (b"access-control-allow-origin", origin),
                    (b"access-control-allow-methods", _ALLOW_METHODS),
                    (b"access-control-allow-headers", _ALLOW_HEADERS),
                    (b"access-control-max-age", _MAX_AGE),
                    (b"vary", b"Origin"),
                    (b"content-length", b"0"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})
