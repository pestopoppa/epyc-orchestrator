"""SEC-EXEC-3: destination deny for the model-reachable web tools (web_fetch, fetch_docs).

A model-issued URL must never reach a loopback, private, link-local, metadata or otherwise
non-global destination (SSRF into the host's own services, the cloud metadata endpoint, the
LAN). Deny by default: a destination is allowed only when EVERY address it resolves to is a
global unicast address. Nothing here fails open -- a URL that cannot be parsed, a name that
does not resolve, or an address form that cannot be classified is refused with a reason.

Defence layers (each independent, so a bypass of one is caught by the next):

1. ``check_url``: syntax + literal-form pre-check. Refuses userinfo (``user@host`` parse
   differentials), backslashes/whitespace/control characters, zone ids, and every
   NON-CANONICAL numeric host (decimal ``2130706433``, octal ``0177.0.0.1``, hex ``0x7f.1``,
   short ``127.1``): libc/curl read those as IPv4, ``ipaddress`` does not, so they are refused
   outright whatever they decode to. Then resolves the name and classifies every answer.
2. ``fetch`` (requests path): every redirect hop is re-checked through ``check_url`` (redirects
   are never followed by the HTTP library), and the connection itself is made by
   ``guarded_create_connection``.
3. ``guarded_create_connection``: resolves the host AGAIN at connect time, classifies every
   answer, and connects only to the address it just classified. A name that answered
   "public" to layer 1 and "127.0.0.1" now (DNS rebinding) is refused here, because the
   address checked is the address connected -- there is no second, unchecked resolution.
4. ``resolve_pinned`` (curl path used by fetch_docs): the validated address is handed to curl
   with ``--resolve`` so curl never resolves the name itself.

Address classification (``ip_denied_reason``): global unicast only. IPv4-mapped IPv6
(``::ffff:127.0.0.1``) is unwrapped and judged as IPv4; IPv6 transition forms that embed an
IPv4 address (NAT64, 6to4, Teredo, IPv4-compatible) are refused outright.

Accepted residuals: environment proxies are ignored on the guarded path (``trust_env`` is
False), so a deployment that REQUIRES an egress proxy must run with the guard flag off or
add the proxy explicitly; a destination that is public but hostile is out of scope (this is
the private-range deny, not a content policy).
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import threading
from typing import Any
from urllib.parse import urljoin, urlsplit

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 5
_ALLOWED_SCHEMES = ("http", "https")
_DENY_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".home.arpa", ".lan")
_DENY_HOSTS = frozenset({"localhost", "ip6-localhost", "ip6-loopback", "broadcasthost"})
_BAD_URL_CHARS = re.compile(r"[\x00-\x20\x7f\\]")
# One dotted part of an IPv4 literal as inet_aton reads it: hex (0x..), octal/decimal digits.
_NUMERIC_PART = re.compile(r"(?:0[xX][0-9a-fA-F]*|[0-9]+)")

# Belt and braces over ``ipaddress.is_global`` (whose table has changed across Python versions).
_V4_DENY = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
    "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24", "192.88.99.0/24", "192.168.0.0/16",
    "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24", "224.0.0.0/4", "240.0.0.0/4",
    "255.255.255.255/32",
))
_V6_DENY = tuple(ipaddress.ip_network(n) for n in (
    "::/96",            # unspecified, loopback, IPv4-compatible
    "64:ff9b::/96",     # NAT64 well-known prefix (embeds IPv4)
    "64:ff9b:1::/48",   # NAT64 local-use prefix
    "100::/64",         # discard-only
    "2001::/32",        # Teredo (embeds IPv4)
    "2001:db8::/32",    # documentation
    "2002::/16",        # 6to4 (embeds IPv4)
    "fc00::/7",         # unique local
    "fe80::/10",        # link-local
    "fec0::/10",        # deprecated site-local
    "ff00::/8",         # multicast
))


class WebGuardRefusal(Exception):
    """The destination was refused; ``str(exc)`` is the human-readable reason."""


class BlockedDestination(OSError):
    """Raised at connect time by ``guarded_create_connection``."""


_tls = threading.local()


def _note_denial(reason: str) -> None:
    lst = getattr(_tls, "denials", None)
    if lst is None:
        lst = _tls.denials = []
    lst.append(reason)


def pop_denials() -> list[str]:
    """Connect-time denials recorded on this thread since the last call."""
    lst = getattr(_tls, "denials", None) or []
    _tls.denials = []
    return lst


# ── address classification ──────────────────────────────────────────────────


def ip_denied_reason(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """None when ``ip`` is an allowed (global unicast) destination, else the reason."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            inner = ip_denied_reason(ip.ipv4_mapped)
            return f"IPv4-mapped IPv6 address ({inner})" if inner else None
        for net in _V6_DENY:
            if ip in net:
                return f"IPv6 address in non-global range {net}"
    else:
        for net in _V4_DENY:
            if ip in net:
                return f"IPv4 address in non-global range {net}"
    if not ip.is_global or ip.is_multicast or ip.is_unspecified or ip.is_loopback or ip.is_link_local:
        return "address is not a global unicast address"
    return None


def _numeric_host_form(host: str) -> bool:
    """True when every dotted part reads as a number to inet_aton (so libc would take it as IPv4)."""
    parts = host.split(".")
    return 1 <= len(parts) <= 4 and all(_NUMERIC_PART.fullmatch(p) for p in parts)


def _classify_host_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The address a host literal denotes, None for a name; raises on ambiguous numeric forms."""
    if ":" in host:
        try:
            return ipaddress.IPv6Address(host)
        except ValueError as exc:
            raise WebGuardRefusal(f"malformed IPv6 literal: {exc}") from exc
    if _numeric_host_form(host):
        try:
            return ipaddress.IPv4Address(host)  # canonical dotted quad only (rejects octal/short/hex)
        except ValueError:
            raise WebGuardRefusal(
                f"non-canonical numeric host {host!r} (decimal/octal/hex/short IPv4 forms are refused)"
            ) from None
    return None


def _split_host(url: str) -> tuple[str, str, int]:
    """(scheme, normalised ascii host, port) or raises WebGuardRefusal."""
    if not isinstance(url, str) or not url:
        raise WebGuardRefusal("empty URL")
    if _BAD_URL_CHARS.search(url):
        raise WebGuardRefusal("URL contains whitespace, control characters or a backslash")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise WebGuardRefusal(f"unparseable URL: {exc}") from exc
    scheme = (parts.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise WebGuardRefusal(f"scheme {scheme!r} not allowed (http/https only)")
    if "@" in parts.netloc:
        raise WebGuardRefusal("URL carries userinfo (user@host) and is ambiguous across parsers")
    host = parts.hostname
    if not host:
        raise WebGuardRefusal("URL has no host")
    if "%" in host:
        raise WebGuardRefusal("host carries a zone id or percent-encoding")
    host = host.rstrip(".")
    if not host:
        raise WebGuardRefusal("URL has no host")
    try:
        # NFKC via IDNA folds full-width digits etc. to the ASCII form the HTTP stack will use.
        host = host.encode("idna").decode("ascii") if not host.isascii() else host
    except UnicodeError as exc:
        raise WebGuardRefusal(f"host is not valid IDNA: {exc}") from exc
    host = host.lower()
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, host, port


def _resolve(host: str, port: int) -> list[tuple[Any, ...]]:
    return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)


def _addresses_or_refuse(host: str, port: int) -> list[tuple[int, int, int, tuple[Any, ...], ipaddress._BaseAddress]]:
    """Resolve ``host`` and return (family, type, proto, sockaddr, ip) rows, ALL of them allowed."""
    lit = _classify_host_literal(host)
    if lit is None:
        if host in _DENY_HOSTS or host.endswith(_DENY_HOST_SUFFIXES):
            raise WebGuardRefusal(f"host name {host!r} is a local/internal name")
    try:
        rows = _resolve(host, port)
    except OSError as exc:
        raise WebGuardRefusal(f"host {host!r} does not resolve: {exc}") from exc
    if not rows:
        raise WebGuardRefusal(f"host {host!r} resolved to no addresses")
    out = []
    for family, stype, proto, _canon, sockaddr in rows:
        raw = str(sockaddr[0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise WebGuardRefusal(f"resolver returned an unparseable address {raw!r}") from exc
        reason = ip_denied_reason(ip)
        if reason:
            # ANY denied answer refuses the host: an attacker-controlled name can mix records.
            raise WebGuardRefusal(f"{host} resolves to {ip}: {reason}")
        out.append((family, stype, proto, sockaddr, ip))
    return out


def check_url(url: str) -> str | None:
    """None when ``url`` may be fetched, else the refusal reason (never raises)."""
    try:
        _scheme, host, port = _split_host(url)
        _addresses_or_refuse(host, port)
    except WebGuardRefusal as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - an unclassifiable URL is refused, never allowed
        return f"URL could not be classified ({type(exc).__name__}: {exc})"
    return None


def resolve_pinned(url: str) -> tuple[str, int, str]:
    """(host, port, ip) for the first validated address; raises WebGuardRefusal. For curl --resolve."""
    _scheme, host, port = _split_host(url)
    rows = _addresses_or_refuse(host, port)
    return host, port, str(rows[0][4])


# ── connect-time guard (closes the DNS-rebinding window) ────────────────────


def _open_socket(family, stype, proto, sockaddr, timeout, source_address, socket_options):
    sock = socket.socket(family, stype, proto)
    try:
        for level, opt, value in socket_options or ():
            sock.setsockopt(level, opt, value)
        if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT and timeout is not None:
            sock.settimeout(timeout)
        if source_address:
            sock.bind(source_address)
        sock.connect(sockaddr)
        return sock
    except BaseException:
        sock.close()
        raise


def guarded_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None,
                              socket_options=None):
    """Drop-in for ``urllib3.util.connection.create_connection`` that only reaches global addresses."""
    host, port = address
    host = str(host).strip("[]")
    try:
        rows = _addresses_or_refuse(host.rstrip(".").lower(), int(port))
    except WebGuardRefusal as exc:
        _note_denial(str(exc))
        raise BlockedDestination(f"web_guard: {exc}") from exc
    err: OSError | None = None
    for family, stype, proto, sockaddr, _ip in rows:
        try:
            return _open_socket(family, stype, proto, sockaddr, timeout, source_address, socket_options)
        except OSError as exc:
            err = exc
    raise err if err is not None else BlockedDestination("web_guard: no usable address")


def _build_session():
    """requests.Session whose every connection goes through ``guarded_create_connection``."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.connection import HTTPConnection, HTTPSConnection
    from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

    class _GuardMixin:
        def _new_conn(self):  # noqa: D401 - urllib3 hook
            return guarded_create_connection(
                (self._dns_host, self.port), self.timeout,
                source_address=self.source_address, socket_options=self.socket_options,
            )

    class _HTTPConn(_GuardMixin, HTTPConnection):
        pass

    class _HTTPSConn(_GuardMixin, HTTPSConnection):
        pass

    class _HTTPPool(HTTPConnectionPool):
        ConnectionCls = _HTTPConn

    class _HTTPSPool(HTTPSConnectionPool):
        ConnectionCls = _HTTPSConn

    class _Adapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            super().init_poolmanager(*args, **kwargs)
            self.poolmanager.pool_classes_by_scheme = {"http": _HTTPPool, "https": _HTTPSPool}

        def proxy_manager_for(self, *args, **kwargs):  # a proxy would resolve the name remotely
            raise WebGuardRefusal("proxies are not allowed on the guarded fetch path")

    session = requests.Session()
    session.trust_env = False  # no environment proxies / netrc
    adapter = _Adapter(max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _session_get(url: str, headers: dict[str, str], timeout: float):
    """One GET hop through the guarded session; redirects are NOT followed. Test seam."""
    session = _build_session()
    try:
        return session.get(url, headers=headers, timeout=timeout, allow_redirects=False)
    finally:
        session.close()


def fetch(url: str, headers: dict[str, str], timeout: float = 30):
    """GET ``url`` following redirects manually; every hop is re-checked. Raises WebGuardRefusal."""
    import requests

    current = url
    for hop in range(MAX_REDIRECTS + 1):
        reason = check_url(current)
        if reason:
            raise WebGuardRefusal(reason if hop == 0 else f"redirect hop {hop} to {_safe(current)}: {reason}")
        pop_denials()
        try:
            resp = _session_get(current, headers, timeout)
        except requests.exceptions.RequestException:
            denials = pop_denials()
            if denials:
                raise WebGuardRefusal(f"connect-time check: {denials[0]}") from None
            raise
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            current = urljoin(current, resp.headers["Location"])
            continue
        return resp
    raise WebGuardRefusal(f"more than {MAX_REDIRECTS} redirects")


def _safe(url: str) -> str:
    """Host-only rendering of a URL for logs (query strings can carry secrets)."""
    try:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.hostname}"
    except ValueError:
        return "<unparseable>"


def log_refusal(tool: str, url: str, reason: str) -> None:
    logger.warning("%s DENIED destination=%s reason=%s", tool, _safe(url), reason)
