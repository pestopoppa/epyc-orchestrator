"""AEI-HTTPPOST-GUARD: http_post applies the SEC-EXEC-3 web_guard destination check.

No network: ``web_guard._resolve`` is stubbed and the opener is replaced where a request would
otherwise leave the process.
"""

import socket
import urllib.request

import pytest

from orchestration.tools import web
from src.features import reset_features
from src.repl_environment import web_guard

PUBLIC4 = "93.184.216.34"


@pytest.fixture(autouse=True)
def _dns(monkeypatch):
    def fake(host, port):
        import ipaddress

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            if host == "public.example.com":
                ip = ipaddress.ip_address(PUBLIC4)
            else:
                raise socket.gaierror(f"no such host {host}")
        fam = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        return [(fam, socket.SOCK_STREAM, 6, "", (str(ip), port))]

    monkeypatch.setattr(web_guard, "_resolve", fake)
    monkeypatch.delenv("ORCHESTRATOR_HTTP_POST_GUARD", raising=False)
    reset_features()
    yield
    reset_features()


@pytest.fixture
def no_network(monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a request left the process")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)
    return calls


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://127.0.0.1:8000/admin",
    "http://localhost/x",
    "http://10.0.0.5/hook",
    "http://192.168.1.1/",
    "http://[::ffff:10.0.0.1]/",
    "http://[fd00:ec2::254]/",
    "http://2130706433/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "file:///etc/passwd",
])
def test_private_and_metadata_destinations_are_refused(url, no_network):
    out = web.http_post(url, {"k": "v"})
    assert out["status_code"] == 0
    assert out["refused"] is True
    assert out["error"].startswith("http_post refused:")
    assert no_network == [], "the request must not be attempted"


def test_public_destination_is_allowed_through(monkeypatch):
    seen = {}

    class _R:
        status = 200
        headers = {"X": "1"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def fake_open(self, request, data=None, timeout=None):
        seen["url"] = request.full_url
        return _R()

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)
    out = web.http_post("http://public.example.com/hook", {"k": "v"})
    assert out["status_code"] == 200 and out["body"] == "ok"
    assert "refused" not in out
    assert seen["url"] == "http://public.example.com/hook"


def test_redirects_are_not_followed():
    handler = urllib.request.HTTPRedirectHandler
    opener_handlers = []

    orig = urllib.request.build_opener

    def spy(*h):
        opener_handlers.extend(h)
        return orig(*h)

    import unittest.mock as m

    with m.patch.object(urllib.request, "build_opener", spy), \
            m.patch.object(urllib.request.OpenerDirector, "open", side_effect=OSError("stop")):
        web.http_post("http://public.example.com/hook", {})
    assert opener_handlers and issubclass(opener_handlers[0], handler)
    assert opener_handlers[0].redirect_request(opener_handlers[0](), None, 302, "", {}, "http://127.0.0.1/") is None


def test_kill_switch_restores_unguarded_path(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_HTTP_POST_GUARD", "0")
    reset_features()
    called = {}

    def fake_urlopen(request, timeout=None):
        called["url"] = request.full_url
        raise OSError("blocked in test")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = web.http_post("http://127.0.0.1:9/x", {})
    assert called["url"] == "http://127.0.0.1:9/x"
    assert "refused" not in out
