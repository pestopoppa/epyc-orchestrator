"""SEC-EXEC-3: web_fetch / fetch_docs destination deny (private, loopback, link-local, metadata).

No network and no real DNS: ``web_guard._resolve`` and ``web_guard._session_get`` are stubbed.
"""

import json
import socket
from unittest.mock import MagicMock, patch

import pytest

from src.features import reset_features
from src.repl_environment import REPLEnvironment, web_guard
from src.registry.tool_registry import ToolRegistry, load_from_yaml


def _make_web_repl():
    from pathlib import Path

    root = Path(__file__).parents[2]
    registry = ToolRegistry()
    load_from_yaml(registry, root / "orchestration" / "tool_registry.yaml")
    registry.load_permissions_from_registry(root / "orchestration" / "model_registry.yaml")
    return REPLEnvironment(context="t", tool_registry=registry, role="frontdoor")

PUBLIC4 = "93.184.216.34"
PUBLIC6 = "2606:2800:220:1:248:1893:25c8:1946"


def _rows(ips, port=80):
    out = []
    for ip in ips:
        if ":" in ip:
            out.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port, 0, 0)))
        else:
            out.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)))
    return out


@pytest.fixture
def dns(monkeypatch):
    """Name -> [ips] resolver. IP literals resolve to themselves; unknown names do not resolve."""
    table: dict = {}

    def fake(host, port):
        if host in table:
            ans = table[host]
            ans = ans.pop(0) if isinstance(ans, list) and ans and isinstance(ans[0], list) else ans
            return _rows(ans, port)
        try:
            import ipaddress

            ipaddress.ip_address(host)
            return _rows([host], port)
        except ValueError:
            raise socket.gaierror(f"no such host {host}")

    monkeypatch.setattr(web_guard, "_resolve", fake)
    return table


@pytest.fixture
def audit(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_TOOL_AUDIT_LOG", str(path))
    return path


@pytest.fixture
def flags(monkeypatch):
    def _set(**kw):
        for name, on in kw.items():
            monkeypatch.setenv("ORCHESTRATOR_" + name.upper(), "1" if on else "0")
        reset_features()

    yield _set
    monkeypatch.undo()
    reset_features()


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ── address classification and URL forms ────────────────────────────────────

DENIED = [
    # loopback / private / link-local / metadata / CGNAT / unspecified
    "http://127.0.0.1/", "http://127.0.0.1:8000/x", "http://127.255.255.254/", "http://0.0.0.0/",
    "http://10.0.0.5/", "http://172.16.0.1/", "http://192.168.1.1/", "http://100.64.0.1/",
    "http://169.254.169.254/latest/meta-data/", "http://169.254.0.1/", "http://100.100.100.200/",
    "http://[::1]/", "http://[::]/", "http://[fe80::1]/", "http://[fd00:ec2::254]/", "http://[fc00::1]/",
    # IPv6-mapped IPv4 (both spellings) and transition forms that embed IPv4
    "http://[::ffff:127.0.0.1]/", "http://[::ffff:7f00:1]/", "http://[::ffff:169.254.169.254]/",
    "http://[0:0:0:0:0:ffff:10.0.0.1]/", "http://[64:ff9b::7f00:1]/", "http://[2002:7f00:1::]/",
    "http://[::127.0.0.1]/",
    # decimal / octal / hex / short IPv4 forms (libc reads them, ipaddress does not)
    "http://2130706433/", "http://0177.0.0.1/", "http://0177" "00000001/", "http://0x7f.0.0.1/",
    "http://0x7f000001/", "http://127.1/", "http://127.0.1/", "http://0x7f.1/", "http://2852039166/",
    "http://0251.0376.0251.0376/",
    # a PUBLIC address in a non-canonical form is still refused (ambiguity, not value)
    "http://1.1.0x1.1/", "http://16843009/",
    # trailing dot, full-width digits (IDNA folds them to 127.0.0.1)
    "http://127.0.0.1./", "http://１２７.０.０.１/",
    # names that are local by convention
    "http://localhost/", "http://LOCALHOST./", "http://foo.localhost/", "http://metadata.google.internal/",
    "http://printer.local/", "http://nas.lan/",
    # parser-differential tricks
    "http://evil.example.com@127.0.0.1/", "http://127.0.0.1\\@evil.example.com/", "http://[fe80::1%25eth0]/",
    "http://example.com:99999/", "http:///nohost", "http://exa mple.com/",
    # schemes
    "file:///etc/passwd", "ftp://example.com/", "gopher://127.0.0.1:70/", "javascript:alert(1)", "",
]


@pytest.mark.parametrize("url", DENIED)
def test_denied_destinations(url, dns):
    reason = web_guard.check_url(url)
    assert reason, f"{url!r} was allowed"


ALLOWED = [
    f"http://{PUBLIC4}/", f"https://{PUBLIC4}:8443/a?b=c", f"http://[{PUBLIC6}]/",
    "https://[2001:4860:4860::8888]/", "http://[::ffff:93.184.216.34]/",
]


@pytest.mark.parametrize("url", ALLOWED)
def test_global_literals_allowed(url, dns):
    assert web_guard.check_url(url) is None


def test_public_name_allowed(dns):
    dns["docs.example.com"] = [PUBLIC4]
    assert web_guard.check_url("https://docs.example.com/page") is None


def test_name_resolving_to_private_is_denied(dns):
    dns["internal.example.com"] = ["10.1.2.3"]
    assert "10.1.2.3" in web_guard.check_url("https://internal.example.com/")


@pytest.mark.parametrize("answers", [
    [PUBLIC4, "127.0.0.1"], ["127.0.0.1", PUBLIC4], [PUBLIC4, "::1"], [PUBLIC4, "169.254.169.254"],
    [PUBLIC6, "::ffff:10.0.0.1"],
])
def test_mixed_answers_deny_the_whole_host(dns, answers):
    dns["mixed.example.com"] = answers
    assert web_guard.check_url("http://mixed.example.com/")


def test_unresolvable_and_empty_answers_are_denied_not_allowed(dns, monkeypatch):
    assert "does not resolve" in web_guard.check_url("http://nonexistent.example.com/")
    monkeypatch.setattr(web_guard, "_resolve", lambda h, p: [])
    assert "no addresses" in web_guard.check_url("http://empty.example.com/")


def test_classifier_exception_is_a_refusal_not_an_allow(monkeypatch):
    def boom(h, p):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(web_guard, "_resolve", boom)
    assert web_guard.check_url("http://anything.example.com/")


# ── DNS rebinding shape: public at check time, private at connect time ─────


def test_dns_rebinding_is_refused_at_connect_time(dns, monkeypatch):
    dns["rebind.example.com"] = [[PUBLIC4], ["127.0.0.1"]]  # first lookup public, second private
    opened = []
    monkeypatch.setattr(web_guard, "_open_socket", lambda *a, **k: opened.append(a) or object())
    assert web_guard.check_url("http://rebind.example.com/") is None  # layer 1 saw the public answer
    web_guard.pop_denials()
    with pytest.raises(web_guard.BlockedDestination):
        web_guard.guarded_create_connection(("rebind.example.com", 80))  # layer 3 sees 127.0.0.1
    assert opened == [], "a socket was opened to a rebound address"
    assert any("127.0.0.1" in d for d in web_guard.pop_denials())


@pytest.mark.parametrize("host", ["127.0.0.1", "[::1]", "[::ffff:127.0.0.1]", "169.254.169.254", "2130706433"])
def test_guarded_connection_refuses_private_literals(host, dns, monkeypatch):
    monkeypatch.setattr(web_guard, "_open_socket", lambda *a, **k: pytest.fail("socket opened"))
    with pytest.raises(web_guard.BlockedDestination):
        web_guard.guarded_create_connection((host, 80))


def test_guarded_connection_connects_to_the_address_it_checked(dns, monkeypatch):
    dns["ok.example.com"] = [PUBLIC4]
    seen = []
    monkeypatch.setattr(
        web_guard, "_open_socket",
        lambda family, stype, proto, sockaddr, *a: seen.append(sockaddr) or "SOCK",
    )
    assert web_guard.guarded_create_connection(("ok.example.com", 8080)) == "SOCK"
    assert seen == [(PUBLIC4, 8080)]


# ── redirects: every hop is re-checked ──────────────────────────────────────


class _Resp:
    def __init__(self, status=200, headers=None, text="body"):
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/plain"}
        self.text = text

    def raise_for_status(self):
        pass


def _hops(monkeypatch, *responses):
    calls = []
    it = iter(responses)

    def fake(url, headers, timeout):
        calls.append(url)
        return next(it)

    monkeypatch.setattr(web_guard, "_session_get", fake)
    return calls


@pytest.mark.parametrize("target", [
    "http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:8000/admin", "//127.0.0.1/x",
    "http://[::ffff:10.0.0.1]/", "http://2130706433/", "http://0177.0.0.1/", "http://localhost/",
    "file:///etc/passwd", "/\\\\127.0.0.1/",
])
def test_redirect_to_private_destination_is_refused(target, dns, monkeypatch):
    calls = _hops(monkeypatch, _Resp(302, {"Location": target}))
    with pytest.raises(web_guard.WebGuardRefusal) as exc:
        web_guard.fetch(f"http://{PUBLIC4}/start", {}, 5)
    assert "redirect hop 1" in str(exc.value)
    assert calls == [f"http://{PUBLIC4}/start"], "the private hop was requested"


def test_relative_and_public_redirects_are_followed(dns, monkeypatch):
    calls = _hops(
        monkeypatch,
        _Resp(301, {"Location": "/next"}),
        _Resp(302, {"Location": f"https://{PUBLIC4}/final"}),
        _Resp(200, text="done"),
    )
    resp = web_guard.fetch(f"http://{PUBLIC4}/start", {}, 5)
    assert resp.text == "done"
    assert calls == [f"http://{PUBLIC4}/start", f"http://{PUBLIC4}/next", f"https://{PUBLIC4}/final"]


def test_redirect_loop_is_bounded(dns, monkeypatch):
    _hops(monkeypatch, *[_Resp(302, {"Location": f"http://{PUBLIC4}/again"})] * 20)
    with pytest.raises(web_guard.WebGuardRefusal, match="redirects"):
        web_guard.fetch(f"http://{PUBLIC4}/", {}, 5)


def test_connect_time_denial_surfaces_as_a_refusal(dns, monkeypatch):
    import requests

    def fake(url, headers, timeout):
        web_guard._note_denial("rebind.example.com resolves to 127.0.0.1: IPv4 address in non-global range")
        raise requests.exceptions.ConnectionError("max retries")

    dns["rebind.example.com"] = [PUBLIC4]
    monkeypatch.setattr(web_guard, "_session_get", fake)
    with pytest.raises(web_guard.WebGuardRefusal, match="connect-time"):
        web_guard.fetch("http://rebind.example.com/", {}, 5)


# ── the REPL tool: flag-gated, refusal visible and audited ──────────────────


def test_repl_web_fetch_denies_and_audits(dns, audit, flags, monkeypatch, caplog):
    flags(web_fetch_guard=False)
    with patch("requests.get") as legacy:
        with caplog.at_level("WARNING"):
            out = _make_web_repl()._web_fetch("http://169.254.169.254/latest/meta-data/")
    assert "WEB_FETCH DENIED" in out
    legacy.assert_not_called()
    assert any("web_fetch DENIED" in r.getMessage() for r in caplog.records)
    rec = _records(audit)[-1]
    assert rec["tool"] == "web_fetch" and rec["outcome"] == "refused"
    assert "non-global" in rec["refusal_reason"] or "global" in rec["refusal_reason"]
    assert rec["host"] == "169.254.169.254" and rec["first_seen_host"] is True


def test_repl_web_fetch_public_url_works_with_guard_on(dns, audit, flags, monkeypatch):
    flags(web_fetch_guard=True)
    _hops(monkeypatch, _Resp(200, text="hello world"))
    out = _make_web_repl()._web_fetch(f"http://{PUBLIC4}/")
    assert out == "hello world"
    assert _records(audit)[-1]["outcome"] == "ok"


def test_repl_web_fetch_redirect_to_private_is_refused(dns, audit, flags, monkeypatch):
    flags(web_fetch_guard=True)
    _hops(monkeypatch, _Resp(302, {"Location": "http://127.0.0.1:8000/"}))
    out = _make_web_repl()._web_fetch(f"http://{PUBLIC4}/r")
    assert "WEB_FETCH DENIED" in out and "redirect hop 1" in out
    assert _records(audit)[-1]["outcome"] == "refused"


def test_repl_web_fetch_guard_is_unconditional_even_when_flag_off(dns, audit, flags, monkeypatch):
    flags(web_fetch_guard=False)
    calls = _hops(monkeypatch, _Resp(text="should not fetch"))
    out = _make_web_repl()._web_fetch("http://127.0.0.1:1/")
    assert "WEB_FETCH DENIED" in out
    assert calls == []


# ── fetch_docs (the curl path) ──────────────────────────────────────────────


def test_fetch_docs_denies_private_and_audits(dns, audit, flags):
    from src.tools.web import fetch as fetch_mod

    flags(web_fetch_guard=True)
    with patch("subprocess.run") as run:
        out = fetch_mod.fetch_docs("http://127.0.0.1:9000/", check_trust=False)
    run.assert_not_called()
    assert out["success"] is False and "WEB_FETCH DENIED" in out["error"]
    rec = _records(audit)[-1]
    assert rec["tool"] == "fetch_docs" and rec["outcome"] == "refused"


def test_fetch_docs_pins_the_validated_address_and_rechecks_redirects(dns, audit, flags):
    from src.tools.web import fetch as fetch_mod

    flags(web_fetch_guard=True)
    dns["docs.example.com"] = [PUBLIC4]
    first = MagicMock(returncode=0, stdout=b"\n__CONTENT_TYPE__:text/html\n__CODE__:302\n__REDIRECT__:http://10.0.0.9/x")
    with patch("subprocess.run", return_value=first) as run:
        out = fetch_mod.fetch_docs("https://docs.example.com/p", check_trust=False)
    argv = run.call_args.args[0]
    assert "--resolve" in argv and argv[argv.index("--resolve") + 1] == f"docs.example.com:443:{PUBLIC4}"
    assert "-L" not in argv and argv[argv.index("--max-redirs") + 1] == "0"
    assert run.call_count == 1, "the private redirect target was fetched"
    assert out["success"] is False and "redirect hop 1" in out["error"]
