"""SEC-ART-2/4 wiring: build_server_command refuses before any Popen; start_server converts the refusal."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from scripts.server import orchestrator_stack as oss
from src.registry import artifact_integrity as ai
from tests.unit.test_artifact_integrity import TEMPLATE, make_gguf, sha


def _setup(tmp_path, monkeypatch, *, pin_template: str | None, with_file: bool):
    gguf = make_gguf(tmp_path / "m.gguf")
    tfile = tmp_path / "t.jinja"
    tfile.write_bytes(TEMPLATE)
    reg = tmp_path / "reg.yaml"
    reg.write_text(yaml.safe_dump({"roles": {"frontdoor": {"model": {"path": str(gguf), "sha256": sha(gguf)}}}}))
    led = tmp_path / "ledger.json"
    ai.register_gguf(gguf, pin_sha256=sha(gguf), ledger=led)
    monkeypatch.setenv("ARTIFACT_INTEGRITY_ENFORCE_IN_TESTS", "1")
    monkeypatch.setenv(ai.REGISTRY_ENV, str(reg))
    monkeypatch.setenv(ai.LEDGER_ENV, str(led))
    ai._REG_CACHE.clear()
    cmd = ["llama-server", "-m", str(gguf), "--port", "1"] + (["--chat-template-file", str(tfile)] if with_file else [])
    flags = {"served_template_sha256": pin_template} if pin_template else {}
    monkeypatch.setattr(oss, "_build_role_command", lambda *a, **k: list(cmd))
    monkeypatch.setattr(oss, "_stack_prior_launch", lambda role: ({}, {"flags": flags}))
    popen_calls: list = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: popen_calls.append(a) or (_ for _ in ()).throw(AssertionError("Popen reached")))
    return gguf, tfile, popen_calls


def test_matching_template_pin_builds_the_command(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, pin_template=ai.template_sha256(TEMPLATE), with_file=True)
    cmd = oss.build_server_command(SimpleNamespace(name="frontdoor"), 8070)
    assert cmd[0] == "llama-server" and "--chat-template-file" in cmd


def test_one_byte_modified_template_file_refuses_before_popen(tmp_path, monkeypatch):
    _gguf, tfile, popen_calls = _setup(tmp_path, monkeypatch, pin_template=ai.template_sha256(TEMPLATE), with_file=True)
    evil = bytearray(TEMPLATE)
    evil[5] ^= 0x01
    tfile.write_bytes(bytes(evil))
    with pytest.raises(ai.TemplatePinError):
        oss.build_server_command(SimpleNamespace(name="frontdoor"), 8070)
    assert popen_calls == []


def test_changed_gguf_refuses_with_gguf_pin_error(tmp_path, monkeypatch):
    gguf, _t, _p = _setup(tmp_path, monkeypatch, pin_template=None, with_file=False)
    st = gguf.stat()
    data = bytearray(gguf.read_bytes())
    data[-3] ^= 0x01
    gguf.write_bytes(bytes(data))
    import os
    os.utime(gguf, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with pytest.raises(ai.GgufPinError):
        oss.build_server_command(SimpleNamespace(name="frontdoor"), 8070)


def test_check_is_skipped_under_pytest_by_default(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, pin_template="f" * 64, with_file=True)
    monkeypatch.delenv("ARTIFACT_INTEGRITY_ENFORCE_IN_TESTS")
    assert oss.build_server_command(SimpleNamespace(name="frontdoor"), 8070)  # wrong pin ignored in tests


def test_start_server_wrapper_turns_the_refusal_into_none_and_logs_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(oss, "LOG_DIR", tmp_path / "logs")

    @oss._refuse_on_artifact_integrity_error
    def start_server(port, roles):
        raise ai.TemplatePinError("frontdoor", "file:/t", "a" * 64, "b" * 64)

    assert start_server(8070, ["frontdoor"]) is None
    assert "[REFUSED] frontdoor" in capsys.readouterr().out
    rows = [json.loads(ln) for ln in (tmp_path / "logs" / "artifact_integrity_refusals.jsonl").read_text().splitlines()]
    assert rows[0]["kind"] == "TemplatePinError" and rows[0]["roles"] == ["frontdoor"]


def test_start_server_wrapper_does_not_swallow_other_errors():
    @oss._refuse_on_artifact_integrity_error
    def start_server(port, roles):
        raise ValueError("unrelated")

    with pytest.raises(ValueError, match="unrelated"):
        start_server(1, ["x"])


def test_real_start_server_is_wrapped():
    assert getattr(oss.start_server, "__wrapped__", None) is not None
