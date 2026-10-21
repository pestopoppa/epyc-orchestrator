"""SEC-EXEC-1/2/4: minimal child env, no shell interpreters / exec forms, durable audit log."""

import json

import pytest

from src.repl_environment import REPLEnvironment
from src.repl_environment.exec_containment import child_env


@pytest.fixture
def audit(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_TOOL_AUDIT_LOG", str(path))
    return path


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_child_env_drops_secrets_keeps_basics(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE-planted-key-123456789")
    monkeypatch.setenv("HF_HOME", "/mnt/raid0/llm/cache/huggingface")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x" * 20)
    monkeypatch.setenv("LANG", "C.UTF-8")
    env = child_env({"EPYC_EVAL_FENCE_CONFIG_FILE": "/x"})
    assert "OPENAI_API_KEY" not in env and "HF_HOME" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["LANG"] == "C.UTF-8" and "PATH" in env
    assert env["EPYC_EVAL_FENCE_CONFIG_FILE"] == "/x"


def test_run_python_code_child_has_no_secrets(monkeypatch, audit):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE-planted-key-123456789")
    monkeypatch.setenv("HF_HOME", "/planted/hf")
    repl = REPLEnvironment(context="t")
    out = repl._run_python_code(
        "import os\nprint('K' if 'OPENAI_API_KEY' in os.environ else 'nokey',"
        " 'H' if 'HF_HOME' in os.environ else 'nohf')"
    )
    assert "nokey nohf" in out


def test_run_shell_child_has_no_secrets(monkeypatch, audit):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE-planted-key-123456789")
    monkeypatch.setenv("HF_HOME", "/planted/hf")
    repl = REPLEnvironment(context="t")
    # printf/echo do not expand env, so use a captured subprocess call to prove the env passed.
    import subprocess
    from unittest.mock import patch

    real = subprocess.run
    seen = {}

    def spy(*a, **kw):
        seen["env"] = kw.get("env")
        return real(*a, **kw)

    with patch("src.repl_environment.external_access.subprocess.run", spy):
        out = repl._run_shell("echo hi")
    assert "hi" in out
    assert seen["env"] is not None
    assert "OPENAI_API_KEY" not in seen["env"] and "HF_HOME" not in seen["env"]


@pytest.mark.parametrize("cmd", [
    "python3 -c 'print(1)'",
    "python -c 'print(1)'",
    "/usr/bin/python3 -c 'print(1)'",
])
def test_run_shell_refuses_python(cmd, audit):
    out = REPLEnvironment(context="t")._run_shell(cmd)
    assert out.startswith("[ERROR")


@pytest.mark.parametrize("cmd", [
    "find . -exec ls {} ;",
    "find . -delete",
    "find . -fprint /tmp/x",
    "awk 'BEGIN{system(\"id\")}'",
    "awk -f prog.awk",
    "sed -f prog.sed x",
    "sed 'e id' x",
])
def test_run_shell_refuses_exec_forms_unscoped(cmd, audit):
    out = REPLEnvironment(context="t")._run_shell(cmd)
    assert "EXEC CONTAINMENT" in out


def test_benign_shell_still_works(audit):
    out = REPLEnvironment(context="t")._run_shell("echo hello")
    assert "hello" in out


def test_refused_call_is_audited(audit):
    repl = REPLEnvironment(context="t", role="frontdoor")
    repl._run_shell("python3 -c 'print(1)'")
    repl._run_shell("echo ok")
    recs = _records(audit)
    assert [r["outcome"] for r in recs] == ["refused", "ok"]
    r = recs[0]
    assert r["tool"] == "run_shell" and r["role"] == "frontdoor"
    assert len(r["sha256"]) == 64 and r["refusal_reason"]
    assert recs[1]["exit"] == 0


def test_timeout_and_secret_redaction_audited(audit):
    repl = REPLEnvironment(context="t")
    repl._run_python_code("import time\ntime.sleep(5)", timeout=1)
    repl._run_python_code("k='sk-ant-api03-" + "A" * 40 + "'\nprint(1)")
    recs = _records(audit)
    assert recs[0]["outcome"] == "timeout" and recs[0]["tool"] == "run_python_code"
    assert recs[1]["exit"] == 0
    assert "A" * 40 not in json.dumps(recs[1])
