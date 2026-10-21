"""SEC-EXEC-6 follow-ups: token-free nextplaid mounts, reappeared-token warning, /proc environ bypass.

The last test class is EMPIRICAL: it demonstrates that SEC-EXEC-1's environment scrub alone does
not hide the API process's secrets from a same-uid child (/proc/<pid>/environ), and that the
exec_containment flag does.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from src.features import reset_features
from src.repl_environment import REPLEnvironment, exec_sandbox
from src.repl_environment.exec_containment import child_env

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def audit(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TOOL_AUDIT_LOG", str(tmp_path / "audit.jsonl"))


@pytest.fixture
def flags(monkeypatch):
    def _set(**kw):
        for name, on in kw.items():
            monkeypatch.setenv("ORCHESTRATOR_" + name.upper(), "1" if on else "0")
        reset_features()
        exec_sandbox.reset_caches()

    yield _set
    monkeypatch.undo()
    reset_features()
    exec_sandbox.reset_caches()


# ── nextplaid mounts ────────────────────────────────────────────────────────


def test_no_docker_service_mounts_the_huggingface_cache_root():
    manifest = yaml.safe_load((ROOT / "orchestration" / "launch_manifest.yaml").read_text())
    seen = 0
    for svc in manifest["docker_services"]:
        for vol in svc.get("volumes", []):
            src = vol.split(":", 1)[0].rstrip("/")
            assert not src.endswith("/huggingface"), f"{svc['name']} mounts the HF cache root: {vol}"
            seen += "huggingface/hub" in src
    assert seen == 2, "both nextplaid services should mount only huggingface/hub"


# ── token reappearance guard ────────────────────────────────────────────────


def test_reappeared_token_files_are_found_by_existence_only(tmp_path):
    hf, home = tmp_path / "hf", tmp_path / "home"
    hf.mkdir()
    env = {"HF_HOME": str(hf)}
    mine = {str(hf / "token"), str(hf / "stored_tokens")}
    assert mine.isdisjoint(exec_sandbox.reappeared_hf_token_files(env, [str(home)]))
    (hf / "token").write_text("hf_FAKE")
    (hf / "stored_tokens").write_text("hf_FAKE")
    assert mine <= set(exec_sandbox.reappeared_hf_token_files(env, [str(home)]))


def test_reappeared_token_warns_loudly_and_is_rate_limited(tmp_path, monkeypatch, caplog):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "token").write_text("hf_FAKE")
    monkeypatch.setattr(exec_sandbox, "_token_warned", {})
    env = {"HF_HOME": str(hf)}
    with caplog.at_level(logging.ERROR):
        exec_sandbox.warn_if_hf_token_files(env, [str(tmp_path / "home")])
        exec_sandbox.warn_if_hf_token_files(env, [str(tmp_path / "home")])
    hits = [r for r in caplog.records if "HF TOKEN FILE PRESENT" in r.getMessage() and str(hf / "token") in r.getMessage()]
    assert len(hits) == 1 and hits[0].levelno == logging.ERROR
    assert "hf_FAKE" not in caplog.text  # the value is never read or logged


def test_child_env_triggers_the_warning_even_with_the_flag_off(tmp_path, monkeypatch, caplog, flags):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "stored_tokens").write_text("hf_FAKE")
    monkeypatch.setattr(exec_sandbox, "_token_warned", {})
    monkeypatch.setenv("HF_HOME", str(hf))
    flags(exec_containment=False)
    with caplog.at_level(logging.ERROR):
        child_env(task_id="g")
    assert any("HF TOKEN FILE PRESENT" in r.getMessage() and "stored_tokens" in r.getMessage() for r in caplog.records)


# ── /proc/<pid>/environ: the SEC-EXEC-1 scrub bypass ────────────────────────

PEER_SECRET = "sk-FAKE-planted-api-secret-0123456789"


@pytest.fixture
def peer():
    """A same-uid process whose launch environment carries a planted secret (stands in for the API)."""
    p = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        env={"PATH": os.environ.get("PATH", ""), "OPENAI_API_KEY": PEER_SECRET},
    )
    time.sleep(0.3)
    yield p
    p.kill()
    p.wait(timeout=10)


def _scan_code(pid):
    return (
        "import os\n"
        "print('OWN_ENV_HAS_KEY', 'OPENAI_API_KEY' in os.environ)\n"
        f"try:\n    data = open('/proc/{pid}/environ','rb').read()\n    print('PEER_SECRET_RECOVERED', {PEER_SECRET.encode()!r} in data)\n"
        "except PermissionError:\n    print('PEER_ENVIRON_DENIED')\n"
        "try:\n    data = open('/proc/%d/environ' % os.getppid(),'rb').read()\n"
        "    print('PARENT_REAL_HOME_RECOVERED', os.fsencode(" + repr(os.path.expanduser("~")) + ") in data)\n"
        "except PermissionError:\n    print('PARENT_ENVIRON_DENIED')\n"
    )


def test_env_scrub_alone_does_not_hide_a_peers_secrets_FINDING(peer, audit, flags):
    """Empirical confirmation of the finding: with exec_containment OFF a child reads /proc/<pid>/environ."""
    flags(exec_containment=False)
    out = REPLEnvironment(context="t")._run_python_code(_scan_code(peer.pid))
    assert "OWN_ENV_HAS_KEY False" in out, out  # SEC-EXEC-1 holds: the child's own environment is clean
    assert "PEER_SECRET_RECOVERED True" in out, (
        "finding NOT confirmed on this host (peer environ unreadable even unsandboxed): " + out
    )
    assert "PARENT_REAL_HOME_RECOVERED True" in out, out  # the REPL process's own launch env is readable too


def test_containment_closes_the_proc_environ_route(peer, audit, flags):
    c = exec_sandbox.capabilities()
    if not (c["landlock"] and c["netdeny"]):
        pytest.skip(f"host cannot enforce kernel containment: {c}")
    flags(exec_containment=True)
    out = REPLEnvironment(context="t")._run_python_code(_scan_code(peer.pid))
    assert "OWN_ENV_HAS_KEY False" in out, out
    assert "PEER_ENVIRON_DENIED" in out and "PEER_SECRET_RECOVERED" not in out, out
    assert "PARENT_ENVIRON_DENIED" in out and "PARENT_REAL_HOME_RECOVERED" not in out, out
    assert PEER_SECRET not in out
