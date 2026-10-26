"""The suite's pre-import runtime paths stay outside the checkout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_STORE_PREFIX = "orchestrator-pytest-stores-"


def _assert_private_runtime_path(path: Path) -> None:
    assert path.is_absolute()
    assert not path.resolve().is_relative_to(REPO_ROOT.resolve())
    assert path.parent.name.startswith(PRIVATE_STORE_PREFIX)


def test_tool_audit_and_seen_host_writes_use_private_preimport_path():
    """The real audit writer persists only beneath the private test runtime root."""
    audit_path = Path(os.environ["ORCHESTRATOR_TOOL_AUDIT_LOG"])
    _assert_private_runtime_path(audit_path)

    from src.repl_environment.exec_containment import audit_tool_call

    host = f"loop-harness-isolation-{uuid.uuid4().hex}.example.invalid"
    audit_tool_call(
        "web_fetch",
        text=f"https://{host}/probe",
        cwd=None,
        started=time.time(),
        outcome="ok",
    )

    seen_hosts = audit_path.with_name("tool_audit_seen_hosts.txt")
    assert host in seen_hosts.read_text().splitlines()
    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    matching_records = [record for record in records if record["host"] == host]
    assert len(matching_records) == 1
    assert matching_records[0]["first_seen_host"] is True


def test_ruff_subprocess_uses_private_cache(tmp_path: Path):
    """An actual Ruff invocation writes its cache outside the checkout."""
    cache_dir = Path(os.environ["RUFF_CACHE_DIR"])
    _assert_private_runtime_path(cache_dir)
    source = tmp_path / "cache_probe.py"
    source.write_text("answer = 42\n")

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(source)],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(path.is_file() for path in cache_dir.rglob("*"))
