"""AKU-25 follow-up: nested region-lock/gpu-quiet audit matrix (A-F).

Shared-in-shared nesting cannot deadlock and must succeed; exclusive overlaps
(same-role, cross-role, excl-outer/shared-inner) must fail fast.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp: Path, *cmd: str) -> tuple[int, str]:
    env = dict(os.environ, ORCHESTRATOR_TMP_DIR=str(tmp), EPYC_LOCK_FIFO="1", PYTHONPATH=str(ROOT))
    env.pop("EPYC_REGION_LOCK_ACTIVE", None)
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout + p.stderr


def _nest(tmp, outer, inner):
    rl = [sys.executable, "-m", "src.runtime.region_lock_cli", "run"]
    return _run(tmp, *rl, *outer, "--timeout-s", "5", "--", *rl, *inner, "--timeout-s", "5", "--", "echo", "INNER_OK")


CASES = [
    ("A", ["--cpu-list", "0-23", "--role", "bench"], ["--cpu-list", "0-23", "--role", "bench"], False),
    ("B", ["--cpu-list", "0-95", "--role", "bench"], ["--cpu-list", "0-95", "--role", "bench-canonical"], False),
    ("D", ["--gpu-quiet", "exclusive", "--role", "bench"], ["--gpu-quiet", "shared", "--cpu-list", "0-23", "--role", "b2"], False),
    ("C", ["--gpu-quiet", "shared", "--cpu-list", "0-23", "--role", "bench"], ["--gpu-quiet", "shared", "--cpu-list", "24-47", "--role", "bench2"], True),
    ("E", ["--cpu-list", "0-23", "--role", "bench"], ["--cpu-list", "24-47", "--role", "build"], True),
    ("F", ["--cpu-list", "0-11", "--role", "bench"], ["--cpu-list", "12-23", "--role", "bench"], True),
]


@pytest.mark.parametrize("name,outer,inner,ok", CASES, ids=[c[0] for c in CASES])
def test_nesting_matrix(tmp_path, name, outer, inner, ok):
    rc, out = _nest(tmp_path, outer, inner)
    ran = "INNER_OK" in out.splitlines()  # the echo's own line, not the "running:" banner
    if ok:
        assert ran and rc == 0 and "Nested" not in out, out
    else:
        assert not ran and rc != 0 and "NestedLockError" in out, out
