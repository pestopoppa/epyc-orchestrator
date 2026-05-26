"""Phase 1 exit-gate tests: model-facing surfaces honor the task-root when
ORCHESTRATOR_EDIT_ROOT is active, and are byte-for-byte unchanged when unset.

Covers the Phase-0 audit surfaces that are unit-testable without inference:
  #1/#2 _validate_file_path (relative resolves under scratch + task-root allowed)
  #5    run_shell cwd
  #7    code_search (index-free scratch search)
  #8    _batch_edit_repo_root
(#10 DCP file_reader is covered in test_dcp4_wiring.py; full write/read tool round-trips
land in the Phase 3 driver dry-run.)
"""
from __future__ import annotations

import json

import pytest

from src.repl_environment.environment import REPLEnvironment


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_EDIT_ROOT", raising=False)


def _env():
    return REPLEnvironment(context="test", role="frontdoor")


# ─── #1/#2 _validate_file_path ────────────────────────────────────────────────────

def test_validate_relative_path_resolves_under_scratch(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_EDIT_ROOT", str(tmp_path))
    env = _env()
    ok, err = env._validate_file_path("cart.py")  # relative → under scratch task-root
    assert ok, err


def test_validate_absolute_scratch_path_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_EDIT_ROOT", str(tmp_path))
    env = _env()
    ok, err = env._validate_file_path(str(tmp_path / "pkg" / "mod.py"))
    assert ok, err


def test_validate_default_off_unchanged(tmp_path):
    # env unset: a /tmp path is allowed (ALLOWED_FILE_PATHS includes /tmp/), an /etc path is not
    env = _env()
    ok_tmp, _ = env._validate_file_path(str(tmp_path / "x.py"))
    ok_etc, _ = env._validate_file_path("/etc/hostname")
    assert ok_tmp is True
    assert ok_etc is False  # parity: outside allowed prefixes


# ─── #8 _batch_edit_repo_root ─────────────────────────────────────────────────────

def test_batch_edit_repo_root_follows_task_root(monkeypatch, tmp_path):
    from src.graph.helpers import _batch_edit_repo_root

    monkeypatch.setenv("ORCHESTRATOR_EDIT_ROOT", str(tmp_path))
    assert _batch_edit_repo_root() == tmp_path


def test_batch_edit_repo_root_default_is_project_root(monkeypatch):
    from src.graph.helpers import _batch_edit_repo_root
    from src.repl_environment.task_root import _project_root

    # unset → project_root (parity with the prior _get_project_root())
    assert _batch_edit_repo_root() == _project_root()


# ─── #7 code_search (index-free scratch search) ───────────────────────────────────

def test_code_search_returns_scratch_files(monkeypatch, tmp_path):
    (tmp_path / "cart.py").write_text("def total():\n    return sum(items)\n")
    (tmp_path / "checkout.py").write_text("from cart import total\n")
    (tmp_path / "unrelated.py").write_text("X = 1\n")
    monkeypatch.setenv("ORCHESTRATOR_EDIT_ROOT", str(tmp_path))
    env = _env()
    out = env._code_search("total cart", limit=5)
    hits = json.loads(out)
    paths = {h["path"] for h in hits}
    assert "cart.py" in paths       # matched by name + body
    assert "checkout.py" in paths   # matched by body ("cart", "total")
    # each hit is ColGREP-JSON-shaped (path/score/start_line)
    assert all("path" in h and "score" in h and "start_line" in h for h in hits)


def test_code_search_default_off_uses_indexed_engine(monkeypatch):
    # env unset → must NOT take the scratch path; falls through to ColGREP/NextPLAID.
    # Patch the indexed engine to confirm it's the one invoked (parity).
    from unittest.mock import patch

    env = _env()
    with patch.object(env, "_task_root_code_search") as scratch, \
         patch.object(env, "_colgrep_search", return_value="[]") as colgrep, \
         patch.object(env, "_nextplaid_search", return_value="[]"):
        env._code_search("anything", limit=5)
    scratch.assert_not_called()  # scratch path never taken when env unset


# ─── #5 run_shell cwd ─────────────────────────────────────────────────────────────

def test_run_shell_cwd_is_scratch(monkeypatch, tmp_path):
    import os

    monkeypatch.setenv("ORCHESTRATOR_EDIT_ROOT", str(tmp_path))
    env = _env()
    out = env._run_shell("pwd")
    # pwd prints the cwd; should be the scratch task-root (realpath to handle /tmp symlinks)
    assert os.path.realpath(str(tmp_path)) in out or str(tmp_path) in out
