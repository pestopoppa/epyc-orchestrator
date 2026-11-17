"""Idempotent write tools (REPL hardening #3)."""

from types import SimpleNamespace

import pytest

from src.repl_environment import write_dedupe as wd
from src.repl_environment.file_mutation import _FileMutationMixin


class _Env(_FileMutationMixin):
    def __init__(self):
        self.calls = []
        self._exploration_calls = 0
        self._exploration_log = SimpleNamespace(add_event=lambda *a, **k: None)

    def _validate_file_path(self, p):
        return True, None


def test_file_write_same_content_skipped_but_changed_content_written(tmp_path):
    env, f = _Env(), str(tmp_path / "a.txt")
    assert env._file_write_safe(f, "one", backup=False).startswith("Wrote")
    assert "DUPLICATE_WRITE_SKIPPED" in env._file_write_safe(f, "one", backup=False)
    assert env._file_write_safe(f, "two", backup=False).startswith("Wrote")
    # A -> B -> A is NOT a duplicate: the last write to the path differed.
    assert env._file_write_safe(f, "one", backup=False).startswith("Wrote")
    assert open(f).read() == "one"


def test_failed_write_not_recorded(tmp_path):
    env = _Env()
    env._validate_file_path = lambda p: (False, "nope")
    assert env._file_write_safe("/x", "c").startswith("[ERROR")
    assert env._file_write_safe("/x", "c").startswith("[ERROR")  # retried, not skipped


def test_log_append_exact_repeat_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("src.repl_environment.file_mutation._get_project_root", lambda: tmp_path)
    env = _Env()
    assert env._log_append("t.log", "hello").startswith("Appended")
    assert "DUPLICATE_WRITE_SKIPPED" in env._log_append("t.log", "hello")
    assert env._log_append("t.log", "other").startswith("Appended")
    assert (tmp_path / "logs" / "t.log").read_text().count("hello") == 1


def test_killswitch(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_WRITE_DEDUPE", "0")
    env, f = _Env(), str(tmp_path / "a.txt")
    env._file_write_safe(f, "x", backup=False)
    assert env._file_write_safe(f, "x", backup=False).startswith("Wrote")


def test_guarded_call_envelope_failure_not_recorded():
    env = SimpleNamespace()
    n = {"c": 0}

    def fn():
        n["c"] += 1
        return SimpleNamespace(ok=False)

    wd.guarded_call(env, "t", ("t",), "same", fn)
    wd.guarded_call(env, "t", ("t",), "same", fn)
    assert n["c"] == 2


def test_invoke_tool_write_bucket_deduped():
    from src.repl_environment.context import _ContextMixin

    class E(_ContextMixin):
        def __init__(self):
            self.tool_registry = SimpleNamespace(_tools={
                "writer": SimpleNamespace(side_effects=["modifies_files"]),
                "reader": SimpleNamespace(side_effects=["read_only"]),
            })
            self._tool_invocations = 0
            self.n = 0

        def _dispatch_tool(self, name, **kw):
            self.n += 1
            return "ok"

    e = E()
    assert e._invoke_tool("writer", path="a") == "ok"
    assert "DUPLICATE_WRITE_SKIPPED" in e._invoke_tool("writer", path="a")
    assert e._invoke_tool("writer", path="b") == "ok"
    assert e._invoke_tool("reader", q=1) == "ok" and e._invoke_tool("reader", q=1) == "ok"
    assert e.n == 4
