"""Per-snippet tool-call budget and write-tool argument validation."""

import pytest

from src.repl_environment.environment import REPLEnvironment
from src.repl_environment.types import REPLConfig


def _env():
    return REPLEnvironment(context="alpha beta gamma", config=REPLConfig(timeout_seconds=20))


def test_budget_exceeded_in_one_snippet(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", "5")
    res = _env().execute("for i in range(6):\n    context_len()\n")
    assert res.error and "Tool-call budget exceeded" in res.error


def test_budget_resets_per_snippet_and_under_cap_ok(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", "5")
    env = _env()
    for _ in range(3):
        assert env.execute("for i in range(5):\n    context_len()\n").error is None


def test_default_1000_and_disable(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", raising=False)
    assert REPLEnvironment._max_tool_calls() == 1000
    res = _env().execute("for i in range(1001):\n    context_len()\n")
    assert res.error and "Tool-call budget exceeded" in res.error
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", "0")
    assert _env().execute("for i in range(1500):\n    context_len()\n").error is None


def test_final_not_charged(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", "1")
    res = _env().execute("context_len()\nFINAL('x')")
    assert res.is_final and res.error is None


@pytest.mark.parametrize(
    "call",
    ["file_write_safe(5, 'x')", "file_write_safe('/tmp/a', None)", "file_write_safe('a\\x00b', 'x')",
     "log_append(None, 'm')", "log_append('l', 3)"],
)
def test_write_tools_validate_args(call):
    res = _env().execute(f"print({call})")
    assert "[ERROR:" in res.output and "must be" in res.output or "NUL" in res.output


def test_patch_methods_validate_args_directly():
    env = _env()
    assert env._prepare_patch("notalist", "d").startswith("[ERROR:")
    assert env._prepare_patch([1], "d").startswith("[ERROR:")
    assert env._apply_approved_patch(None).startswith("[ERROR:")
    assert env._reject_patch("p", None).startswith("[ERROR:")
