"""Codemode reliability fix 1: ORCHESTRATOR_REPL_EXPLICIT_FINAL gates auto_wrap_final's one-line branch (default off)."""

from __future__ import annotations

import pytest

from src.prompt_builders.code_utils import ENV_EXPLICIT_FINAL, auto_wrap_final

TOOL_CALLS = [
    'list_dir("data/inbox")',
    'file_info("data/x.json")',
    "list_tools()",
    'CALL("syn_sku_price", sku="SKU-1")',
    'TOOL("syn_sku_price", sku="SKU-1")',
    'read_file("data/a.json")',
    "json.loads(raw)",
]
ANSWER_VALUES = ['"done"', "42", "-3.5", "'ok'", "answer", "result", "final_answer", 'f"total={total}"', "answer = 42", 'result = "x"']
# Flag OFF behaviour is the pre-existing one: everything single-line and non-assignment is wrapped.
OFF_WRAPPED = TOOL_CALLS + [v for v in ANSWER_VALUES if "=" not in v or v.startswith(("answer =", "result ="))]


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.setenv(ENV_EXPLICIT_FINAL, "0")


@pytest.mark.parametrize("code", OFF_WRAPPED)
def test_flag_off_is_the_legacy_wrap(code):
    assert auto_wrap_final(code) == f"FINAL({code})"


@pytest.mark.parametrize("val", ["0", "off", "false", "no"])
def test_off_values_restore_legacy(monkeypatch, val):
    monkeypatch.setenv(ENV_EXPLICIT_FINAL, val)
    assert auto_wrap_final('list_dir("d")') == 'FINAL(list_dir("d"))'


def test_default_is_explicit_final(monkeypatch):
    monkeypatch.delenv(ENV_EXPLICIT_FINAL, raising=False)
    assert auto_wrap_final('list_dir("d")') == 'list_dir("d")'


@pytest.mark.parametrize("code", TOOL_CALLS)
def test_flag_on_never_wraps_a_tool_call(monkeypatch, code):
    monkeypatch.setenv(ENV_EXPLICIT_FINAL, "1")
    assert auto_wrap_final(code) == code


@pytest.mark.parametrize("code", ANSWER_VALUES)
def test_flag_on_wraps_clear_answer_values(monkeypatch, code):
    monkeypatch.setenv(ENV_EXPLICIT_FINAL, "1")
    assert auto_wrap_final(code) == f"FINAL({code})"


def test_flag_on_assignment_of_a_call_is_not_wrapped(monkeypatch):
    monkeypatch.setenv(ENV_EXPLICIT_FINAL, "1")
    assert auto_wrap_final("result = list_dir('d')") == "result = list_dir('d')"


def test_flag_on_leaves_explicit_final_exploration_and_multiline_alone(monkeypatch):
    monkeypatch.setenv(ENV_EXPLICIT_FINAL, "1")
    for code in ("FINAL('x')", "peek(10, file_path='a')", "a = 1\nprint(a)"):
        assert auto_wrap_final(code) == code


def test_flag_on_keeps_error_payload_and_pure_definition_wraps(monkeypatch):
    monkeypatch.setenv(ENV_EXPLICIT_FINAL, "1")
    assert auto_wrap_final("[ERROR: x]") == 'FINAL("[ERROR: x]")'
    assert auto_wrap_final("def f():\n    return 1").startswith("FINAL('''")


def test_every_call_site_goes_through_the_one_function():
    """The gate lives inside auto_wrap_final, so the five production call sites inherit it."""
    import inspect
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    sites = ["src/graph/helpers.py", "src/api/routes/openai_compat.py", "src/api/routes/chat_pipeline/stages.py",
             "src/api/routes/chat_delegation.py"]
    for rel in sites:
        assert "auto_wrap_final" in (root / rel).read_text(), rel
    assert "_explicit_final_mode()" in inspect.getsource(auto_wrap_final)


@pytest.mark.parametrize("code", ["x == 5", "result = compute_answer()", "json.loads(raw)", "len(items)", "total"])
def test_default_tools_mode_keeps_legacy_wrap_for_non_tool_lines(monkeypatch, code):
    monkeypatch.delenv(ENV_EXPLICIT_FINAL, raising=False)
    assert auto_wrap_final(code) == f"FINAL({code})"


@pytest.mark.parametrize("code", ['list_dir("d")', "list_tools()", 'CALL("t", a=1)', 'peek(5, file_path="f")'])
def test_default_tools_mode_never_wraps_repl_tool_calls(monkeypatch, code):
    monkeypatch.delenv(ENV_EXPLICIT_FINAL, raising=False)
    assert auto_wrap_final(code) == code
