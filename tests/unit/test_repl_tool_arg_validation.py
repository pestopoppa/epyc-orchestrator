"""D-RI-CONTAIN item (e): per-tool argument validation inside each exposed REPL tool."""

from __future__ import annotations

import inspect

import pytest

from src.repl_environment import tool_arg_validation as tav
from src.repl_environment.environment import REPLEnvironment
from src.repl_environment.types import REPLConfig

CONTEXT = "alpha beta gamma\nline two beta"


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    for k in ("ORCHESTRATOR_REPL_TOOL_ARG_VALIDATION", "ORCHESTRATOR_REPL_CHILD_LANE"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def validated(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_TOOL_ARG_VALIDATION", "1")


def _env():
    return REPLEnvironment(context=CONTEXT, config=REPLConfig(timeout_seconds=20))


def _run(env, expr):
    r = env.execute(f"print({expr})")
    return r.output.strip(), r.error


BAD_STR_RETURNING = [
    ("peek(n=[1])", "peek"),
    ("peek(n=True)", "peek"),
    ("peek(n=10**12)", "peek"),
    ("peek(file_path=5)", "peek"),
    ("peek(file_path='a\\x00b')", "peek"),
    ("peek(file_path='')", "peek"),
    ("peek(offset='abc')", "peek"),
    ("read_file(None)", "read_file"),
    ("read_file(7)", "read_file"),
    ("list_dir('x\\x00')", "list_dir"),
    ("list_dir(['a'])", "list_dir"),
    ("file_info('a' * 5000)", "file_info"),
    ("file_info('a\\nb\\x01')", "file_info"),
    ("web_fetch('file:///etc/passwd')", "web_fetch"),
    ("web_fetch('http://a b/')", "web_fetch"),
    ("web_fetch('https://example.com', max_chars=-1)", "web_fetch"),
    ("web_search(None)", "web_search"),
    ("web_search('q', max_results=10**6)", "web_search"),
    ("web_search('q' * 20000)", "web_search"),
    ("batch_web_search('notalist')", "batch_web_search"),
    ("batch_web_search([1, 2])", "batch_web_search"),
    ("run_shell(['ls'])", "run_shell"),
    ("run_shell('echo\\x00x')", "run_shell"),
    ("run_shell('echo', timeout=0)", "run_shell"),
    ("run_shell('echo', timeout='5')", "run_shell"),
    ("run_python_code(None)", "run_python_code"),
    ("run_python_code('1', timeout=10**6)", "run_python_code"),
    ("peek_grep('f', 5)", "peek_grep"),
    ("recall('q', limit=0)", "recall"),
    ("code_search(['q'])", "code_search"),
    ("doc_search('q', limit='x')", "doc_search"),
    ("delegate(None)", "delegate"),
    ("delegate('b', to='bad role!')", "delegate"),
    ("delegate('b', parallel='yes')", "delegate"),
    ("escalate(3)", "escalate"),
    ("registry_lookup(None)", "registry_lookup"),
    ("registry_update('k\\x00', 1)", "registry_update"),
    ("benchmark_run('m', n_tokens=0)", "benchmark_run"),
    ("gate_run(gates='lint')", "gate_run"),
    ("gate_run(fix='yes')", "gate_run"),
    ("fetch_report('../../etc/passwd')", "fetch_report"),
    ("fetch_report('r', max_chars=0)", "fetch_report"),
    ("checkpoint_create(None)", "checkpoint_create"),
    ("checkpoint_restore('a b c')", "checkpoint_restore"),
    ("remember(5)", "remember"),
    ("FINAL_VAR(None)", "FINAL_VAR"),
    ("STUCK(None)", "STUCK"),
    ("archive_open(None)", "archive_open"),
    ("extract_figure('p.pdf', 'one', [1, 2, 3, 4])", "extract_figure"),
    ("extract_figure('p.pdf', 1, [1, 2, 3])", "extract_figure"),
    ("route_advice(None)", "route_advice"),
    ("run_procedure(None)", "run_procedure"),
]


@pytest.mark.parametrize("call,tool", BAD_STR_RETURNING)
def test_bad_arguments_are_refused_inside_the_tool(call, tool, validated):
    out, err = _run(_env(), call)
    assert err is None and out.startswith(f"[ERROR: {tool}: argument "), (out, err)


def test_grep_reports_in_its_own_list_convention(validated):
    out, _ = _run(_env(), "grep(5)")
    assert out.startswith("""["[ERROR: grep: argument 'pattern'""")
    assert _run(_env(), "grep('beta', context_lines=-1)")[0].startswith('["[ERROR: grep:')
    assert _run(_env(), "grep('x' * 100000)")[0].startswith('["[ERROR: grep:')


@pytest.mark.parametrize("call", ["chunk_context(0)", "chunk_context('4')", "chunk_context(2, -5)", "mark_finding(None)",
                                  "mark_finding('f', tags='x')", "summarize_chunks(5)", "TOOL(5)", "TOOL('a b')",
                                  "CALL('x' * 500)"])
def test_non_string_tools_raise_a_value_error(call, validated):
    from unittest.mock import MagicMock

    env = REPLEnvironment(context=CONTEXT, config=REPLConfig(timeout_seconds=20), tool_registry=MagicMock())
    r = env.execute(call)
    assert r.error and r.error.startswith("ToolArgumentError:") and "argument" in r.error, r.error


def test_kwarg_values_of_tool_and_call_are_bounded(validated, monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr(tav, "KWARG_STR_CAP", 100)
    env = REPLEnvironment(context=CONTEXT, config=REPLConfig(timeout_seconds=20), tool_registry=MagicMock())
    r = env.execute("TOOL('t', q='x' * 101)")
    assert r.error and "exceeds 100 chars" in r.error
    r = env.execute("TOOL('t', q='a\\x00b')")
    assert r.error and "NUL" in r.error


@pytest.mark.parametrize(
    "call,expected",
    [
        ("peek(5)", "alpha"),
        ("peek(n='5')", "alpha"),          # the tool itself coerces numeric strings: still accepted
        ("peek(5, offset=-5)", "beta"),
        ("peek(n=5.0)", "alpha"),
        ("context_len()", "30"),
        ("grep('GAMMA')", "['alpha beta gamma']"),
        ("chunk_context(2)[0]['index']", "0"),
    ],
)
def test_valid_calls_are_unchanged_by_the_flag(call, expected, validated):
    out, err = _run(_env(), call)
    assert err is None and (out == expected or out.startswith(expected)), (call, out)


def test_missing_and_extra_arguments_keep_python_errors(validated):
    r = _env().execute("peek(1, 2, 3, 4)")
    assert r.error and r.error.startswith("TypeError")
    r = _env().execute("read_file()")
    assert r.error and r.error.startswith("TypeError")


def test_flag_off_returns_the_very_same_objects_and_does_not_intercept():
    assert tav.wrap_tool("peek", print) is print
    e = _env()
    assert "needs integer" in _run(e, "peek(n=[1])")[0]  # reached the tool's own check, not the validator
    # a value the validator would refuse (> 50M) reaches the tool unchanged
    assert "[ERROR: peek: argument" not in _run(e, "peek(n=10**9, offset=0)")[0]


def test_flag_on_wraps_every_spec_name_that_is_exposed(validated):
    from unittest.mock import MagicMock

    env = REPLEnvironment(context=CONTEXT, config=REPLConfig(timeout_seconds=20), tool_registry=MagicMock(),
                          llm_primitives=MagicMock())
    for name in tav.TOOL_SPECS:
        if name in env._globals:
            assert getattr(env._globals[name], "__wrapped__", None) is not None, name


def test_specs_only_name_real_parameters():
    env = _env()
    from unittest.mock import MagicMock

    full = REPLEnvironment(context=CONTEXT, config=REPLConfig(), tool_registry=MagicMock(), llm_primitives=MagicMock())
    for name, spec in tav.TOOL_SPECS.items():
        fn = full._globals.get(name)
        if fn is None:
            continue
        fn = inspect.unwrap(fn)
        params = inspect.signature(fn).parameters
        for arg in spec.args:
            assert arg in params, f"{name}: spec names {arg!r} which is not a parameter ({list(params)})"
    del env


def test_broker_validates_even_with_the_flag_off():
    e = _env()
    got = tav.call_validated("peek", e._peek, [], {"n": [1]})
    assert got.startswith("[ERROR: peek: argument 'n'")
    assert tav.call_validated("context_len", e._context_len, [], {}) == len(CONTEXT)


def test_write_tools_keep_their_existing_decorator_validation(validated):
    assert "must be" in _run(_env(), "file_write_safe(5, 'x')")[0]
    assert "must be" in _run(_env(), "log_append(None, 'm')")[0]
    assert not {"file_write_safe", "log_append", "prepare_patch", "apply_approved_patch", "reject_patch"} & set(tav.TOOL_SPECS)


def test_child_lane_flag_implies_validation(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_LANE", "1")
    assert tav.flag_enabled()
