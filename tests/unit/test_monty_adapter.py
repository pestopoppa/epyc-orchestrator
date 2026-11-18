"""D-RI-MONTY-LANE: the Monty REPL execution lane (opt-in, default OFF).

Needs ``pydantic-monty`` (extra ``monty``); everything except the flag-off tests skips without it. Pure CPU, no
servers. The lane's contract: flag ON can never make a cell fail that CPython would run. The suite-wide parity runs
use ``-p tests.monty_lane_parity`` (see that module); this file tests the lane's own mechanisms.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from src.features import Features, _FEATURE_REGISTRY, reset_features, set_features
from src.repl_environment import REPLConfig, REPLEnvironment

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_HAS_MONTY = importlib.util.find_spec("pydantic_monty") is not None
needs_monty = pytest.mark.skipif(not _HAS_MONTY, reason="pydantic-monty (extra `monty`) not installed")


@pytest.fixture(autouse=True)
def _flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MONTY_LANE_LOG", str(tmp_path / "lane.jsonl"))
    set_features(Features(monty_repl=True))
    yield
    reset_features()


def _ad(env, **kw):
    from src.repl_environment.monty_adapter import MontyREPLAdapter

    return MontyREPLAdapter(env, **kw)


def _last(ad):
    return ad.route_log[-1]


# ---- flag off: untouched ---------------------------------------------------------------------------------------
def test_flag_is_default_off_in_test_and_prod():
    spec = next(s for s in _FEATURE_REGISTRY if s.name == "monty_repl")
    assert spec.default_test is False and spec.default_prod is False
    assert Features().monty_repl is False


def test_flag_off_leaves_the_environment_byte_identical():
    """Flag OFF: create_repl_environment returns the plain environment and never imports the lane."""
    code = (
        "import sys\n"
        "from src.features import Features, set_features\n"
        "set_features(Features())\n"
        "from src.repl_environment.environment import create_repl_environment, REPLEnvironment\n"
        "env = create_repl_environment(context='c', use_restricted_python=False)\n"
        "assert type(env) is REPLEnvironment and 'execute' not in vars(env) and 'restore' not in vars(env)\n"
        "assert getattr(env, '_monty_lane', None) is None\n"
        "assert env.execute('print(1)').output == '1\\n'\n"
        "assert 'src.repl_environment.monty_adapter' not in sys.modules\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", ".")})
    assert r.stdout.strip().endswith("ok"), r.stderr[-2000:]


def test_per_request_path_and_factory_install_the_lane():
    src = (_ROOT / "src/api/routes/chat_pipeline/repl_executor.py").read_text()
    assert "install_monty_lane(repl)" in src
    env_src = (_ROOT / "src/repl_environment/environment.py").read_text()
    assert "return install_monty_lane(REPLEnvironment(" in env_src


def test_autopilot_can_hot_swap_the_flag():
    sys.path.insert(0, str(_ROOT / "scripts" / "autopilot"))
    try:
        spec = importlib.util.spec_from_file_location("_ca", _ROOT / "scripts/autopilot/config_applicator.py")
        text = spec.origin and pathlib.Path(spec.origin).read_text()
    finally:
        sys.path.pop(0)
    assert '"monty_repl",' in text
    assert "monty_repl" in Features().summary()  # POST /config applies any key of the summary
    assert "| `monty_repl` |" in (_ROOT / "scripts/autopilot/program.md").read_text()


@needs_monty
def test_adapter_refuses_without_flag():
    from src.repl_environment.monty_adapter import MontyUnavailable, maybe_monty_adapter, monty_enabled

    set_features(Features())
    assert monty_enabled() is False
    env = REPLEnvironment(context="c")
    with pytest.raises(MontyUnavailable):
        _ad(env)
    assert maybe_monty_adapter(env) is None


# ---- static helpers --------------------------------------------------------------------------------------------
@needs_monty
def test_binding_names_include_function_globals():
    import ast

    from src.repl_environment.monty_adapter import _binding_names

    code = "a = 1\nfor i in range(3):\n    b = i\nwith x as c:\n    pass\nimport os.path as p\nfrom m import n\n" \
           "def f():\n    global g\n    inner = 1\nclass K:\n    pass\n[j for j in range(2)]\nz += 1\nif (w := 3): pass\n"
    assert _binding_names(ast.parse(code)) == ["a", "i", "b", "c", "p", "n", "f", "g", "K", "z", "w"]


@needs_monty
@pytest.mark.parametrize("code,reason", [
    ("import statistics\nprint(statistics.mean([1, 2]))", "import:statistics"),
    ("print(fractions.Fraction(1, 3))", "module:fractions"),
    ("def g():\n    yield 1\nprint(list(g()))", "feature:generator"),
    ("class E(Exception):\n    pass", "feature:class_inheritance"),
    ("class C:\n    @property\n    def p(self):\n        return 1", "feature:method_decorator"),
    ("x = 1\ndel x", "feature:del"),
    ("print({3, 1, 2})", "feature:set_order"),
    ("print(set([3, 1]))", "feature:set_order"),
    ("from collections import OrderedDict", "import:collections.OrderedDict"),
    ("import heapq", "import:heapq"),
    ("import os", "ast_gate"),
    ("print((1,", "syntax:cpython_rejects"),
    ("remember('x')", "tool_reads_cell_state:remember"),
    ("llm_batch([peek(1, 1)])", "name:undefined:llm_batch"),
    ("undefined_thing", "name:undefined:undefined_thing"),
    ("print(1 + 1)", "in_subset"),
])
def test_static_screen(code, reason):
    from src.repl_environment.monty_adapter import screen

    env = REPLEnvironment(context="c")
    ad = _ad(env)
    assert screen(ad, code).reason == reason
    r1 = REPLEnvironment(context="c").execute(code)
    r2 = ad.execute(code)  # and the routed result is CPython's
    assert (r1.output, r1.error, r1.is_final) == (r2.output, r2.error, r2.is_final)
    ad.close()


@needs_monty
def test_screen_routes_cells_reading_untransferable_state():
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        assert ad.execute("from collections import Counter\nc = Counter('aab')").error is None  # falls back
        assert _last(ad)["fallback"].startswith("state:not_transferable:c")
        r = ad.execute("print(c.most_common(1))")
        assert r.output == "[('a', 2)]\n" and _last(ad)["screen"] == "state:not_transferable:c"


# ---- the slice's twelve cases still hold -----------------------------------------------------------------------
def _load_harness():
    spec = importlib.util.spec_from_file_location("monty_slice_compare", _ROOT / "scripts/benchmark/monty_slice_compare.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["monty_slice_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


_H = _load_harness() if _HAS_MONTY else None


@needs_monty
@pytest.mark.parametrize("cid", [c.id for c in _H.CASES] if _H else [])
def test_twelve_cases_match_cpython(cid):
    case = next(c for c in _H.CASES if c.id == cid)
    cp_env = _H._mk_parent(case)
    cp = [cp_env.execute(c) for c in case.cells]
    env = _H._mk_parent(case)
    with _ad(env) as ad:
        mo = [ad.execute(c) for c in case.cells]
        assert case.oracle(mo, env)
        for a, b in zip(cp, mo):  # byte-identical now, including C07 (Monty-rejected syntax runs on CPython)
            assert (a.output, a.error, a.is_final, a.final_answer) == (b.output, b.error, b.is_final, b.final_answer)
        assert _H._state(cp_env) == _H._state(env)


@needs_monty
def test_security_refusals_are_the_repl_gate():
    for code in ["import os", "from os import system", "eval('1+1')", "''.__class__", "open('/etc/hostname')"]:
        e1 = REPLEnvironment(context="c")
        with _ad(REPLEnvironment(context="c")) as ad:
            assert ad.execute(code).error == e1.execute(code).error
            assert _last(ad)["route"] == "cpython" and _last(ad)["screen"] == "ast_gate"


# ---- callable bridge -------------------------------------------------------------------------------------------
class _Prims:
    def __init__(self):
        self.calls = []

    def llm_call(self, prompt, **kw):
        self.calls.append(("call", prompt))
        return f"answer:{prompt}"

    def llm_batch(self, prompts, **kw):
        self.calls.append(("batch", tuple(prompts)))
        return [f"a:{p}" for p in prompts]


class _Registry:
    def __init__(self):
        self.calls = []

    def invoke(self, name, role, **kw):
        self.calls.append((name, kw))
        return {"tool": name, "args": kw}

    def list_tools(self, role=None):
        return [{"name": "echo", "description": "echo back", "parameters": {"x": {"type": "string"}}}]

    def get_read_only_tools(self):
        return {"echo"}

    def can_use_tool(self, role, name):
        return True


def _full_env(**kw):
    return REPLEnvironment(context="alpha beta\ngamma delta\n", llm_primitives=_Prims(), tool_registry=_Registry(), **kw)


@needs_monty
def test_every_repl_callable_is_bridged_to_the_parents_own_global():
    """Each REPL global is reachable from Monty and the call lands in the parent's global (whatever wraps it)."""
    from src.repl_environment.monty_adapter import tool_names

    env = _full_env()
    ad = _ad(env)
    names = sorted(n for n in tool_names(env) if callable(env._globals.get(n)) and not n.startswith("_")
                   and type(env._globals[n]).__name__ != "module" and n not in ("FINAL", "FINAL_VAR"))
    assert {"peek", "grep", "list_dir", "file_write_safe", "llm_call", "llm_batch", "TOOL", "CALL", "list_tools",
            "run_shell", "file_info", "ocr_document", "context_len"} <= set(names)
    seen = []
    for n in names:
        if n in ("remember", "SCRIPT", "checkpoint_create", "checkpoint_restore"):
            continue  # screened to CPython (they read the cell's namespace)
        env._globals[n] = (lambda name: (lambda *a, **k: seen.append((name, a, k)) or f"{name}-ok"))(n)
        r = ad.execute(f"r = {n}(1, k='v')\nprint(r)")
        assert r.error is None and r.output == f"{n}-ok\n", (n, r)
        assert _last(ad)["route"] == "monty" and _last(ad)["fallback"] is None, (n, _last(ad))
        assert seen[-1] == (n, (1,), {"k": "v"})
    ad.close()


@needs_monty
@pytest.mark.parametrize("cells", [
    ["print(peek(5, file_path='/etc/hostname'))"],
    ["print(grep('a', file_path='/etc/hostname'))"],
    ["print(list_dir('/definitely/not/here'))"],
    ["print(file_info('/etc/shadow'))"],
    ["print(context_len())", "print(peek(5))", "print(grep('gamma'))"],
    ["print(llm_call('hi'))", "print(llm_batch(['a', 'b']))"],
    ["print(list_tools())", "print(TOOL('echo', x='1'))", "print(CALL('echo', x='2'))"],
    ["x = peek(3)\nFINAL(x)"],
    ["artifacts['k'] = 'v'\nFINAL_VAR('k')"],
    ["FINAL_VAR('missing')"],
    ["FINAL(result='ok', extra='bad')"],
])
def test_real_tools_match_cpython(cells):
    e1, e2 = _full_env(), _full_env()
    with _ad(e2) as ad:
        for c in cells:
            a, b = e1.execute(c), ad.execute(c)
            assert (a.output, a.error, a.is_final, a.final_answer) == (b.output, b.error, b.is_final, b.final_answer), c
        assert all(r["route"] == "monty" for r in ad.route_log), ad.route_log
    assert e1.llm_primitives.calls == e2.llm_primitives.calls
    assert e1.tool_registry.calls == e2.tool_registry.calls


@needs_monty
def test_file_write_goes_through_the_repl_path_and_dedupe(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_ALLOWED_PATHS", str(tmp_path))
    target = str(tmp_path / "out.txt")
    e1 = _full_env()
    with _ad(_full_env()) as ad:
        cell = f"print(file_write_safe({target!r}, 'hello'))\nprint(file_write_safe({target!r}, 'hello'))"
        a = e1.execute(cell)
        os.remove(target) if os.path.exists(target) else None
        b = ad.execute(cell)
        assert (a.output, a.error) == (b.output, b.error) and _last(ad)["route"] == "monty"


@needs_monty
def test_tool_call_budget_and_validation_are_the_parents(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", "5")
    e1 = REPLEnvironment(context="ctx")
    with _ad(REPLEnvironment(context="ctx")) as ad:
        code = "for i in range(20):\n    peek(1)"
        r = ad.execute(code)
        assert r.error == e1.execute(code).error and "Tool-call budget exceeded" in r.error
        assert _last(ad)["route"] == "monty" and _last(ad)["fallback"] is None


@needs_monty
def test_tool_errors_surface_with_cpython_text_and_are_not_retried():
    class Boom(REPLEnvironment):
        def _list_dir(self, path="."):
            raise ValueError(f"no such dir {path}")

    e1, e2 = Boom(context="c"), Boom(context="c")
    with _ad(e2) as ad:
        a, b = e1.execute("list_dir('x')"), ad.execute("list_dir('x')")
        assert a.error == b.error == "ValueError: no such dir x"
        assert _last(ad)["route"] == "monty" and _last(ad)["fallback"] is None


@needs_monty
def test_codemode_helpers_are_bridged(monkeypatch):
    from src.repl_environment import codemode_tools

    monkeypatch.setenv(codemode_tools.ENV_HANDLES, "1")
    monkeypatch.setenv(codemode_tools.ENV_SEARCH, "1")
    e1, e2 = _full_env(), _full_env()
    with _ad(e2) as ad:
        for c in ["print(store('k', [1, 2]))", "print(load('handle:k'), handles())", "print(search_tools('echo'))",
                  "print(describe_tool('echo'))", "store('bad', 1)\n1/0", "print(handles())"]:
            a, b = e1.execute(c), ad.execute(c)
            assert (a.output, a.error) == (b.output, b.error), c
            assert _last(ad)["route"] == "monty", _last(ad)
        assert e1._handle_store.snapshot() == e2._handle_store.snapshot()  # the failing cell's write rolled back


# ---- fallback: re-run on CPython from the pre-cell state, callbacks replayed once ------------------------------
@needs_monty
def test_runtime_gap_falls_back_with_no_half_applied_state():
    env = _full_env()
    with _ad(env) as ad:
        ad.execute("keep = [1]")
        r = ad.execute("keep.append(2)\nn = llm_call('once')\nt = 'abc'.translate({97: 98})\nprint(n, t, keep)")
        assert r.error is None and r.output == "answer:once bbc [1, 2]\n"
        row = _last(ad)
        assert row["route"] == "cpython" and row["fallback"] == "runtime:AttributeError"
        assert row["replayed"] == 1 and not row["replay_diverged"]
        assert env.llm_primitives.calls == [("call", "once")]  # the tool ran ONCE across both attempts
        assert env._globals["keep"] == [1, 2]  # appended once, not twice
        assert ad.execute("print(keep, t)").output == "[1, 2] bbc\n"


@needs_monty
def test_unconvertible_tool_result_falls_back_and_is_replayed():
    class Obj(REPLEnvironment):
        calls = 0

        def _file_info(self, path):
            Obj.calls += 1
            return SimpleNamespace(path=path)

    env = Obj(context="c")
    with _ad(env) as ad:
        r = ad.execute("info = file_info('p')\nprint(info.path)")
        assert r.output == "p\n" and _last(ad)["fallback"].startswith("adapter:unconvertible_result:file_info")
        assert Obj.calls == 1


@needs_monty
def test_user_errors_are_not_retried():
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        for code, err in [("1/0", "ZeroDivisionError: division by zero"), ("assert 1 == 2, 'no'", "AssertionError: no"),
                          ("assert 1 == 2", "AssertionError: "), ("{}['k']", "KeyError: 'k'"),
                          ("raise RuntimeError('mine')", "RuntimeError: mine")]:
            cp = REPLEnvironment(context="c").execute(code)
            r = ad.execute(code)
            assert r.error == cp.error and r.error.startswith(err), (code, r.error, cp.error)
            assert _last(ad)["route"] == "monty" and _last(ad)["fallback"] is None


@needs_monty
def test_failing_cell_keeps_partial_mutations_like_cpython():
    e1, e2 = REPLEnvironment(context="c"), REPLEnvironment(context="c")
    with _ad(e2) as ad:
        for c in ["x = 1", "x = 2\ny = 3\n1/0", "print(x, y)"]:
            a, b = e1.execute(c), ad.execute(c)
            assert (a.output, a.error) == (b.output, b.error)


# ---- state boundary: definitions cross by source ---------------------------------------------------------------
@needs_monty
def test_functions_classes_lambdas_imports_cross_cells_and_routes():
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        cells = [
            ("def f(n):\n    return n * 2", "monty"),
            ("sq = lambda v: v * v\nfrom collections import Counter as C", "monty"),
            ("class P:\n    def __init__(self, v):\n        self.v = v\n    def dbl(self):\n        return f(self.v)", "monty"),
            ("import statistics\nprint(statistics.mean([f(1), sq(2)]))", "cpython"),  # CPython sees Monty's defs
            ("def g(x):\n    return statistics.median([x, f(x)])", "cpython"),
            ("print(P(4).dbl(), sq(3), dict(C('aab')))", "monty"),  # Monty sees the defs after a CPython cell
        ]
        cp = REPLEnvironment(context="c")
        for code, route in cells:
            a, b = cp.execute(code), ad.execute(code)
            assert (a.output, a.error) == (b.output, b.error), code
            assert _last(ad)["route"] == route, (code, _last(ad))
        assert callable(env._globals["f"]) and callable(env._globals["P"]) and "f" in ad.defs
        r = ad.execute("print(g(3))")  # g references statistics: Monty NameError on a name CPython has -> fallback
        assert r.output == cp.execute("print(g(3))").output
        assert _last(ad)["fallback"] == "runtime:NameError:statistics"
        # a new worker (crash, restore) gets the definitions again
        assert ad.execute("print(f(1))").output == "2\n" and _last(ad)["route"] == "monty"
        os.kill(ad._session.worker_pid, signal.SIGKILL)
        time.sleep(0.2)
        assert ad.execute("print(f(21))").output == "42\n"


@needs_monty
def test_global_written_by_a_carried_function_syncs_back():
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        ad.execute("def bump():\n    global counter\n    counter = counter + 1 if 'x' else 0\ncounter = 0")
        ad.execute("bump()\nbump()")
        assert env._globals["counter"] == 2 and _last(ad)["route"] == "monty"


@needs_monty
def test_rebinding_a_definition_to_a_value_drops_it():
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        ad.execute("def f():\n    return 1")
        ad.execute("f = 5")
        assert "f" not in ad.defs and env._globals["f"] == 5
        assert ad.execute("print(f + 1)").output == "6\n"


@needs_monty
def test_closure_cannot_cross_and_falls_back():
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        r = ad.execute("def mk(k):\n    return lambda v: v + k\nadd2 = mk(2)\nprint(add2(1))")
        assert r.output == "3\n" and _last(ad)["fallback"].startswith("state:not_transferable:add2")
        assert ad.execute("print(add2(5))").output == "7\n"  # screened to CPython, which owns the closure


@needs_monty
def test_cpython_side_changes_reach_the_live_worker():
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        ad.execute("x = [1]")
        ad.execute("import statistics\nx.append(int(statistics.mean([2, 2])))")  # CPython mutates x in place
        assert ad.execute("print(x)").output == "[1, 2]\n" and _last(ad)["route"] == "monty"
        ad.execute("del x")  # CPython deletes it; the worker must not keep a stale copy
        assert ad.execute("print(x)").error.startswith("NameError")


@needs_monty
def test_json_checkpoint_round_trip_into_new_worker():
    cells = ["x = [i*i for i in range(5)]\nd = {'a': 1}", "artifacts['k'] = x\ncnt = 3"]
    base = REPLEnvironment(context="ctx")
    for c in cells:
        base.execute(c)
    env_a = REPLEnvironment(context="ctx")
    with _ad(env_a) as a:
        for c in cells:
            a.execute(c)
        cp = json.loads(json.dumps(a.checkpoint()))
        assert cp["user_globals"] == base.checkpoint()["user_globals"]
        env_b = REPLEnvironment(context="ctx")
        with _ad(env_b) as b:
            b.restore(cp)
            r = b.execute("cnt += 1\nprint(x, d, artifacts, cnt)")
            assert r.error is None and "[0, 1, 4, 9, 16] {'a': 1} {'k': [0, 1, 4, 9, 16]} 4" in r.output
            assert b.worker_pids[0] != a.worker_pids[0]


# ---- deadline / resources --------------------------------------------------------------------------------------
@needs_monty
def test_infinite_loop_is_terminal_like_cpython():
    env = REPLEnvironment(context="c", config=REPLConfig(timeout_seconds=1))
    ad = _ad(env)
    ad.execute("y = 1")
    t = time.perf_counter()
    r = ad.execute("while True:\n    pass")
    assert r.error == "REPLTimeout: Execution timed out after 1s" and env.timed_out
    assert time.perf_counter() - t < 4  # Monty's compute share (0.5 s), then CPython's full deadline
    assert _last(ad)["fallback"] == "monty_time_limit"
    assert ad.execute("print(1)").error == "REPLTimeout: Execution timed out after 1s"
    assert all(ad.close().values())


@needs_monty
def test_slow_host_callback_is_interrupted_like_the_repl():
    class Slow(REPLEnvironment):
        def _peek(self, *a, **k):
            time.sleep(3)
            return super()._peek(*a, **k)

    env = Slow(context="c", config=REPLConfig(timeout_seconds=1))
    with _ad(env) as ad:
        ad.execute("pass")
        t = time.perf_counter()
        r = ad.execute("peek(1)")
        assert r.error.startswith("REPLTimeout") and env.timed_out
        assert time.perf_counter() - t < 2.5


@needs_monty
def test_memory_limit_falls_back_and_cpython_completes():
    from src.repl_environment.monty_adapter import MontyLimits

    env = REPLEnvironment(context="c")
    with _ad(env, limits=MontyLimits(max_memory_bytes=32 * 1024 * 1024)) as ad:
        ad.execute("keep = [1, 2, 3]")
        old = ad.worker_pids[-1]
        r = ad.execute("n = len([0] * (8 * 10**6))\nprint(n)")
        assert r.output == "8000000\n" and _last(ad)["fallback"] == "runtime:MemoryError"
        assert ad.execute("print(keep, n)").output == "[1, 2, 3] 8000000\n" and ad.worker_pids[-1] != old


@needs_monty
def test_worker_crash_falls_back_and_resumes():
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        ad.execute("x = [1, 2, 3]")
        pid = ad._session.worker_pid
        os.kill(pid, signal.SIGKILL)
        for _ in range(100):
            if not ad.worker_alive(pid):
                break
            time.sleep(0.05)
        r = ad.execute("print(x)")
        assert r.output == "[1, 2, 3]\n" and _last(ad)["fallback"].startswith("worker_crashed")
        assert ad.execute("print(len(x))").output == "3\n" and _last(ad)["route"] == "monty"


@needs_monty
def test_output_is_bounded_before_accumulation():
    from src.repl_environment.monty_adapter import MontyLimits

    with _ad(REPLEnvironment(context="c"), limits=MontyLimits(max_output_chars=10_000)) as ad:
        r = ad.execute("for i in range(100000):\n    print('x' * 50)")
        assert r.error is None and ("OUTPUT TRUNCATED" in r.output or "→" in r.output)


# ---- pool lifecycle --------------------------------------------------------------------------------------------
@needs_monty
def test_warm_pool_is_shared_and_each_request_gets_its_own_worker():
    from src.repl_environment import monty_adapter as m

    a1, a2 = _ad(REPLEnvironment(context="a")), _ad(REPLEnvironment(context="b"))
    a1.execute("v = 1")
    a2.execute("v = 2")
    assert a1._lane_pool is a2._lane_pool and a1.worker_pids[0] != a2.worker_pids[0]
    a1.execute("v += 1")
    assert a1.route_log[-1]["worker_reused"] is True and a1.route_log[0]["worker_reused"] is False
    p1 = a1.worker_pids[0]
    assert a1.close()[p1] is True  # retired and proven dead; the pool stays up for a2
    assert a2.execute("print(v)").output == "2\n"
    pool = a2._lane_pool
    a2.close()
    m.shutdown_pools()
    assert pool.closed and pool.pool is None and not os.path.exists(pool._launcher_dir or "/nonexistent")


@needs_monty
def test_abandoned_environment_releases_its_worker_on_gc():
    env = REPLEnvironment(context="c")
    ad = _ad(env)
    ad.execute("x = 1")
    pid = ad.worker_pids[0]
    del env, ad
    gc.collect()
    from src.repl_environment.monty_adapter import MontyREPLAdapter

    for _ in range(100):
        if not MontyREPLAdapter.worker_alive(pid):
            break
        time.sleep(0.05)
    assert not MontyREPLAdapter.worker_alive(pid)


@needs_monty
def test_telemetry_rows(tmp_path):
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        ad.execute("x = 1")
        ad.execute("import statistics")
        ad.execute("print(x.bit_length())")
    rows = [json.loads(ln) for ln in open(tmp_path / "lane.jsonl")]
    assert [r["route"] for r in rows] == ["monty", "cpython", "cpython"]
    assert rows[1]["screen"] == "import:statistics" and rows[2]["fallback"] == "runtime:AttributeError"
    for r in rows:
        assert {"total_ms", "worker_reused", "refusals", "fallback", "screen", "cell"} <= set(r) or r["route"] == "cpython"


# ---- fail open -------------------------------------------------------------------------------------------------
@needs_monty
def test_fail_open_when_package_missing(monkeypatch):
    import builtins

    from src.repl_environment.environment import install_monty_lane

    real = builtins.__import__

    def no_monty(name, *a, **k):
        if name == "pydantic_monty":
            raise ImportError("simulated")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_monty)
    env = install_monty_lane(REPLEnvironment(context="c"))
    assert getattr(env, "_monty_lane", None) is None and env.execute("print(1)").output == "1\n"


@needs_monty
def test_fail_open_when_containment_cannot_apply(monkeypatch):
    from src.repl_environment import exec_sandbox
    from src.repl_environment import monty_adapter as m

    m.shutdown_pools()
    monkeypatch.setattr(exec_sandbox, "capabilities", lambda: {"abi": 0, "landlock": False, "netdeny": False, "ts": 0})
    from src.repl_environment.environment import install_monty_lane

    env = install_monty_lane(REPLEnvironment(context="c"))
    assert getattr(env, "_monty_lane", None) is None and env.execute("print(2)").output == "2\n"


@needs_monty
def test_fail_open_when_worker_will_not_start(monkeypatch, tmp_path):
    from src.repl_environment import monty_adapter as m

    m.shutdown_pools()
    bad = tmp_path / "monty"
    bad.write_text("#!/bin/sh\nexit 3\n")
    bad.chmod(0o700)
    monkeypatch.setenv("MONTY_BIN", str(bad))
    env = REPLEnvironment(context="c")
    ad = m.maybe_monty_adapter(env)
    assert env.execute("print(3)").output == "3\n"  # with or without an adapter, the cell succeeds
    if ad is not None:
        assert ad.route_log[-1]["route"] == "cpython"
    m.shutdown_pools()


@needs_monty
def test_fail_closed_when_worker_caps_unverifiable(monkeypatch):
    from src.repl_environment import monty_adapter as m

    def refuse(self, pid):
        raise m.MontyUnavailable("caps not verifiable (simulated)")

    monkeypatch.setattr(m._LanePool, "verify_worker", refuse)
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        assert ad.execute("print(4)").output == "4\n"
        assert ad.route_log[0]["fallback"].startswith("worker_unavailable")
        assert ad.execute("print(5)").output == "5\n" and ad.route_log[1]["screen"].startswith("lane_unavailable")


@needs_monty
def test_cgroup_required_but_absent_fails_open(monkeypatch):
    from src.repl_environment import monty_adapter as m

    m.shutdown_pools()
    monkeypatch.setenv("ORCHESTRATOR_MONTY_CGROUP", "require")
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP_PARENT", "/nonexistent-cgroup")
    env = REPLEnvironment(context="c")
    assert m.maybe_monty_adapter(env) is None and env.execute("print(6)").output == "6\n"
    m.shutdown_pools()


# ---- worker containment and caps -------------------------------------------------------------------------------
def _proc(pid: int, path: str, key: str) -> str:
    with open(f"/proc/{pid}/{path}") as fh:
        return next(ln for ln in fh if ln.startswith(key))


@needs_monty
def test_worker_runs_confined_and_capped():
    from src.repl_environment import exec_sandbox

    if not exec_sandbox.capabilities()["landlock"]:
        pytest.skip("Landlock unavailable (the lane fails closed: see the fail-open tests)")
    env = REPLEnvironment(context="c")
    with _ad(env) as ad:
        ad.execute("x = 1")
        pid = ad.worker_pids[-1]
        assert ad.worker_contained and _proc(pid, "status", "Seccomp:").split()[1] == "2"
        proof = ad.worker_proof
        assert proof["mode"] in ("cgroup", "rlimit")
        cap = ad.limits.worker_as_mb * 1024 * 1024
        assert int(_proc(pid, "limits", "Max address space").split()[3]) <= cap
        if proof["mode"] == "rlimit":
            assert _proc(pid, "limits", "Max processes").split()[2] != "unlimited"
        launcher = ad._lane_pool._launcher_dir
        assert (os.stat(launcher).st_mode & 0o077) == 0


@needs_monty
def test_worker_environment_has_no_inherited_secrets(monkeypatch):
    from src.repl_environment import monty_adapter as m

    m.shutdown_pools()  # the canary must be in the environment the pool spawns workers from
    monkeypatch.setenv("CANARY_API_KEY", "sekrit-canary")
    with _ad(REPLEnvironment(context="c")) as ad:
        ad.execute("x = 1")
        with open(f"/proc/{ad.worker_pids[-1]}/environ", "rb") as fh:
            assert b"CANARY" not in fh.read()
    m.shutdown_pools()
