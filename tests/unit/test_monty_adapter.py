"""D-RI-MONTY-SLICE: the restricted Monty adapter (opt-in, default OFF).

Needs ``pydantic-monty`` (optional extra ``monty``); the whole module skips without it. Pure CPU, no servers.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import time

import pytest

pytest.importorskip("pydantic_monty")

from src.features import Features, _FEATURE_REGISTRY, reset_features, set_features  # noqa: E402
from src.repl_environment import REPLConfig, REPLEnvironment  # noqa: E402
from src.repl_environment.monty_adapter import (  # noqa: E402
    MontyLimits,
    MontyREPLAdapter,
    MontyUnavailable,
    _binding_names,
    maybe_monty_adapter,
    monty_enabled,
)

_HARNESS = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "monty_slice_compare.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("monty_slice_compare", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["monty_slice_compare"] = mod  # dataclasses resolves annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _flag_on():
    set_features(Features(monty_repl=True))
    yield
    reset_features()


def test_flag_is_default_off_in_test_and_prod():
    spec = next(s for s in _FEATURE_REGISTRY if s.name == "monty_repl")
    assert spec.default_test is False and spec.default_prod is False
    assert Features().monty_repl is False


def test_adapter_refuses_without_flag():
    set_features(Features())  # monty_repl False
    assert monty_enabled() is False
    env = REPLEnvironment(context="c")
    with pytest.raises(MontyUnavailable):
        MontyREPLAdapter(env)
    assert maybe_monty_adapter(env) is None


def test_adapter_refuses_unsupported_parent():
    env = REPLEnvironment(context="c", structured_mode=True)
    with pytest.raises(MontyUnavailable, match="structured"):
        MontyREPLAdapter(env)


def test_binding_names():
    import ast

    code = "a = 1\nfor i in range(3):\n    b = i\nwith x as c:\n    pass\nimport os.path as p\nfrom m import n\n" \
           "def f():\n    inner = 1\nclass K:\n    pass\n[j for j in range(2)]\nz += 1\ndel a\nif (w := 3): pass\n"
    assert _binding_names(ast.parse(code)) == ["a", "i", "b", "c", "p", "n", "f", "K", "z", "w"]


# ---- the twelve cases -------------------------------------------------------------------------------------------
_H = _load_harness()


@pytest.mark.parametrize("case", _H.CASES, ids=[c.id for c in _H.CASES])
def test_twelve_cases_match_cpython(case):
    cp_env = _H._mk_parent(case)
    cp = [cp_env.execute(c) for c in case.cells]
    assert case.oracle(cp, cp_env)  # the original test's own assertion holds on the baseline
    env = _H._mk_parent(case)
    with MontyREPLAdapter(env) as ad:
        mo = [ad.execute(c) for c in case.cells]
        assert case.oracle(mo, env)  # and on the Monty lane
        for a, b in zip(cp, mo):
            assert (a.output, a.is_final, a.final_answer) == (b.output, b.is_final, b.final_answer)
            assert (a.error or "").split(":")[0] == (b.error or "").split(":")[0]
            if case.id != "C07":  # CPython and Monty word SyntaxError differently
                assert a.error == b.error
        assert _H._state(cp_env) == _H._state(env)


def test_security_refusals_are_the_repl_gate():
    for code in ["import os", "from os import system", "eval('1+1')", "''.__class__", "open('/etc/hostname')"]:
        e1 = REPLEnvironment(context="c")
        e2 = REPLEnvironment(context="c")
        with MontyREPLAdapter(e2) as ad:
            assert ad.execute(code).error == e1.execute(code).error
            assert e1.execute(code).error.startswith("Dangerous operation not allowed")


def test_native_monty_without_gate_differs():
    """Documents (not endorses) what Monty alone does: eval is sandboxed, not refused."""
    env = REPLEnvironment(context="c")
    with MontyREPLAdapter(env, ast_gate=False) as ad:
        r = ad.execute("print(eval('1+1'))")
        assert r.output.strip() == "2" and r.error is None
        assert "PermissionError" in (ad.execute("open('/etc/hostname')").error or "")


# ---- parent boundary: JSON checkpoint export / import ----------------------------------------------------------
def test_json_checkpoint_round_trip_into_new_worker():
    cells = ["x = [i*i for i in range(5)]\nd = {'a': 1}", "artifacts['k'] = x\ncnt = 3"]
    base = REPLEnvironment(context="ctx")
    for c in cells:
        base.execute(c)
    env_a = REPLEnvironment(context="ctx")
    with MontyREPLAdapter(env_a) as a:
        for c in cells:
            a.execute(c)
        wire = json.dumps(a.checkpoint())  # plain JSON; no Monty dump bytes anywhere
        cp = json.loads(wire)
        assert cp["user_globals"] == base.checkpoint()["user_globals"]
        assert cp["artifacts"] == base.checkpoint()["artifacts"]
        env_b = REPLEnvironment(context="ctx")
        with MontyREPLAdapter(env_b) as b:
            rec = b.restore(cp)
            assert sorted(rec["restored"]) == ["cnt", "d", "x"] and not rec["unavailable"]
            r = b.execute("cnt += 1\nprint(x, d, artifacts, cnt)")
            assert r.error is None and "[0, 1, 4, 9, 16] {'a': 1} {'k': [0, 1, 4, 9, 16]} 4" in r.output
            assert env_b._globals["cnt"] == 4
            assert b.worker_pids[0] != a.worker_pids[0]


def test_non_json_state_is_reported_not_silently_carried():
    env = REPLEnvironment(context="c")
    with MontyREPLAdapter(env) as a:
        a.execute("def f():\n    return 1\nok = 1")
        assert "f" in a.unavailable and "ok" not in a.unavailable
        assert "f" not in env._globals and env._globals["ok"] == 1
        # a pickled/non-JSON parent global is refused at import, with a reason
        env2 = REPLEnvironment(context="c")
        env2._globals["blob"] = {1, 2}
        with MontyREPLAdapter(env2) as b:
            assert b.execute("print('hi')").output == "hi\n"
            assert "blob" in b.unavailable


# ---- callbacks ---------------------------------------------------------------------------------------------------
def test_denied_and_bounded_callbacks():
    env = REPLEnvironment(context="hello world")
    with MontyREPLAdapter(env) as a:
        assert "PermissionError" in a.execute("peek(5, file_path='/etc/hostname')").error
        assert "PermissionError" in a.execute("grep('a', file_path='/etc/hostname')").error
        assert "not available in the Monty lane" in a.execute("run_shell('echo hi')").error
        assert "not available in the Monty lane" in a.execute("web_fetch('http://x')").error
        assert a.execute("print(context_len())").output == "11\n"
        assert a.execute("print(peek(5))").output == "hello\n"
        assert len(a.denied_log) == 4
        assert a.execute("peek(10**12)").error is None  # n is clamped, not honoured


def test_tool_call_budget_is_enforced(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", "5")
    env = REPLEnvironment(context="ctx")
    with MontyREPLAdapter(env) as a:
        r = a.execute("for i in range(20):\n    peek(1)")
        assert "Tool-call budget exceeded" in r.error


def test_final_keyword_validation_matches_repl():
    env = REPLEnvironment(context="c")
    with MontyREPLAdapter(env) as a:
        r = a.execute('FINAL(result="ok", extra="bad")')
        assert not r.is_final and "unsupported keyword" in r.error
        r = a.execute('FINAL(result="computed answer")')
        assert r.is_final and r.final_answer == "computed answer"


def test_swallowing_final_does_not_cancel_it():
    env = REPLEnvironment(context="c")
    with MontyREPLAdapter(env) as a:
        r = a.execute("try:\n    FINAL('a')\nexcept BaseException:\n    print('swallowed')\n")
        assert r.is_final and r.final_answer == "a"


# ---- terminal resource errors / disposal -------------------------------------------------------------------------
def test_timeout_is_terminal_and_kills_only_the_captured_worker():
    env = REPLEnvironment(context="c", config=REPLConfig(timeout_seconds=1))
    a = MontyREPLAdapter(env)
    a.execute("y = 1")
    pid = a.worker_pids[-1]
    t = time.perf_counter()
    r = a.execute("while True:\n    pass")
    assert r.error == "REPLTimeout: Execution timed out after 1s" and env.timed_out
    assert time.perf_counter() - t < 4
    assert a.execute("print(1)").error == "REPLTimeout: Execution timed out after 1s"  # terminal, never re-run
    proof = a.close()
    assert proof[pid] is True and all(proof.values())


def test_slow_host_callback_is_interrupted_like_the_repl():
    class Slow(REPLEnvironment):
        def _peek(self, *a, **k):
            time.sleep(3)
            return super()._peek(*a, **k)

    env = Slow(context="c", config=REPLConfig(timeout_seconds=1))
    with MontyREPLAdapter(env) as a:
        a.execute("pass")
        t = time.perf_counter()
        r = a.execute("peek(1)")
        assert r.error.startswith("REPLTimeout") and env.timed_out
        assert time.perf_counter() - t < 2.5  # interrupted at ~1 s, not after the 3 s callback


def test_memory_trip_discards_worker_and_resumes_from_checkpoint():
    env = REPLEnvironment(context="c")
    with MontyREPLAdapter(env, limits=MontyLimits(max_memory_bytes=64 * 1024 * 1024)) as a:
        a.execute("keep = [1, 2, 3]\nartifacts['k'] = 'v'")
        old = a.worker_pids[-1]
        r = a.execute("n = 2 * 10**9\nbig = [0] * n")
        assert r.error.startswith("MemoryError") and not env.timed_out
        r2 = a.execute("print(keep, artifacts)")
        assert r2.output == "[1, 2, 3] {'k': 'v'}\n" and a.sessions_started == 2
        assert a.worker_pids[-1] != old
        for _ in range(50):
            if not MontyREPLAdapter.worker_alive(old):
                break
            time.sleep(0.1)
        assert not MontyREPLAdapter.worker_alive(old)


def test_worker_crash_resumes_from_checkpoint():
    import signal

    env = REPLEnvironment(context="c")
    with MontyREPLAdapter(env) as a:
        a.execute("x = [1, 2, 3]")
        pid = a._session.worker_pid
        os.kill(pid, signal.SIGKILL)  # the pid this adapter captured
        for _ in range(100):
            if not MontyREPLAdapter.worker_alive(pid):
                break
            time.sleep(0.05)
        assert "worker crashed" in a.execute("print(x)").error
        assert a.execute("print(x)").output == "[1, 2, 3]\n"


def test_output_is_bounded_before_accumulation():
    env = REPLEnvironment(context="c")
    with MontyREPLAdapter(env, limits=MontyLimits(max_output_chars=10_000)) as a:
        r = a.execute("for i in range(100000):\n    print('x' * 50)")
        assert r.error is None
        assert "OUTPUT TRUNCATED" in r.output or "→" in r.output  # spill summary or in-band notice


# ---- worker containment (defense in depth) ------------------------------------------------------------------------
def _proc_status(pid: int, key: str) -> str:
    with open(f"/proc/{pid}/status") as fh:
        return next(ln.split(":", 1)[1].strip() for ln in fh if ln.startswith(key + ":"))


def test_worker_runs_under_the_sec_exec_shim_by_default():
    from src.repl_environment import exec_sandbox

    if not exec_sandbox.capabilities()["landlock"]:
        pytest.skip("Landlock unavailable on this host (the adapter fails closed, see next test)")
    env = REPLEnvironment(context="c")
    a = MontyREPLAdapter(env)
    assert a.execute("x = 1").error is None and a.worker_contained
    pid = a.worker_pids[-1]
    assert _proc_status(pid, "Seccomp") == "2" and _proc_status(pid, "NoNewPrivs") == "1"
    launcher = a._launcher_dir
    assert launcher and os.path.isdir(launcher) and (os.stat(launcher).st_mode & 0o077) == 0
    a.close()
    assert not os.path.exists(launcher)  # launcher dir removed on close


def test_uncontained_worker_is_opt_out_only():
    env = REPLEnvironment(context="c")
    with MontyREPLAdapter(env, contain_worker=False) as a:
        a.execute("x = 1")
        assert a.worker_contained is False and _proc_status(a.worker_pids[-1], "Seccomp") == "0"


def test_fails_closed_when_containment_is_unavailable(monkeypatch):
    from src.repl_environment import exec_sandbox

    monkeypatch.setattr(exec_sandbox, "capabilities", lambda: {"abi": 0, "landlock": False, "netdeny": False, "ts": 0})
    env = REPLEnvironment(context="c")
    a = MontyREPLAdapter(env)
    with pytest.raises(MontyUnavailable, match="refusing to run unconfined"):
        a.execute("x = 1")
    a.close()


def test_worker_environment_has_no_inherited_secrets(monkeypatch):
    monkeypatch.setenv("CANARY_API_KEY", "sekrit-canary")
    env = REPLEnvironment(context="c")
    with MontyREPLAdapter(env) as a:
        a.execute("x = 1")
        with open(f"/proc/{a.worker_pids[-1]}/environ", "rb") as fh:
            assert b"CANARY" not in fh.read()
