#!/usr/bin/env python3
"""D-RI-MONTY-SLICE comparison harness: twelve existing REPL cases through warm CPython, contained CPython and Monty.

CPU-only, no inference servers. Run with an interpreter that has ``pydantic-monty`` (a PRIVATE venv; never the
shared one):

    PYTHONPATH=.:/mnt/raid0/llm/epyc-orchestrator/.venv/lib/python3.11/site-packages:/usr/lib/python3/dist-packages \
        /mnt/raid0/llm/tmp/monty-slice-venv/bin/python scripts/benchmark/monty_slice_compare.py \
        --out docs/experiments/monty-slice-results.json

Arms
----
* ``cpython_warm``  the production REPL (``REPLEnvironment.execute``), one env per case, reused for warm timing.
* ``cpython_contained``  the EXISTING SEC-EXEC contained child (``run_python_code``: Landlock + seccomp, env
  scrubbed), one cold child per cell. D-RI-CONTAIN-T2 (a warm, REPL-aware contained lane) is not built, so this arm
  has no REPL surface (no peek/grep/context/artifacts/FINAL, no state across cells): it is only "supported" for
  stateless self-contained cells, and says so.
* ``monty``  ``MontyREPLAdapter`` (flag ``ORCHESTRATOR_MONTY_REPL=1``), AST gate on.
* ``monty_native``  the same adapter with the REPL AST gate OFF, used only to measure what Monty itself refuses.

The twelve cases are copied verbatim (code, context, config) from ``tests/unit/test_repl_environment.py``; each
carries the original test's assertion as an ``oracle``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

os.environ.setdefault("ORCHESTRATOR_MONTY_REPL", "1")

from src.repl_environment import REPLConfig, REPLEnvironment  # noqa: E402
from src.repl_environment.monty_adapter import (  # noqa: E402
    ENV_FLAG,
    MontyLimits,
    MontyREPLAdapter,
    monty_enabled,
)

WARM_REPS = 50


@dataclass
class Case:
    id: str
    test: str  # the existing test this case is taken from
    context: str
    cells: list[str]
    oracle: Callable[[list[Any], Any], bool]  # (results, parent_env) -> the original test's assertion
    config: dict[str, Any] = field(default_factory=dict)
    contained_ok: bool = False  # stateless + self-contained: runnable in a bare child
    needs: str = ""  # REPL surface the case uses


def _o(r: Any) -> str:
    return r.output or ""


CASES: list[Case] = [
    Case("C01", "TestPeekFunction::test_peek_default", "A" * 1000, ["print(len(peek()))"],
         lambda r, e: "500" in _o(r[0]), needs="peek"),
    Case("C02", "TestGrepFunction::test_grep_regex_pattern", "test123\ntest456\nhello",
         ["print(grep(r'test\\d+'))"],
         lambda r, e: "test123" in _o(r[0]) and "test456" in _o(r[0]) and "hello" not in _o(r[0]), needs="grep"),
    Case("C03", "TestGrepFunction::test_grep_truncation", "\n".join(f"match line {i}" for i in range(200)),
         ["print(grep('match'))"], lambda r, e: "truncated" in _o(r[0]), config={"max_grep_results": 50},
         needs="grep (bounded result)"),
    Case("C04", "TestAllowedBuiltins::test_dict_comprehension_works", "test",
         ["print({i: i*2 for i in range(3)})"], lambda r, e: "0: 0" in _o(r[0]), contained_ok=True),
    Case("C05", "TestAllowedBuiltins::test_try_except_works", "test",
         ["try:\n    x = 1/0\nexcept ZeroDivisionError:\n    print('caught')\n"],
         lambda r, e: "caught" in _o(r[0]) and r[0].error is None, contained_ok=True),
    Case("C06", "TestREPLBasicExecution::test_runtime_error_captured", "test", ["x = 1 / 0"],
         lambda r, e: r[0].error is not None and "ZeroDivisionError" in r[0].error, contained_ok=True),
    Case("C07", "TestREPLBasicExecution::test_syntax_error_captured", "test", ["print("],
         lambda r, e: r[0].error is not None and "SyntaxError" in r[0].error, contained_ok=True),
    Case("C08", "TestREPLBasicExecution::test_artifacts_can_be_modified", "test",
         ["artifacts['result'] = 42", "print(artifacts['result'])"],
         lambda r, e: "42" in _o(r[1]) and e.artifacts["result"] == 42, needs="artifacts (state to parent)"),
    Case("C09", "TestFinalFunction::test_output_before_final_captured", "test",
         ["\nprint(\"before final\")\nFINAL(\"done\")\n"],
         lambda r, e: r[0].is_final and "before final" in _o(r[0]) and r[0].final_answer == "done", needs="FINAL"),
    Case("C10", "TestFinalFunction::test_final_var_returns_artifact", "test",
         ["artifacts['result'] = 'computed value'", "FINAL_VAR('result')"],
         lambda r, e: r[1].is_final and r[1].final_answer == "computed value", needs="FINAL_VAR + artifacts"),
    Case("C11", "TestREPLState::test_state_persists_across_executions", "test", ["x = 42", "print(x)"],
         lambda r, e: "42" in _o(r[1]), needs="state continuity across cells"),
    Case("C12", "TestSecuritySandbox::test_import_os_blocked", "test", ["import os"],
         lambda r, e: r[0].error is not None and ("not allowed" in r[0].error.lower() or "Dangerous" in r[0].error),
         contained_ok=True, needs="AST security refusal"),
]

# -------------------------------------------------------------------------------------------------------------
# arm runners


def _mk_parent(case: Case, cls: type = REPLEnvironment, **extra: Any) -> REPLEnvironment:
    cfg = REPLConfig(**{**case.config, **extra}) if (case.config or extra) else REPLConfig()
    return cls(context=case.context, config=cfg)


def _rec(r: Any) -> dict[str, Any]:
    return {"output": r.output, "is_final": r.is_final, "final_answer": r.final_answer, "error": r.error,
            "ms": round(r.elapsed_seconds * 1000, 4)}


def _state(env: REPLEnvironment) -> dict[str, Any]:
    cp = env.checkpoint()
    return {"artifacts": cp["artifacts"], "user_globals": cp["user_globals"]}


def run_cpython_warm(case: Case, reps: int = WARM_REPS) -> dict[str, Any]:
    t0 = time.perf_counter()
    env = _mk_parent(case)
    t_ctor = time.perf_counter() - t0
    res = [env.execute(c) for c in case.cells]
    cold = time.perf_counter() - t0
    out = {"cells": [_rec(r) for r in res], "oracle": bool(case.oracle(res, env)), "state": _state(env),
           "cold_ms": round(cold * 1000, 3), "ctor_ms": round(t_ctor * 1000, 3)}
    passes = []
    for _ in range(reps):
        t = time.perf_counter()
        for c in case.cells:
            env.execute(c)
        passes.append((time.perf_counter() - t) * 1000)
    out["warm_ms_median"] = round(statistics.median(passes), 4)
    out["warm_ms_p95"] = round(sorted(passes)[int(0.95 * len(passes)) - 1], 4)
    return out


def run_monty(case: Case, reps: int = WARM_REPS, *, ast_gate: bool = True, pool: Any = None,
              contain_worker: bool | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    env = _mk_parent(case)
    ad = MontyREPLAdapter(env, ast_gate=ast_gate, pool=pool, contain_worker=contain_worker)
    try:
        res = [ad.execute(c) for c in case.cells]
        cold = time.perf_counter() - t0
        out = {"cells": [_rec(r) for r in res], "oracle": bool(case.oracle(res, env)), "state": _state(env),
               "cold_ms": round(cold * 1000, 3), "sessions_started": ad.sessions_started,
               "unavailable": dict(ad.unavailable), "denied": len(ad.denied_log),
               "callbacks": [n for n, _ in ad.callback_log], "worker_contained": ad.worker_contained}
        pid = ad.worker_pids[-1] if ad.worker_pids else None
        if pid:
            try:
                with open(f"/proc/{pid}/status") as fh:
                    out["worker_rss_kb"] = next(int(ln.split()[1]) for ln in fh if ln.startswith("VmRSS"))
            except OSError:
                pass
        passes = []
        for _ in range(reps):
            t = time.perf_counter()
            for c in case.cells:
                ad.execute(c)
            passes.append((time.perf_counter() - t) * 1000)
        out["warm_ms_median"] = round(statistics.median(passes), 4)
        out["warm_ms_p95"] = round(sorted(passes)[int(0.95 * len(passes)) - 1], 4)
    finally:
        disposal = ad.close()
    out["disposed"] = all(disposal.values()) if disposal else None
    return out


def run_contained(case: Case) -> dict[str, Any]:
    if not case.contained_ok:
        return {"supported": False, "why": f"needs REPL surface ({case.needs}); the existing contained child has none "
                                           "(D-RI-CONTAIN-T2 not built)"}
    env = _mk_parent(case)
    cells = []
    t0 = time.perf_counter()
    for c in case.cells:
        t = time.perf_counter()
        txt = env._run_python_code(c)
        cells.append({"output": txt, "ms": round((time.perf_counter() - t) * 1000, 3)})
    lat = [c["ms"] for c in cells]
    return {"supported": True, "cells": cells, "cold_ms": round((time.perf_counter() - t0) * 1000, 3),
            "per_cell_ms_median": round(statistics.median(lat), 3)}


def compare(cp: dict[str, Any], mo: dict[str, Any]) -> dict[str, Any]:
    cmp: dict[str, Any] = {"output": [], "error_exact": [], "error_class": [], "final": []}
    for a, b in zip(cp["cells"], mo["cells"]):
        cmp["output"].append(a["output"] == b["output"])
        cmp["error_exact"].append(a["error"] == b["error"])
        ca = (a["error"] or "").split(":")[0]
        cb = (b["error"] or "").split(":")[0]
        cmp["error_class"].append(ca == cb)
        cmp["final"].append((a["is_final"], a["final_answer"]) == (b["is_final"], b["final_answer"]))
    cmp["state_equal"] = cp["state"] == mo["state"]
    cmp["all_equal"] = all(cmp["output"]) and all(cmp["error_exact"]) and all(cmp["final"]) and cmp["state_equal"]
    return cmp


# -------------------------------------------------------------------------------------------------------------
# extra scenarios (denied / delayed callbacks, terminal resource errors, worker disposal, checkpoint, coverage)


class SlowPeekEnv(REPLEnvironment):
    DELAY = 0.0

    def _peek(self, *a: Any, **k: Any):  # noqa: D401
        time.sleep(self.DELAY)
        return super()._peek(*a, **k)


def _brief(r: Any) -> dict[str, Any]:
    return {"output": (r.output or "")[:160], "is_final": r.is_final,
            "error": (r.error or "")[:200] or None, "ms": round(r.elapsed_seconds * 1000, 2)}


def s1_checkpoint_roundtrip() -> dict[str, Any]:
    """JSON checkpoint export/import through the parent boundary (no Monty dump bytes)."""
    cells = ["x = [i*i for i in range(5)]\nd = {'a': 1, 'b': [1, 2]}\ns = 'hi'", "artifacts['k'] = x\ncnt = 3", "cnt += 1"]
    env_c = REPLEnvironment(context="ctx")
    for c in cells:
        env_c.execute(c)
    cp_c = env_c.checkpoint()
    env_a = REPLEnvironment(context="ctx")
    a = MontyREPLAdapter(env_a)
    for c in cells:
        a.execute(c)
    cp = a.checkpoint()
    wire = json.dumps(cp)  # plain JSON, no opaque bytes
    first_pid = a.worker_pids[0]
    env_b = REPLEnvironment(context="ctx")
    b = MontyREPLAdapter(env_b)
    rec = b.restore(json.loads(wire))
    r = b.execute("print(x, d, s, artifacts, cnt)\ncnt += 10\nartifacts['k2'] = cnt")
    r2 = b.execute("print(cnt, artifacts['k2'])")
    out = {
        "user_globals_equal_cpython": cp["user_globals"] == cp_c["user_globals"],
        "artifacts_equal_cpython": cp["artifacts"] == cp_c["artifacts"],
        "json_bytes": len(wire), "restore_reconciliation": rec,
        "continue_cell": _brief(r), "second_cell": _brief(r2),
        "state_after": {"cnt": env_b._globals.get("cnt"), "artifacts": env_b.artifacts},
        "restored_into_new_worker": bool(b.worker_pids) and b.worker_pids[0] != first_pid,
        "sessions_b": b.sessions_started,
    }
    a.close(); b.close()
    return out


def s2_denied_callbacks() -> dict[str, Any]:
    probes = {
        "peek_file_path": "print(peek(20, file_path='/etc/hostname'))",
        "grep_file_path": "print(grep('a', file_path='/etc/hostname'))",
        "run_shell": "print(run_shell('echo hi'))",
        "web_fetch": "print(web_fetch('http://127.0.0.1:1/'))",
        "list_dir": "print(list_dir('/'))",
        "unknown_name": "print(totally_not_a_tool(1))",
    }
    out: dict[str, Any] = {}
    for name, code in probes.items():
        env = REPLEnvironment(context="ctx")
        r_cp = env.execute(code)
        env2 = REPLEnvironment(context="ctx")
        with MontyREPLAdapter(env2) as a:
            r_mo = a.execute(code)
            den = [d["reason"] for d in a.denied_log]
        out[name] = {"cpython_warm": _brief(r_cp), "monty": _brief(r_mo), "monty_denied_log": den}
    # budget
    env = REPLEnvironment(context="ctx")
    os.environ["ORCHESTRATOR_REPL_MAX_TOOL_CALLS"] = "5"
    try:
        r_cp = env.execute("for i in range(20):\n    peek(1)")
        env2 = REPLEnvironment(context="ctx")
        with MontyREPLAdapter(env2) as a:
            r_mo = a.execute("for i in range(20):\n    peek(1)")
            out["tool_call_budget_5"] = {"cpython_warm": _brief(r_cp), "monty": _brief(r_mo)}
    finally:
        os.environ.pop("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", None)
    return out


def s3_delayed_callbacks() -> dict[str, Any]:
    out: dict[str, Any] = {}
    # (a) delayed but inside the deadline
    SlowPeekEnv.DELAY = 0.2
    code = "print(len(peek(10)))"
    e1 = SlowPeekEnv(context="A" * 100, config=REPLConfig(timeout_seconds=5))
    r1 = e1.execute(code)
    e2 = SlowPeekEnv(context="A" * 100, config=REPLConfig(timeout_seconds=5))
    with MontyREPLAdapter(e2) as a:
        a.execute("pass")  # warm the session so the delay is the only difference
        r2 = a.execute(code)
    out["delay_0.2s_inside_5s"] = {"cpython_warm": _brief(r1), "monty": _brief(r2)}
    # (b) delay outruns the deadline: REPL interrupts (SIGALRM), the adapter cannot interrupt a host callback
    SlowPeekEnv.DELAY = 1.6
    e1 = SlowPeekEnv(context="A" * 100, config=REPLConfig(timeout_seconds=1))
    r1 = e1.execute(code)
    follow1 = e1.execute("print('after')")
    e2 = SlowPeekEnv(context="A" * 100, config=REPLConfig(timeout_seconds=1))
    with MontyREPLAdapter(e2) as a:
        a.execute("pass")
        r2 = a.execute(code)
        follow2 = a.execute("print('after')")
        pid = a.worker_pids[-1]
        dead = None
        t_end = time.monotonic() + 5
        while time.monotonic() < t_end:
            dead = not MontyREPLAdapter.worker_alive(pid)
            if dead:
                break
            time.sleep(0.05)
    out["delay_1.6s_outruns_1s"] = {
        "cpython_warm": {**_brief(r1), "terminal": e1.timed_out, "followup": _brief(follow1)},
        "monty": {**_brief(r2), "terminal": e2.timed_out, "followup": _brief(follow2), "worker_dead": dead},
    }
    SlowPeekEnv.DELAY = 0.0
    # (c) a slow host-side regex inside the bridged grep (catastrophic backtracking): the REPL's SIGALRM interrupts re
    ctx = "a" * 27 + "b"
    code = "print(len(grep('(a+)+$')))"
    e1 = REPLEnvironment(context=ctx, config=REPLConfig(timeout_seconds=1))
    t = time.perf_counter(); r1 = e1.execute(code); w1 = time.perf_counter() - t
    e2 = REPLEnvironment(context=ctx, config=REPLConfig(timeout_seconds=1))
    with MontyREPLAdapter(e2) as a:
        a.execute("pass")
        t = time.perf_counter(); r2 = a.execute(code); w2 = time.perf_counter() - t
    out["redos_in_bridged_grep_1s"] = {
        "cpython_warm": {**_brief(r1), "terminal": e1.timed_out, "wall_ms": round(w1 * 1000)},
        "monty": {**_brief(r2), "terminal": e2.timed_out, "wall_ms": round(w2 * 1000)},
    }
    return out


def s4_terminal_resource_errors() -> dict[str, Any]:
    out: dict[str, Any] = {}
    # (a) CPU-bound loop, 1 s deadline
    e1 = REPLEnvironment(context="c", config=REPLConfig(timeout_seconds=1))
    r1 = e1.execute("while True:\n    pass")
    f1 = e1.execute("print('after')")
    e2 = REPLEnvironment(context="c", config=REPLConfig(timeout_seconds=1))
    with MontyREPLAdapter(e2) as a:
        a.execute("y = 1")
        pid = a.worker_pids[-1]
        r2 = a.execute("while True:\n    pass")
        f2 = a.execute("print('after')")
        t_end = time.monotonic() + 5
        dead = False
        while time.monotonic() < t_end and not dead:
            dead = not MontyREPLAdapter.worker_alive(pid)
            time.sleep(0.05)
    out["cpu_loop_1s"] = {"cpython_warm": {**_brief(r1), "terminal": e1.timed_out, "followup": _brief(f1)},
                          "monty": {**_brief(r2), "terminal": e2.timed_out, "followup": _brief(f2), "worker_dead": dead}}
    # (b) dynamic memory bomb: NEVER run in-process on CPython (the AST guard only sees literal sizes)
    e3 = REPLEnvironment(context="c")
    # 3e9 x 8 B = 24 GB > the child's 16 GiB RLIMIT_AS, so it fails at mmap instead of touching memory
    # (a first 2e9 = 16 GB attempt, just under the cap, actually allocated; do not go back to that size)
    contained = e3._run_python_code("n = 3 * 10**9\nx = [0] * n\nprint('allocated')")
    e4 = REPLEnvironment(context="c")
    with MontyREPLAdapter(e4, limits=MontyLimits(max_memory_bytes=64 * 1024 * 1024)) as a:
        a.execute("keep = [1, 2, 3]\nartifacts['k'] = 'v'")
        pid1 = a.worker_pids[-1]
        r = a.execute("n = 2 * 10**9\nbig = [0] * n")
        r_after = a.execute("print(keep, artifacts)")
        pid2 = a.worker_pids[-1]
        t_end = time.monotonic() + 5
        dead = False
        while time.monotonic() < t_end and not dead:
            dead = not MontyREPLAdapter.worker_alive(pid1)
            time.sleep(0.05)
        out["memory_bomb_64MiB"] = {
            "cpython_warm": "NOT RUN by design: in-process, dynamic sizes are unguarded (security.py documents the limit)",
            "cpython_contained": {"output": contained[-160:], "note": "child under RLIMIT_AS (16 GiB)"},
            "monty": {**_brief(r), "followup_resumed_from_checkpoint": _brief(r_after),
                      "new_worker": pid2 != pid1, "old_worker_dead": dead, "sessions": a.sessions_started,
                      "parent_terminal": e4.timed_out},
        }
    # (c) recursion
    code = "def f(n):\n    return f(n + 1)\nf(0)"
    e5 = REPLEnvironment(context="c")
    r5 = e5.execute(code)
    e6 = REPLEnvironment(context="c")
    with MontyREPLAdapter(e6) as a:
        r6 = a.execute(code)
        f6 = a.execute("print('still alive')")
    out["recursion"] = {"cpython_warm": _brief(r5), "monty": {**_brief(r6), "followup": _brief(f6)}}
    # (d) output flood: bounded before accumulation?
    code = "for i in range(200000):\n    print('x' * 50)"
    e7 = REPLEnvironment(context="c")
    t = time.perf_counter()
    r7 = e7.execute(code)
    t7 = time.perf_counter() - t
    e8 = REPLEnvironment(context="c")
    with MontyREPLAdapter(e8) as a:
        a.execute("pass")
        t = time.perf_counter()
        r8 = a.execute(code)
        t8 = time.perf_counter() - t
    out["output_flood_10MB"] = {"cpython_warm": {**_brief(r7), "wall_ms": round(t7 * 1000, 1), "output_chars": len(r7.output)},
                                "monty": {**_brief(r8), "wall_ms": round(t8 * 1000, 1), "output_chars": len(r8.output)}}
    return out


def s5_disposal_and_crash() -> dict[str, Any]:
    import signal

    out: dict[str, Any] = {}
    env = REPLEnvironment(context="c")
    a = MontyREPLAdapter(env)
    a.execute("x = [1, 2, 3]\nartifacts['k'] = 'v'")
    pid = a._session.worker_pid
    # crash: SIGKILL the worker pid this adapter captured
    os.kill(pid, signal.SIGKILL)
    t_end = time.monotonic() + 5
    while time.monotonic() < t_end and MontyREPLAdapter.worker_alive(pid):
        time.sleep(0.02)
    r1 = a.execute("print(x)")
    r2 = a.execute("print(x, artifacts)")
    out["external_kill_then_cells"] = {"killed_pid_dead": not MontyREPLAdapter.worker_alive(pid),
                                       "cell_after_kill": _brief(r1), "next_cell": _brief(r2),
                                       "sessions_started": a.sessions_started}
    disposal = a.close()
    out["close_proof"] = {"pids_captured": len(a.worker_pids), "all_dead_after_close": all(disposal.values()),
                          "per_pid": {str(k): v for k, v in disposal.items()}}
    return out


PROBES = {
    "import os": "import os",
    "from os import system": "from os import system",
    "__import__('os')": "__import__('os')",
    "eval('1+1')": "print(eval('1+1'))",
    "exec('x=1')": "exec('x=1')",
    "open('/etc/hostname')": "print(open('/etc/hostname').read())",
    "''.__class__": "print(''.__class__)",
    "getattr(object,'__class__')": "print(getattr(object, '__class__'))",
    "object.__subclasses__()": "print(object.__subclasses__())",
    "globals()": "print(list(globals())[:3])",
}


def s6_security_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, code in PROBES.items():
        env = REPLEnvironment(context="c")
        r_cp = env.execute(code)
        env2 = REPLEnvironment(context="c")
        with MontyREPLAdapter(env2) as a:
            r_mo = a.execute(code)
        env3 = REPLEnvironment(context="c")
        with MontyREPLAdapter(env3, ast_gate=False) as a:
            r_nat = a.execute(code)
        env4 = REPLEnvironment(context="c")
        cont = env4._run_python_code(code)
        out[name] = {"cpython_warm": _brief(r_cp), "monty": _brief(r_mo), "monty_native_no_gate": _brief(r_nat),
                     "cpython_contained": cont[-200:]}
    return out


COVERAGE = {
    "statistics.mean": "print(statistics.mean([1, 2, 3]))",
    "numpy (preloaded name)": "print(numpy.arange(3).tolist())",
    "import fractions": "import fractions\nprint(fractions.Fraction(1, 3))",
    "generator": "def g():\n    yield 1\nprint(list(g()))",
    "exception subclass": "class E(Exception):\n    pass\ntry:\n    raise E('x')\nexcept E:\n    print('caught')",
    "set + sorted + f-string": "s = {3, 1, 2}\nprint(f'{sorted(s)}')",
    "json round trip": "print(json.loads(json.dumps({'a': [1, 2]})))",
    "re + collections": "print(re.findall(r'a+', 'caab'), collections.Counter('aab').most_common(1))",
}


def s7_coverage() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, code in COVERAGE.items():
        env = REPLEnvironment(context="c")
        r_cp = env.execute(code)
        env2 = REPLEnvironment(context="c")
        with MontyREPLAdapter(env2) as a:
            r_mo = a.execute(code)
        out[name] = {"cpython_warm": _brief(r_cp), "monty": _brief(r_mo),
                     "same": (r_cp.output, r_cp.error) == (r_mo.output, r_mo.error)}
    return out


def s8_startup() -> dict[str, Any]:
    """Cold start: parent env, own-pool adapter first cell, session on a pre-warmed shared pool."""
    import pydantic_monty as pm

    out: dict[str, Any] = {}
    reps = 8
    t_env, t_own, t_unc, t_sess = [], [], [], []
    for _ in range(reps):
        t = time.perf_counter(); env = REPLEnvironment(context="c"); t_env.append((time.perf_counter() - t) * 1000)
        for contain, acc in ((True, t_own), (False, t_unc)):
            env = REPLEnvironment(context="c")
            t = time.perf_counter()
            a = MontyREPLAdapter(env, contain_worker=contain); a.execute("x = 1")
            acc.append((time.perf_counter() - t) * 1000)
            a.close()
    with pm.Monty(min_processes=2, max_processes=2, max_checkouts_per_worker=1) as pool:
        time.sleep(0.5)
        for _ in range(reps):
            env = REPLEnvironment(context="c")
            t = time.perf_counter()
            a = MontyREPLAdapter(env, pool=pool); a.execute("x = 1")
            t_sess.append((time.perf_counter() - t) * 1000)
            a.dispose_session(); time.sleep(0.4)  # let the pool refill a spare worker
    out["parent_env_ctor_ms_median"] = round(statistics.median(t_env), 2)
    out["adapter_own_pool_contained_worker_first_cell_ms_median"] = round(statistics.median(t_own), 2)
    out["adapter_own_pool_uncontained_worker_first_cell_ms_median"] = round(statistics.median(t_unc), 2)
    out["adapter_shared_uncontained_pool_first_cell_ms_median"] = round(statistics.median(t_sess), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/experiments/monty-slice-results.json")
    ap.add_argument("--reps", type=int, default=WARM_REPS)
    ap.add_argument("--skip-extras", action="store_true")
    args = ap.parse_args()
    assert monty_enabled(), f"{ENV_FLAG} must be on for the adapter arm"
    import pydantic_monty as pm

    result: dict[str, Any] = {"pydantic_monty": pm.__version__, "python": sys.version.split()[0], "reps": args.reps,
                              "cases": {}}
    for case in CASES:
        cp = run_cpython_warm(case, args.reps)
        mo = run_monty(case, args.reps)  # default: worker under the SEC-EXEC Landlock+seccomp shim
        mo_unc = run_monty(case, args.reps, contain_worker=False)
        cn = run_contained(case)
        result["cases"][case.id] = {"test": case.test, "cpython_warm": cp, "monty": mo,
                                    "monty_uncontained_worker": {k: mo_unc[k] for k in ("cold_ms", "warm_ms_median", "warm_ms_p95", "oracle") if k in mo_unc},
                                    "cpython_contained": cn, "compare": compare(cp, mo)}
        print(f"{case.id} {case.test}: cpython_oracle={cp['oracle']} monty_oracle={mo['oracle']} "
              f"identical={result['cases'][case.id]['compare']['all_equal']} "
              f"warm_ms cp={cp['warm_ms_median']} monty={mo['warm_ms_median']}", flush=True)
    if not args.skip_extras:
        for name, fn in [("S1_checkpoint_roundtrip", s1_checkpoint_roundtrip), ("S2_denied_callbacks", s2_denied_callbacks),
                         ("S3_delayed_callbacks", s3_delayed_callbacks),
                         ("S4_terminal_resource_errors", s4_terminal_resource_errors),
                         ("S5_disposal_and_crash", s5_disposal_and_crash), ("S6_security_probes", s6_security_probes),
                         ("S7_coverage", s7_coverage), ("S8_startup", s8_startup)]:
            result[name] = fn()
            print("done", name, flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
