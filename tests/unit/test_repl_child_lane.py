"""D-RI-CONTAIN-T2: the opt-in contained child lane (ORCHESTRATOR_REPL_CHILD_LANE).

Real children, real kernel containment, no mocks of the process boundary. Only children that
THESE tests spawn are ever signalled (their pid is captured from the lane).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src.repl_environment import child_lane, exec_sandbox
from src.repl_environment.environment import REPLEnvironment
from src.repl_environment.types import REPLConfig

_CAPS = exec_sandbox.capabilities()
needs_kernel = pytest.mark.skipif(
    not (_CAPS["landlock"] and _CAPS["netdeny"]), reason="host cannot enforce Landlock+seccomp (the lane refuses to run)"
)

CONTEXT = "alpha beta gamma\nline two beta\nline three delta"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    from src.features import reset_features

    reset_features()
    for k in list(os.environ):
        if k.startswith("ORCHESTRATOR_REPL_CHILD") or k in ("ORCHESTRATOR_REPL_CHILD_LANE",
                                                          "ORCHESTRATOR_REPL_TOOL_ARG_VALIDATION"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", raising=False)
    yield
    reset_features()


@pytest.fixture
def lane_on(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_LANE", "1")
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP", "off")  # cgroup behaviour has its own tests


def _env(timeout=15, ctx=CONTEXT):
    return REPLEnvironment(context=ctx, config=REPLConfig(timeout_seconds=timeout))


@pytest.fixture
def make_env():
    made = []

    def make(**kw):
        e = _env(**kw)
        made.append(e)
        return e

    yield make
    for e in made:
        e.close_child_lane()


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


# ── flag-off identity ────────────────────────────────────────────────────────


def test_flag_off_is_the_in_process_path(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_REPL_CHILD_LANE", raising=False)
    monkeypatch.setattr(child_lane, "ChildLane", None)  # any lane construction would blow up
    e = _env()
    r = e.execute("x = 41\nprint(x + 1, peek(5))\nartifacts['a'] = 1")
    assert (r.output, r.error, r.is_final) == ("42 alpha\n", None, False)
    assert not hasattr(e, "_child_lane")
    assert e._globals["x"] == 41 and e.artifacts == {"a": 1}


def test_flag_off_installs_no_validation_wrappers():
    e = _env()
    peek = e._globals["peek"]
    assert not hasattr(peek, "__wrapped__") or peek.__wrapped__.__self__ is e  # counted() -> bound method only
    # a bad argument reaches the tool's own (pre-existing) handling, not the validator
    assert "needs integer" in e.execute("print(peek(n=[1]))").output
    assert set(e._globals) == set(_env()._globals)


def test_flag_off_and_on_expose_the_same_names(monkeypatch, lane_on):
    on = _env()
    monkeypatch.delenv("ORCHESTRATOR_REPL_CHILD_LANE")
    assert set(on._globals) == set(_env()._globals)


PARITY_CELLS = [
    "print('hello')",
    "x = [1, 2, 3]\nprint(sum(x))",
    "print(len(context), context_len())",
    "print(peek(10))",
    "print(peek(10, offset=-8))",
    "print(grep('beta'))",
    "print(grep('nomatch'))",
    "print(chunk_context(2, 3)[0]['count' if False else 'char_count'])",
    "1/0",
    "d = {}\nd['missing']",
    "raise ValueError('boom')",
    "x = (\n",
    "print(undefined_name)",
    "import heapq\nprint(heapq.nsmallest(2, [5, 3, 9]))",
    "import os",
    "from collections import Counter\nprint(Counter('aab').most_common(1))",
    "FINAL('the answer')",
    "FINAL(42)",
    "FINAL([1, 2])",
    "FINAL(result='aliased')",
    "FINAL()",
    "artifacts['k'] = {'n': [1, 2]}\nFINAL_VAR('k')",
    "FINAL_VAR('absent')",
    "print('a' * 5000)",
    "import sys",
    "print(f'{1+1}')\nprint(my_role())",
]


@needs_kernel
@pytest.mark.parametrize("code", PARITY_CELLS)
def test_lane_result_matches_in_process(code, monkeypatch, make_env):
    monkeypatch.delenv("ORCHESTRATOR_REPL_CHILD_LANE", raising=False)
    ref = _env().execute(code)
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_LANE", "1")
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP", "off")
    env = make_env()
    got = env.execute(code)
    want_error = ref.error
    if want_error and not want_error.startswith("SyntaxError") and want_error.partition(": ")[0].isidentifier():
        # the lane implies the structured exception schema (item (a)); same type and message, schema-rendered
        from src.repl_environment import exception_schema

        etype, _, rest = want_error.partition(": ")
        msg, sep, hint = rest.partition("\n--- Tool Usage Hint")
        want_error = exception_schema.render(etype, msg, "model_code", (sep + hint) if sep else "")
        if "FINAL" in code:  # FINAL/FINAL_VAR errors are raised by the brokered parent tool
            want_error = want_error.replace('"model_code"', '"tool:%s"' % ("FINAL_VAR" if "FINAL_VAR" in code else "FINAL"))
    assert (got.output, got.is_final, got.final_answer, got.error) == (ref.output, ref.is_final, ref.final_answer, want_error)
    assert type(got) is type(ref)


@needs_kernel
def test_state_and_artifact_parity_across_cells(monkeypatch, make_env):
    cells = ["a = 1\nb = [a, 2]", "b.append(a + 2)\nc = {'k': b}", "artifacts['x'] = c\nprint(b, c)", "del a\ntry:\n    a\nexcept NameError:\n    print('del ok')"]
    monkeypatch.delenv("ORCHESTRATOR_REPL_CHILD_LANE", raising=False)
    ref = _env()
    ref_out = [ref.execute(c).output for c in cells]
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_LANE", "1")
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP", "off")
    env = make_env()
    assert [env.execute(c).output for c in cells] == ref_out
    user = lambda e: {k: v for k, v in e._globals.items() if k not in e._builtin_global_keys}  # noqa: E731
    assert user(env) == user(ref) and env.artifacts == ref.artifacts
    assert env.get_state() == ref.get_state()


# ── warm reuse ───────────────────────────────────────────────────────────────


@needs_kernel
def test_one_warm_child_serves_every_cell_of_a_request(lane_on, make_env):
    env = make_env()
    env.execute("n = 0")
    lane = env._child_lane
    pid = lane.pid
    assert pid and _alive(pid)
    for i in range(1, 6):
        r = env.execute("n += 1\nprint(n)")
        assert r.output == f"{i}\n" and lane.pid == pid
    assert lane.spawns == 1 and lane.respawns == 0
    # the child is a different process, in its own session, not the parent
    assert pid != os.getpid() and os.getsid(pid) == pid


@needs_kernel
def test_close_kills_and_reaps_the_child(lane_on, make_env):
    env = make_env()
    env.execute("x = 1")
    pid = env._child_lane.pid
    rep = env.close_child_lane()
    assert rep["reaped"] and rep["survivors"] == [] and not _alive(pid)
    assert env.close_child_lane() == rep  # idempotent
    assert env.execute("print(x)").output == "1\n"  # a later cell restarts from the parent's state


@needs_kernel
def test_garbage_collected_environment_does_not_leak_its_child(lane_on):
    import gc

    env = _env()
    env.execute("x = 1")
    pid = env._child_lane.pid
    del env
    gc.collect()
    deadline = time.time() + 5
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


# ── deadline kill ────────────────────────────────────────────────────────────


@needs_kernel
@pytest.mark.parametrize(
    "code",
    [
        "while True:\n    pass",
        "while True:\n    try:\n        pass\n    except Exception:\n        pass",
        "while True:\n    try:\n        pass\n    except BaseException:\n        pass",
        "i = 0\nwhile True:\n    i += 1\n    try:\n        context_len()\n    except Exception:\n        pass",
    ],
)
def test_deadline_kills_the_child_even_if_code_swallows_exceptions(code, lane_on, make_env):
    env = make_env(timeout=2)
    env.execute("warm = 1")
    pid = env._child_lane.pid
    t0 = time.monotonic()
    r = env.execute(code)
    assert time.monotonic() - t0 < 6
    assert r.error == env.timeout_error and env.timed_out
    assert not _alive(pid)
    assert env._child_lane.last_teardown["reaped"] and env._child_lane.last_teardown["survivors"] == []
    # terminal exactly like the in-process timeout: no reuse, no new child
    again = env.execute("print('late')")
    assert again.error == env.timeout_error and again.output == ""
    assert env._child_lane.pid is None


@needs_kernel
def test_blocking_syscall_is_killed_too(lane_on, make_env):
    env = make_env(timeout=2)
    r = env.execute("import time\ntime.sleep(60)")
    assert r.error == env.timeout_error


@needs_kernel
def test_cpu_budget_is_finite(monkeypatch, lane_on, make_env):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CPU_S", "1")
    env = make_env(timeout=30)
    env.execute("x = 1")
    pid = env._child_lane.pid
    t0 = time.monotonic()
    r = env.execute("while True:\n    pass")
    assert time.monotonic() - t0 < 10  # SIGXCPU long before the 30 s wall deadline
    assert r.error and "REPLChildCrashed" in r.error and "SIGXCPU" in r.error
    assert not _alive(pid) and not env.timed_out
    assert env.execute("print(x)").output == "1\n"


# ── memory cap ───────────────────────────────────────────────────────────────


@needs_kernel
def test_rlimit_as_caps_the_child_not_the_parent(monkeypatch, lane_on, make_env):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_AS_MB", "300")
    env = make_env()
    env.execute("ok = 1")
    pid = env._child_lane.pid
    assert env._child_lane.limits["child_reported"]["as_bytes"] == 300 * 1024 * 1024
    row = next(ln for ln in Path(f"/proc/{pid}/limits").read_text().splitlines() if ln.startswith("Max address space"))
    assert row.split()[3:5] == [str(300 * 1024 * 1024)] * 2  # soft == hard: the child cannot raise it
    r = env.execute("n = 90_000_000\nbig = list(range(n))\nprint(len(big))")
    assert r.error and "MemoryError" in r.error  # or the child died: both are an error result, never a hang
    # the request survives: state intact, lane usable
    r2 = env.execute("print(ok)")
    assert r2.output == "1\n" and r2.error is None
    import resource

    assert resource.getrlimit(resource.RLIMIT_AS)[0] != 300 * 1024 * 1024  # the API process was never limited


@needs_kernel
def test_other_budgets_are_applied_and_verified(lane_on, make_env):
    env = make_env()
    env.execute("x = 1")
    lim = env._child_lane.limits["child_reported"]
    assert lim["core"] == 0 and lim["nofile"] == 64 and lim["cpu"] == 600 and lim["fsize"] == 8 * 1024 * 1024
    pid = env._child_lane.pid
    rows = {ln[:25].strip(): ln[25:].split() for ln in Path(f"/proc/{pid}/limits").read_text().splitlines()[1:]}
    assert rows["Max core file size"][:2] == ["0", "0"]
    assert rows["Max open files"][:2] == ["64", "64"]
    assert rows["Max file size"][:2] == [str(8 * 1024 * 1024)] * 2
    assert rows["Max cpu time"][0] == "600"


@needs_kernel
def test_output_is_capped_at_the_source(monkeypatch, lane_on, make_env):
    monkeypatch.setattr(child_lane.LaneConfig, "from_env", classmethod(lambda cls: cls(output_chars=10_000, cgroup_mode="off")))
    env = make_env()
    r = env.execute("for _ in range(2000):\n    print('x' * 1000)")
    assert r.error is None and "truncated" in r.output and len(r.output) < 200_000  # 2 MB printed, ~10 kB kept


@needs_kernel
def test_oversized_child_frame_is_a_crash_not_an_allocation(monkeypatch, lane_on, make_env):
    monkeypatch.setattr(
        child_lane.LaneConfig, "from_env", classmethod(lambda cls: cls(max_frame=50_000, cgroup_mode="off", output_chars=1_000_000))
    )
    env = make_env()
    env.execute("a = 1")
    pid = env._child_lane.pid
    r = env.execute("print('y' * 400000)")
    assert r.error and "REPLChildCrashed" in r.error and "exceeds" in r.error
    assert not _alive(pid) and not env.timed_out
    assert env.execute("print(a)").output == "1\n"


# ── fail closed ──────────────────────────────────────────────────────────────


def test_refuses_when_containment_cannot_be_enforced(monkeypatch, lane_on, make_env):
    def boom(*a, **k):
        raise exec_sandbox.ContainmentUnavailable("Landlock is not usable on this host")

    monkeypatch.setattr(exec_sandbox, "wrap", boom)
    env = make_env()
    r = env.execute("ran_inline = 1")
    assert r.error and "CONTAINED REPL LANE unavailable" in r.error and "NOT run" in r.error
    assert "ran_inline" not in env._globals  # never executed inline
    assert env._child_lane.pid is None


def test_refuses_to_run_without_an_address_space_cap(monkeypatch, lane_on, make_env):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_AS_MB", "0")
    env = make_env()
    r = env.execute("ran_inline = 1")
    assert r.error and "RLIMIT_AS cannot be disabled" in r.error and "ran_inline" not in env._globals


@needs_kernel
def test_refuses_when_the_child_does_not_report_the_requested_limits(monkeypatch, lane_on, make_env):
    real = child_lane._ChildProc.spawn

    def spawn(self, task_id):
        real(self, task_id)
        self.limits = dict(self.limits, nofile=63)  # the parent expects something the child did not apply

    monkeypatch.setattr(child_lane._ChildProc, "spawn", spawn)
    env = make_env()
    r = env.execute("ran_inline = 1")
    assert r.error and "limit nofile not applied" in r.error and "ran_inline" not in env._globals
    assert env._child_lane.pid is None


@needs_kernel
def test_shim_refusal_is_fail_closed(monkeypatch, lane_on, make_env):
    # an unsatisfiable limit makes the shim exit 126 before exec: the lane refuses, nothing runs
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_AS_MB", "1")
    env = make_env()
    r = env.execute("ran_inline = 1")
    assert r.error and "unavailable" in r.error and "ran_inline" not in env._globals


def test_bundle_context_is_refused(lane_on, make_env):
    env = make_env()
    env._context_bundle = object()
    with pytest.raises(child_lane.LaneUnavailable, match="context bundle"):
        child_lane.ChildLane(env)._build_init(env)


@needs_kernel
def test_unsupported_restored_state_is_refused(lane_on, make_env):
    env = make_env()
    env._globals["s"] = (i for i in range(2))
    r = env.execute("print('hi')")
    assert r.error and "unsupported REPL state" in r.error and "s (generator)" in r.error
    assert env._child_lane.pid is None


# ── state rules ──────────────────────────────────────────────────────────────


@needs_kernel
def test_unsupported_new_state_rolls_the_cell_back(lane_on, make_env):
    env = make_env()
    env.execute("keep = 1")
    r = env.execute("keep = 2\nt = (i for i in range(3))\nprint('out stays')")
    assert r.output == "out stays\n" and "REPLLaneStateError" in r.error and "t: generator" in r.error
    assert env._globals["keep"] == 1 and "t" not in env._globals
    assert env.execute("print(keep)").output == "1\n"  # the child was re-seeded from the parent's state
    assert env._child_lane.respawns == 0  # a deliberate rollback is not a crash


@needs_kernel
def test_reset_and_restore_reseed_the_child(lane_on, make_env):
    env = make_env()
    env.execute("v = 7")
    env.reset()
    assert env.execute("try:\n    v\nexcept NameError:\n    print('gone')").output == "gone\n"


@needs_kernel
def test_nested_definitions_are_refused_with_a_reason(lane_on, make_env):
    r = make_env().execute("h = [].append")
    assert "h: builtin_function_or_method" in r.error and "top-level def" in r.error


# ── crash isolation ──────────────────────────────────────────────────────────


@needs_kernel
def test_killed_child_does_not_take_the_request_down(lane_on, make_env):
    env = make_env(timeout=30)
    env.execute("kept = [1, 2, 3]\nartifacts['a'] = 'v'")
    lane = env._child_lane
    old = lane.pid
    result = {}

    def run():
        result["r"] = env.execute("while True:\n    pass")

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.6)
    os.kill(old, signal.SIGKILL)  # a child THIS test spawned
    t.join(10)
    assert not t.is_alive()
    r = result["r"]
    assert r.error and "REPLChildCrashed" in r.error and "SIGKILL" in r.error
    assert not env.timed_out and not _alive(old)
    # next cell: fresh child, last completed state restored, parent untouched
    r2 = env.execute("print(kept, artifacts['a'])")
    assert r2.output == "[1, 2, 3] v\n" and r2.error is None
    assert lane.pid and lane.pid != old and lane.respawns == 1


@needs_kernel
def test_repeated_crashes_stop_respawning(monkeypatch, lane_on, make_env):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_MAX_RESPAWNS", "2")
    env = make_env()
    env.execute("x = 1")
    lane = env._child_lane
    for _ in range(2):  # two genuine crashes (SIGKILL of a child this test spawned) are tolerated
        os.kill(lane.pid, signal.SIGKILL)
        deadline = time.time() + 5
        while _alive(lane.pid) and time.time() < deadline:
            time.sleep(0.02)
        assert env.execute("print(x)").output == "1\n"
    assert lane.respawns == 2
    os.kill(lane.pid, signal.SIGKILL)
    time.sleep(0.2)
    r = env.execute("y = 2")
    assert r.error and "refusing to respawn" in r.error and "y" not in env._globals


@needs_kernel
def test_descendants_are_killed_with_the_child(tmp_path):
    cfg = child_lane.LaneConfig(cgroup_mode="off")
    proc = child_lane._ChildProc(cfg)
    proc.popen = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 300 & sleep 300 & wait"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, start_new_session=True
    )
    proc.pid = proc.popen.pid
    deadline = time.time() + 5
    while len(child_lane._session_members(proc.pid)) < 3 and time.time() < deadline:
        time.sleep(0.05)
    members = child_lane._session_members(proc.pid)
    assert len(members) >= 3
    rep = proc.terminate()
    assert rep["reaped"] and rep["survivors"] == []
    assert not any(_alive(p) for p in members)


@needs_kernel
def test_child_has_no_secrets_no_network_and_a_private_scratch(monkeypatch, lane_on, make_env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("HF_TOKEN", "hf_not_real")
    env = make_env()
    env.execute("x = 1")
    pid = env._child_lane.pid
    environ = Path(f"/proc/{pid}/environ").read_bytes().decode()
    assert "sk-test-not-real" not in environ and "hf_not_real" not in environ
    cwd = os.readlink(f"/proc/{pid}/cwd")
    assert Path(cwd).name.startswith("epyc-repl-lane-") and oct(Path(cwd).stat().st_mode & 0o777) == "0o700"
    assert os.getpgid(pid) == pid
    env.close_child_lane()
    assert not Path(cwd).exists()  # scratch removed on teardown


# ── brokered callbacks ───────────────────────────────────────────────────────


@needs_kernel
@pytest.mark.parametrize(
    "call",
    ["web_fetch('http://127.0.0.1:1/')", "run_shell('echo hi')", "run_python_code('1')",
     "file_write_safe('/nonexistent-root/x', 'y')", "TOOL('anything')", "CALL('anything')",
     "list_tools()", "llm_call('hi')", "llm_batch(['hi'])", "delegate('b')", "mark_finding('f')",
     "remember('x')", "STUCK('why')"],
)
def test_mutating_and_unbounded_tools_are_brokered_not_refused(call, lane_on):
    """Every one of these used to be a stub that raised ``not available in the contained REPL lane``.
    They now reach the parent (what each returns or raises is the tool's own business; the exact
    in-process equivalence is asserted in test_repl_child_lane_tools.py)."""
    from unittest.mock import MagicMock

    env = REPLEnvironment(context=CONTEXT, config=REPLConfig(timeout_seconds=15),
                          tool_registry=MagicMock(), llm_primitives=MagicMock())
    try:
        r = env.execute(f"try:\n    print({call})\nexcept Exception as e:\n    print('RAISED', type(e).__name__, e)")
        assert "not available in the contained REPL lane" not in (r.error or "") + r.output
    finally:
        env.close_child_lane()


@needs_kernel
def test_non_tool_module_preloads_stay_refused(lane_on):
    env = REPLEnvironment(context=CONTEXT, config=REPLConfig(timeout_seconds=15))
    try:
        if "numpy" not in env._globals:
            pytest.skip("numpy is not installed")
        r = env.execute("print(numpy.array([1]))")
        assert r.error and "not available in the contained REPL lane" in r.error
    finally:
        env.close_child_lane()


@needs_kernel
def test_callback_budget_is_the_parents(monkeypatch, lane_on, make_env):
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", "5")
    env = make_env()
    r = env.execute("for i in range(6):\n    context_len()")
    assert r.error and "Tool-call budget exceeded" in r.error
    assert env.execute("for i in range(5):\n    context_len()").error is None  # reset per cell


@needs_kernel
def test_callback_counters_land_in_the_parent(lane_on, make_env):
    env = make_env()
    env.execute("peek(5)\ngrep('beta')")
    assert env._exploration_calls == 2 and len(env._exploration_log.events) >= 2


@needs_kernel
def test_callback_per_cell_hard_cap(monkeypatch, lane_on, make_env):
    monkeypatch.setattr(child_lane.LaneConfig, "from_env", classmethod(lambda cls: cls(max_calls_per_cell=3, cgroup_mode="off")))
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_TOOL_CALLS", "0")
    r = make_env().execute("for i in range(4):\n    context_len()")
    assert r.error and "callback cap exceeded" in r.error


@needs_kernel
def test_oversized_reply_is_refused_not_streamed(monkeypatch, lane_on, make_env):
    monkeypatch.setattr(child_lane.LaneConfig, "from_env", classmethod(lambda cls: cls(max_reply_bytes=1000, cgroup_mode="off")))
    r = make_env(ctx="z" * 50_000).execute("print(len(peek(40000)))")
    assert r.error and "reply cap" in r.error


def test_callbacks_outside_an_admitted_cell_are_rejected(lane_on, make_env):
    env = make_env()
    lane = child_lane.ChildLane(env)
    lane._build_init(env)  # captures the brokered callables, as a started lane does
    frame = {"t": "call", "id": "x", "seq": 1, "name": "peek", "args": [3], "kwargs": {}}
    with pytest.raises(child_lane.LaneCrash):
        lane._serve_call(env, frame, time.monotonic() + 5)  # no admitted cell: a late/forged reply path
    lane.admission_open, lane._nonce = True, "real"
    with pytest.raises(child_lane.LaneCrash):
        lane._serve_call(env, dict(frame, id="other"), time.monotonic() + 5)  # wrong request identity
    lane._nonce = "x"
    with pytest.raises(child_lane.LaneCrash):
        lane._serve_call(env, dict(frame, seq=7), time.monotonic() + 5)  # replayed/skipped sequence
    lane._expect_seq = 0
    ok = lane._serve_call(env, frame, time.monotonic() + 5)
    assert ok["ok"] and ok["value"] == "alp"
    lane.close()


@needs_kernel
def test_lease_loss_revokes_admission_and_kills_the_child(lane_on, make_env):
    t_end = time.monotonic() + 1.0
    env = make_env(timeout=30)
    env._lane_lease_validator = lambda: time.monotonic() < t_end
    env.execute("x = 1")
    pid = env._child_lane.pid
    t0 = time.monotonic()
    r = env.execute("while True:\n    pass")
    assert time.monotonic() - t0 < 8
    assert r.error and "REPLLaneRevoked" in r.error and "lease lost" in r.error
    assert not _alive(pid) and not env.timed_out


@needs_kernel
def test_cancel_kills_the_child(lane_on, make_env):
    env = make_env(timeout=30)
    env.execute("x = 1")
    pid = env._child_lane.pid
    threading.Timer(0.5, env._child_lane.cancel).start()
    r = env.execute("while True:\n    pass")
    assert r.error and "REPLLaneRevoked" in r.error and not _alive(pid)


@needs_kernel
def test_late_state_from_a_killed_cell_is_never_applied(lane_on, make_env):
    env = make_env(timeout=2)
    env.execute("v = 1")
    r = env.execute("v = 2\nwhile True:\n    pass")
    assert r.error == env.timeout_error
    assert env._globals["v"] == 1  # the half-finished cell left no trace in the parent


# ── cgroup v2 ────────────────────────────────────────────────────────────────


def _fake_cgroup_parent(tmp_path, controllers="memory pids"):
    parent = tmp_path / "cg"
    parent.mkdir()
    (parent / "cgroup.subtree_control").write_text(controllers + "\n")
    return parent


def _fake_mkdir(monkeypatch):
    import shutil

    real = Path.mkdir
    real_rmdir = Path.rmdir

    def rmdir(self):
        if self.name.startswith("epyc-repl-"):
            shutil.rmtree(self)  # a real cgroup dir holds no regular files; the fake one does
        else:
            real_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", rmdir)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        real(self, mode, parents, exist_ok)
        if self.name.startswith("epyc-repl-"):
            for f in ("memory.max", "memory.swap.max", "memory.oom.group", "pids.max", "cgroup.procs", "cgroup.kill", "memory.events"):
                (self / f).write_text("")

    monkeypatch.setattr(Path, "mkdir", mkdir)


def test_cgroup_limits_are_written_when_controllers_are_delegated(tmp_path, monkeypatch):
    parent = _fake_cgroup_parent(tmp_path)
    _fake_mkdir(monkeypatch)
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP_PARENT", str(parent))
    cfg = child_lane.LaneConfig(as_mb=1000, cgroup_mode="require", cgroup_pids_max=33)
    cg = child_lane._Cgroup.create(cfg)
    assert cg is not None and cg.memory_applied and cg.pids_applied
    assert (cg.path / "memory.max").read_text() == str(1512 * 1024 * 1024)
    assert (cg.path / "memory.swap.max").read_text() == "0"
    assert (cg.path / "memory.oom.group").read_text() == "1"
    assert (cg.path / "pids.max").read_text() == "33"
    cg.attach(4242)
    assert (cg.path / "cgroup.procs").read_text() == "4242"
    cg.kill_all()
    assert (cg.path / "cgroup.kill").read_text() == "1"
    assert (parent / "cgroup.subtree_control").read_text().strip() == "memory pids"  # never edited


def test_cgroup_required_but_controller_not_delegated_fails_closed(tmp_path, monkeypatch):
    parent = _fake_cgroup_parent(tmp_path, controllers="")
    _fake_mkdir(monkeypatch)
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP_PARENT", str(parent))
    with pytest.raises(child_lane.LaneUnavailable):
        child_lane._Cgroup.create(child_lane.LaneConfig(cgroup_mode="require"))
    assert not [p for p in parent.iterdir() if p.name.startswith("epyc-repl-")]  # cleaned up
    # auto: continues without the cgroup limit (RLIMIT_AS still applies)
    assert child_lane._Cgroup.create(child_lane.LaneConfig(cgroup_mode="auto")).memory_applied is False


def test_cgroup_required_but_unwritable_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP_PARENT", str(tmp_path / "does-not-exist"))
    with pytest.raises(child_lane.LaneUnavailable):
        child_lane._Cgroup.create(child_lane.LaneConfig(cgroup_mode="require"))
    assert child_lane._Cgroup.create(child_lane.LaneConfig(cgroup_mode="auto")) is None


@needs_kernel
def test_lane_refuses_cells_when_a_required_cgroup_limit_is_unavailable(tmp_path, monkeypatch, lane_on, make_env):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP", "require")
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP_PARENT", str(tmp_path / "nope"))
    env = make_env()
    r = env.execute("ran_inline = 1")
    assert r.error and "cgroup" in r.error and "ran_inline" not in env._globals


@needs_kernel
@pytest.mark.skipif(not os.access("/sys/fs/cgroup", os.W_OK), reason="no writable cgroup v2 directory on this host")
def test_live_cgroup_membership_kill_and_removal(monkeypatch, lane_on, make_env):
    monkeypatch.setenv("ORCHESTRATOR_REPL_CHILD_CGROUP", "auto")
    env = make_env(timeout=2)
    env.execute("x = 1")
    cg = env._child_lane._proc.cgroup
    if cg is None:
        pytest.skip("cgroup not usable here")
    pid = env._child_lane.pid
    assert pid in cg.members()
    r = env.execute("while True:\n    pass")
    assert r.error == env.timeout_error
    assert not _alive(pid) and not cg.path.exists()
