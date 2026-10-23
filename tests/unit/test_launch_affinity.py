"""Deterministic host-thread pinning for pinned launches (workspace-58, 2026-10-22).

Fixture shape mirrors the live finding: a `taskset -c 184-191` HIP llama-server whose
main thread and 194 others read 184-191 while ONE early ROCr thread read 0-191.
Everything runs against a synthetic /proc tree; no real process is touched.
"""

from __future__ import annotations

import errno
import types
from pathlib import Path

import pytest

from scripts.server import launch_affinity as la

LANE = "184-191"
PREFIX = ["numactl", "--membind=3", "--", "taskset", "-c", LANE]


def _write_task(proc: Path, pid: int, tid: int, mask: str, comm: str = "llama-server") -> None:
    task = proc / str(pid) / "task" / str(tid)
    task.mkdir(parents=True, exist_ok=True)
    (task / "status").write_text(f"Name:\t{comm}\nCpus_allowed_list:\t{mask}\n")
    (task / "comm").write_text(comm + "\n")


def _make_proc(tmp_path: Path, pid: int, masks: dict[int, str], ppid: int) -> Path:
    proc = tmp_path / "proc"
    for tid, mask in masks.items():
        _write_task(proc, pid, tid, mask)
    (proc / str(pid) / "stat").write_text(f"{pid} (llama-server) S {ppid} {pid} {pid} 0\n")
    return proc


def _finding(tmp_path: Path, ppid: int = 1000) -> tuple[Path, int]:
    pid = 3638945
    masks = {pid + i: LANE for i in range(196)}
    masks[pid + 7] = "0-191"  # the ROCr thread the diag found (tid = pid + 7)
    return _make_proc(tmp_path, pid, masks, ppid), pid


def _fs_setter(proc: Path, pid: int, calls: list[tuple[int, str]]):
    def setter(tid: int, cpus) -> None:
        mask = la.format_cpu_list(cpus)
        calls.append((tid, mask))
        _write_task(proc, pid, tid, mask)

    return setter


# ── parsing ──────────────────────────────────────────────────────────────────


def test_parse_and_format_round_trip():
    assert la.parse_cpu_list("184-191") == frozenset(range(184, 192))
    assert la.parse_cpu_list("0-3,96-99,7") == frozenset([0, 1, 2, 3, 7, 96, 97, 98, 99])
    assert la.format_cpu_list(la.parse_cpu_list("0-3,96-99,7")) == "0-3,7,96-99"
    assert la.format_cpu_list([5]) == "5"


@pytest.mark.parametrize("bad", ["", " ", "0-", "a-3", "3-1", "1,,2"])
def test_parse_rejects_garbage(bad):
    with pytest.raises(la.LaunchAffinityError):
        la.parse_cpu_list(bad)


def test_prefix_cpuset():
    assert la.prefix_cpuset(PREFIX) == LANE
    assert la.prefix_cpuset(["taskset", "-c", "0-95"]) == "0-95"
    assert la.prefix_cpuset([]) is None
    assert la.prefix_cpuset(["numactl", "--interleave=all", "--"]) is None
    with pytest.raises(la.LaunchAffinityError):
        la.prefix_cpuset(["taskset", "0x3"])


# ── layer 1: the ROCr env knob ───────────────────────────────────────────────


def test_pinned_launch_gets_rocr_override_even_over_ambient():
    env = {"HSA_OVERRIDE_CPU_AFFINITY_DEBUG": "1", "PATH": "/bin"}
    assert la.apply_pinned_launch_env(env, PREFIX) is True
    assert env["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] == "0"
    assert env["PATH"] == "/bin"


def test_unpinned_launch_env_untouched():
    env = {"PATH": "/bin"}
    assert la.apply_pinned_launch_env(env, []) is False
    assert env == {"PATH": "/bin"}


# ── layer 2: re-pin + assert ─────────────────────────────────────────────────


def test_escapes_find_the_rocr_thread(tmp_path):
    proc, pid = _finding(tmp_path)
    escapes = la.affinity_escapes(pid, LANE, proc)
    assert [(t.tid, t.cpu_list) for t in escapes] == [(pid + 7, "0-191")]
    assert len(la.read_task_affinities(pid, proc)) == 196


def test_narrower_submask_is_not_an_escape(tmp_path):
    proc = _make_proc(tmp_path, 10, {10: LANE, 11: "184", 12: "185,189"}, ppid=1)
    assert la.affinity_escapes(10, LANE, proc) == []


def test_pin_all_tasks_repins_only_the_escape(tmp_path):
    proc, pid = _finding(tmp_path)
    _write_task(proc, pid, pid + 300, "186")  # an OpenMP-place thread on a sub-mask
    calls: list[tuple[int, str]] = []
    result = la.pin_all_tasks(pid, LANE, proc_root=proc, setter=_fs_setter(proc, pid, calls))
    assert calls == [(pid + 7, LANE)]
    assert result.ok and result.escapes_after == [] and result.tasks_seen == 197
    # the sub-mask thread was NOT widened back to the lane (what `taskset -a` would do)
    assert la.read_task_affinities(pid, proc)[-1].cpu_list == "186"


def test_pin_all_tasks_catches_thread_born_mid_walk(tmp_path):
    proc = _make_proc(tmp_path, 20, {20: LANE, 21: "0-191"}, ppid=1)
    calls: list[tuple[int, str]] = []
    base = _fs_setter(proc, 20, calls)
    born = {"done": False}

    def setter(tid, cpus):
        base(tid, cpus)
        if not born["done"]:  # a new ROCr thread appears after the first pass read
            born["done"] = True
            _write_task(proc, 20, 22, "0-191")

    result = la.pin_all_tasks(20, LANE, proc_root=proc, setter=setter)
    assert [c[0] for c in calls] == [21, 22]
    assert result.ok and result.passes == 3  # move 21, move 22, then a clean pass


def test_pin_all_tasks_esrch_is_not_an_escape_eperm_is_a_failure(tmp_path):
    proc = _make_proc(tmp_path, 30, {30: LANE, 31: "0-191", 32: "0-191"}, ppid=1)

    def setter(tid, cpus):
        if tid == 31:
            (proc / "30" / "task" / "31" / "status").unlink()  # thread exited
            raise OSError(errno.ESRCH, "No such process")
        raise OSError(errno.EPERM, "Operation not permitted")

    result = la.pin_all_tasks(30, LANE, proc_root=proc, setter=setter, passes=1)
    assert [e["tid"] for e in result.errors] == [32]
    assert [e["tid"] for e in result.escapes_after] == [32]
    assert not result.ok


def test_pin_all_tasks_on_vanished_pid_is_not_ok(tmp_path):
    result = la.pin_all_tasks(999, LANE, proc_root=tmp_path, setter=lambda *a: None)
    assert result.tasks_seen == 0 and not result.ok


def test_enforce_refuses_a_pid_that_is_not_our_child(tmp_path):
    proc, pid = _finding(tmp_path, ppid=1)
    calls: list = []
    out: list[str] = []
    got = la.enforce_launch_affinity(
        pid, PREFIX, label="t", proc_root=proc, setter=lambda *a: calls.append(a),
        emit=out.append, launcher_pid=1000,
    )
    assert got is None and calls == [] and "not a child" in out[0]


def test_enforce_unpinned_prefix_is_not_asserted(tmp_path):
    proc, pid = _finding(tmp_path)
    assert la.enforce_launch_affinity(pid, [], label="t", proc_root=proc, launcher_pid=1000) is None


def test_enforce_pins_our_child(tmp_path):
    proc, pid = _finding(tmp_path, ppid=1000)
    calls: list = []
    out: list[str] = []
    got = la.enforce_launch_affinity(
        pid, PREFIX, label="gpu", proc_root=proc, setter=_fs_setter(proc, pid, calls),
        emit=out.append, launcher_pid=1000,
    )
    assert got is not None and got.ok and len(calls) == 1
    assert "196 tasks inside 184-191, re-pinned 1" in out[0]


def test_parent_pid_handles_comm_with_spaces(tmp_path):
    d = tmp_path / "5"
    d.mkdir()
    (d / "stat").write_text("5 (a) b (c) S 77 5 5 0\n")
    assert la.parent_pid(5, tmp_path) == 77
    assert la.parent_pid(6, tmp_path) is None


# ── layer 3: status / attestation assertion ──────────────────────────────────


def test_attestation_warning_names_the_escape_and_the_remedy(tmp_path):
    proc, pid = _finding(tmp_path)
    warnings = la.affinity_attestation_warnings("architect_critic", pid, 8083, proc_root=proc,
                                                declared=LANE)
    assert len(warnings) == 1
    w = warnings[0]
    assert "cpu_affinity_escape: 1/196" in w and f"{pid + 7}(llama-server)=0-191" in w
    assert f"launch_affinity.py --repin {pid} {LANE}" in w


def test_attestation_clean_and_undeclared_are_silent(tmp_path):
    proc = _make_proc(tmp_path, 40, {40: LANE, 41: "188"}, ppid=1)
    assert la.affinity_attestation_warnings("x", 40, 8083, proc_root=proc, declared=LANE) == []
    assert la.affinity_attestation_warnings("x", 40, 1, proc_root=proc, declared="") == []
    assert la.affinity_attestation_warnings("x", -1, 8083, proc_root=proc, declared=LANE) == []


def test_declared_cpuset_for_port_reads_the_gpu_host_lane():
    from scripts.server.stack_numa import NUMA_CONFIG

    lane_ports = [
        (inst[1], inst[0])
        for cfg in NUMA_CONFIG.values() if cfg.get("gpu_host_lane")
        for inst in cfg["instances"]
    ]
    assert lane_ports, "no gpu_host_lane role declared -- fixture assumption broken"
    for port, cpus in lane_ports:
        assert la.declared_cpuset_for_port(port) == cpus
    assert la.declared_cpuset_for_port(1) is None


def test_cli_check_exit_codes(tmp_path, capsys):
    proc, pid = _finding(tmp_path)
    assert la.main(["--check", str(pid), LANE, "--proc-root", str(proc)]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert la.main(["--check", str(pid), "0-191", "--proc-root", str(proc)]) == 0
    assert la.main(["--check", "424242", LANE, "--proc-root", str(proc)]) == 1  # no tasks = FAIL


# ── belief-kernel reading record (VB-LAUNCH-AFFINITY-WIRE) ─────────────────────


def _reading_proc(tmp_path: Path, knob: str | None = "0") -> tuple[Path, int, Path]:
    proc, pid = _finding(tmp_path)
    stat = " ".join([str(pid), "(llama-server)", "S", "1000"] + ["0"] * 17 + ["987654"])
    (proc / str(pid) / "stat").write_text(stat + "\n")
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"\x7fELF fake server")
    (proc / str(pid) / "exe").symlink_to(exe)
    env = [b"PATH=/usr/bin"] + ([f"{la.ROCR_AFFINITY_ENV_KEY}={knob}".encode()] if knob is not None else [])
    (proc / str(pid) / "environ").write_bytes(b"\0".join(env) + b"\0")
    return proc, pid, exe


def test_reading_records_pid_identity_exe_digest_knob_and_every_task(tmp_path):
    import datetime
    import hashlib

    proc, pid, exe = _reading_proc(tmp_path)
    when = datetime.datetime(2026, 10, 23, 6, 37, 31, tzinfo=datetime.timezone.utc)
    rec = la.affinity_reading(pid, LANE, moment="post_health_pin", label="arm-a", recipe="r1",
                              proc_root=proc, now=when)
    assert rec["schema"] == la.READING_SCHEMA and rec["read_at"] == "2026-10-23T06:37:31.000000Z"
    assert rec["pid_starttime_ticks"] == 987654 and rec["exe"] == str(exe)
    assert rec["exe_sha256"] == hashlib.sha256(exe.read_bytes()).hexdigest()
    assert rec["rocr_env_readable"] is True and rec["rocr_env_value"] == "0"
    assert rec["tasks_seen"] == 196 and len(rec["tasks"]) == 196
    assert rec["escapes"] == [{"tid": pid + 7, "comm": "llama-server", "cpus": "0-191"}]
    assert rec["content_sha256"] == la.reading_body_sha256(rec)
    assert la.reading_body_sha256(dict(rec, tasks_seen=195)) != rec["content_sha256"]


def test_reading_keeps_absence_absent(tmp_path):
    proc, pid, _ = _reading_proc(tmp_path, knob=None)
    rec = la.affinity_reading(pid, LANE, moment="check", label="x", proc_root=proc)
    assert rec["rocr_env_readable"] is True and rec["rocr_env_value"] is None
    gone = la.affinity_reading(424242, LANE, moment="check", label="x", proc_root=proc)
    assert gone["tasks_seen"] == 0 and gone["exe"] is None and gone["exe_sha256"] is None
    assert gone["pid_starttime_ticks"] is None and gone["rocr_env_readable"] is False
    with pytest.raises(la.LaunchAffinityError):
        la.affinity_reading(pid, LANE, moment="whenever", label="x", proc_root=proc)


def test_cli_record_appends_one_reading_per_call(tmp_path, capsys):
    import json

    proc, pid, _ = _reading_proc(tmp_path)
    out = tmp_path / "affinity_readings.jsonl"
    assert la.main(["--check", str(pid), LANE, "--record", str(out), "--label", "probe",
                    "--proc-root", str(proc)]) == 1
    assert la.main(["--check", str(pid), "0-191", "--record", str(out), "--moment", "pre_stop",
                    "--proc-root", str(proc)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["moment"] for r in rows] == ["check", "pre_stop"]
    assert [len(r["escapes"]) for r in rows] == [1, 0]
    assert all(r["content_sha256"] == la.reading_body_sha256(r) for r in rows)
    assert "reading_sha256" in capsys.readouterr().out


def test_serving_gate_server_appends_readings_and_never_fails_on_a_reading_error(tmp_path):
    import json
    import os

    from scripts.benchmark import serving_gate as sg

    srv = sg.Server.__new__(sg.Server)       # no launch: read our OWN pid, read-only
    srv.pid, srv.label, srv.out = os.getpid(), "arm-a", tmp_path
    srv.argv = ["taskset", "-c", la.format_cpu_list(os.sched_getaffinity(0)), "llama-server"]
    srv._record_affinity("post_health_pin")
    rec = json.loads((tmp_path / "affinity_readings.jsonl").read_text())
    assert rec["label"] == "arm-a" and rec["moment"] == "post_health_pin" and rec["escapes"] == []
    assert rec["exe_sha256"] and srv.affinity_readings == [rec["content_sha256"]]
    srv._record_affinity("not-a-moment")
    assert srv.affinity_reading_errors and "not-a-moment" in srv.affinity_reading_errors[0]


# ── generate_attestation: the lenient match is gone ──────────────────────────


def test_generate_attestation_flags_a_single_escaping_task():
    from scripts.attest import generate_attestation as attest

    row = {
        "pid": 1, "kind": "llama_server", "port": 8083, "registry_matches": [{}], "args": {},
        "cpus_allowed_list": LANE, "task_cpu_masks": {LANE: 195, "0-191": 1},
    }
    rows = attest.build_serving_config(
        [row], numa_ports={8083: {"role": "architect_critic", "cpu_list": LANE, "threads": 8}}
    )
    assert rows[0]["numa_match"] is False
    assert rows[0]["task_cpu_escape"] == ["0-191"]
    summary = attest.summarize([], serving_config=rows)
    assert any("cpu_affinity_escape:0-191" in i for issue in summary["issues"] for i in issue["issues"])

    clean = dict(row, task_cpu_masks={LANE: 190, "186": 6})
    rows = attest.build_serving_config(
        [clean], numa_ports={8083: {"role": "architect_critic", "cpu_list": LANE, "threads": 8}}
    )
    assert rows[0]["numa_match"] is True and rows[0]["task_cpu_escape"] == []


# ── orchestrator_stack: every launch branch carries both layers ──────────────


@pytest.fixture
def pinned_harness(monkeypatch, tmp_path):
    from scripts.server import orchestrator_stack as osk

    captured: list[dict[str, str]] = []
    gates: list[tuple[int, list[str], str]] = []

    def fake_popen(argv, **kwargs):
        captured.append({"argv": list(argv), "env": dict(kwargs["env"])})
        return types.SimpleNamespace(pid=4242)

    def fake_enforce(pid, prefix, *, label, **kw):
        gates.append((pid, list(prefix), label))
        return types.SimpleNamespace(ok=True)

    monkeypatch.setenv("HSA_OVERRIDE_CPU_AFFINITY_DEBUG", "1")
    monkeypatch.setattr(osk.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(osk, "enforce_launch_affinity", fake_enforce)
    monkeypatch.setattr(osk, "build_server_command", lambda *a, **k: ["llama-server", "--port", "1"])
    monkeypatch.setattr(osk, "_bench_guarded_numa_prefix", lambda *a, **k: list(PREFIX))
    monkeypatch.setattr(osk, "_embedding_spawn_prefix", lambda *a, **k: list(PREFIX))
    monkeypatch.setattr(osk, "_write_llama_marker", lambda *a, **k: None)
    monkeypatch.setattr(osk, "wait_for_health", lambda port, timeout=0, path="/health": True)
    monkeypatch.setattr(osk, "pre_evict_nodes", lambda gib, **kw: (True, "faked"))
    monkeypatch.setattr(osk, "placement_summary", lambda pid: "faked")
    monkeypatch.setattr(osk, "_renice_all_threads", lambda *a, **k: None)
    monkeypatch.setattr(osk, "LOG_DIR", tmp_path)
    monkeypatch.setattr(osk, "_stack_prior_runtime_overrides", lambda role: ("/k/bin/llama-server", []))
    monkeypatch.setattr(osk, "_runtime_requirements_for_role", lambda registry, role: ("/k/bin", []))
    monkeypatch.setattr(osk, "_stack_prior_launch", lambda role: ({"model_path": "/m/fake.gguf"}, {}))
    model = types.SimpleNamespace(name="fake-model", full_path="/m/fake.gguf")
    registry = types.SimpleNamespace(get_role=lambda r: types.SimpleNamespace(model=model, name=r))
    return types.SimpleNamespace(osk=osk, captured=captured, gates=gates, registry=registry)


BRANCHES = {
    "registry": dict(roles=["architect_critic"]),
    "vision": dict(roles=["worker_vision"], vision_mode=True),
    "gpu_shadow_lane": dict(roles=["frontdoor"], gpu_shadow_lane_mode=True),
    "worker_pool": dict(roles=["worker_general"], worker_pool_mode=True, worker_type="__first__"),
    "embedding": dict(roles=["embedder"], embedding_mode=True),
    "eval_batch": dict(roles=["frontdoor"], eval_batch_frontdoor_mode=True),
}


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_every_llama_server_branch_pins_env_and_gates(pinned_harness, branch):
    from scripts.server.stack_manifest import WORKER_POOL_MODELS

    kw = dict(BRANCHES[branch])
    roles = kw.pop("roles")
    if kw.get("worker_type") == "__first__":
        kw["worker_type"] = sorted(WORKER_POOL_MODELS)[0]
    info = pinned_harness.osk.start_server(8083, roles, pinned_harness.registry, **kw)
    assert info is not None
    launch = pinned_harness.captured[-1]
    assert launch["argv"][: len(PREFIX)] == PREFIX
    assert launch["env"]["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] == "0"
    assert pinned_harness.gates and pinned_harness.gates[-1][:2] == (4242, PREFIX)


def test_failed_affinity_gate_fails_the_launch_and_kills_our_pid(pinned_harness, monkeypatch):
    osk = pinned_harness.osk
    killed: list[int] = []
    monkeypatch.setattr(osk, "kill_process", lambda pid, timeout=5: killed.append(pid) or True)
    monkeypatch.setattr(
        osk, "enforce_launch_affinity", lambda *a, **k: types.SimpleNamespace(ok=False)
    )
    assert osk.start_server(8083, ["architect_critic"], pinned_harness.registry) is None
    assert killed == [4242]


def test_unpinned_launch_passes_the_gate_untouched(pinned_harness, monkeypatch):
    osk = pinned_harness.osk
    monkeypatch.setattr(osk, "_bench_guarded_numa_prefix", lambda *a, **k: [])
    monkeypatch.setattr(osk, "enforce_launch_affinity", la.enforce_launch_affinity)
    assert osk.start_server(8083, ["architect_critic"], pinned_harness.registry) is not None
    assert pinned_harness.captured[-1]["env"]["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] == "1"


def test_aux_launch_source_applies_env_and_gate():
    """start_aux_service has heavier prerequisites; assert its source carries both layers."""
    import inspect

    from scripts.server import orchestrator_stack as osk

    src = inspect.getsource(osk.start_aux_service)
    assert "apply_pinned_launch_env(env, spawn_prefix)" in src
    assert "_post_health_affinity_gate(" in src
    assert src.index("apply_pinned_launch_env") < src.index("subprocess.Popen")


def test_d1_refuse_has_no_warn_mode_or_bypass(monkeypatch):
    """Operator decision D1 (2026-10-22): refuse, with no flag or env that downgrades to warn."""
    import inspect

    from scripts.server import orchestrator_stack as osk

    src = inspect.getsource(osk._post_health_affinity_gate)
    assert "kill_process(pid)" in src and "return False" in src
    assert "environ" not in src and "getenv" not in src
    assert "environ" not in inspect.getsource(la.enforce_launch_affinity)
