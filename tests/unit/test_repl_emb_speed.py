"""REPL-EMB-2.7 speed-block harness: fake encoders, a fake /proc and a fake clock (no sleeps, no
models, no servers, no sockets). Everything the real run depends on is exercised here except the
numbers themselves."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
import repl_emb_arms as arms  # noqa: E402
import repl_emb_speed as sp  # noqa: E402

cs = arms.load_context_search()
ME = os.getpid()


# ───────────────────────────────────────────────────────────────────────────── fakes


def stat_line(pid, comm, state="S", ppid=1, utime=0, stime=0, processor=0):
    rest = ["0"] * 50
    rest[0], rest[1] = state, str(ppid)
    rest[11], rest[12], rest[36] = str(utime), str(stime), str(processor)
    return f"{pid} ({comm}) " + " ".join(rest) + "\n"


class FakeProc(sp.Proc):
    """Dict-backed /proc and /sys: the REAL parsers run on text this class synthesises."""

    def __init__(self, clk_tck=100):
        super().__init__(proc_root="/nonexistent", sys_cpu_root="/nonexistent", clk_tck=clk_tck)
        self.procs: dict[int, dict] = {}
        self.cpu: dict[int, list[int]] = {}
        self.load = (1.0, 1.0, 1.0)
        self.freq: dict[int, int] = {}
        self.siblings: dict[int, list[int]] = {}
        self.clock: dict[int, float] = {}

    def add(self, pid, comm="python", state="S", ppid=1, utime=0, stime=0, processor=0,
            allowed="0-191", cmdline=(), tasks=None, rss_kb=1000, hwm_kb=2000):  # fmt: skip
        self.procs[pid] = dict(comm=comm, state=state, ppid=ppid, utime=utime, stime=stime,
                               processor=processor, allowed=allowed, cmdline=list(cmdline),
                               tasks=tasks or {pid: (state, processor)}, rss=rss_kb, hwm=hwm_kb)  # fmt: skip
        return self.procs[pid]

    def set_cpus(self, cpus):
        for c in cpus:
            self.cpu[c] = [0, 0]

    def _p(self, pid):
        if pid not in self.procs:
            raise FileNotFoundError(pid)
        return self.procs[pid]

    def _read(self, rel):
        if rel == "loadavg":
            return "%.2f %.2f %.2f 1/100 12345\n" % self.load
        if rel == "stat":
            out = ["cpu  1 2 3 4 5 6 7 8 0 0"]
            out += [f"cpu{c} {b} 0 0 {t - b} 0 0 0 0 0 0" for c, (b, t) in sorted(self.cpu.items())]
            return "\n".join(out) + "\n"
        parts = rel.split("/")
        p = self._p(int(parts[0]))
        if parts[1:] == ["stat"]:
            return stat_line(
                parts[0], p["comm"], p["state"], p["ppid"], p["utime"], p["stime"], p["processor"]
            )
        if len(parts) == 4 and parts[1] == "task" and parts[3] == "stat":
            st, proc = p["tasks"][int(parts[2])]
            return stat_line(parts[2], p["comm"], st, p["ppid"], 0, 0, proc)
        if parts[1:] == ["status"]:
            return (f"Name:\t{p['comm']}\nCpus_allowed_list:\t{p['allowed']}\n"
                    f"VmHWM:\t{p['hwm']} kB\nVmRSS:\t{p['rss']} kB\n")  # fmt: skip
        raise FileNotFoundError(rel)

    def _read_bytes(self, rel):
        pid = int(rel.split("/")[0])
        return b"\0".join(a.encode() for a in self._p(pid)["cmdline"])

    def _listdir(self, rel):
        if rel == "":
            return [str(p) for p in self.procs] + ["self", "loadavg"]
        pid = int(rel.split("/")[0])
        return [str(t) for t in self._p(pid)["tasks"]]

    def _read_sys(self, rel):
        c = int(rel.split("/")[0][3:])
        if rel.endswith("scaling_cur_freq") and c in self.freq:
            return f"{self.freq[c]}\n"
        if rel.endswith("thread_siblings_list") and c in self.siblings:
            return sp.format_cpu_list(self.siblings[c]) + "\n"
        raise FileNotFoundError(rel)

    def cpu_clock_s(self, pid):
        return self.clock.get(pid)


class World:
    """Fake clock + fake CPU accounting. ``step()`` is one encode call taking ``step_s`` seconds
    during which this process burns ``own_ticks`` and strangers burn ``foreign_ticks`` on the cpus."""

    def __init__(self, proc, cpus, step_s=1.0, own_ticks=100, foreign_ticks=0):
        self.proc, self.cpus = proc, list(cpus)
        self.step_s, self.own_ticks, self.foreign_ticks = step_s, own_ticks, foreign_ticks
        self.t = 1000.0

    def perf(self):
        return self.t

    def time(self):
        return 1.7e9 + self.t

    def step(self):
        self.t += self.step_s
        per = (self.own_ticks + self.foreign_ticks) // len(self.cpus)
        for c in self.cpus:
            self.proc.cpu[c][0] += per
            self.proc.cpu[c][1] += int(self.step_s * self.proc.clk_tck)
        self.proc.procs[ME]["utime"] += self.own_ticks


class TickingRuntime(sp.FakeRuntime):
    def __init__(self, entry, world, **kw):
        super().__init__(entry, **kw)
        self.world = world

    def encode_docs(self, texts, batch_size):
        self.world.step()
        return super().encode_docs(texts, batch_size)

    def encode_query(self, query):
        self.world.step()
        return super().encode_query(query)


def roster():
    return sp.speed_roster(cs, "st-rt", "llama-rt", "m2v-rt")


CORPUS = [f"chunk {i} " + " ".join(f"w{(i * 3 + j) % 17}" for j in range(30)) for i in range(64)]
QUERIES = [f"query number {i} about w{i % 7}" for i in range(12)]


def make_runner(
    tmp_path, arm_ids, *, proc=None, world=None, runtimes=None, dry=True, extra=None, **cfg_over
):
    r = roster()
    fp = proc or FakeProc()
    cpus = cfg_over.pop("cpu_list", [0, 1, 2, 3])
    fp.add(ME, ppid=500)
    fp.set_cpus(cpus)
    cfg = sp.RunConfig(
        out_dir=tmp_path, cpu_list=cpus, claim_tag="TAG", dry_run=dry, run_id="t1", **cfg_over
    )
    rts = runtimes or {
        a: (TickingRuntime(r[a], world) if world else sp.FakeRuntime(r[a])) for a in arm_ids
    }
    kw = dict(now=world.time, perf=world.perf) if world else {}
    runner = sp.SpeedRunner(
        cfg, r, rts, list(CORPUS), list(QUERIES), cs=cs, proc=fp, affinity=sp.DryAffinity(cpus),
        inputs_meta={"corpus_sha256": "c", "queries_sha256": "q"}, **(extra or {}), **kw,
    )  # fmt: skip
    return runner, fp


# ───────────────────────────────────────────────────────────────────────────── pure arithmetic


def test_percentile_arithmetic_matches_numpy_linear():
    xs = list(range(1, 201))
    assert sp.percentile(xs, 50) == pytest.approx(100.5)
    assert sp.percentile(xs, 95) == pytest.approx(190.05)
    assert sp.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert sp.percentile([7.0], 95) == 7.0
    rng = np.random.default_rng(3)
    for n in (2, 3, 17, 600):
        v = rng.exponential(size=n)
        for q in (50, 95, 99):
            assert sp.percentile(v, q) == pytest.approx(float(np.percentile(v, q)))
    with pytest.raises(ValueError):
        sp.percentile([], 50)
    st = sp.latency_stats([5.0, 1.0, 3.0])
    assert (st["n"], st["p50_ms"], st["max_ms"]) == (3, 3.0, 5.0)
    assert sp.latency_stats([])["p50_ms"] is None


def test_aba_order_is_forward_then_reverse_and_mirrored():
    assert sp.aba_order(["A", "B", "C"], 2) == [
        (0, 0, "A"), (0, 1, "B"), (0, 2, "C"), (1, 3, "C"), (1, 4, "B"), (1, 5, "A"),
    ]  # fmt: skip
    for n in (1, 2, 5, 9):
        arms_ = [f"a{i}" for i in range(n)]
        order = sp.aba_order(arms_, 2)
        pos: dict[str, list[int]] = {}
        for _, p, a in order:
            pos.setdefault(a, []).append(p)
        # every arm twice, at mirrored positions: the position sum is constant, so linear drift cancels
        assert {a: len(v) for a, v in pos.items()} == {a: 2 for a in arms_}
        assert {sum(v) for v in pos.values()} == {2 * n - 1}
        assert [a for _, _, a in order] == arms_ + arms_[::-1]
    assert [a for _, _, a in sp.aba_order(["A", "B"], 4)] == [
        "A",
        "B",
        "B",
        "A",
        "A",
        "B",
        "B",
        "A",
    ]


def test_cpu_list_helpers_and_kv_parsing():
    assert sp.parse_cpu_list("24-27,30,8-9") == [8, 9, 24, 25, 26, 27, 30]
    assert sp.format_cpu_list([8, 9, 24, 25, 26, 27, 30]) == "8-9,24-27,30"
    with pytest.raises(ValueError):
        sp.parse_cpu_list("5-3")
    assert sp.parse_kv(["a=http://x:1", "b=2"], "t") == {"a": "http://x:1", "b": "2"}
    with pytest.raises(sp.Refusal):
        sp.parse_kv(["nokey"], "t")
    assert sp.parse_server_threads(["llama-server", "-m", "x.gguf", "-t", "16"]) == 16
    assert sp.parse_server_threads(["llama-server", "--threads=8"]) == 8
    assert sp.parse_server_threads(["llama-server"]) is None
    assert sp.parse_server_model(["llama-server", "-m", "/m/x.gguf"]) == "/m/x.gguf"


def test_pick_cpus_takes_physical_cores_before_smt_siblings():
    fp = FakeProc()
    fp.siblings = {c: [c, c + 4] for c in range(4)} | {c: [c - 4, c] for c in range(4, 8)}
    cl = list(range(8))
    assert sp.pick_cpus(cl, 1, fp) == [0]
    assert sp.pick_cpus(cl, 4, fp) == [0, 1, 2, 3]
    assert sp.pick_cpus(cl, 6, fp) == [0, 1, 2, 3, 4, 5]  # 4 cores, then 2 siblings
    assert sp.pick_cpus(cl, 99, fp) == cl
    assert sp.pick_cpus([5, 6, 7], 2, FakeProc()) == [5, 6]  # no topology: first n


def test_disk_usage_follows_symlinks_and_dedupes_by_inode(tmp_path):
    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    blobs.mkdir()
    snap.mkdir()
    (blobs / "w").write_bytes(b"x" * 1000)
    (blobs / "c").write_bytes(b"y" * 10)
    (snap / "model.safetensors").symlink_to(blobs / "w")
    (snap / "again.safetensors").symlink_to(blobs / "w")  # same inode: counted once
    (snap / "config.json").symlink_to(blobs / "c")
    du = sp.disk_usage(snap)
    assert (du["bytes"], du["weights_bytes"], du["files"]) == (1010, 1000, 2)
    assert sp.disk_usage(blobs / "w")["bytes"] == 1000


# ───────────────────────────────────────────────────────────────────────────── budget arithmetic


def test_implied_budget_flag_arithmetic_against_the_shipped_constants():
    assert cs.index_build_timeout_s(4096) == 60.0  # clamp(5 + 4096/25, 10, 60)
    # the shipped docstring's own figure: at the ASSUMED rate a full bundle takes ~164 s = 3 searches
    r = sp.index_budget_report(25.0, 4096, cs)
    assert r["implied_build_s"] == pytest.approx(163.84)
    assert (r["budget_s"], r["exposes_index_timeout"], r["searches_to_complete"]) == (60.0, True, 3)
    assert r["assumed_build_s"] == pytest.approx(163.84) and r[
        "measured_over_assumed"
    ] == pytest.approx(1.0)
    fits = sp.index_budget_report(4096 / 59.9, 4096, cs)  # builds in 59.9 s
    assert fits["implied_build_s"] == pytest.approx(59.9) and fits["exposes_index_timeout"] is False
    exact = sp.index_budget_report(4096 / 60.0, 4096, cs)  # exactly the budget does NOT expose
    assert exact["exposes_index_timeout"] is False and exact["searches_to_complete"] == 1
    over = sp.index_budget_report(4096 / 60.1, 4096, cs)
    assert over["exposes_index_timeout"] is True and over["searches_to_complete"] == 2
    zero = sp.index_budget_report(0.0, 4096, cs)
    assert math.isinf(zero["implied_build_s"]) and zero["exposes_index_timeout"] is True
    fake = SimpleNamespace(
        INDEX_ASSUMED_TEXTS_PER_S=50.0, INDEX_TIMEOUT_BASE_S=2.0, INDEX_TIMEOUT_MIN_S=5.0,
        INDEX_TIMEOUT_MAX_S=30.0, MAX_DENSE_CHUNKS=1000,
        index_build_timeout_s=lambda n: min(30.0, max(5.0, 2.0 + n / 50.0)),
    )  # fmt: skip
    rr = sp.index_budget_report(10.0, 1000, fake)  # 100 s implied vs min(30, 2+20)=22 s budget
    assert (rr["budget_s"], rr["exposes_index_timeout"], rr["searches_to_complete"]) == (
        22.0,
        True,
        5,
    )


# ───────────────────────────────────────────────────────────────────────────── cpu seconds from /proc


def test_stat_parser_survives_awkward_comm_names():
    st = sp.Proc.parse_stat(stat_line(7, "we ird) (name", "R", 3, 250, 50, 9))
    assert (st["comm"], st["state"], st["ppid"], st["utime"], st["stime"], st["processor"]) == (
        "we ird) (name", "R", 3, 250, 50, 9,
    )  # fmt: skip


def test_cpu_seconds_accounting_from_a_fake_proc_reader():
    fp = FakeProc()
    fp.add(100, utime=1000, stime=200)  # client
    fp.add(200, comm="llama-server", utime=5000, stime=1000)  # caller-owned server
    fp.clock = {100: 12.0, 200: 60.0}
    meter = sp.CpuMeter(fp, {"client": 100, "server": 200})
    meter.start()
    fp.procs[100].update(utime=1000 + 300, stime=200 + 50)  # +3.5 s
    fp.procs[200].update(utime=5000 + 400, stime=1000 + 100)  # +5.0 s
    fp.clock = {100: 15.5, 200: 65.0}
    t = meter.total()
    assert t["cpu_s_stat"] == pytest.approx(8.5) and t["cpu_s"] == pytest.approx(8.5)
    assert t["cpu_source"] == "stat"
    assert t["by_role"]["client"]["cpu_s_stat"] == pytest.approx(3.5)
    assert t["by_role"]["server"]["cpu_s_stat"] == pytest.approx(5.0)
    assert t["cpu_s_clock"] == pytest.approx(8.5)
    # a window with < 1 cpu-second is below clock-tick resolution: the ns clock replaces it, labelled
    snap = meter.read()
    fp.procs[100]["utime"] += 3
    fp.clock[100] += 0.0312
    short = meter.since(snap)
    assert short["cpu_source"] == "clock" and short["cpu_s"] == pytest.approx(0.0312)
    assert short["cpu_s_stat"] == pytest.approx(0.03)
    fp.clock = {}
    assert meter.since(snap)["cpu_source"] == "stat"  # no ns clock readable: stay on /proc ticks


def test_cpu_s_per_query_row_counts_client_plus_server_over_the_query_window(tmp_path):
    fp = FakeProc()
    fp.add(900, comm="llama-server", utime=0)
    world = World(fp, [0, 1, 2, 3], own_ticks=100)
    r = roster()["eg2-gguf-q8_0"]
    rt = TickingRuntime(r, world)
    rt.server_pid = 900  # the caller-owned server's pid

    class Both(World):
        def step(self):
            super().step()
            fp.procs[900]["utime"] += 50  # the server burns 0.5 s per query as well

    both = Both(fp, [0, 1, 2, 3], own_ticks=100)
    rt.world = both
    runner, _ = make_runner(tmp_path, ["eg2-gguf-q8_0"], proc=fp, world=both, runtimes={"eg2-gguf-q8_0": rt},
                            only=("query",), repeats=3, cpu_list=[0, 1, 2, 3])  # fmt: skip
    runner.run()
    rows = [
        x
        for x in sp.load_rows(tmp_path / "speed.jsonl")
        if x["leg"] == "q_served" and x["phase"] == "main"
    ]
    assert rows and all(x["status"] == "ok" for x in rows)
    x = rows[0]
    assert x["n_queries"] == 36  # 12 distinct queries x 3 repeats
    # per query: 1.0 s client + 0.5 s server
    assert x["cpu_s_window"] == pytest.approx(36 * 1.5)
    assert x["cpu_s_per_query"] == pytest.approx(1.5)
    assert set(x["cpu"]["by_role"]) == {"client", "server"}
    assert [r_["repeat"] for r_ in x["repeats"]] == [0, 1, 2]


# ───────────────────────────────────────────────────────────────────────────── safety


@pytest.mark.parametrize("port", sorted(sp.PRODUCTION_PORTS))
def test_every_production_port_is_refused(port):
    with pytest.raises(sp.Refusal, match="production/stack port"):
        sp.check_arm_url("bge-large-gguf-f16", f"http://127.0.0.1:{port}")


def test_stack_ports_remote_hosts_and_portless_urls_are_refused_but_a_free_port_is_not():
    assert {
        8000,
        8083,
        8090,
        8091,
        8092,
        8093,
        8094,
        8095,
        9000,
        9002,
        8088,
        8089,
    } <= sp.REFUSED_PORTS
    for bad in ("http://127.0.0.1:8096", "http://localhost:8190", "http://10.0.0.5:18741",
                "http://127.0.0.1", "ftp://127.0.0.1:18741", "127.0.0.1:18741"):  # fmt: skip
        with pytest.raises(sp.Refusal):
            sp.check_arm_url("a", bad)
    ok = sp.check_arm_url("a", "http://127.0.0.1:18741/")
    assert (ok["host"], ok["port"], ok["base_url"]) == (
        "127.0.0.1",
        18741,
        "http://127.0.0.1:18741",
    )
    assert sp.check_arm_url("a", "http://[::1]:18742")["base_url"] == "http://[::1]:18742"


def test_cli_refuses_a_production_port_before_doing_anything(tmp_path, capsys):
    out = tmp_path / "o"
    rc = sp.main(
        [
            "run",
            "--dry-run",
            "--out",
            str(out),
            "--arm-url",
            "bge-large-gguf-f16=http://127.0.0.1:8090",
        ]
    )
    assert rc == 64
    assert "REFUSING" in capsys.readouterr().err
    assert not (out / "speed.jsonl").exists() and (out / "refusal.json").exists()


def test_cli_refuses_to_start_without_a_claim_tag_or_without_claim_verification(tmp_path, capsys):
    out = tmp_path / "o"
    assert sp.main(["run", "--out", str(out), "--cpu-list", "0-3"]) == 64
    assert "--claim-tag is required" in capsys.readouterr().err
    assert not (out / "speed.jsonl").exists()
    assert sp.main(["run", "--out", str(out), "--cpu-list", "0-3", "--claim-tag", " "]) == 64
    assert (
        sp.main(
            [
                "run",
                "--out",
                str(out),
                "--cpu-list",
                "0-3",
                "--claim-tag",
                "T",
                "--no-require-claim-env",
            ]
        )
        == 64
    )
    assert "only allowed with --dry-run" in capsys.readouterr().err
    assert sp.main(["run", "--out", str(out), "--claim-tag", "T"]) == 64  # no --cpu-list
    assert sp.main(["run", "--out", str(out), "--claim-tag", "T", "--cpu-list", "9-3"]) == 64
    assert "--cpu-list" in capsys.readouterr().err
    # in-process arms need a model path, served arms a url AND a pid: refused before any load
    assert sp.main(["run", "--out", str(out), "--claim-tag", "T", "--cpu-list", "0-3"]) == 64
    assert not (out / "speed.jsonl").exists()


def _claim_world():
    """A fake host: this process (pid ME) under a region-lock CLI (pid 500), claim = cpus 0-3."""
    fp = FakeProc()
    fp.add(ME, ppid=500, allowed="0-3")
    fp.add(500, comm="python3", ppid=1, allowed="0-3")
    fp.add(1, comm="systemd", allowed="0-191")
    holders = [{"region": "q0", "pid": 500, "request_tag": "TAG", "role": "bench-cpu"}]
    return fp, (lambda: holders)


def _verify(fp, holders_fn, **kw):
    base = dict(cpu_list=[0, 1, 2, 3], claim_tag="TAG", proc=fp, affinity_now={0, 1, 2, 3},
                exempt_pids=(), holders_fn=holders_fn, scan_samples=1, sleep=lambda s: None)  # fmt: skip
    base.update(kw)
    return sp.verify_claim_env(**base)


def test_claim_env_passes_on_a_clean_claim():
    fp, holders = _claim_world()
    fp.add(40, comm="python", state="R", processor=100, allowed="0-191")  # busy, but on cpu 100
    fp.add(41, comm="python", state="S", processor=2, allowed="0-191")  # unpinned idle overlap
    rep = _verify(fp, holders)
    assert rep["ok"] and rep["n_offenders"] == 0 and rep["warnings"] == []
    assert {c["name"] for c in rep["checks"]} >= {
        "claim_tag",
        "affinity_equals_cpu_list",
        "region_lock_holder_is_ancestor",
    }


def test_claim_env_reports_offenders_without_signalling_anything():
    fp, holders = _claim_world()
    fp.add(
        40, comm="python", state="R", processor=2, allowed="0-191", cmdline=["python", "bench.py"]
    )
    fp.add(
        41,
        comm="llama-server",
        state="S",
        processor=2,
        allowed="0-3",
        cmdline=["llama-server", "-m", "x"],
    )
    fp.add(
        42,
        comm="python",
        state="S",
        processor=2,
        allowed="0-191",
        tasks={42: ("S", 2), 43: ("R", 1)},
    )
    fp.add(43, comm="bash", state="R", processor=1, allowed="0-191")  # not llama/python: ignored
    fp.add(44, comm="llama-server", state="R", processor=1, allowed="0-3")  # exempt (caller-owned)
    rep = _verify(fp, holders, exempt_pids={44})
    assert not rep["ok"]
    by_pid = {o["pid"]: o for o in rep["offenders"]}
    assert set(by_pid) == {40, 41, 42}
    assert by_pid[40]["reasons"] == ["running_on_cpu2"]
    assert by_pid[41]["reasons"] == ["pinned_inside_claim"]
    assert "running_on_cpu1" in by_pid[42]["reasons"]  # a non-main THREAD running on a claimed cpu
    failed = [c for c in rep["checks"] if not c["ok"]]
    assert [c["name"] for c in failed] == ["no_foreign_llama_python_on_cpus"]


def test_claim_env_fails_on_wrong_affinity_missing_tag_missing_holder_or_stray_server():
    fp, holders = _claim_world()
    assert not _verify(fp, holders, affinity_now=set(range(192)))["ok"]
    assert not _verify(fp, holders, claim_tag="  ")["ok"]
    assert not _verify(fp, lambda: [])["ok"]  # nobody in the ancestry holds a region-lock claim
    assert _verify(fp, lambda: [], require_region_lock_holder=False)[
        "ok"
    ]  # downgraded, still reported
    assert not _verify(fp, None)["ok"]
    fp.add(900, comm="llama-server", allowed="0-191")
    bad = _verify(fp, holders, server_pids={"bge-large-gguf-f16": 900}, exempt_pids={900})
    assert not bad["ok"] and any(
        c["name"] == "server_bge-large-gguf-f16_inside_claim" and not c["ok"] for c in bad["checks"]
    )
    fp.procs[900]["allowed"] = "0-3"
    assert _verify(fp, holders, server_pids={"bge-large-gguf-f16": 900}, exempt_pids={900})["ok"]
    boom = _verify(fp, lambda: (_ for _ in ()).throw(RuntimeError("no orchestrator venv")))
    assert (
        not boom["ok"]
        and "unavailable"
        in [c for c in boom["checks"] if c["name"].startswith("region")][0]["detail"]
    )
    mism = _verify(
        fp, lambda: [{"region": "q0", "pid": 500, "request_tag": "OTHER"}], exempt_pids={900}
    )
    assert mism["ok"] and mism["warnings"]


def test_region_lock_holders_flattens_the_status_json():
    payload = [
        {
            "region": "q0",
            "global_held": True,
            "holders": [{"pid": 11, "request_tag": "a", "role": "build"}],
        },
        {"region": "q1", "global_held": False, "holders": []},
    ]
    cp = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
    seen = {}

    def runner(cmd, **kw):
        seen["cmd"] = cmd
        return cp

    h = sp.region_lock_holders("/x/region-lock", runner=runner)
    assert h == [{"region": "q0", "pid": 11, "request_tag": "a", "role": "build"}]
    assert seen["cmd"] == ["/x/region-lock", "status", "--json"]  # read-only subcommand
    with pytest.raises(RuntimeError):
        sp.region_lock_holders(
            "/x", runner=lambda *a, **k: SimpleNamespace(returncode=3, stdout="", stderr="no")
        )


def test_the_module_never_signals_or_kills_anything():
    src = Path(sp.__file__).read_text()
    for forbidden in (
        "os.kill",
        "pkill",
        "pgrep",
        "killpg",
        "signal.",
        ".terminate(",
        ".kill(",
        "Popen(",
    ):
        assert forbidden not in src, forbidden


def test_affinity_is_applied_to_every_thread_and_read_back():
    state = {1: {0, 1, 2, 3, 4, 5}, 2: {0, 1, 2, 3, 4, 5}, 3: {0, 1, 2, 3, 4, 5}}
    calls = []

    def set_fn(tid, cpus):
        calls.append(tid)
        if tid != 3:  # thread 3 "ignores" the request (e.g. a GOMP team re-pinning itself)
            state[tid] = set(cpus)

    aff = sp.OsAffinity(
        set_fn=set_fn, get_fn=lambda tid: state.get(tid, state[1]), tids_fn=lambda: [1, 2, 3]
    )
    rb = aff.apply([2, 3])
    assert sorted(calls) == [1, 2, 3]  # every tid, not just the caller
    assert rb["applied"] == [2, 3] and rb["violations"] == [3] and rb["uniform"] is False
    state[3] = {2, 3}
    assert aff.readback({2, 3})["violations"] == []
    dry = sp.DryAffinity([0, 1, 2, 3])
    assert dry.apply([1])["applied"] == [1] and dry.current() == {1}


# ───────────────────────────────────────────────────────────────────────────── contention watch


def test_contention_watch_flags_foreign_busy_time_on_the_leg_cpus_not_its_own():
    fp = FakeProc()
    fp.add(ME)
    fp.set_cpus([0, 1, 2, 3])
    w = World(fp, [0, 1, 2, 3], own_ticks=100)
    watch = sp.ContentionWatch(fp, [0, 1, 2, 3], [ME], own_threads=4, max_foreign_busy_frac=0.10,
                               max_loadavg_excess=6.0, interval_s=0.5, min_window_s=0.5, perf=w.perf)  # fmt: skip
    watch.start()
    for _ in range(3):  # only our own time: never contended
        w.step()
        watch.check()
    fin = watch.finish()
    assert not fin["contended"] and fin["foreign_busy_frac"] == pytest.approx(0.0, abs=1e-9)
    w2 = World(fp, [0, 1, 2, 3], own_ticks=100, foreign_ticks=200)  # 2 cpu-equivalents of strangers
    watch2 = sp.ContentionWatch(fp, [0, 1, 2, 3], [ME], own_threads=4, max_foreign_busy_frac=0.10,
                                max_loadavg_excess=6.0, interval_s=0.5, min_window_s=0.5, perf=w2.perf)  # fmt: skip
    watch2.start()
    w2.step()
    with pytest.raises(sp.LegContended) as ei:
        watch2.check()
    assert "foreign_busy_interval" in ei.value.reasons
    assert watch2.max_interval_frac == pytest.approx(0.5)  # 2 of 4 cpus


def test_contention_watch_loadavg_excess_is_the_rise_beyond_the_legs_own_threads():
    fp = FakeProc()
    fp.add(ME)
    fp.set_cpus([0, 1])
    fp.load = (10.0, 5.0, 3.0)
    w = World(fp, [0, 1], own_ticks=100)
    watch = sp.ContentionWatch(fp, [0, 1], [ME], own_threads=2, max_foreign_busy_frac=1.0,
                               max_loadavg_excess=6.0, interval_s=0.5, min_window_s=0.5, perf=w.perf)  # fmt: skip
    watch.start()
    fp.load = (17.5, 5.0, 3.0)  # +7.5, of which 2 are this leg's own threads: excess 5.5 <= 6
    w.step()
    watch.check()
    assert watch.load_excess == pytest.approx(5.5)
    fp.load = (18.5, 5.0, 3.0)  # excess 6.5 > 6
    w.step()
    with pytest.raises(sp.LegContended) as ei:
        watch.check()
    assert ei.value.reasons == ["loadavg_excess"]


# ───────────────────────────────────────────────────────────────────────────── runner on a fake world


def test_ok_row_carries_identity_window_loadavg_cpus_frequency_and_claim_tag(tmp_path):
    fp = FakeProc()
    fp.freq = {0: 3000000, 1: 3100000, 2: 3200000, 3: 3300000}
    fp.load = (4.0, 3.0, 2.0)
    world = World(fp, [0, 1, 2, 3], own_ticks=100)
    runner, _ = make_runner(tmp_path, ["bge-large-st-fp32"], proc=fp, world=world, only=("throughput",),
                            batches=(8,), threads=(4,))  # fmt: skip
    runner.run()
    rows = [x for x in sp.load_rows(tmp_path / "speed.jsonl") if x["leg"] == "tp_b8_t4"]
    assert [x["rep"] for x in sorted(rows, key=lambda r: r["order_pos"])] == [0, 1]
    x = rows[0]
    assert x["status"] == "ok" and x["claim_tag"] == "TAG" and x["leg"] == "tp_b8_t4"
    assert x["arm_identity_digest"] == roster()["bge-large-st-fp32"].spec.index_identity()["digest"]
    assert x["loadavg_before"] == [4.0, 3.0, 2.0] and x["loadavg_after"] == [4.0, 3.0, 2.0]
    assert x["cpus_requested"] == "0-3" and x["cpus_applied"] == "0-3"
    assert x["t_end"] - x["t_start"] == pytest.approx(8.0)  # 8 slices of 1 s on the fake clock
    assert (x["freq_khz"]["min"], x["freq_khz"]["median"], x["freq_khz"]["max"]) == (
        3000000,
        3150000,
        3300000,
    )
    assert x["n_chunks"] == 64 and x["wall_s"] == pytest.approx(8.0)
    assert x["chunks_per_s"] == pytest.approx(8.0) and x["cpu_s_window"] == pytest.approx(8.0)
    assert x["tokens_source"] == "tokenizer" and x["tokens_total"] == x["tokens_per_s"] * 8.0
    assert (
        x["config"]["batch"] == 8
        and x["config"]["threads"] == 4
        and x["config"]["threads_kind"] == "torch"
    )


def test_a_contended_leg_is_aborted_recorded_and_retried_on_resume(tmp_path):
    fp = FakeProc()
    world = World(fp, [0, 1, 2, 3], own_ticks=100, foreign_ticks=400)
    kw = dict(
        only=("throughput",), batches=(8,), threads=(4,), check_interval_s=0.5, min_window_s=0.5
    )
    runner, _ = make_runner(tmp_path, ["bge-large-st-fp32"], proc=fp, world=world, **kw)
    res = runner.run()
    assert res["status_counts"] == {"contended": 2}
    rows = sp.load_rows(tmp_path / "speed.jsonl")
    x = rows[0]
    assert x["status"] == "contended" and x["aborted"] is True
    assert "foreign_busy_interval" in x["contention"]["reasons"]
    assert (
        x["n_chunks"] == 8 < x["config"]["n_chunks_planned"]
    )  # aborted after the first slice, still recorded
    assert (
        (sp.aggregate(rows)["arms"]["bge-large-st-fp32"]["throughput"]["tp_b8_t4"]["n_contended"])
        == 2
    )
    assert (
        sp.aggregate(rows)["arms"]["bge-large-st-fp32"]["throughput"]["tp_b8_t4"]["chunks_per_s"]
        is None
    )
    world.foreign_ticks = 0  # the strangers are gone: a re-run retries exactly those units
    runner2, _ = make_runner(tmp_path, ["bge-large-st-fp32"], proc=fp, world=world, **kw)
    res2 = runner2.run()
    assert res2["executed"] == 2 and res2["status_counts"] == {"ok": 2}
    final = sp.load_rows(tmp_path / "speed.jsonl")
    assert {r["status"] for r in final} == {"ok"}
    assert (
        len((tmp_path / "speed.jsonl").read_text().splitlines()) == 4
    )  # the contended rows stay in the log
    runner3, _ = make_runner(tmp_path, ["bge-large-st-fp32"], proc=fp, world=world, **kw)
    assert runner3.run()["executed"] == 0


def test_an_affinity_violation_marks_the_leg_contended(tmp_path):
    class Leaky(sp.DryAffinity):
        def readback(self, want=None):
            rb = super().readback(want)
            rb["violations"] = [4242]
            return rb

    runner, fp = make_runner(
        tmp_path, ["bge-large-st-fp32"], only=("throughput",), batches=(8,), threads=(4,)
    )
    runner.affinity = Leaky([0, 1, 2, 3])
    runner.run()
    rows = sp.load_rows(tmp_path / "speed.jsonl")
    assert {r["status"] for r in rows} == {"contended"}
    assert "affinity_violation" in rows[0]["contention"]["reasons"] and rows[0][
        "affinity_violations"
    ] == [4242]


def test_resumability_skips_finished_units_and_refuses_a_changed_plan(tmp_path):
    ids = ["bge-large-st-fp32", "potion-base-8m"]

    class Stop(Exception):
        pass

    n_seen = []

    def stop_after(u, row):
        n_seen.append(u.key)
        if len(n_seen) == 15:
            raise KeyboardInterrupt

    runner, _ = make_runner(tmp_path, ids, only=("throughput", "query", "index"))
    runner.after_unit = stop_after
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    first = (tmp_path / "speed.jsonl").read_text().splitlines()
    assert len(first) == 15
    runner2, _ = make_runner(tmp_path, ids, only=("throughput", "query", "index"))
    total = len(runner2.build_plan())
    res = runner2.run()
    assert res["skipped_done"] == 15 and res["executed"] == total - 15
    keys = [
        json.loads(line)["unit_key"] for line in (tmp_path / "speed.jsonl").read_text().splitlines()
    ]
    assert len(keys) == total == len(set(keys))  # nothing measured twice
    assert runner2.run()["executed"] == 0  # fully done: a third run measures nothing
    assert (tmp_path / "noise_floor.json").exists()
    runner3, _ = make_runner(tmp_path, ids, only=("throughput", "query", "index"), batches=(1, 8))
    with pytest.raises(sp.Refusal, match="plan changed"):
        runner3.run()


def test_row_store_final_states(tmp_path):
    st = sp.RowStore(tmp_path / "x.jsonl")
    for key, status in (("ok", "ok"), ("na", "not_applicable"), ("c", "contended"), ("e", "error")):
        st.append({"unit_key": key, "status": status})
    assert st.is_final("ok") and st.is_final("na")
    assert not st.is_final("c") and st.is_final("c", retry_contended=False)
    assert not st.is_final("e") and not st.is_final("missing")
    st.append({"unit_key": "c", "status": "ok"})  # the retry wins
    again = sp.RowStore(tmp_path / "x.jsonl")
    assert again.latest["c"]["status"] == "ok" and again.is_final("c")
    (tmp_path / "x.jsonl").write_text((tmp_path / "x.jsonl").read_text() + "{torn line\n")
    assert sp.RowStore(tmp_path / "x.jsonl").bad_lines == 1


def test_a_failing_arm_is_recorded_on_every_unit_never_dropped(tmp_path):
    r = roster()
    rts = {"bge-large-st-fp32": sp.FakeRuntime(r["bge-large-st-fp32"], fail_open="weights missing"),
           "potion-base-8m": sp.FakeRuntime(r["potion-base-8m"])}  # fmt: skip
    runner, _ = make_runner(
        tmp_path, [], runtimes=rts, only=("throughput", "query"), batches=(8,), threads=(4,)
    )
    runner.run()
    rows = sp.load_rows(tmp_path / "speed.jsonl")
    bad = [x for x in rows if x["arm_id"] == "bge-large-st-fp32" and x["status"] == "error"]
    assert bad and all("weights missing" in x["error"] for x in bad)
    assert any(x["arm_id"] == "potion-base-8m" and x["status"] == "ok" for x in rows)
    arm = sp.aggregate(rows)["arms"]["bge-large-st-fp32"]
    assert (
        arm["status_counts"]["error"] >= 1 and arm["throughput"]["tp_b8_t4"]["chunks_per_s"] is None
    )


def test_plan_has_explicit_not_applicable_rows_aba_logs_and_aa_slots(tmp_path):
    ids = ["bge-large-gguf-f16", "bge-large-st-fp32", "potion-base-8m"]
    runner, _ = make_runner(tmp_path, ids)
    plan = runner.build_plan()
    keys = [u.key for u in plan]
    assert len(keys) == len(set(keys))
    na = [u for u in plan if u.order_dir == "na"]
    assert {(u.leg_id, u.arm_id) for u in na} == {
        ("q_inproc", "bge-large-gguf-f16"), ("q_served", "bge-large-st-fp32"), ("q_served", "potion-base-8m"),
    }  # fmt: skip
    # decision-relevant legs first, the slow single-thread sweep last
    legs = [leg.leg_id for leg in runner.legs]
    assert legs[:3] == ["idx_b32", "q_inproc", "q_served"] and legs[-1].endswith("_t1")
    # A/A slots: the incumbent at its reference legs, before the first ABA leg and after the last
    first_main = next(i for i, u in enumerate(plan) if u.phase == "main")
    last_main = max(i for i, u in enumerate(plan) if u.phase == "main")
    early = [i for i, u in enumerate(plan) if u.phase == "aa_early"]
    late = [i for i, u in enumerate(plan) if u.phase == "aa_late"]
    assert early and late and max(early) < first_main and min(late) > last_main
    assert {plan[i].arm_id for i in early + late} == {"bge-large-gguf-f16"}
    assert {plan[i].leg_id for i in early} == {"tp_b32_srv", "q_served", "idx_b32"}
    # every leg's measuring units follow aba_order over the applicable arms only
    leg = next(u.leg_id for u in plan if u.leg_id == "tp_b32_t4")
    seq = [u.arm_id for u in plan if u.leg_id == leg and u.phase == "main"]
    assert seq == ["bge-large-st-fp32", "bge-large-st-fp32"]
    seq_q = [
        u.arm_id
        for u in plan
        if u.leg_id == "q_inproc" and u.phase == "main" and u.order_dir != "na"
    ]
    assert seq_q == ["bge-large-st-fp32", "potion-base-8m", "potion-base-8m", "bge-large-st-fp32"]


def test_full_socket_leg_exists_only_when_asked():
    base = sp.RunConfig(out_dir=Path("/x"), cpu_list=[0, 1, 2, 3], claim_tag="t")
    assert not [leg for leg in sp.build_legs(base) if leg.threads == "full"]
    full = sp.RunConfig(out_dir=Path("/x"), cpu_list=[0, 1, 2, 3], claim_tag="t", full_socket=True)
    assert [leg.leg_id for leg in sp.build_legs(full) if leg.threads == "full"] == [
        "tp_b32_tfull",
        "tp_b128_tfull",
    ]


def test_threads_and_batches_reach_the_encoder_and_subsets_follow_the_spec(tmp_path):
    runner, _ = make_runner(tmp_path, ["bge-large-st-fp32"], only=("throughput",), batches=(1, 8), threads=(1, 4),
                            subset_n=32, subset_n_b1=10)  # fmt: skip
    rt = runner.runtimes["bge-large-st-fp32"]
    runner.run()
    assert set(rt.thread_calls) == {1, 4}
    rows = {
        r["leg"]: r
        for r in sp.load_rows(tmp_path / "speed.jsonl")
        if r["phase"] == "main" and r["rep"] == 0
    }
    assert (
        rows["tp_b1_t4"]["n_chunks"] == 10 and rows["tp_b8_t4"]["n_chunks"] == 32
    )  # 64 for batch 1, else 256
    assert rows["tp_b1_t1"]["cpus_requested"] == "0" and rows["tp_b8_t4"]["cpus_requested"] == "0-3"
    assert sp.stride_subset(list(range(100)), 4) == [0, 25, 50, 75]
    assert sp.stride_subset([1, 2], 9) == [1, 2]


# ───────────────────────────────────────────────────────────────────────────── adapters (no weights)


class _FakeSTModel:
    def __init__(self):
        self.calls = []

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ):
        self.calls.append((list(texts), batch_size))
        return np.ones((len(texts), 768), dtype=np.float32)


def test_st_adapter_subclass_carries_client_batch_and_the_document_prefix():
    spec = roster()["eg2-st-fp32"].spec
    ad = sp.SpeedST.__new__(sp.SpeedST)
    arms._AdapterBase.__init__(ad, spec)
    ad.model = _FakeSTModel()
    rt = sp.STRuntime(roster()["eg2-st-fp32"], "/nonexistent")
    rt.adapter = ad
    v = rt.encode_docs(["a", "b"], 128)
    assert ad.model.calls[-1] == (["title: none | text: a", "title: none | text: b"], 128)
    assert v.shape == (2, 768) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    rt.encode_query("where")
    assert ad.model.calls[-1] == (["task: search result | query: where"], 1)


def test_bge_query_prefix_is_the_shipped_instruction_string():
    prefix = cs.query_text("bge-large-en-v1.5", "")
    assert prefix and prefix.startswith("Represent this sentence")
    for arm in ("bge-large-gguf-f16", "bge-large-st-fp32"):
        assert roster()[arm].spec.query_prompt == prefix and roster()[arm].spec.doc_prompt == ""
    assert roster()["granite-97m-r2-gguf-f16"].spec.query_prompt == ""


def test_static_adapter_runs_single_threaded_with_the_leg_batch(monkeypatch):
    seen = {}

    class FakeStatic:
        def encode(
            self,
            sentences,
            show_progress_bar=False,
            max_length=512,
            batch_size=1024,
            use_multiprocessing=True,
        ):
            seen.update(batch_size=batch_size, mp=use_multiprocessing, n=len(sentences))
            return np.ones((len(sentences), 256), dtype=np.float32)

    def fake_init(self, spec, path):
        arms._AdapterBase.__init__(self, spec)
        self.model = FakeStatic()

    monkeypatch.setattr(arms.Model2VecAdapter, "__init__", fake_init)
    e = roster()["potion-base-8m"]
    rt = sp.StaticRuntime(e, "/nonexistent")
    rt.adapter = sp.SpeedM2V(e.spec, "/nonexistent")
    rt.encode_docs(["x", "y", "z"], 32)
    assert seen == {
        "batch_size": 32,
        "mp": False,
        "n": 3,
    }  # a worker pool would make "single-threaded" false
    assert rt.encode_docs(["x"], 1).shape == (1, 256)


class _Resp:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_server(monkeypatch, dim=384, prompt_tokens_each=7, usage=True, build="b10305-966c24d6e"):
    urls = []

    def urlopen(req, timeout=None):
        url = getattr(req, "full_url", req)
        urls.append(url)
        if url.endswith("/props"):
            return _Resp(
                {
                    "total_slots": 4,
                    "build_info": build,
                    "default_generation_settings": {"n_ctx": 8192},
                }
            )
        n = len(json.loads(req.data)["input"])
        body = {"data": [{"index": i, "embedding": [1.0] * dim} for i in reversed(range(n))]}
        if usage:
            body["usage"] = {
                "prompt_tokens": prompt_tokens_each * n,
                "total_tokens": prompt_tokens_each * n,
            }
        return _Resp(body)

    monkeypatch.setattr(sp.urllib.request, "urlopen", urlopen)
    return urls


def test_served_runtime_reads_usage_tokens_threads_and_resolves_unknown_dim_and_identity(
    monkeypatch, tmp_path
):
    urls = _fake_server(monkeypatch, dim=384)
    gguf = tmp_path / "granite.gguf"
    gguf.write_bytes(b"gguf-bytes")
    fp = FakeProc()
    fp.add(
        900,
        comm="llama-server",
        cmdline=["llama-server", "-m", str(gguf), "--threads", "16", "--embedding"],
    )
    e = roster()["granite-97m-r2-gguf-f16"]
    assert e.spec.native_dim == 0 and e.spec.revision.startswith("unresolved")
    rt = sp.ServedRuntime(
        e, "http://127.0.0.1:18751", 900, fp, server_threads=12, server_flags="-t 12"
    )
    meta = rt.open()
    assert meta["native_dim"] == 384 and rt.spec.dim == 384 and rt.spec.native_dim == 384
    assert rt.spec.revision == "sha256:" + sp.sha256_text(
        "gguf-bytes"
    ) or rt.spec.revision.startswith("sha256:")
    assert rt.server_threads == 12 and any("cmdline --threads 16" in w for w in rt.warnings)
    assert rt.server_meta["props"]["total_slots"] == 4 and rt.server_meta["flags_echo"] == "-t 12"
    assert rt.take_tokens() is None  # probe tokens were discarded at open
    v = rt.encode_docs(["a", "b", "c"], 32)
    assert v.shape == (3, 384)
    assert (
        rt.take_tokens() == 21 and rt.take_tokens() is None
    )  # server-reported prompt_tokens, then reset
    assert rt.count_tokens(["a"]) is None
    assert urls[0].endswith("/props") and urls[-1].endswith("/v1/embeddings")


def test_served_runtime_without_usage_reports_no_token_count(monkeypatch):
    _fake_server(monkeypatch, dim=1024, usage=False)
    fp = FakeProc()
    fp.add(901, comm="llama-server", cmdline=["llama-server"])
    rt = sp.ServedRuntime(roster()["bge-large-gguf-f16"], "http://127.0.0.1:18752", 901, fp)
    assert roster()["bge-large-gguf-f16"].spec.runtime.startswith("unresolved")
    rt.open()
    # the incumbent's runtime is whatever the caller's server reports, not another arm's build string
    assert rt.spec.runtime == "llama-server b10305-966c24d6e"
    assert rt.spec.revision == f"sha256:{sp.BGE_GGUF_SHA256}"
    rt.encode_docs(["a"], 1)
    assert rt.take_tokens() is None
    bad = sp.ServedRuntime(roster()["eg2-gguf-bf16"], "http://127.0.0.1:18753", 901, fp)
    with pytest.raises(RuntimeError, match="native dim 1024 != declared 768"):
        bad.open()


# ───────────────────────────────────────────────────────────────────────────── noise floor / labels


def mk(arm, kind, phase, leg, rep, t, status="ok", **kw):
    leg_kind = (
        "throughput"
        if leg.startswith("tp_")
        else ("query" if leg.startswith("q_") else "index_build")
    )
    entry = roster().get(arm)
    ident = entry.spec.index_identity() if entry else {"digest": f"d-{arm}", "model": arm}
    base = {
        "unit_key": f"{phase}|{leg}|{arm}|{rep}|{t}",
        "arm_id": arm,
        "arm_kind": kind,
        "phase": phase,
        "leg": leg,
        "leg_kind": leg_kind,
        "rep": rep,
        "t_start": t,
        "status": status,
        "arm_identity": ident,
        "arm_identity_digest": ident["digest"],
        "run_id": "r",
        "claim_tag": "TAG",
        "cpu_list_claimed": "24-71",
        "recall_arm_ids": list(entry.recall_arm_ids) if entry else [],
        "shares_forward_pass": [dict(x) for x in entry.shares_forward_pass] if entry else [],
    }
    if leg_kind == "throughput":
        base["config"] = {
            "batch": 32,
            "threads": None if kind == "gguf" else 4,
            "threads_kind": "server" if kind == "gguf" else "torch",
            "server_threads": 8 if kind == "gguf" else None,
        }
        base["n_chunks"] = 256
    return base | kw


def aa_rows(arm="bge-large-gguf-f16", vals=(100.0, 104.0, 98.0, 102.0)):
    slots = [("aa_early", None, 10), ("main", 0, 20), ("main", 1, 30), ("aa_late", None, 40)]
    return [
        mk(arm, "gguf", ph, "tp_b32_srv", rep, t, chunks_per_s=v, tokens_per_s=v * 40)
        for (ph, rep, t), v in zip(slots, vals)
    ]


def test_noise_floor_is_the_aa_spread_in_the_metrics_own_unit():
    rows = aa_rows()
    rows += [
        mk("bge-large-gguf-f16", "gguf", ph, "q_served", rep, t, p50_ms=v, p95_ms=2 * v, max_ms=3 * v, cpu_s_per_query=v / 100)
        for (ph, rep, t), v in zip([("aa_early", None, 11), ("main", 0, 21), ("main", 1, 31), ("aa_late", None, 41)], (50.0, 52.0, 49.0, 51.0))
    ]  # fmt: skip
    nf = sp.compute_noise_floor(rows, "bge-large-gguf-f16")
    c = nf["metrics"]["chunks_per_s"]
    assert (c["unit"], c["n"], c["abs"], c["leg"]) == ("chunks/s", 4, 6.0, "tp_b32_srv")
    assert c["rel"] == pytest.approx(6.0 / 101.0) and c["median"] == pytest.approx(101.0)
    assert [s["slot"] for s in c["samples"]] == [
        "aa_early",
        "aba_p0",
        "aba_p1",
        "aa_late",
    ]  # time order
    assert nf["metrics"]["tokens_per_s"]["unit"] == "tokens/s" and nf["metrics"]["tokens_per_s"][
        "abs"
    ] == pytest.approx(240.0)
    p50 = nf["metrics"]["query_p50_ms"]
    assert (p50["unit"], p50["abs"]) == ("ms", pytest.approx(3.0)) and nf["metrics"][
        "query_p95_ms"
    ]["abs"] == pytest.approx(6.0)
    assert nf["metrics"]["cpu_s_per_query"]["unit"] == "cpu-s/query"
    assert (
        nf["metrics"]["index_build_s_4096"]["abs"] is None
    )  # no index rows: stated, never invented
    assert nf["metrics"]["index_build_s_4096"]["reason"]
    assert nf["rep_to_rep"]["chunks_per_s"]["n_pairs"] == 1


def test_noise_floor_ignores_contended_samples_and_needs_two():
    rows = aa_rows()
    rows[1]["status"] = "contended"
    rows[2]["status"] = "error"
    nf = sp.compute_noise_floor(rows, "bge-large-gguf-f16")
    assert nf["metrics"]["chunks_per_s"]["n"] == 2 and nf["metrics"]["chunks_per_s"][
        "abs"
    ] == pytest.approx(2.0)
    rows[3]["status"] = "contended"
    one = sp.compute_noise_floor(rows, "bge-large-gguf-f16")["metrics"]["chunks_per_s"]
    assert one["n"] == 1 and one["abs"] is None and "fewer than 2" in one["reason"]
    assert sp.compute_noise_floor([], None)["metrics"] == {}


def test_within_noise_labelling():
    floor = {"abs": 6.0, "rel": 0.06, "leg": "tp_b32_srv"}
    assert sp.is_within_noise(100.0, 105.0, floor, leg="tp_b32_srv") is True  # |d|=5 <= 6
    assert sp.is_within_noise(100.0, 107.0, floor, leg="tp_b32_srv") is False
    # another leg: the abs floor is in the A/A leg's scale, so only the relative floor applies there
    assert sp.is_within_noise(1000.0, 1050.0, floor, leg="tp_b128_srv") is True  # 50 <= 0.06 x 1025
    assert sp.is_within_noise(1000.0, 1100.0, floor, leg="tp_b128_srv") is False
    assert (
        sp.is_within_noise(10.0, 14.0, floor, leg=None) is True
    )  # unit-comparable metrics: abs applies
    assert sp.is_within_noise(1.0, 2.0, None) is None  # no floor: never guess
    assert sp.is_within_noise(1.0, 2.0, {"abs": None, "rel": None}) is None
    assert sp.is_within_noise(None, 2.0, floor) is None


def test_summary_labels_a_cross_arm_difference_smaller_than_the_floor_within_noise():
    rows = aa_rows()  # baseline: main medians 104, 98 -> 101; floor abs 6.0 (4 samples)
    rows += [
        mk("eg2-gguf-q8_0", "gguf", "main", "tp_b32_srv", r, 50 + r, chunks_per_s=v)
        for r, v in ((0, 103.0), (1, 101.0))
    ]
    rows += [
        mk("eg2-gguf-bf16", "gguf", "main", "tp_b32_srv", r, 60 + r, chunks_per_s=v)
        for r, v in ((0, 151.0), (1, 149.0))
    ]
    s = sp.build_summary(rows)
    assert s["baseline_arm"] == "bge-large-gguf-f16"
    near = next(i for i in s["vs_baseline"]["eg2-gguf-q8_0"] if i["metric"] == "chunks_per_s")
    far = next(i for i in s["vs_baseline"]["eg2-gguf-bf16"] if i["metric"] == "chunks_per_s")
    assert near["within_noise"] is True and near["diff"] == pytest.approx(1.0)
    assert far["within_noise"] is False and far["ratio"] == pytest.approx(150.0 / 101.0)
    md = sp.render_markdown(s)
    assert "**within_noise**" in md and "A/A arm: `bge-large-gguf-f16`" in md and "chunks/s" in md
    # with no A/A rows at all there is no floor: the label is "n/a", not a guess
    s2 = sp.build_summary([r for r in rows if r["phase"] == "main"])
    assert (
        next(i for i in s2["vs_baseline"]["eg2-gguf-q8_0"] if i["metric"] == "chunks_per_s")[
            "within_noise"
        ]
        is None
    )


# ───────────────────────────────────────────────────────────────────────────── selection / capture


def full_arm_rows(
    arm,
    kind,
    *,
    tp=100.0,
    qleg=None,
    lat=(1.0, 2.0, 3.0, 4.0),
    cpu=2.0,
    impl=70.0,
    rss_kb=2048,
    hwm_kb=4096,
    disk=1_000_000 * 3,
):
    """Two ABA visits per leg that differ a little (so each metric has a repeat spread), two RSS
    children, a repeated disk stat."""
    qleg = qleg or ("q_served" if kind == "gguf" else "q_inproc")
    tp_leg = "tp_b32_srv" if kind == "gguf" else "tp_b32_t4"
    rows = []
    for r in (0, 1):
        rows.append(
            mk(
                arm,
                kind,
                "main",
                tp_leg,
                r,
                100 + r,
                chunks_per_s=tp + r,
                tokens_per_s=40 * (tp + r),
            )
        )
        scaled = [x * (1.0 + 0.1 * r) for x in lat]
        rows.append(mk(arm, kind, "main", qleg, r, 200 + r, latencies_ms=scaled, n_queries=len(lat),
                       cpu_s_window=cpu * (1 + 0.1 * r), cpu_s_per_query=cpu * (1 + 0.1 * r) / len(lat),
                       config={"threads": 4, "n_distinct_queries": 200, "repeats": 3, "server_threads": 8},
                       cpu={"cpu_source": "stat"}, **{k: v for k, v in sp.latency_stats(scaled).items() if k != "n"}))  # fmt: skip
        imp_r = impl * (1 + 0.05 * r)
        rows.append(mk(arm, kind, "main", "idx_b32", r, 300 + r, chunks_per_s=4096 / imp_r, implied_build_s_4096=imp_r, finite=True,
                       n_index_chunks=4096, dim=16, nonfinite_rows=0, budget=sp.index_budget_report(4096 / imp_r, 4096, cs),
                       config={"batch": 32, "threads": 4, "threads_kind": "torch"}))  # fmt: skip
        rows.append(mk(arm, kind, "static", "rss", r, 1 + r, leg_kind="rss", rss_after_load_kb=rss_kb + 100 * r,
                       rss_baseline_kb=rss_kb // 4, hwm_kb=hwm_kb + 100 * r, rss_source="isolated_child_process"))  # fmt: skip
    rows.append(mk(arm, kind, "static", "disk", 0, 2, leg_kind="disk",
                   disk={"bytes": disk, "weights_bytes": disk - 10, "files": 3, "path": "/m", "repeat_bytes": disk}))  # fmt: skip
    return rows


def test_selection_costs_use_the_as_served_leg_for_gguf_and_in_process_for_the_rest():
    rows = full_arm_rows("bge-large-gguf-f16", "gguf", lat=(10.0, 20.0, 30.0, 40.0), cpu=8.0)
    # a stray in-process leg on a gguf arm must not leak into its selection cost
    rows += [
        mk(
            "bge-large-gguf-f16",
            "gguf",
            "main",
            "q_inproc",
            0,
            9,
            status="not_applicable",
            leg_kind="query",
        )
    ]
    rows += full_arm_rows("eg2-st-fp32", "st", lat=(1.0, 2.0, 3.0, 5.0), cpu=0.4)
    rows += [
        mk(
            "eg2-st-fp32",
            "st",
            "main",
            "q_served",
            None,
            9,
            status="not_applicable",
            leg_kind="query",
        )
    ]
    s = sp.build_summary(rows)
    costs = s["selection"]["costs"]
    assert set(costs) == {"bge-large-gguf-f16", "eg2-st-fp32"}
    for v in costs.values():
        assert set(v) == {"cpu_s_per_query", "p95_ms"}  # exactly the 2.1 inputs
    # two visits of 4 queries each: (8.0 + 8.8) cpu-s / 8 queries; latencies pooled across visits
    assert costs["bge-large-gguf-f16"]["cpu_s_per_query"] == pytest.approx((8.0 + 8.8) / 8)
    assert costs["bge-large-gguf-f16"]["p95_ms"] == pytest.approx(
        sp.percentile([10, 20, 30, 40, 11, 22, 33, 44], 95)
    )
    assert costs["eg2-st-fp32"]["cpu_s_per_query"] == pytest.approx((0.4 + 0.44) / 8)
    assert s["selection"]["legs"] == {
        "bge-large-gguf-f16": "as_served",
        "eg2-st-fp32": "in_process",
    }
    assert [r["arm_id"] for r in s["selection"]["ranking_by_cpu_s_per_query"]] == [
        "eg2-st-fp32",
        "bge-large-gguf-f16",
    ]
    assert (
        "as-served leg for GGUF arms, in-process leg for ST/static arms" in s["selection"]["rule"]
    )


CAPTURE_METRICS = {"chunks_per_s", "tokens_per_s", "query_p50_ms", "query_p95_ms", "query_max_ms",
                   "cpu_s_per_query", "index_build_s_4096", "rss_mb", "disk_mb"}  # fmt: skip


def capture_fixture():
    rows = (
        aa_rows()
        + [r for r in full_arm_rows("bge-large-gguf-f16", "gguf") if r["leg"] != "tp_b32_srv"]
        + full_arm_rows("eg2-st-fp32", "st", tp=40.0)
    )
    rows += full_arm_rows("eg2-gguf-q8_0", "gguf", tp=90.0)
    nf = sp.compute_noise_floor(rows, "bge-large-gguf-f16")
    return rows, nf


def test_rows_for_capture_has_exactly_the_add_speed_keys_and_the_nine_metrics():
    rows, nf = capture_fixture()
    out = sp.rows_for_capture(rows, nf)
    assert out and all(set(r) == set(sp.CAPTURE_KEYS) for r in out)
    assert sp.CAPTURE_KEYS == (
        "arm_identity",
        "metric",
        "value",
        "unit",
        "n",
        "spread",
        "spread_unit",
        "config_echo",
    )
    assert {r["metric"] for r in out} == CAPTURE_METRICS
    for r in out:
        assert r["spread_unit"] == r["unit"] == sp.CAPTURE_UNITS[r["metric"]]
        assert isinstance(r["config_echo"], dict) and r["config_echo"]
        assert isinstance(r["value"], float) and r["value"] > 0
        assert isinstance(r["n"], int) and r["n"] >= 1
        assert (
            isinstance(r["spread"], float) and r["spread"] >= 0
        )  # never None: the writer refuses it
        assert set(r["arm_identity"]) == set(sp.ARM_FIELDS)
    assert {r["unit"] for r in out if r["metric"] in ("rss_mb", "disk_mb")} == {
        "MB"
    }  # the writer's unit
    q = next(
        r
        for r in out
        if r["metric"] == "query_p95_ms" and r["arm_identity"]["arm_id"] == "bge-large-gguf-f16"
    )
    assert (
        q["config_echo"]["served"] == "as_served" and q["n"] == 8
    )  # 2 visits x 4 latencies pooled
    assert q["spread"] == pytest.approx(
        sp.percentile([1, 2, 3, 4], 95) * 0.1
    )  # range of the per-visit p95
    assert q["config_echo"]["spread_kind"] == "visit_range"
    tp = next(
        r
        for r in out
        if r["metric"] == "chunks_per_s" and r["arm_identity"]["arm_id"] == "eg2-st-fp32-768"
    )
    assert (
        tp["spread"] == pytest.approx(1.0) and tp["config_echo"]["spread_kind"] == "rep_range"
    )  # |41 - 40|
    assert (
        tp["config_echo"]["leg"] == "tp_b32_t4"
        and tp["config_echo"]["speed_arm_id"] == "eg2-st-fp32"
    )
    idx = next(
        r
        for r in out
        if r["metric"] == "index_build_s_4096"
        and r["arm_identity"]["arm_id"] == "bge-large-gguf-f16"
    )
    assert (
        idx["value"] == pytest.approx(71.75) and idx["config_echo"]["exposes_index_timeout"] is True
    )
    assert idx["spread"] == pytest.approx(3.5)
    assert (
        idx["config_echo"]["budget_s"] == 60.0 and idx["config_echo"]["searches_to_complete"] == 2
    )
    rss = [
        r
        for r in out
        if r["metric"] == "rss_mb" and r["arm_identity"]["arm_id"] == "eg2-st-fp32-768"
    ]
    assert {r["config_echo"]["kind"] for r in rss} == {"after_load", "peak_hwm"}
    after = next(r for r in rss if r["config_echo"]["kind"] == "after_load")
    assert (
        after["value"] == pytest.approx(2098 * 1024 / 1e6) and after["n"] == 2
    )  # decimal MB, 2 children
    assert after["spread"] == pytest.approx(100 * 1024 / 1e6)
    assert next(r for r in rss if r["config_echo"]["kind"] == "peak_hwm")["value"] == pytest.approx(
        4196 * 1024 / 1e6
    )
    disk = next(r for r in out if r["metric"] == "disk_mb")
    assert disk["value"] == pytest.approx(3.0) and disk["spread"] == 0.0


def test_capture_identity_roles_and_the_256_alias_sharing_a_forward_pass():
    rows, nf = capture_fixture()
    out = sp.rows_for_capture(rows, nf)
    ids = {r["arm_identity"]["arm_id"]: r["arm_identity"] for r in out}
    assert set(ids) == {
        "bge-large-gguf-f16",
        "eg2-st-fp32-768",
        "eg2-st-fp32-256",
        "eg2-gguf-q8_0-768",
        "eg2-gguf-q8_0-256",
    }
    base = roster()["bge-large-gguf-f16"].spec.index_identity()
    b = ids["bge-large-gguf-f16"]
    assert (b["role"], b["dim"], b["dtype_or_quant"], b["model"]) == (
        "BASELINE",
        1024,
        "f16",
        base["model"],
    )
    assert b["index_identity_digest"] == base["digest"] and b["prompt_hash"] == base["prompt_hash"]
    assert ids["eg2-st-fp32-768"]["role"] == "CANDIDATE" and ids["eg2-st-fp32-768"]["dim"] == 768
    assert ids["eg2-st-fp32-256"]["dim"] == 256
    # the frozen manifest's own digests, not new ones
    base_specs = {s.arm_id: s for s in arms.arm_specs("st-rt", "llama-rt", "m2v-rt")}
    assert (
        ids["eg2-st-fp32-256"]["index_identity_digest"]
        == base_specs["eg2-st-fp32-256"].index_identity()["digest"]
    )
    a768 = [
        r
        for r in out
        if r["arm_identity"]["arm_id"] == "eg2-st-fp32-768" and r["metric"] == "chunks_per_s"
    ][0]
    a256 = [
        r
        for r in out
        if r["arm_identity"]["arm_id"] == "eg2-st-fp32-256" and r["metric"] == "chunks_per_s"
    ][0]
    assert a256["value"] == a768["value"] and a256["config_echo"]["alias_of"] == "eg2-st-fp32-768"
    assert a768["config_echo"]["alias_of"] is None  # one measurement, labelled, never a second leg
    plain = sp.rows_for_capture(rows, nf, include_aliases=False)
    assert not any(r["arm_identity"]["arm_id"].endswith("-256") for r in plain)
    flipped = sp.rows_for_capture(rows, nf, roles={"eg2-st-fp32": "BASELINE"})
    assert {
        r["arm_identity"]["role"]
        for r in flipped
        if r["arm_identity"]["arm_id"] == "eg2-st-fp32-768"
    } == {"BASELINE"}


def test_noise_floors_for_capture_cover_every_speed_row_in_the_metrics_unit():
    rows, nf = capture_fixture()
    out = sp.rows_for_capture(rows, nf)
    floors = sp.noise_floors_for_capture(rows, nf)
    assert all(set(f) == set(sp.FLOOR_KEYS) for f in floors)
    keyed = {(f["arm_identity"]["arm_id"], f["metric"]): f for f in floors}
    assert len(keyed) == len(floors)  # one per (arm, metric): the writer refuses duplicates
    assert {(r["arm_identity"]["arm_id"], r["metric"]) for r in out} == set(
        keyed
    )  # nothing without a floor
    for f in floors:
        assert f["unit"] == sp.CAPTURE_UNITS[f["metric"]] and f["n"] >= 2 and f["aa_spread"] >= 0
        assert f["config_echo"] is None and f["replication_unit"] in ("arm", "process")
    a = nf["metrics"]["chunks_per_s"]
    # the A/A arm keeps its absolute floor; another arm gets the relative floor x its own scale
    assert keyed[("bge-large-gguf-f16", "chunks_per_s")]["aa_spread"] == pytest.approx(
        max(a["abs"], a["rel"] * 100.0)
    )
    assert keyed[("eg2-st-fp32-768", "chunks_per_s")]["aa_spread"] == pytest.approx(a["rel"] * 40.5)
    assert keyed[("eg2-st-fp32-256", "chunks_per_s")]["aa_spread"] == pytest.approx(a["rel"] * 40.5)
    # RSS and disk floors come from the arm's own repeats, in their own replication units
    assert keyed[("eg2-st-fp32-768", "rss_mb")]["replication_unit"] == "process"
    assert keyed[("eg2-st-fp32-768", "rss_mb")]["aa_spread"] == pytest.approx(100 * 1024 / 1e6)
    assert keyed[("eg2-st-fp32-768", "disk_mb")]["aa_spread"] == 0.0


def test_a_row_without_a_spread_or_a_floor_is_omitted_and_reported_never_defaulted():
    rows, nf = capture_fixture()
    for (
        r
    ) in rows:  # one visit of the ST arm's throughput leg was contended: 1 ok repeat => no spread
        if r["arm_id"] == "eg2-st-fp32" and r["leg"] == "tp_b32_t4" and r["rep"] == 1:
            r["status"] = "contended"
    out = sp.rows_for_capture(rows, nf)
    assert not [
        r
        for r in out
        if r["arm_identity"]["arm_id"] == "eg2-st-fp32-768" and r["metric"] == "chunks_per_s"
    ]
    gaps = sp.capture_gaps(rows, nf)
    assert any(
        g["arm_id"] == "eg2-st-fp32"
        and g["metric"] == "chunks_per_s"
        and "no spread" in g["reason"]
        for g in gaps
    )
    # no A/A floor at all => every row that needs one is dropped, with the reason, never defaulted
    nf2 = {"aa_arm": "bge-large-gguf-f16", "metrics": {}}
    out2 = sp.rows_for_capture(rows, nf2)
    floors2 = sp.noise_floors_for_capture(rows, nf2)
    assert {(r["arm_identity"]["arm_id"], r["metric"]) for r in out2} == {
        (f["arm_identity"]["arm_id"], f["metric"]) for f in floors2
    }  # whatever is emitted always has its floor
    assert {r["metric"] for r in out2} <= {
        "rss_mb",
        "disk_mb",
    }  # only those carry their own repeats
    assert any("noise floor" in g["reason"] for g in sp.capture_gaps(rows, nf2))
    # a zero cpu reading (below tick resolution) is not a positive value: omitted, not written as 0
    rows3, nf3 = capture_fixture()
    for r in rows3:
        if r["arm_id"] == "eg2-st-fp32" and r["leg"] == "q_inproc":
            r["cpu_s_window"], r["cpu_s_per_query"] = 0.0, 0.0
    assert not [
        r
        for r in sp.rows_for_capture(rows3, nf3)
        if r["metric"] == "cpu_s_per_query" and r["arm_identity"]["arm_id"] == "eg2-st-fp32-768"
    ]
    assert any(
        g["metric"] == "cpu_s_per_query" and "positive" in g["reason"]
        for g in sp.capture_gaps(rows3, nf3)
    )


def test_the_noise_floor_may_be_a_path_and_rows_may_be_a_path(tmp_path):
    rows, nf = capture_fixture()
    (tmp_path / "speed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "noise_floor.json").write_text(json.dumps(nf))
    a = sp.rows_for_capture(tmp_path / "speed.jsonl", tmp_path / "noise_floor.json")
    assert a == sp.rows_for_capture(rows, nf)


def test_rows_for_capture_refuses_dry_run_rows_and_drops_contended_ones():
    rows = full_arm_rows("eg2-st-fp32", "st")
    dry = [dict(r, dry_run=True) for r in rows]
    for fn in (sp.rows_for_capture, sp.noise_floors_for_capture, sp.capture_gaps):
        with pytest.raises(ValueError, match="dry-run"):
            fn(dry, None)
    assert sp.capture_gaps(dry, None, allow_dry_run=True)  # not refused when explicitly allowed
    for r in rows:
        if r["leg"] == "tp_b32_t4":
            r["status"] = "contended"
    gaps = sp.capture_gaps(rows, None)
    assert "chunks_per_s" not in {
        r["metric"] for r in sp.rows_for_capture(rows, None)
    }  # contended never gets through
    assert gaps  # and without any A/A floor nothing is emitted, with the reason listed


def test_rows_and_floors_are_accepted_by_the_real_speed_writer(tmp_path):
    cap = pytest.importorskip("repl_retrieval_capture")
    rows, nf = capture_fixture()
    out = sp.rows_for_capture(rows, nf)
    floors = sp.noise_floors_for_capture(rows, nf)
    host = {
        "hostname": "epyc-test", "cpu_list": "24-71", "cpus_allowed_observed": "24-71",
        "region_lock_claim": "q1+q2:build:sess-1",
        "loadavg": {"start": [1.0, 1.2, 1.5], "end": [1.1, 1.2, 1.5]},
        "frequency": {"governor": "performance", "mhz_min": 3500.0, "mhz_max": 4200.0, "throttled": False},
    }  # fmt: skip
    digest = "0" * 63 + "1"
    stamps = iter(["2026-10-20T10:00:00+00:00", "2026-10-20T10:30:00+00:00"])
    run = cap.SpeedRun(tmp_path, "run-1", {"corpus": digest}, {"q200": digest}, digest, host, clock=lambda: next(stamps, "2026-10-20T10:30:00+00:00"))  # fmt: skip
    for r in out:
        run.add_speed(**r)  # exactly the eight add_speed arguments
    for f in floors:
        run.add_noise_floor(**f)
    sidecar = run.finalize()
    written = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(written) == len(out) and cap.validate_rows(written) == []
    assert {w["metric"] for w in written} == CAPTURE_METRICS
    assert all(w["samples"]["noise_floor"]["aa_spread"] >= 0 for w in written)


# ───────────────────────────────────────────────────────────────────────────── end to end (CLI dry run)


def test_dry_run_executes_the_whole_pipeline_on_fake_encoders(tmp_path, capsys):
    out = tmp_path / "dry"
    assert sp.main(["run", "--dry-run", "--out", str(out)]) == 0
    for name in (
        "speed.jsonl",
        "run_manifest.json",
        "noise_floor.json",
        "speed_summary.json",
        "speed_summary.md",
        "selection_costs.json",
    ):
        assert (out / name).exists(), name
    man = json.loads((out / "run_manifest.json").read_text())
    assert (
        man["dry_run"] is True
        and man["finished"] is True
        and man["claim_report"]["verified"] is False
    )
    rows = sp.load_rows(out / "speed.jsonl")
    assert {r["status"] for r in rows} == {"ok", "not_applicable"}
    assert all(r["dry_run"] is True and r["claim_tag"] == "DRY-RUN" for r in rows)
    # one measuring row per (arm, leg, pass) and the ABA order is in the log: A B C C B A
    seq = [
        r["arm_id"]
        for r in sorted(
            (r for r in rows if r["leg"] == "tp_b32_t4" and r["phase"] == "main"),
            key=lambda r: r["order_pos"],
        )
    ]
    assert seq == [
        "bge-large-st-fp32",
        "eg2-st-bf16",
        "eg2-st-fp32",
        "eg2-st-fp32",
        "eg2-st-bf16",
        "bge-large-st-fp32",
    ]
    # every required field is on a measuring row
    need = {"arm_identity_digest", "leg", "rep", "order_pos", "t_start", "t_end", "wall_s", "loadavg_before", "loadavg_after",
            "cpus_requested", "cpus_applied", "freq_khz", "claim_tag", "status", "contention"}  # fmt: skip
    for r in rows:
        if r["status"] == "ok" and r["leg_kind"] in ("throughput", "query", "index_build"):
            assert need <= set(r), (r["unit_key"], need - set(r))
    # both query sides exist for every arm, the wrong one as an explicit not_applicable row
    for a in ("bge-large-gguf-f16", "eg2-st-fp32", "potion-base-8m"):
        legs = {
            r["leg"]: r["status"] for r in rows if r["arm_id"] == a and r["leg_kind"] == "query"
        }
        assert set(legs) == {"q_inproc", "q_served"} and sorted(legs.values()) == [
            "not_applicable",
            "ok",
        ]
    summary = json.loads((out / "speed_summary.json").read_text())
    costs = json.loads((out / "selection_costs.json").read_text())
    assert set(costs) == set(sp.ROSTER_ORDER) and all(
        set(v) == {"cpu_s_per_query", "p95_ms"} for v in costs.values()
    )
    assert (
        summary["selection"]["legs"]["eg2-gguf-q8_0"] == "as_served"
        and summary["selection"]["legs"]["potion-base-8m"] == "in_process"
    )
    assert (
        summary["arm_aliases"]["eg2-gguf-bf16-256"] == "eg2-gguf-bf16"
    )  # @256 shares the forward pass
    assert (
        summary["arm_aliases"]["eg2-st-fp32-256"] == "eg2-st-fp32"
        and "eg2-st-bf16-256" not in summary["arm_aliases"]
    )
    assert "DRY RUN" in (out / "speed_summary.md").read_text()
    nf = json.loads((out / "noise_floor.json").read_text())
    assert nf["aa_arm"] == "bge-large-gguf-f16" and nf["metrics"]["chunks_per_s"]["n"] == 4
    # the capture adapter refuses fake numbers
    with pytest.raises(ValueError):
        sp.rows_for_capture(out / "speed.jsonl", nf)
    shaped = sp.rows_for_capture(out / "speed.jsonl", nf, allow_dry_run=True)
    fl = sp.noise_floors_for_capture(out / "speed.jsonl", nf, allow_dry_run=True)
    assert all(set(r) == set(sp.CAPTURE_KEYS) for r in shaped)
    # fake model dirs have no bytes (disk_mb is a gap), everything else is shaped for the writer
    assert {r["metric"] for r in shaped} >= CAPTURE_METRICS - {"disk_mb", "rss_mb"}
    assert {(r["arm_identity"]["arm_id"], r["metric"]) for r in shaped} == {
        (f["arm_identity"]["arm_id"], f["metric"]) for f in fl
    }
    # resumable through the CLI too, and `summarize` reproduces the same selection costs
    capsys.readouterr()
    assert sp.main(["run", "--dry-run", "--out", str(out), "--run-id", "other"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["executed"] == 0
    (out / "selection_costs.json").unlink()
    assert sp.main(["summarize", "--out", str(out)]) == 0
    assert set(json.loads((out / "selection_costs.json").read_text())) == set(sp.ROSTER_ORDER)


def test_prepare_freezes_inputs_and_run_refuses_a_tampered_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.json"
    corpus.write_text(
        "".join(
            json.dumps({"src": "f", "i": i, "text": f"chunk {i} text"}) + "\n" for i in range(50)
        )
    )
    queries.write_text(json.dumps([f"q{i}" for i in range(12)] + ["q1", "q2"]))
    out = tmp_path / "o"
    args = [
        "prepare",
        "--corpus",
        str(corpus),
        "--queries",
        str(queries),
        "--out",
        str(out),
        "--n-chunks",
        "40",
        "--min-queries",
        "10",
    ]
    assert sp.main(args) == 0
    meta = json.loads((out / "speed_inputs.json").read_text())
    assert meta["corpus"]["n_chunks"] == 40 and meta["queries"]["n"] == 12  # duplicates dropped
    assert meta["shipped_context_search"]["budget_s_at_4096"] == 60.0
    assert meta["shipped_context_search"]["constants"]["INDEX_ASSUMED_TEXTS_PER_S"] == 25.0
    c, q, im = sp.load_inputs(out)
    assert len(c) == 40 and len(q) == 12 and im["corpus_sha256"] == meta["corpus"]["sha256"]
    (out / "inputs" / "corpus.jsonl").write_text("tampered\n")
    with pytest.raises(sp.Refusal, match="changed since"):
        sp.load_inputs(out)
    assert (
        sp.main(
            [
                "prepare",
                "--corpus",
                str(corpus),
                "--queries",
                str(queries),
                "--out",
                str(tmp_path / "o2"),
                "--n-chunks",
                "9999",
            ]
        )
        == 64
    )
    assert (
        sp.main(
            [
                "prepare",
                "--corpus",
                str(corpus),
                "--queries",
                str(queries),
                "--out",
                str(tmp_path / "o3"),
                "--n-chunks",
                "40",
                "--min-queries",
                "13",
            ]
        )
        == 64
    )


def test_roster_matches_the_frozen_manifest_identities_for_the_arms_it_covers():
    r = sp.speed_roster(cs, "st", "ll", "m2v")
    assert list(r) == list(sp.ROSTER_ORDER) and len(r) == 9
    assert {a: e.kind for a, e in r.items()} == {
        "bge-large-gguf-f16": "gguf", "bge-large-st-fp32": "st", "granite-97m-r2-gguf-f16": "gguf",
        "eg2-gguf-bf16": "gguf", "eg2-gguf-q8_0": "gguf", "eg2-st-bf16": "st", "eg2-st-fp32": "st",
        "potion-code-16m-v2": "static", "potion-base-8m": "static",
    }  # fmt: skip
    base = {s.arm_id: s for s in arms.arm_specs("st", "ll", "m2v")}
    assert (
        r["eg2-st-fp32"].spec.index_identity()["digest"]
        == base["eg2-st-fp32-768"].index_identity()["digest"]
    )
    assert (
        r["eg2-gguf-q8_0"].spec.index_identity()["digest"]
        == base["eg2-gguf-q8_0-768"].index_identity()["digest"]
    )
    assert r["eg2-gguf-q8_0"].shares_forward_pass[0]["arm_id"] == "eg2-gguf-q8_0-256"
    assert r["eg2-st-fp32"].text_only and not r["bge-large-st-fp32"].text_only
    assert r["bge-large-gguf-f16"].spec.revision == f"sha256:{sp.BGE_GGUF_SHA256}"
    assert (
        len({e.spec.index_identity()["digest"] for e in r.values()}) == 9
    )  # no two arms share an index
    assert sp.ARM_KINDS == {a: e.kind for a, e in r.items()}
    ov = sp.speed_roster(
        cs, "st", "ll", "m2v", runtime_overrides={"bge-large-gguf-f16": "llama-server b10303"}
    )
    assert ov["bge-large-gguf-f16"].spec.runtime == "llama-server b10303"


def test_host_facts_survive_an_unreadable_sys_tree():
    f = sp.host_facts(FakeProc())
    assert f["governor"] is None and f["python"] and f["numpy"]


def test_run_manifest_records_plan_digest_and_every_arm(tmp_path):
    ids = ["bge-large-st-fp32", "potion-base-8m"]
    runner, _ = make_runner(tmp_path, ids, only=("rss", "disk"))
    runner.run()
    man = json.loads((tmp_path / "run_manifest.json").read_text())
    assert man["plan_sha256"] == runner.plan_sha256() and set(man["arms"]) == set(ids)
    assert man["arms"]["potion-base-8m"]["identity"]["digest"]


def test_rss_child_runner_spawns_an_isolated_interpreter_and_parses_its_json(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return SimpleNamespace(
            returncode=0,
            stdout='noise\n{"rss_after_load_kb": 123, "hwm_peak_kb": 456}\n',
            stderr="",
        )

    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    run = sp._rss_child_runner(
        None,
        ("st-rt", "ll-rt", "m2v-rt", None),
        {"eg2-st-fp32": "/m/eg2"},
        tmp_path / "c.jsonl",
        60.0,
    )
    assert run(roster()["eg2-st-fp32"]) == {"rss_after_load_kb": 123, "hwm_peak_kb": 456}
    cmd = seen["cmd"]
    assert cmd[0] == sys.executable and cmd[1] == "-I" and cmd[3] == "_rss-child"
    assert cmd[cmd.index("--arm-id") + 1] == "eg2-st-fp32"
    assert cmd[cmd.index("--model-path") + 1] == "/m/eg2" and seen["kw"]["timeout"] == 60.0

    monkeypatch.setattr(
        sp.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=3, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="rss child rc=3"):
        run(roster()["eg2-st-fp32"])


def test_rss_and_disk_units_for_in_process_and_served_arms(tmp_path):
    weights = tmp_path / "snap"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"w" * 1_000_000)
    gguf = tmp_path / "q8.gguf"
    gguf.write_bytes(b"g" * 2_000_000)
    fp = FakeProc()
    fp.add(900, comm="llama-server", cmdline=["llama-server", "-m", str(gguf), "-t", "16"])
    fp.procs[900].update(rss=111000, hwm=222000)
    r = roster()
    served = sp.FakeRuntime(r["eg2-gguf-q8_0"])
    served.server_pid = 900
    children = iter(
        [
            {"rss_baseline_kb": 1000, "rss_after_load_kb": 3000, "hwm_after_load_kb": 3500, "hwm_peak_kb": 5000, "load_s": 4.0},
            {"rss_baseline_kb": 1000, "rss_after_load_kb": 3400, "hwm_after_load_kb": 3900, "hwm_peak_kb": 5400, "load_s": 4.4},
        ]
    )  # fmt: skip
    asked = []
    runner, _ = make_runner(
        tmp_path / "out", [], proc=fp, dry=False, only=("rss", "disk"),
        runtimes={"bge-large-st-fp32": sp.FakeRuntime(r["bge-large-st-fp32"]), "eg2-gguf-q8_0": served},
        extra=dict(rss_child=lambda e: asked.append(e.arm_id) or next(children), model_paths={"bge-large-st-fp32": str(weights)}),
    )  # fmt: skip
    runner.run()
    assert (
        asked == ["bge-large-st-fp32"] * 2
    )  # an isolated child per in-process arm, twice; none for a server
    keys = [
        json.loads(x)["unit_key"]
        for x in (tmp_path / "out" / "speed.jsonl").read_text().splitlines()
    ]
    assert "static|rss|bge-large-st-fp32|p0" in keys and "static|rss|bge-large-st-fp32|p1" in keys
    assert (
        keys.count("static|rss|eg2-gguf-q8_0|p0") == 1 and "static|rss|eg2-gguf-q8_0|p1" not in keys
    )
    agg = sp.aggregate(sp.load_rows(tmp_path / "out" / "speed.jsonl"))["arms"]
    st, gg = agg["bge-large-st-fp32"], agg["eg2-gguf-q8_0"]
    kb = 1024 / 1e6  # /proc kB is KiB; the capture unit is decimal MB
    assert st["rss"]["rss_mb"] == pytest.approx(3200 * kb) and st["rss"]["n_rss"] == 2
    assert st["rss"]["rss_spread_mb"] == pytest.approx(400 * kb)
    assert st["rss"]["rss_delta_mb"] == pytest.approx(2200 * kb)
    assert st["rss"]["peak_hwm_mb"] == pytest.approx(5400 * kb)
    assert st["rss"]["source"] == "isolated_child_process"
    assert st["disk"]["disk_mb"] == pytest.approx(1.0) and st["disk"][
        "weights_mb"
    ] == pytest.approx(1.0)
    assert st["disk"]["repeat_spread_mb"] == 0.0
    assert gg["rss"]["source"] == "server_proc_status"
    assert gg["rss"]["rss_mb"] == pytest.approx(111000 * kb)
    assert gg["rss"]["peak_hwm_mb"] == pytest.approx(222000 * kb)
    assert gg["disk"]["disk_mb"] == pytest.approx(2.0)  # path came from the server's own cmdline
    assert gg["disk"]["sha256"] == sp.sha256_file(gguf)


def test_served_arm_rss_is_sampled_across_the_run_for_its_spread(tmp_path):
    fp = FakeProc()
    fp.add(900, comm="llama-server")
    fp.procs[900].update(rss=100000, hwm=100000)
    world = World(fp, [0, 1, 2, 3], own_ticks=100)

    class Growing(World):
        def step(self):
            super().step()
            fp.procs[900]["rss"] += 500  # the server's RSS creeps up while it works

    g = Growing(fp, [0, 1, 2, 3], own_ticks=100)
    rt = TickingRuntime(roster()["eg2-gguf-q8_0"], g)
    rt.server_pid = 900
    runner, _ = make_runner(
        tmp_path,
        [],
        proc=fp,
        world=g,
        runtimes={"eg2-gguf-q8_0": rt},
        only=("rss", "throughput"),
        batches=(32,),
    )
    runner.run()
    rs = sp.aggregate(sp.load_rows(tmp_path / "speed.jsonl"))["arms"]["eg2-gguf-q8_0"]["rss"]
    assert (
        rs["n_rss"] > 3 and rs["rss_spread_mb"] > 0
    )  # start reading plus before/after every timed window
    assert rs["rss_mb"] >= 100000 * 1024 / 1e6
    del world
