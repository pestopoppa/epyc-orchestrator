"""Cross-process scout slot reservation: contention, crash reaping, release, fail-safe."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

from src.api.routes.chat_pipeline import scout_reservations as R
from src.api.routes.chat_pipeline import scout_stage as ss

URL = "http://127.0.0.1:59990"


class _Resolver:
    def __init__(self, total, processing=0):
        self.occ = SimpleNamespace(slots=list(range(total)), processing=processing)

    def pool_occupancy(self, url):
        return self.occ


def _cap(resolver, hold, *, max_scouts=1, url=URL):
    return ss.resolve_cap(url, max_scouts=max_scouts, n_targets=8, reserve_slots=1,
                          resolver=resolver, hold=hold)


def _inflight(url=URL):
    with R.locked(url) as slate:
        return slate.inflight()


def _contender(total, q, go, done):
    hold: dict = {}
    go.wait(10)
    out = _cap(_Resolver(total), hold)
    q.put(out["cap"])
    done.wait(20)                       # keep the reservation until everyone has decided
    ss._release_inflight(hold.get("url"), hold.get("token"))


def test_n_processes_contending_never_exceed_the_cap():
    n_procs, k = 10, 3                   # 4 slots, reserve 1 -> cap 3 in total
    ctx = mp.get_context("fork")
    q, go, done = ctx.Queue(), ctx.Event(), ctx.Event()
    procs = [ctx.Process(target=_contender, args=(k + 1, q, go, done)) for _ in range(n_procs)]
    for p in procs:
        p.start()
    go.set()
    caps = [q.get(timeout=20) for _ in procs]
    assert sum(caps) == k, caps          # never more than K, and the free slots are used
    assert _inflight() == k
    done.set()
    for p in procs:
        p.join(20)
        assert p.exitcode == 0
    assert _inflight() == 0              # every exit released


def _die_holding(url):
    out = _cap(_Resolver(6), {}, url=url)
    assert out["cap"] == 1
    os._exit(0)                          # crash: no release


def test_dead_owner_is_reaped():
    p = mp.get_context("fork").Process(target=_die_holding, args=(URL,))
    p.start()
    p.join(20)
    assert p.exitcode == 0
    assert _inflight() == 0              # reaped on the next locked read
    assert _cap(_Resolver(6), {})["cap"] == 1


def _plant(entry):
    path = R.reserve_dir() / (R.hashlib.sha1(URL.encode()).hexdigest()[:20] + ".json")
    R.reserve_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": URL, "entries": [entry]}))


def test_pid_reuse_and_ttl_are_reaped(monkeypatch):
    me = os.getpid()
    base = {"id": "x", "pid": me, "t": time.time(), "n": 4}
    _plant({**base, "start": "1"})       # live pid, but a different process incarnation
    assert _inflight() == 0
    _plant({**base, "start": R.proc_start_ticks(me)})
    assert _inflight() == 4              # genuinely live owner is honoured
    monkeypatch.setenv(R.TTL_ENV, "0.01")
    time.sleep(0.05)
    assert _inflight() == 0              # TTL bound


def test_release_on_cancel_flag_and_token():
    hold: dict = {}
    assert _cap(_Resolver(6), hold, max_scouts=4)["cap"] == 4
    assert _inflight() == 4
    ss._release_inflight(hold["url"], hold["token"])
    assert _inflight() == 0
    ss._release_inflight(hold["url"], hold["token"])      # idempotent
    late = {"released": True}
    out = _cap(_Resolver(6), late)
    assert out["cap"] == 0 and out["source"] == "released" and _inflight() == 0


def test_fail_safe_when_store_unavailable(monkeypatch, tmp_path, caplog):
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    monkeypatch.setenv(R.DIR_ENV, str(blocker / "sub"))   # mkdir under a file -> OSError
    hold: dict = {}
    with caplog.at_level("WARNING"):
        out = _cap(_Resolver(6), hold)
    assert out["cap"] == 0 and out["source"] == "reserve_unavailable"
    assert "token" not in hold and "admitting 0" in caplog.text
    ss._release_inflight(URL, "t")                         # must not raise


def test_fail_safe_on_lock_timeout_and_corrupt_file(monkeypatch):
    import fcntl

    d = R.reserve_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / (R.hashlib.sha1(URL.encode()).hexdigest()[:20] + ".json")
    path.write_text("{not json")
    assert _cap(_Resolver(6), {})["source"] == "reserve_unavailable"
    path.write_text("")
    fd = os.open(path, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)                         # another holder
    try:
        t0 = time.monotonic()
        try:
            with R.locked(URL, timeout=0.1):
                raise AssertionError("should not acquire")
        except R.ReservationUnavailable:
            pass
        assert time.monotonic() - t0 < 2
    finally:
        os.close(fd)
