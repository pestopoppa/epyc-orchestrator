"""Host-wide KV-pool token reservations and queue bound (multi-worker).

``uvicorn --workers 6`` gives six ``SharedKVPoolAdmission`` instances. Their in-flight
reserved tokens and queue depth must be summed host-wide: total admitted never exceeds the
pool, the queue limit is not multiplied by the worker count, a dead worker's reservation is
reaped, and an unavailable store admits nothing (fail closed).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time

import pytest

from src.runtime import xproc_reservations as xr
from src.scheduling import kv_pool_admission as kpa
from src.scheduling.kv_pool_admission import KVPoolQueueFull, SharedKVPoolAdmission

URL = "http://127.0.0.1:59991"
POOL = 1000
WANT = 400          # two fit in POOL, a third does not


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(kpa.KV_POOL_LONG_PREFILL_ENV, "0")
    monkeypatch.delenv(kpa.KV_POOL_MAX_QUEUED_ENV, raising=False)
    monkeypatch.delenv(kpa.KV_POOL_CROSS_PROCESS_RESERVE_ENV, raising=False)
    monkeypatch.setenv(kpa.KV_POOL_CROSS_PROCESS_LEASE_ENV, "0")


def _adm() -> SharedKVPoolAdmission:
    return SharedKVPoolAdmission(occupancy=lambda url: None, cross_process=False)


def _host_totals():
    with kpa._XP_STORE.locked(URL) as slate:
        return slate.foreign_total(kpa._XP_INFLIGHT), slate.foreign_total(kpa._XP_QUEUE)


def _admitter(q, go, done, timeout_s):
    adm = _adm()
    go.wait(10)
    ticket = adm.acquire(URL, WANT, POOL, timeout_s=timeout_s, max_queued=0)
    q.put(WANT if ticket is not None else 0)
    done.wait(30)
    adm.release(URL, ticket)


def test_total_admitted_across_processes_never_exceeds_pool():
    n_procs = 8
    ctx = mp.get_context("fork")
    q, go, done = ctx.Queue(), ctx.Event(), ctx.Event()
    procs = [ctx.Process(target=_admitter, args=(q, go, done, 1.0)) for _ in range(n_procs)]
    for p in procs:
        p.start()
    go.set()
    admitted = [q.get(timeout=30) for _ in procs]
    assert sum(admitted) <= POOL, admitted
    assert sum(admitted) == 2 * WANT, admitted       # the free tokens are used, not wasted
    assert _host_totals()[0] == 2 * WANT
    done.set()
    for p in procs:
        p.join(30)
        assert p.exitcode == 0
    assert _host_totals() == (0, 0)                   # every release returned its tokens


def test_releasing_in_one_process_unblocks_a_waiter_in_another():
    ctx = mp.get_context("fork")
    q, go, done = ctx.Queue(), ctx.Event(), ctx.Event()
    holders = [ctx.Process(target=_admitter, args=(q, go, done, 1.0)) for _ in range(2)]
    for p in holders:
        p.start()
    go.set()
    assert [q.get(timeout=30) for _ in holders] == [WANT, WANT]
    adm = _adm()
    assert adm.acquire(URL, WANT, POOL, timeout_s=0.5, max_queued=0) is None   # pool full
    done.set()
    for p in holders:
        p.join(30)
    ticket = adm.acquire(URL, WANT, POOL, timeout_s=2.0, max_queued=0)
    assert ticket is not None
    adm.release(URL, ticket)


def _queue_filler(q, go, max_queued):
    adm = _adm()
    go.wait(10)
    try:
        t = adm.acquire(URL, WANT, POOL, timeout_s=3.0, max_queued=max_queued)
        q.put("timeout" if t is None else "admitted")
    except KVPoolQueueFull:
        q.put("full")


def test_queue_bound_is_host_wide_not_per_worker():
    holder = _adm()
    t1 = holder.acquire(URL, POOL, POOL, timeout_s=1.0, max_queued=0)   # pool exhausted
    assert t1 is not None
    max_queued, n_procs = 3, 9
    ctx = mp.get_context("fork")
    q, go = ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=_queue_filler, args=(q, go, max_queued)) for _ in range(n_procs)]
    for p in procs:
        p.start()
    go.set()
    outcomes = [q.get(timeout=30) for _ in procs]
    for p in procs:
        p.join(30)
        assert p.exitcode == 0
    assert outcomes.count("full") == n_procs - max_queued, outcomes   # not x workers
    assert outcomes.count("timeout") == max_queued, outcomes
    assert "admitted" not in outcomes
    holder.release(URL, t1)
    assert _host_totals() == (0, 0)


def _leaky(go_dir_env):
    adm = _adm()
    ticket = adm.acquire(URL, POOL, POOL, timeout_s=1.0, max_queued=0)
    os._exit(0 if ticket is not None else 3)         # dies holding the reservation


def test_dead_owner_reservation_is_reaped():
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_leaky, args=(None,))
    p.start()
    p.join(30)
    assert p.exitcode == 0
    # The corpse's entry is on disk, but its owner is gone: reaped before the fit check.
    adm = _adm()
    ticket = adm.acquire(URL, POOL, POOL, timeout_s=2.0, max_queued=0)
    assert ticket is not None
    adm.release(URL, ticket)
    assert _host_totals() == (0, 0)


def test_pid_reuse_entry_is_reaped(monkeypatch):
    with kpa._XP_STORE.locked(URL) as slate:        # a live pid with a different start time
        slate.entries.append({"id": "x", "pid": os.getppid(), "start": "0", "t": time.time(),
                              "n": POOL, "tag": kpa._XP_INFLIGHT})
        slate.dirty = True
    adm = _adm()
    ticket = adm.acquire(URL, WANT, POOL, timeout_s=1.0, max_queued=0)
    assert ticket is not None
    adm.release(URL, ticket)


def test_store_unavailable_fails_closed_and_logs(monkeypatch, tmp_path, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setenv(kpa.KV_POOL_RESERVE_DIR_ENV, str(blocker / "sub"))   # mkdir under a file
    adm = _adm()
    with caplog.at_level(logging.WARNING, logger=kpa.logger.name):
        ticket = adm.acquire(URL, 10, POOL, timeout_s=0.4, max_queued=0)
    assert ticket is None                             # nothing admitted
    assert adm.in_flight_tokens(URL) == 0
    assert any("failing closed" in r.getMessage() for r in caplog.records)


def test_store_flock_timeout_fails_closed(monkeypatch):
    import fcntl

    d = kpa._XP_STORE.reserve_dir()
    with kpa._XP_STORE.locked(URL):
        pass                                          # create the file
    path = next(d.glob("*.json"))
    fd = os.open(path, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)                    # another "worker" sits on the flock
    try:
        monkeypatch.setattr(kpa._XP_STORE, "lock_timeout_s", 0.05)
        adm = _adm()
        assert adm.acquire(URL, 10, POOL, timeout_s=0.3, max_queued=0) is None
        assert adm.in_flight_tokens(URL) == 0
    finally:
        os.close(fd)


def test_corrupt_store_file_fails_closed():
    with kpa._XP_STORE.locked(URL):
        pass
    path = next(kpa._XP_STORE.reserve_dir().glob("*.json"))
    path.write_text("{not json")
    adm = _adm()
    assert adm.acquire(URL, 10, POOL, timeout_s=0.3, max_queued=0) is None


def test_off_switch_keeps_per_process_behaviour(monkeypatch):
    monkeypatch.setenv(kpa.KV_POOL_CROSS_PROCESS_RESERVE_ENV, "0")
    monkeypatch.setenv(kpa.KV_POOL_RESERVE_DIR_ENV, "/proc/definitely/not/writable")
    adm = _adm()
    t = adm.acquire(URL, WANT, POOL, timeout_s=0.5, max_queued=0)
    assert t is not None
    adm.release(URL, t)


def test_primitive_waits_with_backoff_not_a_spin(monkeypatch):
    import fcntl
    import time

    with xr.ReservationStore("t", dir_env="EPYC_KVPOOL_RESERVE_DIR", ttl_env="X").locked(URL):
        pass
    store = xr.ReservationStore("t", dir_env="EPYC_KVPOOL_RESERVE_DIR", ttl_env="X")
    path = next(store.reserve_dir().glob("*.json"))
    fd = os.open(path, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    calls = {"n": 0}
    real = fcntl.flock

    def counting(f, op):
        calls["n"] += 1
        return real(f, op)

    monkeypatch.setattr(xr.fcntl, "flock", counting)
    t0 = time.monotonic()
    try:
        with pytest.raises(xr.ReservationUnavailable):
            with store.locked(URL, timeout=0.5):
                pass
    finally:
        os.close(fd)
    assert time.monotonic() - t0 >= 0.5
    assert calls["n"] < 40, calls            # a 10 ms fixed poll would be ~50; backoff is far fewer
