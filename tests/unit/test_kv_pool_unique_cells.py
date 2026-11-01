"""KPF-24 native attention-cell admission. No servers or model calls."""

from __future__ import annotations

import threading

import httpx
import pytest

from src.backends.context_limits import (
    ContextLimitResolver,
    PoolOccupancy,
    SlotState,
    parse_kv_cell_metrics,
)
from src.inference import prefix_index as pi
from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

URL = "http://private-fixture:18083"


def metrics(used=60, size=100):
    return f"# TYPE llamacpp:kv_cells_used gauge\nllamacpp:kv_cells_used {used}\nllamacpp:kv_cells_size {size}\n"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv("ORCHESTRATOR_KV_POOL_LONG_PREFILL_TOKENS", "0")


@pytest.mark.parametrize("used", [0, 60, 100])
def test_native_counter_is_integer_unique_cells_including_measured_zero(used):
    assert parse_kv_cell_metrics(metrics(used)) == {"size": 100, "used": used}


def test_integer_cells_do_not_round_through_float():
    used = 9007199254740993
    assert parse_kv_cell_metrics(metrics(used, used + 1)) == {"size": used + 1, "used": used}


@pytest.mark.parametrize("body", [
    "", "llamacpp:kv_cells_logical 90\n", metrics("NaN"), metrics("Inf"),
    metrics("-1"), metrics("1.5"), metrics(101), metrics(0, 0),
    metrics(0, -1), metrics("nope"), metrics() + "llamacpp:kv_cells_used 60\n",
    metrics().replace("kv_cells_used 60", 'kv_cells_used{model="other"} 60'),
    metrics().replace("kv_cells_used 60", "kv_cells_used 60 123456"), None,
])
def test_missing_off_arm_or_malformed_is_unknown_not_zero(body):
    assert parse_kv_cell_metrics(body) is None


def slot(i, n, *, busy=True, remain=0):
    return SlotState(i, 100, busy, n, remain)


def test_native_used_includes_idle_and_unattributed_cells_not_logical_sum():
    occ = PoolOccupancy(URL, (slot(0, 40), slot(1, 40, remain=10), slot(2, 7, busy=False)),
                        kv_metrics={"size": 100, "used": 61})
    assert occ.projected_tokens() == 90
    assert occ.projected_unique_tokens() == 71  # actual 61, plus 10 pending decode
    assert occ.projected_unique_tokens(0.3) == 64


def test_consecutive_native_snapshots_keep_higher_usage_and_reject_capacity_drift():
    occ = PoolOccupancy(URL, (), kv_pool={"size": 100, "used": 70},
                        kv_metrics={"size": 100, "used": 60})
    assert occ.unique_cells == 70
    mismatch = PoolOccupancy(URL, (slot(0, 80),), kv_pool=occ.kv_pool,
                             kv_metrics={"size": 200, "used": 60})
    assert mismatch.unique_cells is None
    assert mismatch.projected_unique_tokens() == 80
    assert mismatch.unique_cell_capacity == 0
    pool = SharedKVPoolAdmission(occupancy=lambda u: mismatch, cross_process=False)
    assert pool.acquire(URL, 1, 200, timeout_s=0) is None
    # Even zero processing slots cannot fall into the legacy lone-request rule.
    idle_drift = PoolOccupancy(URL, (), kv_pool=occ.kv_pool,
                              kv_metrics={"size": 200, "used": 0})
    pool = SharedKVPoolAdmission(occupancy=lambda u: idle_drift, cross_process=False)
    assert pool.acquire(URL, 1, 200, timeout_s=0) is None


def test_resolver_binds_metrics_to_same_endpoint_refresh_and_drops_stale_success():
    clock = [0.0]
    calls = []
    body = [metrics()]

    def fetch(endpoint, timeout):
        calls.append((endpoint, timeout))
        return body[0]

    resolver = ContextLimitResolver(live=True, clock=lambda: clock[0],
                                    fetch_slots=lambda u, t: [], fetch_metrics=fetch,
                                    occupancy_ttl_s=1.5)
    assert resolver.pool_occupancy(URL).unique_cells == 60
    body[0] = ""  # off arm / producer disappears; old success must expire
    assert resolver.pool_occupancy(URL).unique_cells == 60
    assert calls == [(URL, 1.0)]
    clock[0] = 1.6
    assert resolver.pool_occupancy(URL).unique_cells is None
    assert resolver.pool_occupancy(URL + "1").unique_cells is None
    assert calls == [(URL, 1.0), (URL, 1.0), (URL + "1", 1.0)]


def test_metrics_failure_and_flag_off_preserve_slots_fallback(monkeypatch):
    def failed(*args):
        raise OSError("private fixture unavailable")

    resolver = ContextLimitResolver(live=True, fetch_slots=lambda u, t: [], fetch_metrics=failed)
    assert resolver.pool_occupancy(URL).unique_cells is None
    monkeypatch.setenv(pi.FLAG_ENV, "0")
    resolver = ContextLimitResolver(live=True, fetch_slots=lambda u, t: [],
                                    fetch_metrics=lambda *a: pytest.fail("flag off read metrics"))
    assert resolver.pool_occupancy(URL).unique_cells is None


def test_capacity_drift_refuses_until_a_consistent_refresh():
    clock = [0.0]
    body = [metrics(0, 200)]
    resolver = ContextLimitResolver(live=True, clock=lambda: clock[0],
                                    fetch_slots=lambda u, t: [{
                                        "id": 0, "is_processing": False,
                                        "kv_pool": {"size": 100, "used": 0}}],
                                    fetch_metrics=lambda u, t: body[0])
    pool = SharedKVPoolAdmission(occupancy=resolver.pool_occupancy, cross_process=False)
    assert pool.acquire(URL, 1, 200, timeout_s=0) is None
    body[0] = metrics(0, 100)
    assert pool.acquire(URL, 1, 200, timeout_s=0) is None  # drift snapshot still cached
    clock[0] = 1.6
    assert pool.acquire(URL, 1, 200, timeout_s=0) is not None


def test_default_readers_use_native_metrics_on_the_selected_server_only(monkeypatch):
    calls = []

    def get(endpoint, *, timeout):
        calls.append(endpoint)
        request = httpx.Request("GET", endpoint)
        if endpoint.endswith("/slots"):
            return httpx.Response(200, json=[], request=request)
        assert endpoint == URL + "/metrics"
        return httpx.Response(200, text=metrics(), request=request)

    monkeypatch.setattr(httpx, "get", get)
    resolver = ContextLimitResolver(live=True)
    assert resolver.pool_occupancy(URL).unique_cells == 60
    assert calls == [URL + "/slots", URL + "/metrics"]
    assert ContextLimitResolver(live=False).pool_occupancy(URL) is None
    assert calls == [URL + "/slots", URL + "/metrics"]


def test_unique_native_snapshot_plus_pending_tickets_does_not_hide_external_load():
    occ = PoolOccupancy(URL, (), kv_metrics={"size": 100, "used": 60})
    pool = SharedKVPoolAdmission(occupancy=lambda u: occ, cross_process=False)
    first = pool.acquire(URL, 30, 100, timeout_s=0)
    assert first is not None
    # This ticket has not reached the server. max(own=30, used=60)+20
    # would admit 110 cells into a 100-cell pool.
    assert pool.acquire(URL, 20, 100, timeout_s=0) is None
    pool.release(URL, first)
    assert pool.acquire(URL, 20, 100, timeout_s=0) is not None


def test_native_idle_usage_and_native_capacity_are_binding():
    occ = PoolOccupancy(URL, (slot(0, 90, busy=False),),
                        kv_metrics={"size": 100, "used": 90})
    pool = SharedKVPoolAdmission(occupancy=lambda u: occ, cross_process=False)
    assert pool.acquire(URL, 11, 200, timeout_s=0) is None
    assert pool.acquire(URL, 10, 200, timeout_s=0) is not None


def test_measured_zero_counts_outstanding_reservations_and_no_counter_keeps_legacy():
    occ = PoolOccupancy(URL, (), kv_metrics={"size": 100, "used": 0})
    pool = SharedKVPoolAdmission(occupancy=lambda u: occ, cross_process=False)
    assert pool.acquire(URL, 60, 100, timeout_s=0) is not None
    assert pool.acquire(URL, 41, 100, timeout_s=0) is None
    legacy = SharedKVPoolAdmission(occupancy=lambda u: PoolOccupancy(u, ()), cross_process=False)
    assert legacy.acquire(URL, 200, 100, timeout_s=0) is not None  # legacy lone-request rule


def test_concurrent_acquires_cannot_spend_same_headroom_and_release_wakes_waiter():
    occ = PoolOccupancy(URL, (), kv_metrics={"size": 100, "used": 60})
    pool = SharedKVPoolAdmission(occupancy=lambda u: occ, cross_process=False)
    barrier = threading.Barrier(3)
    results = []

    def acquire():
        barrier.wait(timeout=5)
        results.append(pool.acquire(URL, 30, 100, timeout_s=0))

    threads = [threading.Thread(target=acquire) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert sum(ticket is not None for ticket in results) == 1
    winner = next(ticket for ticket in results if ticket is not None)
    waiting = threading.Event()
    admitted = []

    def waiter():
        waiting.set()
        admitted.append(pool.acquire(URL, 30, 100, timeout_s=5, poll_s=0.01))

    thread = threading.Thread(target=waiter)
    thread.start()
    assert waiting.wait(timeout=5)
    pool.release(URL, winner)
    thread.join(timeout=5)
    assert not thread.is_alive() and admitted[0] is not None
    pool.release(URL, admitted[0])
    assert pool.in_flight_tokens(URL) == 0
