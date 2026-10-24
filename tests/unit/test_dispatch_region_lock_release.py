"""_dispatch releases the region lock it took even when post-acquisition setup raises (A2-13).

The per-region path entered the lock by hand and only exited it in the finally around
the yield, so an exception from the migration kick-off (MigrationTransaction, a
Thread that cannot start) left the flock to the GC and the in-process active flag stuck.
"""

from __future__ import annotations

import threading

import pytest

import src.backends.concurrency_aware as ca_mod


class _Stub:
    def __init__(self, url: str):
        self.config = type("C", (), {"base_url": url})()


class _RecordingLock:
    def __init__(self, log: list[str]):
        self.log = log

    def __enter__(self):
        self.log.append("enter")
        return ["/tmp/mock.lock"]

    def __exit__(self, *exc):
        self.log.append("exit")
        return False


@pytest.fixture
def backend(monkeypatch):
    log: list[str] = []
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setattr(
        "src.runtime.cpu_region_lock.cpu_region_lock_for_instance",
        lambda role, idx, **kw: _RecordingLock(log),
    )
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", lambda: {})
    monkeypatch.setattr("src.runtime.instance_topology.get_instance_regions", lambda: {})
    cab = ca_mod.ConcurrencyAwareBackend(
        full_backend=_Stub("http://localhost:8070"),
        quarter_backends=[_Stub(f"http://localhost:80{80 + i * 100}") for i in range(4)],
        role="frontdoor",
        full_port=8070,
    )
    cab._full_last_session = "old-sess"  # next full holder triggers a migration kick-off
    return cab, log


def _boom(*_a, **_kw):
    raise RuntimeError("can't start new thread")


@pytest.mark.parametrize(
    "target",
    ["threading.Thread.start", "src.scheduling.migration_transaction.MigrationTransaction"],
)
def test_lock_released_when_migration_kickoff_raises(backend, monkeypatch, target):
    cab, log = backend
    monkeypatch.setattr(target, _boom)

    with pytest.raises(RuntimeError):
        with cab._dispatch(session_id="new-sess"):
            pytest.fail("must not reach the caller")

    assert log == ["enter", "exit"]
    assert cab._full_active is False


def test_success_path_still_enters_and_exits_once(backend, monkeypatch):
    cab, log = backend
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)

    with cab._dispatch(session_id="new-sess") as (_backend, idx, is_full):
        assert (idx, is_full) == (-1, True)
        assert log == ["enter"]

    assert log == ["enter", "exit"]
    assert cab._full_active is False
