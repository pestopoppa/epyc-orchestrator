"""P1-8 (INF-45): region_lock_wait_s_by_holder is observation-only telemetry."""
import fcntl
import json
import threading
import time

import pytest

import src.runtime.cpu_region_lock as crl


@pytest.fixture(autouse=True)
def _reset():
    crl.reset_region_lock_wait_stats_for_tests()
    yield
    crl.reset_region_lock_wait_stats_for_tests()


def _held(tmp_path, role):
    path = tmp_path / "q0.lock"
    holder = open(path, "w+b")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    holder.write(json.dumps({"role": role}).encode())
    holder.flush()
    return path, holder


def _waiter(path):
    return open(path, "r+b")


def test_wait_attributed_to_holder_role(tmp_path):
    path, holder = _held(tmp_path, "frontdoor")
    threading.Timer(0.3, holder.close).start()
    fh = _waiter(path)
    waited = crl._acquire_one_with_timeout(
        fh, region="q0", role="coder", timeout_s=10, deadline_s=None,
        cancel_check=None, request_tag="t",
    )
    snap = crl.region_lock_wait_s_by_holder()
    assert waited >= 0.25
    assert snap["frontdoor"]["waits"] == 1.0
    assert snap["frontdoor"]["wait_s"] >= 0.25
    fh.close()


def test_timeout_still_recorded_and_raised(tmp_path):
    path, holder = _held(tmp_path, "worker")
    fh = _waiter(path)
    with pytest.raises(crl.CpuRegionLockTimeout):
        crl._acquire_one_with_timeout(
            fh, region="q0", role="coder", timeout_s=0.2, deadline_s=None,
            cancel_check=None, request_tag="t",
        )
    assert crl.region_lock_wait_s_by_holder()["worker"]["wait_s"] >= 0.15
    holder.close(); fh.close()


def test_uncontended_acquire_records_nothing(tmp_path):
    fh = open(tmp_path / "free.lock", "w+b")
    crl._acquire_one_with_timeout(
        fh, region="q0", role="coder", timeout_s=5, deadline_s=None,
        cancel_check=None, request_tag="t",
    )
    assert crl.region_lock_wait_s_by_holder() == {}
    fh.close()
