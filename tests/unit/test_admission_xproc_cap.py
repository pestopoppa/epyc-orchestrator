"""Multi-worker audit: the per-backend execution cap holds across processes.

``AdmissionController`` is per process (``uvicorn --workers 6`` -> up to 6n requests past
the gate for an n-slot backend). The cap on concurrent inference per serving instance is the
downstream host-wide flock (`cpu_region_lock`, exclusive per region), taken by every dispatch
after admission. This proves it: many processes, each with its own AdmissionController that
happily admits them all, never run more than one holder of the same region at a time.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from src.api.admission import AdmissionController

URL = "http://localhost:59992"


@pytest.fixture(autouse=True)
def _lock_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TMP_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_INFERENCE_LOCK_POLL_MS", "5")


def _worker(live, peak, admitted, go, hold_s):
    from src.runtime.cpu_region_lock import cpu_region_lock

    ctl = AdmissionController({URL: 2})            # this process's own gate
    go.wait(10)
    ok = ctl.acquire(URL)
    with admitted.get_lock():
        admitted.value += 1 if ok else 0
    try:
        with cpu_region_lock("frontdoor", {"q0"}, timeout_s=60):
            with live.get_lock():
                live.value += 1
                peak.value = max(peak.value, live.value)
            time.sleep(hold_s)
            with live.get_lock():
                live.value -= 1
    finally:
        ctl.release(URL)


def test_region_flock_caps_concurrency_across_processes_despite_per_process_gates():
    n_procs = 6
    ctx = mp.get_context("fork")
    live, peak, admitted = ctx.Value("i", 0), ctx.Value("i", 0), ctx.Value("i", 0)
    go = ctx.Event()
    procs = [ctx.Process(target=_worker, args=(live, peak, admitted, go, 0.05))
             for _ in range(n_procs)]
    for p in procs:
        p.start()
    go.set()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert admitted.value == n_procs     # every worker's own gate admitted its request ...
    assert peak.value == 1               # ... yet one region never had two holders
    assert live.value == 0


def test_within_one_process_the_gate_still_caps_at_n():
    ctl = AdmissionController({URL: 2})
    assert ctl.acquire(URL) and ctl.acquire(URL)
    assert ctl.acquire(URL) is False
    ctl.release(URL)
    assert ctl.acquire(URL)
