"""OAB-17 review fixes: F8 cancel-leak, F9 claim/cap pool, F15 mode prompt, F16 bundle.

Offline and inference-free; reuses the OAB-8 scout fixtures.
"""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.api.models import ChatRequest
from src.api.routes.chat_pipeline import scout_stage as S
from src.repl_environment.context import _ContextMixin
from src.repl_environment.context_bundle import (
    MAX_BUNDLE_BYTES,
    ContextBundle,
    ContextPullBudgetExceeded,
    Section,
)
from tests.unit.test_oab8_scouts import (  # noqa: F401  (fixtures + helpers)
    FakeResolver,
    ScriptedTransport,
    _no_leaked_scope,
    _run_handle_chat,
    _spec,
    _summary,
    lane,
    pipeline,
)


# ── F8: a cancel while resolve_cap is still in its pool thread must not leak ─────────────


class _BlockingResolver:
    """`pool_occupancy` parks until released, like a slow /slots read."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def pool_occupancy(self, url):
        self.entered.set()
        assert self.release.wait(10)
        return FakeResolver(total=6, busy=0).pool_occupancy(url)


def test_cancel_mid_read_does_not_leak_the_reservation(lane, monkeypatch):
    url = "http://127.0.0.1:59871"
    resolver = _BlockingResolver()
    finished = threading.Event()
    decisions: list[dict] = []
    real = S.resolve_cap

    def wrapped(*a, **kw):
        try:
            out = real(*a, **kw)
            decisions.append(out)
            return out
        finally:
            finished.set()

    monkeypatch.setattr(S, "resolve_cap", wrapped)

    async def go():
        task = asyncio.ensure_future(S.run_scouts(
            _spec([{"symbol": "helper", "label": "T"}]), scope=lane.scope, role="frontdoor",
            url=url, transport=ScriptedTransport({"T": [_summary("t")]}), resolver=resolver))
        for _ in range(500):                       # the read is in flight in its pool thread
            if resolver.entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert resolver.entered.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        resolver.release.set()                     # the thread now finishes AFTER the cancel
        await asyncio.get_running_loop().run_in_executor(None, finished.wait, 10)

    try:
        asyncio.run(go())
        assert finished.is_set()
        assert decisions and decisions[0]["cap"] == 0 and decisions[0]["source"] == "released"
        assert not S._INFLIGHT.get(url), "a cancelled request must not leave slots reserved"
    finally:
        S._INFLIGHT.pop(url, None)


def test_resolve_cap_refuses_once_released():
    url = "http://127.0.0.1:59872"
    hold = {"released": True}
    out = S.resolve_cap(url, max_scouts=8, n_targets=8, reserve_slots=1,
                        resolver=FakeResolver(total=6, busy=0), hold=hold)
    assert out["cap"] == 0 and not S._INFLIGHT.get(url) and "url" not in hold


# ── F9: claim acquire and resolve_cap do not share the (full) scout pool ─────────────────


def test_full_scout_pool_does_not_block_the_claim(lane, monkeypatch):
    from src.runtime import direct_region_claim as drc

    entered_on: list[str] = []
    acquired = threading.Event()

    class _Claim:
        def __enter__(self):
            entered_on.append(threading.current_thread().name)
            acquired.set()

        def __exit__(self, *exc):
            return False

    class _Target:
        def as_dict(self):
            return {"instance": "fake"}

    monkeypatch.setattr(drc, "resolve_claim_target", lambda *a, **k: _Target())
    monkeypatch.setattr(drc, "region_claim", lambda *a, **k: _Claim())
    block = threading.Event()

    async def go():
        for _ in range(S.SCOUT_EXECUTOR_WORKERS):       # the scout pool is completely full
            S._scout_thread(block.wait, 20)
        task = asyncio.ensure_future(S.run_scouts(
            _spec([{"symbol": "helper", "label": "T"}]), scope=lane.scope, role="frontdoor",
            url="http://x:1", transport=ScriptedTransport({"T": [_summary("t")]}),
            resolver=FakeResolver()))
        got = False
        for _ in range(300):
            if acquired.is_set():
                got = True
                break
            await asyncio.sleep(0.01)
        block.set()                                      # free the scout pool for the scouts
        stage = await asyncio.wait_for(task, 20)
        return got, stage

    got, stage = asyncio.run(go())
    assert got, "the claim must be acquired while every scout thread is busy"
    assert entered_on and not entered_on[0].startswith("scout_") and entered_on[0].startswith("scoutctl")
    assert stage.report["region_claim"]["status"] == "held"
    assert stage.report["completed"] == 1


# ── F15: mode selection judges the caller's prompt, not the scout block ──────────────────


def _scouted(lane, prompt):
    return ChatRequest(prompt=prompt, mock_mode=False, real_mode=True, task_root=str(lane.root),
                       quiescent_after=True,
                       scouts={"enabled": True, "targets": [{"symbol": "helper", "label": "T"}]})


def test_select_mode_receives_the_original_short_question(lane, pipeline):
    seen: dict = {}

    def recorder(prompt, context, state):
        seen["prompt"] = prompt
        return "repl"

    req = _scouted(lane, "what is helper?")
    captured: dict = {}
    with patch("src.api.routes.chat._select_mode", side_effect=recorder):
        _run_handle_chat(pipeline, req, captured, ScriptedTransport({"T": [_summary("t")]}))
    assert seen["prompt"] == "what is helper?"
    assert captured["prompt"].startswith("## Orchestrator scout findings"), "scouts did run"


def test_memrl_objective_is_task_based_not_the_scout_block(lane, pipeline):
    router = MagicMock()
    router.route_with_mode.return_value = (None, None, "repl")
    pipeline.state.hybrid_router = router
    req = _scouted(lane, "PROPOSE ONE HYPOTHESIS")
    captured: dict = {}
    _run_handle_chat(pipeline, req, captured, ScriptedTransport({"T": [_summary("t")]}))
    task_ir = router.route_with_mode.call_args.args[0]
    assert task_ir["objective"] == "PROPOSE ONE HYPOTHESIS"
    assert "scout findings" not in task_ir["objective"]


# ── F16: a refused chunk_context charges nothing; size check precedes JSON parse ─────────


class _Env(_ContextMixin):
    def __init__(self, bundle):
        self._exploration_calls = 0
        self._context_bundle = bundle
        self.context = bundle.full_text()
        self._exploration_log = MagicMock()


def _bundle(budget, size=10_000):
    return ContextBundle([Section("big", "x" * size)], pull_budget_bytes=budget)


def test_refused_chunk_context_does_not_charge_the_budget():
    bundle = _bundle(6000)                      # 4 chunks of ~2500: two would fit, four don't
    env = _Env(bundle)
    with pytest.raises(ContextPullBudgetExceeded):
        env._chunk_context(4, 0)
    assert bundle._bytes_pulled == 0 and bundle._pull_calls == 0
    # the budget is still wholly available to a pull that fits
    assert len(bundle.get("big", max_chars=5000)) == 5000


def test_admitted_chunk_context_still_charges_every_chunk():
    bundle = _bundle(20_000)
    chunks = _Env(bundle)._chunk_context(4, 0)
    assert len(chunks) == 4 and bundle._bytes_pulled == sum(c["char_count"] for c in chunks)


def test_size_cap_runs_before_any_json_parse():
    big = Section("j", "[" + "1," * (MAX_BUNDLE_BYTES // 2) + "1]", kind="json")
    with patch("src.repl_environment.context_bundle.json.loads",
               side_effect=AssertionError("parsed before the size check")):
        with pytest.raises(ValueError, match="exceeds"):
            ContextBundle([big])
