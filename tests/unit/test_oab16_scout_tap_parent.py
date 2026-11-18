"""OAB-16: every scout serving_call record carries the parent request id."""
from __future__ import annotations

import asyncio

from src.api.routes.chat_pipeline import scout_stage as ss
from src.backends import serving_calls


def test_scout_record_is_tagged_with_the_parent_request(monkeypatch):
    seen = []
    monkeypatch.setattr(serving_calls, "record_http_call", lambda **kw: seen.append(kw) or True)

    async def go():
        token = ss._SCOUT_PARENT.set("req-123")
        try:
            await ss._scout_thread(
                ss._record_scout_call, "scout_stage_chat_stream", "http://x:1",
                [{"role": "user", "content": "hi"}], 8, 1.0, 200, "stop",
                {"id_slot": 2, "prompt_n": 10, "predicted_n": 4}, {"completion_tokens": 4},
                None, {"scout_admission": None})
        finally:
            ss._SCOUT_PARENT.reset(token)

    asyncio.run(go())
    assert len(seen) == 1
    assert seen[0]["role"] == "scout"
    assert seen[0]["notes"]["parent_request_id"] == "req-123"
    assert seen[0]["timings"]["id_slot"] == 2


def test_no_parent_leaves_notes_untouched(monkeypatch):
    seen = []
    monkeypatch.setattr(serving_calls, "record_http_call", lambda **kw: seen.append(kw) or True)
    ss._record_scout_call("scout_stage_chat_stream", "http://x:1", [], 1, 1.0, 200, "stop",
                          None, None, None, {"k": 1})
    assert seen[0]["notes"] == {"k": 1}


def test_run_scouts_wrapper_scopes_the_parent_and_resets_it(monkeypatch):
    captured = {}

    async def fake_inner(*a, _held=None, **kw):
        captured["during"] = ss._SCOUT_PARENT.get()
        return "stage"

    monkeypatch.setattr(ss, "_run_scouts", fake_inner)
    out = asyncio.run(ss.run_scouts(object(), scope=None, role="r", parent_request_id="req-9"))
    assert out == "stage" and captured["during"] == "req-9"
    assert ss._SCOUT_PARENT.get() is None
