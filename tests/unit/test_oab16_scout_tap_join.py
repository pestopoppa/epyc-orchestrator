"""OAB-16: scout calls appear in the inference tap NEXT TO the planner turn.

Offline, real writer (no monkeypatched ``record_http_call``): two concurrent scout calls and
one planner-turn record are written to a temp serving-calls log, then joined on the request
id the way an analyst would for an OAB-4 ``orch``-arm call. The live-run clause of the box
(a real call on :8083) is not exercised here.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from src.api.routes.chat_pipeline import scout_stage as ss
from src.backends import serving_calls


@pytest.fixture
def tap(tmp_path, monkeypatch):
    path = tmp_path / "serving_calls.jsonl"
    monkeypatch.setenv(serving_calls.LOG_ENV, str(path))
    return path


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_scout_calls_join_the_planner_turn_by_request_id(tap):
    req_id = "req-oab16-join"

    async def scouts():
        token = ss._SCOUT_PARENT.set(req_id)
        try:
            await asyncio.gather(*[
                ss._scout_thread(
                    ss._record_scout_call, "scout_stage_chat_stream", "http://x:8083",
                    [{"role": "user", "content": f"scout {i}"}], 64, time.time(), 200, "stop",
                    {"id_slot": i, "prompt_n": 100 + i, "predicted_n": 7 + i},
                    {"prompt_tokens": 100 + i, "completion_tokens": 7 + i}, None,
                    {"scout_admission": None})
                for i in (0, 1)
            ])
        finally:
            ss._SCOUT_PARENT.reset(token)

    asyncio.run(scouts())

    # The planner turn, written through the same writer with the staged caller block the
    # primitives layer produces (caller.parent_request_id is the /chat request id).
    serving_calls.stage_caller(parent_request_id=req_id, request_id=f"{req_id}:abcd1234")
    staged = serving_calls.peek_staged()
    now = time.time()
    planner = serving_calls.build_record(
        method="planner_turn", role_config=SimpleNamespace(name="architect", model=None),
        request=SimpleNamespace(role="architect", prompt="plan", n_tokens=128, timeout=None,
                                slot_id=None, chat_payload=None),
        base_url="http://x:8083", ts_start=now, ts_end=now + 1,
        result=SimpleNamespace(success=True, completion_reason="stop", failure_reason=None,
                               error_message=None, tokens_generated=9, prompt_tokens=300,
                               prompt_eval_ms=None, generation_ms=None, first_token_ms=None,
                               stream_chunks=None),
        exc=None, staged=staged, notes={})
    assert serving_calls.write_record(planner)
    serving_calls.clear_staged()

    recs = _records(tap)
    scouts_rec = [r for r in recs if r.get("role") == "scout"]
    planner_rec = [r for r in recs if r.get("role") == "architect"]
    assert len(scouts_rec) == 2 and len(planner_rec) == 1

    # one tap record per scout call: role, slot, prompt/decoded tokens, wall
    for r in scouts_rec:
        assert r["notes"]["parent_request_id"] == req_id
        assert r["timings"]["id_slot"] in (0, 1)
        assert r["result"]["tokens_generated"] in (7, 8)
        assert r["result"]["prompt_tokens"] in (100, 101)
        assert r["wall_ms"] >= 0 and r["ts_start"] and r["ts_end"]

    # the join: scouts' notes.parent_request_id == planner's caller.parent_request_id
    assert planner_rec[0]["caller"]["parent_request_id"] == req_id
    joined = {r["notes"]["parent_request_id"] for r in scouts_rec}
    assert joined == {planner_rec[0]["caller"]["parent_request_id"]}


def test_scouts_of_another_request_do_not_join(tap):
    ss._record_scout_call("scout_stage_chat_stream", "http://x:8083", [], 1, time.time(), 200,
                          "stop", None, None, None, {})
    (rec,) = _records(tap)
    assert "parent_request_id" not in (rec.get("notes") or {})
