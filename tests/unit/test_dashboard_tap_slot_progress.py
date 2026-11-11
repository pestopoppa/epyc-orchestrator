"""RTG-47 — live ``slot_progress`` on structured tap requests (2026-08-10).

The tap stream is silent through a long prefill (no chunks until the first
token), but the llama-server slot knows ``n_prompt_tokens_processed`` /
``n_prompt_tokens`` / ``next_token[].n_decoded`` the whole time. The machine
page renders these as live ↑ingest/↓decode counters. Attach rules under test:
fresh-sample-only, processing-slots-only, one busy slot ⇒ unambiguous, several
⇒ port aggregate flagged ``ambiguous``, none/stale ⇒ no field at all.

All fixtures are synthetic; no sockets, no processes.
"""

from __future__ import annotations

from src.api.routes.dashboard import _SLOT_PROGRESS_FRESH_S, _attach_slot_progress

_NOW = 1_000_000.0


def _slot(processing: bool, *, total: int = 100, done: int = 40,
          decoded: int | None = None) -> dict:
    slot = {
        "id": 0,
        "is_processing": processing,
        "n_prompt_tokens": total,
        "n_prompt_tokens_processed": done,
    }
    if decoded is not None:
        slot["next_token"] = [{"n_decoded": decoded, "has_next_token": True}]
    return slot


def test_single_busy_slot_attaches_unambiguous() -> None:
    reqs = [{"request_id": "r1", "port": 8085}]
    out = _attach_slot_progress(
        reqs, slots_by_port={8085: [_slot(True, total=48726, done=48447, decoded=0)]},
        sampled_at=_NOW, now=_NOW)
    sp = out[0]["slot_progress"]
    assert sp["n_prompt_tokens"] == 48726
    assert sp["n_prompt_tokens_processed"] == 48447
    assert sp["n_decoded"] == 0
    assert sp["ambiguous"] is False


def test_multiple_busy_slots_aggregate_and_flag_ambiguous() -> None:
    reqs = [{"request_id": "r1", "port": 8072}]
    out = _attach_slot_progress(
        reqs,
        slots_by_port={8072: [_slot(True, total=100, done=60, decoded=5),
                              _slot(True, total=200, done=80, decoded=7)]},
        sampled_at=_NOW, now=_NOW)
    sp = out[0]["slot_progress"]
    assert sp["n_prompt_tokens"] == 300
    assert sp["n_prompt_tokens_processed"] == 140
    assert sp["n_decoded"] == 12
    assert sp["ambiguous"] is True


def test_idle_slots_attach_nothing() -> None:
    """A finished slot retains its counters — attributing it would revive a
    dead task's numbers, so only ``is_processing`` slots count."""
    reqs = [{"request_id": "r1", "port": 8070}]
    out = _attach_slot_progress(
        reqs, slots_by_port={8070: [_slot(False, total=30, done=30, decoded=512)]},
        sampled_at=_NOW, now=_NOW)
    assert "slot_progress" not in out[0]


def test_stale_sample_attaches_nothing() -> None:
    reqs = [{"request_id": "r1", "port": 8085}]
    out = _attach_slot_progress(
        reqs, slots_by_port={8085: [_slot(True)]},
        sampled_at=_NOW - _SLOT_PROGRESS_FRESH_S - 1, now=_NOW)
    assert "slot_progress" not in out[0]


def test_missing_decoded_reads_none_and_bad_port_is_skipped() -> None:
    reqs = [{"request_id": "r1", "port": 8085},
            {"request_id": "r2", "port": "not-a-port"}]
    out = _attach_slot_progress(
        reqs, slots_by_port={8085: [_slot(True)]}, sampled_at=_NOW, now=_NOW)
    assert out[0]["slot_progress"]["n_decoded"] is None
    assert "slot_progress" not in out[1]


def test_complete_request_never_wears_the_live_slot() -> None:
    """The busy slot on a port belongs to whichever request is still running
    there — a completed request re-wearing it was the live defect this rule
    closed (observed 2026-08-10: same-port complete + running both attached)."""
    reqs = [{"request_id": "done", "port": 8072, "status": "complete"},
            {"request_id": "live", "port": 8072, "status": "running"}]
    out = _attach_slot_progress(
        reqs, slots_by_port={8072: [_slot(True, decoded=432)]},
        sampled_at=_NOW, now=_NOW)
    assert "slot_progress" not in out[0]
    sp = out[1]["slot_progress"]
    assert sp["n_decoded"] == 432
    assert sp["ambiguous"] is False  # one busy slot, one LIVE candidate


def test_two_live_candidates_on_one_port_read_ambiguous() -> None:
    reqs = [{"request_id": "a", "port": 8072, "status": "running"},
            {"request_id": "b", "port": 8072, "status": "quiet"}]
    out = _attach_slot_progress(
        reqs, slots_by_port={8072: [_slot(True)]}, sampled_at=_NOW, now=_NOW)
    assert out[0]["slot_progress"]["ambiguous"] is True
    assert out[1]["slot_progress"]["ambiguous"] is True


def test_original_request_objects_are_not_mutated() -> None:
    req = {"request_id": "r1", "port": 8085}
    _attach_slot_progress(
        [req], slots_by_port={8085: [_slot(True)]}, sampled_at=_NOW, now=_NOW)
    assert "slot_progress" not in req


# ---- cross-worker cache + unanswered reasons (f2-slots-diag) ----------------

import asyncio

import httpx

from src.api.routes import dashboard as _dash


def test_slots_cache_roundtrip_and_cold_worker_hit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_dash, "_SLOTS_CACHE_PATH", tmp_path / "slots.json")
    monkeypatch.setattr(_dash, "_LAST_SLOTS", {"ts": 0.0, "by_port": {}})
    _dash._write_slots_cache(_NOW - 3, {8085: [_slot(True, total=10, done=5)]})
    assert not list(tmp_path.glob("*.tmp"))
    reqs = [{"request_id": "r1", "port": 8085}]
    out = _dash._attach_slot_progress(reqs, now=_NOW)
    assert out[0]["slot_progress"]["n_prompt_tokens"] == 10
    out = _dash._attach_slot_progress(reqs, now=_NOW + _SLOT_PROGRESS_FRESH_S + 10)
    assert "slot_progress" not in out[0]


def test_slots_cache_missing_or_corrupt(tmp_path, monkeypatch) -> None:
    p = tmp_path / "slots.json"
    monkeypatch.setattr(_dash, "_SLOTS_CACHE_PATH", p)
    assert _dash._read_slots_cache() == (0.0, {})
    p.write_text("{not json")
    assert _dash._read_slots_cache() == (0.0, {})


def test_poll_slot_reasons() -> None:
    def run(handler):
        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                r: dict = {}
                _dash._SLOT_REASONS.set(r)
                res = await _dash._poll_slot(c, 1)
                return res, r
        return asyncio.run(go())

    def refused(req): raise httpx.ConnectError("no", request=req)
    def timeout(req): raise httpx.ReadTimeout("t", request=req)
    assert run(refused) == (None, {1: "refused"})
    assert run(timeout) == (None, {1: "timeout"})
    assert run(lambda r: httpx.Response(503)) == (None, {1: "http_503"})
    assert run(lambda r: httpx.Response(200, content=b"zz"))[1] == {1: "malformed"}
    assert run(lambda r: httpx.Response(200, json={"a": 1})) == (None, {1: "malformed"})
    assert run(lambda r: httpx.Response(200, json=[]))[1] == {}


def test_poll_all_slots_meta_reasons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_dash, "_SLOTS_CACHE_PATH", tmp_path / "s.json")
    monkeypatch.setattr(_dash, "_LAST_SLOTS", {"ts": 0.0, "by_port": {}})
    monkeypatch.setattr(_dash, "_discover_llama_ports", lambda: {1: "a", 2: "b", 3: "c"})

    async def fake(client, port):
        if port == 1:
            return [{"id": 0}]
        _dash._SLOT_REASONS.get()[port] = "refused" if port == 2 else "http_500"
        return None
    monkeypatch.setattr(_dash, "_poll_slot", fake)
    _, meta = asyncio.run(_dash._poll_all_slots())
    assert meta["unanswered_ports"] == [2, 3]
    assert meta["unanswered_reasons"] == {"2": "refused", "3": "http_500"}
    assert meta["unanswered_reason_counts"] == {"http_500": 1, "refused": 1}
    assert meta["timed_out"] == 2 and meta["answered"] == 1
    assert (tmp_path / "s.json").exists()
