"""EA-HYBRID-FOLD-GUARD: Expected Attention must not compact the hybrid 27B on :8083."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

kv_compress = importlib.import_module("kv_compress")
actions = importlib.import_module("actions")


def _ok_post(calls):
    def post(url, json=None, timeout=None):
        calls.append((url, json))
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"n_evicted": 3, "scorer": json["scorer"], "pos_max_after": 10},
        )

    return post


def _patch_all(monkeypatch):
    seen = []
    monkeypatch.setattr(kv_compress, "production_ports", lambda: {"hyb": 8083, "fd": 8070})

    def fake_auto(port, **kw):
        seen.append(port)
        return kv_compress.CompressResult(success=True, port=port)

    monkeypatch.setattr(kv_compress, "auto_compress_if_needed", fake_auto)
    return seen


def test_auto_compress_ea_skips_8083(monkeypatch):
    seen = _patch_all(monkeypatch)
    res = kv_compress.auto_compress_all(keep_ratio=0.5, scorer="expected_attention")
    assert seen == [8070]
    assert res["hyb"] is None


def test_auto_compress_default_scorer_is_ea_and_skips(monkeypatch):
    seen = _patch_all(monkeypatch)
    kv_compress.auto_compress_all(keep_ratio=0.5)
    assert seen == [8070]


def test_explicit_8083_ea_refused_by_action(monkeypatch):
    calls = []
    monkeypatch.setattr(kv_compress.httpx, "post", _ok_post(calls))
    out, species = actions._action_slot_compact(
        {"type": "slot_compact", "port": 8083, "keep_ratio": 0.9,
         "scorer": "expected_attention"},
        SimpleNamespace(),
    )
    assert out.status == "invalid"
    assert "8083" in out.reason and "EA-HYBRID-FOLD-GUARD" in out.reason
    assert species == "slot_management"
    assert calls == []


def test_compress_slot_chokepoint_refuses_ea(monkeypatch):
    calls = []
    monkeypatch.setattr(kv_compress.httpx, "post", _ok_post(calls))
    r = kv_compress.compress_slot(port=8083, keep_ratio=0.9)
    assert not r.success and "refused" in r.error
    assert calls == []


def test_non_ea_scorers_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(kv_compress.httpx, "post", _ok_post(calls))
    r = kv_compress.compress_slot(port=8083, keep_ratio=0.5, scorer="knorm")
    assert r.success and len(calls) == 1
    seen = _patch_all(monkeypatch)
    kv_compress.auto_compress_all(keep_ratio=0.5, scorer="knorm")
    assert sorted(seen) == [8070, 8083]
    # EA on a non-hybrid port is unchanged too.
    r = kv_compress.compress_slot(port=8070, keep_ratio=0.5)
    assert r.success


def test_floor_override_admits_only_at_or_above_floor(monkeypatch):
    calls = []
    monkeypatch.setattr(kv_compress.httpx, "post", _ok_post(calls))
    monkeypatch.setattr(kv_compress, "EA_HYBRID_MIN_KEEP_RATIO", 0.8)
    assert not kv_compress.compress_slot(port=8083, keep_ratio=0.79).success
    assert kv_compress.compress_slot(port=8083, keep_ratio=0.8).success
    assert kv_compress.compress_slot(port=8083, keep_ratio=0.9).success
    assert len(calls) == 2
    seen = _patch_all(monkeypatch)
    kv_compress.auto_compress_all(keep_ratio=0.9, scorer="expected_attention")
    assert sorted(seen) == [8070, 8083]
    seen.clear()
    kv_compress.auto_compress_all(keep_ratio=0.5, scorer="expected_attention")
    assert seen == [8070]
