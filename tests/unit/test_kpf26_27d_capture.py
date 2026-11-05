"""KPF-26 fork attestation witness and KPF-27d per-worker shadow capture.

Conditions: docs/reports/kv-cache/high-impact-reviews-20261105.md (epyc-root)
§1 (tests a-c) and §2 (tests a-c). Offline, inference-free.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from src.inference import prefix_index as pi
from src.scheduling import kv_pool_admission as kpa
from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

URL = "http://localhost:8083"
KV_CAPS = {"min_tokens": 0, "mode": "kv", "checkpoint_at": False}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    from src.runtime import long_prefill_lease

    monkeypatch.setenv(pi.BLOCK_CHARS_ENV, "256")
    monkeypatch.setattr(long_prefill_lease, "lease_dir", lambda: tmp_path)
    for name in (pi.FLAG_ENV, pi.FORK_ENV, pi.PIN_ENV, pi.LPM_ENV, pi.LPM_MIN_TOKENS_ENV,
                 pi.TRUNK_MIN_TOKENS_ENV, pi.TRUNK_HOLD_S_ENV, kpa.KV_POOL_LONG_PREFILL_ENV,
                 kpa.KV_POOL_CACHE_CREDIT_MARGIN_ENV, kpa.KV_POOL_WAIT_ENV,
                 kpa.KV_POOL_MAX_QUEUED_ENV):
        monkeypatch.delenv(name, raising=False)
    pi.reset_indexes()
    yield
    pi.reset_indexes()


def _doc(tag: str, lines: int) -> str:
    return "".join(f"{tag} line {i}: shared context for every turn\n" for i in range(lines))


def _pool(fork_caps) -> SharedKVPoolAdmission:
    return SharedKVPoolAdmission(occupancy=lambda url: None, cross_process=False,
                                 fork_caps=fork_caps)


def _run(pool, tag="A"):
    t = pool.acquire(URL, 13000, 393216, prefix_key=_doc("W", 600) + tag, timeout_s=0)
    assert t is not None
    rec = pool.admission_record(t)
    inflight = dict(pool._inflight[URL])
    pool.release(URL, t)
    return rec, inflight


def _strip(rec):
    rec = {k: v for k, v in rec.items() if k not in ("queue_wait_ms", "admission_ticket")}
    rec["prefix_index"] = {k: v for k, v in rec["prefix_index"].items()
                           if k not in ("fork_witness", "match")}
    return rec


# -- KPF-26 (a): raising / non-dict caps -> unknown, verdict + ticket unchanged ----

@pytest.mark.parametrize("bad", ["raise", "nondict"])
def test_kpf26_bad_caps_are_unknown_and_admission_is_unchanged(monkeypatch, bad):
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.FORK_ENV, "auto")

    def caps(url):
        if bad == "raise":
            raise RuntimeError("props unreadable")
        return ["not", "a", "dict"]

    rec, inflight = _run(_pool(caps))
    pi.reset_indexes()
    base, base_inflight = _run(_pool(lambda url: None))
    w = rec["prefix_index"]["fork_witness"]
    assert w["caps_state"] == "unknown" and w["attestation"] == "unknown"
    assert w["view_fork"] is False and w["view_fork_without_capable_caps"] is False
    assert _strip(rec) == _strip(base)  # verdict, credit, reservation identical
    assert list(inflight.values()) == list(base_inflight.values())


def test_kpf26_fork_caps_contract_kept():
    pool = _pool(MagicMock(side_effect=RuntimeError("x")))
    assert pool.fork_caps(URL) is None
    assert _pool(lambda url: 3).fork_caps(URL) is None
    assert _pool(lambda url: KV_CAPS).fork_caps(URL) == KV_CAPS


# -- KPF-26 (b): flags off -> fork_caps never called, no witness key -------------

@pytest.mark.parametrize("flags", [{}, {pi.FLAG_ENV: "1"}, {pi.FLAG_ENV: "1", pi.FORK_ENV: "0"}])
def test_kpf26_flag_off_never_calls_fork_caps_or_emits_a_witness(monkeypatch, flags):
    for k, v in flags.items():
        monkeypatch.setenv(k, v)
    caps = MagicMock(return_value=KV_CAPS)
    rec, _ = _run(_pool(caps))
    caps.assert_not_called()
    assert "fork_witness" not in str(rec)


# -- KPF-26 (c): anti-vacuity -- a null witness can never read as capable --------

def test_kpf26_null_witness_never_passes(monkeypatch):
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.FORK_ENV, "1")
    rec, _ = _run(_pool(lambda url: None))
    w = rec["prefix_index"]["fork_witness"]
    assert w["attestation"] == "unknown" and w["attestation"] != "fork_capable"
    assert w["server_fork_mode"] is None and w["server_min_tokens"] is None
    assert w["fork_credit_tokens"] == 0
    assert kpa._fork_witness(None)["attestation"] == "unknown"
    assert kpa._fork_witness({})["attestation"] == "fork_not_capable"
    pi.reset_indexes()
    ok, _ = _run(_pool(lambda url: KV_CAPS))
    w2 = ok["prefix_index"]["fork_witness"]
    assert w2["caps_state"] == "present" and w2["attestation"] == "fork_capable"
    assert w2["server_fork_mode"] == "kv" and w2["view_fork"] is True


# -- KPF-27d (a): singleton identity across both import paths --------------------

def test_kpf27d_shared_pool_singleton_identity():
    from src.api import admission as api_admission
    from src.scheduling.kv_pool_admission import get_shared_pool_admission

    assert get_shared_pool_admission() is api_admission.get_shared_pool_admission()
    assert get_shared_pool_admission() is get_shared_pool_admission()


# -- KPF-27d (b): flag off -> get_status key set unchanged -----------------------

EXPECTED_KEYS = {
    "reserved_tokens", "in_flight", "queued", "new_token_ratio", "long_prefill_threshold",
    "long_prefill_lease_held", "long_prefill_lease_held_here",
    "long_prefill_lease_cross_process", "queued_long_prefills", "admission_stats",
}


def test_kpf27d_flag_off_get_status_key_set_unchanged():
    pool = _pool(lambda url: None)
    t = pool.acquire(URL, 100, 393216, timeout_s=0)
    st = pool.get_status()
    assert set(st[URL]) == EXPECTED_KEYS
    pool.release(URL, t)


def test_kpf27d_flag_on_capture_is_additive_and_per_worker(monkeypatch):
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    pool = _pool(lambda url: None)
    t = pool.acquire(URL, 13000, 393216, prefix_key=_doc("S", 600), timeout_s=0)
    st = pool.get_status()[URL]
    assert set(st) == EXPECTED_KEYS | {"prefix_index"}
    cap = st["prefix_index"]["shadow_capture"]
    assert cap["worker_pid"] == os.getpid()
    for k in ("prediction_abs_err_tokens", "slot_prediction_hits"):
        assert set(cap[k]) == {"numerator", "denominator", "value"}
    assert set(cap["stale_drops"]) == {"by_reason", "total", "lookups"}
    pool.release(URL, t)


# -- KPF-27d (c): empty denominator -> null, not 0 --------------------------------

def test_kpf27d_empty_denominator_is_null():
    cap = pi.shadow_capture(pi._new_stats())
    assert cap["prediction_abs_err_tokens"] == {"numerator": 0, "denominator": 0, "value": None}
    assert cap["slot_prediction_hits"]["value"] is None
    full = dict(pi._new_stats(), predictions=4, prediction_abs_err_tokens=10,
                slot_predictions=2, slot_prediction_hits=1, stale_drops={"epoch": 3})
    cap = pi.shadow_capture(full)
    assert cap["prediction_abs_err_tokens"]["value"] == 2.5
    assert cap["slot_prediction_hits"]["value"] == 0.5
    assert cap["stale_drops"]["total"] == 3
