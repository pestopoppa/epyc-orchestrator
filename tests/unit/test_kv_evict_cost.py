"""Unit tests for the KV evict-cost probe (fakes only, no server)."""

import pytest

from scripts.autopilot import kv_evict_cost as ec
from scripts.autopilot import kv_evict_cost_smoke as sm

CAL = ec.Calibration(a=0.5, b=0.01)


def test_exact_checkpoint():
    r = ec.predict_evict_cost(1000, 400, [100, 400, 700], CAL)
    assert (r.p_star, r.reprefilled_tokens, r.alignment) == (400, 600, ec.ALIGN_EXACT)
    assert r.seconds == pytest.approx(0.5 + 6.0)


def test_aligned_and_misaligned():
    a = ec.predict_evict_cost(1000, 430, [400], CAL, align_tol=64)
    assert a.alignment == ec.ALIGN_ALIGNED and a.cut_to_checkpoint_gap == 30
    m = ec.predict_evict_cost(1000, 600, [400], CAL, align_tol=64)
    assert m.alignment == ec.ALIGN_MISALIGNED and m.reprefilled_tokens == 600


def test_no_checkpoint_full_reprefill():
    r = ec.predict_evict_cost(1000, 50, [100, 200], CAL)
    assert r.p_star is None and r.reprefilled_tokens == 1000
    assert r.alignment == ec.ALIGN_NO_CHECKPOINT


def test_hybrid_cannot_shift():
    r = ec.predict_evict_cost(1000, 400, [400], CAL, hybrid=True)
    assert r.alignment == ec.ALIGN_CANNOT_SHIFT
    assert r.reprefilled_tokens is None and r.seconds is None


def test_arch_detection():
    assert ec.is_hybrid_arch("qwen35") and ec.is_hybrid_arch("Qwen3Next")
    assert not ec.is_hybrid_arch("qwen3") and not ec.is_hybrid_arch(None)


def test_bad_inputs():
    with pytest.raises(ValueError):
        ec.predict_evict_cost(100, 200, [], CAL)
    with pytest.raises(ValueError):
        ec.load_calibration({"a": -1, "b": 1})


def test_no_calibration_gives_no_seconds():
    assert ec.predict_evict_cost(100, 50, [50], None).seconds is None


def test_parse_slot_state_and_fetch_fake():
    payload = [{"id": 0, "n_past": 900, "checkpoints": [{"pos_max": 100}, {"pos_max": 500}]}]
    st = ec.parse_slot_state(payload)
    assert st == {"t_prime": 900, "checkpoints": [100, 500], "arch": None}
    with pytest.raises(KeyError):
        ec.parse_slot_state(payload, slot_id=3)

    class R:
        status_code = 200
        def json(self): return payload

    calls = []
    def fake_get(url, timeout):
        calls.append(url)
        return R()
    assert ec.fetch_slot_state(8070, http_get=fake_get)["t_prime"] == 900
    assert calls == ["http://localhost:8070/slots"]


def test_smoke_passes_on_fake_state():
    res = sm.run_smoke({"t_prime": 4000, "checkpoints": [500, 1500, 2500, 3500]}, CAL)
    assert res["n"] == 20 and res["k"] == 20 and res["pass"]
    classes = {r["class"] for r in res["rows"]}
    assert {ec.ALIGN_EXACT, ec.ALIGN_ALIGNED, ec.ALIGN_MISALIGNED} <= classes


def test_smoke_no_checkpoints_and_hybrid():
    assert sm.run_smoke({"t_prime": 800, "checkpoints": []}, CAL)["pass"]
    h = sm.run_smoke({"t_prime": 800, "checkpoints": [100], "arch": "qwen35"}, CAL)
    assert h["hybrid"] and h["pass"]
    assert {r["class"] for r in h["rows"]} == {ec.ALIGN_CANNOT_SHIFT}


def test_smoke_detects_broken_probe(monkeypatch):
    real = ec.predict_evict_cost
    def bad(t, cut, cps, cal, **kw):
        r = real(t, cut, cps, cal, **kw)
        return ec.EvictCost(r.t_prime, r.cut, r.p_star, (r.reprefilled_tokens or 0) + 1,
                            r.seconds, r.alignment, r.cut_to_checkpoint_gap)
    monkeypatch.setattr(sm, "predict_evict_cost", bad)
    assert not sm.run_smoke({"t_prime": 4000, "checkpoints": [500, 1500]}, CAL)["pass"]
