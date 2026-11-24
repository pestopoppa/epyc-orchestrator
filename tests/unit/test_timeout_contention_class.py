"""HF-TIMEOUT-CLASS: flag ``timeout_contention_class`` labels timeouts, never excludes rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from src.autopilot_core import timeout_contention as tc  # noqa: E402
from src.features import reset_features  # noqa: E402


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_TIMEOUT_CONTENTION_CLASS", raising=False)
    reset_features()
    yield
    reset_features()


def _on(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TIMEOUT_CONTENTION_CLASS", "1")
    reset_features()


@pytest.mark.parametrize(
    "cov,expected",
    [
        ({"host_inflight": 3, "loadavg_per_core": 0.1}, "contention"),
        ({"host_inflight": 1, "loadavg_per_core": 1.0}, "contention"),
        ({"host_inflight": 1, "loadavg_per_core": 0.49}, "model_slow"),
        ({"host_inflight": 0, "loadavg_per_core": 0.0}, "model_slow"),
        ({"host_inflight": 1, "loadavg_per_core": 0.7}, "unknown"),
        ({"host_inflight": 1}, "unknown"),
        ({"loadavg_per_core": 0.1}, "unknown"),
        ({}, "unknown"),
        (None, "unknown"),
        ({"host_inflight": float("nan"), "loadavg_per_core": "x"}, "unknown"),
    ],
)
def test_classifier_thresholds(cov, expected):
    assert tc.classify_contention(cov) == expected


def test_label_only_touches_timeout_rows():
    t = {"class": "backend_timeout", "code": "inband_timeout"}
    tc.label_timeout_provenance(t, {"host_inflight": 4})
    assert t["contention_class"] == "contention" and t["contention_inputs"] == {"host_inflight": 4}
    other = {"class": "backend_failure"}
    tc.label_timeout_provenance(other, {"host_inflight": 4})
    assert "contention_class" not in other


def test_rates_per_class():
    rows = [
        ({"class": "backend_timeout", "contention_class": "contention"}, None),
        ({"class": "client_transport_timeout", "contention_class": "model_slow"}, None),
        ({"class": "admission_timeout"}, None),  # unlabelled timeout counts as unknown
        ({"class": "backend_failure"}, None),
        (None, None),
    ]
    out = tc.timeout_rate_by_class(rows)
    assert out["total_rows"] == 5 and out["timeout_rows"] == 3
    assert out["counts"] == {"contention": 1, "model_slow": 1, "unknown": 1}
    assert out["rates"]["contention"] == pytest.approx(0.2)


def test_capture_covariates_off_is_identical_and_on_adds_loadavg(monkeypatch):
    import types

    raw = {
        "host_inflight": 2, "loadavg_1min": 3.14159265, "loadavg_per_core": 0.0123456,
        "min_core_mhz": 2000, "n_cores_online": 256, "page_cache_mb": 1.0,
    }
    fake = types.ModuleType("host_health")
    fake.host_timing_covariates = lambda **kw: dict(raw)
    monkeypatch.setitem(sys.modules, "host_health", fake)
    off = eval_tower._capture_host_timing_covariates(tokens_generated=1, elapsed_s=1.0)
    assert "loadavg_per_core" not in off and off == eval_tower._compact_host_covariates(raw)
    _on(monkeypatch)
    on = eval_tower._capture_host_timing_covariates(tokens_generated=1, elapsed_s=1.0)
    assert on["loadavg_per_core"] == 0.0123 and on["host_inflight"] == 2
    assert {k: v for k, v in on.items() if not k.startswith("loadavg")} == off


def test_flag_default_off_and_aggregate_helper(monkeypatch):
    assert eval_tower._timeout_contention_class_on() is False
    _on(monkeypatch)
    assert eval_tower._timeout_contention_class_on() is True
    R = type("R", (), {})
    a, b = R(), R()
    a.failure_provenance = {"class": "backend_timeout", "contention_class": "model_slow"}
    b.failure_provenance = None
    out = eval_tower._timeout_class_rates([a, b])
    assert out["counts"]["model_slow"] == 1 and out["rates"]["model_slow"] == 0.5


def _genexc_result(monkeypatch, exc):
    import types

    fake = types.ModuleType("host_health")
    fake.host_timing_covariates = lambda **kw: {
        "host_inflight": 4, "loadavg_1min": 9.0, "loadavg_per_core": 0.9,
    }
    monkeypatch.setitem(sys.modules, "host_health", fake)

    def _boom(**kw):
        raise exc

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _boom)
    tower = eval_tower.EvalTower()
    q = {"id": "q", "suite": "unit", "prompt": "2+2", "expected": "4",
         "scoring_method": "exact_match"}
    return tower._generate_question(q, None).final_result


def test_generation_exception_timeout_labelled_when_on(monkeypatch):
    import httpx

    _on(monkeypatch)
    row = _genexc_result(monkeypatch, httpx.ReadTimeout("slow"))
    prov = row.failure_provenance
    assert prov["class"] == "backend_timeout" and prov["phase"] == "generation_exception"
    assert prov["contention_class"] == "contention"
    assert prov["contention_inputs"]["host_inflight"] == 4
    # Non-timeout exceptions stay unlabelled.
    other = _genexc_result(monkeypatch, ValueError("bad"))
    assert other.failure_provenance is None
    # Aggregate counts it.
    assert eval_tower._timeout_class_rates([row])["counts"]["contention"] == 1


def test_generation_exception_timeout_unchanged_when_off(monkeypatch):
    import httpx

    row = _genexc_result(monkeypatch, httpx.ReadTimeout("slow"))
    assert row.failure_provenance is None
    assert row.disposition == eval_tower.DISPOSITION_INFRA_FAILED
