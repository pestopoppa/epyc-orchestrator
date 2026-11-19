"""AP-72: per-model physical speed ceilings (orchestration/autopilot_speed_ceilings.yaml).

ceiling = measured decode roofline x concurrent slots x 1.10, default OFF behind the runtime flag
``autopilot_speed_physical_bound`` (env AUTOPILOT_SPEED_PHYSICAL_BOUND). Unknown model = no bound.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts" / "autopilot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.autopilot import host_health as hh  # noqa: E402
from scripts.autopilot import safety_gate as sg  # noqa: E402

FLAG = sg.SPEED_PHYSICAL_BOUND_FLAG_ENV
CEIL_ENV = sg.SPEED_PHYSICAL_CEILING_ENV
TABLE = yaml.safe_load((ROOT / "orchestration" / "autopilot_speed_ceilings.yaml").read_text())
A3B = "qwen3.6-35b-a3b-mtp-q8_0"
Q27 = "qwen3.8-27b-q8_0"
FLASH = "qwen3.8-flash-next-ud-iq4_xs"


def _result(speed=10.0, routing=None, details=None, conc=1):
    return sg.EvalResult(
        tier=1, quality=1.7, speed=speed, cost=0.5, reliability=1.0,
        per_suite_quality={"math": 2.0}, per_suite_counts={"math": 6},
        routing_distribution=routing if routing is not None else {"frontdoor": 1.0},
        details=details or {}, eval_concurrency=conc,
    )


def _state():
    return {
        "quality": 1.6, "speed": 12.0, "cost": 0.5, "reliability": 0.9, "frontdoor_speed": 12.0,
        "per_suite_quality": {"math": 2.0},
        "baselines_by_tier": {"1": 1.6},
        "per_suite_quality_by_tier": {"1": {"math": 2.0}},
        "per_suite_counts_by_tier": {"1": {"math": 6}},
    }


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    class _S:
        def is_throttled(self):
            return False, []

    monkeypatch.setattr(hh.HostHealthState, "snapshot", staticmethod(lambda: _S()))
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(CEIL_ENV, raising=False)


def test_table_arithmetic_is_roofline_x_slots_x_margin():
    assert TABLE["margin"] == 0.10 and TABLE["bound_enabled_default"] is False
    for mid, e in TABLE["models"].items():
        assert e["ceiling_tps"] == pytest.approx(e["roofline_tps"] * e["slots"] * 1.10), mid
        assert e["roofline_source"] and e["slots_source"], mid
    assert TABLE["models"][Q27]["ceiling_tps"] == pytest.approx(210.276)
    assert TABLE["models"][A3B]["ceiling_tps"] == pytest.approx(176.968)
    assert TABLE["models"][FLASH]["ceiling_tps"] == pytest.approx(57.97)


def test_table_matches_stack_priors_measured_values_and_slots():
    pri = yaml.safe_load((ROOT / "orchestration" / "derived" / "stack_priors.yaml").read_text())["roles"]
    for mid, e in TABLE["models"].items():
        role = e["roles"][0]
        assert pri[role]["model_id"] == mid
        assert pri[role]["priors"]["throughput_tps"] == pytest.approx(e["roofline_tps"]), mid
        assert pri[role]["serving"]["slots"] == e["slots"], mid


def test_flag_is_registered_default_off_and_runtime_toggleable():
    from src.features import _REGISTRY_BY_NAME, Features

    spec = _REGISTRY_BY_NAME["autopilot_speed_physical_bound"]
    assert spec.default_test is False and spec.default_prod is False
    assert spec.env_name == FLAG and spec.env_on == "1" and spec.env_off == "0"
    assert Features().autopilot_speed_physical_bound is False


def test_default_off_ignores_table():
    assert sg._speed_physical_ceiling(_result()) is None
    v = sg.SafetyGate(baseline_state=_state(), use_sequential=False).check(_result(speed=1e6))
    assert "speed_above_physical_bound" not in v.categories


def test_flag_on_applies_per_model_ceiling_with_eval_concurrency(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    # frontdoor/worker -> A3B; 4 slots at conc 4 = 40.22*4*1.1
    assert sg._speed_physical_ceiling(_result(routing={"frontdoor": 0.6, "worker": 0.4}, conc=4)) == pytest.approx(176.968)
    # serial eval (conc 1): one stream's roofline + margin
    assert sg._speed_physical_ceiling(_result(conc=1)) == pytest.approx(40.22 * 1.1)
    # conc above slots is capped at slots
    assert sg._speed_physical_ceiling(_result(conc=16)) == pytest.approx(176.968)
    # explicit model id
    assert sg._speed_physical_ceiling(_result(details={"speed_ceiling_model": FLASH}, conc=4)) == pytest.approx(57.97)


def test_multi_model_eval_sums_full_slot_ceilings(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    c = sg._speed_physical_ceiling(_result(routing={"frontdoor": 0.5, "architect": 0.5}, conc=4))
    assert c == pytest.approx((40.22 * 4 + 47.79 * 4 + 52.7) * 1.1)


def test_unknown_model_is_no_bound_and_logged(monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "1")
    with caplog.at_level(logging.WARNING, logger="autopilot.safety"):
        assert sg._speed_physical_ceiling(_result(details={"speed_ceiling_model": "mystery-70b"})) is None
        assert sg._speed_physical_ceiling(_result(routing={"frontdoor": 0.5, "novel_role": 0.5})) is None
    assert "fail-open" in caplog.text and "mystery-70b" in caplog.text
    v = sg.SafetyGate(baseline_state=_state(), use_sequential=False).check(
        _result(speed=1e6, details={"speed_ceiling_model": "mystery-70b"}))
    assert "speed_above_physical_bound" not in v.categories


def test_gate_flags_above_ceiling_when_on_and_toggles_at_runtime(monkeypatch):
    gate = lambda: sg.SafetyGate(baseline_state=_state(), use_sequential=False)  # noqa: E731
    def over():  # fresh result each time: check() caches its verdict on the result
        return _result(speed=300.0, conc=4)  # > 176.968

    assert "speed_above_physical_bound" not in gate().check(over()).categories  # flag off
    monkeypatch.setenv(FLAG, "1")
    bad = gate().check(over())
    assert not bad.passed and "speed_above_physical_bound" in bad.categories
    assert "speed_above_physical_bound" not in gate().check(_result(speed=150.0, conc=4)).categories
    monkeypatch.setenv(FLAG, "0")  # toggled off again, no restart
    assert "speed_above_physical_bound" not in gate().check(over()).categories


def test_pure_verdict_equals_check_with_table(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    for speed in (10.0, 176.9, 177.1, 900.0):
        gate = sg.SafetyGate(baseline_state=_state(), use_sequential=False)
        r = _result(speed=speed, conc=4)
        pure = sg.verdict_from_evidence(r, sg.gate_evidence_from(gate, r))
        live = gate.check(r)
        assert (pure.passed, pure.violations, pure.categories) == (live.passed, live.violations, live.categories)


def test_env_number_overrides_table(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(CEIL_ENV, "20")
    assert sg._speed_physical_ceiling(_result(conc=4)) == 20.0


def test_missing_or_malformed_table_fails_open(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(sg, "SPEED_CEILINGS_PATH", tmp_path / "nope.yaml")
    with caplog.at_level(logging.WARNING, logger="autopilot.safety"):
        assert sg._speed_physical_ceiling(_result()) is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [not, a, mapping]\n")
    monkeypatch.setattr(sg, "SPEED_CEILINGS_PATH", bad)
    assert sg._speed_physical_ceiling(_result()) is None
