"""progress_aware_retry_cap (default OFF): attempts that advanced are not charged."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "autopilot"))

from src.autopilot_core.progress_retry import advanced, charged_attempts, qids_of
from src.features import _REGISTRY_BY_NAME, Features, reset_features


def _flag(monkeypatch, on: bool) -> None:
    flagset = Features(progress_aware_retry_cap=on)
    monkeypatch.setattr("src.features.features", lambda: flagset)


def test_flag_registered_default_off():
    assert _REGISTRY_BY_NAME["progress_aware_retry_cap"].default_prod is False
    assert Features().progress_aware_retry_cap is False
    reset_features()


def test_pure_helpers():
    assert qids_of({"a": True, "b": False}) == {"a", "b"}
    assert qids_of([{"qid": "x"}, {"question_id": "y"}, {}]) == {"x", "y"}
    assert advanced(None, {"a"}) is False
    assert advanced({"a"}, {"a"}) is False
    assert advanced({"a"}, {"a", "b"}) is True
    assert charged_attempts([{"a"}, {"a"}, {"a"}]) == 3
    assert charged_attempts([{"a"}, {"a", "b"}, {"a", "b", "c"}]) == 1


def _attempts(*qid_sets):
    return [{"outcomes": {q: True for q in s}} for s in qid_sets]


def test_multitier_charge(monkeypatch):
    import autopilot as ap

    prog = _attempts({"a"}, {"a", "b"}, {"a", "b", "c"})
    flat = _attempts({"a"}, {"a"}, {"a"})
    _flag(monkeypatch, False)
    assert ap._multitier_charged_attempts(prog) == 3
    assert ap._multitier_charged_attempts(flat) == 3
    _flag(monkeypatch, True)
    assert ap._multitier_charged_attempts(prog) == 1
    assert ap._multitier_charged_attempts(flat) == 3


def _result(qids):
    from safety_gate import EvalResult

    return EvalResult(
        tier=1, quality=0.0, speed=1.0, cost=0.1, reliability=1.0,
        question_results=[{"qid": q, "correct": False} for q in qids],
    )


def _gate_failures(monkeypatch, on, series):
    from safety_gate import SafetyGate

    _flag(monkeypatch, on)
    gate = SafetyGate.__new__(SafetyGate)
    gate._consecutive_failures = 0
    gate._failure_streak_qids = None
    for qids in series:
        if not gate._failure_streak_advanced(_result(qids), None):
            gate._consecutive_failures += 1
    return gate._consecutive_failures


def test_gate_streak_progress_not_charged(monkeypatch):
    assert _gate_failures(monkeypatch, True, [{"a"}, {"a", "b"}, {"a", "b", "c"}]) == 1


def test_gate_streak_no_progress_charged(monkeypatch):
    assert _gate_failures(monkeypatch, True, [{"a"}, {"a"}, {"a"}]) == 3


def test_gate_flag_off_unchanged(monkeypatch):
    assert _gate_failures(monkeypatch, False, [{"a"}, {"a", "b"}, {"a", "b", "c"}]) == 3


def test_gate_reset_clears_streak(monkeypatch):
    from safety_gate import SafetyGate

    _flag(monkeypatch, True)
    gate = SafetyGate.__new__(SafetyGate)
    gate._consecutive_failures = 0
    gate._failure_streak_qids = {"a"}
    gate.reset_failures()
    assert gate._failure_streak_qids is None
