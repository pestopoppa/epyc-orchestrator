"""M-19 record spec and confidence gate (synthetic, zero inference)."""
import pytest

from src.typed_decisions.memory_write_record import (
    CalibrationRecord, ConfidenceStatistic, MemoryWriteRecord, MemoryWriteRecordError,
    QuestionAnswer, evaluate_gate,
)

PIN = {"gguf_sha256": "aa"}


def _rec(value=0.8, **kw):
    base = dict(
        op="ADD", target_key="k", content_sha256="c" * 8,
        answers=(QuestionAnswer("q1", "yes", {"yes": 0.7, "no": 0.3}),),
        statistic=ConfidenceStatistic("choice_confidence", value),
        catalogue_id="mem-write", catalogue_version="1", model_pin=PIN, source_event_ids=("e1",),
    )
    base.update(kw)
    return MemoryWriteRecord(**base)


def _cal(**kw):
    base = dict(calibration_id="cal-1", catalogue_id="mem-write", catalogue_version="1",
                model_pin=PIN, statistic_name="choice_confidence", threshold=0.7)
    base.update(kw)
    return CalibrationRecord(**base)


def test_no_calibration_is_unscored_never_enforced():
    v = evaluate_gate(_rec(0.99), None)
    assert (v.status, v.enforce) == ("unscored", False)


def test_unratified_calibration_is_shadow_only():
    assert evaluate_gate(_rec(0.8), _cal()).status == "shadow_pass"
    v = evaluate_gate(_rec(0.5), _cal())
    assert (v.status, v.enforce, v.calibration_id) == ("shadow_hold", False, "cal-1")


def test_ratified_calibration_enforces():
    assert evaluate_gate(_rec(0.5), _cal(operator_ratified=True)).status == "hold"
    assert evaluate_gate(_rec(0.9), _cal(operator_ratified=True)).enforce is True


@pytest.mark.parametrize("override", [
    {"catalogue_version": "2"}, {"model_pin": {"gguf_sha256": "bb"}}, {"statistic_name": "other"},
    {"threshold": float("nan")},
])
def test_calibration_never_transfers_across_catalogue_pin_or_statistic(override):
    assert evaluate_gate(_rec(), _cal(operator_ratified=True, **override)).status == "unscored"


def test_candidate_softmax_and_model_self_report_are_refused():
    with pytest.raises(MemoryWriteRecordError):
        ConfidenceStatistic("softmax_max", 0.95)
    with pytest.raises(MemoryWriteRecordError):
        ConfidenceStatistic("choice_confidence", 0.5, computed_by="model")


def test_record_validation_and_digest_stability():
    with pytest.raises(MemoryWriteRecordError):
        _rec(source_event_ids=())
    with pytest.raises(MemoryWriteRecordError):
        _rec(model_pin={})
    with pytest.raises(MemoryWriteRecordError):
        QuestionAnswer("q", "x", {"a": 0.9, "b": 0.3})
    assert _rec().digest() == _rec().digest() != _rec(0.81).digest()
    assert _rec(op="NOOP", source_event_ids=()).op == "NOOP"
