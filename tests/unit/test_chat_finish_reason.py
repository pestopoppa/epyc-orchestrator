"""AEI-CHAT-FINISH-REASON: /chat carries finish_reason and truncated from completion_reason."""

import pytest

from src.api.models import ChatResponse
from src.api.routes.chat_pipeline.finish_reason import finish_reason_fields
from src.features import reset_features


class _Prim:
    def __init__(self, meta):
        self._meta = meta

    def get_last_inference_meta(self):
        return self._meta


@pytest.fixture
def flag(monkeypatch):
    def _set(on):
        monkeypatch.setenv("ORCHESTRATOR_CHAT_FINISH_REASON", "1" if on else "0")
        reset_features()

    yield _set
    monkeypatch.undo()
    reset_features()


def _resp(**extra):
    return ChatResponse(answer="a", turns=1, elapsed_seconds=0.1, mock_mode=False, **extra)


@pytest.mark.parametrize("reason,truncated", [
    ("stop", False),
    ("length", True),
    ("context_limit", True),
])
def test_reason_is_carried_and_truncation_flagged(flag, reason, truncated):
    flag(True)
    fields = finish_reason_fields(_Prim({"completion_reason": reason}))
    assert fields == {"finish_reason": reason, "truncated": truncated}
    body = _resp(**fields).model_dump(mode="json")
    assert body["finish_reason"] == reason and body["truncated"] is truncated


def test_complete_answer_is_not_marked_truncated(flag):
    flag(True)
    assert finish_reason_fields(_Prim({"completion_reason": "stop"}))["truncated"] is False


def test_flag_off_response_is_unchanged(flag):
    flag(False)
    assert finish_reason_fields(_Prim({"completion_reason": "length"})) == {}
    body = _resp().model_dump(mode="json")
    assert "finish_reason" not in body and "truncated" not in body


def test_unknown_meta_adds_nothing(flag):
    flag(True)
    assert finish_reason_fields(_Prim(None)) == {}
    assert finish_reason_fields(_Prim({})) == {}
