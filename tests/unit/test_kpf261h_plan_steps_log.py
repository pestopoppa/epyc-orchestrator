"""KPF-26.1h: plan_steps_parsed event."""
from unittest.mock import MagicMock

from orchestration.repl_memory.progress_logger import EventType
from src.api.routes.chat_pipeline.proactive_stage import (
    PLAN_STEPS_PARSED_ACTION_CAP,
    _log_plan_steps_parsed,
)


def test_emits_one_event_with_schema_and_cap():
    state = MagicMock()
    steps = [
        {"id": "s1", "action": "x" * 5000, "actor": "coder", "persona": "terse"},
        {"id": "s2", "action": "summarize", "actor": "worker"},
    ]
    _log_plan_steps_parsed(state, "t-1", steps)
    state.progress_logger.log.assert_called_once()
    e = state.progress_logger.log.call_args[0][0]
    assert e.event_type == EventType.PLAN_STEPS_PARSED
    assert e.event_type.value == "plan_steps_parsed"
    assert e.task_id == "t-1" and e.timestamp is not None
    assert e.data["plan_id"] == "t-1" and e.data["step_ids"] == ["s1", "s2"]
    assert len(e.data["steps"][0]["action"]) == PLAN_STEPS_PARSED_ACTION_CAP
    assert e.data["steps"][0]["action_len"] == 5000
    assert e.data["steps"][0]["persona"] == "terse"
    assert e.data["steps"][1]["persona"] is None


def test_redacts_credentials(monkeypatch):
    import src.repl_environment.redaction as r
    monkeypatch.setattr(r, "redact_if_enabled", lambda t: t.replace("SECRET", "[R]"))
    state = MagicMock()
    _log_plan_steps_parsed(state, "t", [{"id": "a", "action": "use SECRET"}])
    assert state.progress_logger.log.call_args[0][0].data["steps"][0]["action"] == "use [R]"


def test_no_logger_or_failure_is_silent():
    _log_plan_steps_parsed(MagicMock(progress_logger=None), "t", [{"id": "a", "action": "b"}])
    state = MagicMock()
    state.progress_logger.log.side_effect = RuntimeError("boom")
    _log_plan_steps_parsed(state, "t", [{"id": "a", "action": "b"}])
