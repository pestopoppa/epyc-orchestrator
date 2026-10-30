import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.api.routes.chat_pipeline import proactive_stage as stage


@pytest.fixture(autouse=True)
def journal_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_PROGRESS_LOG_DIR", str(tmp_path))
    return tmp_path


def test_journal_contains_parsed_identity_without_progress_logger(journal_dir):
    steps = stage._parse_plan_steps('[{"id":"s1","action":"Investigate"},'
                                    '{"id":"s2","actor":"coder","action":"Implement"}]')
    stage._log_plan_steps_parsed(SimpleNamespace(progress_logger=None), "task-1", steps)
    record = json.loads((journal_dir / "plan_steps_parsed.jsonl").read_text())
    assert record["schema"] == "epyc.plan_steps_parsed.v1"
    assert record["task_id"] == "task-1"
    assert [(r["step_id"], r["actor"], r["action"]) for r in record["data"]["steps"]] == [
        ("s1", "worker", "Investigate"), ("s2", "coder", "Implement")]


def test_failed_write_rolls_back_and_reports(monkeypatch, journal_dir, caplog):
    path = journal_dir / "plan_steps_parsed.jsonl"
    path.write_text('{"prior":true}\n')
    original_write = stage.os.write
    def broken_write(fd, data):
        original_write(fd, data[:5])
        raise OSError("disk failure")
    monkeypatch.setattr(stage.os, "write", broken_write)
    state = SimpleNamespace(progress_logger=MagicMock())
    stage._log_plan_steps_parsed(state, "task-2", [{"id": "s", "action": "x"}])
    assert path.read_text() == '{"prior":true}\n'
    assert "journal capture failed" in caplog.text
    state.progress_logger.log.assert_not_called()


def test_next_append_recovers_interrupted_tail(journal_dir):
    path = journal_dir / "plan_steps_parsed.jsonl"
    path.write_text('{"prior":true}\n{"partial":')
    stage._log_plan_steps_parsed(SimpleNamespace(progress_logger=None), "task-3",
                                 [{"id": "s", "actor": "worker", "action": "go"}])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0] == {"prior": True}
    assert records[1]["task_id"] == "task-3"
