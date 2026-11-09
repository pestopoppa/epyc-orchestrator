"""M-27d.1: request-source label on episodic routing rows, default off."""
from __future__ import annotations

from unittest.mock import MagicMock

from orchestration.repl_memory.progress_logger import ProgressLogger, ProgressReader
from orchestration.repl_memory.q_scorer import QScorer


def _scored_context(tmp_path, monkeypatch, flag, source):
    if flag:
        monkeypatch.setenv("ORCHESTRATOR_EPISODIC_SOURCE_LABEL", "1")
    else:
        monkeypatch.delenv("ORCHESTRATOR_EPISODIC_SOURCE_LABEL", raising=False)
    log = ProgressLogger(log_dir=tmp_path, buffer_size=1)
    ir = {"objective": "label me", "task_type": "chat", "priority": "interactive"}
    if source:
        ir["request_source"] = source
    log.log_task_started("t1", ir, ["frontdoor"], "rules")
    log.log_task_completed("t1", success=True)
    log.flush()
    store = MagicMock()
    store.store.return_value = "mem-1"
    store.get_by_id.return_value = None
    embedder = MagicMock()
    embedder.embed_task_ir.return_value = [0.0]
    scorer = QScorer(store=store, embedder=embedder, logger=MagicMock(),
                     reader=ProgressReader(log_dir=tmp_path))
    scorer._score_task("t1")
    assert store.store.call_count == 1
    return store.store.call_args.kwargs["context"]


def test_flag_on_stamps_source_in_extra(tmp_path, monkeypatch):
    ctx = _scored_context(tmp_path, monkeypatch, True, "autopilot")
    assert "autopilot" in repr(ctx)


def test_flag_off_leaves_row_unlabelled(tmp_path, monkeypatch):
    ctx = _scored_context(tmp_path, monkeypatch, False, "autopilot")
    assert "autopilot" not in repr(ctx)
