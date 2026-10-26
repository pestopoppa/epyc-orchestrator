"""HIL-16: the real _execute_turn records the model turn (prompt bytes and raw output) when, and only when,
replay_recording is on."""
from __future__ import annotations

import sqlite3

import pytest
from pydantic_graph import GraphRunContext

from src.features import reset_features
from src.graph.helpers import _execute_turn
from src.roles import Role
from src.trace import store as trace_store
from src.trace.replay.cas import CAS
from src.trace.replay.model import load_run, verify_integrity

pytestmark = pytest.mark.integration


def _setup(monkeypatch, tmp_path, on):
    db = tmp_path / "e.sqlite"
    trace_store.ensure_schema(db).close()
    monkeypatch.setattr(trace_store, "DEFAULT_DB_PATH", db)
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.delenv("ORCHESTRATOR_STEP_EVENT_LOGGING", raising=False)
    if on:
        monkeypatch.setenv("ORCHESTRATOR_REPLAY_RECORDING", "1")
    else:
        monkeypatch.delenv("ORCHESTRATOR_REPLAY_RECORDING", raising=False)
    reset_features()
    return db


async def test_turn_recorded_when_on(graph_ctx, monkeypatch, tmp_path):
    db = _setup(monkeypatch, tmp_path, True)
    state, deps = graph_ctx(responses=["x = 2 + 2\nFINAL(x)"])
    await _execute_turn(GraphRunContext(state=state, deps=deps), Role.FRONTDOOR)
    run_id = str(deps.repl._session_id)
    run = load_run(db, run_id)
    cas = CAS(tmp_path / "cas")
    assert len(run.turns) == 1 and verify_integrity(run, cas) == []
    assert cas.get_text(run.turns[0].output_cas) == "x = 2 + 2\nFINAL(x)"
    assert len(cas.get_text(run.turns[0].prompt_cas)) > 0
    assert run.turns[0].role == "frontdoor"


async def test_nothing_recorded_when_off(graph_ctx, monkeypatch, tmp_path):
    db = _setup(monkeypatch, tmp_path, False)
    state, deps = graph_ctx(responses=["FINAL(1)"])
    await _execute_turn(GraphRunContext(state=state, deps=deps), Role.FRONTDOOR)
    assert deps.repl._replay_recorder is None
    assert sqlite3.connect(db).execute("select count(*) from event").fetchone()[0] == 0
    assert not (tmp_path / "cas").exists()
