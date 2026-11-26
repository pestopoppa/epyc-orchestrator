"""Verbatim log coexistence with normal compaction and arm conflict guard."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.features import Features, reset_features, set_features
from src.graph import context_arm_ab as ab
from src.graph.compaction import _maybe_compact_context
from src.graph.session_summary import _init_session_log, _record_session_turn
from src.graph.state import TaskState
from src.repl_environment import REPLEnvironment


@pytest.fixture
def case(tmp_path, monkeypatch):
    """Normal-compaction fixture: session_compaction on, context_arm_ab off."""
    monkeypatch.setenv("ORCHESTRATOR_CONTEXT_ARM", "")
    monkeypatch.setenv("ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR", str(tmp_path / "trajectory"))
    monkeypatch.setenv("ORCHESTRATOR_SESSION_SUMMARY_PROMPT", "openclaw_0e7b5c3")
    monkeypatch.setenv("ORCHESTRATOR_CONTEXT_BUDGET_TOKENS", "16384")
    monkeypatch.setenv("ORCHESTRATOR_CONTEXT_TRIGGER_FRACTION", ".5")
    monkeypatch.setattr(
        "src.repl_environment.session_artifacts.session_artifact_root",
        lambda: tmp_path / "tmp",
    )
    set_features(
        Features(
            session_log=True,
            session_log_verbatim=True,
            session_compaction=True,
            output_spill_to_file=True,
            trajectory_artifact=True,
            trace_navigation_tools=True,
        )
    )
    repl = REPLEnvironment(context="test", task_id="x")
    repl._session_artifact_dir.mkdir(parents=True, exist_ok=True)
    state = TaskState(task_id="x", context="x" * 15000)
    _init_session_log(state, repl=repl)
    # Provide a dummy primitives so _maybe_compact_context doesn't bail early
    state._mock_primitives = SimpleNamespace(_count_tokens=lambda t: len(t) // 4)
    yield state, repl
    reset_features()


def test_verbatim_log_with_normal_compaction_is_owner_only(case):
    """Verbatim log written via _record_session_turn alongside unforced session_compaction;
    grep_history scoped to owning task only."""
    state, repl = case

    # Record turns through the normal session_summary writer (not ab.record_turn directly)
    for i in range(5):
        state.turns += 1
        _record_session_turn(
            state,
            role="worker",
            code=f"print({i})",
            output=f"item-{i} " + "x" * 200,
            error=None,
        )

    # Build a context namespace with primitives so compaction runs naturally
    ctx = SimpleNamespace(
        state=state,
        deps=SimpleNamespace(repl=repl, primitives=state._mock_primitives),
    )
    # Unforced: context is already >12000 chars, so compaction triggers on that threshold
    asyncio.run(_maybe_compact_context(ctx))

    assert state.compaction_count >= 1
    verbatim_path = Path(state.session_log_path).with_name("session.verbatim.md")
    assert verbatim_path.exists() and verbatim_path.stat().st_size > 0

    # History log path was set by ab.record_turn (called from _record_session_turn)
    assert state.history_log_path == str(verbatim_path)

    # After compaction, history_search.install wired grep_history for the owner
    assert "grep_history" in repl._globals

    # Owner can search
    result = repl._grep_history("item-0")
    page = json.loads(result)
    assert page["matches"]
    assert page["matches"][0]["line"] > 0

    # A second task cannot register or search this artifact
    repl2 = REPLEnvironment(context="test", task_id="y")
    assert "grep_history" not in repl2._globals

    # Cross-owner registration attempt is refused
    registered = repl2._register_session_artifact(str(verbatim_path), "y")
    assert registered is False

    # No foreign history path is installed
    assert not hasattr(repl2, "_history_search_path")

    # Cross-owner grep_history attempt: the tool simply does not exist
    with pytest.raises((KeyError, AttributeError)):
        repl2._grep_history("item-0")


def test_context_arm_refuses_normal_session_compaction(case, monkeypatch):
    """With an otherwise-valid arm setup, session_compaction confounder is refused."""
    state, repl = case
    set_features(
        Features(
            session_log=True,
            session_log_verbatim=True,
            context_arm_ab=True,
            session_compaction=True,
            output_spill_to_file=True,
            trajectory_artifact=True,
        )
    )
    monkeypatch.setenv("ORCHESTRATOR_CONTEXT_ARM", "pointer_only")
    with pytest.raises(ValueError, match="session_compaction"):
        ab.settings()
