"""Integration tests for failure recording and observability.

Tests _record_failure, _add_evidence, _record_mitigation, _log_escalation
with real stub implementations (not MagicMock).

Target: observability.py 80% → 95%+
"""

from __future__ import annotations

import pytest
from pydantic_graph import GraphRunContext

from src.escalation import ErrorCategory
from src.graph.observability import (
    _add_evidence,
    _log_escalation,
    _record_failure,
    _record_mitigation,
)
from src.graph.decision_gates import _make_end_result
from src.graph.state import TaskDeps, TaskState
from src.roles import Role

from .conftest import StubFailureGraph, StubHypothesisGraph

pytestmark = pytest.mark.integration

Ctx = GraphRunContext[TaskState, TaskDeps]


def _make_ctx(state: TaskState, deps: TaskDeps) -> Ctx:
    return GraphRunContext(state=state, deps=deps)


# ── _record_failure ───────────────────────────────────────────────────


class TestRecordFailure:
    """Tests for _record_failure with real StubFailureGraph."""

    def test_records_failure_with_graph(self, graph_ctx):
        """Failure recorded when failure_graph is present."""
        state, deps = graph_ctx(with_failure_graph=True)
        state.task_id = "test-task"
        state.current_role = Role.FRONTDOOR
        state.consecutive_failures = 1
        ctx = _make_ctx(state, deps)

        fid = _record_failure(ctx, ErrorCategory.CODE, "NameError: x is not defined")

        assert fid is not None
        fg = deps.failure_graph
        assert len(fg.failures) == 1
        assert fg.failures[0]["memory_id"] == "test-task"
        assert "code" in fg.failures[0]["symptoms"]
        assert "NameError" in fg.failures[0]["description"]
        assert fg.failures[0]["severity"] == 3  # consecutive_failures(1) + 2

    def test_returns_none_without_graph(self, graph_ctx):
        """Returns None when failure_graph is None."""
        state, deps = graph_ctx(with_failure_graph=False)
        ctx = _make_ctx(state, deps)

        result = _record_failure(ctx, ErrorCategory.CODE, "some error")

        assert result is None

    def test_severity_scales_with_failures(self, graph_ctx):
        """Severity increases with consecutive failures, capped at 5."""
        state, deps = graph_ctx(with_failure_graph=True)
        ctx = _make_ctx(state, deps)

        for n in range(4):
            state.consecutive_failures = n
            _record_failure(ctx, ErrorCategory.CODE, f"error {n}")

        fg = deps.failure_graph
        severities = [f["severity"] for f in fg.failures]
        assert severities == [2, 3, 4, 5]  # min(n+2, 5)

    def test_updates_last_failure_id(self, graph_ctx):
        """Recording a failure sets state.last_failure_id."""
        state, deps = graph_ctx(with_failure_graph=True)
        ctx = _make_ctx(state, deps)

        fid = _record_failure(ctx, ErrorCategory.LOGIC, "bad logic")

        assert state.last_failure_id == fid

    def test_truncates_error_message(self, graph_ctx):
        """Long error messages are truncated in symptoms and description."""
        state, deps = graph_ctx(with_failure_graph=True)
        ctx = _make_ctx(state, deps)

        long_error = "x" * 500
        _record_failure(ctx, ErrorCategory.CODE, long_error)

        fg = deps.failure_graph
        assert len(fg.failures[0]["symptoms"][1]) <= 100
        assert len(fg.failures[0]["description"]) <= 250  # role prefix + 200


# ── _add_evidence ─────────────────────────────────────────────────────


class TestAddEvidence:
    """Tests for _add_evidence with real StubHypothesisGraph."""

    def test_records_success_evidence(self, graph_ctx):
        """Success outcome is recorded."""
        state, deps = graph_ctx(with_hypothesis_graph=True)
        state.task_id = "test-task"
        state.current_role = Role.FRONTDOOR
        state.turns = 3
        ctx = _make_ctx(state, deps)

        _add_evidence(ctx, "success")

        hg = deps.hypothesis_graph
        assert len(hg.evidence) == 1
        assert hg.evidence[0]["outcome"] == "success"
        assert hg.evidence[0]["source"] == "frontdoor:turn_3"

    def test_records_failure_evidence(self, graph_ctx):
        """Non-success outcome normalized to 'failure'."""
        state, deps = graph_ctx(with_hypothesis_graph=True)
        state.current_role = Role.THINKING_REASONING
        state.turns = 2
        ctx = _make_ctx(state, deps)

        _add_evidence(ctx, "error")

        hg = deps.hypothesis_graph
        assert hg.evidence[0]["outcome"] == "failure"

    def test_noop_without_graph(self, graph_ctx):
        """No error raised when hypothesis_graph is None."""
        state, deps = graph_ctx(with_hypothesis_graph=False)
        ctx = _make_ctx(state, deps)

        # Should not raise
        _add_evidence(ctx, "success")

    def test_confidence_updates(self, graph_ctx):
        """Confidence increases on success, decreases on failure."""
        state, deps = graph_ctx(with_hypothesis_graph=True)
        ctx = _make_ctx(state, deps)

        _add_evidence(ctx, "success")
        c1 = deps.hypothesis_graph._confidence

        _add_evidence(ctx, "failure")
        c2 = deps.hypothesis_graph._confidence

        assert c1 > 0.5  # Started at 0.5, success increased it
        assert c2 < c1   # Failure decreased it


# ── _record_mitigation ────────────────────────────────────────────────


class TestRecordMitigation:
    """Tests for _record_mitigation with real StubFailureGraph."""

    def test_records_mitigation(self, graph_ctx):
        """Mitigation recorded with correct failure_id."""
        state, deps = graph_ctx(with_failure_graph=True)
        ctx = _make_ctx(state, deps)

        # Record a failure first
        fid = _record_failure(ctx, ErrorCategory.CODE, "error")

        # Record mitigation
        _record_mitigation(ctx, "frontdoor", "coder_escalation", failure_id=fid)

        fg = deps.failure_graph
        assert len(fg.mitigations) == 1
        assert fg.mitigations[0]["failure_id"] == fid
        assert fg.mitigations[0]["action"] == "escalate:frontdoor->coder_escalation"
        assert fg.mitigations[0]["worked"] is True

    def test_uses_last_failure_id_as_fallback(self, graph_ctx):
        """When no failure_id passed, uses state.last_failure_id."""
        state, deps = graph_ctx(with_failure_graph=True)
        ctx = _make_ctx(state, deps)

        fid = _record_failure(ctx, ErrorCategory.CODE, "error")
        assert state.last_failure_id == fid

        _record_mitigation(ctx, "coder", "architect")

        fg = deps.failure_graph
        assert len(fg.mitigations) == 1
        assert fg.mitigations[0]["failure_id"] == fid

    def test_noop_without_failure_graph(self, graph_ctx):
        """No error when failure_graph is None."""
        state, deps = graph_ctx(with_failure_graph=False)
        ctx = _make_ctx(state, deps)

        _record_mitigation(ctx, "a", "b")

    def test_noop_without_failure_id(self, graph_ctx):
        """No mitigation recorded when no failure_id available."""
        state, deps = graph_ctx(with_failure_graph=True)
        state.last_failure_id = None
        ctx = _make_ctx(state, deps)

        _record_mitigation(ctx, "a", "b")

        fg = deps.failure_graph
        assert len(fg.mitigations) == 0


# ── _log_escalation ──────────────────────────────────────────────────


class TestLogEscalation:
    """Tests for _log_escalation."""

    def test_logs_escalation_with_progress_logger(self, graph_ctx):
        """Escalation logged via progress_logger.log_escalation."""
        from unittest.mock import MagicMock

        state, deps = graph_ctx()
        state.task_id = "test-task"
        pl = MagicMock()
        deps.progress_logger = pl
        ctx = _make_ctx(state, deps)

        _log_escalation(ctx, "frontdoor", "coder_escalation", "test reason")

        pl.log_escalation.assert_called_once_with(
            task_id="test-task",
            from_tier="frontdoor",
            to_tier="coder_escalation",
            reason="test reason",
        )

    def test_noop_without_logger(self, graph_ctx):
        """No error when progress_logger is None."""
        state, deps = graph_ctx()
        deps.progress_logger = None
        ctx = _make_ctx(state, deps)

        _log_escalation(ctx, "a", "b", "reason")


# ── _make_end_result ──────────────────────────────────────────────────


class TestMakeEndResult:
    """Tests for _make_end_result with real observability stubs."""

    @pytest.mark.asyncio
    async def test_end_result_records_success_evidence(self, graph_ctx):
        """_make_end_result records success evidence in hypothesis graph."""
        state, deps = graph_ctx(with_hypothesis_graph=True)
        state.turns = 2
        state.record_role(Role.FRONTDOOR)
        ctx = _make_ctx(state, deps)

        result = _make_end_result(ctx, "the answer", True)

        hg = deps.hypothesis_graph
        assert len(hg.evidence) >= 1
        assert any(e["outcome"] == "success" for e in hg.evidence)

    @pytest.mark.asyncio
    async def test_end_result_records_failure_evidence(self, graph_ctx):
        """_make_end_result records failure evidence."""
        state, deps = graph_ctx(with_hypothesis_graph=True)
        state.turns = 3
        state.record_role(Role.FRONTDOOR)
        ctx = _make_ctx(state, deps)

        result = _make_end_result(ctx, "[FAILED]", False)

        hg = deps.hypothesis_graph
        assert any(e["outcome"] == "failure" for e in hg.evidence)

    @pytest.mark.asyncio
    async def test_end_result_updates_workspace_decisions(self, graph_ctx):
        """_make_end_result appends to workspace_state decisions."""
        state, deps = graph_ctx()
        state.turns = 1
        state.record_role(Role.FRONTDOOR)
        ctx = _make_ctx(state, deps)

        _make_end_result(ctx, "answer", True)

        ws = state.workspace_state
        assert len(ws["decisions"]) >= 1
        assert ws["decisions"][-1]["rationale"] == "success"

    @pytest.mark.asyncio
    async def test_end_result_contains_metadata(self, graph_ctx):
        """End result includes turns, role_history, delegation_events."""
        state, deps = graph_ctx()
        state.turns = 5
        state.record_role(Role.FRONTDOOR)
        state.record_role(Role.THINKING_REASONING)
        state.delegation_events.append({"type": "escalation", "from": "frontdoor"})
        ctx = _make_ctx(state, deps)

        result = _make_end_result(ctx, "final answer", True)

        assert result.data.turns == 5
        assert len(result.data.role_history) == 2
        assert len(result.data.delegation_events) == 1
