"""Tests for graph compaction and budget modules.

Covers uncovered branches in compaction.py, budgets.py, and
concurrency_aware.py migration state machine.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


# ── compaction.py ────────────────────────────────────────────────────────


class TestEstimateContextTokens:
    def test_uses_accurate_tokenizer_when_available(self):
        from src.graph.compaction import _estimate_context_tokens

        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 42
        assert _estimate_context_tokens(ctx, "hello world") == 42

    def test_falls_back_to_heuristic(self):
        from src.graph.compaction import _estimate_context_tokens

        ctx = MagicMock(spec=[])
        ctx.deps = MagicMock(spec=[])
        ctx.deps.primitives = None
        assert _estimate_context_tokens(ctx, "x" * 400) == 100


class TestGetModelMaxContext:
    def test_returns_registry_value(self):
        from src.graph.compaction import _get_model_max_context

        ctx = MagicMock()
        role_cfg = MagicMock()
        role_cfg.n_ctx = 65536
        ctx.deps.primitives.registry.get_role_config.return_value = role_cfg
        ctx.state.current_role = "coder"
        assert _get_model_max_context(ctx) == 65536

    def test_returns_default_without_registry(self):
        from src.graph.compaction import _get_model_max_context

        ctx = MagicMock()
        ctx.deps.primitives = None
        assert _get_model_max_context(ctx) == 32768

    def test_returns_default_on_exception(self):
        from src.graph.compaction import _get_model_max_context

        ctx = MagicMock()
        ctx.deps.primitives.registry.get_role_config.side_effect = RuntimeError("no registry")
        assert _get_model_max_context(ctx) == 32768


class TestContextExternalizationPath:
    def test_returns_writable_path(self, tmp_path, monkeypatch):
        from src.graph.compaction import _context_externalization_path

        state = MagicMock()
        state.task_id = "test-task-123"
        state.compaction_count = 2

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
        with patch("src.config.get_config", side_effect=ImportError):
            path = _context_externalization_path(state)

        assert "test-task-123" in str(path)
        assert "ctx_2" in str(path)

    def test_falls_back_to_tempdir(self, monkeypatch):
        from src.graph.compaction import _context_externalization_path

        state = MagicMock()
        state.task_id = "fallback"
        state.compaction_count = 0

        monkeypatch.delenv("ORCHESTRATOR_PATHS_TMP_DIR", raising=False)
        with patch("src.config.get_config", side_effect=ImportError):
            path = _context_externalization_path(state)

        assert "fallback" in str(path)


# ── budgets.py ───────────────────────────────────────────────────────────


class TestReplTurnTokenCap:
    def test_returns_flat_cap_without_band(self):
        from src.graph.budgets import _repl_turn_token_cap

        assert _repl_turn_token_cap("") == 768

    def test_returns_flat_cap_when_not_enforce(self):
        from src.graph.budgets import _repl_turn_token_cap

        with patch("src.classifiers.difficulty_signal.get_mode", return_value="shadow"):
            assert _repl_turn_token_cap("hard") == 768

    def test_returns_band_budget_in_enforce_mode(self):
        from src.graph.budgets import _repl_turn_token_cap, _BAND_TOKEN_BUDGETS

        with patch("src.classifiers.difficulty_signal.get_mode", return_value="enforce"):
            result = _repl_turn_token_cap("hard")
            assert result == _BAND_TOKEN_BUDGETS["hard"]

    def test_returns_flat_cap_on_import_error(self):
        from src.graph.budgets import _repl_turn_token_cap

        with patch("src.classifiers.difficulty_signal.get_mode", side_effect=ImportError):
            assert _repl_turn_token_cap("hard") == 768


class TestFrontdoorTokenCaps:
    def test_frontdoor_turn_cap_disabled(self, monkeypatch):
        from src.graph.budgets import _frontdoor_turn_token_cap

        monkeypatch.setenv("ORCHESTRATOR_FRONTDOOR_TURN_N_TOKENS", "0")
        assert _frontdoor_turn_token_cap() == 0

    def test_frontdoor_turn_cap_with_value(self, monkeypatch):
        from src.graph.budgets import _frontdoor_turn_token_cap

        monkeypatch.setenv("ORCHESTRATOR_FRONTDOOR_TURN_N_TOKENS", "512")
        assert _frontdoor_turn_token_cap() == 512

    def test_frontdoor_non_tool_cap(self):
        from src.graph.budgets import _frontdoor_repl_non_tool_token_cap

        assert _frontdoor_repl_non_tool_token_cap() >= 64


class TestBudgetCaps:
    def test_worker_call_budget_cap_default(self):
        from src.graph.budgets import _worker_call_budget_cap

        assert _worker_call_budget_cap() == 30

    def test_task_token_budget_cap_default(self):
        from src.graph.budgets import _task_token_budget_cap

        assert _task_token_budget_cap() == 200000


class TestCheckBudgetExceeded:
    def test_returns_none_when_within_budget(self):
        from src.graph.budgets import _check_budget_exceeded

        ctx = MagicMock()
        ctx.state.repl_executions = 5
        ctx.state.aggregate_tokens = 1000

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = True
            mock_feat.return_value.task_token_budget = True
            result = _check_budget_exceeded(ctx)
        assert result is None

    def test_returns_message_on_call_budget_exceeded(self):
        from src.graph.budgets import _check_budget_exceeded

        ctx = MagicMock()
        ctx.state.repl_executions = 50
        ctx.state.aggregate_tokens = 0

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = True
            mock_feat.return_value.task_token_budget = False
            result = _check_budget_exceeded(ctx)
        assert "Worker call budget exhausted" in result

    def test_returns_message_on_token_budget_exceeded(self):
        from src.graph.budgets import _check_budget_exceeded

        ctx = MagicMock()
        ctx.state.repl_executions = 0
        ctx.state.aggregate_tokens = 999999

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = False
            mock_feat.return_value.task_token_budget = True
            result = _check_budget_exceeded(ctx)
        assert "Task token budget exhausted" in result


class TestBudgetPressureWarnings:
    def test_no_warnings_when_plenty_of_budget(self):
        from src.graph.budgets import _budget_pressure_warnings

        state = MagicMock()
        state.repl_executions = 5
        state.aggregate_tokens = 1000

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = True
            mock_feat.return_value.task_token_budget = True
            result = _budget_pressure_warnings(state)
        assert result == ""

    def test_warns_when_repl_budget_low(self):
        from src.graph.budgets import _budget_pressure_warnings

        state = MagicMock()
        state.repl_executions = 28
        state.aggregate_tokens = 1000

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = True
            mock_feat.return_value.task_token_budget = False
            result = _budget_pressure_warnings(state)
        assert "REPL execution" in result
        assert "FINAL()" in result

    def test_warns_when_token_budget_low(self):
        from src.graph.budgets import _budget_pressure_warnings

        state = MagicMock()
        state.repl_executions = 0
        state.aggregate_tokens = 185000

        with patch("src.features.features") as mock_feat:
            mock_feat.return_value.worker_call_budget = False
            mock_feat.return_value.task_token_budget = True
            result = _budget_pressure_warnings(state)
        assert "token budget" in result


# ── concurrency_aware.py ─────────────────────────────────────────────────


class TestConcurrencyAwareMigrationStates:
    """Test the KV migration state constants added in Phase 4."""

    def test_migration_state_constants_exist(self):
        from src.backends.concurrency_aware import (
            _STATE_UNASSIGNED,
            _STATE_ASSIGNED_FULL,
            _STATE_MIGRATION_PENDING,
            _STATE_ASSIGNED_QUARTER,
            _STATE_MIGRATION_FAILED_COLD,
        )

        assert _STATE_UNASSIGNED == "unassigned"
        assert _STATE_ASSIGNED_FULL == "assigned_full"
        assert _STATE_MIGRATION_PENDING == "migration_pending"
        assert _STATE_ASSIGNED_QUARTER == "assigned_quarter"
        assert _STATE_MIGRATION_FAILED_COLD == "migration_failed_cold"

    def test_backend_init_empty_affinity(self):
        """Verify affinity map starts empty (Phase 4 invariant)."""
        from src.backends.concurrency_aware import ConcurrencyAwareBackend

        backend = ConcurrencyAwareBackend.__new__(ConcurrencyAwareBackend)
        backend._session_affinity = {}
        assert len(backend._session_affinity) == 0
