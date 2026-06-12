"""Tests for src/api/routes/chat_routing.py.

Covers: _select_mode, _should_use_direct, _classify_and_route,
tool requirement detection.
"""

from unittest.mock import MagicMock, patch


from src.api.routes.chat_routing import (
    _classify_and_route,
    _select_mode,
    _should_use_direct,
)


# ── _should_use_direct ───────────────────────────────────────────────────


class TestShouldUseDirect:
    """Test the direct-mode heuristic for simple questions."""

    def test_mcq_detected(self):
        prompt = (
            "What is the capital of France?\n"
            "A) Paris\n"
            "B) London\n"
            "C) Berlin\n"
            "D) Madrid"
        )
        assert _should_use_direct(prompt, None) is True

    def test_mcq_parenthetical(self):
        prompt = (
            "Which element has atomic number 6?\n"
            "(A) Oxygen\n"
            "(B) Carbon\n"
            "(C) Nitrogen\n"
            "(D) Hydrogen"
        )
        assert _should_use_direct(prompt, None) is True

    def test_short_factual_question(self):
        assert _should_use_direct("What is the speed of light?", None) is True

    def test_short_factual_who(self):
        assert _should_use_direct("Who wrote Romeo and Juliet?", None) is True

    def test_coding_task_rejected(self):
        assert _should_use_direct("Implement a binary search function", None) is False

    def test_code_block_rejected(self):
        assert _should_use_direct("Fix this code:\n```\nx = 1\n```", None) is False

    def test_long_prompt_rejected(self):
        long_prompt = "What is X? " * 500  # > 4000 chars
        assert _should_use_direct(long_prompt, None) is False

    def test_long_context_rejected(self):
        assert _should_use_direct("What is this?", "x" * 9000) is False

    def test_research_task_rejected(self):
        assert _should_use_direct("Research the history of AI", None) is False

    def test_step_by_step_rejected(self):
        assert _should_use_direct("Explain step by step how to sort a list", None) is False

    def test_non_question_not_matched(self):
        assert _should_use_direct("Hello there", None) is False

    def test_long_non_mcq_not_matched(self):
        assert _should_use_direct("Tell me everything about quantum physics in detail", None) is False


# ── _select_mode ─────────────────────────────────────────────────────────


class TestSelectMode:
    """Test execution mode selection."""

    def test_returns_repl_by_default(self):
        state = MagicMock(hybrid_router=None)
        assert _select_mode("hello", "", state) == "repl"

    def test_returns_direct_for_mcq(self):
        state = MagicMock(hybrid_router=None)
        prompt = "Q?\nA) X\nB) Y\nC) Z\nD) W"
        assert _select_mode(prompt, "", state) == "direct"

    def test_returns_direct_for_short_factual(self):
        state = MagicMock(hybrid_router=None)
        assert _select_mode("What is the capital of France?", "", state) == "direct"

    def test_returns_repl_for_coding(self):
        state = MagicMock(hybrid_router=None)
        assert _select_mode("Implement a sorting algorithm", "", state) == "repl"

    def test_returns_repl_with_no_hybrid_router(self):
        state = MagicMock(spec=[])
        # "What is 2+2?" is short factual — but spec=[] means no hybrid_router attr
        # The heuristic still runs before the MemRL check
        assert _select_mode("What is 2+2?", "", state) == "direct"

    def test_uses_hybrid_router_when_available(self):
        router = MagicMock()
        router.route_with_mode.return_value = (["frontdoor"], "memrl", "direct")
        state = MagicMock(hybrid_router=router)
        assert _select_mode("prompt", "", state) == "direct"

    def test_maps_react_to_repl(self):
        router = MagicMock()
        router.route_with_mode.return_value = (["frontdoor"], "memrl", "react")
        state = MagicMock(hybrid_router=router)
        assert _select_mode("prompt", "", state) == "repl"

    def test_falls_back_on_hybrid_router_error(self):
        router = MagicMock()
        router.route_with_mode.side_effect = RuntimeError("fail")
        state = MagicMock(hybrid_router=router)
        assert _select_mode("prompt", "", state) == "repl"

    def test_rejects_invalid_mode_from_router(self):
        router = MagicMock()
        router.route_with_mode.return_value = (["frontdoor"], "memrl", "invalid")
        state = MagicMock(hybrid_router=router)
        assert _select_mode("prompt", "", state) == "repl"


# ── _classify_and_route ──────────────────────────────────────────────────


class TestClassifyAndRoute:
    """Test intent classification and routing."""

    @patch("src.classifiers.classify_and_route")
    def test_returns_role_and_strategy(self, mock_classify):
        mock_result = MagicMock(role="coder_escalation", strategy="keyword")
        mock_classify.return_value = mock_result
        role, strategy = _classify_and_route("Write a function")
        assert role == "coder_escalation"
        assert strategy == "keyword"

    @patch("src.classifiers.classify_and_route")
    def test_passes_image_flag(self, mock_classify):
        mock_result = MagicMock(role="worker_vision", strategy="keyword")
        mock_classify.return_value = mock_result
        _classify_and_route("Describe this", has_image=True)
        mock_classify.assert_called_once_with("Describe this", "", True)
