#!/usr/bin/env python3
"""Tests for CC Local MCP chat delegation tools.

Tests tool functions directly (bypassing MCP transport).
All HTTP calls and feature flags are mocked.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from src.mcp_server import (
    _format_chat_response,
    _post_chat,
    orchestrator_chat,
    orchestrator_route_explain,
)


# ---------------------------------------------------------------------------
# _format_chat_response unit tests
# ---------------------------------------------------------------------------


class TestFormatChatResponse:
    def test_success_response(self):
        resp = {
            "answer": "The answer is 42.",
            "routed_to": "frontdoor",
            "routing_strategy": "learned",
            "mode": "direct",
            "elapsed_seconds": 2.5,
        }
        result = _format_chat_response(resp)
        assert "The answer is 42." in result
        assert "role=frontdoor" in result
        assert "strategy=learned" in result
        assert "elapsed=2.5s" in result

    def test_error_in_response(self):
        resp = {"error": "Connection refused"}
        result = _format_chat_response(resp)
        assert "Error: Connection refused" in result

    def test_orchestrator_error_code(self):
        resp = {
            "answer": "",
            "error_code": 504,
            "error_detail": "Backend timeout",
        }
        result = _format_chat_response(resp)
        assert "[Error 504]" in result
        assert "Backend timeout" in result

    def test_empty_response(self):
        result = _format_chat_response({})
        assert result == "(empty response)"


# ---------------------------------------------------------------------------
# _post_chat unit tests
# ---------------------------------------------------------------------------


class TestPostChat:
    @patch("src.mcp_server._get_api_url", return_value="http://localhost:8000")
    @patch("urllib.request.urlopen")
    def test_success(self, mock_urlopen, mock_url):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"answer": "ok"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _post_chat({"prompt": "test", "timeout_s": 30})
        assert result == {"answer": "ok"}

    @patch("src.mcp_server._get_api_url", return_value="http://localhost:8000")
    @patch("urllib.request.urlopen", side_effect=URLError("Connection refused"))
    def test_connection_error(self, mock_urlopen, mock_url):
        result = _post_chat({"prompt": "test"})
        assert "error" in result
        assert "not reachable" in result["error"]

    @patch("src.mcp_server._get_api_url", return_value="http://localhost:8000")
    @patch("urllib.request.urlopen", side_effect=TimeoutError)
    def test_timeout(self, mock_urlopen, mock_url):
        result = _post_chat({"prompt": "test", "timeout_s": 5})
        assert "error" in result
        assert "timed out" in result["error"]


# ---------------------------------------------------------------------------
# orchestrator_chat tool tests
# ---------------------------------------------------------------------------


class TestOrchestratorChat:
    @patch.dict(os.environ, {"ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT": "0"}, clear=False)
    def test_feature_disabled(self):
        result = orchestrator_chat("Hello")
        assert "disabled" in result.lower()
        assert "ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT" in result

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat")
    def test_success(self, mock_post, mock_enabled):
        mock_post.return_value = {
            "answer": "The answer is 42.",
            "routed_to": "coder_escalation",
            "routing_strategy": "learned",
            "mode": "repl",
            "elapsed_seconds": 3.1,
        }
        result = orchestrator_chat("What is 6 times 7?")
        assert "42" in result
        assert "coder_escalation" in result
        # Verify payload
        call_args = mock_post.call_args[0][0]
        assert call_args["prompt"] == "What is 6 times 7?"
        assert call_args["real_mode"] is True
        assert call_args["mock_mode"] is False

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat")
    def test_force_role(self, mock_post, mock_enabled):
        mock_post.return_value = {"answer": "done", "routed_to": "architect_coding"}
        orchestrator_chat("Fix the bug", force_role="architect_coding")
        call_args = mock_post.call_args[0][0]
        assert call_args["force_role"] == "architect_coding"

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat")
    def test_connection_error(self, mock_post, mock_enabled):
        mock_post.return_value = {"error": "Orchestrator not reachable at http://localhost:8000: Connection refused"}
        result = orchestrator_chat("Hello")
        assert "not reachable" in result

    @patch.dict(os.environ, {
        "ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT": "1",
        "ORCHESTRATOR_API_URL": "http://localhost:9999",
    }, clear=False)
    @patch("src.mcp_server._ORCHESTRATOR_API_URL", None)  # Reset cached value
    @patch("src.mcp_server._post_chat")
    def test_custom_api_url(self, mock_post):
        mock_post.return_value = {"answer": "ok"}
        # _get_api_url should pick up the env var
        from src.mcp_server import _get_api_url
        # Reset the cached value to force re-read
        import src.mcp_server
        src.mcp_server._ORCHESTRATOR_API_URL = None
        url = _get_api_url()
        assert url == "http://localhost:9999"
        # Clean up
        src.mcp_server._ORCHESTRATOR_API_URL = None


# ---------------------------------------------------------------------------
# orchestrator_route_explain tool tests
# ---------------------------------------------------------------------------


class TestOrchestratorRouteExplain:
    @patch.dict(os.environ, {"ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT": "0"}, clear=False)
    def test_feature_disabled(self):
        result = orchestrator_route_explain("Hello")
        assert "disabled" in result.lower()

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat")
    def test_success(self, mock_post, mock_enabled):
        mock_post.return_value = {
            "routed_to": "architect_general",
            "routing_strategy": "rules",
            "mode": "delegated",
            "tool_required": True,
            "timeout_s": 60,
        }
        result = orchestrator_route_explain("Explain quantum computing")
        assert "architect_general" in result
        assert "rules" in result
        assert "delegated" in result
        assert "True" in result
        # Verify mock_mode was set
        call_args = mock_post.call_args[0][0]
        assert call_args["mock_mode"] is True
        assert call_args["real_mode"] is False

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat")
    def test_connection_error(self, mock_post, mock_enabled):
        mock_post.return_value = {"error": "Orchestrator not reachable"}
        result = orchestrator_route_explain("Hello")
        assert "Error:" in result
