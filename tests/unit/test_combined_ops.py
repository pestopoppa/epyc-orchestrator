#!/usr/bin/env python3
"""Unit tests for REPL combined operations (_CombinedOpsMixin)."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.repl_environment import REPLEnvironment


@pytest.fixture(autouse=True)
def _clear_feature_flag():
    """Ensure REPL_COMBINED_OPS is unset before/after each test."""
    old = os.environ.pop("REPL_COMBINED_OPS", None)
    yield
    if old is not None:
        os.environ["REPL_COMBINED_OPS"] = old
    else:
        os.environ.pop("REPL_COMBINED_OPS", None)


def _make_repl_with_registry(use_toon=False):
    """Create a REPLEnvironment with a mocked tool_registry."""
    from src.repl_environment.types import REPLConfig

    config = REPLConfig(use_toon_encoding=use_toon)
    registry = MagicMock()
    repl = REPLEnvironment(
        context="test", tool_registry=registry, role="worker_general", config=config,
    )
    return repl, registry


def _make_repl(use_toon=False):
    """Create a plain REPLEnvironment with TOON disabled for JSON tests."""
    from src.repl_environment.types import REPLConfig

    config = REPLConfig(use_toon_encoding=use_toon)
    return REPLEnvironment(context="test", config=config)


class TestBatchWebSearchDisabled:
    """Tests for batch_web_search when feature flag is off."""

    def test_batch_web_search_disabled(self):
        """Flag off returns feature disabled message."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["r"] = batch_web_search(["q1", "q2"])')
        assert result.error is None
        assert "Combined ops disabled" in repl.artifacts["r"]

    def test_batch_web_search_disabled_explicit_off(self):
        """Explicit '0' value keeps feature disabled."""
        os.environ["REPL_COMBINED_OPS"] = "0"
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["r"] = batch_web_search(["q1"])')
        assert result.error is None
        assert "Combined ops disabled" in repl.artifacts["r"]


class TestBatchWebSearchEnabled:
    """Tests for batch_web_search when feature flag is on."""

    def test_batch_web_search_enabled(self):
        """Flag on, mock tool registry, returns consolidated results."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl, registry = _make_repl_with_registry(use_toon=False)

        registry.invoke.return_value = [{"title": "Result", "url": "https://example.com"}]

        result = repl._batch_web_search(["python asyncio", "rust tokio"], max_results=2)
        parsed = json.loads(result)

        assert parsed["operation"] == "batch_web_search"
        assert parsed["query_count"] == 2
        assert "python asyncio" in parsed["results"]
        assert "rust tokio" in parsed["results"]
        assert registry.invoke.call_count == 2

    def test_batch_web_search_caps_at_5(self):
        """More than 5 queries gets capped to 5."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl, registry = _make_repl_with_registry(use_toon=False)

        registry.invoke.return_value = []
        queries = [f"query_{i}" for i in range(8)]

        result = repl._batch_web_search(queries)
        parsed = json.loads(result)

        assert parsed["query_count"] == 5
        assert parsed["capped"] is True
        assert registry.invoke.call_count == 5

    def test_batch_web_search_empty_queries(self):
        """Empty query list returns error."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl = _make_repl()
        result = repl._batch_web_search([])
        assert "[ERROR: No queries provided]" in result

    def test_batch_web_search_no_registry(self):
        """No tool registry returns error."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl = _make_repl()
        result = repl._batch_web_search(["query"])
        assert "No tool registry" in result

    def test_batch_web_search_handles_tool_error(self):
        """Individual tool errors are captured per-query."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl, registry = _make_repl_with_registry(use_toon=False)

        registry.invoke.side_effect = [
            [{"title": "OK"}],
            RuntimeError("search failed"),
        ]

        result = repl._batch_web_search(["good_query", "bad_query"])
        parsed = json.loads(result)

        assert "good_query" in parsed["results"]
        assert "ERROR" in parsed["results"]["bad_query"]

    def test_batch_web_search_toon_encoding(self):
        """TOON encoding produces text format."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl, registry = _make_repl_with_registry(use_toon=True)

        registry.invoke.return_value = [{"title": "Result"}]

        result = repl._batch_web_search(["test query"])
        assert "=== batch_web_search" in result
        assert '--- "test query" ---' in result


class TestSearchAndVerify:
    """Tests for search_and_verify."""

    def test_search_and_verify_disabled(self):
        """Flag off returns feature disabled message."""
        repl = _make_repl()
        result = repl._search_and_verify("test query")
        assert "Combined ops disabled" in result

    def test_search_and_verify_enabled(self):
        """Returns both web and wikipedia results."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl, registry = _make_repl_with_registry(use_toon=False)

        def invoke_side_effect(tool_name, role, **kwargs):
            if tool_name == "web_search":
                return [{"title": "Web Result", "url": "https://example.com"}]
            elif tool_name == "search_wikipedia":
                return {"summary": "Wikipedia summary text"}
            return None

        registry.invoke.side_effect = invoke_side_effect

        result = repl._search_and_verify("quantum computing", max_results=2)
        parsed = json.loads(result)

        assert parsed["operation"] == "search_and_verify"
        assert parsed["query"] == "quantum computing"
        assert parsed["web_results"] is not None
        assert parsed["wikipedia"] is not None
        assert registry.invoke.call_count == 2

    def test_search_and_verify_no_registry(self):
        """No tool registry returns error."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl = _make_repl()
        result = repl._search_and_verify("test")
        assert "No tool registry" in result

    def test_search_and_verify_toon_encoding(self):
        """TOON encoding produces labeled sections."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl, registry = _make_repl_with_registry(use_toon=True)

        registry.invoke.return_value = "some result"

        result = repl._search_and_verify("test")
        assert "## Web Results" in result
        assert "## Wikipedia" in result


class TestPeekGrep:
    """Tests for peek_grep."""

    def test_peek_grep_disabled(self):
        """Flag off returns feature disabled message."""
        repl = _make_repl()
        result = repl._peek_grep("/tmp/test.txt", "pattern")
        assert "Combined ops disabled" in result

    def test_peek_grep_enabled(self):
        """Reads file, applies pattern, returns matches with context."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl = _make_repl(use_toon=False)

        content = "line 1\nline 2\nfoo target bar\nline 4\nline 5\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", dir="/tmp", delete=False
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            result = repl._peek_grep(tmp_path, "target", context_lines=1)
            parsed = json.loads(result)

            assert parsed["operation"] == "peek_grep"
            assert parsed["match_count"] == 1
            assert len(parsed["matches"]) == 1
            # The match block should contain the target line and context
            assert "target" in parsed["matches"][0]
            assert ">>>" in parsed["matches"][0]
        finally:
            os.unlink(tmp_path)

    def test_peek_grep_no_matches(self):
        """Pattern not found returns empty matches message."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl = _make_repl()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", dir="/tmp", delete=False
        ) as f:
            f.write("nothing interesting here\n")
            tmp_path = f.name

        try:
            result = repl._peek_grep(tmp_path, "nonexistent_pattern")
            assert "No matches" in result
        finally:
            os.unlink(tmp_path)

    def test_peek_grep_file_not_found(self):
        """Missing file returns error."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl = _make_repl()
        result = repl._peek_grep("/tmp/does_not_exist_xyz.txt", "pattern")
        assert "File not found" in result

    def test_peek_grep_invalid_regex(self):
        """Invalid regex returns error."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl = _make_repl()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", dir="/tmp", delete=False
        ) as f:
            f.write("content\n")
            tmp_path = f.name

        try:
            result = repl._peek_grep(tmp_path, "[invalid")
            assert "Invalid regex" in result
        finally:
            os.unlink(tmp_path)

    def test_peek_grep_toon_encoding(self):
        """TOON encoding produces text format."""
        os.environ["REPL_COMBINED_OPS"] = "1"
        repl = _make_repl(use_toon=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", dir="/tmp", delete=False
        ) as f:
            f.write("hello world\n")
            tmp_path = f.name

        try:
            result = repl._peek_grep(tmp_path, "hello")
            assert "=== peek_grep:" in result
            assert "1 matches" in result
        finally:
            os.unlink(tmp_path)
