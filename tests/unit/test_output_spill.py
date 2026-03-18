"""Tests for CMV-style output spill to file (Action 11)."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

# Ensure src is importable
sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator")

from src.features import Features, set_features, reset_features
from src.graph.helpers import _spill_if_truncated


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_features()


class _FakeState:
    """Minimal TaskState stand-in for spill tests."""

    def __init__(self, task_id: str = "test_task", turns: int = 3):
        self.task_id = task_id
        self.turns = turns


class TestSpillIfTruncated:
    """Unit tests for _spill_if_truncated."""

    def test_short_text_returned_unchanged(self):
        set_features(Features(output_spill_to_file=True))
        state = _FakeState()
        text = "Hello world"
        result = _spill_if_truncated(text, 1500, "output", state)
        assert result == text

    def test_exact_limit_returned_unchanged(self):
        set_features(Features(output_spill_to_file=True))
        state = _FakeState()
        text = "x" * 1500
        result = _spill_if_truncated(text, 1500, "output", state)
        assert result == text

    def test_feature_flag_off_returns_unchanged(self):
        set_features(Features(output_spill_to_file=False))
        state = _FakeState()
        text = "x" * 5000
        result = _spill_if_truncated(text, 1500, "output", state)
        assert result == text
        assert "peek" not in result

    def test_long_text_spills_to_file(self, tmp_path):
        set_features(Features(output_spill_to_file=True))
        state = _FakeState(task_id="spill_test", turns=2)
        text = "A" * 5000

        with patch("src.graph.helpers.os.makedirs"):
            # Patch open to write to tmp_path instead
            spill_path = str(tmp_path / "spill_test_output_t2.txt")
            result = _spill_if_truncated(text, 1500, "output", state)

        assert "peek(99999" in result
        assert "output" in result
        assert "chars truncated" in result
        # Result should be within limit (1500 chars + pointer)
        assert len(result) <= 1500 + 50  # generous margin for pointer

    def test_spill_file_contains_full_content(self):
        set_features(Features(output_spill_to_file=True))
        state = _FakeState(task_id="content_check", turns=1)
        text = "B" * 3000

        result = _spill_if_truncated(text, 500, "error", state)

        # Verify the spill file was created with full content
        spill_path = "/mnt/raid0/llm/tmp/content_check_error_t1.txt"
        try:
            assert os.path.exists(spill_path)
            with open(spill_path) as f:
                content = f.read()
            assert content == text
            assert len(content) == 3000
        finally:
            if os.path.exists(spill_path):
                os.remove(spill_path)

    def test_pointer_includes_file_path(self):
        set_features(Features(output_spill_to_file=True))
        state = _FakeState(task_id="ptr_test", turns=5)
        text = "C" * 2000

        result = _spill_if_truncated(text, 500, "output", state)

        assert "ptr_test_output_t5.txt" in result
        # Clean up
        spill_path = "/mnt/raid0/llm/tmp/ptr_test_output_t5.txt"
        if os.path.exists(spill_path):
            os.remove(spill_path)

    def test_truncated_preview_fits_within_limit(self):
        set_features(Features(output_spill_to_file=True))
        state = _FakeState(task_id="fit_test", turns=1)
        text = "D" * 10000

        result = _spill_if_truncated(text, 1500, "output", state)

        # The result (preview + pointer) should be under the original limit
        # so the builder's own truncation doesn't clip the pointer
        assert len(result) <= 1500
        assert "peek" in result

        # Clean up
        spill_path = "/mnt/raid0/llm/tmp/fit_test_output_t1.txt"
        if os.path.exists(spill_path):
            os.remove(spill_path)

    def test_error_label_in_spill_path(self):
        set_features(Features(output_spill_to_file=True))
        state = _FakeState(task_id="label_test", turns=7)
        text = "E" * 2000

        result = _spill_if_truncated(text, 500, "error", state)

        assert "label_test_error_t7.txt" in result
        # Clean up
        spill_path = "/mnt/raid0/llm/tmp/label_test_error_t7.txt"
        if os.path.exists(spill_path):
            os.remove(spill_path)

    def test_sanitizes_task_id(self):
        set_features(Features(output_spill_to_file=True))
        state = _FakeState(task_id="task/with:bad<chars>", turns=1)
        text = "F" * 2000

        result = _spill_if_truncated(text, 500, "output", state)

        # Should not contain unsafe characters in the path
        assert "task_with_bad_chars_" in result
        # Clean up
        import glob
        for f in glob.glob("/mnt/raid0/llm/tmp/task_with_bad_chars*_output_t1.txt"):
            os.remove(f)
