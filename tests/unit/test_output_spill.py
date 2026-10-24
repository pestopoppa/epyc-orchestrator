"""Tests for CMV-style output spill to file (Action 11) and exact paged recall (TOC-SP-2)."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features import Features, set_features, reset_features
from src.graph import file_artifacts
from src.graph.helpers import _spill_if_truncated

_POINTER_RE = re.compile(r'peek\((\d+), file_path="([^"]+)", offset=(\d+)\)')


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_features()


@pytest.fixture()
def spill_dir(tmp_path, monkeypatch):
    """A private spill dir; /tmp/ is always a REPL-allowed prefix."""
    path = tmp_path / "spill"
    path.mkdir()
    monkeypatch.setattr(file_artifacts, "SPILL_DIR", str(path))
    return str(path)


class _FakeState:
    """Minimal TaskState stand-in for spill tests."""

    def __init__(self, task_id: str = "test_task", turns: int = 3):
        self.task_id = task_id
        self.turns = turns


def _pointer(result: str) -> tuple[int, str, int]:
    match = _POINTER_RE.search(result)
    assert match, result
    return int(match.group(1)), match.group(2), int(match.group(3))


def _split(result: str) -> tuple[str, str]:
    head, _, rest = result.partition("\n[... ")
    return head, rest.split("]\n", 1)[1]


def _varied(n: int) -> str:
    """Non-uniform text so head/tail/offset mistakes are visible."""
    return "".join(f"line {i:05d} value={i * 7 % 101}\n" for i in range(n))


class TestSpillIfTruncated:
    """Unit tests for _spill_if_truncated."""

    def test_short_text_returned_unchanged(self, spill_dir):
        set_features(Features(output_spill_to_file=True))
        text = "Hello world"
        assert _spill_if_truncated(text, 1500, "output", _FakeState()) == text

    def test_exact_limit_returned_unchanged(self, spill_dir):
        set_features(Features(output_spill_to_file=True))
        text = "x" * 1500
        assert _spill_if_truncated(text, 1500, "output", _FakeState()) == text

    def test_feature_flag_off_returns_unchanged(self, spill_dir):
        set_features(Features(output_spill_to_file=False))
        text = "x" * 5000
        result = _spill_if_truncated(text, 1500, "output", _FakeState())
        assert result == text
        assert "peek" not in result
        assert os.listdir(spill_dir) == []

    def test_long_text_spills_with_head_tail_and_pointer(self, spill_dir):
        set_features(Features(output_spill_to_file=True))
        text = _varied(400)

        result = _spill_if_truncated(text, 1500, "output", _FakeState("spill_test", 2))

        assert "chars truncated" in result
        assert f"of {len(text)} chars" in result
        n, _path, offset = _pointer(result)
        head, tail = _split(result)
        assert text.startswith(head)
        assert tail and text.endswith(tail)
        assert offset == len(head)
        assert len(head) + n + len(tail) == len(text)
        assert len(result) <= 1500

    def test_spill_file_contains_full_content(self, spill_dir):
        set_features(Features(output_spill_to_file=True))
        text = "B" * 3000

        result = _spill_if_truncated(text, 500, "error", _FakeState("content_check", 1))

        _, path, _ = _pointer(result)
        with open(path, encoding="utf-8", newline="") as f:
            assert f.read() == text

    def test_spill_name_is_content_hash(self, spill_dir):
        set_features(Features(output_spill_to_file=True))
        text = "C" * 2000
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        result = _spill_if_truncated(text, 500, "output", _FakeState("ptr_test", 5))

        assert f"{spill_dir}/ptr_test_output_{digest}.txt" in result
        assert "_t5" not in result

    def test_same_content_reuses_file_and_new_content_never_overwrites(self, spill_dir):
        set_features(Features(output_spill_to_file=True))
        first = "D" * 2000
        second = "E" * 2000

        p1 = _pointer(_spill_if_truncated(first, 500, "output", _FakeState("dedup", 1)))[1]
        p1_again = _pointer(_spill_if_truncated(first, 500, "output", _FakeState("dedup", 9)))[1]
        p2 = _pointer(_spill_if_truncated(second, 500, "output", _FakeState("dedup", 1)))[1]

        assert p1 == p1_again
        assert p2 != p1
        with open(p1, encoding="utf-8") as f:
            assert f.read() == first  # the earlier pointer still yields its own bytes
        assert sorted(os.listdir(spill_dir)) == sorted(
            [os.path.basename(p1), os.path.basename(p2)]
        )

    def test_truncated_preview_fits_within_limit(self, spill_dir):
        set_features(Features(output_spill_to_file=True))
        text = "D" * 10000

        result = _spill_if_truncated(text, 1500, "output", _FakeState("fit_test", 1))

        # preview + pointer stays under the limit so a downstream cut cannot clip the pointer
        assert len(result) <= 1500
        assert "peek" in result

    def test_error_label_in_spill_path(self, spill_dir):
        set_features(Features(output_spill_to_file=True))
        result = _spill_if_truncated("E" * 2000, 500, "error", _FakeState("label_test", 7))
        assert "/label_test_error_" in result

    def test_sanitizes_task_id(self, spill_dir):
        set_features(Features(output_spill_to_file=True))
        result = _spill_if_truncated(
            "F" * 2000, 500, "output", _FakeState("task/with:bad<chars>", 1)
        )
        _, path, _ = _pointer(result)
        assert os.path.basename(path).startswith("task_with_bad_chars__output_")
        assert os.path.dirname(path) == spill_dir


class TestExactRecallThroughRepl:
    """The emitted pointer, executed verbatim in the REPL, returns exactly the omitted span."""

    def test_emitted_pointer_recalls_omitted_span_exactly(self, spill_dir):
        from src.repl_environment import REPLEnvironment

        set_features(Features(output_spill_to_file=True))
        text = _varied(200) + "Traceback: boom\r\nwindows line\rcr only\n" + _varied(300)
        result = _spill_if_truncated(text, 1200, "output", _FakeState("recall", 1))
        call = _POINTER_RE.search(result).group(0)
        n, _, offset = _pointer(result)

        repl = REPLEnvironment(context="ctx")
        out = repl.execute(f"artifacts['span'] = {call}")

        assert out.error is None
        assert repl.artifacts["span"] == text[offset:offset + n]
        head, tail = _split(result)
        assert head + repl.artifacts["span"] + tail == text

    def test_legacy_pointer_format_still_resolves(self, spill_dir):
        """Pointers already in transcripts: `peek(99999, file_path="..._t{turn}.txt")`."""
        from src.repl_environment import REPLEnvironment

        legacy = os.path.join(spill_dir, "old_task_output_t3.txt")
        content = "legacy spilled output\n" * 50
        with open(legacy, "w", encoding="utf-8") as f:
            f.write(content)
        legacy_pointer = (
            f"\n[... 900 chars truncated; "
            f'full output: peek(99999, file_path="{legacy}")]'
        )
        call = re.search(r'peek\(99999, file_path="[^"]+"\)', legacy_pointer).group(0)

        # a new spill for the same task/label/turn must not disturb the legacy file
        set_features(Features(output_spill_to_file=True))
        _spill_if_truncated("Z" * 3000, 500, "output", _FakeState("old_task", 3))

        repl = REPLEnvironment(context="ctx")
        out = repl.execute(f"artifacts['full'] = {call}")
        assert out.error is None
        assert repl.artifacts["full"] == content
        # the TOC-SP-1 footer form keeps working too
        repl.execute(f'artifacts["head"] = peek(2000, file_path="{legacy}")')
        assert repl.artifacts["head"] == content[:2000]

    def test_peek_offset_paging_and_negative_offset(self, spill_dir):
        from src.repl_environment import REPLEnvironment

        path = os.path.join(spill_dir, "paged.txt")
        content = "".join(chr(0x41 + i % 26) for i in range(3000)) + "é✓end"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        repl = REPLEnvironment(context="0123456789")
        pages = []
        for off in range(0, len(content), 1000):
            repl.execute(f'artifacts["p"] = peek(1000, file_path="{path}", offset={off})')
            pages.append(repl.artifacts["p"])
        assert "".join(pages) == content

        repl.execute(f'artifacts["t"] = peek(5, file_path="{path}", offset=-5)')
        assert repl.artifacts["t"] == "é✓end"
        repl.execute(f'artifacts["past"] = peek(10, file_path="{path}", offset=999999)')
        assert repl.artifacts["past"] == ""

        repl.execute('artifacts["c"] = peek(3, offset=4)')
        assert repl.artifacts["c"] == "456"
        repl.execute('artifacts["ct"] = peek(3, offset=-3)')
        assert repl.artifacts["ct"] == "789"
        repl.execute("artifacts['d'] = peek(3)")
        assert repl.artifacts["d"] == "012"


class TestSpillErrorPreserve:
    """TOC-SP-8: error-bearing output is never worker-summarised (flag default off)."""

    @staticmethod
    def _repl(tmp_path, worker):
        from src.repl_environment import REPLConfig, REPLEnvironment

        cfg = REPLConfig(output_cap=100, spill_dir=str(tmp_path))
        return REPLEnvironment(context="t", config=cfg, llm_primitives=worker)

    @staticmethod
    def _worker():
        from unittest.mock import MagicMock

        w = MagicMock()
        w.llm_call.return_value = "WORKER SUMMARY"
        return w

    ERR = "ok line\n" * 100 + "[STDERR]\nTraceback (most recent call last):\nboom\n"
    OK = "ok line\n" * 100

    def test_flag_on_error_keeps_head_tail_and_peek_marker_no_worker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_REPL_SPILL_ERROR_PRESERVE", "1")
        w = self._worker()
        out = self._repl(tmp_path, w)._spill_output(self.ERR)
        w.llm_call.assert_not_called()
        assert "WORKER SUMMARY" not in out and "boom" in out
        assert 'peek(2000, file_path="' in out
        assert "chars omitted" in out
        assert len(out.split("\n...[")[0].split("\n", 1)[1]) <= 300

    def test_flag_on_success_still_summarised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_REPL_SPILL_ERROR_PRESERVE", "1")
        w = self._worker()
        out = self._repl(tmp_path, w)._spill_output(self.OK)
        w.llm_call.assert_called_once()
        assert "WORKER SUMMARY" in out

    def test_structured_error_argument_counts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_REPL_SPILL_ERROR_PRESERVE", "1")
        w = self._worker()
        out = self._repl(tmp_path, w)._spill_output(self.OK, error="ValueError: x")
        w.llm_call.assert_not_called()
        assert "chars omitted" in out

    def test_flag_off_is_byte_identical_for_errors(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_REPL_SPILL_ERROR_PRESERVE", raising=False)
        w = self._worker()
        r = self._repl(tmp_path, w)
        a = r._spill_output(self.ERR)
        w.llm_call.assert_called_once()
        assert "WORKER SUMMARY" in a
        b = r._spill_output(self.ERR, error="e")  # flag off: error kwarg is inert
        assert a.split("\n")[1:-1] == b.split("\n")[1:-1] == ["WORKER SUMMARY"]

    def test_exception_text_over_cap_routed_through_spill_flag_on(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_REPL_SPILL_ERROR_PRESERVE", "1")
        w = self._worker()
        r = self._repl(tmp_path, w)
        res = r.execute("raise ValueError('x' * 500)")
        w.llm_call.assert_not_called()
        assert res.error and "chars omitted" in res.error and "peek(2000" in res.error

    def test_exception_text_unchanged_flag_off_and_under_cap(self, tmp_path, monkeypatch):
        w = self._worker()
        r = self._repl(tmp_path, w)
        monkeypatch.delenv("ORCHESTRATOR_REPL_SPILL_ERROR_PRESERVE", raising=False)
        res = r.execute("raise ValueError('x' * 500)")
        assert res.error == "ValueError: " + "x" * 500
        monkeypatch.setenv("ORCHESTRATOR_REPL_SPILL_ERROR_PRESERVE", "1")
        res = r.execute("raise ValueError('short')")
        assert res.error == "ValueError: short"
