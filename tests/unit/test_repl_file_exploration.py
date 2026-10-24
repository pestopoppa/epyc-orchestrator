#!/usr/bin/env python3
"""Unit tests for REPL file exploration tools (_FileExplorationMixin)."""

import json

from src.repl_environment import REPLConfig, REPLEnvironment


class TestPeekTool:
    """Test the peek() tool (_peek)."""

    def test_peek_default_from_context(self):
        """Test peek with default n from context."""
        context = "A" * 1000
        repl = REPLEnvironment(context=context)
        result = repl.execute("artifacts['result'] = peek()")

        assert result.error is None
        assert repl.artifacts["result"] == "A" * 500  # Default n=500

    def test_peek_custom_n_from_context(self):
        """Test peek with custom n from context."""
        context = "Hello World"
        repl = REPLEnvironment(context=context)
        result = repl.execute("artifacts['result'] = peek(3)")

        assert result.error is None
        assert repl.artifacts["result"] == "Hel"

    def test_peek_from_valid_file(self, tmp_path):
        """Test peek from valid file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("X" * 1000)

        repl = REPLEnvironment(context="test")
        result = repl.execute(f'artifacts["result"] = peek(100, file_path="{test_file}")')

        assert result.error is None
        assert repl.artifacts["result"] == "X" * 100

    def test_peek_from_nonexistent_file(self, tmp_path):
        """Test peek from nonexistent file returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute(
            f'artifacts["result"] = peek(file_path="{tmp_path}/nonexistent.txt")'
        )

        assert result.error is None
        assert "[ERROR: File not found:" in repl.artifacts["result"]

    def test_peek_from_blocked_path(self):
        """Test peek from blocked path (e.g. /etc/passwd) returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = peek(file_path="/etc/passwd")')

        assert result.error is None
        assert "[ERROR:" in repl.artifacts["result"]
        assert "not in allowed locations" in repl.artifacts["result"]

    def test_peek_increments_exploration_calls(self):
        """Test peek increments _exploration_calls counter."""
        repl = REPLEnvironment(context="test")
        assert repl._exploration_calls == 0

        repl.execute("peek()")
        assert repl._exploration_calls == 1

        repl.execute("peek(10)")
        assert repl._exploration_calls == 2


class TestGrepTool:
    """Test the grep() tool (_grep)."""

    def test_grep_matching_pattern_in_context(self):
        """Test grep with matching pattern in context."""
        context = "Line 1\nLine 2\nSpecial pattern\nLine 4"
        repl = REPLEnvironment(context=context)
        result = repl.execute('artifacts["result"] = grep("Special")')

        assert result.error is None
        assert isinstance(repl.artifacts["result"], list)
        assert len(repl.artifacts["result"]) == 1
        assert "Special pattern" in repl.artifacts["result"][0]

    def test_grep_case_insensitive_matching(self):
        """Test grep is case-insensitive."""
        context = "Hello WORLD\nhello world\nHeLLo WoRLd"
        repl = REPLEnvironment(context=context)
        result = repl.execute('artifacts["result"] = grep("hello")')

        assert result.error is None
        assert len(repl.artifacts["result"]) == 3

    def test_grep_in_file_with_matches(self, tmp_path):
        """Test grep in file with matches."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nMatch here\nLine 3\nAnother match\n")

        repl = REPLEnvironment(context="test")
        result = repl.execute(f'artifacts["result"] = grep("match", file_path="{test_file}")')

        assert result.error is None
        assert len(repl.artifacts["result"]) == 2

    def test_grep_in_file_no_matches(self, tmp_path):
        """Test grep in file with no matches returns empty list."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n")

        repl = REPLEnvironment(context="test")
        result = repl.execute(f'artifacts["result"] = grep("NOMATCH", file_path="{test_file}")')

        assert result.error is None
        assert repl.artifacts["result"] == []

    def test_grep_invalid_regex(self):
        """Test grep with invalid regex returns error message."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = grep("[invalid")')

        assert result.error is None
        assert "[REGEX ERROR:" in repl.artifacts["result"][0]

    def test_grep_nonexistent_file(self, tmp_path):
        """Test grep on nonexistent file returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute(
            f'artifacts["result"] = grep("pattern", file_path="{tmp_path}/nonexistent.txt")'
        )

        assert result.error is None
        assert "[ERROR: File not found:" in repl.artifacts["result"][0]

    def test_grep_blocked_path(self):
        """Test grep on blocked path returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = grep("pattern", file_path="/etc/passwd")')

        assert result.error is None
        assert "[ERROR:" in repl.artifacts["result"][0]

    def test_grep_populates_hits_buffer(self):
        """Test grep populates _grep_hits_buffer."""
        context = "Match line 1\nNo result here\nMatch line 2"
        repl = REPLEnvironment(context=context)
        assert len(repl._grep_hits_buffer) == 0

        repl.execute('artifacts["result"] = grep("Match")')

        assert len(repl._grep_hits_buffer) == 1
        assert repl._grep_hits_buffer[0]["pattern"] == "Match"
        assert repl._grep_hits_buffer[0]["match_count"] == 2

    def test_grep_truncates_at_max_results(self):
        """Test grep truncates at max_grep_results."""
        # Create context with many matches
        lines = [f"Match line {i}" for i in range(200)]
        context = "\n".join(lines)

        config = REPLConfig(max_grep_results=50)
        repl = REPLEnvironment(context=context, config=config)
        result = repl.execute('artifacts["result"] = grep("Match")')

        assert result.error is None
        # Should be 50 matches + 1 truncation message
        assert len(repl.artifacts["result"]) == 51
        assert "truncated at 50 results" in repl.artifacts["result"][-1]


class TestListDirTool:
    """Test the LIST_DIR() tool (_list_dir)."""

    def test_list_dir_valid_directory(self, tmp_path):
        """Test list_dir on valid directory."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "subdir").mkdir()

        repl = REPLEnvironment(context="test")
        result = repl.execute(f'artifacts["result"] = list_dir("{tmp_path}")')

        assert result.error is None
        # Result is wrapped in <<<TOOL_OUTPUT>>>...<<<END_TOOL_OUTPUT>>>
        assert "<<<TOOL_OUTPUT>>>" in repl.artifacts["result"]

    def test_list_dir_sorts_dirs_before_files(self, tmp_path):
        """Test list_dir sorts directories before files."""
        (tmp_path / "afile.txt").touch()
        (tmp_path / "zdir").mkdir()

        repl = REPLEnvironment(context="test")
        repl.execute(f'list_dir("{tmp_path}")')

        # Check that directory comes first in the output
        output_list = repl.artifacts.get("_tool_outputs", [])
        assert len(output_list) > 0
        output_data = json.loads(output_list[-1])
        files = output_data["files"]

        # First entry should be directory
        assert files[0]["type"] == "dir"
        assert files[0]["name"] == "zdir"
        # Second entry should be file
        assert files[1]["type"] == "file"
        assert files[1]["name"] == "afile.txt"

    def test_list_dir_nonexistent_path(self, tmp_path):
        """Test list_dir on nonexistent path returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute(f'artifacts["result"] = list_dir("{tmp_path}/nonexistent")')

        assert result.error is None
        assert "[ERROR: Directory not found:" in repl.artifacts["result"]

    def test_list_dir_blocked_path(self):
        """Test list_dir on blocked path returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = list_dir("/etc")')

        assert result.error is None
        assert "[ERROR:" in repl.artifacts["result"]

    def test_list_dir_caps_at_100_entries(self, tmp_path):
        """Test list_dir caps at 100 entries."""
        for i in range(150):
            (tmp_path / f"file{i:03d}.txt").touch()

        repl = REPLEnvironment(context="test")
        repl.execute(f'list_dir("{tmp_path}")')

        output_list = repl.artifacts.get("_tool_outputs", [])
        raw = output_list[-1]

        # Output may be TOON (tabular) or JSON depending on toon_format availability
        try:
            output_data = json.loads(raw)
        except json.JSONDecodeError:
            from toon_format import decode
            output_data = decode(raw)

        assert len(output_data["files"]) == 100
        assert output_data["total"] == 150


class TestFileInfoTool:
    """Test the FILE_INFO() tool (_file_info)."""

    def test_file_info_existing_file(self, tmp_path):
        """Test file_info on existing file returns JSON with metadata."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        repl = REPLEnvironment(context="test")
        result = repl.execute(f'artifacts["result"] = file_info("{test_file}")')

        assert result.error is None
        info = json.loads(repl.artifacts["result"])

        assert info["path"] == str(test_file)
        assert info["exists"] is True
        assert info["type"] == "file"
        assert info["size"] == 11  # len("Hello World")
        assert "modified" in info
        assert info["extension"] == ".txt"

    def test_file_info_directory(self, tmp_path):
        """Test file_info on directory returns type 'dir'."""
        repl = REPLEnvironment(context="test")
        result = repl.execute(f'artifacts["result"] = file_info("{tmp_path}")')

        assert result.error is None
        info = json.loads(repl.artifacts["result"])

        assert info["exists"] is True
        assert info["type"] == "dir"
        assert info["extension"] is None

    def test_file_info_nonexistent_file(self, tmp_path):
        """Test file_info on nonexistent file returns exists: false."""
        repl = REPLEnvironment(context="test")
        result = repl.execute(
            f'artifacts["result"] = file_info("{tmp_path}/nonexistent.txt")'
        )

        assert result.error is None
        info = json.loads(repl.artifacts["result"])

        assert info["exists"] is False

    def test_file_info_blocked_path(self):
        """Test file_info on blocked path returns error."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["result"] = file_info("/etc/passwd")')

        assert result.error is None
        assert "[ERROR:" in repl.artifacts["result"]
