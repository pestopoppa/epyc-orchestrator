"""Safety regressions for the tools loaded from the YAML registry."""

import tracemalloc

from orchestration.tools import system


def test_read_file_oversize_branch_reads_only_the_configured_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "ALLOWED_PATHS", [str(tmp_path)])
    large_file = tmp_path / "sparse-large-file"
    with large_file.open("wb") as stream:
        stream.write(b"prefix")
        stream.truncate(128 * 1024 * 1024)

    # Import-time/setup allocations should not count against the bounded-read
    # assertion below.
    assert system._is_path_allowed(str(large_file))
    tracemalloc.start()
    try:
        result = system.read_file(str(large_file), max_bytes=1024)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result["content"].startswith("prefix")
    assert "truncated at 1024 bytes" in result["content"]
    assert peak_bytes < 16 * 1024 * 1024


def test_read_file_rejects_symlink_escape_from_allowed_root(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_file = outside_root / "secret.txt"
    outside_file.write_text("secret")
    escaped_link = allowed_root / "link.txt"
    escaped_link.symlink_to(outside_file)
    monkeypatch.setattr(system, "ALLOWED_PATHS", [str(allowed_root)])

    result = system.read_file(str(escaped_link))

    assert "error" in result
    assert "Path not allowed" in result["error"]


def test_read_file_rejects_negative_byte_limit(tmp_path):
    result = system.read_file(str(tmp_path / "missing"), max_bytes=-1)
    assert result == {"error": "max_bytes must be non-negative"}
