from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.corpus import build_static_ngram_cache as builder


def _fake_tool_dir(tmp_path: Path) -> Path:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    for name in ("llama-lookup-create", "llama-lookup-merge"):
        path = tool_dir / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return tool_dir


def _fake_runner(commands: list[list[str]]):
    def run(command):
        command = list(command)
        commands.append(command)
        if command[0].endswith("llama-lookup-create"):
            part = Path(command[command.index("-lcs") + 1])
            part.write_bytes(b"part")
        elif command[0].endswith("llama-lookup-merge"):
            Path(command[-1]).write_bytes(b"merged")
        else:  # pragma: no cover - defensive guard
            raise AssertionError(command)

    return run


def test_build_static_ngram_cache_chunks_input_file_and_merges(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    source = tmp_path / "source.txt"
    source.write_text("alpha beta gamma\n" * 4, encoding="utf-8")
    output = tmp_path / "cache.bin"
    commands: list[list[str]] = []

    result = builder.build_static_ngram_cache(
        builder.BuildConfig(
            model=model,
            output=output,
            input_file=source,
            chunk_bytes=20,
            max_bytes=10_000,
            llama_bin_dir=_fake_tool_dir(tmp_path),
            threads=3,
        ),
        runner=_fake_runner(commands),
    )

    assert output.read_bytes() == b"merged"
    assert result.source == {"kind": "input_file", "path": str(source)}
    assert result.bytes_written == source.stat().st_size
    assert len(result.chunks) > 1
    assert len(commands) == len(result.chunks) + 1
    first_create = commands[0]
    assert first_create[:2] == [str(tmp_path / "bin" / "llama-lookup-create"), "-m"]
    assert first_create[first_create.index("-f") + 1].endswith(".txt")
    assert first_create[first_create.index("-lcs") + 1].endswith(".bin")
    assert first_create[-2:] == ["-t", "3"]
    merge = commands[-1]
    assert merge[0].endswith("llama-lookup-merge")
    assert merge[-1] == str(output)
    manifest = json.loads(Path(result.manifest).read_text(encoding="utf-8"))
    assert manifest["model"] == str(model)
    assert manifest["output"] == str(output)
    assert "vocab-locked" in manifest["notes"][0]


def test_dry_run_can_export_filtered_snippets_without_model_or_tools(tmp_path: Path) -> None:
    db = tmp_path / "snippets.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE snippets (id INTEGER PRIMARY KEY, code TEXT NOT NULL, source TEXT DEFAULT '', hash TEXT NOT NULL, language TEXT DEFAULT '')"
    )
    conn.executemany(
        "INSERT INTO snippets (id, code, source, hash, language) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "def alpha():\n    return 1", "a.py", "h1", "python"),
            (2, "function beta() { return 2; }", "b.js", "h2", "javascript"),
            (3, "def gamma():\n    return 3", "c.py", "h3", "python"),
        ],
    )
    conn.commit()
    conn.close()

    result = builder.build_static_ngram_cache(
        builder.BuildConfig(
            model=tmp_path / "missing-model.gguf",
            output=tmp_path / "cache.bin",
            snippets_db=db,
            language=("python",),
            limit_snippets=1,
            chunk_bytes=1024,
            max_bytes=10_000,
            dry_run=True,
            llama_bin_dir=tmp_path / "missing-bin",
        )
    )

    assert result.dry_run is True
    assert result.source["kind"] == "snippets_db"
    assert result.source["language"] == ["python"]
    assert result.source["limit_snippets"] == 1
    assert len(result.chunks) == 1
    assert len(result.commands) == 2
    assert result.commands[0][0].endswith("llama-lookup-create")
    assert result.commands[-1][0].endswith("llama-lookup-merge")
    assert not (tmp_path / "cache.bin").exists()
    assert Path(result.manifest).exists()


def test_large_scan_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow-large-scan"):
        builder.build_static_ngram_cache(
            builder.BuildConfig(
                model=tmp_path / "missing-model.gguf",
                output=tmp_path / "cache.bin",
                input_file=tmp_path / "source.txt",
                max_bytes=builder.LARGE_SCAN_BYTES + 1,
            )
        )
