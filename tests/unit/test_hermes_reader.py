"""Synthetic/public fixtures only: no private session text belongs in this suite."""

import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import stat

import pytest

from src.trace.hermes_reader import read_session

CLI = Path(__file__).resolve().parents[2] / "scripts/trace/export_hermes.py"
spec = importlib.util.spec_from_file_location("export_hermes", CLI)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def make_db(path, envelope):
    session = envelope["session"]
    messages = envelope["messages"]
    fields = sorted({key for row in messages for key in row})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE sessions (" + ",".join('"' + k + '"' for k in session) + ")"
        )
        connection.execute(
            "INSERT INTO sessions VALUES (" + ",".join("?" for _ in session) + ")",
            list(session.values()),
        )
        connection.execute(
            "CREATE TABLE messages (" + ",".join('"' + k + '"' for k in fields) + ")"
        )
        for row in messages:
            values = [
                json.dumps(row[k]) if isinstance(row.get(k), (dict, list)) else row.get(k)
                for k in fields
            ]
            connection.execute(
                "INSERT INTO messages VALUES (" + ",".join("?" for _ in fields) + ")", values
            )


@pytest.mark.parametrize("name", ["tool-calls", "cleanup"])
def test_pinned_public_upstream_rows_recovered_without_node(tmp_path, name):
    source = (
        Path(__file__).resolve().parents[1] / "fixtures/hermes_trajectory" / name / "input.json"
    )
    envelope = json.loads(source.read_text())
    if isinstance(envelope, list):
        envelope = {"session": {"id": envelope[0]["session_id"]}, "messages": envelope}
    db = tmp_path / "state.db"
    make_db(db, envelope)
    output = read_session(db, envelope["session"]["id"])
    expected = [
        row
        for row in envelope["messages"]
        if row.get("active") not in (False, 0) and row["role"] in ("user", "assistant", "tool")
    ]
    expected.sort(key=lambda row: row["id"])
    conversational = [
        row for row in output["messages"] if row["role"] in ("user", "assistant", "tool")
    ]
    assert [row["id"] for row in conversational] == [row["id"] for row in expected]
    for actual, original in zip(conversational, expected):
        for field in ("content", "reasoning", "reasoning_content", "tool_call_id", "tool_calls"):
            value = actual.get(field)
            if isinstance(original.get(field), (list, dict)):
                value = json.loads(value)
            assert value == original.get(field)
    assert output["fidelity"]["measurement_grade"] is False


def test_optional_columns_clock_regression_and_sql_parameters(tmp_path):
    db = tmp_path / "odd?#name.db"
    envelope = {
        "session": {"id": "session' OR 1=1 --", "model": "fixture"},
        "messages": [
            {
                "id": 2,
                "session_id": "session' OR 1=1 --",
                "role": "assistant",
                "content": "second",
                "timestamp": 1,
            },
            {
                "id": 1,
                "session_id": "session' OR 1=1 --",
                "role": "user",
                "content": "first",
                "timestamp": 3,
            },
        ],
    }
    make_db(db, envelope)
    before = db.read_bytes()
    output = read_session(db, envelope["session"]["id"])
    assert [row["content"] for row in output["messages"]] == ["first", "second"]
    assert "tool_calls" not in output["messages"][0]
    assert db.read_bytes() == before
    with pytest.raises(ValueError, match="not found"):
        read_session(db, "different' OR 1=1 --")


def test_non_conversational_and_measurement_fields_never_project(tmp_path):
    db = tmp_path / "state.db"
    envelope = {
        "session": {
            "id": "s",
            "user_id": "private-user",
            "system_prompt": "private-system",
            "actual_cost_usd": 1,
        },
        "messages": [
            {
                "id": 1,
                "session_id": "s",
                "role": "assistant",
                "content": "private-sentinel",
                "token_count": 42,
                "finish_reason": "stop",
                "active": 1,
            },
            {
                "id": 2,
                "session_id": "s",
                "role": "system",
                "content": "not-agent-view",
                "active": 1,
            },
            {"id": 3, "session_id": "s", "role": "user", "content": "rewound", "active": 0},
        ],
    }
    make_db(db, envelope)
    output = read_session(db, "s")
    assert len(output["messages"]) == 2
    assert output["messages"][1]["role"] == "system"
    assert "content" not in output["messages"][1]
    assert set(output["session"]) == {"id"}
    assert not {"token_count", "finish_reason"} & set(output["messages"][0])
    assert "outcome" not in output["messages"][0]


def test_missing_db_schema_and_empty_session_refuse(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_session(tmp_path / "absent.db", "s")
    assert not (tmp_path / "absent.db").exists()
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    with pytest.raises(ValueError, match="incompatible"):
        read_session(db, "s")
    with pytest.raises(ValueError, match="nonempty"):
        read_session(db, "")


def test_private_output_fresh0600_and_stdout_counts_only(tmp_path, capsys):
    db = tmp_path / "state.db"
    make_db(
        db,
        {
            "session": {"id": "s"},
            "messages": [
                {"id": 1, "session_id": "s", "role": "user", "content": "private-sentinel"}
            ],
        },
    )
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    output = private / "projection.json"
    assert cli.main([str(db), "s", "--out", str(output)]) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert "private-sentinel" not in capsys.readouterr().out
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli.write_private_projection({}, output)
    assert output.read_bytes() == original


def test_public_git_directory_and_parent_symlink_refuse(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(private, target_is_directory=True)
    with pytest.raises(PermissionError, match="real unambiguous"):
        cli.write_private_projection({}, link / "export.json")
    repo = tmp_path / "repo"
    repo.mkdir(mode=0o700)
    (repo / ".git").mkdir()
    with pytest.raises(PermissionError, match="Git tree"):
        cli.write_private_projection({}, repo / "export.json")
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="0700"):
        cli.write_private_projection({}, public / "export.json")


def test_output_symlink_never_touches_target(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    target = tmp_path / "unrelated"
    target.write_text("original")
    output = private / "projection.json"
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        cli.write_private_projection({"messages": ["private-sentinel"]}, output)
    assert target.read_text() == "original"


def test_directory_swap_refuses_before_writing_private_text(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    held = tmp_path / "held"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir(mode=0o700)
    real_open = os.open

    def swap(name, flags, *args, **kwargs):
        if name == "projection.json" and kwargs.get("dir_fd") is not None:
            private.rename(held)
            private.symlink_to(unrelated, target_is_directory=True)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(cli.os, "open", swap)
    with pytest.raises(PermissionError, match="namespace changed"):
        cli.write_private_projection(
            {"messages": ["private-sentinel"]}, private / "projection.json"
        )
    assert list(unrelated.iterdir()) == []
    assert (held / "projection.json").read_bytes() == b""


def test_long_multibyte_tool_payload_and_reasoning_are_untruncated(tmp_path):
    text = "synthetic-αβγ\n" * 5000
    envelope = {
        "session": {"id": "s"},
        "messages": [
            {
                "id": 1,
                "session_id": "s",
                "role": "assistant",
                "content": None,
                "reasoning_content": text,
                "tool_calls": [{"id": "c", "function": {"name": "fixture", "arguments": "{}"}}],
            },
            {"id": 2, "session_id": "s", "role": "tool", "content": text, "tool_call_id": "c"},
        ],
    }
    db = tmp_path / "state.db"
    make_db(db, envelope)
    rows = read_session(db, "s")["messages"]
    assert rows[0]["reasoning_content"] == text
    assert rows[1]["content"] == text


def test_strict_upstream_active_types_preserve_string_zero(tmp_path):
    db = tmp_path / "state.db"
    make_db(
        db,
        {
            "session": {"id": "s"},
            "messages": [
                {"id": 1, "session_id": "s", "role": "user", "content": "kept", "active": "0"},
                {
                    "id": 2,
                    "session_id": "s",
                    "role": "assistant",
                    "content": "rewound",
                    "active": 0,
                },
            ],
        },
    )
    assert [row["id"] for row in read_session(db, "s")["messages"]] == [1]


def test_write_permission_failure_emits_no_success_and_retains_partial(
    tmp_path, monkeypatch, capsys
):
    db = tmp_path / "state.db"
    make_db(
        db,
        {
            "session": {"id": "s"},
            "messages": [
                {"id": 1, "session_id": "s", "role": "user", "content": "private-sentinel"}
            ],
        },
    )
    private = tmp_path / "private"
    private.mkdir(mode=0o700)

    def refuse(*_):
        raise PermissionError("fixture write refusal")

    monkeypatch.setattr(cli.os, "write", refuse)
    with pytest.raises(PermissionError, match="fixture write refusal"):
        cli.main([str(db), "s", "--out", str(private / "partial.json")])
    assert capsys.readouterr().out == ""
    assert (private / "partial.json").read_bytes() == b""
    with pytest.raises(PermissionError, match="unambiguous"):
        cli.write_private_projection({}, private / ".." / "unexpected.json")
