"""Private agent-readable Hermes projection; never a measurement source of record.

The stdlib reader emits the session/messages input envelope accepted by the
pinned Letta Hermes adapter. It does not adopt the Node-backed normalizer,
mutate/ingest a database, truncate tool results, or infer execution outcomes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

SESSION_FIELDS = ("id", "model", "cwd", "started_at", "ended_at", "title")
MESSAGE_FIELDS = (
    "id",
    "session_id",
    "role",
    "content",
    "timestamp",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "reasoning_content",
    "reasoning",
    "active",
)


def read_session(db_path: str | Path, session_id: str) -> dict[str, Any]:
    """Read one consistent, local-private projection using a read-only transaction.

    Only allowlisted conversational fields leave this function. Billing, user
    identity, system prompts and measurement fields remain in the original DB.
    Content itself is private and callers must not put the result in Git/logs.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("a nonempty session ID is required")
    path = Path(db_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("expected a SQLite database file")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        session_columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
        message_columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
        if (
            "id" not in session_columns
            or not {"id", "session_id", "role", "content"} <= message_columns
        ):
            raise ValueError("incompatible Hermes schema")
        session_fields = [name for name in SESSION_FIELDS if name in session_columns]
        session = connection.execute(
            "SELECT " + ",".join(session_fields) + " FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if session is None:
            raise ValueError("session not found")
        message_fields = [name for name in MESSAGE_FIELDS if name in message_columns]
        rows = connection.execute(
            "SELECT " + ",".join(message_fields) + " FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        )
        messages = []
        for native in rows:
            row = dict(native)
            active = row.get("active")
            # Match upstream's strict numeric/boolean zero check, not string truthiness.
            if active is False or (type(active) in (int, float) and active == 0):
                continue
            if row["role"] not in ("user", "assistant", "tool"):
                # Keep insertion-order barriers for upstream call/result linking,
                # without exporting injected/system content the adapter ignores.
                row = {
                    name: row[name]
                    for name in ("id", "session_id", "role", "timestamp", "active")
                    if name in row
                }
            messages.append(row)
        return {
            "session": dict(session),
            "messages": messages,
            "fidelity": {
                "tier": "agent_readable_projection",
                "privacy_class": "local_private",
                "measurement_grade": False,
                "omitted": ["tokens", "latency", "cost", "outcome", "phase", "executor_build"],
            },
        }
    finally:
        connection.close()
