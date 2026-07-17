"""AutoMem-style write actions for agent-facing memory projections.

This module is intentionally default-inert: it defines a safe write surface and
file-backed store, but no autopilot runtime path calls it yet. The trace store
remains read-only; these actions target an agent-facing memory ledger with
generated projection files.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

DEFAULT_MEMORY_ACTION_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/memory_actions"
)

MEMORY_ACTION_SCHEMA_VERSION = 1
VALID_ACTIONS = frozenset({"APPEND", "CREATE", "UPSERT"})
VALID_CHANNELS = frozenset({"log", "plan", "status", "inventory", "strategy"})
PROJECTION_CHANNELS = ("status", "inventory", "strategy", "plan", "log")

MemoryActionName = Literal["APPEND", "CREATE", "UPSERT"]
MemoryChannel = Literal["log", "plan", "status", "inventory", "strategy"]


class MemoryActionError(ValueError):
    """Raised when a memory action or persisted ledger row is invalid."""


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_token(value: str, *, field_name: str, allow_slash: bool = True) -> str:
    normalized = " ".join(str(value or "").strip().split())
    if not normalized:
        raise MemoryActionError(f"{field_name} must be non-empty")
    if "\x00" in normalized:
        raise MemoryActionError(f"{field_name} must not contain NUL bytes")
    if not allow_slash and any(separator in normalized for separator in ("/", "\\")):
        raise MemoryActionError(f"{field_name} must not contain path separators")
    if normalized in {".", ".."} or "/../" in f"/{normalized}/" or "\\..\\" in f"\\{normalized}\\":
        raise MemoryActionError(f"{field_name} must not contain traversal segments")
    return normalized


def _normalize_tags(tags: Iterable[str]) -> tuple[str, ...]:
    clean: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = _normalize_token(tag, field_name="tag", allow_slash=False).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        clean.append(normalized)
    return tuple(clean)


def _identity(channel: str, coordinate: str, key: str) -> str:
    digest = hashlib.sha256(f"{channel}\0{coordinate}\0{key}".encode("utf-8")).hexdigest()
    return digest[:24]


def _event_id(payload: dict[str, Any]) -> str:
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(stable).hexdigest()[:24]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True)
class MemoryAction:
    """A first-class memory write request.

    ``coordinate`` and ``key`` form the stable deduplication identity for
    CREATE/UPSERT actions. APPEND actions always add a new ledger event.
    """

    action: MemoryActionName
    channel: MemoryChannel
    coordinate: str
    key: str
    content: str
    source: str = "autopilot"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "MemoryAction":
        action = _normalize_token(self.action, field_name="action", allow_slash=False).upper()
        channel = _normalize_token(self.channel, field_name="channel", allow_slash=False).lower()
        if action not in VALID_ACTIONS:
            raise MemoryActionError(f"unknown memory action: {self.action!r}")
        if channel not in VALID_CHANNELS:
            raise MemoryActionError(f"unknown memory channel: {self.channel!r}")
        content = str(self.content or "").strip()
        if not content:
            raise MemoryActionError("content must be non-empty")
        if "\x00" in content:
            raise MemoryActionError("content must not contain NUL bytes")
        return MemoryAction(
            action=action,  # type: ignore[arg-type]
            channel=channel,  # type: ignore[arg-type]
            coordinate=_normalize_token(self.coordinate, field_name="coordinate"),
            key=_normalize_token(self.key, field_name="key"),
            content=content,
            source=_normalize_token(self.source, field_name="source"),
            tags=_normalize_tags(self.tags),
            metadata=dict(self.metadata or {}),
        )


@dataclass(frozen=True)
class MemoryActionResult:
    """Result of applying a memory action."""

    changed: bool
    status: str
    event_id: str | None
    memory_id: str
    projection_paths: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "status": self.status,
            "event_id": self.event_id,
            "memory_id": self.memory_id,
            "projection_paths": {
                channel: str(path) for channel, path in sorted(self.projection_paths.items())
            },
        }


class MemoryActionStore:
    """File-backed AutoMem action ledger plus generated memory projections."""

    def __init__(self, path: Path = DEFAULT_MEMORY_ACTION_PATH):
        self.path = Path(path)
        self.ledger_path = self.path / "memory_actions.jsonl"
        self.lock_path = self.path / ".memory_actions.lock"

    def apply(self, action: MemoryAction, *, now: datetime | None = None) -> MemoryActionResult:
        """Validate and apply a memory action.

        The operation is serialized with a file lock. CREATE/UPSERT actions are
        deduplicated by ``channel`` + ``coordinate`` + ``key``; APPEND actions
        always add a new event.
        """

        normalized = action.normalized()
        timestamp = (now or _utcnow()).astimezone(timezone.utc).isoformat()
        memory_id = _identity(normalized.channel, normalized.coordinate, normalized.key)

        with _exclusive_file_lock(self.lock_path):
            rows = self._load_rows()
            current = _fold_current(rows)
            existing = current.get(memory_id)

            status = "appended"
            row: dict[str, Any] | None = None
            changed = True

            if normalized.action == "CREATE" and existing is not None:
                status = "exists"
                changed = False
            elif normalized.action == "UPSERT" and _same_payload(existing, normalized):
                status = "unchanged"
                changed = False
            else:
                row = _row_from_action(
                    normalized,
                    memory_id=memory_id,
                    timestamp=timestamp,
                    previous_event_id=existing.get("event_id") if existing else None,
                )
                self._append_row(row)
                rows = [*rows, row]
                if normalized.action == "CREATE":
                    status = "created"
                elif normalized.action == "UPSERT":
                    status = "updated" if existing else "inserted"

            projections = self._sync_projections(rows)
            return MemoryActionResult(
                changed=changed,
                status=status,
                event_id=row.get("event_id") if row else None,
                memory_id=memory_id,
                projection_paths=projections,
            )

    def _load_rows(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MemoryActionError(
                    f"invalid memory action ledger JSON at line {line_number}"
                ) from exc
            _validate_row(row, line_number=line_number)
            rows.append(row)
        return rows

    def _append_row(self, row: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _sync_projections(self, rows: list[dict[str, Any]]) -> dict[str, Path]:
        projection_paths: dict[str, Path] = {}
        entries_by_channel = _projection_entries(rows)
        for channel in PROJECTION_CHANNELS:
            path = self.path / f"{channel}.md"
            _atomic_write_text(
                path, _render_projection(channel, entries_by_channel.get(channel, []))
            )
            projection_paths[channel] = path
        return projection_paths


def _row_from_action(
    action: MemoryAction,
    *,
    memory_id: str,
    timestamp: str,
    previous_event_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": MEMORY_ACTION_SCHEMA_VERSION,
        "action": action.action,
        "channel": action.channel,
        "coordinate": action.coordinate,
        "key": action.key,
        "content": action.content,
        "source": action.source,
        "tags": list(action.tags),
        "metadata": action.metadata,
        "memory_id": memory_id,
        "previous_event_id": previous_event_id,
        "created_at": timestamp,
    }
    payload["event_id"] = _event_id(payload)
    return payload


def _validate_row(row: Any, *, line_number: int) -> None:
    if not isinstance(row, dict):
        raise MemoryActionError(f"memory action ledger line {line_number} is not an object")
    if row.get("schema_version") != MEMORY_ACTION_SCHEMA_VERSION:
        raise MemoryActionError(f"unsupported memory action schema at line {line_number}")
    required = {
        "action",
        "channel",
        "coordinate",
        "key",
        "content",
        "source",
        "tags",
        "metadata",
        "memory_id",
        "event_id",
        "created_at",
    }
    missing = sorted(required - set(row))
    if missing:
        raise MemoryActionError(
            f"memory action ledger line {line_number} missing fields: {', '.join(missing)}"
        )
    MemoryAction(
        action=row["action"],
        channel=row["channel"],
        coordinate=row["coordinate"],
        key=row["key"],
        content=row["content"],
        source=row["source"],
        tags=tuple(row.get("tags") or ()),
        metadata=dict(row.get("metadata") or {}),
    ).normalized()
    expected_memory_id = _identity(row["channel"], row["coordinate"], row["key"])
    if row["memory_id"] != expected_memory_id:
        raise MemoryActionError(f"memory action ledger line {line_number} has bad memory_id")


def _same_payload(existing: dict[str, Any] | None, action: MemoryAction) -> bool:
    if existing is None:
        return False
    return (
        existing.get("content") == action.content
        and tuple(existing.get("tags") or ()) == action.tags
        and existing.get("source") == action.source
        and dict(existing.get("metadata") or {}) == action.metadata
    )


def _fold_current(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("action") == "APPEND":
            continue
        current[row["memory_id"]] = row
    return current


def _projection_entries(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    append_rows = [row for row in rows if row.get("action") == "APPEND"]
    current_rows = list(_fold_current(rows).values())
    entries_by_channel: dict[str, list[dict[str, Any]]] = {}
    for row in [*append_rows, *current_rows]:
        entries_by_channel.setdefault(row["channel"], []).append(row)
    for entries in entries_by_channel.values():
        entries.sort(key=lambda row: (row.get("created_at", ""), row.get("event_id", "")))
    return entries_by_channel


def _render_projection(channel: str, entries: list[dict[str, Any]]) -> str:
    title = channel.replace("_", " ").title()
    lines = [
        f"# {title} Memory",
        "",
        "<!-- Generated by MemoryActionStore; edit memory_actions.jsonl through actions. -->",
        "",
    ]
    if not entries:
        lines.append("_No entries._")
        lines.append("")
        return "\n".join(lines)
    for row in entries:
        tags = ", ".join(row.get("tags") or [])
        tag_suffix = f" [{tags}]" if tags else ""
        lines.append(f"## {row['coordinate']} :: {row['key']}{tag_suffix}")
        lines.append("")
        lines.append(row["content"])
        lines.append("")
        lines.append(
            f"- action: {row['action']} | source: {row['source']} | event: {row['event_id']}"
        )
        lines.append("")
    return "\n".join(lines)
