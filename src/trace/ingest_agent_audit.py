"""Parse /workspace/logs/agent_audit.log into normalized events.

Two formats coexist in the file:
1. JSON (~1459/2698 lines, recent): `{"ts","session","level","cat","msg","details"}`
2. Legacy text (~827/2698 lines, older): `[ts] CATEGORY: msg | k=v | k=v ...`
3. Other (~412/2698 lines): blank lines, comments, malformed entries.

Both formats normalize to the same Event shape. Dedup key is (source_path, line_number).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.trace.store import Event, EventCategory, EventSource, detail_to_json

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("/workspace/logs/agent_audit.log")

# Map upstream category strings to canonical EventCategory values.
# Categories not in this map are stored verbatim (lowercased).
_CATEGORY_MAP = {
    "SESSION_START": EventCategory.SESSION_START,
    "SESSION_END": EventCategory.SESSION_END,
    "SESSION_SUMMARY": EventCategory.SESSION_SUMMARY,
    "TASK_START": EventCategory.TASK_START,
    "TASK_END": EventCategory.TASK_END,
    "CMD_INTENT": EventCategory.CMD_INTENT,
    "CMD_RESULT": EventCategory.CMD_RESULT,
    "DECISION": EventCategory.DECISION,
    "OBSERVE": EventCategory.OBSERVE,
    "WARN": EventCategory.WARN,
    "WARNING": EventCategory.WARN,
    "ERROR": EventCategory.ERROR,
    "FILE_MODIFY": EventCategory.FILE_MODIFY,
    "FILE_CREATE": EventCategory.FILE_MODIFY,
    "FILE_ADD": EventCategory.FILE_MODIFY,
    "FILE_MOVE": EventCategory.FILE_MODIFY,
    "EDIT": EventCategory.FILE_MODIFY,
    "ROLLBACK": EventCategory.ROLLBACK,
    "DISCOVERY": EventCategory.DISCOVERY,
    "VERIFY": EventCategory.VERIFY,
    "DOCS": EventCategory.DOCS,
}

_TEXT_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+(?P<cat>[A-Z_]+)(?::\s*|\s+)(?P<msg>.+?)(?:\s*\|\s*(?P<extras>.+))?$"
)
_KV_RE = re.compile(r"(?P<k>[A-Za-z_][\w]*)\s*=\s*(?P<v>[^|]+?)(?=\s*\||$)")


def _normalize_ts(raw: str) -> str:
    """Best-effort UTC ISO8601 normalization."""
    raw = raw.strip()
    try:
        # Try ISO 8601 directly (handles +HH:MM and Z).
        if raw.endswith("Z"):
            dt = datetime.fromisoformat(raw[:-1]).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return raw  # store as-is if unparseable; query layer can still sort lexically


def _canonical_category(raw: str) -> str:
    raw = (raw or "").strip().upper()
    return _CATEGORY_MAP.get(raw, raw.lower() if raw else "unknown")


def _status_from_details(details: str) -> str | None:
    """Extract status= or outcome= from details string, if present."""
    if not details:
        return None
    for key in ("status", "outcome", "result"):
        m = re.search(rf"\b{key}\s*=\s*(\w+)", details)
        if m:
            return m.group(1).lower()
    return None


def _parse_json_line(line: str, source_path: str, lineno: int) -> Event | None:
    try:
        d = json.loads(line)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None

    ts = _normalize_ts(d.get("ts", ""))
    cat_raw = d.get("cat") or d.get("category") or ""
    msg = d.get("msg") or d.get("message") or ""
    details = d.get("details") or ""
    return Event(
        ts_utc=ts,
        source=EventSource.AGENT_AUDIT,
        source_path=source_path,
        source_line=lineno,
        session_id=d.get("session"),
        category=_canonical_category(str(cat_raw)),
        status=_status_from_details(str(details)),
        summary=str(msg),
        detail_json=detail_to_json(d),
    )


def _parse_text_line(line: str, source_path: str, lineno: int) -> Event | None:
    m = _TEXT_LINE_RE.match(line)
    if not m:
        return None
    ts = _normalize_ts(m.group("ts"))
    cat = m.group("cat")
    msg = m.group("msg") or ""
    extras = m.group("extras") or ""

    extras_dict: dict[str, str] = {}
    for kv in _KV_RE.finditer(extras):
        extras_dict[kv.group("k")] = kv.group("v").strip()

    detail = {
        "ts": ts,
        "cat": cat,
        "msg": msg,
        "extras": extras_dict,
    }
    return Event(
        ts_utc=ts,
        source=EventSource.AGENT_AUDIT,
        source_path=source_path,
        source_line=lineno,
        session_id=extras_dict.get("session"),
        category=_canonical_category(cat),
        status=extras_dict.get("status") or extras_dict.get("outcome"),
        summary=msg,
        detail_json=detail_to_json(detail),
    )


def parse_lines(lines: list[str], source_path: str) -> Iterator[Event]:
    """Parse a list of log lines (1-indexed line numbers in source_line)."""
    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("{"):
            ev = _parse_json_line(line, source_path, i)
        elif line.startswith("["):
            ev = _parse_text_line(line, source_path, i)
        else:
            ev = None
        if ev is not None:
            yield ev


def parse_file(path: Path | str = DEFAULT_LOG_PATH) -> list[Event]:
    """Parse the log file at `path`. Returns [] if file is absent."""
    path = Path(path)
    if not path.exists():
        logger.info("agent_audit.log not found at %s — skipping", path)
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return list(parse_lines(f.readlines(), str(path)))
