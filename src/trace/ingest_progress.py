"""Parse /workspace/progress/YYYY-MM/*.md (+ sibling .jsonl when present).

Each `## ` heading inside a daily progress file becomes a session_summary
event. The heading text becomes summary; the section body (until next `## `
or EOF) becomes detail_json.

When a sibling YYYY-MM-DD.jsonl exists, its lines are emitted as granular
events with the JSONL line number as source_line.
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

DEFAULT_PROGRESS_ROOT = Path("/workspace/progress")

_DATE_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")


def _date_from_filename(path: Path) -> str | None:
    m = _DATE_RE.search(path.name)
    return m.group("date") if m else None


def _to_iso_utc(date_str: str, time_str: str | None = None) -> str:
    """Convert YYYY-MM-DD (+ optional HH:MM[:SS]) to UTC ISO8601.

    Best-effort — progress markdown timestamps are not reliable; default
    to T00:00:00Z when only date is known.
    """
    base = f"{date_str}T{time_str or '00:00:00'}"
    try:
        dt = datetime.fromisoformat(base).replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return f"{date_str}T00:00:00+00:00"


def parse_markdown_file(path: Path) -> Iterator[Event]:
    """Yield one event per ## heading, plus one for the file overall."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    date_str = _date_from_filename(path) or "1970-01-01"
    ts = _to_iso_utc(date_str)

    # File-level marker (line 0 — non-conflicting with per-section line numbers).
    file_summary = f"progress {date_str}"
    yield Event(
        ts_utc=ts,
        source=EventSource.PROGRESS,
        source_path=str(path),
        source_line=0,
        category="progress_file",
        summary=file_summary,
        detail_json=detail_to_json({"date": date_str, "byte_size": len(text)}),
    )

    # Section-level: each ## heading.
    lines = text.splitlines()
    section_start: int | None = None
    section_heading: str | None = None
    section_body: list[str] = []

    def flush(end_line: int):
        nonlocal section_start, section_heading, section_body
        if section_start is None or section_heading is None:
            return
        body = "\n".join(section_body).strip()
        yield Event(
            ts_utc=ts,
            source=EventSource.PROGRESS,
            source_path=str(path),
            source_line=section_start,
            category=EventCategory.SESSION_SUMMARY,
            summary=section_heading.strip(),
            detail_json=detail_to_json(
                {
                    "date": date_str,
                    "heading": section_heading.strip(),
                    "line_range": [section_start, end_line],
                    "body": body[:4000],  # cap to avoid pathological detail rows
                }
            ),
        )
        section_start = None
        section_heading = None
        section_body = []

    for i, line in enumerate(lines, start=1):
        if line.startswith("## ") and not line.startswith("### "):
            yield from flush(i - 1)
            section_start = i
            section_heading = line[3:].strip()
        elif section_start is not None:
            section_body.append(line)
    # Flush trailing section.
    yield from flush(len(lines))


def parse_jsonl_file(path: Path) -> Iterator[Event]:
    """Yield one event per JSONL line. Path is sibling of a markdown file."""
    if not path.exists():
        return
    date_str = _date_from_filename(path) or "1970-01-01"
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(d, dict):
                continue
            ts = d.get("ts") or _to_iso_utc(date_str, d.get("time"))
            yield Event(
                ts_utc=ts,
                source=EventSource.PROGRESS_JSONL,
                source_path=str(path),
                source_line=i,
                session_id=d.get("session"),
                category=str(d.get("event") or d.get("category") or "progress_event").lower(),
                status=d.get("status") or d.get("outcome"),
                summary=d.get("summary") or d.get("msg"),
                detail_json=detail_to_json(d),
            )


def walk_progress_root(root: Path | str = DEFAULT_PROGRESS_ROOT) -> list[Event]:
    """Walk progress/YYYY-MM/*.md (+ matching .jsonl) under `root`.

    Returns aggregated events. Returns [] if root does not exist.
    """
    root = Path(root)
    if not root.exists():
        logger.info("progress root %s not found — skipping", root)
        return []

    events: list[Event] = []
    for md in sorted(root.glob("*/*.md")):
        events.extend(parse_markdown_file(md))
        sibling = md.with_suffix(".jsonl")
        if sibling.exists():
            events.extend(parse_jsonl_file(sibling))
    return events
