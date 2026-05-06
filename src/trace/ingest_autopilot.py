"""Parse autopilot journal artifacts.

Three sources, all under epyc-orchestrator/orchestration/:
- autopilot_journal.tsv  (one row per trial, summary)
- autopilot_journal.jsonl (full per-trial detail)
- autopilot_state.json (controller state snapshot at trial boundary)

Files may be absent on a given host (autopilot has not run, or this is a
fresh checkout). When absent, emit a single `source_unavailable` event so
the unified store records the gap rather than silently skipping. When the
files appear later, idempotent re-ingest picks them up.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from src.trace.store import Event, EventCategory, EventSource, detail_to_json

logger = logging.getLogger(__name__)

# Default paths under the orchestrator repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOURNAL_TSV = _REPO_ROOT / "orchestration" / "autopilot_journal.tsv"
DEFAULT_JOURNAL_JSONL = _REPO_ROOT / "orchestration" / "autopilot_journal.jsonl"
DEFAULT_STATE_JSON = _REPO_ROOT / "orchestration" / "autopilot_state.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_ts(raw: str | None) -> str:
    if not raw:
        return _now_utc()
    raw = raw.strip()
    try:
        if raw.endswith("Z"):
            dt = datetime.fromisoformat(raw[:-1]).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return raw


def _emit_unavailable(path: Path, kind: str) -> Event:
    return Event(
        ts_utc=_now_utc(),
        source=kind,
        source_path=str(path),
        source_line=0,
        category=EventCategory.SOURCE_UNAVAILABLE,
        status="absent",
        summary=f"{kind} not found at {path}",
        detail_json=detail_to_json({"path": str(path), "kind": kind}),
    )


def _parse_tsv(path: Path) -> Iterator[Event]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader, start=2):  # +1 for header
            trial_id_raw = row.get("trial_id") or row.get("trial") or ""
            try:
                trial_id = int(trial_id_raw) if trial_id_raw else None
            except ValueError:
                trial_id = None
            ts = _normalize_ts(row.get("ts") or row.get("timestamp") or row.get("created_at"))
            yield Event(
                ts_utc=ts,
                source=EventSource.AUTOPILOT_JOURNAL,
                source_path=str(path),
                source_line=i,
                trial_id=trial_id,
                role=row.get("role") or row.get("species"),
                category="trial_summary",
                status=row.get("status") or row.get("outcome"),
                summary=row.get("note") or row.get("summary") or row.get("msg"),
                detail_json=detail_to_json(row),
            )


def _parse_jsonl(path: Path) -> Iterator[Event]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(d, dict):
                continue
            trial_id_raw = d.get("trial_id") or d.get("trial")
            try:
                trial_id = int(trial_id_raw) if trial_id_raw is not None else None
            except (ValueError, TypeError):
                trial_id = None
            ts = _normalize_ts(d.get("ts") or d.get("timestamp") or d.get("created_at"))
            cat = d.get("event") or d.get("category") or "trial_event"
            yield Event(
                ts_utc=ts,
                source=EventSource.AUTOPILOT_JOURNAL,
                source_path=str(path),
                source_line=i,
                trial_id=trial_id,
                role=d.get("role") or d.get("species"),
                category=str(cat).lower(),
                status=d.get("status") or d.get("outcome"),
                summary=d.get("summary") or d.get("msg") or d.get("note"),
                detail_json=detail_to_json(d),
            )


def _parse_state(path: Path) -> Iterator[Event]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        try:
            d = json.load(f)
        except (ValueError, json.JSONDecodeError):
            return
    if not isinstance(d, dict):
        return
    trial_id_raw = d.get("trial_id") or d.get("trial") or d.get("current_trial")
    try:
        trial_id = int(trial_id_raw) if trial_id_raw is not None else None
    except (ValueError, TypeError):
        trial_id = None
    ts = _normalize_ts(d.get("ts") or d.get("timestamp"))
    yield Event(
        ts_utc=ts,
        source=EventSource.AUTOPILOT_STATE,
        source_path=str(path),
        source_line=0,
        trial_id=trial_id,
        category=EventCategory.CONTROLLER_SNAPSHOT,
        summary=f"controller snapshot @ trial {trial_id}",
        detail_json=detail_to_json(d),
    )


def parse_all(
    tsv_path: Path | str = DEFAULT_JOURNAL_TSV,
    jsonl_path: Path | str = DEFAULT_JOURNAL_JSONL,
    state_path: Path | str = DEFAULT_STATE_JSON,
) -> list[Event]:
    """Parse all three autopilot artifacts. Returns [] entries with
    `source_unavailable` events for any absent file.
    """
    events: list[Event] = []
    for path, parser, kind in (
        (Path(tsv_path), _parse_tsv, EventSource.AUTOPILOT_JOURNAL),
        (Path(jsonl_path), _parse_jsonl, EventSource.AUTOPILOT_JOURNAL),
        (Path(state_path), _parse_state, EventSource.AUTOPILOT_STATE),
    ):
        if not path.exists():
            logger.info("autopilot source not found at %s — emitting source_unavailable", path)
            events.append(_emit_unavailable(path, kind))
            continue
        try:
            events.extend(parser(path))
        except Exception as e:  # noqa: BLE001 — defensive, log and continue per source
            logger.warning("autopilot parser for %s failed: %s", path, e)
            events.append(
                Event(
                    ts_utc=_now_utc(),
                    source=kind,
                    source_path=str(path),
                    source_line=0,
                    category="parse_error",
                    status="error",
                    summary=f"parser raised {type(e).__name__}: {e}",
                    detail_json=detail_to_json({"path": str(path), "error": str(e)}),
                )
            )
    return events
