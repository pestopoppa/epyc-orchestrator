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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

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


def shard_paths(main_path: Path | str) -> list[Path]:
    """Return the primary journal file plus its rotated shards, in read order.

    The autopilot journal rotates by appending ``_<n>`` before the suffix,
    e.g. ``autopilot_journal.jsonl`` rotates to ``autopilot_journal_1.jsonl``,
    ``autopilot_journal_2.jsonl``, ... (memory: journal rotates to ``_<n>``).
    We must ingest EVERY shard, not just the live file, or historical trials
    silently disappear from the store.

    Ordering: primary first, then shards by ascending ``<n>``. Dedup in the
    store is keyed by ``(source_path, source_line)`` and each shard has a
    distinct path, so ordering only affects presentation, never correctness.

    ``.bak-*`` / ``.run3-poisoned`` backups are deliberately excluded: they do
    not end in the bare suffix and never match the strict ``_<n><suffix>`` regex.
    """
    main_path = Path(main_path)
    parent = main_path.parent
    stem = main_path.stem  # e.g. "autopilot_journal"
    suffix = main_path.suffix  # e.g. ".jsonl"

    paths: list[Path] = []
    if main_path.exists():
        paths.append(main_path)

    shard_re = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
    numbered: list[tuple[int, Path]] = []
    if parent.exists():
        for candidate in parent.glob(f"{stem}_*{suffix}"):
            m = shard_re.match(candidate.name)
            if m:
                numbered.append((int(m.group(1)), candidate))
    paths.extend(p for _, p in sorted(numbered))
    return paths


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

    def _run_parser(path: Path, parser, kind: str) -> None:
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

    # TSV + JSONL journals rotate into ``_<n>`` shards — ingest every shard.
    for primary, parser in (
        (Path(tsv_path), _parse_tsv),
        (Path(jsonl_path), _parse_jsonl),
    ):
        shards = shard_paths(primary)
        if not shards:
            logger.info(
                "autopilot journal not found at %s (no shards) — emitting source_unavailable",
                primary,
            )
            events.append(_emit_unavailable(primary, EventSource.AUTOPILOT_JOURNAL))
            continue
        for shard in shards:
            _run_parser(shard, parser, EventSource.AUTOPILOT_JOURNAL)

    # Controller state snapshot is a single file (not rotated).
    state = Path(state_path)
    if not state.exists():
        logger.info("autopilot state not found at %s — emitting source_unavailable", state)
        events.append(_emit_unavailable(state, EventSource.AUTOPILOT_STATE))
    else:
        _run_parser(state, _parse_state, EventSource.AUTOPILOT_STATE)

    return events
