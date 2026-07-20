"""A5 journal durability: tolerant read + torn-tail quarantine on append.

Covers ExperimentJournal's crash-durability contract:
  - a torn *trailing* line (partial final append, no newline) loads tolerantly,
  - the next append quarantines those bytes to a `.corrupt-*` sidecar and
    truncates the shard back to the last good newline,
  - mid-file corruption (a bad line with good lines after it) is fatal, and
  - normal appends leave a clean newline-terminated shard.

All fixtures use tmp_path; the live orchestration/ journals are never touched.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

from experiment_journal import (  # noqa: E402
    ExperimentJournal,
    ExperimentJournalCorruptError,
    JournalEntry,
)


def _entry(trial_id: int, *, quality: float = 1.0) -> JournalEntry:
    return JournalEntry(
        trial_id=trial_id,
        timestamp=f"2026-06-14T00:00:0{trial_id}Z",
        species="unit",
        action_type="seed_batch",
        tier=1,
        quality=quality,
        speed=40.0,
        cost=0.2,
        reliability=0.9,
        pareto_status="frontier",
    )


def _line(entry: JournalEntry) -> str:
    """Serialize exactly as ExperimentJournal.record writes JSONL rows."""
    return json.dumps(asdict(entry), default=str)


def _shard(journal_dir: Path) -> Path:
    return journal_dir / "autopilot_journal.jsonl"


def test_torn_trailing_line_loads_tolerantly(tmp_path: Path) -> None:
    # 2 good lines + a torn partial (no trailing newline).
    partial = _line(_entry(3))[:80]  # truncated mid-record, not valid JSON
    _shard(tmp_path).write_bytes(
        (_line(_entry(1)) + "\n" + _line(_entry(2)) + "\n" + partial).encode("utf-8")
    )

    journal = ExperimentJournal(journal_dir=tmp_path)

    assert journal.count() == 2
    assert journal.torn_lines_skipped == 1
    assert [e.trial_id for e in journal.all_entries()] == [1, 2]


def test_next_append_quarantines_torn_tail_to_sidecar(tmp_path: Path) -> None:
    partial = _line(_entry(3))[:80]
    partial_bytes = partial.encode("utf-8")
    _shard(tmp_path).write_bytes(
        (_line(_entry(1)) + "\n" + _line(_entry(2)) + "\n" + partial).encode("utf-8")
    )

    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(3, quality=1.5))

    # The torn bytes were quarantined verbatim, not deleted.
    sidecars = list(tmp_path.glob("autopilot_journal.jsonl.corrupt-*"))
    assert len(sidecars) == 1
    assert sidecars[0].read_bytes() == partial_bytes

    # The shard now parses cleanly line-by-line and a fresh reload sees 3.
    raw = _shard(tmp_path).read_bytes()
    assert raw.endswith(b"\n")
    parsed = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    assert [row["trial_id"] for row in parsed] == [1, 2, 3]

    reloaded = ExperimentJournal(journal_dir=tmp_path)
    assert reloaded.count() == 3
    assert reloaded.torn_lines_skipped == 0
    assert [e.trial_id for e in reloaded.all_entries()] == [1, 2, 3]


def test_midfile_corruption_raises(tmp_path: Path) -> None:
    # A bad line (2) followed by a good line (3) is NOT a torn tail: fatal.
    _shard(tmp_path).write_bytes(
        (
            _line(_entry(1))
            + "\n"
            + '{"trial_id": 2, "timestamp":'  # truncated / invalid JSON
            + "\n"
            + _line(_entry(3))
            + "\n"
        ).encode("utf-8")
    )

    with pytest.raises(ExperimentJournalCorruptError) as excinfo:
        ExperimentJournal(journal_dir=tmp_path)

    message = str(excinfo.value)
    assert "autopilot_journal.jsonl:2:" in message
    assert "scrub_journal.py" in message


def test_append_path_writes_newline_terminated_parseable_rows(tmp_path: Path) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(1))
    journal.record(_entry(2))

    raw = _shard(tmp_path).read_bytes()
    assert raw.endswith(b"\n")
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    parsed = [json.loads(ln) for ln in lines]
    assert [row["trial_id"] for row in parsed] == [1, 2]
