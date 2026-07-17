#!/usr/bin/env python3
"""Remove stale AutoPilot state archive cache in favor of journal replay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from archive_authority_report import build_archive_authority_report  # noqa: E402
from preflight_audit import JOURNAL_PATH, STATE_PATH, _load_jsonl  # noqa: E402
from preflight_audit import archive_replay_kwargs_from_state  # noqa: E402
from src.autopilot_core.journal_reconstruction import (  # noqa: E402
    reconstruct_archive_from_journal_rows,
)
from state_store import save_state  # noqa: E402
from state_lock import state_write_lock  # noqa: E402
from contextlib import nullcontext  # noqa: E402


@dataclass(frozen=True)
class ArchiveRepairResult:
    status: str
    before: dict[str, Any]
    after: dict[str, Any] | None = None
    backup_path: str = ""
    warning: str = ""


def _load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"state file is not a JSON object: {path}")
    return state


def build_repaired_state(
    state: dict[str, Any],
    journal_rows: list[dict[str, Any]],
    *,
    max_examples: int = 20,
) -> tuple[dict[str, Any], ArchiveRepairResult]:
    """Return state with the legacy pareto_archive cache removed."""
    before = build_archive_authority_report(
        state,
        journal_rows,
        max_examples=max_examples,
    )
    state_archive_present = isinstance(state.get("pareto_archive"), dict) and bool(
        state.get("pareto_archive")
    )
    if before["ok"] and not state_archive_present:
        return state, ArchiveRepairResult(status="already_aligned", before=before)

    archive = reconstruct_archive_from_journal_rows(
        journal_rows,
        None,
        current_run_only=False,
        **archive_replay_kwargs_from_state(state),
    )
    if archive is None:
        return state, ArchiveRepairResult(
            status="unreconstructable",
            before=before,
            warning="journal replay produced no archive payload",
        )

    repaired = dict(state)
    repaired.pop("pareto_archive", None)
    after = build_archive_authority_report(
        repaired,
        journal_rows,
        max_examples=max_examples,
    )
    status = "ready" if after["ok"] else "postcheck_failed"
    warning = "" if after["ok"] else "repaired state still does not match journal authority"
    return repaired, ArchiveRepairResult(
        status=status,
        before=before,
        after=after,
        warning=warning,
    )


def _backup_state(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_suffix(path.suffix + f".bak-archive-repair-{timestamp}")
    shutil.copy2(path, backup)
    return backup


def repair_state_file(
    state_path: Path,
    journal_path: Path,
    *,
    write: bool = False,
    expect_trial_counter: int | None = None,
    max_examples: int = 20,
) -> ArchiveRepairResult:
    """Build and optionally write a journal-authoritative state-cache removal.

    H4: when actually writing, hold the cross-process ``state_write_lock`` across
    the WHOLE read (``_load_state``) -> build -> write so the repair's read and
    its whole-file overwrite are one atomic critical section that cannot lose a
    concurrent daemon/dashboard update. Read-only (``write=False``) inspection
    takes no lock.
    """
    lock_cm = state_write_lock(state_path) if write else nullcontext()
    with lock_cm:
        state = _load_state(state_path)
        actual_counter = state.get("trial_counter")
        if expect_trial_counter is not None and actual_counter != expect_trial_counter:
            before = build_archive_authority_report(
                state,
                _load_jsonl(journal_path),
                max_examples=max_examples,
            )
            return ArchiveRepairResult(
                status="trial_counter_mismatch",
                before=before,
                warning=(
                    f"expected trial_counter {expect_trial_counter}, "
                    f"found {actual_counter}"
                ),
            )

        journal_rows = _load_jsonl(journal_path)
        repaired, result = build_repaired_state(
            state,
            journal_rows,
            max_examples=max_examples,
        )
        if not write or result.status != "ready":
            return result

        backup = _backup_state(state_path)
        save_state(state_path, repaired)
        return ArchiveRepairResult(
            status="written",
            before=result.before,
            after=result.after,
            backup_path=str(backup),
        )


def _summary_lines(result: ArchiveRepairResult) -> list[str]:
    before = result.before.get("diagnostic", {})
    after = (result.after or {}).get("diagnostic", {})
    lines = [
        f"Archive repair status: {result.status}",
        (
            "Before: "
            f"status={before.get('status', 'n/a')} "
            f"state_entries={before.get('state_entry_count', 'n/a')} "
            f"journal_entries={before.get('journal_entry_count', 'n/a')} "
            f"state_frontier={before.get('state_frontier_count', 'n/a')} "
            f"journal_frontier={before.get('journal_frontier_count', 'n/a')}"
        ),
    ]
    if after:
        lines.append(
            "After: "
            f"status={after.get('status', 'n/a')} "
            f"state_entries={after.get('state_entry_count', 'n/a')} "
            f"journal_entries={after.get('journal_entry_count', 'n/a')} "
            f"state_frontier={after.get('state_frontier_count', 'n/a')} "
            f"journal_frontier={after.get('journal_frontier_count', 'n/a')}"
        )
    if result.backup_path:
        lines.append(f"Backup: {result.backup_path}")
    if result.warning:
        lines.append(f"Warning: {result.warning}")
    return lines


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run by default. With --write, remove only the legacy "
            "autopilot_state.json:pareto_archive cache after journal replay verifies."
        )
    )
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--journal", type=Path, default=JOURNAL_PATH)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--write", action="store_true", help="Write the repair.")
    parser.add_argument(
        "--expect-trial-counter",
        type=int,
        help="Refuse to write if state trial_counter changed since inspection.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    state_path = args.state.expanduser().resolve()
    journal_path = args.journal.expanduser().resolve()
    if not state_path.exists():
        print(f"state file does not exist: {state_path}", file=sys.stderr)
        return 2
    if not journal_path.exists():
        print(f"journal file does not exist: {journal_path}", file=sys.stderr)
        return 2

    result = repair_state_file(
        state_path,
        journal_path,
        write=args.write,
        expect_trial_counter=args.expect_trial_counter,
        max_examples=max(0, args.max_examples),
    )
    if args.json:
        print(json.dumps(result.__dict__, sort_keys=True, default=str))
    else:
        print("\n".join(_summary_lines(result)))
    if result.status in {"ready", "already_aligned", "written"}:
        return 0
    if result.status == "trial_counter_mismatch":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
