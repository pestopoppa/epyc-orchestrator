#!/usr/bin/env python3
"""Replay the autopilot journal under legacy and task-rate objective policies."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.autopilot_core.journal_reconstruction import (  # noqa: E402
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.pareto_math import dominates  # noqa: E402
from src.autopilot_core.tier_specs import (  # noqa: E402
    DEFAULT_FRONTIER_TIER,
    LEGACY_OBJECTIVE_POLICY,
    TASK_RATE_OBJECTIVE_POLICY,
    goodput_qph_from_row,
    task_rate_objectives_from_row,
    task_rate_qph_from_row,
)

DEFAULT_JOURNAL = REPO / "orchestration" / "autopilot_journal.jsonl"
DEFAULT_STATE = REPO / "orchestration" / "autopilot_state.json"
DEFAULT_REPORT_DIR = REPO / "orchestration" / "reports"


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows, malformed


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _details(row: dict[str, Any]) -> dict[str, Any]:
    eval_details = row.get("eval_details") or {}
    details = eval_details.get("details") or {}
    return details if isinstance(details, dict) else {}


def _n_questions(row: dict[str, Any]) -> int | None:
    details = _details(row)
    direct = row.get("n_questions") or details.get("total") or details.get("n_questions")
    n_questions = _as_int(direct)
    if n_questions is not None:
        return n_questions
    counts = details.get("per_suite_counts")
    if isinstance(counts, dict):
        total = 0
        for value in counts.values():
            count = _as_int(value)
            if count and count > 0:
                total += count
        return total or None
    return None


def _eval_wall_s(row: dict[str, Any]) -> float | None:
    eval_details = row.get("eval_details") or {}
    return (
        _as_float(row.get("eval_wall_s"))
        or _as_float(eval_details.get("eval_wall_s"))
        or _as_float(_details(row).get("eval_wall_s"))
    )


def _tokens_per_solved_task(row: dict[str, Any]) -> float | None:
    details = _details(row)
    existing = _as_float(details.get("tokens_per_solved_task"))
    if existing is not None:
        return existing
    tokens = _as_float(details.get("tokens_generated"))
    solved = _as_float(details.get("correct"))
    if tokens is None or solved is None or solved <= 0:
        return None
    return tokens / solved


def _fmt(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _trial_id(entry: dict[str, Any]) -> int | None:
    return _as_int(entry.get("trial_id"))


def _frontier_for(archive: dict[str, Any]) -> list[dict[str, Any]]:
    frontiers = archive.get("frontiers_by_tier") or {}
    return (
        frontiers.get(str(DEFAULT_FRONTIER_TIER))
        or frontiers.get(DEFAULT_FRONTIER_TIER)
        or archive.get("frontier")
        or []
    )


def _archive(
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    objective_policy: str,
    current_run_only: bool,
) -> dict[str, Any]:
    try:
        pareto_epoch_ts = float(state.get("pareto_epoch_ts") or 0.0) or None
    except (TypeError, ValueError):
        pareto_epoch_ts = None
    try:
        deinflate_factor = float(state.get("pareto_pre_epoch_speed_factor", 0.5))
    except (TypeError, ValueError):
        deinflate_factor = 0.5
    archive = reconstruct_archive_from_journal_rows(
        rows,
        None,
        current_run_only=current_run_only,
        deinflate_before_ts=pareto_epoch_ts,
        deinflate_factor=deinflate_factor,
        objective_policy=objective_policy,
    )
    if archive is None:
        raise SystemExit("no reconstructable journal rows")
    return archive


def _dominators(
    row: dict[str, Any],
    task_rate_frontier: list[dict[str, Any]],
) -> list[int]:
    objectives = task_rate_objectives_from_row(row)
    if objectives is None:
        return []
    dominated_by: list[int] = []
    for entry in task_rate_frontier:
        if dominates(entry.get("objectives") or [], objectives):
            tid = _trial_id(entry)
            if tid is not None:
                dominated_by.append(tid)
    return dominated_by


def _row_table(
    title: str,
    entries: list[dict[str, Any]],
    rows_by_tid: dict[int, dict[str, Any]],
    task_rate_frontier: list[dict[str, Any]],
    *,
    include_dominators: bool,
) -> list[str]:
    lines = [f"## {title}", ""]
    if not entries:
        lines.extend(["None.", ""])
        return lines
    header = (
        "| Trial | Quality | Speed t/s | Wall s | N | task_rate q/h | "
        "goodput q/h | Tokens/solved | Dominated by task-rate |"
    )
    lines.extend([
        header,
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for entry in entries:
        tid = _trial_id(entry)
        row = rows_by_tid.get(tid or -1, {})
        dom = _dominators(row, task_rate_frontier) if include_dominators else []
        lines.append(
            "| "
            + " | ".join([
                str(tid if tid is not None else "n/a"),
                _fmt(row.get("quality"), 3),
                _fmt(row.get("speed"), 2),
                _fmt(_eval_wall_s(row), 1),
                str(_n_questions(row) or "n/a"),
                _fmt(task_rate_qph_from_row(row), 2),
                _fmt(goodput_qph_from_row(row), 2),
                _fmt(_tokens_per_solved_task(row), 1),
                ", ".join(str(item) for item in dom) or "n/a",
            ])
            + " |"
        )
    lines.append("")
    return lines


def _write_report(
    path: Path,
    *,
    journal: Path,
    rows: list[dict[str, Any]],
    malformed: int,
    legacy: dict[str, Any],
    task_rate: dict[str, Any],
    current_run_only: bool,
) -> None:
    rows_by_tid = {
        tid: row
        for row in rows
        if (tid := _as_int(row.get("trial_id"))) is not None
    }
    legacy_frontier = _frontier_for(legacy)
    task_rate_frontier = _frontier_for(task_rate)
    legacy_ids = {_trial_id(entry) for entry in legacy_frontier}
    task_rate_ids = {_trial_id(entry) for entry in task_rate_frontier}
    dropped_ids = sorted(tid for tid in legacy_ids - task_rate_ids if tid is not None)
    added_ids = sorted(tid for tid in task_rate_ids - legacy_ids if tid is not None)
    dropped = [entry for entry in legacy_frontier if _trial_id(entry) in dropped_ids]
    added = [entry for entry in task_rate_frontier if _trial_id(entry) in added_ids]

    criterion_met = len(dropped_ids) >= 2 and len(legacy_frontier) == 5
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    scope = "latest trial-id reset segment" if current_run_only else "full journal"
    lines = [
        "# Task-rate / Goodput Replay Report",
        "",
        f"Generated: {generated}",
        f"Journal: `{journal}`",
        f"Scope: {scope}",
        f"Rows parsed: {len(rows)} ({malformed} malformed skipped)",
        "",
        "## Verdict",
        "",
        (
            f"{len(dropped_ids)} of {len(legacy_frontier)} legacy canonical T"
            f"{DEFAULT_FRONTIER_TIER} frontier points fall off under "
            f"`{TASK_RATE_OBJECTIVE_POLICY}`."
        ),
        (
            "Fable criterion (`>=2 of 5`) is "
            f"{'met' if criterion_met else 'not met'} on this replay."
        ),
        "",
        "## Frontier Summary",
        "",
        "| Policy | Frontier points | All admitted entries | Hypervolume final |",
        "|---|---:|---:|---:|",
        (
            f"| `{LEGACY_OBJECTIVE_POLICY}` | {len(legacy_frontier)} | "
            f"{len(legacy.get('all_entries') or [])} | "
            f"{_fmt((legacy.get('hypervolume_history') or [[None, 0]])[-1][1], 4)} |"
        ),
        (
            f"| `{TASK_RATE_OBJECTIVE_POLICY}` | {len(task_rate_frontier)} | "
            f"{len(task_rate.get('all_entries') or [])} | "
            f"{_fmt((task_rate.get('hypervolume_history') or [[None, 0]])[-1][1], 4)} |"
        ),
        "",
    ]
    lines.extend(
        _row_table(
            "Legacy Frontier Points Dropped Under Task-rate",
            dropped,
            rows_by_tid,
            task_rate_frontier,
            include_dominators=True,
        )
    )
    lines.extend(
        _row_table(
            "Task-rate Frontier Additions",
            added,
            rows_by_tid,
            task_rate_frontier,
            include_dominators=False,
        )
    )
    lines.extend([
        "## Notes",
        "",
        "- Dominance uses `(quality, task_rate_qph, reliability)` for the task-rate policy.",
        "- `goodput_qph` is reported as a diagnostic: `(quality / 3) * task_rate_qph`.",
        "- Legacy speed de-inflation is preserved for `legacy_4d_v1` and ignored for task-rate replay.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--current-run-only",
        action="store_true",
        help="replay only the latest contiguous trial-id segment",
    )
    args = parser.parse_args()

    rows, malformed = _read_jsonl(args.journal)
    state = _read_state(args.state)
    legacy = _archive(
        rows,
        state,
        objective_policy=LEGACY_OBJECTIVE_POLICY,
        current_run_only=args.current_run_only,
    )
    task_rate = _archive(
        rows,
        state,
        objective_policy=TASK_RATE_OBJECTIVE_POLICY,
        current_run_only=args.current_run_only,
    )

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output = DEFAULT_REPORT_DIR / f"task_rate_goodput_replay_{stamp}.md"
    _write_report(
        output,
        journal=args.journal,
        rows=rows,
        malformed=malformed,
        legacy=legacy,
        task_rate=task_rate,
        current_run_only=args.current_run_only,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
