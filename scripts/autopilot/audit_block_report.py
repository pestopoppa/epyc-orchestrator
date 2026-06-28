#!/usr/bin/env python3
"""Read-only W6 rotating-audit report for partitioned question results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

UNTRUSTED_OUTCOME_STATUSES = frozenset({"invalid", "skipped"})
DEFAULT_ALARM_WINDOW = 30


def _expand_journal_paths(raw_paths: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = sorted(
                {
                    *path.glob("autopilot_journal*.jsonl"),
                    *path.glob("*.jsonl"),
                },
                key=str,
            )
            paths.extend(candidate for candidate in candidates if candidate.is_file())
        else:
            paths.append(path)
    return paths


def load_journal_rows(journal_paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load JSONL rows from one or more journals, skipping non-trial rows."""
    rows: list[dict[str, Any]] = []
    for path in _expand_journal_paths(journal_paths):
        with Path(path).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
                if _is_trial_row(row):
                    rows.append(row)
    return rows


def _is_trial_row(row: Any) -> bool:
    if not isinstance(row, Mapping):
        return False
    row_type = str(row.get("type") or "").strip().lower()
    if row_type in {"ledger", "supersession"}:
        return False
    trial_id = row.get("trial_id")
    if isinstance(trial_id, bool) or trial_id is None:
        return False
    try:
        int(trial_id)
    except (TypeError, ValueError):
        return False
    return True


def _question_results(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    eval_details = row.get("eval_details")
    if not isinstance(eval_details, Mapping):
        return []
    raw_results = eval_details.get("question_results")
    if not isinstance(raw_results, list):
        nested_details = eval_details.get("details")
        if isinstance(nested_details, Mapping):
            raw_results = nested_details.get("question_results")
    if not isinstance(raw_results, list):
        return []
    results: list[dict[str, Any]] = []
    for item in raw_results:
        if isinstance(item, Mapping):
            results.append(dict(item))
    return results


def _partition_name(item: Mapping[str, Any]) -> str:
    partition = str(item.get("partition") or "core").strip().lower()
    return partition or "core"


def _quality_0_3(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((correct / total) * 3.0, 6)


def _quality_step(total: int | float | None) -> float:
    try:
        n = int(total or 0)
    except (TypeError, ValueError):
        return 0.0
    if n <= 0:
        return 0.0
    return round(3.0 / n, 6)


def _trial_summary(row: Mapping[str, Any]) -> dict[str, Any] | None:
    core_correct = core_total = audit_correct = audit_total = 0
    for item in _question_results(row):
        partition = _partition_name(item)
        if partition not in {"core", "audit"}:
            continue
        correct = bool(item.get("correct"))
        if partition == "core":
            core_total += 1
            core_correct += int(correct)
        else:
            audit_total += 1
            audit_correct += int(correct)

    if audit_total == 0:
        return None

    core_quality = _quality_0_3(core_correct, core_total)
    audit_quality = _quality_0_3(audit_correct, audit_total)
    return {
        "trial_id": int(row["trial_id"]),
        "core_correct": core_correct,
        "core_total": core_total,
        "audit_correct": audit_correct,
        "audit_total": audit_total,
        "core_quality_0_3": core_quality,
        "audit_quality_0_3": audit_quality,
        "delta_audit_minus_core": round(audit_quality - core_quality, 6),
    }


def build_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    alarm_window: int | None = None,
    exclude_before_ts: float | None = None,
) -> dict[str, Any]:
    trial_rows = [row for row in rows if _is_trial_row(row)]
    audit_rows = [
        (row, summary)
        for row in trial_rows
        if (summary := _trial_summary(row)) is not None
    ]
    era_excluded_audit_rows = [
        (row, summary)
        for row, summary in audit_rows
        if _excluded_by_timestamp(row, exclude_before_ts)
    ]
    eligible_audit_rows = [
        (row, summary)
        for row, summary in audit_rows
        if not _excluded_by_timestamp(row, exclude_before_ts)
    ]
    trusted_audit_rows = [
        (row, summary) for row, summary in eligible_audit_rows if _is_trusted_trial_row(row)
    ]
    untrusted_audit_rows = [
        (row, summary)
        for row, summary in eligible_audit_rows
        if not _is_trusted_trial_row(row)
    ]
    trial_summaries = [summary for _row, summary in trusted_audit_rows]
    totals = {
        "core_correct": sum(summary["core_correct"] for summary in trial_summaries),
        "core_total": sum(summary["core_total"] for summary in trial_summaries),
        "audit_correct": sum(summary["audit_correct"] for summary in trial_summaries),
        "audit_total": sum(summary["audit_total"] for summary in trial_summaries),
    }
    totals["core_quality_0_3"] = _quality_0_3(totals["core_correct"], totals["core_total"])
    totals["audit_quality_0_3"] = _quality_0_3(totals["audit_correct"], totals["audit_total"])
    totals["delta_audit_minus_core"] = round(
        totals["audit_quality_0_3"] - totals["core_quality_0_3"], 6
    )
    gaming_diagnostic = _gaming_diagnostic(trial_summaries, alarm_window=alarm_window)
    return {
        "trial_count": len(trial_rows),
        "raw_audited_trial_count": len(eligible_audit_rows),
        "all_raw_audited_trial_count": len(audit_rows),
        "era_exclude_before_ts": exclude_before_ts,
        "era_excluded_audited_trial_count": len(era_excluded_audit_rows),
        "era_excluded_audited_trial_ids": _trial_ids(
            row for row, _summary in era_excluded_audit_rows
        ),
        "trusted_audited_trial_count": len(trusted_audit_rows),
        "untrusted_audited_trial_count": len(untrusted_audit_rows),
        "untrusted_audited_trial_ids": _trial_ids(row for row, _summary in untrusted_audit_rows),
        "audited_trial_count": len(trial_summaries),
        "totals": totals,
        "trials": trial_summaries,
        "gaming_alarm": gaming_diagnostic["gaming_alarm"],
        "gaming_events": gaming_diagnostic["gaming_events"],
        "gaming_alarm_window": gaming_diagnostic["gaming_alarm_window"],
        "gaming_alarm_window_trial_count": gaming_diagnostic[
            "gaming_alarm_window_trial_count"
        ],
        "gaming_alarm_clearance_clean_trials_required": gaming_diagnostic[
            "gaming_alarm_clearance_clean_trials_required"
        ],
        "cumulative_gaming_alarm": gaming_diagnostic["cumulative_gaming_alarm"],
        "cumulative_gaming_events": gaming_diagnostic["cumulative_gaming_events"],
        "transfer_diagnostic": {
            "audited_trial_count": len(trial_summaries),
            "potential_overfit_divergences": len(gaming_diagnostic["gaming_events"]),
            "events": gaming_diagnostic["gaming_events"],
            "cumulative_potential_overfit_divergences": len(
                gaming_diagnostic["cumulative_gaming_events"]
            ),
            "cumulative_events": gaming_diagnostic["cumulative_gaming_events"],
            "alarm_window": gaming_diagnostic["gaming_alarm_window"],
            "alarm_window_trial_count": gaming_diagnostic[
                "gaming_alarm_window_trial_count"
            ],
            "clearance_clean_trials_required": gaming_diagnostic[
                "gaming_alarm_clearance_clean_trials_required"
            ],
        },
    }


def _is_trusted_trial_row(row: Mapping[str, Any]) -> bool:
    if row.get("bug_corrupted_by"):
        return False
    try:
        if int(row.get("tier", 1)) < 1:
            return False
    except (TypeError, ValueError):
        return False
    status = str(row.get("outcome_status") or "ok")
    if status in UNTRUSTED_OUTCOME_STATUSES:
        return False
    return True


def _excluded_by_timestamp(row: Mapping[str, Any], exclude_before_ts: float | None) -> bool:
    if exclude_before_ts is None:
        return False
    row_ts = _row_timestamp(row)
    if row_ts is None:
        return True
    return row_ts < exclude_before_ts


def _row_timestamp(row: Mapping[str, Any]) -> float | None:
    raw = row.get("timestamp")
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int | float):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def _trial_ids(rows: Iterable[Mapping[str, Any]]) -> list[int]:
    trial_ids: list[int] = []
    for row in rows:
        try:
            trial_ids.append(int(row["trial_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(trial_ids)


def _gaming_diagnostic(
    trials: list[dict[str, Any]],
    *,
    alarm_window: int | None = None,
) -> dict[str, Any]:
    if len(trials) < 3:
        return {
            "gaming_alarm": False,
            "gaming_events": [],
            "gaming_alarm_window": alarm_window,
            "gaming_alarm_window_trial_count": len(trials),
            "gaming_alarm_clearance_clean_trials_required": 0,
            "cumulative_gaming_alarm": False,
            "cumulative_gaming_events": [],
        }

    cumulative_events = _gaming_events(trials)
    window_trials = _alarm_window_trials(trials, alarm_window)
    active_events = _gaming_events(window_trials)
    return {
        "gaming_alarm": bool(active_events),
        "gaming_events": active_events,
        "gaming_alarm_window": alarm_window,
        "gaming_alarm_window_trial_count": len(window_trials),
        "gaming_alarm_clearance_clean_trials_required": (
            _gaming_alarm_clearance_clean_trials_required(
                trials,
                active_events,
                alarm_window=alarm_window,
            )
        ),
        "cumulative_gaming_alarm": bool(cumulative_events),
        "cumulative_gaming_events": cumulative_events,
    }


def _alarm_window_trials(
    trials: list[dict[str, Any]],
    alarm_window: int | None,
) -> list[dict[str, Any]]:
    if alarm_window is None or alarm_window <= 0 or alarm_window >= len(trials):
        return trials
    return trials[-alarm_window:]


def _gaming_events(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(trials) < 3:
        return []

    events: list[dict[str, Any]] = []
    prev = trials[0]
    for trial in trials[1:]:
        core_delta = round(trial["core_quality_0_3"] - prev["core_quality_0_3"], 6)
        audit_delta = round(trial["audit_quality_0_3"] - prev["audit_quality_0_3"], 6)
        core_step = max(_quality_step(prev.get("core_total")), _quality_step(trial.get("core_total")))
        audit_step = max(_quality_step(prev.get("audit_total")), _quality_step(trial.get("audit_total")))
        if core_delta > core_step and audit_delta < -audit_step:
            events.append(
                {
                    "trial_id": trial["trial_id"],
                    "previous_trial_id": prev["trial_id"],
                    "core_delta": core_delta,
                    "audit_delta": audit_delta,
                }
            )
        prev = trial

    return events


def _gaming_alarm_clearance_clean_trials_required(
    trials: list[dict[str, Any]],
    active_events: list[dict[str, Any]],
    *,
    alarm_window: int | None,
) -> int | None:
    """Rows needed to age out active events, assuming no new gaming events occur."""
    if not active_events:
        return 0
    if alarm_window is None or alarm_window <= 0:
        return None

    index_by_trial_id = {
        trial["trial_id"]: index
        for index, trial in enumerate(trials)
        if "trial_id" in trial
    }
    remaining = 0
    trial_count = len(trials)
    for event in active_events:
        event_index = index_by_trial_id.get(event.get("trial_id"))
        if event_index is None:
            return None
        remaining = max(remaining, event_index + alarm_window - trial_count)
    return max(0, remaining)


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report["totals"]
    transfer_diagnostic = report["transfer_diagnostic"]
    gaming_events = report.get("gaming_events") or []
    gaming_alarm = bool(report.get("gaming_alarm"))
    alarm_window = report.get("gaming_alarm_window")
    cumulative_events = report.get("cumulative_gaming_events") or gaming_events
    lines = [
        "# W6 Rotating Audit Block Report",
        "",
        f"- Trial rows: `{report['trial_count']}`",
        f"- Audited trials: `{report['audited_trial_count']}`",
        (
            "- Totals: "
            f"core={totals['core_correct']}/{totals['core_total']} "
            f"({totals['core_quality_0_3']:.3f}), "
            f"audit={totals['audit_correct']}/{totals['audit_total']} "
            f"({totals['audit_quality_0_3']:.3f}), "
            f"delta={totals['delta_audit_minus_core']:+.3f}"
        ),
        (
            "- Transfer diagnostic: "
            f"potential overfit divergences={transfer_diagnostic['potential_overfit_divergences']}"
        ),
        (
            "- Cumulative transfer diagnostic: "
            f"potential overfit divergences={len(cumulative_events)}"
        ),
        (
            "- Gaming alarm: "
            f"{'triggered' if gaming_alarm else 'clear'} "
            f"({len(gaming_events)} event{'s' if len(gaming_events) != 1 else ''})"
        ),
    ]
    if alarm_window is not None:
        lines.append(
            "- Gaming alarm window: "
            f"last {alarm_window} audited trial"
            f"{'s' if alarm_window != 1 else ''} "
            f"(available={report.get('gaming_alarm_window_trial_count')})"
        )
    clearance_trials = report.get("gaming_alarm_clearance_clean_trials_required")
    if gaming_alarm and clearance_trials is not None:
        lines.append(
            "- Gaming alarm clearance: "
            f"{clearance_trials} future clean audited trial"
            f"{'s' if clearance_trials != 1 else ''} required "
            "to age active events out of the window"
        )
    lines.extend(
        [
            "",
            "| trial_id | core | audit | core_q | audit_q | delta_audit_minus_core |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for trial in report["trials"]:
        lines.append(
            "| {trial_id} | {core_correct}/{core_total} | {audit_correct}/{audit_total} | "
            "{core_quality_0_3:.3f} | {audit_quality_0_3:.3f} | {delta_audit_minus_core:+.3f} |".format(
                **trial
            )
        )

    if gaming_alarm:
        lines.extend(["", "## Audit Gaming Alarm", ""])
        for event in gaming_events:
            lines.append(
                "- trial {trial_id} vs {previous_trial_id}: core_delta={core_delta:+.3f} "
                "audit_delta={audit_delta:+.3f}".format(**event)
            )
        lines.append("- Action: review transition(s) for audit overfitting risk.")
    else:
        lines.append("")
        lines.append("## Audit Gaming Alarm")
        if cumulative_events:
            lines.append("- No suspicious gaming trend detected in the current window.")
            lines.append(
                "- Historical divergences remain in cumulative evidence: "
                f"{len(cumulative_events)} event"
                f"{'s' if len(cumulative_events) != 1 else ''}."
            )
        else:
            lines.append("- No suspicious gaming trend detected.")

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(report: Mapping[str, Any], out_json: Path | None, out_md: Path | None) -> None:
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(report), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only W6 rotating-audit summary for partitioned question results."
    )
    parser.add_argument(
        "--journal",
        type=Path,
        nargs="+",
        action="append",
        required=True,
        help="One or more AutoPilot journal JSONL files or directories.",
    )
    parser.add_argument("--out-json", type=Path, help="Write JSON summary to this path.")
    parser.add_argument("--out-md", type=Path, help="Write Markdown summary to this path.")
    parser.add_argument(
        "--alarm-window",
        type=int,
        default=DEFAULT_ALARM_WINDOW,
        help=(
            "Evaluate the active gaming alarm over the last N audited trials while "
            "preserving cumulative divergence evidence in the report. "
            f"Defaults to {DEFAULT_ALARM_WINDOW}; use 0 for all audited history."
        ),
    )
    parser.add_argument(
        "--exclude-before-ts",
        type=float,
        help=(
            "Exclude audited rows whose journal timestamp is before this Unix timestamp. "
            "Rows without parseable timestamps are excluded when this fence is set."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    journal_paths = [path for group in args.journal for path in group]
    rows = load_journal_rows(journal_paths)
    report = build_report(
        rows,
        alarm_window=args.alarm_window,
        exclude_before_ts=args.exclude_before_ts,
    )
    if args.out_json is None and args.out_md is None:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        write_outputs(report, args.out_json, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
