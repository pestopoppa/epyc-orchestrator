#!/usr/bin/env python3
"""Read-only report for AutoPilot handoff closure candidates.

The report intentionally separates "planner memory exists" from "handoff can be
closed". Operator-seeded StrategyStore rows can guide AutoPilot, but handoff
completion remains a governance action. This script only emits review
suggestions with provenance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from experiment_journal import DEFAULT_JOURNAL_DIR, ExperimentJournal  # noqa: E402
from orchestration.repl_memory.strategy_store import DEFAULT_STRATEGY_PATH  # noqa: E402
from seed_operator_strategies import (  # noqa: E402
    DEFAULT_CAMPAIGN,
    DEFAULT_SEED_FILE,
    SeedRow,
    load_seed_rows,
)


def _strategy_db_path(strategy_path: Path) -> Path:
    return strategy_path / "strategies.db" if strategy_path.is_dir() else strategy_path


def _load_campaign_strategy_rows(
    *,
    strategy_path: Path,
    campaign: str,
    expected_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    db_path = _strategy_db_path(strategy_path)
    if not db_path.exists():
        return {}, [f"strategy db does not exist: {db_path}"]

    rows: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return {}, [f"cannot open strategy db read-only: {db_path}: {exc}"]
    conn.row_factory = sqlite3.Row
    try:
        try:
            raw_rows = conn.execute(
                "SELECT id, description, insight, source_trial_id, species, "
                "created_at, metadata_json, entry_type, evidence_trial_ids "
                "FROM strategies"
            ).fetchall()
        except sqlite3.Error as exc:
            return {}, [f"cannot read strategies table: {exc}"]
        for raw in raw_rows:
            try:
                metadata = json.loads(raw["metadata_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
                warnings.append(f"strategy {raw['id']} has invalid metadata_json")
            if not isinstance(metadata, dict):
                metadata = {}
            if raw["id"] not in expected_ids and metadata.get("seed_campaign") != campaign:
                continue
            try:
                evidence_trial_ids = json.loads(raw["evidence_trial_ids"] or "[]")
            except (TypeError, json.JSONDecodeError):
                evidence_trial_ids = []
                warnings.append(f"strategy {raw['id']} has invalid evidence_trial_ids")
            rows[str(raw["id"])] = {
                "id": raw["id"],
                "description": raw["description"],
                "insight": raw["insight"],
                "source_trial_id": raw["source_trial_id"],
                "species": raw["species"],
                "created_at": raw["created_at"],
                "metadata": metadata,
                "entry_type": raw["entry_type"] or "raw",
                "evidence_trial_ids": [
                    int(tid)
                    for tid in evidence_trial_ids
                    if isinstance(tid, int) or str(tid).isdigit()
                ],
            }
    finally:
        conn.close()
    return rows, warnings


def _source_handoff_refs(source_handoff: str) -> list[str]:
    refs: list[str] = []
    for part in source_handoff.replace(",", " / ").split("/"):
        ref = " ".join(part.split())
        if ref:
            refs.append(ref)
    return refs or [source_handoff]


def _journal_entries_by_trial(journal_dir: Path) -> tuple[dict[int, Any], list[str]]:
    warnings: list[str] = []
    if not journal_dir.exists():
        return {}, [f"journal directory does not exist: {journal_dir}"]
    try:
        journal = ExperimentJournal(journal_dir=journal_dir)
        entries = (
            journal.entries_with_supersessions()
            if hasattr(journal, "entries_with_supersessions")
            else journal.all_entries()
        )
    except Exception as exc:  # noqa: BLE001
        return {}, [f"cannot load experiment journal: {exc}"]
    by_trial: dict[int, Any] = {}
    for entry in entries:
        try:
            by_trial[int(entry.trial_id)] = entry
        except (TypeError, ValueError):
            warnings.append(f"journal entry has non-integer trial_id: {entry!r}")
    return by_trial, warnings


def _entry_is_clean_evidence(entry: Any) -> bool:
    if entry is None:
        return False
    if getattr(entry, "bug_corrupted_by", ""):
        return False
    if getattr(entry, "outcome_status", "ok") != "ok":
        return False
    if getattr(entry, "keep_revert_decision", "") == "excluded":
        return False
    eval_details = getattr(entry, "eval_details", {}) or {}
    if isinstance(eval_details, dict) and eval_details.get("learning_exclusion"):
        return False
    return True


def _entry_summary(entry: Any) -> dict[str, Any]:
    if entry is None:
        return {"present": False}
    return {
        "present": True,
        "trial_id": int(entry.trial_id),
        "species": getattr(entry, "species", ""),
        "action_type": getattr(entry, "action_type", ""),
        "quality": getattr(entry, "quality", None),
        "reliability": getattr(entry, "reliability", None),
        "pareto_status": getattr(entry, "pareto_status", ""),
        "outcome_status": getattr(entry, "outcome_status", "ok"),
        "git_tag": getattr(entry, "git_tag", ""),
        "bug_corrupted_by": getattr(entry, "bug_corrupted_by", ""),
        "learning_exclusion": (
            (getattr(entry, "eval_details", {}) or {}).get("learning_exclusion")
            if isinstance(getattr(entry, "eval_details", {}) or {}, dict)
            else None
        ),
        "clean_evidence": _entry_is_clean_evidence(entry),
    }


def _row_report(
    row: SeedRow,
    *,
    strategy_row: dict[str, Any] | None,
    journal_by_trial: dict[int, Any],
) -> dict[str, Any]:
    declared_evidence_ids = list(row.evidence_trial_ids)
    strategy_evidence_ids = (
        list(strategy_row.get("evidence_trial_ids", [])) if strategy_row else []
    )
    evidence_ids = declared_evidence_ids or strategy_evidence_ids
    evidence_entries = [
        _entry_summary(journal_by_trial.get(int(trial_id)))
        for trial_id in evidence_ids
    ]
    declared_clean_count = sum(
        1
        for trial_id in declared_evidence_ids
        if _entry_is_clean_evidence(journal_by_trial.get(int(trial_id)))
    )
    applied = strategy_row is not None
    closure_candidate = (
        applied
        and bool(declared_evidence_ids)
        and declared_clean_count == len(declared_evidence_ids)
    )
    if not applied:
        closure_status = "pending_seed"
    elif not declared_evidence_ids:
        closure_status = "memory_only_not_closure"
    elif closure_candidate:
        closure_status = "candidate_review_required"
    else:
        closure_status = "evidence_incomplete_or_unclean"

    metadata = strategy_row.get("metadata", {}) if strategy_row else {}
    return {
        "strategy_id": row.entry_id,
        "slug": row.slug,
        "tranche": row.tranche,
        "species": row.species,
        "entry_type": row.entry_type,
        "source_handoff": row.source_handoff,
        "source_handoff_refs": _source_handoff_refs(row.source_handoff),
        "seed_status": "applied" if applied else "pending_seed",
        "metadata_campaign": metadata.get("seed_campaign"),
        "metadata_source_handoff": metadata.get("source_handoff"),
        "declared_evidence_trial_ids": declared_evidence_ids,
        "strategy_evidence_trial_ids": strategy_evidence_ids,
        "evidence_entries": evidence_entries,
        "clean_declared_evidence_count": declared_clean_count,
        "closure_candidate": closure_candidate,
        "closure_status": closure_status,
        "recommendation": (
            "review handoff for possible archival; do not auto-write"
            if closure_candidate
            else "do not archive from memory evidence alone"
        ),
    }


def build_handoff_closure_candidate_report(
    *,
    seed_file: Path = DEFAULT_SEED_FILE,
    strategy_path: Path = DEFAULT_STRATEGY_PATH,
    journal_dir: Path = DEFAULT_JOURNAL_DIR,
    campaign: str = DEFAULT_CAMPAIGN,
) -> dict[str, Any]:
    seed_rows = load_seed_rows(seed_file)
    expected_ids = {row.entry_id for row in seed_rows}
    strategy_rows, strategy_warnings = _load_campaign_strategy_rows(
        strategy_path=strategy_path,
        campaign=campaign,
        expected_ids=expected_ids,
    )
    journal_by_trial, journal_warnings = _journal_entries_by_trial(journal_dir)
    row_reports = [
        _row_report(
            row,
            strategy_row=strategy_rows.get(row.entry_id),
            journal_by_trial=journal_by_trial,
        )
        for row in seed_rows
    ]

    handoff_index: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in row_reports:
        for ref in item["source_handoff_refs"]:
            grouped[ref].append(item)
    for ref, items in sorted(grouped.items()):
        handoff_index[ref] = {
            "row_count": len(items),
            "applied_count": sum(1 for item in items if item["seed_status"] == "applied"),
            "pending_seed_count": sum(
                1 for item in items if item["seed_status"] == "pending_seed"
            ),
            "closure_candidate_count": sum(
                1 for item in items if item["closure_candidate"]
            ),
            "statuses": sorted({item["closure_status"] for item in items}),
            "strategy_ids": [item["strategy_id"] for item in items],
        }

    warnings = strategy_warnings + journal_warnings
    return {
        "ok": not warnings,
        "campaign": campaign,
        "governance_mode": "suggest_only",
        "handoff_writes_permitted": False,
        "row_count": len(row_reports),
        "applied_count": sum(1 for item in row_reports if item["seed_status"] == "applied"),
        "pending_seed_count": sum(
            1 for item in row_reports if item["seed_status"] == "pending_seed"
        ),
        "closure_candidate_count": sum(
            1 for item in row_reports if item["closure_candidate"]
        ),
        "memory_only_count": sum(
            1 for item in row_reports if item["closure_status"] == "memory_only_not_closure"
        ),
        "warnings": warnings,
        "handoffs": handoff_index,
        "rows": row_reports,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AutoPilot Handoff Closure Candidate Report",
        "",
        f"- Campaign: `{report['campaign']}`",
        f"- Governance mode: `{report['governance_mode']}`",
        f"- Handoff writes permitted: {str(report['handoff_writes_permitted']).lower()}",
        (
            "- Rows: "
            f"total={report['row_count']}, applied={report['applied_count']}, "
            f"pending_seed={report['pending_seed_count']}, "
            f"memory_only={report['memory_only_count']}, "
            f"closure_candidates={report['closure_candidate_count']}"
        ),
    ]
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    candidates = [row for row in report["rows"] if row["closure_candidate"]]
    if candidates:
        lines.extend(["", "## Closure Candidates", ""])
        for row in candidates[:50]:
            lines.append(
                f"- `{row['strategy_id']}` -> {row['source_handoff']}: "
                f"{row['recommendation']}"
            )
    else:
        lines.extend(
            [
                "",
                "## Closure Candidates",
                "",
                "- None. Seeded planner memory alone is not handoff closure evidence.",
            ]
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report operator-seeded StrategyStore rows that may support handoff "
            "closure review. This script is read-only and never edits handoffs."
        )
    )
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--strategy-path", type=Path, default=DEFAULT_STRATEGY_PATH)
    parser.add_argument("--journal-dir", type=Path, default=DEFAULT_JOURNAL_DIR)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if report warnings were emitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_handoff_closure_candidate_report(
            seed_file=args.seed_file.expanduser().resolve(),
            strategy_path=args.strategy_path.expanduser().resolve(),
            journal_dir=args.journal_dir.expanduser().resolve(),
            campaign=args.campaign,
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"handoff closure candidate report failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(render_markdown(report))
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
