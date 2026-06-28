#!/usr/bin/env python3
"""Report F3 intake-triage reviewed-label readiness without creating labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datasets._common import load_jsonl, utc_now


DEFAULT_QUEUE = Path("orchestration/datasets/intake_triage_review_queue.jsonl")
DEFAULT_REVIEWED_LABELS = Path("orchestration/datasets/intake_triage_reviewed.jsonl")
DEFAULT_REPORT = Path("orchestration/reports/intake_triage_review_status.json")
REPORT_VERSION = "intake_triage_review_status.v1"


def _load_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _intake_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("intake_id") or "") for row in rows if row.get("intake_id")}


def summarize(
    *,
    queue_path: Path,
    reviewed_labels_path: Path,
    min_reviewed_labels: int,
) -> dict[str, Any]:
    queue_rows = _load_jsonl_if_present(queue_path)
    reviewed_rows = _load_jsonl_if_present(reviewed_labels_path)
    queue_ids = _intake_ids(queue_rows)
    reviewed_ids = _intake_ids(reviewed_rows)
    reviewed_queue_ids = queue_ids & reviewed_ids
    remaining_queue_ids = queue_ids - reviewed_ids
    labels_needed = max(0, min_reviewed_labels - len(reviewed_ids))

    if len(reviewed_ids) >= min_reviewed_labels:
        status = "ready_for_baseline"
    elif not queue_ids:
        status = "missing_review_queue"
    elif len(reviewed_ids) + len(remaining_queue_ids) < min_reviewed_labels:
        status = "queue_exhausted_below_gate"
    else:
        status = "needs_reviewed_labels"

    return {
        "schema_version": REPORT_VERSION,
        "generated_at": utc_now(),
        "queue_path": str(queue_path),
        "reviewed_labels_path": str(reviewed_labels_path),
        "min_reviewed_labels": min_reviewed_labels,
        "status": status,
        "queue_rows": len(queue_rows),
        "queue_unique_intake_ids": len(queue_ids),
        "reviewed_rows": len(reviewed_rows),
        "reviewed_unique_intake_ids": len(reviewed_ids),
        "reviewed_queue_items": len(reviewed_queue_ids),
        "remaining_queue_items": len(remaining_queue_ids),
        "labels_needed": labels_needed,
        "ready_for_baseline": status == "ready_for_baseline",
        "privacy": {
            "raw_text_in_report": False,
            "reported_fields": "aggregate counts only",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    queue_path = Path(args.queue).expanduser()
    reviewed_labels_path = Path(args.reviewed_labels).expanduser()
    report = summarize(
        queue_path=queue_path,
        reviewed_labels_path=reviewed_labels_path,
        min_reviewed_labels=args.min_reviewed_labels,
    )
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--reviewed-labels", default=str(DEFAULT_REVIEWED_LABELS))
    parser.add_argument("--min-reviewed-labels", type=int, default=100)
    parser.add_argument(
        "--report",
        nargs="?",
        const=str(DEFAULT_REPORT),
        default="",
        help="Optionally write a JSON report; defaults to the standard report path when no value is supplied.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
