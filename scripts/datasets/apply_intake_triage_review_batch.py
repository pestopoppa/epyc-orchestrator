#!/usr/bin/env python3
"""Apply reviewed intake-triage verdicts from a batch file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datasets import record_intake_triage_verdict as recorder


def _load_batch(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    else:
        data = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"batch must be a list of objects or JSONL object rows: {path}")
    return data


def _required_text(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"batch row missing required field: {key}")
    return value


def _record_for(
    decision: dict[str, Any],
    intake_rows: list[dict[str, Any]],
    *,
    intake_path: Path,
    default_reviewer: str,
    default_label_source: str,
) -> dict[str, Any]:
    intake_id = _required_text(decision, "intake_id")
    intake_row = recorder._find_intake(intake_rows, intake_id)
    return recorder.build_record(
        intake_row,
        source_index_path=intake_path,
        verdict=_required_text(decision, "verdict"),
        destination_handoff=str(decision.get("destination_handoff") or ""),
        destination_index=str(decision.get("destination_index") or ""),
        reviewer=str(decision.get("reviewer") or default_reviewer),
        label_source=str(decision.get("label_source") or default_label_source),
        notes=str(decision.get("notes") or ""),
        reviewed_at=str(decision.get("reviewed_at") or "") or None,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    intake_path = Path(args.intake).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    batch_path = Path(args.batch).expanduser().resolve()
    decisions = _load_batch(batch_path)
    intake_rows = recorder._load_intake(intake_path)

    records = [
        _record_for(
            decision,
            intake_rows,
            intake_path=intake_path,
            default_reviewer=args.reviewer,
            default_label_source=args.label_source,
        )
        for decision in decisions
    ]
    if args.apply:
        for record in records:
            recorder.append_record(output_path, record)

    return {
        "applied": args.apply,
        "batch": str(batch_path),
        "output": str(output_path),
        "records": len(records),
        "intake_ids": [record["intake_id"] for record in records],
        "review_ids": [record["review_id"] for record in records],
        "source_text_excluded": all(record["source_text_excluded"] for record in records),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--intake", default=str(recorder.DEFAULT_INTAKE))
    parser.add_argument("--output", default=str(recorder.DEFAULT_OUTPUT))
    parser.add_argument("--reviewer", default=os.environ.get("USER", "operator"))
    parser.add_argument("--label-source", choices=recorder.LABEL_SOURCES, default="operator")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="append records to --output; without this flag the batch is validated only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
