#!/usr/bin/env python3
"""Append a reviewed intake-triage verdict row for F3 dataset capture."""

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

from scripts.datasets._common import file_sha256, stable_hash, utc_now


DEFAULT_INTAKE = Path("/mnt/raid0/llm/epyc-root/research/intake_index.yaml")
DEFAULT_OUTPUT = Path("orchestration/datasets/intake_triage_reviewed.jsonl")
RECORDER_VERSION = "intake_triage_verdict_recorder.v1"
SCHEMA_VERSION = "reviewed_intake_triage_verdict.v1"
QUARANTINE_POLICY_VERSION = "f5-quarantine-v1"
OUTPUT_CONTRACT_VERSION = "intake-triage-reviewed-label.v1"
LABEL_SOURCES = ("operator", "shadow_job")


def _load_intake(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text()) or []
    if not isinstance(data, list):
        raise ValueError(f"intake index must be a YAML list: {path}")
    return [row for row in data if isinstance(row, dict)]


def _find_intake(rows: list[dict[str, Any]], intake_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("id") or "") == intake_id:
            return row
    raise ValueError(f"intake id not found: {intake_id}")


def source_features(row: dict[str, Any]) -> dict[str, Any]:
    """Extract non-instructional classification features from an intake row."""
    return {
        "title": row.get("title") or "",
        "url": row.get("url") or "",
        "source_type": row.get("source_type") or "",
        "categories": row.get("categories") or [],
        "novelty": row.get("novelty") or "",
        "relevance": row.get("relevance") or "",
        "discovered_via": row.get("discovered_via") or "",
        "ingested_date": row.get("ingested_date") or "",
    }


def build_record(
    row: dict[str, Any],
    *,
    source_index_path: Path,
    verdict: str,
    destination_handoff: str = "",
    destination_index: str = "",
    reviewer: str = "operator",
    label_source: str = "operator",
    notes: str = "",
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    if label_source not in LABEL_SOURCES:
        raise ValueError(f"label_source must be one of {LABEL_SOURCES}: {label_source}")
    intake_id = str(row.get("id") or "")
    if not intake_id:
        raise ValueError("intake row is missing id")
    if not verdict:
        raise ValueError("verdict is required")
    reviewed_at = reviewed_at or utc_now()
    features = source_features(row)
    review_payload = {
        "intake_id": intake_id,
        "verdict": verdict,
        "destination_handoff": destination_handoff,
        "destination_index": destination_index,
        "reviewer": reviewer,
        "label_source": label_source,
        "reviewed_at": reviewed_at,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "recorder_version": RECORDER_VERSION,
        "review_id": stable_hash(review_payload),
        "source_index_path": str(source_index_path),
        "source_index_sha256": file_sha256(source_index_path),
        "intake_id": intake_id,
        **features,
        "source_features": features,
        "features_text": json.dumps(features, sort_keys=True),
        "verdict": verdict,
        "destination_handoff": destination_handoff,
        "destination_index": destination_index,
        "reviewer": reviewer,
        "label_source": label_source,
        "notes": notes,
        "reviewed_at": reviewed_at,
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "source_text_excluded": True,
    }


def append_record(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    intake_path = Path(args.intake).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    rows = _load_intake(intake_path)
    intake_row = _find_intake(rows, args.intake_id)
    record = build_record(
        intake_row,
        source_index_path=intake_path,
        verdict=args.verdict,
        destination_handoff=args.destination_handoff,
        destination_index=args.destination_index,
        reviewer=args.reviewer,
        label_source=args.label_source,
        notes=args.notes,
        reviewed_at=args.reviewed_at or None,
    )
    if not args.dry_run:
        append_record(output_path, record)
    return {
        "dry_run": args.dry_run,
        "output": str(output_path),
        "intake_id": record["intake_id"],
        "review_id": record["review_id"],
        "record": record,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", default=str(DEFAULT_INTAKE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--intake-id", required=True)
    parser.add_argument("--verdict", required=True)
    parser.add_argument("--destination-handoff", default="")
    parser.add_argument("--destination-index", default="")
    parser.add_argument("--reviewer", default=os.environ.get("USER", "operator"))
    parser.add_argument("--label-source", choices=LABEL_SOURCES, default="operator")
    parser.add_argument("--notes", default="")
    parser.add_argument("--reviewed-at", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
