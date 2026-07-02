#!/usr/bin/env python3
"""Prepare prompt-free intake-triage rows for reviewed-label collection."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datasets._common import load_jsonl, stable_hash, utc_now, write_jsonl, write_manifest
from scripts.datasets.record_intake_triage_verdict import (
    DEFAULT_INTAKE,
    DEFAULT_OUTPUT as DEFAULT_REVIEWED_LABELS,
    LABEL_SOURCES,
    source_features,
)


DEFAULT_OUTPUT = Path("orchestration/datasets/intake_triage_review_queue.jsonl")
DEFAULT_MANIFEST = Path("orchestration/datasets/intake_triage_review_queue.manifest.json")
DEFAULT_BATCH_TEMPLATE = Path("orchestration/datasets/intake_triage_review_batch_template.jsonl")
BUILDER_VERSION = "intake_triage_review_queue_builder.v1"
DEFAULT_TRUSTED_REVIEWED_LABEL_SOURCES = ("operator",)


def _load_intake(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text()) or []
    if not isinstance(data, list):
        raise ValueError(f"intake index must be a YAML list: {path}")
    return [row for row in data if isinstance(row, dict)]


def _effective_label_sources(raw_sources: list[str] | tuple[str, ...] | None) -> set[str]:
    sources = {source for source in (raw_sources or DEFAULT_TRUSTED_REVIEWED_LABEL_SOURCES) if source}
    return sources or set(DEFAULT_TRUSTED_REVIEWED_LABEL_SOURCES)


def _label_source(row: dict[str, Any]) -> str:
    return str(row.get("label_source") or "operator")


def _latest_reviewed_ids(
    path: Path | None,
    *,
    trusted_label_sources: set[str] | None = None,
) -> set[str]:
    if path is None or not path.exists():
        return set()
    trusted_sources = trusted_label_sources or set(DEFAULT_TRUSTED_REVIEWED_LABEL_SOURCES)
    reviewed: set[str] = set()
    for row in load_jsonl(path):
        if _label_source(row) not in trusted_sources:
            continue
        intake_id = str(row.get("intake_id") or "")
        if intake_id:
            reviewed.add(intake_id)
    return reviewed


def _destination(row: dict[str, Any], kind: str) -> str:
    refs = row.get("cross_references") or {}
    if not isinstance(refs, dict):
        return ""
    key = "handoffs" if kind == "handoff" else "indices"
    values = refs.get(key)
    if isinstance(values, list) and values:
        return str(values[0])
    return ""


def _record_command(
    *,
    intake_path: Path,
    reviewed_labels_path: Path,
    intake_id: str,
    verdict: str,
    destination_handoff: str,
    destination_index: str,
    label_source: str = "operator",
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "scripts/datasets/record_intake_triage_verdict.py",
        "--intake",
        str(intake_path),
        "--output",
        str(reviewed_labels_path),
        "--intake-id",
        intake_id,
        "--verdict",
        verdict,
    ]
    if destination_handoff:
        parts.extend(["--destination-handoff", destination_handoff])
    if destination_index:
        parts.extend(["--destination-index", destination_index])
    parts.extend(["--label-source", label_source])
    return " ".join(shlex.quote(part) for part in parts)


def build_review_item(
    row: dict[str, Any],
    *,
    intake_path: Path,
    reviewed_labels_path: Path,
    label_source: str = "operator",
) -> dict[str, Any]:
    if label_source not in LABEL_SOURCES:
        raise ValueError(f"label_source must be one of {LABEL_SOURCES}: {label_source}")
    features = source_features(row)
    intake_id = str(row.get("id") or stable_hash(features))
    verdict = str(row.get("verdict") or "")
    destination_handoff = _destination(row, "handoff")
    destination_index = _destination(row, "index")
    return {
        "schema_version": "intake_triage_review_queue.v1",
        "builder_version": BUILDER_VERSION,
        "review_item_id": stable_hash(
            {"source": str(intake_path), "reviewed_labels": str(reviewed_labels_path), "intake_id": intake_id}
        ),
        "source_index_path": str(intake_path),
        "intake_id": intake_id,
        **features,
        "current_verdict": verdict,
        "destination_handoff": destination_handoff,
        "destination_index": destination_index,
        "label_source": label_source,
        "features_text": json.dumps(features, sort_keys=True),
        "record_command": _record_command(
            intake_path=intake_path,
            reviewed_labels_path=reviewed_labels_path,
            intake_id=intake_id,
            verdict=verdict or "<verdict>",
            destination_handoff=destination_handoff,
            destination_index=destination_index,
            label_source=label_source,
        ),
        "source_text_excluded": True,
    }


def build_queue(
    *,
    intake_path: Path,
    reviewed_labels_path: Path | None,
    include_verdicts: set[str] | None = None,
    exclude_verdicts: set[str] | None = None,
    label_source: str = "operator",
    trusted_reviewed_label_sources: set[str] | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_intake(intake_path)
    trusted_sources = trusted_reviewed_label_sources or set(DEFAULT_TRUSTED_REVIEWED_LABEL_SOURCES)
    reviewed_ids = _latest_reviewed_ids(
        reviewed_labels_path,
        trusted_label_sources=trusted_sources,
    )
    output_reviewed_path = reviewed_labels_path or DEFAULT_REVIEWED_LABELS
    queue: list[dict[str, Any]] = []
    skipped_reviewed = 0
    skipped_verdict_filter = 0
    for row in rows:
        intake_id = str(row.get("id") or "")
        if intake_id and intake_id in reviewed_ids:
            skipped_reviewed += 1
            continue
        verdict = str(row.get("verdict") or "")
        if include_verdicts is not None and verdict not in include_verdicts:
            skipped_verdict_filter += 1
            continue
        if exclude_verdicts is not None and verdict in exclude_verdicts:
            skipped_verdict_filter += 1
            continue
        queue.append(
            build_review_item(
                row,
                intake_path=intake_path,
                reviewed_labels_path=output_reviewed_path,
                label_source=label_source,
            )
        )
        if limit is not None and len(queue) >= limit:
            break
    counts = {
        "source_rows": len(rows),
        "reviewed_labels_loaded": len(reviewed_ids),
        "trusted_reviewed_label_sources": sorted(trusted_sources),
        "skipped_already_reviewed": skipped_reviewed,
        "skipped_verdict_filter": skipped_verdict_filter,
        "emitted": len(queue),
    }
    return queue, counts


def build_batch_template(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    template: list[dict[str, Any]] = []
    for row in rows:
        template.append(
            {
                "intake_id": row["intake_id"],
                "title": row.get("title") or "",
                "url": row.get("url") or "",
                "source_type": row.get("source_type") or "",
                "categories": row.get("categories") or [],
                "suggested_verdict": row.get("current_verdict") or "",
                "verdict": "",
                "destination_handoff": row.get("destination_handoff") or "",
                "destination_index": row.get("destination_index") or "",
                "label_source": row.get("label_source") or "operator",
                "notes": "",
                "source_text_excluded": True,
            }
        )
    return template


def run(args: argparse.Namespace) -> dict[str, Any]:
    intake_path = Path(args.intake).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    reviewed_labels_path = (
        Path(args.reviewed_labels).expanduser() if args.reviewed_labels else DEFAULT_REVIEWED_LABELS
    )
    limit = args.limit if args.limit and args.limit > 0 else None
    include_verdicts = set(args.include_verdict) if args.include_verdict else None
    exclude_verdicts = set(args.exclude_verdict) if args.exclude_verdict else None
    trusted_reviewed_label_sources = _effective_label_sources(
        getattr(args, "trusted_reviewed_label_source", [])
    )
    rows, counts = build_queue(
        intake_path=intake_path,
        reviewed_labels_path=reviewed_labels_path,
        include_verdicts=include_verdicts,
        exclude_verdicts=exclude_verdicts,
        label_source=args.label_source,
        trusted_reviewed_label_sources=trusted_reviewed_label_sources,
        limit=limit,
    )
    written = write_jsonl(output_path, rows)
    batch_template_path = Path(args.batch_template).expanduser() if args.batch_template else None
    batch_template_written = 0
    if batch_template_path is not None:
        batch_template_written = write_jsonl(batch_template_path, build_batch_template(rows))
    generated_at = utc_now()
    write_manifest(
        manifest_path,
        builder=BUILDER_VERSION,
        generated_at=generated_at,
        source_path=intake_path,
        output_path=output_path,
        counts={**counts, "written": written},
        options={
            "reviewed_labels": str(reviewed_labels_path),
            "batch_template": str(batch_template_path) if batch_template_path else "",
            "include_verdict": sorted(include_verdicts) if include_verdicts is not None else [],
            "exclude_verdict": sorted(exclude_verdicts) if exclude_verdicts is not None else [],
            "label_source": args.label_source,
            "trusted_reviewed_label_sources": sorted(trusted_reviewed_label_sources),
            "limit": limit,
        },
    )
    return {
        "output": str(output_path),
        "manifest": str(manifest_path),
        "batch_template": str(batch_template_path) if batch_template_path else "",
        "batch_template_written": batch_template_written,
        "counts": counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", default=str(DEFAULT_INTAKE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--batch-template",
        nargs="?",
        const=str(DEFAULT_BATCH_TEMPLATE),
        default="",
        help=(
            "Optionally write operator-fillable JSONL decisions for "
            "apply_intake_triage_review_batch.py; defaults to the standard "
            "template path when no value is supplied."
        ),
    )
    parser.add_argument("--reviewed-labels", default=str(DEFAULT_REVIEWED_LABELS))
    parser.add_argument("--include-verdict", action="append", default=[])
    parser.add_argument("--exclude-verdict", action="append", default=[])
    parser.add_argument("--label-source", choices=LABEL_SOURCES, default="operator")
    parser.add_argument(
        "--trusted-reviewed-label-source",
        action="append",
        default=[],
        help="Reviewed label source that suppresses queue items; defaults to operator.",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
