#!/usr/bin/env python3
"""Build an intake-triage JSONL corpus from the root research intake index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datasets._common import load_jsonl, stable_hash, utc_now, write_jsonl, write_manifest


DEFAULT_INTAKE = Path("/mnt/raid0/llm/epyc-root/research/intake_index.yaml")
DEFAULT_OUTPUT = Path("orchestration/datasets/intake_triage.jsonl")
DEFAULT_MANIFEST = Path("orchestration/datasets/intake_triage.manifest.json")
DEFAULT_REVIEWED_LABELS = Path("orchestration/datasets/intake_triage_reviewed.jsonl")
BUILDER_VERSION = "intake_triage_builder.v1"
QUARANTINE_POLICY_VERSION = "f5-quarantine-v1"


def _load_intake(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text()) or []
    if not isinstance(data, list):
        raise ValueError(f"intake index must be a YAML list: {path}")
    return [row for row in data if isinstance(row, dict)]


def _destination(row: dict[str, Any]) -> str:
    refs = row.get("cross_references") or {}
    if isinstance(refs, dict):
        for key in ("handoffs", "indices", "chapters"):
            values = refs.get(key)
            if isinstance(values, list) and values:
                return str(values[0])
    return ""


def _reviewed_label_by_intake_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    labels: dict[str, dict[str, Any]] = {}
    for label in load_jsonl(path):
        intake_id = str(label.get("intake_id") or "")
        if not intake_id:
            continue
        current = labels.get(intake_id)
        if current is None or str(label.get("reviewed_at") or "") >= str(
            current.get("reviewed_at") or ""
        ):
            labels[intake_id] = label
    return labels


def build_example(
    row: dict[str, Any],
    *,
    source_path: Path,
    reviewed_label: dict[str, Any] | None = None,
    require_reviewed_label: bool = False,
) -> dict[str, Any]:
    verdict = reviewed_label.get("verdict") if reviewed_label is not None else row.get("verdict")
    exclude_reason = "" if verdict else "missing_verdict"
    if require_reviewed_label and reviewed_label is None:
        exclude_reason = "missing_reviewed_label"
    feature_payload = {
        "title": row.get("title") or "",
        "source_type": row.get("source_type") or "",
        "categories": row.get("categories") or [],
        "novelty": row.get("novelty") or "",
        "relevance": row.get("relevance") or "",
        "discovered_via": row.get("discovered_via") or "",
    }
    intake_id = str(row.get("id") or stable_hash(feature_payload))
    destination_handoff = _destination(row)
    destination_index = ""
    label_source = "research-intake"
    reviewed_at = ""
    output_contract_version = ""
    if reviewed_label is not None:
        destination_handoff = str(reviewed_label.get("destination_handoff") or destination_handoff)
        destination_index = str(reviewed_label.get("destination_index") or "")
        label_source = str(reviewed_label.get("label_source") or "operator")
        reviewed_at = str(reviewed_label.get("reviewed_at") or "")
        output_contract_version = str(reviewed_label.get("output_contract_version") or "")
    return {
        "schema_version": "intake_triage_example.v1",
        "builder_version": BUILDER_VERSION,
        "example_id": stable_hash({"source": str(source_path), "intake_id": intake_id}),
        "source_index_path": str(source_path),
        "intake_id": intake_id,
        "url": row.get("url") or "",
        "source_type": row.get("source_type") or "",
        "title": row.get("title") or "",
        "categories": row.get("categories") or [],
        "novelty": row.get("novelty") or "",
        "relevance": row.get("relevance") or "",
        "verdict": verdict or "",
        "discovered_via": row.get("discovered_via") or "",
        "ingested_date": row.get("ingested_date") or "",
        "destination_handoff": destination_handoff,
        "destination_index": destination_index,
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
        "label_source": label_source,
        "reviewed_at": reviewed_at,
        "output_contract_version": output_contract_version,
        "exclude_reason": exclude_reason,
        "features_text": json.dumps(feature_payload, sort_keys=True),
    }


def build_dataset(
    *,
    intake_path: Path,
    reviewed_labels_path: Path | None = None,
    require_reviewed_labels: bool = False,
    include_excluded: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_intake(intake_path)
    reviewed_labels = _reviewed_label_by_intake_id(reviewed_labels_path)
    examples = [
        build_example(
            row,
            source_path=intake_path,
            reviewed_label=reviewed_labels.get(str(row.get("id") or "")),
            require_reviewed_label=require_reviewed_labels,
        )
        for row in rows
    ]
    counts: dict[str, Any] = {
        "source_rows": len(rows),
        "emitted": 0,
        "reviewed_labels_loaded": len(reviewed_labels),
        "reviewed_labels_used": 0,
        "verdicts": {},
        "label_sources": {},
    }
    for example in examples:
        verdict = example["verdict"] or "<missing>"
        counts["verdicts"][verdict] = counts["verdicts"].get(verdict, 0) + 1
        label_source = example["label_source"] or "<missing>"
        counts["label_sources"][label_source] = counts["label_sources"].get(label_source, 0) + 1
        if example["reviewed_at"]:
            counts["reviewed_labels_used"] += 1
    if not include_excluded:
        examples = [row for row in examples if row["exclude_reason"] == ""]
    counts["emitted"] = len(examples)
    return examples, counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    intake_path = Path(args.intake).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    reviewed_labels_arg = getattr(args, "reviewed_labels", str(DEFAULT_REVIEWED_LABELS))
    reviewed_labels_path = Path(reviewed_labels_arg).expanduser() if reviewed_labels_arg else None
    require_reviewed_labels = bool(getattr(args, "require_reviewed_labels", False))
    examples, counts = build_dataset(
        intake_path=intake_path,
        reviewed_labels_path=reviewed_labels_path,
        require_reviewed_labels=require_reviewed_labels,
        include_excluded=args.include_excluded,
    )
    written = write_jsonl(output_path, examples)
    generated_at = utc_now()
    write_manifest(
        manifest_path,
        builder=BUILDER_VERSION,
        generated_at=generated_at,
        source_path=intake_path,
        output_path=output_path,
        counts={**counts, "written": written},
        options={
            "include_excluded": args.include_excluded,
            "reviewed_labels": reviewed_labels_arg,
            "require_reviewed_labels": require_reviewed_labels,
        },
    )
    return {"output": str(output_path), "manifest": str(manifest_path), "counts": counts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", default=str(DEFAULT_INTAKE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--reviewed-labels", default=str(DEFAULT_REVIEWED_LABELS))
    parser.add_argument("--require-reviewed-labels", action="store_true")
    parser.add_argument("--include-excluded", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
