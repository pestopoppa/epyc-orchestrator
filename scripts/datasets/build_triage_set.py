#!/usr/bin/env python3
"""Build an intake-triage JSONL corpus from the root research intake index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.datasets._common import stable_hash, utc_now, write_jsonl, write_manifest


DEFAULT_INTAKE = Path("/mnt/raid0/llm/epyc-root/research/intake_index.yaml")
DEFAULT_OUTPUT = Path("orchestration/datasets/intake_triage.jsonl")
DEFAULT_MANIFEST = Path("orchestration/datasets/intake_triage.manifest.json")
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


def build_example(row: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    verdict = row.get("verdict")
    exclude_reason = "" if verdict else "missing_verdict"
    feature_payload = {
        "title": row.get("title") or "",
        "source_type": row.get("source_type") or "",
        "categories": row.get("categories") or [],
        "novelty": row.get("novelty") or "",
        "relevance": row.get("relevance") or "",
        "discovered_via": row.get("discovered_via") or "",
    }
    intake_id = str(row.get("id") or stable_hash(feature_payload))
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
        "destination_handoff": _destination(row),
        "quarantine_policy_version": QUARANTINE_POLICY_VERSION,
        "label_source": "research-intake",
        "exclude_reason": exclude_reason,
        "features_text": json.dumps(feature_payload, sort_keys=True),
    }


def build_dataset(
    *,
    intake_path: Path,
    include_excluded: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_intake(intake_path)
    examples = [build_example(row, source_path=intake_path) for row in rows]
    counts: dict[str, Any] = {"source_rows": len(rows), "emitted": 0, "verdicts": {}}
    for example in examples:
        verdict = example["verdict"] or "<missing>"
        counts["verdicts"][verdict] = counts["verdicts"].get(verdict, 0) + 1
    if not include_excluded:
        examples = [row for row in examples if row["exclude_reason"] == ""]
    counts["emitted"] = len(examples)
    return examples, counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    intake_path = Path(args.intake).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    examples, counts = build_dataset(
        intake_path=intake_path,
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
        options={"include_excluded": args.include_excluded},
    )
    return {"output": str(output_path), "manifest": str(manifest_path), "counts": counts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", default=str(DEFAULT_INTAKE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--include-excluded", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
