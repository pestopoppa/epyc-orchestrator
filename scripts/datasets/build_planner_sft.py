#!/usr/bin/env python3
"""Build a weakly labeled planner-SFT JSONL corpus from planner_archive.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.datasets._common import load_jsonl, stable_hash, utc_now, write_jsonl, write_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_ARCHIVE = _REPO_ROOT / "logs/planner_archive.jsonl"
DEFAULT_OUTPUT = Path("orchestration/datasets/planner_sft.jsonl")
DEFAULT_MANIFEST = Path("orchestration/datasets/planner_sft.manifest.json")
BUILDER_VERSION = "planner_sft_builder.v1"


def _action_fingerprint(row: dict[str, Any]) -> dict[str, Any]:
    action = row.get("action")
    if isinstance(action, dict):
        return action
    action_json = row.get("action_json")
    if isinstance(action_json, dict):
        return action_json
    action_type = row.get("action_type") or row.get("draft_action_type")
    return {"type": action_type} if action_type else {}


def _label_for(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("bug_corrupted_by"):
        return "contaminated", str(row.get("bug_corrupted_by"))
    subtype = str(row.get("subtype") or row.get("status") or "").lower()
    if subtype and subtype not in {"success", "ok", "passed"}:
        return "failed", subtype
    critique = str(row.get("critique_decision") or row.get("critic_decision") or "").lower()
    if critique in {"approve", "approved", "accept", "accepted"}:
        return "critic_approved", ""
    if critique in {"reject", "rejected"}:
        return "rejected", "critic_rejected"
    if row.get("type") == "planner_coordinator" and row.get("action_type"):
        return "critic_approved" if row.get("degraded") is False else "unlabeled", ""
    if row.get("result_preview") or row.get("events"):
        return "unlabeled", "no_explicit_action_or_outcome"
    return "failed", "empty_or_missing_result"


def build_example(row: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    action = _action_fingerprint(row)
    label, exclude_reason = _label_for(row)
    if label in {"critic_approved", "confirmed"}:
        training_exclude = ""
    elif label in {"failed", "rejected"}:
        training_exclude = "negative_example_only"
    elif label == "contaminated":
        training_exclude = exclude_reason or "contaminated"
    else:
        training_exclude = exclude_reason or "unlabeled"
    source_line = row.get("_source_line")
    prompt_hash = row.get("prompt_sha256_16") or row.get("prompt_hash")
    example_id = stable_hash(
        {
            "source": str(source_path),
            "line": source_line,
            "prompt": prompt_hash,
            "action": action,
        }
    )
    return {
        "schema_version": "planner_sft_example.v1",
        "builder_version": BUILDER_VERSION,
        "example_id": example_id,
        "source_archive_path": str(source_path),
        "source_archive_line": source_line,
        "source_timestamp": row.get("ts_iso") or row.get("timestamp") or row.get("ts"),
        "prompt_sha256_16": prompt_hash,
        "prompt_chars": row.get("prompt_chars"),
        "provider": row.get("provider") or row.get("draft_provider") or "unknown",
        "role": row.get("role") or row.get("model") or "planner",
        "subtype": row.get("subtype") or row.get("status") or row.get("type") or "",
        "action_type": action.get("type"),
        "action_json": action,
        "trial_id": row.get("trial_id"),
        "candidate_fingerprint": row.get("candidate_fingerprint"),
        "label": label,
        "era_label": "planner_archive_v1",
        "exclude_reason": training_exclude,
        "cost_usd": row.get("total_cost_usd"),
        "duration_ms": row.get("duration_ms"),
        "result_preview": row.get("result_preview", "")[:1200],
    }


def build_dataset(
    *,
    archive_path: Path,
    include_excluded: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(archive_path)
    examples = [build_example(row, source_path=archive_path) for row in rows]
    counts: dict[str, Any] = {"source_rows": len(rows), "emitted": 0, "labels": {}}
    for example in examples:
        counts["labels"][example["label"]] = counts["labels"].get(example["label"], 0) + 1
    if not include_excluded:
        examples = [row for row in examples if row["exclude_reason"] == ""]
    counts["emitted"] = len(examples)
    return examples, counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = Path(args.archive).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    examples, counts = build_dataset(
        archive_path=archive_path,
        include_excluded=args.include_excluded,
    )
    written = write_jsonl(output_path, examples)
    generated_at = utc_now()
    write_manifest(
        manifest_path,
        builder=BUILDER_VERSION,
        generated_at=generated_at,
        source_path=archive_path,
        output_path=output_path,
        counts={**counts, "written": written},
        options={"include_excluded": args.include_excluded},
    )
    return {"output": str(output_path), "manifest": str(manifest_path), "counts": counts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
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
