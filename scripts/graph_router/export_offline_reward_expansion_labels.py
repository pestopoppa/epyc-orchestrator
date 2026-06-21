#!/usr/bin/env python3
"""Export prompt-free labels for offline verifier expansion candidates.

This bridges the A9 sparse-action expansion plan to verifier-data rebuilding.
It reuses an already adopted offline oracle's identity and threshold, but it
does not weaken the primary adoption exporter row-count gate: expansion rows
must instead match the prompt-free expansion candidate manifest exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.build_offline_reward_oracle_manifest import load_json
from scripts.graph_router.export_offline_reward_oracle_labels import (
    LABEL_SCHEMA_VERSION,
    LabelExportError,
    _label_row,
    _manifest_oracle,
    load_jsonl,
)


SUMMARY_SCHEMA_VERSION = "offline_reward_expansion_label_export_summary.v1"
EXPANSION_TARGET_SOURCE = "verifier_sparse_action_expansion"
PRIVATE_FIELDS = {"answer", "expected", "prompt", "reference", "response"}


def _candidate_key(row: dict[str, Any]) -> tuple[str, int, str]:
    source_path = str(row.get("source_path") or "")
    offset = row.get("source_record_offset")
    role_key = str(row.get("role_key") or "")
    if not source_path or not isinstance(offset, int) or not role_key:
        raise LabelExportError("candidate/source rows require source_path, offset, and role_key")
    return source_path, offset, role_key


def _candidate_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise LabelExportError(f"{path}:{row_number}: missing candidate_id")
        if candidate_id in out:
            raise LabelExportError(f"{path}:{row_number}: duplicate candidate_id={candidate_id!r}")
        _candidate_key(row)
        out[candidate_id] = row
    return out


def _assert_prompt_free(rows: Iterable[dict[str, Any]]) -> None:
    for row_number, row in enumerate(rows, start=1):
        present = sorted(PRIVATE_FIELDS & set(row))
        if present:
            raise LabelExportError(
                f"output row {row_number}: private fields present: {', '.join(present)}"
            )


def export_expansion_labels(
    *,
    manifest_path: Path,
    scored_rows_path: Path,
    candidates_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    oracle = _manifest_oracle(manifest, manifest_path=manifest_path)
    threshold = float(oracle["oracle_threshold"])
    model_id = str(oracle["model_id"])
    score_source = str(oracle["oracle_score_source"])

    candidates = _candidate_map(candidates_path)
    scored_rows = load_jsonl(scored_rows_path)
    labels: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row_number, row in enumerate(scored_rows, start=1):
        item_id = str(row.get("item_id") or "")
        if item_id not in candidates:
            raise LabelExportError(f"{scored_rows_path}:{row_number}: {item_id!r} not in candidates")
        candidate = candidates[item_id]
        if _candidate_key(row) != _candidate_key(candidate):
            raise LabelExportError(
                f"{scored_rows_path}:{row_number}: source/role key does not match candidate"
            )
        row_for_label = dict(row)
        row_for_label["target_source"] = EXPANSION_TARGET_SOURCE
        label = _label_row(
            row_for_label,
            threshold=threshold,
            model_id=model_id,
            score_source=score_source,
            row_number=row_number,
        )
        label["expansion_candidate_id"] = item_id
        label["expansion_candidate_schema_version"] = candidate.get("schema_version")
        if item_id in seen:
            raise LabelExportError(f"duplicate scored item_id={item_id!r}")
        seen.add(item_id)
        labels.append(label)

    missing = sorted(set(candidates) - seen)
    if missing:
        raise LabelExportError(f"missing scored candidate rows: {len(missing)}")
    _assert_prompt_free(labels)

    counts = Counter(int(row["oracle_binary_label"]) for row in labels)
    action_counts = Counter(str(row.get("role_key") or "") for row in labels)
    compared = [row for row in labels if "oracle_matches_target" in row]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "manifest_json": str(manifest_path),
        "scored_rows_jsonl": str(scored_rows_path),
        "candidates_jsonl": str(candidates_path),
        "rows": len(labels),
        "oracle": {
            "model_id": model_id,
            "oracle_score_source": score_source,
            "oracle_threshold": threshold,
        },
        "oracle_positive": int(counts[1]),
        "oracle_negative": int(counts[0]),
        "target_source": EXPANSION_TARGET_SOURCE,
        "target_compared": len(compared),
        "target_agreement": (
            sum(bool(row["oracle_matches_target"]) for row in compared) / len(compared)
            if compared
            else None
        ),
        "role_counts": dict(sorted(action_counts.items())),
        "privacy": {
            "private_fields_excluded": sorted(PRIVATE_FIELDS),
            "commits_private_rows": False,
        },
    }
    return labels, summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Offline Reward Expansion Label Export",
        "",
        f"- Manifest: `{summary['manifest_json']}`",
        f"- Candidates: `{summary['candidates_jsonl']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Oracle: `{summary['oracle']['model_id']}`",
        f"- Score source: `{summary['oracle']['oracle_score_source']}`",
        f"- Threshold: `{summary['oracle']['oracle_threshold']}`",
        f"- Oracle positives / negatives: `{summary['oracle_positive']}` / `{summary['oracle_negative']}`",
        f"- Role counts: `{summary['role_counts']}`",
        "",
        "The exported label rows exclude prompt, reference, response, expected, and answer text.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export prompt-free labels for verifier expansion candidates",
    )
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--scored-rows-jsonl", required=True, type=Path)
    parser.add_argument("--candidates-jsonl", required=True, type=Path)
    parser.add_argument("--labels-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--summary-md", type=Path)
    args = parser.parse_args(argv)

    try:
        labels, summary = export_expansion_labels(
            manifest_path=args.manifest_json,
            scored_rows_path=args.scored_rows_jsonl,
            candidates_path=args.candidates_jsonl,
        )
    except (LabelExportError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_jsonl(args.labels_jsonl, labels)
    write_json(args.summary_json, summary)
    if args.summary_md:
        write_markdown(args.summary_md, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
