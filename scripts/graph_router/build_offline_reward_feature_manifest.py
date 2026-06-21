#!/usr/bin/env python3
"""Build prompt-free feature-input manifests for offline reward labels.

The A9 label export has source file, source record, and role metadata, but not
episodic memory IDs. This bridge validates those join keys against the original
benchmark result files and emits a text-free manifest that a later embedding
extractor can consume deliberately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.export_offline_reward_oracle_labels import (
    LABEL_SCHEMA_VERSION,
    load_jsonl,
)


FEATURE_ROW_SCHEMA_VERSION = "offline_reward_feature_input.v1"
SUMMARY_SCHEMA_VERSION = "offline_reward_feature_manifest_summary.v1"
PRIVATE_FIELDS = {"answer", "expected", "prompt", "reference", "response"}
TASK_TYPES = ["code", "chat", "architecture", "ingest", "general"]


class FeatureManifestError(ValueError):
    """Raised when label rows cannot be joined to source result records."""


def _read_source_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise FeatureManifestError(f"{path}:{line_number}: expected object")
                records.append(value)
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("results", "questions", "records", "rows"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [value]
    raise FeatureManifestError(f"{path}: unsupported JSON source shape")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _task_type_onehot(task_type: str) -> tuple[str, list[float]]:
    task_type_lower = (task_type or "general").lower()
    for index, task_name in enumerate(TASK_TYPES):
        if task_name in task_type_lower:
            vec = [0.0] * len(TASK_TYPES)
            vec[index] = 1.0
            return task_name, vec
    vec = [0.0] * len(TASK_TYPES)
    vec[TASK_TYPES.index("general")] = 1.0
    return "general", vec


def _context_features(record: dict[str, Any]) -> dict[str, Any]:
    prompt = str(record.get("prompt") or "")
    suite = str(record.get("suite") or "general")
    task_type, task_vec = _task_type_onehot(suite)
    has_images = bool(record.get("images") or record.get("image") or record.get("has_images"))
    context_length = len(prompt)
    return {
        "task_type": task_type,
        "task_type_onehot": task_vec,
        "context_length_chars": context_length,
        "has_images": has_images,
        "expected_classifier_feature_dim_without_embedding": 7,
    }


def _resolve_source_record(
    records: list[dict[str, Any]],
    source_index: int,
    label_question_id: str,
) -> tuple[dict[str, Any], int, str]:
    candidates: list[tuple[int, str]] = [(source_index, "zero_based")]
    if source_index > 0:
        candidates.append((source_index - 1, "one_based"))

    mismatches: list[str] = []
    for offset, index_base in candidates:
        if offset < 0 or offset >= len(records):
            continue
        record = records[offset]
        source_question_id = str(record.get("question_id") or "")
        if not source_question_id or not label_question_id or source_question_id == label_question_id:
            return record, offset, index_base
        mismatches.append(
            f"{index_base}: offset={offset} question_id={source_question_id!r}"
        )

    bounds = f"valid zero-based offsets 0..{len(records) - 1}"
    mismatch_text = "; ".join(mismatches) if mismatches else "no in-range candidate"
    raise FeatureManifestError(
        f"source_record_index={source_index} could not resolve "
        f"question_id={label_question_id!r} ({bounds}; {mismatch_text})"
    )


def _role_result(record: dict[str, Any], role_key: str) -> dict[str, Any]:
    role_results = record.get("role_results")
    if not isinstance(role_results, dict):
        raise FeatureManifestError("source record missing role_results object")
    value = role_results.get(role_key)
    if value is None and role_key == "frontdoor:direct":
        value = role_results.get("frontdoor")
    if not isinstance(value, dict):
        raise FeatureManifestError(f"source record missing role result for {role_key!r}")
    return value


def _assert_prompt_free(rows: Iterable[dict[str, Any]]) -> None:
    for row_number, row in enumerate(rows, start=1):
        present = sorted(PRIVATE_FIELDS & set(row))
        if present:
            raise FeatureManifestError(
                f"manifest row {row_number}: private fields present: {', '.join(present)}"
            )


def build_feature_manifest(
    labels_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = load_jsonl(labels_path)
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    by_source = Counter()
    by_role = Counter()
    by_suite = Counter()

    for row_number, label in enumerate(labels, start=1):
        if label.get("schema_version") != LABEL_SCHEMA_VERSION:
            raise FeatureManifestError(
                f"{labels_path}:{row_number}: expected schema_version={LABEL_SCHEMA_VERSION!r}"
            )
        source_path = Path(str(label.get("source_path") or ""))
        if not source_path.exists():
            raise FeatureManifestError(f"{labels_path}:{row_number}: missing source_path {source_path}")
        source_index = label.get("source_record_index")
        if not isinstance(source_index, int):
            raise FeatureManifestError(f"{labels_path}:{row_number}: source_record_index must be int")
        if source_path not in source_cache:
            source_cache[source_path] = _read_source_records(source_path)
        records = source_cache[source_path]
        label_question_id = str(label.get("question_id") or "")
        try:
            record, source_record_offset, source_record_index_base = _resolve_source_record(
                records,
                source_index,
                label_question_id,
            )
        except FeatureManifestError as exc:
            raise FeatureManifestError(f"{labels_path}:{row_number}: {exc}") from exc
        role_key = str(label.get("role_key") or "")
        role_result = _role_result(record, role_key)
        prompt = str(record.get("prompt") or "")
        expected = str(record.get("expected") or "")
        answer = str(role_result.get("answer") or "")
        result_role = str(role_result.get("role") or role_key)
        join_key = (
            f"{source_path}:{source_record_offset}:{role_key}:"
            f"{label.get('item_id')}"
        )
        source_question_id = str(record.get("question_id") or "")
        if source_question_id and label_question_id and source_question_id != label_question_id:
            raise FeatureManifestError(
                f"{labels_path}:{row_number}: question_id mismatch "
                f"{label_question_id!r} != {source_question_id!r}"
            )
        feature_context = _context_features(record)
        manifest_row = {
            "schema_version": FEATURE_ROW_SCHEMA_VERSION,
            "item_id": label.get("item_id"),
            "join_key": join_key,
            "question_id": label.get("question_id"),
            "suite": label.get("suite"),
            "role_key": role_key,
            "source_path": str(source_path),
            "source_record_index": source_index,
            "source_record_offset": source_record_offset,
            "source_record_index_base": source_record_index_base,
            "source_record_count": len(records),
            "source_role": result_role,
            "source_passed": role_result.get("passed"),
            "source_elapsed_seconds": role_result.get("elapsed_seconds"),
            "source_error_present": bool(role_result.get("error")),
            "prompt_sha256": _hash_text(prompt),
            "expected_sha256": _hash_text(expected),
            "answer_sha256": _hash_text(answer),
            "prompt_chars": len(prompt),
            "expected_chars": len(expected),
            "answer_chars": len(answer),
            "feature_context": feature_context,
            "oracle_binary_label": label.get("oracle_binary_label"),
            "oracle_score": label.get("oracle_score"),
            "oracle_threshold": label.get("oracle_threshold"),
            "oracle_score_source": label.get("oracle_score_source"),
            "target_binary_label": label.get("target_binary_label"),
            "target_source": label.get("target_source"),
            "label_source": label.get("label_source"),
            "label_status": label.get("label_status"),
        }
        manifest_rows.append(manifest_row)
        by_source[str(source_path)] += 1
        by_role[role_key] += 1
        by_suite[str(label.get("suite") or "unknown")] += 1

    _assert_prompt_free(manifest_rows)
    source_record_keys = {
        (row["source_path"], row["source_record_offset"]) for row in manifest_rows
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "labels_jsonl": str(labels_path),
        "feature_row_schema_version": FEATURE_ROW_SCHEMA_VERSION,
        "rows": len(manifest_rows),
        "unique_source_records": len(source_record_keys),
        "sources": dict(sorted(by_source.items())),
        "roles": dict(sorted(by_role.items())),
        "suites": dict(sorted(by_suite.items())),
        "feature_contract": {
            "embedding_dim_required": 1024,
            "engineered_feature_dim": 7,
            "engineered_features": [
                "task_type_onehot[5]",
                "log1p(context_length)/12.0",
                "has_images",
            ],
            "next_step": (
                "embed source prompt/context text for each row, append engineered "
                "features, then join to oracle_binary_label by join_key"
            ),
        },
        "privacy": {
            "private_fields_excluded": sorted(PRIVATE_FIELDS),
            "text_represented_by_sha256_and_lengths": True,
        },
    }
    return manifest_rows, summary


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
        "# Offline Reward Feature Manifest",
        "",
        f"- Labels: `{summary['labels_jsonl']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Unique source records: `{summary['unique_source_records']}`",
        f"- Embedding dimension required: `{summary['feature_contract']['embedding_dim_required']}`",
        f"- Engineered feature dimension: `{summary['feature_contract']['engineered_feature_dim']}`",
        "",
        "The manifest is prompt-free: text is represented only by SHA-256 hashes and lengths.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a prompt-free feature-input manifest for offline reward labels",
    )
    parser.add_argument("--labels-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--summary-md", type=Path)
    args = parser.parse_args(argv)
    try:
        rows, summary = build_feature_manifest(args.labels_jsonl)
    except (FeatureManifestError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_jsonl(args.manifest_jsonl, rows)
    write_json(args.summary_json, summary)
    if args.summary_md:
        write_markdown(args.summary_md, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
