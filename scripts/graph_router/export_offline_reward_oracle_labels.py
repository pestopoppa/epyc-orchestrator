#!/usr/bin/env python3
"""Export prompt-free labels from an adopted offline reward oracle.

This is the NEXT-A2/A3 bridge after an oracle clears the offline decision gate:
the adoption manifest proves the scorer is eligible, while the private scored
JSONL provides row-level scores. The output deliberately strips prompt/reference
and response text so it can be committed as a durable label source.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.build_offline_reward_oracle_manifest import (
    ADOPTABLE_STATUS,
    MANIFEST_SCHEMA_VERSION,
    load_json,
)


LABEL_SCHEMA_VERSION = "offline_reward_oracle_label.v1"
SUMMARY_SCHEMA_VERSION = "offline_reward_oracle_label_export_summary.v1"
PRIVATE_FIELDS = {
    "answer",
    "expected",
    "prompt",
    "reference",
    "response",
}


class LabelExportError(ValueError):
    """Raised when manifest or row data cannot be exported safely."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise LabelExportError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise LabelExportError(f"{path}:{line_number}: expected object")
            rows.append(value)
    if not rows:
        raise LabelExportError(f"{path}: no rows")
    return rows


def _as_float(value: Any, *, field: str, row_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LabelExportError(f"{row_id}: {field} must be numeric, got {value!r}") from exc
    return parsed


def _manifest_oracle(manifest: dict[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise LabelExportError(
            f"{manifest_path}: expected schema_version={MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.get("status") != ADOPTABLE_STATUS:
        raise LabelExportError(
            f"{manifest_path}: expected status={ADOPTABLE_STATUS!r}, "
            f"got {manifest.get('status')!r}"
        )
    oracle = manifest.get("oracle")
    if not isinstance(oracle, dict):
        raise LabelExportError(f"{manifest_path}: missing oracle object")
    for key in ("model_id", "oracle_score_source", "oracle_threshold"):
        if key not in oracle:
            raise LabelExportError(f"{manifest_path}: missing oracle.{key}")
    return oracle


def _expected_rows(manifest: dict[str, Any], *, manifest_path: Path) -> int:
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict):
        raise LabelExportError(f"{manifest_path}: missing evidence object")
    rows = evidence.get("rows")
    if not isinstance(rows, int) or rows <= 0:
        raise LabelExportError(f"{manifest_path}: evidence.rows must be positive int")
    return rows


def _target_binary(row: dict[str, Any]) -> int | None:
    if "target_score" in row and row["target_score"] is not None:
        return 1 if float(row["target_score"]) >= 0.5 else 0
    if "binary_reward" in row and row["binary_reward"] is not None:
        return 1 if float(row["binary_reward"]) >= 0.5 else 0
    return None


def _label_row(
    row: dict[str, Any],
    *,
    threshold: float,
    model_id: str,
    score_source: str,
    row_number: int,
) -> dict[str, Any]:
    item_id = str(row.get("item_id") or row.get("id") or "").strip()
    if not item_id:
        raise LabelExportError(f"row {row_number}: missing item_id")
    row_model = str(row.get("oracle_model_id") or "").strip()
    row_source = str(row.get("oracle_score_source") or "").strip()
    if row_model != model_id:
        raise LabelExportError(
            f"{item_id}: oracle_model_id mismatch: expected {model_id!r}, got {row_model!r}"
        )
    if row_source != score_source:
        raise LabelExportError(
            f"{item_id}: oracle_score_source mismatch: "
            f"expected {score_source!r}, got {row_source!r}"
        )
    score = _as_float(row.get("oracle_score"), field="oracle_score", row_id=item_id)
    target_binary = _target_binary(row)
    labeled = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "item_id": item_id,
        "question_id": row.get("question_id"),
        "suite": row.get("suite"),
        "role_key": row.get("role_key") or row.get("role"),
        "source_path": row.get("source_path"),
        "source_record_index": row.get("source_record_index"),
        "target_source": row.get("target_source"),
        "oracle_model_id": model_id,
        "oracle_score_source": score_source,
        "oracle_threshold": threshold,
        "oracle_score": score,
        "oracle_binary_label": 1 if score >= threshold else 0,
        "label_source": f"{score_source}@{threshold:.6g}",
        "label_status": "oracle_labeled",
    }
    for key in ("binary_reward", "q_reward", "target_score"):
        if key in row:
            labeled[key] = row.get(key)
    if target_binary is not None:
        labeled["target_binary_label"] = target_binary
        labeled["oracle_matches_target"] = bool(labeled["oracle_binary_label"] == target_binary)
    return labeled


def _assert_prompt_free(rows: Iterable[dict[str, Any]]) -> None:
    for row_number, row in enumerate(rows, start=1):
        present = sorted(PRIVATE_FIELDS & set(row))
        if present:
            raise LabelExportError(
                f"output row {row_number}: private fields present: {', '.join(present)}"
            )


def _slice_counts(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row.get(key) or "unknown")].append(row)
    out: dict[str, dict[str, Any]] = {}
    for value, group in sorted(by_key.items()):
        positives = sum(int(row["oracle_binary_label"]) for row in group)
        matched = [row for row in group if "oracle_matches_target" in row]
        out[value] = {
            "rows": len(group),
            "oracle_positive": positives,
            "oracle_negative": len(group) - positives,
            "target_compared": len(matched),
            "target_agreement": (
                sum(bool(row["oracle_matches_target"]) for row in matched) / len(matched)
                if matched
                else None
            ),
        }
    return out


def export_labels(
    *,
    manifest_path: Path,
    scored_rows_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    oracle = _manifest_oracle(manifest, manifest_path=manifest_path)
    threshold = float(oracle["oracle_threshold"])
    model_id = str(oracle["model_id"])
    score_source = str(oracle["oracle_score_source"])

    raw_rows = load_jsonl(scored_rows_path)
    expected_rows = _expected_rows(manifest, manifest_path=manifest_path)
    if len(raw_rows) != expected_rows:
        raise LabelExportError(
            f"row-count mismatch: manifest evidence.rows={expected_rows}, "
            f"scored rows={len(raw_rows)}"
        )

    labels: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(raw_rows, start=1):
        label = _label_row(
            row,
            threshold=threshold,
            model_id=model_id,
            score_source=score_source,
            row_number=row_number,
        )
        item_id = str(label["item_id"])
        if item_id in seen:
            raise LabelExportError(f"duplicate item_id={item_id!r}")
        seen.add(item_id)
        labels.append(label)
    _assert_prompt_free(labels)

    compared = [row for row in labels if "oracle_matches_target" in row]
    counts = Counter(int(row["oracle_binary_label"]) for row in labels)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "manifest_json": str(manifest_path),
        "scored_rows_jsonl": str(scored_rows_path),
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "oracle": {
            "model_id": model_id,
            "oracle_score_source": score_source,
            "oracle_threshold": threshold,
        },
        "rows": len(labels),
        "oracle_positive": int(counts[1]),
        "oracle_negative": int(counts[0]),
        "target_compared": len(compared),
        "target_agreement": (
            sum(bool(row["oracle_matches_target"]) for row in compared) / len(compared)
            if compared
            else None
        ),
        "privacy": {
            "private_fields_excluded": sorted(PRIVATE_FIELDS),
            "commits_private_rows": False,
        },
        "slices": {
            "target_source": _slice_counts(labels, "target_source"),
            "suite": _slice_counts(labels, "suite"),
            "role_key": _slice_counts(labels, "role_key"),
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
        "# Offline Reward Oracle Label Export",
        "",
        f"- Manifest: `{summary['manifest_json']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Oracle: `{summary['oracle']['model_id']}`",
        f"- Score source: `{summary['oracle']['oracle_score_source']}`",
        f"- Threshold: `{summary['oracle']['oracle_threshold']}`",
        f"- Oracle positives / negatives: `{summary['oracle_positive']}` / `{summary['oracle_negative']}`",
        f"- Target agreement: `{summary['target_agreement']}`",
        "",
        "The exported label rows exclude prompt, reference, response, expected, and answer text.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export prompt-free labels from an adopted offline reward oracle",
    )
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--scored-rows-jsonl", required=True, type=Path)
    parser.add_argument("--labels-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--summary-md", type=Path)
    args = parser.parse_args(argv)

    try:
        labels, summary = export_labels(
            manifest_path=args.manifest_json,
            scored_rows_path=args.scored_rows_jsonl,
        )
    except (LabelExportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_jsonl(args.labels_jsonl, labels)
    write_json(args.summary_json, summary)
    if args.summary_md:
        write_markdown(args.summary_md, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
