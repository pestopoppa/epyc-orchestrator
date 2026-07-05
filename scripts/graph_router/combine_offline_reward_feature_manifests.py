#!/usr/bin/env python3
"""Combine prompt-free offline reward feature manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.build_offline_reward_feature_manifest import (  # noqa: E402
    FEATURE_ROW_SCHEMA_VERSION,
    PRIVATE_FIELDS,
)


SUMMARY_SCHEMA_VERSION = "offline_reward_feature_manifest_combined_summary.v1"


class FeatureManifestCombineError(ValueError):
    """Raised when feature manifests cannot be safely combined."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise FeatureManifestCombineError(f"{path}:{line_number}: expected object")
            rows.append(value)
    if not rows:
        raise FeatureManifestCombineError(f"{path}: no rows")
    return rows


def _row_key(row: dict[str, Any]) -> str:
    join_key = str(row.get("join_key") or "")
    if join_key:
        return join_key
    item_id = str(row.get("item_id") or "")
    if item_id:
        return item_id
    raise FeatureManifestCombineError("feature row missing join_key and item_id")


def _source_family(row: dict[str, Any]) -> str:
    context = row.get("feature_context")
    if isinstance(context, dict) and context.get("source_family"):
        return str(context["source_family"])
    return "unknown"


def _validate_prompt_free(row: dict[str, Any], *, source: Path, line_number: int) -> None:
    if row.get("schema_version") != FEATURE_ROW_SCHEMA_VERSION:
        raise FeatureManifestCombineError(
            f"{source}:{line_number}: expected schema_version={FEATURE_ROW_SCHEMA_VERSION!r}"
        )
    present = sorted(PRIVATE_FIELDS & set(row))
    if present:
        raise FeatureManifestCombineError(
            f"{source}:{line_number}: private fields present: {', '.join(present)}"
        )


def combine_feature_manifests(
    *,
    base_manifest: Path,
    expansion_manifests: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not expansion_manifests:
        raise FeatureManifestCombineError("at least one expansion manifest is required")
    inputs = [base_manifest, *expansion_manifests]
    combined: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    source_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()

    for path in inputs:
        for line_number, row in enumerate(_load_jsonl(path), start=1):
            _validate_prompt_free(row, source=path, line_number=line_number)
            key = _row_key(row)
            if key in seen:
                raise FeatureManifestCombineError(
                    f"duplicate feature row {key!r} in {path}; first seen in {seen[key]}"
                )
            seen[key] = path
            combined.append(row)
            source_counts[_source_family(row)] += 1
            role_counts[str(row.get("role_key") or "unknown")] += 1
            suite_counts[str(row.get("suite") or "unknown")] += 1

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "base_manifest_jsonl": str(base_manifest),
        "expansion_manifest_jsonl": str(expansion_manifests[0]),
        "expansion_manifest_jsonls": [str(path) for path in expansion_manifests],
        "rows": len(combined),
        "source_family_counts": dict(sorted(source_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "suite_counts": dict(sorted(suite_counts.items())),
        "privacy": {
            "commits_private_text": False,
            "text_represented_by_sha256_and_lengths": True,
        },
    }
    return combined, summary


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_markdown(summary: dict[str, Any]) -> str:
    expansions = ", ".join(f"`{path}`" for path in summary["expansion_manifest_jsonls"])
    return "\n".join(
        [
            "# Offline Reward Feature Manifest Combination",
            "",
            f"- Base manifest: `{summary['base_manifest_jsonl']}`",
            f"- Expansion manifests: {expansions}",
            f"- Output manifest: `{summary['manifest_jsonl']}`",
            f"- Rows: `{summary['rows']}`",
            f"- Source families: `{summary['source_family_counts']}`",
            "- Runtime gate change allowed: `False`",
            "",
            "This artifact is prompt-free; private prompt/answer/reference text is not committed.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine prompt-free offline reward feature manifests.",
    )
    parser.add_argument("--base-manifest-jsonl", required=True, type=Path)
    parser.add_argument("--expansion-manifest-jsonl", required=True, action="append", type=Path)
    parser.add_argument("--manifest-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--summary-md", type=Path)
    args = parser.parse_args(argv)

    try:
        rows, summary = combine_feature_manifests(
            base_manifest=args.base_manifest_jsonl,
            expansion_manifests=args.expansion_manifest_jsonl,
        )
    except (FeatureManifestCombineError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = {**summary, "manifest_jsonl": str(args.manifest_jsonl)}
    _write_jsonl(args.manifest_jsonl, rows)
    _write_summary(args.summary_json, summary)
    if args.summary_md:
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text(_render_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
