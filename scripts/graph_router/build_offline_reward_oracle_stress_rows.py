#!/usr/bin/env python3
"""Build mandatory stress rows for offline reward-oracle evaluation.

The AVB A9 lane requires paraphrase/synonym and confound stress tests before an
offline scorer can be trusted. This script turns scorer-input rows from
`build_offline_reward_oracle_rows.py` into grouped stress rows:

  - base: the original positive row
  - paraphrase: a deterministic correct rewording wrapper around the response
  - confound: a plausible wrong answer borrowed from a different row

It does not assign `oracle_score`; score these rows with the candidate offline
reward model before passing them to `evaluate_offline_reward_oracle.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(raw)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _target_positive(row: dict[str, Any]) -> bool:
    try:
        return float(row.get("binary_reward", row.get("q_reward", 0.0))) >= 0.5
    except (TypeError, ValueError):
        return False


def _text(row: dict[str, Any], field: str) -> str:
    return str(row.get(field) or "").strip()


def _optional_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _paraphrase_response(response: str) -> str:
    if not response:
        return response
    return f"In other words: {response}"


def _candidate_confounds(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        response = _text(row, "response")
        reference = _text(row, "reference")
        if response and reference:
            candidates.append(row)
    return candidates


def _pick_confound(
    base: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    base_reference = _text(base, "reference")
    base_response = _text(base, "response")
    base_item_id = _text(base, "item_id")
    for candidate in candidates:
        if _text(candidate, "item_id") == base_item_id:
            continue
        candidate_response = _text(candidate, "response")
        candidate_reference = _text(candidate, "reference")
        if not candidate_response or not candidate_reference:
            continue
        if candidate_response == base_response:
            continue
        if candidate_reference == base_reference:
            continue
        return candidate
    return None


def _strip_scores(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.pop("oracle_score", None)
    return out


def _variant(
    base: dict[str, Any],
    *,
    variant_type: str,
    variant_group: str,
    response: str,
    binary_reward: float,
    q_reward: float | None = None,
    confound_source_item_id: str | None = None,
) -> dict[str, Any]:
    row = _strip_scores(base)
    base_item_id = str(base.get("item_id") or "")
    row["item_id"] = f"{base_item_id}:{variant_type}"
    row["variant_group"] = variant_group
    row["variant_type"] = variant_type
    row["response"] = response
    row["binary_reward"] = binary_reward
    if q_reward is None:
        row.pop("q_reward", None)
    else:
        row["q_reward"] = q_reward
    if confound_source_item_id:
        row["confound_source_item_id"] = confound_source_item_id
    return row


def build_stress_rows(
    rows: list[dict[str, Any]],
    *,
    max_groups: int | None = None,
    include_base: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = _candidate_confounds(rows)
    out: list[dict[str, Any]] = []
    stats = Counter()

    for row in rows:
        if not _target_positive(row):
            stats["skipped_non_positive_base"] += 1
            continue
        response = _text(row, "response")
        reference = _text(row, "reference")
        if not response or not reference:
            stats["skipped_missing_text"] += 1
            continue
        confound = _pick_confound(row, candidates)
        if confound is None:
            stats["skipped_no_confound"] += 1
            continue

        variant_group = str(row.get("item_id") or f"group-{stats['groups'] + 1}")
        if include_base:
            out.append(
                _variant(
                    row,
                    variant_type="base",
                    variant_group=variant_group,
                    response=response,
                    binary_reward=1.0,
                    q_reward=_optional_float(row.get("q_reward"), 1.0),
                )
            )
        out.append(
            _variant(
                row,
                variant_type="paraphrase",
                variant_group=variant_group,
                response=_paraphrase_response(response),
                binary_reward=1.0,
                q_reward=_optional_float(row.get("q_reward"), 1.0),
            )
        )
        out.append(
            _variant(
                row,
                variant_type="confound",
                variant_group=variant_group,
                response=_text(confound, "response"),
                binary_reward=0.0,
                q_reward=0.0,
                confound_source_item_id=str(confound.get("item_id") or ""),
            )
        )
        stats["groups"] += 1
        if max_groups is not None and stats["groups"] >= max_groups:
            break

    summary = {
        "schema_version": "offline_reward_oracle_stress_rows.v1",
        "groups": int(stats["groups"]),
        "rows": len(out),
        "include_base": include_base,
        "stats": {key: int(value) for key, value in sorted(stats.items())},
    }
    return out, summary


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build paraphrase/confound stress rows for offline reward-oracle eval",
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--no-base", action="store_true")
    args = parser.parse_args(argv)

    rows = load_jsonl(args.input_jsonl)
    stress_rows, summary = build_stress_rows(
        rows,
        max_groups=args.max_groups,
        include_base=not args.no_base,
    )
    if not stress_rows:
        raise SystemExit("no stress rows generated")
    write_jsonl(stress_rows, args.output_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
