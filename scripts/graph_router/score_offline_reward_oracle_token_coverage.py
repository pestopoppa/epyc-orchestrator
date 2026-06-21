#!/usr/bin/env python3
"""Score offline reward-oracle rows with deterministic token coverage.

This A9 scorer is a local, dependency-free baseline for reference-grounded
answer equivalence. It measures how much of the reference token set appears in
the candidate response. The output schema matches the other offline oracle
scorers so `evaluate_offline_reward_oracle.py` can calibrate thresholds and
apply the decision gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ORACLE_MODEL_ID = "deterministic/reference-token-coverage-v1"
ORACLE_SCORE_SOURCE = "reference_token_coverage"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


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


def _text(row: dict[str, Any], field: str) -> str:
    return str(row.get(field) or "").strip()


def tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def reference_token_coverage(reference: str, response: str) -> float:
    reference_tokens = tokenize(reference)
    if not reference_tokens:
        return 0.0
    response_tokens = tokenize(response)
    return len(reference_tokens & response_tokens) / len(reference_tokens)


def score_rows(
    rows: Iterable[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    stats = Counter()
    values: list[float] = []

    for index, row in enumerate(rows, start=1):
        reference = _text(row, "reference")
        response = _text(row, "response")
        if not reference:
            raise ValueError(f"row {index}: reference is required")
        if not response:
            raise ValueError(f"row {index}: response is required")
        if "oracle_score" in row and not overwrite:
            raise ValueError(f"row {index}: oracle_score already present; pass --overwrite")

        out = dict(row)
        score = reference_token_coverage(reference, response)
        out["oracle_score"] = score
        out["oracle_score_source"] = ORACLE_SCORE_SOURCE
        out["oracle_model_id"] = ORACLE_MODEL_ID
        scored.append(out)
        values.append(score)
        stats["rows"] += 1
        if out.get("target_source"):
            stats[f"target_source:{out['target_source']}"] += 1
        if out.get("variant_type"):
            stats[f"variant_type:{out['variant_type']}"] += 1

    summary = {
        "schema_version": "offline_reward_oracle_token_coverage_scores.v1",
        "model_id": ORACLE_MODEL_ID,
        "oracle_score_source": ORACLE_SCORE_SOURCE,
        "rows": int(stats["rows"]),
        "score_min": min(values) if values else None,
        "score_max": max(values) if values else None,
        "score_mean": (sum(values) / len(values)) if values else None,
        "stats": {key: int(value) for key, value in sorted(stats.items())},
        "score_definition": (
            "unique lowercase alphanumeric/underscore reference tokens present "
            "in response divided by unique reference tokens"
        ),
    }
    return scored, summary


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score offline reward-oracle rows with reference-token coverage",
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    rows = load_jsonl(args.input_jsonl)
    scored, summary = score_rows(rows, overwrite=args.overwrite)
    write_jsonl(scored, args.output_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
