#!/usr/bin/env python3
"""Build offline reward-oracle rows from seeding/eval artifacts.

This prepares the A9 learned-routing reward-oracle lane by extracting
`(reference, response, binary_reward)` pairs from existing benchmark outputs.
The default output is scorer input: it intentionally omits `oracle_score` until
an offline scorer such as the AVB tiny reward model has produced scores.

For plumbing/baseline smoke tests, `--oracle-score-mode binary_reward|q_reward`
can copy an existing reward into `oracle_score`; those modes are not a model
evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROLE_RESULT_KEYS = ("role_results", "results_by_role")


def _load_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                raw = json.loads(stripped)
                if not isinstance(raw, dict):
                    raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
                yield raw
        return

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if isinstance(raw.get("results"), list):
            yield from raw["results"]
            return
        if isinstance(raw.get("records"), list):
            yield from raw["records"]
            return
        yield raw
        return
    if isinstance(raw, list):
        yield from raw
        return
    raise ValueError(f"{path}: unsupported JSON root {type(raw).__name__}")


def _role_results(record: dict[str, Any]) -> dict[str, Any]:
    for key in ROLE_RESULT_KEYS:
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _binary_reward(role_result: dict[str, Any], reward_value: Any) -> float:
    if isinstance(role_result.get("passed"), bool):
        return 1.0 if role_result["passed"] else 0.0
    if reward_value is not None:
        try:
            return 1.0 if float(reward_value) >= 0.5 else 0.0
        except (TypeError, ValueError):
            pass
    return 0.0


def _row_id(path: Path, record_index: int, role_key: str) -> str:
    safe_role = role_key.replace(":", "_").replace("/", "_")
    return f"{path.stem}:{record_index}:{safe_role}"


def build_rows(
    paths: Iterable[Path],
    *,
    oracle_score_mode: str = "omit",
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = Counter()
    suite_counts = Counter()
    role_counts = Counter()

    for path in paths:
        for record_index, record in enumerate(_load_records(path), start=1):
            if not isinstance(record, dict):
                stats["skipped_non_object_record"] += 1
                continue
            reference = str(record.get("expected") or record.get("reference") or "")
            if not reference:
                stats["skipped_missing_reference"] += 1
                continue
            role_results = _role_results(record)
            if not role_results:
                stats["skipped_missing_role_results"] += 1
                continue
            rewards = record.get("rewards") if isinstance(record.get("rewards"), dict) else {}
            suite = str(record.get("suite") or "unknown")
            question_id = str(record.get("question_id") or record.get("qid") or "")

            for role_key, role_result in role_results.items():
                if not isinstance(role_result, dict):
                    stats["skipped_bad_role_result"] += 1
                    continue
                response = str(role_result.get("answer") or role_result.get("response") or "")
                if not response:
                    stats["skipped_missing_response"] += 1
                    continue
                reward_value = rewards.get(role_key)
                binary_reward = _binary_reward(role_result, reward_value)
                row = {
                    "item_id": _row_id(path, record_index, str(role_key)),
                    "source_path": str(path),
                    "source_record_index": record_index,
                    "question_id": question_id,
                    "suite": suite,
                    "role_key": str(role_key),
                    "role": str(role_result.get("role") or ""),
                    "reference": reference,
                    "response": response,
                    "binary_reward": binary_reward,
                }
                if reward_value is not None:
                    try:
                        row["q_reward"] = float(reward_value)
                    except (TypeError, ValueError):
                        stats["skipped_non_numeric_q_reward"] += 1
                if oracle_score_mode == "binary_reward":
                    row["oracle_score"] = binary_reward
                elif oracle_score_mode == "q_reward":
                    row["oracle_score"] = float(row.get("q_reward", binary_reward))

                rows.append(row)
                suite_counts[suite] += 1
                role_counts[str(role_key)] += 1
                stats["rows"] += 1
                if max_rows is not None and len(rows) >= max_rows:
                    return rows, _summary(stats, suite_counts, role_counts, oracle_score_mode)

    return rows, _summary(stats, suite_counts, role_counts, oracle_score_mode)


def _summary(
    stats: Counter,
    suite_counts: Counter,
    role_counts: Counter,
    oracle_score_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": "offline_reward_oracle_rows.v1",
        "oracle_score_mode": oracle_score_mode,
        "rows": int(stats["rows"]),
        "stats": {key: int(value) for key, value in sorted(stats.items())},
        "suite_counts": {key: int(value) for key, value in sorted(suite_counts.items())},
        "role_counts": {key: int(value) for key, value in sorted(role_counts.items())},
    }


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build offline reward-oracle rows from seeding/eval artifacts",
    )
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument(
        "--oracle-score-mode",
        choices=("omit", "binary_reward", "q_reward"),
        default="omit",
        help="How to populate oracle_score. Default omits it for external scorer input.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)

    rows, summary = build_rows(
        args.input,
        oracle_score_mode=args.oracle_score_mode,
        max_rows=args.max_rows,
    )
    if not rows:
        raise SystemExit("no rows extracted")

    write_jsonl(rows, args.output_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
