#!/usr/bin/env python3
"""Score RI-10 canary responses against a scored request-plan answer key."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from scripts.graph_router.reconstruct_answer_equivalence_targets import (
    equivalence_features,
)

DEFAULT_F1_THRESHOLD = 0.8
ANSWER_KEY_SCHEMA = "ri10_canary_answer_key.v1"
SCORE_REPORT_SCHEMA = "ri10_canary_scored_response_report.v1"


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _response_text(row: dict[str, Any]) -> str:
    for field in ("response", "answer", "content", "text", "output"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"].strip()
            if isinstance(first.get("text"), str):
                return first["text"].strip()
    raise ValueError(
        "response row must contain one of response/answer/content/text/output "
        "or choices[0].message.content"
    )


def _request_id(row: dict[str, Any], *, source: str) -> str:
    request_id = str(row.get("request_id") or "").strip()
    if not request_id:
        raise ValueError(f"{source} row missing request_id")
    return request_id


def _index_responses(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = _request_id(row, source="response")
        if request_id in indexed:
            raise ValueError(f"duplicate response request_id: {request_id}")
        indexed[request_id] = row
    return indexed


def _score_one(
    answer_key_row: dict[str, Any],
    response_row: dict[str, Any] | None,
    *,
    f1_threshold: float,
) -> dict[str, Any]:
    request_id = _request_id(answer_key_row, source="answer-key")
    expected = str(answer_key_row.get("expected_answer") or "").strip()
    if not expected:
        raise ValueError(f"{request_id}: expected_answer is required")

    base = {
        "request_id": request_id,
        "role": answer_key_row.get("role"),
        "expected_factual_risk_mode": answer_key_row.get("expected_factual_risk_mode"),
        "prompt_hash": answer_key_row.get("prompt_hash"),
        "domain": answer_key_row.get("domain"),
        "label_source": answer_key_row.get("label_source"),
        "expected_answer": expected,
    }
    if response_row is None:
        return {
            **base,
            "status": "missing_response",
            "response": "",
            "token_f1": 0.0,
            "binary_correct": False,
            "outcome": "MISSING_RESPONSE",
        }

    response = _response_text(response_row)
    features = equivalence_features(expected, response)
    binary = _answer_matches(features, f1_threshold=f1_threshold)
    return {
        **base,
        "status": "scored",
        "response": response,
        "token_f1": features["token_f1"],
        "binary_correct": binary,
        "outcome": "CORRECT" if binary else "INCORRECT",
        "equivalence_features": features,
        "response_meta": {
            key: value
            for key, value in response_row.items()
            if key
            not in {
                "response",
                "answer",
                "content",
                "text",
                "output",
                "choices",
            }
        },
    }


def _answer_matches(features: dict[str, Any], *, f1_threshold: float) -> bool:
    """RI-10 exact-answer scoring for short factual canaries.

    The shared A9 equivalence proxy is conservative about substring matches
    because a one-token reference can appear in a long answer by accident. RI-10
    uses curated exact-answer factual QA rows, so an exact normalized answer
    appearing inside an explanatory sentence is the expected success shape.
    """
    return bool(
        features["normalized_exact"]
        or features["reference_in_response"]
        or features["token_f1"] >= f1_threshold
    )


def _bucket_summary(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        role = str(row.get("role") or "unknown")
        arm = str(row.get("expected_factual_risk_mode") or "unknown")
        for key in ("overall", f"role:{role}", f"arm:{arm}", f"role_arm:{role}:{arm}"):
            bucket = buckets.setdefault(
                key,
                {
                    "rows": 0,
                    "scored": 0,
                    "missing": 0,
                    "correct": 0,
                    "token_f1_sum": 0.0,
                },
            )
            bucket["rows"] += 1
            if row["status"] == "missing_response":
                bucket["missing"] += 1
                continue
            bucket["scored"] += 1
            bucket["correct"] += int(bool(row.get("binary_correct")))
            bucket["token_f1_sum"] += float(row.get("token_f1") or 0.0)

    for bucket in buckets.values():
        scored = int(bucket["scored"])
        bucket["accuracy"] = (bucket["correct"] / scored) if scored else None
        bucket["mean_token_f1"] = (bucket["token_f1_sum"] / scored) if scored else None
        del bucket["token_f1_sum"]
    return buckets


def _arm_comparison(buckets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enforce = buckets.get("arm:enforce", {})
    shadow = buckets.get("arm:shadow", {})
    if enforce.get("accuracy") is None or shadow.get("accuracy") is None:
        return {"status": "insufficient_scored_arms"}
    return {
        "status": "ready",
        "accuracy_delta_enforce_minus_shadow": enforce["accuracy"] - shadow["accuracy"],
        "mean_token_f1_delta_enforce_minus_shadow": (
            enforce["mean_token_f1"] - shadow["mean_token_f1"]
        ),
    }


def score_responses(
    *,
    answer_key_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    f1_threshold: float = DEFAULT_F1_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    responses = _index_responses(response_rows)
    scored = [
        _score_one(row, responses.get(_request_id(row, source="answer-key")), f1_threshold=f1_threshold)
        for row in answer_key_rows
    ]
    buckets = _bucket_summary(scored)
    counts = Counter(row["status"] for row in scored)
    summary = {
        "schema_version": SCORE_REPORT_SCHEMA,
        "generated_at": _iso_now(),
        "answer_key_schema": ANSWER_KEY_SCHEMA,
        "f1_threshold": f1_threshold,
        "status": "ready" if counts.get("missing_response", 0) == 0 else "missing_responses",
        "rows": len(scored),
        "status_counts": dict(sorted(counts.items())),
        "buckets": buckets,
        "arm_comparison": _arm_comparison(buckets),
    }
    return scored, summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RI-10 Canary Scored Response Report",
        "",
        f"- Schema: `{summary['schema_version']}`",
        f"- Status: `{summary['status']}`",
        f"- Rows: `{summary['rows']}`",
        f"- F1 threshold: `{summary['f1_threshold']}`",
        f"- Arm comparison: `{summary['arm_comparison']['status']}`",
        "",
        "## Buckets",
        "",
        "| Bucket | Rows | Scored | Missing | Correct | Accuracy | Mean Token F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, bucket in sorted(summary["buckets"].items()):
        accuracy = bucket["accuracy"]
        mean_f1 = bucket["mean_token_f1"]
        lines.append(
            f"| `{key}` | {bucket['rows']} | {bucket['scored']} | "
            f"{bucket['missing']} | {bucket['correct']} | "
            f"{accuracy:.4f} | {mean_f1:.4f} |"
            if accuracy is not None and mean_f1 is not None
            else (
                f"| `{key}` | {bucket['rows']} | {bucket['scored']} | "
                f"{bucket['missing']} | {bucket['correct']} | n/a | n/a |"
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--responses-jsonl", type=Path, required=True)
    parser.add_argument("--scored-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--f1-threshold", type=float, default=DEFAULT_F1_THRESHOLD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    answer_key_rows = load_jsonl(args.answer_key)
    response_rows = load_jsonl(args.responses_jsonl)
    scored, summary = score_responses(
        answer_key_rows=answer_key_rows,
        response_rows=response_rows,
        f1_threshold=args.f1_threshold,
    )
    write_jsonl(scored, args.scored_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(render_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
