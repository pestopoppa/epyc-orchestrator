#!/usr/bin/env python3
"""Prepare private review packets for answer-equivalence disagreement rows.

The committed audit artifact is intentionally prompt-free. This helper rebuilds
the private reviewer packet from source coordinates and writes a separate
redacted manifest that can be committed safely.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PRIVATE_FIELDS = {"prompt", "expected", "reference", "response", "answer"}
LABEL_OPTIONS = ("equivalent", "not_equivalent", "needs_semantic_judge")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{line_number}: row must be an object")
            rows.append(obj)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def _source_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return obj["results"]
    raise SystemExit(f"{path}: unsupported source result shape")


def _source_row(path: Path, index: int, *, question_id: str | None = None) -> dict[str, Any]:
    rows = _source_records(path)
    candidates = [index]
    if index > 0:
        candidates.append(index - 1)
    for candidate in candidates:
        if candidate < 0 or candidate >= len(rows):
            continue
        row = rows[candidate]
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{candidate}: source row must be an object")
        if not question_id or str(row.get("question_id") or "") == question_id:
            return row
    raise SystemExit(
        f"{path}: source_record_index {index} could not be resolved "
        f"for question_id={question_id!r} across {len(rows)} rows"
    )


def _role_result(source_row: dict[str, Any], role_key: str) -> dict[str, Any]:
    role_results = source_row.get("role_results")
    if not isinstance(role_results, dict):
        return {}
    candidates = [role_key]
    if ":" in role_key:
        candidates.append(role_key.replace(":", "_"))
        candidates.append(role_key.split(":", 1)[0])
    for candidate in candidates:
        value = role_results.get(candidate)
        if isinstance(value, dict):
            return value
    return {}


def _private_review_row(disagreement: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(str(disagreement["source_path"]))
    source_index = int(disagreement["source_record_index"])
    source = _source_row(
        source_path,
        source_index,
        question_id=str(disagreement.get("question_id") or "") or None,
    )
    role_key = str(disagreement.get("role_key") or disagreement.get("role") or "")
    role = _role_result(source, role_key)
    prompt = source.get("prompt")
    reference = source.get("expected")
    response = role.get("answer")
    if prompt is None or reference is None or response is None:
        raise SystemExit(
            f"{disagreement.get('item_id')}: missing prompt/expected/answer "
            f"from {source_path}:{source_index} role={role_key}"
        )
    source_passed = role.get("passed")
    disagreement_type = _disagreement_type(disagreement)
    seed_label = _seed_label(
        source_passed=source_passed,
        disagreement_type=disagreement_type,
    )
    return {
        "schema_version": "answer_equivalence_review_private.v1",
        "item_id": disagreement.get("item_id", ""),
        "source_path": str(source_path),
        "source_record_index": source_index,
        "question_id": disagreement.get("question_id") or source.get("question_id", ""),
        "suite": disagreement.get("suite") or source.get("suite", ""),
        "role_key": role_key,
        "truth_label": disagreement.get("truth_label"),
        "equivalence_proxy_label": disagreement.get("equivalence_proxy_label"),
        "q_reward": disagreement.get("q_reward"),
        "binary_reward": disagreement.get("binary_reward"),
        "oracle_score": disagreement.get("oracle_score"),
        "token_f1": disagreement.get("token_f1"),
        "source_passed": source_passed if isinstance(source_passed, bool) else None,
        "source_error_type": role.get("error_type"),
        "prompt": prompt,
        "reference": reference,
        "response": response,
        "manual_label": None,
        "judge_label": None,
        "semantic_label": seed_label["semantic_label"],
        "final_label": seed_label["final_label"],
        "label_source": seed_label["label_source"],
        "label_status": seed_label["label_status"],
        "label_options": list(LABEL_OPTIONS),
    }


def _disagreement_type(row: dict[str, Any]) -> str:
    truth = row.get("truth_label")
    proxy = row.get("equivalence_proxy_label")
    if truth == 1 and proxy == 0:
        return "current_positive_not_deterministically_reconstructable"
    if truth == 0 and proxy == 1:
        return "current_negative_deterministically_equivalent"
    return "other_disagreement"


def _seed_label(*, source_passed: Any, disagreement_type: str) -> dict[str, str | None]:
    if (
        source_passed is True
        and disagreement_type == "current_positive_not_deterministically_reconstructable"
    ):
        return {
            "semantic_label": "equivalent",
            "final_label": "equivalent",
            "label_source": "source_passed_true",
            "label_status": "seeded",
        }
    return {
        "semantic_label": None,
        "final_label": None,
        "label_source": None,
        "label_status": "needs_semantic_judge",
    }


def _public_manifest_row(private_row: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in private_row.items() if key not in PRIVATE_FIELDS}
    row["schema_version"] = "answer_equivalence_review_manifest.v1"
    row["disagreement_type"] = _disagreement_type(private_row)
    row["review_bucket"] = row["disagreement_type"]
    row["prompt_chars"] = len(str(private_row["prompt"]))
    row["reference_chars"] = len(str(private_row["reference"]))
    row["response_chars"] = len(str(private_row["response"]))
    return row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Answer-Equivalence Review Packet",
        "",
        f"- Status: `{summary['status']}`",
        f"- Review rows: `{summary['review_rows']}`",
        f"- Private packet: `{summary['private_review_jsonl']}`",
        f"- Public manifest: `{summary['public_manifest_jsonl']}`",
        "",
        "## Disagreement Types",
        "",
        "| Type | Rows |",
        "|---|---:|",
    ]
    for key, value in summary["by_disagreement_type"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Label Status", "", "| Status | Rows |", "|---|---:|"])
    for key, value in summary["by_label_status"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Suites", "", "| Suite | Rows |", "|---|---:|"])
    for key, value in summary["by_suite"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "The committed manifest excludes prompt, reference, expected, response, and answer text. "
            "Those fields are present only in the private packet path above.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_review(
    disagreement_rows: list[dict[str, Any]],
    *,
    private_review_jsonl: Path,
    public_manifest_jsonl: Path,
    summary_json: Path,
    summary_md: Path,
) -> dict[str, Any]:
    private_rows = [_private_review_row(row) for row in disagreement_rows]
    public_rows = [_public_manifest_row(row) for row in private_rows]
    write_jsonl(private_review_jsonl, private_rows)
    write_jsonl(public_manifest_jsonl, public_rows)
    by_type = Counter(row["disagreement_type"] for row in public_rows)
    by_suite = Counter(str(row.get("suite") or "unknown") for row in public_rows)
    by_role = Counter(str(row.get("role_key") or "unknown") for row in public_rows)
    by_label_status = Counter(str(row.get("label_status") or "unknown") for row in public_rows)
    summary = {
        "schema_version": "answer_equivalence_review_packet.v1",
        "status": "ready_for_manual_or_judge_labeling",
        "review_rows": len(private_rows),
        "private_review_jsonl": str(private_review_jsonl),
        "public_manifest_jsonl": str(public_manifest_jsonl),
        "label_options": list(LABEL_OPTIONS),
        "by_disagreement_type": dict(sorted(by_type.items())),
        "by_label_status": dict(sorted(by_label_status.items())),
        "by_suite": dict(sorted(by_suite.items())),
        "by_role": dict(sorted(by_role.items())),
        "privacy": {
            "public_manifest_excludes": sorted(PRIVATE_FIELDS),
            "private_packet_committable": False,
        },
    }
    write_json(summary_json, summary)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disagreements-jsonl", type=Path, required=True)
    parser.add_argument("--private-review-jsonl", type=Path, required=True)
    parser.add_argument("--public-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.disagreements_jsonl)
    summary = prepare_review(
        rows,
        private_review_jsonl=args.private_review_jsonl,
        public_manifest_jsonl=args.public_manifest_jsonl,
        summary_json=args.summary_json,
        summary_md=args.summary_md,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
