#!/usr/bin/env python3
"""Train/evaluate a stdlib intake-triage baseline over reviewed labels."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datasets._common import load_jsonl, stable_hash, utc_now


DEFAULT_DATA = Path("orchestration/datasets/intake_triage.jsonl")
DEFAULT_REPORT = Path("orchestration/reports/intake_triage_baseline_report.json")
BASELINE_VERSION = "intake_triage_nb_baseline.v1"
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.-]*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _is_reviewed(row: dict[str, Any]) -> bool:
    return bool(row.get("reviewed_at") or row.get("output_contract_version")) or str(
        row.get("label_source") or ""
    ) not in {"", "research-intake"}


def _eligible_rows(
    rows: list[dict[str, Any]],
    *,
    target_field: str,
    text_field: str,
    require_reviewed: bool,
) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        if row.get("exclude_reason"):
            continue
        if require_reviewed and not _is_reviewed(row):
            continue
        if not row.get(target_field) or not row.get(text_field):
            continue
        eligible.append(row)
    return eligible


def _split_rows(rows: list[dict[str, Any]], heldout_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: stable_hash(
            {
                "example_id": row.get("example_id"),
                "intake_id": row.get("intake_id"),
                "target": row.get("verdict"),
            }
        ),
    )
    if len(ordered) < 2:
        return ordered, []
    n_heldout = max(1, min(len(ordered) - 1, round(len(ordered) * heldout_frac)))
    return ordered[n_heldout:], ordered[:n_heldout]


class NaiveBayesBaseline:
    def __init__(self, *, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.class_docs: Counter[str] = Counter()
        self.class_tokens: dict[str, Counter[str]] = defaultdict(Counter)
        self.class_token_totals: Counter[str] = Counter()
        self.vocab: set[str] = set()

    def fit(self, rows: list[dict[str, Any]], *, target_field: str, text_field: str) -> None:
        for row in rows:
            label = str(row[target_field])
            self.class_docs[label] += 1
            tokens = tokenize(str(row[text_field]))
            self.class_tokens[label].update(tokens)
            self.class_token_totals[label] += len(tokens)
            self.vocab.update(tokens)

    def predict(self, text: str) -> str:
        if not self.class_docs:
            raise ValueError("baseline is not fitted")
        tokens = tokenize(text)
        total_docs = sum(self.class_docs.values())
        vocab_size = max(1, len(self.vocab))
        scores: dict[str, float] = {}
        for label, doc_count in self.class_docs.items():
            score = math.log(doc_count / total_docs)
            denom = self.class_token_totals[label] + self.smoothing * vocab_size
            for token in tokens:
                score += math.log(
                    (self.class_tokens[label][token] + self.smoothing) / denom
                )
            scores[label] = score
        return max(scores, key=scores.get)


def evaluate(
    rows: list[dict[str, Any]],
    *,
    target_field: str,
    text_field: str,
    heldout_frac: float,
    smoothing: float,
) -> dict[str, Any]:
    train_rows, heldout_rows = _split_rows(rows, heldout_frac)
    if not train_rows or not heldout_rows:
        return {
            "status": "insufficient_examples",
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
            "accuracy": None,
            "correct": 0,
        }
    baseline = NaiveBayesBaseline(smoothing=smoothing)
    baseline.fit(train_rows, target_field=target_field, text_field=text_field)
    correct = 0
    per_label: dict[str, dict[str, int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for row in heldout_rows:
        expected = str(row[target_field])
        predicted = baseline.predict(str(row[text_field]))
        correct += int(predicted == expected)
        per_label.setdefault(expected, {"total": 0, "correct": 0})
        per_label[expected]["total"] += 1
        per_label[expected]["correct"] += int(predicted == expected)
        confusion.setdefault(expected, {})
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1
    return {
        "status": "evaluated",
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "accuracy": correct / len(heldout_rows),
        "correct": correct,
        "per_label": per_label,
        "confusion": confusion,
        "labels": sorted(baseline.class_docs),
        "vocab_size": len(baseline.vocab),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data).expanduser()
    report_path = Path(args.report).expanduser()
    rows = load_jsonl(data_path)
    reviewed_rows = _eligible_rows(
        rows,
        target_field=args.target_field,
        text_field=args.text_field,
        require_reviewed=True,
    )
    eligible_rows = _eligible_rows(
        rows,
        target_field=args.target_field,
        text_field=args.text_field,
        require_reviewed=args.require_reviewed,
    )
    result = evaluate(
        eligible_rows,
        target_field=args.target_field,
        text_field=args.text_field,
        heldout_frac=args.heldout_frac,
        smoothing=args.smoothing,
    )
    status = result["status"]
    if len(reviewed_rows) < args.min_reviewed_labels:
        status = "insufficient_reviewed_labels"
    elif result["status"] == "evaluated" and result["accuracy"] is not None:
        status = "acceptance_pass" if result["accuracy"] >= args.min_accuracy else "acceptance_fail"
    report = {
        "schema_version": "intake_triage_baseline_report.v1",
        "baseline_version": BASELINE_VERSION,
        "generated_at": utc_now(),
        "data_path": str(data_path),
        "target_field": args.target_field,
        "text_field": args.text_field,
        "status": status,
        "min_reviewed_labels": args.min_reviewed_labels,
        "reviewed_rows": len(reviewed_rows),
        "eligible_rows": len(eligible_rows),
        "source_rows": len(rows),
        "min_accuracy": args.min_accuracy,
        "heldout_frac": args.heldout_frac,
        "smoothing": args.smoothing,
        "evaluation": result,
        "privacy": {
            "raw_text_in_report": False,
            "reported_fields": "aggregate counts only",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"report": str(report_path), "status": status, "reviewed_rows": len(reviewed_rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--target-field", default="verdict")
    parser.add_argument("--text-field", default="features_text")
    parser.add_argument("--min-reviewed-labels", type=int, default=100)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--heldout-frac", type=float, default=0.2)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument(
        "--include-unreviewed",
        dest="require_reviewed",
        action="store_false",
        help="Allow weak research-intake labels for smoke/debug only.",
    )
    parser.set_defaults(require_reviewed=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
