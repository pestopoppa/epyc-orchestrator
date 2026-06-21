#!/usr/bin/env python3
"""Audit answer-equivalence target reconstruction for reward-oracle rows.

This is a no-inference A9 utility. It does not try to replace a semantic judge;
it separates rows whose reference/response equivalence is deterministically
recoverable from rows that still need manual or judged labels before the offline
reward scorer can be trusted as a NEXT-A2/A3 target source.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TARGET_FIELDS = ("binary_reward", "q_reward", "target_score", "score")
PRIVATE_FIELDS = {"reference", "response"}
NEGATIVE_MARKERS = {"none", "n/a", "na", "no answer", "empty"}


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(reference: str, response: str) -> float:
    ref_tokens = normalize_text(reference).split()
    resp_tokens = normalize_text(response).split()
    if not ref_tokens or not resp_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    resp_counts = Counter(resp_tokens)
    overlap = sum((ref_counts & resp_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(resp_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _target_score(row: dict[str, Any]) -> float | None:
    for field in TARGET_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    outcome = str(row.get("outcome") or "").strip().lower()
    if outcome in {"success", "passed", "pass", "ok", "correct", "true"}:
        return 1.0
    if outcome in {"failure", "failed", "fail", "incorrect", "false", "error"}:
        return 0.0
    return None


def _truth_label(row: dict[str, Any], *, threshold: float) -> int | None:
    score = _target_score(row)
    if score is None:
        return None
    return 1 if score >= threshold else 0


def equivalence_features(reference: str, response: str) -> dict[str, Any]:
    ref_norm = normalize_text(reference)
    resp_norm = normalize_text(response)
    exact = bool(ref_norm and ref_norm == resp_norm)
    ref_in_resp = bool(ref_norm and resp_norm and ref_norm in resp_norm)
    resp_in_ref = bool(ref_norm and resp_norm and resp_norm in ref_norm)
    f1 = token_f1(reference, response)
    marker_exact = bool(exact and ref_norm in NEGATIVE_MARKERS)
    return {
        "normalized_exact": exact,
        "reference_in_response": ref_in_resp,
        "response_in_reference": resp_in_ref,
        "token_f1": f1,
        "negative_marker_exact": marker_exact,
        "reference_token_count": len(ref_norm.split()) if ref_norm else 0,
        "response_token_count": len(resp_norm.split()) if resp_norm else 0,
    }


def proxy_label(features: dict[str, Any], *, f1_threshold: float) -> int:
    if features["negative_marker_exact"]:
        return 1
    if features["normalized_exact"] or features["reference_in_response"]:
        if features["normalized_exact"]:
            return 1
        # Substring containment is only a high-confidence equivalence signal
        # when the response is still close to the reference. A one-token
        # reference appearing somewhere inside a long answer is usually not
        # enough to reconstruct an answer-equivalence label.
        ref_tokens = int(features["reference_token_count"])
        resp_tokens = int(features["response_token_count"])
        if features["token_f1"] >= 0.5:
            return 1
        if ref_tokens <= 3 and resp_tokens <= ref_tokens + 4:
            return 1
    if features["token_f1"] >= f1_threshold:
        return 1
    return 0


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


def _safe_row(row: dict[str, Any], *, truth: int | None, proxy: int, features: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": row.get("item_id", ""),
        "source_path": row.get("source_path", ""),
        "source_record_index": row.get("source_record_index"),
        "question_id": row.get("question_id", ""),
        "suite": row.get("suite", ""),
        "role_key": row.get("role_key", ""),
        "role": row.get("role", ""),
        "truth_label": truth,
        "equivalence_proxy_label": proxy,
        "q_reward": row.get("q_reward"),
        "binary_reward": row.get("binary_reward"),
        "oracle_score": row.get("oracle_score"),
        "normalized_exact": features["normalized_exact"],
        "reference_in_response": features["reference_in_response"],
        "response_in_reference": features["response_in_reference"],
        "negative_marker_exact": features["negative_marker_exact"],
        "token_f1": round(float(features["token_f1"]), 6),
        "reference_token_count": features["reference_token_count"],
        "response_token_count": features["response_token_count"],
    }


def audit_rows(
    rows: Iterable[dict[str, Any]],
    *,
    target_threshold: float = 0.5,
    f1_threshold: float = 0.8,
    include_agreed_negatives: bool = False,
    max_agreed_negatives: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stats = Counter()
    by_suite = Counter()
    by_role = Counter()
    disagreement_types = Counter()
    disagreement_suites = Counter()
    review_candidate_types = Counter()
    review_candidate_suites = Counter()
    review_candidates: list[dict[str, Any]] = []
    recoverable_positive = 0
    recoverable_negative = 0
    agreed_negative_candidates = 0

    for row in rows:
        reference = str(row.get("reference") or "")
        response = str(row.get("response") or "")
        if not reference or not response:
            stats["missing_reference_or_response"] += 1
            continue
        features = equivalence_features(reference, response)
        proxy = proxy_label(features, f1_threshold=f1_threshold)
        truth = _truth_label(row, threshold=target_threshold)
        stats["rows"] += 1
        by_suite[str(row.get("suite") or "unknown")] += 1
        by_role[str(row.get("role_key") or row.get("role") or "unknown")] += 1
        if proxy:
            recoverable_positive += 1
        else:
            recoverable_negative += 1
        if truth is None:
            stats["missing_target"] += 1
            continue
        if truth == proxy:
            stats["agreement"] += 1
            if truth == 0 and include_agreed_negatives:
                if max_agreed_negatives is None or agreed_negative_candidates < max_agreed_negatives:
                    review_candidates.append(_safe_row(row, truth=truth, proxy=proxy, features=features))
                    review_candidate_types["agreed_negative_not_equivalent"] += 1
                    review_candidate_suites[str(row.get("suite") or "unknown")] += 1
                    agreed_negative_candidates += 1
        else:
            stats["disagreement"] += 1
            dtype = (
                "current_positive_not_deterministically_reconstructable"
                if truth == 1 and proxy == 0
                else "current_negative_deterministically_equivalent"
            )
            disagreement_types[dtype] += 1
            disagreement_suites[str(row.get("suite") or "unknown")] += 1
            review_candidates.append(_safe_row(row, truth=truth, proxy=proxy, features=features))
            review_candidate_types[dtype] += 1
            review_candidate_suites[str(row.get("suite") or "unknown")] += 1

    total = int(stats["rows"])
    compared = int(stats["agreement"] + stats["disagreement"])
    summary = {
        "schema_version": "answer_equivalence_target_audit.v1",
        "status": "observation_not_decision",
        "target_threshold": target_threshold,
        "f1_threshold": f1_threshold,
        "counts": {
            "rows": total,
            "compared_rows": compared,
            "agreement": int(stats["agreement"]),
            "disagreement": int(stats["disagreement"]),
            "missing_target": int(stats["missing_target"]),
            "missing_reference_or_response": int(stats["missing_reference_or_response"]),
            "proxy_positive": recoverable_positive,
            "proxy_negative": recoverable_negative,
        },
        "rates": {
            "agreement": (stats["agreement"] / compared) if compared else None,
            "disagreement": (stats["disagreement"] / compared) if compared else None,
            "proxy_positive": recoverable_positive / total if total else None,
        },
        "by_suite": dict(sorted(by_suite.items())),
        "by_role": dict(sorted(by_role.items())),
        "disagreements": {
            "by_type": dict(sorted(disagreement_types.items())),
            "by_suite": dict(sorted(disagreement_suites.items())),
        },
        "review_candidates": {
            "rows": len(review_candidates),
            "included_agreed_negative": agreed_negative_candidates,
            "include_agreed_negatives": include_agreed_negatives,
            "max_agreed_negatives": max_agreed_negatives,
            "by_type": dict(sorted(review_candidate_types.items())),
            "by_suite": dict(sorted(review_candidate_suites.items())),
        },
        "interpretation": (
            "Deterministic answer-equivalence proxies are an audit target, not a "
            "semantic-judge replacement. Disagreement rows, plus any explicitly "
            "included agreed-negative rows, are the review set for the next A9 "
            "label-construction pass."
        ),
    }
    return summary, review_candidates


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def render_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    rates = summary["rates"]
    lines = [
        "# Answer-Equivalence Target Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rows: `{counts['rows']}`",
        f"- Compared rows: `{counts['compared_rows']}`",
        f"- Agreement: `{counts['agreement']}` ({_fmt_rate(rates['agreement'])})",
        f"- Disagreement: `{counts['disagreement']}` ({_fmt_rate(rates['disagreement'])})",
        f"- Proxy positives: `{counts['proxy_positive']}` ({_fmt_rate(rates['proxy_positive'])})",
        f"- F1 threshold: `{summary['f1_threshold']}`",
        f"- Review candidates: `{summary['review_candidates']['rows']}`",
        f"- Included agreed negatives: `{summary['review_candidates']['included_agreed_negative']}`",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Disagreements",
        "",
        "| Type | Rows |",
        "|---|---:|",
    ]
    for dtype, count in summary["disagreements"]["by_type"].items():
        lines.append(f"| `{dtype}` | {count} |")
    lines.extend([
        "",
        "## Suite Counts",
        "",
        "| Suite | Rows |",
        "|---|---:|",
    ])
    for suite, count in summary["by_suite"].items():
        lines.append(f"| `{suite}` | {count} |")
    return "\n".join(lines) + "\n"


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def assert_prompt_free(path: Path) -> None:
    for row_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            continue
        row = json.loads(raw)
        private = sorted(PRIVATE_FIELDS.intersection(row))
        if private:
            raise SystemExit(f"{path}:{row_number}: private fields present: {private}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--disagreements-jsonl", type=Path)
    parser.add_argument("--review-candidates-jsonl", type=Path)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--f1-threshold", type=float, default=0.8)
    parser.add_argument(
        "--include-agreed-negatives",
        action="store_true",
        help="Also export target-negative/proxy-negative rows as review candidates.",
    )
    parser.add_argument(
        "--max-agreed-negatives",
        type=int,
        help="Optional cap on exported agreed-negative review candidates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_jsonl = args.review_candidates_jsonl or args.disagreements_jsonl
    if output_jsonl is None:
        raise SystemExit("--review-candidates-jsonl or --disagreements-jsonl is required")
    rows = load_jsonl(args.input_jsonl)
    summary, disagreements = audit_rows(
        rows,
        target_threshold=args.target_threshold,
        f1_threshold=args.f1_threshold,
        include_agreed_negatives=args.include_agreed_negatives,
        max_agreed_negatives=args.max_agreed_negatives,
    )
    summary["input_jsonl"] = str(args.input_jsonl)
    write_json(args.summary_json, summary)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(render_markdown(summary), encoding="utf-8")
    write_jsonl(output_jsonl, disagreements)
    assert_prompt_free(output_jsonl)
    print(
        "answer-equivalence audit: "
        f"rows={summary['counts']['rows']} "
        f"disagreements={summary['counts']['disagreement']} "
        f"-> {args.summary_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
