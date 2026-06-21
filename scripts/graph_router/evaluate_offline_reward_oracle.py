#!/usr/bin/env python3
"""Evaluate an offline reference-grounded reward oracle.

This is the gate harness for the learned-routing NEXT-A2/A3 offline quality
oracle lane. It does not score model outputs itself; it evaluates already
produced oracle scores against existing binary/graded labels and mandatory
paraphrase/synonym stress rows.

Expected JSONL fields per row:
  - item_id: stable row id, optional
  - reference: reference answer or rubric text
  - response: candidate response
  - oracle_score: scalar oracle score, normally 0..1
  - one target field: binary_reward, q_reward, target_score, score, or outcome
  - optional variant_group: groups verbatim/paraphrase/synonym variants
  - optional variant_type: base, verbatim, paraphrase, synonym, confound, ...

The output is observations only. Promotion/enablement still belongs to the
owning handoff and measurement policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TARGET_FIELDS = ("binary_reward", "q_reward", "target_score", "score")
SUCCESS_OUTCOMES = {"success", "passed", "pass", "ok", "correct", "true"}
PARTIAL_OUTCOMES = {"partial", "mixed", "incomplete"}
FAILURE_OUTCOMES = {"failure", "failed", "fail", "incorrect", "false", "error"}
BASE_VARIANTS = {"base", "verbatim", "original", "reference"}
PARAPHRASE_VARIANTS = {"paraphrase", "synonym", "synonym_swap", "reworded"}
CONFOUND_VARIANTS = {"confound", "decoy", "plausible_wrong"}


@dataclass(frozen=True)
class OracleRow:
    item_id: str
    reference: str
    response: str
    oracle_score: float
    target_score: float
    binary_target: int
    target_source: str = "unspecified"
    suite: str = "unknown"
    role_key: str = "unknown"
    variant_group: str | None = None
    variant_type: str | None = None


def _is_finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def _as_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not _is_finite(parsed):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return parsed


def _target_from_outcome(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in SUCCESS_OUTCOMES:
        return 1.0
    if normalized in PARTIAL_OUTCOMES:
        return 0.3
    if normalized in FAILURE_OUTCOMES:
        return 0.0
    return None


def _target_score(raw: dict[str, Any]) -> float:
    for field in TARGET_FIELDS:
        if field in raw and raw[field] is not None:
            return _as_float(raw[field], field=field)
    outcome_score = _target_from_outcome(raw.get("outcome"))
    if outcome_score is not None:
        return outcome_score
    raise ValueError(
        "row needs one target field: binary_reward, q_reward, target_score, "
        "score, or recognized outcome"
    )


def parse_row(
    raw: dict[str, Any],
    *,
    row_number: int,
    target_threshold: float,
) -> OracleRow:
    if not isinstance(raw, dict):
        raise ValueError("row must be an object")
    oracle_score = _as_float(raw.get("oracle_score"), field="oracle_score")
    target_score = _target_score(raw)
    reference = str(raw.get("reference", ""))
    response = str(raw.get("response", ""))
    if not reference:
        raise ValueError("reference is required")
    if not response:
        raise ValueError("response is required")

    item_id = str(raw.get("item_id") or raw.get("id") or f"row-{row_number}")
    variant_group = raw.get("variant_group")
    variant_type = raw.get("variant_type")
    return OracleRow(
        item_id=item_id,
        reference=reference,
        response=response,
        oracle_score=oracle_score,
        target_score=target_score,
        binary_target=1 if target_score >= target_threshold else 0,
        target_source=str(raw.get("target_source") or "unspecified"),
        suite=str(raw.get("suite") or "unknown"),
        role_key=str(raw.get("role_key") or raw.get("role") or "unknown"),
        variant_group=str(variant_group) if variant_group else None,
        variant_type=str(variant_type).strip().lower() if variant_type else None,
    )


def load_jsonl(path: Path, *, target_threshold: float) -> list[OracleRow]:
    rows: list[OracleRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                rows.append(
                    parse_row(
                        raw,
                        row_number=row_number,
                        target_threshold=target_threshold,
                    )
                )
            except Exception as exc:
                raise SystemExit(f"{path}:{row_number}: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0 + 1.0
        for idx in order[cursor:end]:
            ranks[idx] = average_rank
        cursor = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(_average_ranks(xs), _average_ranks(ys))


def _confusion(
    rows: Iterable[OracleRow],
    *,
    oracle_threshold: float,
) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        pred = 1 if row.oracle_score >= oracle_threshold else 0
        truth = row.binary_target
        if pred == 1 and truth == 1:
            counts["tp"] += 1
        elif pred == 1 and truth == 0:
            counts["fp"] += 1
        elif pred == 0 and truth == 1:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return {key: int(counts[key]) for key in ("tp", "fp", "fn", "tn")}


def _agreement_from_confusion(confusion: dict[str, int]) -> float | None:
    total = sum(confusion.values())
    if total == 0:
        return None
    return (confusion["tp"] + confusion["tn"]) / total


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _threshold_metrics(
    rows: Iterable[OracleRow],
    *,
    oracle_threshold: float,
) -> dict[str, Any]:
    confusion = _confusion(rows, oracle_threshold=oracle_threshold)
    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    tn = confusion["tn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = (recall + specificity) / 2
    return {
        "threshold": oracle_threshold,
        "agreement": _agreement_from_confusion(confusion),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "predicted_positive": tp + fp,
        "predicted_negative": tn + fn,
        "confusion": confusion,
    }


def _metric_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        return -1.0
    return float(value)


def _best_threshold(
    sweep: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any] | None:
    if not sweep:
        return None
    return max(
        sweep,
        key=lambda row: (
            _metric_value(row, metric),
            row["confusion"]["tp"],
            -row["confusion"]["fp"],
            -row["threshold"],
        ),
    )


def _best_no_false_positive_threshold(
    sweep: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [row for row in sweep if row["confusion"]["fp"] == 0]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["confusion"]["tp"],
            _metric_value(row, "recall"),
            _metric_value(row, "agreement"),
            -row["threshold"],
        ),
    )


def _threshold_sweep(
    rows: list[OracleRow],
    *,
    include_threshold: float,
) -> dict[str, Any]:
    thresholds = {round(step / 100, 2) for step in range(101)}
    thresholds.add(round(include_threshold, 6))
    sweep = [
        _threshold_metrics(rows, oracle_threshold=threshold)
        for threshold in sorted(thresholds)
    ]
    return {
        "schema_version": "offline_reward_oracle_calibration.v1",
        "threshold_step": 0.01,
        "threshold_count": len(sweep),
        "best": {
            "agreement": _best_threshold(sweep, metric="agreement"),
            "balanced_accuracy": _best_threshold(sweep, metric="balanced_accuracy"),
            "f1": _best_threshold(sweep, metric="f1"),
            "no_false_positive": _best_no_false_positive_threshold(sweep),
        },
        "threshold_sweep": sweep,
    }


def _stress_summary(
    rows: list[OracleRow],
    *,
    oracle_threshold: float,
) -> dict[str, Any]:
    by_group: dict[str, list[OracleRow]] = defaultdict(list)
    for row in rows:
        if row.variant_group:
            by_group[row.variant_group].append(row)

    groups: list[dict[str, Any]] = []
    variant_counts = Counter(row.variant_type or "unknown" for row in rows)
    paraphrase_penalized = 0
    paraphrase_total = 0
    confound_fooled = 0
    confound_total = 0

    for group, group_rows in sorted(by_group.items()):
        base_rows = [r for r in group_rows if (r.variant_type or "") in BASE_VARIANTS]
        if not base_rows:
            continue
        base_score = sum(r.oracle_score for r in base_rows) / len(base_rows)
        variants: dict[str, dict[str, Any]] = {}
        for variant_type in sorted({r.variant_type or "unknown" for r in group_rows}):
            variant_rows = [r for r in group_rows if (r.variant_type or "unknown") == variant_type]
            mean_score = sum(r.oracle_score for r in variant_rows) / len(variant_rows)
            mean_target = sum(r.target_score for r in variant_rows) / len(variant_rows)
            delta = mean_score - base_score
            variants[variant_type] = {
                "n": len(variant_rows),
                "mean_oracle_score": mean_score,
                "mean_target_score": mean_target,
                "delta_vs_base": delta,
            }
            if variant_type in PARAPHRASE_VARIANTS:
                paraphrase_total += len(variant_rows)
                paraphrase_penalized += sum(1 for r in variant_rows if r.binary_target == 1 and r.oracle_score < base_score - 0.1)
            if variant_type in CONFOUND_VARIANTS:
                confound_total += len(variant_rows)
                confound_fooled += sum(
                    1
                    for r in variant_rows
                    if r.binary_target == 0 and r.oracle_score >= oracle_threshold
                )
        groups.append(
            {
                "variant_group": group,
                "base_score": base_score,
                "variants": variants,
            }
        )

    return {
        "variant_counts": dict(sorted(variant_counts.items())),
        "groups_evaluated": len(groups),
        "paraphrase_penalty_rate": (
            paraphrase_penalized / paraphrase_total if paraphrase_total else None
        ),
        "paraphrase_penalized": paraphrase_penalized,
        "paraphrase_total": paraphrase_total,
        "confound_fooled_rate": confound_fooled / confound_total if confound_total else None,
        "confound_fooled": confound_fooled,
        "confound_total": confound_total,
        "groups": groups,
    }


def _slice_summary(
    rows: list[OracleRow],
    *,
    key: str,
    oracle_threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[OracleRow]] = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, key) or "unknown")].append(row)

    summary: dict[str, Any] = {}
    for value, group_rows in sorted(grouped.items()):
        scores = [row.oracle_score for row in group_rows]
        targets = [row.target_score for row in group_rows]
        confusion = _confusion(group_rows, oracle_threshold=oracle_threshold)
        class_counts = Counter(row.binary_target for row in group_rows)
        mean_abs_error = sum(
            abs(row.oracle_score - row.target_score) for row in group_rows
        ) / len(group_rows)
        summary[value] = {
            "n": len(group_rows),
            "target_positive": int(class_counts[1]),
            "target_negative": int(class_counts[0]),
            "spearman": spearman(scores, targets),
            "pearson": pearson(scores, targets),
            "mean_abs_error": mean_abs_error,
            "agreement_at_threshold": _agreement_from_confusion(confusion),
            "confusion": confusion,
        }
    return summary


def evaluate(
    rows: list[OracleRow],
    *,
    oracle_threshold: float = 0.5,
) -> dict[str, Any]:
    scores = [row.oracle_score for row in rows]
    targets = [row.target_score for row in rows]
    confusion = _confusion(rows, oracle_threshold=oracle_threshold)
    class_counts = Counter(row.binary_target for row in rows)
    mean_abs_error = sum(abs(row.oracle_score - row.target_score) for row in rows) / len(rows)
    return {
        "schema_version": "offline_reward_oracle_eval.v1",
        "status": "observation_not_decision",
        "n": len(rows),
        "target_positive": int(class_counts[1]),
        "target_negative": int(class_counts[0]),
        "oracle_threshold": oracle_threshold,
        "score": {
            "spearman": spearman(scores, targets),
            "pearson": pearson(scores, targets),
            "mean_abs_error": mean_abs_error,
            "agreement_at_threshold": _agreement_from_confusion(confusion),
            "confusion": confusion,
        },
        "calibration": _threshold_sweep(rows, include_threshold=oracle_threshold),
        "stress": _stress_summary(rows, oracle_threshold=oracle_threshold),
        "slices": {
            "target_source": _slice_summary(
                rows,
                key="target_source",
                oracle_threshold=oracle_threshold,
            ),
            "suite": _slice_summary(rows, key="suite", oracle_threshold=oracle_threshold),
            "role_key": _slice_summary(
                rows,
                key="role_key",
                oracle_threshold=oracle_threshold,
            ),
        },
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    score = summary["score"]
    stress = summary["stress"]
    lines = [
        "# Offline Reward-Oracle Evaluation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rows: {summary['n']}",
        f"- Binary target positives: {summary['target_positive']}",
        f"- Binary target negatives: {summary['target_negative']}",
        f"- Oracle threshold: {summary['oracle_threshold']}",
        "",
        "## Score Metrics",
        "",
        f"- Spearman vs target: {_fmt(score['spearman'])}",
        f"- Pearson vs target: {_fmt(score['pearson'])}",
        f"- Mean absolute error: {_fmt(score['mean_abs_error'])}",
        f"- Agreement at threshold: {_fmt(score['agreement_at_threshold'])}",
        f"- Confusion: `tp={score['confusion']['tp']} fp={score['confusion']['fp']} "
        f"fn={score['confusion']['fn']} tn={score['confusion']['tn']}`",
        "",
        "## Calibration",
        "",
        *_calibration_markdown_lines(summary["calibration"]),
        "",
        "## Stress Metrics",
        "",
        f"- Groups evaluated: {stress['groups_evaluated']}",
        f"- Variant counts: `{json.dumps(stress['variant_counts'], sort_keys=True)}`",
        f"- Paraphrase penalty rate: {_fmt(stress['paraphrase_penalty_rate'])} "
        f"({stress['paraphrase_penalized']}/{stress['paraphrase_total']})",
        f"- Confound fooled rate: {_fmt(stress['confound_fooled_rate'])} "
        f"({stress['confound_fooled']}/{stress['confound_total']})",
        "",
        "## Slices",
        "",
        *_slice_markdown_lines("Target source", summary["slices"]["target_source"]),
        "",
        *_slice_markdown_lines("Suite", summary["slices"]["suite"]),
        "",
        *_slice_markdown_lines("Role", summary["slices"]["role_key"]),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _calibration_markdown_lines(calibration: dict[str, Any]) -> list[str]:
    best = calibration["best"]
    lines = [
        f"- Thresholds evaluated: {calibration['threshold_count']}",
    ]
    for label, key in (
        ("Best F1", "f1"),
        ("Best balanced accuracy", "balanced_accuracy"),
        ("Best agreement", "agreement"),
        ("Best no-false-positive recall", "no_false_positive"),
    ):
        row = best.get(key)
        if not row:
            lines.append(f"- {label}: `null`")
            continue
        confusion = row["confusion"]
        lines.append(
            f"- {label}: threshold `{row['threshold']:.2f}`, "
            f"agreement {_fmt(row['agreement'])}, recall {_fmt(row['recall'])}, "
            f"precision {_fmt(row['precision'])}, "
            f"tp={confusion['tp']} fp={confusion['fp']} "
            f"fn={confusion['fn']} tn={confusion['tn']}"
        )
    return lines


def _slice_markdown_lines(title: str, rows: dict[str, Any]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in rows.items():
        confusion = row["confusion"]
        lines.append(
            f"| `{name}` | {row['n']} | {row['target_positive']} | "
            f"{row['target_negative']} | {_fmt(row['agreement_at_threshold'])} | "
            f"{_fmt(row['spearman'])} | "
            f"`tp={confusion['tp']} fp={confusion['fp']} "
            f"fn={confusion['fn']} tn={confusion['tn']}` |"
        )
    return lines


def _fmt(value: float | None) -> str:
    if value is None:
        return "`null`"
    return f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate offline reward-oracle scores against q_reward labels",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL rows")
    parser.add_argument("--output-json", required=True, type=Path, help="Summary JSON path")
    parser.add_argument("--output-md", type=Path, help="Optional Markdown summary path")
    parser.add_argument(
        "--oracle-threshold",
        type=float,
        default=0.5,
        help="Threshold for oracle_score -> positive agreement label",
    )
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=0.5,
        help="Threshold for target_score -> binary target",
    )
    args = parser.parse_args(argv)

    rows = load_jsonl(args.input, target_threshold=args.target_threshold)
    summary = evaluate(rows, oracle_threshold=args.oracle_threshold)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        write_markdown(summary, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
