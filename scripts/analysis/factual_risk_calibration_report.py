#!/usr/bin/env python3
"""Deterministic factual-risk calibration report/aggregator.

Reads the v2 calibration corpus plus optional result JSONL files or result
directories, then emits a machine-readable summary with counts and basic
cross-tabs. The script performs no inference.

Default input:
    orchestration/factual_risk_calibration_v2.jsonl

Optional result inputs:
    - explicit ``--result-path`` arguments
    - auto-discovered ``g10`` / ``g11`` paths under ``orchestration/``

Output:
    JSON summary on stdout, or ``--output`` if supplied.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ORCH_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_INPUT = ORCH_ROOT / "orchestration" / "factual_risk_calibration_v2.jsonl"
DEFAULT_SPLITS = ("train", "val", "test")
RESULT_NAME_HINTS = ("g10", "g11")
DEFAULT_EXPECTED_ROLES = ("architect_general", "frontdoor", "worker_general")
DEFAULT_ROLE_TO_TIER = {
    "architect_general": "tier_1",
    "frontdoor": "tier_2",
    "worker_general": "tier_3",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except FileNotFoundError:
        return rows
    return rows


def _sorted_dict(counter: Counter[Any] | dict[Any, Any]) -> dict[str, Any]:
    items = counter.items() if isinstance(counter, Counter) else counter.items()
    return {str(key): value for key, value in sorted(items, key=lambda item: str(item[0]))}


def _normalize_outcome(record: dict[str, Any]) -> str:
    if "outcome" in record and record["outcome"] is not None:
        return str(record["outcome"])
    if "result" in record and record["result"] is not None:
        return str(record["result"])
    if "status" in record and record["status"] is not None:
        return str(record["status"])
    passed = record.get("passed")
    if isinstance(passed, bool):
        return "passed" if passed else "failed"
    label = record.get("label_4class")
    if isinstance(label, str) and label:
        return label
    return "unknown"


def _normalize_tier(record: dict[str, Any]) -> str | None:
    tier = record.get("tier")
    if tier is None:
        return None
    return str(tier)


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _outcome_metrics(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    correct = counts.get("CORRECT", 0)
    incorrect = counts.get("INCORRECT", 0)
    partial = counts.get("PARTIAL_ANSWER", 0) + counts.get("PARTIAL", 0)
    not_attempted = counts.get("NOT_ATTEMPTED", 0)
    answered_denominator = incorrect + partial + not_attempted
    return {
        "accuracy": _round_or_none(_rate(correct, total)),
        "hallucination_rate": _round_or_none(_rate(incorrect, answered_denominator)),
        "not_attempted_rate": _round_or_none(_rate(not_attempted, total)),
        "partial_rate": _round_or_none(_rate(partial, total)),
        "total": total,
    }


def _load_dataset(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    label_counts = Counter()
    source_counts = Counter()
    tier_counts = Counter()
    tier_source = defaultdict(Counter)
    field_presence = Counter()

    for row in rows:
        for field in ("label_source", "label_4class", "tier", "risk_band_v1", "risk_features", "risk_score_computed"):
            if field in row and row[field] is not None:
                field_presence[field] += 1

        source = row.get("label_source") or row.get("source") or "unknown"
        label = row.get("label_4class") or row.get("risk_label") or "unknown"
        tier = _normalize_tier(row)

        source_counts[str(source)] += 1
        label_counts[str(label)] += 1
        if tier is not None:
            tier_counts[tier] += 1
            tier_source[tier][str(source)] += 1

    return {
        "path": str(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "field_presence": _sorted_dict(field_presence),
        "source_counts": _sorted_dict(source_counts),
        "risk_label_counts": _sorted_dict(label_counts),
        "tier_counts": _sorted_dict(tier_counts),
        "tier_source_crosstab": {
            tier: _sorted_dict(counts) for tier, counts in sorted(tier_source.items(), key=lambda item: str(item[0]))
        },
    }


def _load_split_counts(dataset_path: Path) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in DEFAULT_SPLITS:
        split_path = dataset_path.with_name(f"{dataset_path.stem}_{split}.jsonl")
        splits[split] = {
            "path": str(split_path),
            "exists": split_path.exists(),
            "row_count": len(_read_jsonl(split_path)),
        }
    return splits


def _discover_result_paths(base_dirs: tuple[Path, ...] | None = None) -> list[Path]:
    if base_dirs is None:
        base_dirs = (
            ORCH_ROOT / "orchestration",
            ORCH_ROOT / "orchestration" / "reports",
        )
    candidates: list[Path] = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.iterdir(), key=lambda p: p.name):
            name = path.name.lower()
            if any(hint in name for hint in RESULT_NAME_HINTS):
                candidates.append(path)
    return candidates


def _iter_result_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*.jsonl"), key=lambda p: str(p)):
            if child.is_file():
                yield child


def _aggregate_results(paths: list[Path]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    by_role_model: dict[str, Counter[str]] = defaultdict(Counter)
    role_sources: dict[str, Counter[str]] = defaultdict(Counter)
    model_sources: dict[str, Counter[str]] = defaultdict(Counter)

    for source_path in paths:
        for file_path in _iter_result_files(source_path):
            rows = _read_jsonl(file_path)
            files.append({
                "path": str(file_path),
                "row_count": len(rows),
            })
            for row in rows:
                role = str(row.get("role") or row.get("agent_role") or row.get("worker_role") or "unknown")
                model = str(row.get("model") or row.get("model_name") or row.get("engine") or "unknown")
                source = str(row.get("source") or row.get("label_source") or row.get("suite") or "unknown")
                outcome = _normalize_outcome(row)

                by_role[role][outcome] += 1
                by_model[model][outcome] += 1
                by_role_model[f"{role}::{model}"][outcome] += 1
                role_sources[role][source] += 1
                model_sources[model][source] += 1

    return {
        "enabled": bool(files),
        "files": files,
        "by_role": {
            key: {
                "total": sum(counts.values()),
                "outcomes": _sorted_dict(counts),
                "metrics": _outcome_metrics(counts),
                "source_counts": _sorted_dict(role_sources[key]),
            }
            for key, counts in sorted(by_role.items(), key=lambda item: item[0])
        },
        "by_model": {
            key: {
                "total": sum(counts.values()),
                "outcomes": _sorted_dict(counts),
                "metrics": _outcome_metrics(counts),
                "source_counts": _sorted_dict(model_sources[key]),
            }
            for key, counts in sorted(by_model.items(), key=lambda item: item[0])
        },
        "by_role_model": {
            key: {
                "total": sum(counts.values()),
                "outcomes": _sorted_dict(counts),
                "metrics": _outcome_metrics(counts),
            }
            for key, counts in sorted(by_role_model.items(), key=lambda item: item[0])
        },
    }


def _build_tier_calibration_readiness(
    results: dict[str, Any],
    expected_roles: tuple[str, ...],
    role_to_tier: dict[str, str] | None = None,
) -> dict[str, Any]:
    role_to_tier = role_to_tier or DEFAULT_ROLE_TO_TIER
    by_role = results.get("by_role", {})
    observed_roles = tuple(sorted(role for role in by_role if role in expected_roles))
    missing_roles = tuple(role for role in expected_roles if role not in by_role)

    role_metrics: dict[str, Any] = {}
    hallucination_rates: dict[str, float] = {}
    for role in observed_roles:
        metrics = dict(by_role[role].get("metrics", {}))
        metrics["tier"] = role_to_tier.get(role)
        role_metrics[role] = metrics
        rate = metrics.get("hallucination_rate")
        if isinstance(rate, (float, int)):
            hallucination_rates[role] = float(rate)

    worst_hallucination_rate = max(hallucination_rates.values()) if hallucination_rates else None
    role_multiplier_preview: dict[str, float | None] = {}
    if worst_hallucination_rate and worst_hallucination_rate > 0:
        for role, rate in sorted(hallucination_rates.items()):
            role_multiplier_preview[role] = round(rate / worst_hallucination_rate, 6)
    else:
        for role in observed_roles:
            role_multiplier_preview[role] = None

    tier_values: dict[str, list[float]] = defaultdict(list)
    for role, multiplier in role_multiplier_preview.items():
        tier = role_to_tier.get(role)
        if tier and multiplier is not None:
            tier_values[tier].append(multiplier)

    tier_multiplier_preview = {
        tier: round(sum(values) / len(values), 6)
        for tier, values in sorted(tier_values.items())
        if values
    }

    complete = not missing_roles and set(expected_roles).issubset(set(role_metrics))
    return {
        "complete": complete,
        "status": "ready_for_tier_update" if complete else "blocked_missing_roles",
        "expected_roles": list(expected_roles),
        "observed_roles": list(observed_roles),
        "missing_roles": list(missing_roles),
        "role_metrics": role_metrics,
        "worst_observed_hallucination_rate": _round_or_none(worst_hallucination_rate),
        "role_multiplier_preview_vs_worst": role_multiplier_preview,
        "tier_multiplier_preview_vs_worst": tier_multiplier_preview if complete else {},
        "note": (
            "Preview only: do not update factual-risk role tiers until all expected roles "
            "are present and the deterministic-vs-LLM-judge scoring decision is resolved."
        ),
    }


def build_report(
    dataset_path: Path = DEFAULT_INPUT,
    result_paths: list[Path] | None = None,
    auto_discover_results: bool = True,
    expected_roles: tuple[str, ...] = DEFAULT_EXPECTED_ROLES,
) -> dict[str, Any]:
    dataset = _load_dataset(dataset_path)
    splits = _load_split_counts(dataset_path)

    explicit_result_paths = result_paths or []
    discovered_result_paths = _discover_result_paths() if auto_discover_results else []
    ordered_result_paths: list[Path] = []
    seen: set[str] = set()
    for path in [*explicit_result_paths, *discovered_result_paths]:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered_result_paths.append(path)

    results = _aggregate_results(ordered_result_paths)

    return {
        "dataset": dataset,
        "splits": splits,
        "results": results,
        "tier_calibration_readiness": _build_tier_calibration_readiness(results, expected_roles),
    }


def _parse_paths(values: list[str] | None) -> list[Path]:
    if not values:
        return []
    return [Path(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--result-path",
        dest="result_paths",
        action="append",
        default=[],
        help="Optional result JSONL file or directory to aggregate. May be repeated.",
    )
    parser.add_argument(
        "--no-auto-discover-results",
        action="store_true",
        help="Disable g10/g11 result path discovery under orchestration/.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--expected-role",
        dest="expected_roles",
        action="append",
        default=[],
        help="Role required before tier calibration can be marked complete. May be repeated.",
    )
    args = parser.parse_args()

    result_paths = _parse_paths(args.result_paths)
    expected_roles = tuple(args.expected_roles) if args.expected_roles else DEFAULT_EXPECTED_ROLES
    report = build_report(
        args.input,
        result_paths,
        auto_discover_results=not args.no_auto_discover_results,
        expected_roles=expected_roles,
    )

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
