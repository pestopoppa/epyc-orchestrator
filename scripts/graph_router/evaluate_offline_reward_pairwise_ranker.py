#!/usr/bin/env python3
"""Train/evaluate offline pairwise reward rankers from the A9 preference contract.

This is the next A9 step after the absolute verifier family stop condition. It
uses prompt-free pairwise preference rows, trains tabular pairwise rankers, and
reports held-out preference-prediction signal. It does not write runtime weights
or authorize any live routing gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.build_offline_reward_feature_manifest import PRIVATE_FIELDS
from scripts.graph_router.build_offline_reward_pairwise_contract import (
    CONTRACT_NAME,
    PAIRWISE_ROW_SCHEMA_VERSION,
)
from scripts.graph_router.train_verifier_head import _metrics

SUMMARY_SCHEMA_VERSION = "offline_reward_pairwise_ranker_eval.v1"
FEATURE_CONTRACT = "pairwise_action_response_delta_v1"
DEFAULT_SEEDS = [42, 7, 13, 101, 2026]
MODEL_FAMILIES = ("logistic_l2", "hist_gradient_boosting", "random_forest")
HOLDOUT_FIELDS = ("source_family", "suite")
TARGET_LEAKAGE_FIELDS = {"oracle_score_delta", "preferred_oracle_score", "rejected_oracle_score"}


class PairwiseRankerError(ValueError):
    """Raised when pairwise ranker inputs are invalid."""


@dataclass(frozen=True)
class Encoders:
    actions: tuple[str, ...]
    source_families: tuple[str, ...]
    suites: tuple[str, ...]

    @property
    def feature_names(self) -> list[str]:
        return (
            [f"action_delta[{action}]" for action in self.actions]
            + ["cross_action", "same_action"]
            + [f"source_family[{family}]" for family in self.source_families]
            + [f"suite[{suite}]" for suite in self.suites]
            + ["answer_chars_log_delta", "elapsed_log_delta", "error_present_delta"]
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise PairwiseRankerError(f"{path}:{line_number}: expected object")
            _validate_row(value, row_number=line_number)
            rows.append(value)
    if not rows:
        raise PairwiseRankerError(f"{path}: no pairwise rows")
    return rows


def _validate_row(row: dict[str, Any], *, row_number: int) -> None:
    if row.get("schema_version") != PAIRWISE_ROW_SCHEMA_VERSION:
        raise PairwiseRankerError(
            f"row {row_number}: expected schema_version={PAIRWISE_ROW_SCHEMA_VERSION!r}"
        )
    if row.get("contract_name") != CONTRACT_NAME:
        raise PairwiseRankerError(f"row {row_number}: expected contract_name={CONTRACT_NAME!r}")
    private_present = sorted(PRIVATE_FIELDS & set(row))
    if private_present:
        raise PairwiseRankerError(
            f"row {row_number}: private fields present: {', '.join(private_present)}"
        )
    for key in ("preferred_canonical_action", "rejected_canonical_action", "group_key"):
        if not isinstance(row.get(key), str) or not str(row[key]).strip():
            raise PairwiseRankerError(f"row {row_number}: {key} must be non-empty string")


def _float(row: dict[str, Any], key: str, *, default: float = 0.0) -> float:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_delta(preferred: Any, rejected: Any) -> float:
    return float(bool(preferred)) - float(bool(rejected))


def build_encoders(rows: Iterable[dict[str, Any]]) -> Encoders:
    row_list = list(rows)
    actions = sorted(
        {
            str(row["preferred_canonical_action"])
            for row in row_list
        }
        | {
            str(row["rejected_canonical_action"])
            for row in row_list
        }
    )
    source_families = sorted({str(row.get("source_family") or "unknown") for row in row_list})
    suites = sorted({str(row.get("suite") or "unknown") for row in row_list})
    return Encoders(tuple(actions), tuple(source_families), tuple(suites))


def _base_features(row: dict[str, Any], encoders: Encoders) -> np.ndarray:
    preferred_action = str(row["preferred_canonical_action"])
    rejected_action = str(row["rejected_canonical_action"])
    values: list[float] = []
    for action in encoders.actions:
        values.append(float(preferred_action == action) - float(rejected_action == action))
    cross_action = preferred_action != rejected_action
    values.extend([1.0 if cross_action else 0.0, 0.0 if cross_action else 1.0])
    source_family = str(row.get("source_family") or "unknown")
    values.extend(1.0 if source_family == family else 0.0 for family in encoders.source_families)
    suite = str(row.get("suite") or "unknown")
    values.extend(1.0 if suite == item else 0.0 for item in encoders.suites)
    values.extend(
        [
            _float(row, "answer_chars_log_delta"),
            _float(row, "elapsed_log_delta"),
            _bool_delta(row.get("preferred_error_present"), row.get("rejected_error_present")),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def build_symmetric_examples(
    rows: list[dict[str, Any]],
    encoders: Encoders,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        base = _base_features(row, encoders)
        positive_meta = _example_metadata(row, flipped=False)
        features.append(base)
        labels.append(1.0)
        metadata.append(positive_meta)
        features.append(_flip_feature_vector(base, encoders))
        labels.append(0.0)
        metadata.append(_example_metadata(row, flipped=True))
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=np.float32), metadata


def _flip_feature_vector(features: np.ndarray, encoders: Encoders) -> np.ndarray:
    flipped = features.copy()
    signed_indexes = list(range(len(encoders.actions)))
    signed_indexes.extend(
        [
            len(features) - 3,
            len(features) - 2,
            len(features) - 1,
        ]
    )
    flipped[signed_indexes] *= -1.0
    return flipped


def _example_metadata(row: dict[str, Any], *, flipped: bool) -> dict[str, Any]:
    if not flipped:
        preferred_action = row["preferred_canonical_action"]
        rejected_action = row["rejected_canonical_action"]
        preferred_item = row["preferred_item_id"]
        rejected_item = row["rejected_item_id"]
    else:
        preferred_action = row["rejected_canonical_action"]
        rejected_action = row["preferred_canonical_action"]
        preferred_item = row["rejected_item_id"]
        rejected_item = row["preferred_item_id"]
    return {
        "pair_id": row.get("pair_id"),
        "group_key": row.get("group_key"),
        "source_family": row.get("source_family") or "unknown",
        "suite": row.get("suite") or "unknown",
        "preferred_canonical_action": preferred_action,
        "rejected_canonical_action": rejected_action,
        "preferred_item_id": preferred_item,
        "rejected_item_id": rejected_item,
        "flipped": flipped,
    }


def split_group_keys(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    test_split: float,
) -> tuple[set[str], set[str]]:
    if not 0.0 < test_split < 1.0:
        raise PairwiseRankerError("test_split must be between 0 and 1")
    groups = sorted({str(row["group_key"]) for row in rows})
    if len(groups) < 2:
        raise PairwiseRankerError("at least two source-record groups are required")
    rng = np.random.default_rng(seed)
    indexes = np.arange(len(groups))
    rng.shuffle(indexes)
    n_test = max(1, int(round(len(groups) * test_split)))
    if n_test >= len(groups):
        n_test = len(groups) - 1
    test_groups = {groups[idx] for idx in indexes[:n_test].tolist()}
    train_groups = set(groups) - test_groups
    return train_groups, test_groups


def _rows_for_groups(rows: list[dict[str, Any]], groups: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["group_key"]) in groups]


def _split_rows_by_holdout(
    rows: list[dict[str, Any]],
    *,
    field: str,
    value: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_rows = [row for row in rows if str(row.get(field) or "unknown") == value]
    train_rows = [row for row in rows if str(row.get(field) or "unknown") != value]
    return train_rows, test_rows


def _model_for_family(family: str, seed: int) -> Any:
    if family == "logistic_l2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            l2_regularization=0.01,
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            random_state=seed,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    raise PairwiseRankerError(f"unsupported model family: {family}")


def _predict_positive(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = [int(value) for value in model.classes_.tolist()]
    if 1 not in classes:
        return np.zeros(x.shape[0], dtype=np.float32)
    return probabilities[:, classes.index(1)].astype(np.float32)


def _metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {"min": min(values), "max": max(values), "mean": float(np.mean(values))}


def _stratum_counts(metadata: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in metadata).items()))


def _random_baseline_metrics(labels: np.ndarray) -> dict[str, float]:
    metrics = _metrics(np.full_like(labels, 0.5, dtype=np.float32), labels)
    # The shared lightweight AUC helper is intentionally simple and does not
    # tie-correct constant scores. For the random pairwise reference, the
    # statistical AUC is exactly 0.5 on balanced symmetric examples.
    metrics["auc"] = 0.5
    return metrics


def _stratum_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    key: str,
    *,
    min_rows: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stratum in sorted({str(row.get(key) or "unknown") for row in metadata}):
        indexes = np.asarray(
            [idx for idx, row in enumerate(metadata) if str(row.get(key) or "unknown") == stratum],
            dtype=np.int64,
        )
        y = labels[indexes]
        p = probs[indexes]
        entry: dict[str, Any] = {
            "rows": int(indexes.shape[0]),
            "positive_rows": int(y.sum()),
            "negative_rows": int(indexes.shape[0] - int(y.sum())),
        }
        if indexes.shape[0] >= min_rows and len(set(int(value) for value in y.tolist())) == 2:
            entry["metrics"] = _metrics(p, y)
        else:
            entry["metrics"] = None
        out[stratum] = entry
    return out


def _evaluate_model_runs(
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    encoders: Encoders,
    seed: int,
    families: list[str],
    min_stratum_rows: int,
) -> list[dict[str, Any]]:
    x_train, y_train, _train_meta = build_symmetric_examples(train_rows, encoders)
    x_test, y_test, test_meta = build_symmetric_examples(test_rows, encoders)
    random_metrics = _random_baseline_metrics(y_test)
    runs: list[dict[str, Any]] = []
    for family in families:
        model = _model_for_family(family, seed)
        model.fit(x_train, y_train.astype(np.int64))
        probs = _predict_positive(model, x_test)
        runs.append(
            {
                "seed": seed,
                "family": family,
                "train_pair_rows": len(train_rows),
                "test_pair_rows": len(test_rows),
                "train_examples": int(x_train.shape[0]),
                "test_examples": int(x_test.shape[0]),
                "train_groups": len({str(row["group_key"]) for row in train_rows}),
                "test_groups": len({str(row["group_key"]) for row in test_rows}),
                "metrics": _metrics(probs, y_test),
                "random_baseline": random_metrics,
                "source_family_metrics": _stratum_metrics(
                    probs,
                    y_test,
                    test_meta,
                    "source_family",
                    min_rows=min_stratum_rows,
                ),
                "suite_metrics": _stratum_metrics(
                    probs,
                    y_test,
                    test_meta,
                    "suite",
                    min_rows=min_stratum_rows,
                ),
            }
        )
    return runs


def _leakage_policy() -> dict[str, Any]:
    return {
        "target_fields_excluded_from_features": sorted(TARGET_LEAKAGE_FIELDS),
        "private_fields_excluded": sorted(PRIVATE_FIELDS),
        "uses_prompt_answer_expected_text": False,
        "runtime_gate_change_allowed": False,
    }


def aggregate_runs(
    runs: list[dict[str, Any]],
    *,
    min_mean_accuracy: float,
    min_mean_auc: float,
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    blockers: list[str] = []
    for family in sorted({run["family"] for run in runs}):
        family_runs = [run for run in runs if run["family"] == family]
        stats = {
            "runs": len(family_runs),
            "accuracy": _metric_stats([run["metrics"]["acc"] for run in family_runs]),
            "auc": _metric_stats([run["metrics"]["auc"] for run in family_runs]),
            "brier": _metric_stats([run["metrics"]["brier"] for run in family_runs]),
            "ece": _metric_stats([run["metrics"]["ece"] for run in family_runs]),
            "acc_delta_vs_random": _metric_stats(
                [run["metrics"]["acc"] - run["random_baseline"]["acc"] for run in family_runs]
            ),
        }
        stats["meets_signal_gate"] = bool(
            stats["accuracy"]["mean"] >= min_mean_accuracy
            and stats["auc"]["mean"] >= min_mean_auc
        )
        if not stats["meets_signal_gate"]:
            blockers.append(f"{family}_mean_signal_below_threshold")
        families[family] = stats
    best_family = max(
        families,
        key=lambda family: (
            families[family]["accuracy"]["mean"],
            families[family]["auc"]["mean"],
            -families[family]["brier"]["mean"],
        ),
    )
    best_stats = families[best_family]
    status = "pairwise_ranker_signal" if best_stats["meets_signal_gate"] else "insufficient_pairwise_signal"
    return {
        "families": families,
        "criteria": {
            "min_mean_accuracy": min_mean_accuracy,
            "min_mean_auc": min_mean_auc,
        },
        "decision": {
            "status": status,
            "best_family": best_family,
            "blockers": [] if status == "pairwise_ranker_signal" else blockers,
            "runtime_gate_change_allowed": False,
            "recommended_next": (
                "cross_validate_pairwise_ranker_on_expanded_contract"
                if status == "pairwise_ranker_signal"
                else "collect_more_cross_action_pairwise_preferences"
            ),
        },
    }


def _holdout_candidates(
    rows: list[dict[str, Any]],
    *,
    field: str,
    min_holdout_pair_rows: int,
    min_train_pair_rows: int,
) -> list[str]:
    counts = Counter(str(row.get(field) or "unknown") for row in rows)
    candidates: list[str] = []
    for value, count in sorted(counts.items()):
        if count < min_holdout_pair_rows:
            continue
        if len(rows) - count < min_train_pair_rows:
            continue
        candidates.append(value)
    return candidates


def _evaluate_holdout_splits(
    rows: list[dict[str, Any]],
    encoders: Encoders,
    args: argparse.Namespace,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for field in args.holdout_fields:
        field_results: dict[str, Any] = {}
        for value in _holdout_candidates(
            rows,
            field=field,
            min_holdout_pair_rows=args.min_holdout_pair_rows,
            min_train_pair_rows=args.min_train_pair_rows,
        ):
            runs: list[dict[str, Any]] = []
            train_rows, test_rows = _split_rows_by_holdout(rows, field=field, value=value)
            for seed in args.seeds:
                runs.extend(
                    _evaluate_model_runs(
                        train_rows=train_rows,
                        test_rows=test_rows,
                        encoders=encoders,
                        seed=seed,
                        families=args.families,
                        min_stratum_rows=args.min_stratum_rows,
                    )
                )
            aggregate = aggregate_runs(
                runs,
                min_mean_accuracy=args.min_mean_accuracy,
                min_mean_auc=args.min_mean_auc,
            )
            field_results[value] = {
                "holdout_field": field,
                "holdout_value": value,
                "train_pair_rows": len(train_rows),
                "test_pair_rows": len(test_rows),
                "train_groups": len({str(row["group_key"]) for row in train_rows}),
                "test_groups": len({str(row["group_key"]) for row in test_rows}),
                "runs": runs,
                "aggregate": aggregate,
            }
        results[field] = {
            "eligible_holdout_values": sorted(field_results),
            "skipped_values": sorted(
                set(str(row.get(field) or "unknown") for row in rows) - set(field_results)
            ),
            "results": field_results,
        }
    return results


def _holdout_decision(holdout: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    eligible_count = 0
    passing_count = 0
    for field, payload in sorted(holdout.items()):
        for value, result in sorted(payload["results"].items()):
            eligible_count += 1
            status = result["aggregate"]["decision"]["status"]
            if status == "pairwise_ranker_signal":
                passing_count += 1
            else:
                blockers.append(f"{field}:{value}:{status}")
    if eligible_count == 0:
        status = "no_eligible_holdouts"
    elif blockers:
        status = "mixed_holdout_signal"
    else:
        status = "holdout_signal_consistent"
    return {
        "status": status,
        "eligible_holdouts": eligible_count,
        "passing_holdouts": passing_count,
        "blockers": blockers,
        "runtime_gate_change_allowed": False,
        "recommended_next": (
            "collect_more_non_overlapping_cross_action_preferences"
            if blockers or eligible_count == 0
            else "preregister_downstream_pairwise_reward_use"
        ),
    }


def run_pairwise_ranker_eval(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(Path(args.pairwise_jsonl))
    encoders = build_encoders(rows)
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        train_groups, test_groups = split_group_keys(rows, seed=seed, test_split=args.test_split)
        train_rows = _rows_for_groups(rows, train_groups)
        test_rows = _rows_for_groups(rows, test_groups)
        runs.extend(
            _evaluate_model_runs(
                train_rows=train_rows,
                test_rows=test_rows,
                encoders=encoders,
                seed=seed,
                families=args.families,
                min_stratum_rows=args.min_stratum_rows,
            )
        )
    source_family_counts = Counter(str(row.get("source_family") or "unknown") for row in rows)
    suite_counts = Counter(str(row.get("suite") or "unknown") for row in rows)
    pairing_mode_counts = Counter(str(row.get("pairing_mode") or "unknown") for row in rows)
    action_pair_counts = Counter(
        f"{row['preferred_canonical_action']}>{row['rejected_canonical_action']}"
        for row in rows
    )
    cross_action_rows = sum(
        count for pair, count in action_pair_counts.items() if pair.split(">", 1)[0] != pair.split(">", 1)[1]
    )
    holdout = _evaluate_holdout_splits(rows, encoders, args)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "pairwise_jsonl": str(args.pairwise_jsonl),
        "feature_contract": {
            "name": FEATURE_CONTRACT,
            "feature_names": encoders.feature_names,
            "feature_dim": len(encoders.feature_names),
            "symmetric_augmentation": True,
        },
        "input": {
            "pair_rows": len(rows),
            "cross_action_pair_rows": cross_action_rows,
            "same_action_pair_rows": len(rows) - cross_action_rows,
            "group_count": len({str(row["group_key"]) for row in rows}),
            "source_family_pair_counts": dict(sorted(source_family_counts.items())),
            "suite_pair_counts": dict(sorted(suite_counts.items())),
            "pairing_mode_counts": dict(sorted(pairing_mode_counts.items())),
            "action_pair_counts": dict(sorted(action_pair_counts.items())),
        },
        "split": {
            "test_split": args.test_split,
            "group_disjoint": True,
        },
        "families_requested": args.families,
        "seeds": args.seeds,
        "leakage_policy": _leakage_policy(),
        "runs": runs,
        "aggregate": aggregate_runs(
            runs,
            min_mean_accuracy=args.min_mean_accuracy,
            min_mean_auc=args.min_mean_auc,
        ),
        "independent_holdout": holdout,
        "holdout_decision": _holdout_decision(holdout),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    decision = summary["aggregate"]["decision"]
    lines = [
        "# Offline Reward Pairwise Ranker Eval",
        "",
        f"- Pairwise JSONL: `{summary['pairwise_jsonl']}`",
        f"- Feature contract: `{summary['feature_contract']['name']}`",
        f"- Pair rows: `{summary['input']['pair_rows']}`",
        f"- Cross-action pair rows: `{summary['input']['cross_action_pair_rows']}`",
        f"- Same-action pair rows: `{summary['input']['same_action_pair_rows']}`",
        f"- Group count: `{summary['input']['group_count']}`",
        f"- Pairing mode counts: `{summary['input']['pairing_mode_counts']}`",
        f"- Families: `{summary['families_requested']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Decision: `{decision['status']}`",
        f"- Best family: `{decision['best_family']}`",
        f"- Runtime gate change allowed: `{decision['runtime_gate_change_allowed']}`",
        f"- Recommended next: `{decision['recommended_next']}`",
        "",
        "## Family Summary",
        "",
    ]
    for family, stats in summary["aggregate"]["families"].items():
        lines.append(
            f"- `{family}`: acc mean `{stats['accuracy']['mean']:.4f}`, "
            f"AUC mean `{stats['auc']['mean']:.4f}`, "
            f"Brier mean `{stats['brier']['mean']:.4f}`, "
            f"ECE mean `{stats['ece']['mean']:.4f}`, "
            f"acc delta vs random `{stats['acc_delta_vs_random']['mean']:.4f}`"
        )
    holdout = summary.get("independent_holdout") or {}
    if holdout:
        lines.extend(["", "## Independent Holdout Summary", ""])
        decision = summary.get("holdout_decision") or {}
        if decision:
            lines.extend(
                [
                    f"- Holdout decision: `{decision['status']}`",
                    f"- Passing holdouts: `{decision['passing_holdouts']}/{decision['eligible_holdouts']}`",
                    f"- Runtime gate change allowed: `{decision['runtime_gate_change_allowed']}`",
                    f"- Recommended next: `{decision['recommended_next']}`",
                    "",
                ]
            )
        for field, payload in holdout.items():
            lines.append(f"### `{field}`")
            if not payload["results"]:
                lines.append("- no eligible holdout values")
                continue
            for value, result in payload["results"].items():
                split_decision = result["aggregate"]["decision"]
                best_family = split_decision["best_family"]
                best_stats = result["aggregate"]["families"][best_family]
                lines.append(
                    f"- `{value}`: decision `{split_decision['status']}`, "
                    f"best `{best_family}`, acc mean `{best_stats['accuracy']['mean']:.4f}`, "
                    f"AUC mean `{best_stats['auc']['mean']:.4f}`, "
                    f"test pairs `{result['test_pair_rows']}`"
                )
    lines.extend(
        [
            "",
            "## Leakage Controls",
            "",
        ]
    )
    for key, value in summary["leakage_policy"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "This is an offline diagnostic artifact. It does not write runtime",
            "ranker weights or enable a live routing gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_csv(value: str, *, allowed: tuple[str, ...] | None = None) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one value")
    if allowed is not None:
        invalid = sorted(set(items) - set(allowed))
        if invalid:
            raise argparse.ArgumentTypeError(f"unsupported value(s): {', '.join(invalid)}")
    return items


def _parse_csv_ints(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate offline pairwise reward rankers from A9 preference rows."
    )
    parser.add_argument("--pairwise-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument(
        "--families",
        type=lambda value: _parse_csv(value, allowed=MODEL_FAMILIES),
        default=list(MODEL_FAMILIES),
    )
    parser.add_argument("--seeds", type=_parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--test-split", type=float, default=0.25)
    parser.add_argument("--min-mean-accuracy", type=float, default=0.60)
    parser.add_argument("--min-mean-auc", type=float, default=0.60)
    parser.add_argument("--min-stratum-rows", type=int, default=10)
    parser.add_argument(
        "--holdout-fields",
        type=lambda value: _parse_csv(value, allowed=HOLDOUT_FIELDS),
        default=list(HOLDOUT_FIELDS),
    )
    parser.add_argument("--min-holdout-pair-rows", type=int, default=20)
    parser.add_argument("--min-train-pair-rows", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_pairwise_ranker_eval(args)
    except (OSError, PairwiseRankerError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(render_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
