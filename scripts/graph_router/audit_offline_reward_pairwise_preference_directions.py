#!/usr/bin/env python3
"""Audit action-pair preference directions and source/suite×action intersections
in A9 offline-reward pairwise-preference contracts.

OFFLINE, READ-ONLY, EVIDENCE-ONLY. This loads a pairwise-preference contract
(`offline_reward_pairwise_preference.v1` rows) and emits a per-stratum breakdown
of *why* the pairwise ranker fails to generalize to particular independent
holdouts (e.g. `source_family:seeding_eval`, `suite:thinking`): thin action-pair
coverage, one-sided preference directions, and covariate shift between a holdout
stratum and its training complement. The output converts "needs more data" into
a concrete, targeted cross-action collection list.

It makes NO model calls, changes NO runtime gate, and writes NO adoption
manifest — it is purely diagnostic. It therefore sits OUTSIDE the
human-amendment-only measurement trust boundary (MEASUREMENT.md): it can gate
no keep/revert/deploy/promote decision on its own; it only describes the data.

Companion to `evaluate_offline_reward_pairwise_ranker.py` (which decides signal /
holdout pass-fail). This script explains the holdout failures that one reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

AUDIT_SCHEMA_VERSION = "offline_reward_pairwise_preference_direction_audit.v1"
COLLECTION_TARGETS_SCHEMA_VERSION = "offline_reward_pairwise_collection_targets.v1"
PAIRWISE_ROW_SCHEMA_VERSION = "offline_reward_pairwise_preference.v1"
HOLDOUT_FIELDS = ("source_family", "suite")

# A cross-action pair needs both preference directions represented to be
# learnable; below this minority share it carries little directional signal.
DEFAULT_DIRECTION_BALANCE_WARN = 0.20
# Action-pairs (or strata) with fewer rows than this are "thin".
DEFAULT_MIN_ACTION_PAIR_ROWS = 8
DEFAULT_MIN_STRATUM_ROWS = 20
# An action-pair well-represented in a holdout stratum but with fewer than this
# many rows in the training complement is effectively "untrained" for it.
DEFAULT_MIN_TRAIN_ROWS = 10


class AuditError(Exception):
    """Raised on malformed pairwise contract input."""


def _action_pair_key(a: str, b: str) -> tuple[str, str, bool]:
    """Return (canonical_key, lo, cross_action) for an unordered action pair.

    The canonical key sorts the two actions so {X,Y} and {Y,X} collapse; `lo` is
    the sorted-first action so directional counts have a stable reference side.
    """
    cross = a != b
    lo, hi = (a, b) if a <= b else (b, a)
    return (f"{lo}>{hi}", lo, cross)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise AuditError(f"pairwise contract not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            sv = row.get("schema_version")
            if sv != PAIRWISE_ROW_SCHEMA_VERSION:
                raise AuditError(
                    f"{path}:{lineno}: unexpected schema_version {sv!r} "
                    f"(want {PAIRWISE_ROW_SCHEMA_VERSION!r})"
                )
            for field in ("preferred_canonical_action", "rejected_canonical_action"):
                if not row.get(field):
                    raise AuditError(f"{path}:{lineno}: missing {field}")
            rows.append(row)
    if not rows:
        raise AuditError(f"no pairwise rows in {path}")
    return rows


def _direction_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per action-pair: rows, cross_action flag, and the directional split
    (how many rows prefer the sorted-first action vs the sorted-second)."""
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        pref = str(row["preferred_canonical_action"])
        rej = str(row["rejected_canonical_action"])
        key, lo, cross = _action_pair_key(pref, rej)
        entry = table.setdefault(
            key,
            {"rows": 0, "cross_action": cross, "prefer_lo": 0, "prefer_hi": 0, "lo_action": lo},
        )
        entry["rows"] += 1
        if pref == lo:
            entry["prefer_lo"] += 1
        else:
            entry["prefer_hi"] += 1
    for entry in table.values():
        total = entry["rows"]
        minority = min(entry["prefer_lo"], entry["prefer_hi"])
        entry["direction_balance"] = round(minority / total, 4) if total else 0.0
        entry["one_sided"] = bool(entry["cross_action"] and minority == 0)
    return dict(sorted(table.items()))


def _counts_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(field) or "unknown")] += 1
    return dict(sorted(counts.items()))


def _stratum_audit(
    stratum_rows: list[dict[str, Any]],
    complement_rows: list[dict[str, Any]],
    *,
    min_action_pair_rows: int,
    direction_balance_warn: float,
    min_train_rows: int,
) -> dict[str, Any]:
    table = _direction_table(stratum_rows)
    complement_table = _direction_table(complement_rows)
    cross_rows = sum(e["rows"] for e in table.values() if e["cross_action"])
    same_rows = sum(e["rows"] for e in table.values() if not e["cross_action"])

    thin: list[str] = []
    one_sided: list[str] = []
    untrained: list[str] = []  # present in stratum, near-absent in train complement
    for key, entry in table.items():
        if not entry["cross_action"]:
            continue
        if entry["rows"] < min_action_pair_rows:
            thin.append(key)
        if entry["one_sided"] or entry["direction_balance"] < direction_balance_warn:
            one_sided.append(key)
        comp_rows = complement_table.get(key, {}).get("rows", 0)
        if comp_rows < min_train_rows:
            untrained.append(key)
    return {
        "rows": len(stratum_rows),
        "cross_action_rows": cross_rows,
        "same_action_rows": same_rows,
        "distinct_action_pairs": len([e for e in table.values() if e["cross_action"]]),
        "action_pair_directions": table,
        "thin_cross_action_pairs": sorted(thin),
        "one_sided_cross_action_pairs": sorted(one_sided),
        "untrained_in_complement_pairs": sorted(untrained),
    }


def audit_pairwise_preference_directions(
    rows: list[dict[str, Any]],
    *,
    min_stratum_rows: int = DEFAULT_MIN_STRATUM_ROWS,
    min_action_pair_rows: int = DEFAULT_MIN_ACTION_PAIR_ROWS,
    direction_balance_warn: float = DEFAULT_DIRECTION_BALANCE_WARN,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
) -> dict[str, Any]:
    overall_table = _direction_table(rows)
    overall_cross = sum(e["rows"] for e in overall_table.values() if e["cross_action"])

    strata: dict[str, dict[str, Any]] = {}
    collection_targets: list[dict[str, Any]] = []
    weak_strata: list[str] = []

    for field in HOLDOUT_FIELDS:
        by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_value[str(row.get(field) or "unknown")].append(row)
        field_out: dict[str, Any] = {}
        for value, stratum_rows in sorted(by_value.items()):
            complement = [r for r in rows if str(r.get(field) or "unknown") != value]
            audit = _stratum_audit(
                stratum_rows,
                complement,
                min_action_pair_rows=min_action_pair_rows,
                direction_balance_warn=direction_balance_warn,
                min_train_rows=min_train_rows,
            )
            # A stratum is "weak" (intrinsically un-learnable / under-covered) if it
            # is thin overall, has no cross-action signal, or its cross-action pairs
            # are one-sided. `untrained_in_complement_pairs` is reported as context
            # (it explains a holdout failure for minority strata) but is NOT a weak
            # trigger on its own: for a majority stratum it is a hold-one-out artifact
            # (its pairs are simply absent from the small complement), which would
            # otherwise flag a perfectly healthy, well-balanced stratum.
            weak_reasons: list[str] = []
            if audit["rows"] < min_stratum_rows:
                weak_reasons.append("thin_stratum")
            if audit["cross_action_rows"] == 0:
                weak_reasons.append("no_cross_action_rows")
            if audit["one_sided_cross_action_pairs"]:
                weak_reasons.append("one_sided_directions")
            audit["weak_reasons"] = weak_reasons
            field_out[value] = audit
            if weak_reasons:
                weak_strata.append(f"{field}:{value}")
                # Emit concrete collection targets for this weak stratum.
                table = audit["action_pair_directions"]
                # Targets = genuinely under-covered / one-directional cross-action
                # pairs only. `untrained_in_complement` is reported as stratum
                # context but does not by itself generate a target: for a majority
                # stratum it is a hold-one-out artifact (the pair can be fine and
                # well-balanced within the stratum), which would otherwise emit a
                # misleading "collect more" row.
                flagged = set(
                    audit["thin_cross_action_pairs"]
                    + audit["one_sided_cross_action_pairs"]
                )
                for key in sorted(flagged):
                    entry = table[key]
                    need_dir: list[str] = []
                    if entry["prefer_lo"] == 0:
                        need_dir.append(f"prefer {entry['lo_action']}")
                    if entry["prefer_hi"] == 0:
                        need_dir.append(f"prefer other-side of {key}")
                    collection_targets.append(
                        {
                            "stratum_field": field,
                            "stratum_value": value,
                            "action_pair": key,
                            "current_rows": entry["rows"],
                            "current_direction_balance": entry["direction_balance"],
                            "prefer_lo": entry["prefer_lo"],
                            "prefer_hi": entry["prefer_hi"],
                            "needs_direction": need_dir or ["balance both directions"],
                            "suggested_min_rows": max(min_action_pair_rows * 2, min_train_rows * 2),
                        }
                    )
        strata[field] = field_out

    status = "preference_coverage_gaps_found" if weak_strata else "preference_coverage_adequate"
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "thresholds": {
            "min_stratum_rows": min_stratum_rows,
            "min_action_pair_rows": min_action_pair_rows,
            "direction_balance_warn": direction_balance_warn,
            "min_train_rows": min_train_rows,
        },
        "input": {
            "pair_rows": len(rows),
            "cross_action_pair_rows": overall_cross,
            "same_action_pair_rows": len(rows) - overall_cross,
            "group_count": len({str(r.get("group_key")) for r in rows}),
            "source_family_pair_counts": _counts_by(rows, "source_family"),
            "suite_pair_counts": _counts_by(rows, "suite"),
            "action_pair_counts": {k: v["rows"] for k, v in overall_table.items()},
        },
        "overall_action_pair_directions": overall_table,
        "strata": strata,
        "collection_targets": collection_targets,
        "decision": {
            "status": status,
            "weak_strata": sorted(weak_strata),
            "runtime_gate_change_allowed": False,
            "recommended_next": (
                "collect non-overlapping cross-action preference rows for the listed "
                "collection_targets (balance both directions); re-run "
                "evaluate_offline_reward_pairwise_ranker.py holdouts after collection. "
                "Do NOT retune the absolute MLP/calibrator/pairwise family."
                if weak_strata
                else "no targeted collection indicated by direction/coverage audit"
            ),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Pairwise preference-direction audit (A9, offline/evidence-only)")
    lines.append("")
    dec = summary["decision"]
    lines.append(f"- **status**: `{dec['status']}`")
    lines.append(f"- **weak strata**: {', '.join(dec['weak_strata']) or '(none)'}")
    lines.append(f"- runtime_gate_change_allowed: `{dec['runtime_gate_change_allowed']}`")
    inp = summary["input"]
    lines.append(
        f"- input: {inp['pair_rows']} pairs "
        f"({inp['cross_action_pair_rows']} cross-action, "
        f"{inp['same_action_pair_rows']} same-action), {inp['group_count']} groups"
    )
    lines.append("")
    lines.append(f"**Recommended next**: {dec['recommended_next']}")
    lines.append("")
    for field, values in summary["strata"].items():
        lines.append(f"## holdout field: `{field}`")
        lines.append("")
        lines.append("| value | rows | cross | distinct pairs | weak reasons |")
        lines.append("|---|---|---|---|---|")
        for value, audit in values.items():
            lines.append(
                f"| `{value}` | {audit['rows']} | {audit['cross_action_rows']} | "
                f"{audit['distinct_action_pairs']} | "
                f"{', '.join(audit['weak_reasons']) or 'ok'} |"
            )
        lines.append("")
    if summary["collection_targets"]:
        lines.append("## Concrete collection targets")
        lines.append("")
        lines.append("| stratum | action pair | rows | dir balance | needs | suggest ≥ |")
        lines.append("|---|---|---|---|---|---|")
        for t in summary["collection_targets"]:
            lines.append(
                f"| `{t['stratum_field']}:{t['stratum_value']}` | `{t['action_pair']}` | "
                f"{t['current_rows']} | {t['current_direction_balance']} | "
                f"{'; '.join(t['needs_direction'])} | {t['suggested_min_rows']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def collection_targets_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the small downstream artifact consumed by the A9 collection planner."""
    return {
        "schema_version": COLLECTION_TARGETS_SCHEMA_VERSION,
        "source_audit_schema_version": summary.get("schema_version"),
        "thresholds": summary.get("thresholds", {}),
        "decision": summary.get("decision", {}),
        "collection_targets": summary.get("collection_targets", []),
    }


def run_pairwise_preference_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(args.pairwise_jsonl)
    return audit_pairwise_preference_directions(
        rows,
        min_stratum_rows=args.min_stratum_rows,
        min_action_pair_rows=args.min_action_pair_rows,
        direction_balance_warn=args.direction_balance_warn,
        min_train_rows=args.min_train_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pairwise preference directions and source_family/suite × action "
            "intersections in A9 offline-reward contracts (offline, evidence-only)."
        )
    )
    parser.add_argument("--pairwise-jsonl", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-md", type=Path, required=True)
    parser.add_argument(
        "--collection-targets-json",
        type=Path,
        help=(
            "Optional compact JSON artifact containing only collection_targets "
            "plus provenance, for plan_offline_reward_pairwise_holdout_expansion.py."
        ),
    )
    parser.add_argument("--min-stratum-rows", type=int, default=DEFAULT_MIN_STRATUM_ROWS)
    parser.add_argument("--min-action-pair-rows", type=int, default=DEFAULT_MIN_ACTION_PAIR_ROWS)
    parser.add_argument(
        "--direction-balance-warn", type=float, default=DEFAULT_DIRECTION_BALANCE_WARN
    )
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_pairwise_preference_audit(args)
    except (OSError, AuditError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.audit_md.parent.mkdir(parents=True, exist_ok=True)
    args.audit_md.write_text(render_markdown(summary), encoding="utf-8")
    if args.collection_targets_json:
        args.collection_targets_json.parent.mkdir(parents=True, exist_ok=True)
        args.collection_targets_json.write_text(
            json.dumps(collection_targets_payload(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
