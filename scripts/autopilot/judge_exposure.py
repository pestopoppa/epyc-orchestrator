#!/usr/bin/env python3
"""Offline judge exposure inventory; never invokes a scorer or changes an instrument.

Pass the pool JSONL for inventory, or the exact selected question JSONL for a
draw's potential quality weight. Runtime scoring failures change the denominator,
so neither input establishes realized scored exposure. Use --results for one
complete native sidecar and its actual quality denominator. Methods describe
scoring-method exposure, not proof that a model-backed judge was invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from collections.abc import Iterable

JUDGE_METHODS = frozenset({"llm_judge", "rubric"})


def summarize(rows: Iterable[dict], *, selected: bool = False) -> dict:
    """Count distinct declared inputs and equal-row potential quality weights.

    Identity includes suite, prompt, context and image path, rather than local
    ids (which can be reused). Image bytes are not inspected; this is an input
    inventory, not an evaluation identity or measurement warrant.
    """
    suites: dict[str, Counter] = defaultdict(Counter)
    identities: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        if row.get("__pool_metadata__"):
            continue
        suite = row.get("suite")
        if not isinstance(suite, str) or not suite or not isinstance(row.get("prompt"), str):
            raise ValueError("every question requires a suite and string prompt")
        method = row.get("scoring_method") or "exact_match"
        if not isinstance(method, str):
            raise ValueError("scoring_method must be a string")
        identity = json.dumps(
            [suite, row["prompt"], row.get("context", ""), row.get("image_path", "")],
            sort_keys=True,
        )
        identity = hashlib.sha256(identity.encode()).hexdigest()
        previous = identities[suite].get(identity)
        if previous is not None and previous != method:
            raise ValueError(f"conflicting scoring methods for one input in {suite}")
        identities[suite][identity] = method
        suites[suite]["rows"] += 1
        suites[suite]["judge_rows"] += method in JUDGE_METHODS
    total = sum(counts["rows"] for counts in suites.values())
    output = {}
    for suite, records in sorted(suites.items()):
        methods = Counter(identities[suite].values())
        judge_rows = records["judge_rows"]
        distinct_judge = sum(methods[m] for m in JUDGE_METHODS)
        output[suite] = {
            "rows": records["rows"],
            "distinct_inputs": len(identities[suite]),
            "duplicate_rows": records["rows"] - len(identities[suite]),
            "distinct_by_method": dict(sorted(methods.items())),
            "distinct_judge_inputs": distinct_judge,
            "distinct_judge_fraction": distinct_judge / len(identities[suite]),
            "judge_rows": judge_rows,
            "potential_quality_axis_share": judge_rows / total if selected else None,
        }
    judge_rows = sum(s["judge_rows"] for s in output.values())
    distinct = sum(s["distinct_inputs"] for s in output.values())
    judge_distinct = sum(s["distinct_judge_inputs"] for s in output.values())
    return {
        "schema_version": 1,
        "scope": "selected_draw" if selected else "pool_inventory",
        "quality_semantics": "higher is better; 3 * correct / scored denominator; equal row weight",
        "caveat": "tier-stratified rotating or designed-core draws differ from pool frequencies; "
        "scoring failures change weights; method labels do not prove model judge invocation",
        "identity": "declared suite/prompt/context/image_path inventory only; other fields and image bytes not inspected",
        "realized_scored_quality_axis_share": None,
        "rows": total,
        "distinct_inputs": distinct,
        "distinct_judge_inputs": judge_distinct,
        "distinct_judge_fraction": judge_distinct / distinct if distinct else None,
        "potential_quality_axis_share": judge_rows / total if selected and total else None,
        "suites": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--selected", action="store_true", help="input is an exact selected draw")
    mode.add_argument(
        "--results", action="store_true", help="input is one complete native result sidecar"
    )
    parser.add_argument(
        "--partition", default="core", help="result decision partition (default: core)"
    )
    args = parser.parse_args()
    digest = hashlib.sha256()

    def rows():
        with args.questions.open("rb") as source:
            for line in source:
                digest.update(line)
                if line.strip():
                    yield json.loads(line)

    report = (
        summarize_results(rows(), partition=args.partition)
        if args.results
        else summarize(rows(), selected=args.selected)
    )
    report["source"] = {"path": str(args.questions), "sha256": digest.hexdigest()}
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


def summarize_results(rows: Iterable[dict], *, partition: str = "core") -> dict:
    """Project one complete native batch using the ETR-1 quality denominator."""
    batch_ids = set()
    ordinals = set()
    starts, completions, questions = [], [], []
    for row in rows:
        batch_ids.add(row.get("eval_batch_id"))
        kind = row.get("row_type")
        if kind == "batch_start":
            starts.append(row)
        elif kind == "batch_complete":
            completions.append(row)
        elif kind == "question_result":
            ordinal = row.get("ordinal")
            if not isinstance(ordinal, int) or ordinal in ordinals:
                raise ValueError("missing or duplicate result ordinal")
            ordinals.add(ordinal)
            questions.append(row["result"])
        else:
            raise ValueError(f"unknown result row type: {kind}")
    if len(batch_ids) != 1 or None in batch_ids or "" in batch_ids:
        raise ValueError("exactly one named eval batch required")
    if len(starts) != 1 or len(completions) != 1 or completions[0].get("complete") is not True:
        raise ValueError("one batch_start and complete marker required")
    if completions[0].get("completed_n") != len(questions) or starts[0].get("requested_n") != len(
        questions
    ):
        raise ValueError("incomplete batch question count")
    # Same predicate as EvalTower._quality_denominator, including legacy rows.
    decision = [r for r in questions if (r.get("partition") or "core") == partition]
    scored = [r for r in decision if r.get("disposition") == "task_failed" or not r.get("error")]
    report = summarize(
        (
            {
                "suite": r["suite"],
                "prompt": r["qid"],
                "scoring_method": r.get("scoring_method", "exact_match"),
            }
            for r in scored
        ),
        selected=True,
    )
    report.update(
        {
            "scope": "complete_result_batch",
            "eval_batch_id": next(iter(batch_ids)),
            "identity": "native suite/qid",
            "batch_rows": len(questions),
            "partition": partition,
            "excluded_partition_rows": len(questions) - len(decision),
            "excluded_error_rows": len(decision) - len(scored),
            "quality_denominator": len(scored),
            "realized_scored_quality_axis_share": report["potential_quality_axis_share"],
            "caveat": "method-label exposure for this complete batch only; "
            "not proof of model judge invocation or a claim about current production",
        }
    )
    for suite in report["suites"].values():
        suite["realized_scored_quality_axis_share"] = suite["potential_quality_axis_share"]
    return report


if __name__ == "__main__":
    main()
