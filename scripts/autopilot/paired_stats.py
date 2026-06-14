#!/usr/bin/env python3
"""Paired per-question statistics for AutoPilot eval journals.

This is intentionally useful before the ledger writer lands: current journals
will report zero vector-bearing trials, while synthetic/unit fixtures can test
the replay math. Once EvalTower journals ``eval_details.question_results``, the
same CLI provides the W2 replay gate for the sequential-verdict workstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.autopilot_core.journal_reconstruction import fold_supersession_events

DEFAULT_JOURNAL_DIR = Path(__file__).resolve().parents[2] / "orchestration"


@dataclass(frozen=True)
class QuestionOutcome:
    qid: str
    suite: str
    correct: bool
    trial_id: int


@dataclass(frozen=True)
class McNemarResult:
    trial_a: str
    trial_b: str
    shared_qids: int
    a_correct_b_wrong: int
    a_wrong_b_correct: int
    same_correct: int
    same_wrong: int
    p_value_two_sided: float
    odds_ratio_b_over_a: float
    accuracy_a: float
    accuracy_b: float
    delta_b_minus_a: float


def iter_journal_rows(path: Path | str) -> Iterable[dict[str, Any]]:
    """Yield JSONL rows from one file or every autopilot_journal*.jsonl in a dir."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("autopilot_journal*.jsonl"))
    rows: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {file_path}:{line_no}: {exc}") from exc
    folded_rows, _ = fold_supersession_events(rows)
    yield from (row for row in folded_rows if row.get("trial_id") is not None)


def extract_question_outcomes(row: dict[str, Any]) -> list[QuestionOutcome]:
    """Extract per-question vectors from a journal row, if present.

    Planned W1 location is ``eval_details.question_results``. This reader also
    accepts ``eval_details.details.question_results`` and a top-level
    ``question_results`` key so early experiments do not need a schema fork.
    """
    trial_id = int(row.get("trial_id", -1))
    eval_details = row.get("eval_details") or {}
    raw_results = (
        eval_details.get("question_results")
        or (eval_details.get("details") or {}).get("question_results")
        or row.get("question_results")
        or []
    )
    outcomes: list[QuestionOutcome] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("qid") or item.get("question_id") or "").strip()
        if not qid:
            continue
        suite = str(item.get("suite") or "").strip()
        outcomes.append(
            QuestionOutcome(
                qid=qid,
                suite=suite,
                correct=bool(item.get("correct")),
                trial_id=trial_id,
            )
        )
    return outcomes


def row_fingerprint(row: dict[str, Any]) -> str:
    """Stable config fingerprint for grouping replay candidates."""
    eval_details = row.get("eval_details") or {}
    explicit = (
        eval_details.get("config_fingerprint")
        or (row.get("config_snapshot") or {}).get("config_fingerprint")
        or row.get("config_fingerprint")
    )
    if explicit:
        return str(explicit)

    snapshot = row.get("config_snapshot") or {}
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    return "sha1:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def trial_vectors(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, QuestionOutcome]]:
    """Return trial_id -> qid -> outcome for vector-bearing rows."""
    vectors: dict[int, dict[str, QuestionOutcome]] = {}
    for row in rows:
        outcomes = extract_question_outcomes(row)
        if not outcomes:
            continue
        vectors[int(row["trial_id"])] = {outcome.qid: outcome for outcome in outcomes}
    return vectors


def mcnemar_from_vectors(
    vector_a: dict[str, QuestionOutcome],
    vector_b: dict[str, QuestionOutcome],
    label_a: str = "a",
    label_b: str = "b",
) -> McNemarResult:
    shared = sorted(set(vector_a) & set(vector_b))
    same_correct = same_wrong = a_correct_b_wrong = a_wrong_b_correct = 0
    for qid in shared:
        a = vector_a[qid].correct
        b = vector_b[qid].correct
        if a and b:
            same_correct += 1
        elif not a and not b:
            same_wrong += 1
        elif a and not b:
            a_correct_b_wrong += 1
        else:
            a_wrong_b_correct += 1

    n = len(shared)
    acc_a = (same_correct + a_correct_b_wrong) / n if n else 0.0
    acc_b = (same_correct + a_wrong_b_correct) / n if n else 0.0
    b_disc = a_correct_b_wrong
    c_disc = a_wrong_b_correct
    return McNemarResult(
        trial_a=label_a,
        trial_b=label_b,
        shared_qids=n,
        a_correct_b_wrong=b_disc,
        a_wrong_b_correct=c_disc,
        same_correct=same_correct,
        same_wrong=same_wrong,
        p_value_two_sided=_exact_two_sided_binomial_p(b_disc, c_disc),
        odds_ratio_b_over_a=round((c_disc + 0.5) / (b_disc + 0.5), 6),
        accuracy_a=round(acc_a, 6),
        accuracy_b=round(acc_b, 6),
        delta_b_minus_a=round(acc_b - acc_a, 6),
    )


def _exact_two_sided_binomial_p(b: int, c: int) -> float:
    """Exact sign-test p-value over discordant McNemar pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
    return round(min(1.0, 2.0 * tail), 8)


def majority_vector(rows: list[dict[str, Any]]) -> dict[str, QuestionOutcome]:
    """Collapse repeated trials into per-qid majority outcomes.

    Ties are dropped rather than forced, because a tied item has no majority
    evidence for either side.
    """
    votes: dict[str, list[QuestionOutcome]] = defaultdict(list)
    for row in rows:
        for outcome in extract_question_outcomes(row):
            votes[outcome.qid].append(outcome)

    out: dict[str, QuestionOutcome] = {}
    for qid, outcomes in votes.items():
        counts = Counter(o.correct for o in outcomes)
        if counts[True] == counts[False]:
            continue
        winner = counts[True] > counts[False]
        exemplar = outcomes[-1]
        out[qid] = QuestionOutcome(
            qid=qid,
            suite=exemplar.suite,
            correct=winner,
            trial_id=exemplar.trial_id,
        )
    return out


def group_rows_by_fingerprint(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if extract_question_outcomes(row):
            grouped[row_fingerprint(row)].append(row)
    return dict(grouped)


def compare_fingerprints(
    rows: list[dict[str, Any]],
    candidate_fingerprint: str,
    baseline_fingerprint: str,
) -> dict[str, Any]:
    grouped = group_rows_by_fingerprint(rows)
    candidate_rows = grouped.get(candidate_fingerprint, [])
    baseline_rows = grouped.get(baseline_fingerprint, [])
    candidate = majority_vector(candidate_rows)
    baseline = majority_vector(baseline_rows)
    result = mcnemar_from_vectors(
        baseline,
        candidate,
        label_a=f"baseline:{baseline_fingerprint}",
        label_b=f"candidate:{candidate_fingerprint}",
    )
    return {
        **asdict(result),
        "candidate_trials": [row["trial_id"] for row in candidate_rows],
        "baseline_trials": [row["trial_id"] for row in baseline_rows],
        "candidate_vector_qids": len(candidate),
        "baseline_vector_qids": len(baseline),
    }


def summarize(path: Path | str) -> dict[str, Any]:
    rows = list(iter_journal_rows(path))
    vectors = trial_vectors(rows)
    grouped = group_rows_by_fingerprint(rows)
    return {
        "rows": len(rows),
        "vector_trials": len(vectors),
        "vector_trial_ids": sorted(vectors),
        "fingerprints_with_vectors": {
            fingerprint: [row["trial_id"] for row in group]
            for fingerprint, group in sorted(grouped.items())
        },
    }


def _cmd_summary(args: argparse.Namespace) -> int:
    print(json.dumps(summarize(args.journal), indent=2, sort_keys=True))
    return 0


def _cmd_mcnemar(args: argparse.Namespace) -> int:
    vectors = trial_vectors(iter_journal_rows(args.journal))
    try:
        a = vectors[args.trial_a]
        b = vectors[args.trial_b]
    except KeyError as exc:
        raise SystemExit(f"trial has no question_results vector: {exc}") from exc
    result = mcnemar_from_vectors(a, b, str(args.trial_a), str(args.trial_b))
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


def _cmd_config_vs_baseline(args: argparse.Namespace) -> int:
    rows = list(iter_journal_rows(args.journal))
    result = compare_fingerprints(
        rows,
        candidate_fingerprint=args.candidate_fingerprint,
        baseline_fingerprint=args.baseline_fingerprint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paired per-question AutoPilot statistics")
    p.add_argument("--journal", default=str(DEFAULT_JOURNAL_DIR), help="journal dir or JSONL file")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("summary", help="show vector-bearing trial availability")
    ps.set_defaults(func=_cmd_summary)

    pm = sub.add_parser("mcnemar", help="compare two vector-bearing trials")
    pm.add_argument("trial_a", type=int)
    pm.add_argument("trial_b", type=int)
    pm.set_defaults(func=_cmd_mcnemar)

    pc = sub.add_parser("config-vs-baseline", help="compare majority vectors by fingerprint")
    pc.add_argument("candidate_fingerprint")
    pc.add_argument("baseline_fingerprint")
    pc.set_defaults(func=_cmd_config_vs_baseline)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
