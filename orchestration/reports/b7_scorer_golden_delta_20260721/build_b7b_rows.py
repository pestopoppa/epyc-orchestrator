#!/usr/bin/env python3
"""Extend the B7 golden-delta corpus with the B7b scorer-semantics remainder.

B7b covers SCORE-07/08/09/12 — rubric judge range VALIDATION (not clamping),
judge-vs-heuristic provenance, empty-rubric ≠ perfect, and the phantom
`pass_rate` code_execution confidence. Unlike the B7 rows (whose scorer is
`scripts/benchmark/debug_scorer.py`), these behaviors live in
`scripts/autopilot/eval_tower.py` + `rubric_scoring.py`, so the rows carry a
`scorer` discriminator and are pinned via those surfaces, not `score_answer`.

The `new` column is computed here from the CURRENT (post-B7b) code, so it is the
ratified behavior; the pin test (`test_b7_golden_corpus_pin.py`) re-derives it
and fails on drift. The `old` column reproduces the pre-B7b behavior inline
(clamp / 1.0-default / pass_rate read) for the operator-facing delta only.

Idempotent: strips any existing ``b7b_*`` rows before appending fresh ones.
Run:  .venv/bin/python orchestration/reports/b7_scorer_golden_delta_20260721/build_b7b_rows.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from rubric_scoring import DEFAULT_RUBRIC_CRITERIA, aggregate_rubric_score  # noqa: E402

CORPUS_PATH = REPORT_DIR / "golden_corpus.jsonl"
RESULTS_PATH = REPORT_DIR / "results.jsonl"


def _old_parse_clamp(text: str) -> dict[str, float]:
    """Pre-B7b _parse_rubric_judge_scores: clamped, not validated."""
    parsed = eval_tower._extract_json_object(text)
    if not parsed:
        return {}
    raw = parsed.get("scores")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            out[str(key)] = min(max(numeric, 0.0), 1.0)
    return out


def _old_aggregate_score(scores: dict) -> float:
    """Pre-B7b aggregate: zero positive weight defaulted to 1.0."""
    positive_total = positive_weight = 0.0
    negative_total = negative_weight = 0.0
    for criterion in DEFAULT_RUBRIC_CRITERIA:
        weight = max(0.0, float(criterion.weight))
        if weight == 0.0:
            continue
        value = scores.get(criterion.name)
        if value is None or not math.isfinite(float(value)):
            continue
        bounded = min(max(float(value), 0.0), 1.0)
        if criterion.polarity == "negative":
            negative_total += bounded * weight
            negative_weight += weight
        else:
            positive_total += bounded * weight
            positive_weight += weight
    positive_score = positive_total / positive_weight if positive_weight else 1.0  # old default
    negative_penalty = negative_total / negative_weight if negative_weight else 0.0
    return min(max(positive_score - negative_penalty, 0.0), 1.0)


def _new_aggregate_score(scores: dict) -> float:
    return round(aggregate_rubric_score(scores).score, 6)


def _old_code_exec_conf(scoring_config: dict, correct: bool) -> list:
    return [float(scoring_config.get("pass_rate", correct)), "code_execution_pass_rate"]


def _new_code_exec_conf(correct: bool) -> list:
    conf, source = eval_tower._derive_question_confidence(
        scoring_method="code_execution",
        correct=correct,
        probability_confidence=None,
        rubric_scores={},
    )
    return [conf, source]


ALL8_HALF = {c.name: 0.5 for c in DEFAULT_RUBRIC_CRITERIA}


def build_rows() -> tuple[list[dict], list[dict]]:
    corpus: list[dict] = []
    results: list[dict] = []

    def add(case_id, scorer, finding, provenance, corpus_extra, old, new, note):
        row = {
            "case_id": case_id,
            "scorer": scorer,
            "finding": finding,
            "provenance": provenance,
        }
        row.update(corpus_extra)
        corpus.append(row)
        results.append({
            "case_id": case_id,
            "changed": old != new,
            "new": new,
            "old": old,
            "pin_fast": True,
            "provenance": provenance,
            "scoring_method": corpus_extra.get("scoring_method", scorer),
            "finding": finding,
            "note": note,
        })

    NOTE_RUBRIC = (
        "rubric path lives in eval_tower/rubric_scoring, not debug_scorer; "
        "new-behavior pinned against the real function"
    )

    # ── SCORE-07: rubric judge score RANGE VALIDATION, not clamping ──
    for cid, prov, answer in [
        ("b7b_score07_out_of_range_high", "b7b:score07/out_of_range_high",
         '{"scores": {"factual_accuracy": 7}}'),
        ("b7b_score07_out_of_range_negative", "b7b:score07/out_of_range_negative",
         '{"scores": {"citation": -0.5}}'),
        ("b7b_score07_in_range_kept", "b7b:score07/in_range_kept",
         '{"scores": {"outline": 0.7}}'),
        ("b7b_score07_boundary_one_kept", "b7b:score07/boundary_one_kept",
         '{"scores": {"presentation": 1.0}}'),
        ("b7b_score07_boundary_zero_kept", "b7b:score07/boundary_zero_kept",
         '{"scores": {"tool_calls": 0}}'),
        ("b7b_score07_mixed_reject_and_keep", "b7b:score07/mixed_reject_and_keep",
         '{"scores": {"reasoning_trajectory": 0.25, "tool_calls": 1.7, "content_stage": 3}}'),
    ]:
        add(
            cid, "rubric_parse", "SCORE-07", prov,
            {"answer": answer, "scoring_method": "rubric"},
            old=_old_parse_clamp(answer),
            new=eval_tower._parse_rubric_judge_scores(answer),
            note=NOTE_RUBRIC,
        )

    # ── SCORE-09: empty / all-missing rubric ≠ perfect ──
    for cid, prov, scores in [
        ("b7b_score09_empty", "b7b:score09/empty", {}),
        ("b7b_score09_all_missing", "b7b:score09/all_missing", {"unknown_dimension": 0.5}),
        ("b7b_score09_all_present_half", "b7b:score09/all_present_half", ALL8_HALF),
        ("b7b_score09_partial_positive", "b7b:score09/partial_positive", {"factual_accuracy": 0.8}),
    ]:
        answer = json.dumps(scores, sort_keys=True)
        add(
            cid, "rubric_aggregate", "SCORE-09", prov,
            {"answer": answer, "scoring_method": "rubric"},
            old=round(_old_aggregate_score(scores), 6),
            new=_new_aggregate_score(scores),
            note=NOTE_RUBRIC,
        )

    # ── SCORE-12: phantom pass_rate confidence removed ──
    for cid, prov, scoring_config, correct in [
        ("b7b_score12_pass_rate_correct", "b7b:score12/pass_rate_correct",
         {"pass_rate": 0.9}, True),
        ("b7b_score12_pass_rate_wrong", "b7b:score12/pass_rate_wrong",
         {"pass_rate": 0.9}, False),
        ("b7b_score12_no_pass_rate_source_renamed", "b7b:score12/no_pass_rate_source_renamed",
         {}, True),
    ]:
        add(
            cid, "code_exec_confidence", "SCORE-12", prov,
            {
                "answer": "",
                "scoring_method": "code_execution",
                "scoring_config": scoring_config,
                "correct": correct,
            },
            old=_old_code_exec_conf(scoring_config, correct),
            new=_new_code_exec_conf(correct),
            note="[confidence, confidence_source]; pass_rate dropped, source stamped non-real",
        )

    return corpus, results


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    corpus = [r for r in _load(CORPUS_PATH) if not r["case_id"].startswith("b7b_")]
    results = [r for r in _load(RESULTS_PATH) if not r["case_id"].startswith("b7b_")]
    b7b_corpus, b7b_results = build_rows()
    _write(CORPUS_PATH, corpus + b7b_corpus)
    _write(RESULTS_PATH, results + b7b_results)
    changed = sum(1 for r in b7b_results if r["changed"])
    print(f"B7b rows: {len(b7b_results)} added, {changed} changed")
    print(f"corpus total: {len(corpus) + len(b7b_corpus)}  results total: {len(results) + len(b7b_results)}")


if __name__ == "__main__":
    main()
