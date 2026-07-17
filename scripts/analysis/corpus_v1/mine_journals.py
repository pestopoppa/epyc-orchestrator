#!/usr/bin/env python3
"""Source 3: Autopilot-journal miner -> domain-labeled outcome rows (RC-3).

The autopilot journals hold thousands of *scored per-question outcomes* across
general/hotpotqa/simpleqa/instruction_precision/thinking/code + more. Each
``eval_details.question_results`` entry carries (qid, suite, correct, scoring
method) but crucially NOT the question text nor the model's answer text -- the
harness persists pass/fail + hashes only.

FORMAT SURPRISE + RECOVERY: the qid is a stable
``sha1(f"{suite}\\x00{prompt}")[:16]`` (eval_tower._stable_question_qid), so we
JOIN journal outcomes back to the research question pool
(question_pool.jsonl) and recover the *task* prompt and *reference gold answer*
for ~99% of distinct qids.

What we still CANNOT recover is the model's candidate ANSWER text -> these rows
carry candidate=null and are flagged ``candidate_recovery_needed`` (a later,
non-inference join to a captured-answer store, or an inference re-run, completes
them). They are gold_confidence="observation" (non-gating) domain priors.

NATURAL. Read-only journal access. Pure stdlib (no inference).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

try:
    from corpus_v1 import common
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from corpus_v1 import common


def mine() -> tuple[list[dict], dict]:
    pool = common.load_question_pool()

    # Aggregate per qid across all shards/trials.
    agg: dict[str, dict] = {}
    for _trial, qr in common.iter_journal_question_results():
        qid = qr["qid"]
        suite = qr.get("suite") or "unknown"
        rec = agg.setdefault(
            qid,
            {"suite": suite, "n": 0, "correct": 0, "scoring": defaultdict(int)},
        )
        rec["n"] += 1
        if qr.get("correct") is True or str(qr.get("correct")).lower() == "true":
            rec["correct"] += 1
        sm = qr.get("scoring_method")
        if sm:
            rec["scoring"][sm] += 1

    rows: list[dict] = []
    matched = 0
    for qid, rec in sorted(agg.items()):
        pinfo = pool.get(qid)
        if pinfo:
            matched += 1
        suite = rec["suite"]
        # Prefer the pool suite when available (identical by construction).
        if pinfo and pinfo.get("suite"):
            suite = pinfo["suite"]
        domain = common.map_domain(suite)
        scoring_method = (
            max(rec["scoring"].items(), key=lambda kv: kv[1])[0] if rec["scoring"] else None
        )
        pass_rate = rec["correct"] / rec["n"] if rec["n"] else None
        task = common.truncate(pinfo["prompt"], 8000) if pinfo else None
        reference_answer = (
            common.truncate(str(pinfo.get("expected")), 4000) if pinfo else None
        )
        rows.append(
            common.make_row(
                source_benchmark="autopilot-journal",
                source_suite=suite,
                domain=domain,
                task=task,
                candidate=None,  # model answer text not persisted -> recovery needed
                gold_label=None,  # no specific candidate to verdict yet
                gold_source=f"programmatic_scorer:{scoring_method or 'unknown'}",
                gold_confidence="observation",
                defect_origin="natural",
                row_key=f"journal|{qid}",
                executable_oracle=None,
                reasoning_module_labels=None,  # needs a later inference labeling pass
                rationale_gold_cause=None,
                ambiguous_tail=False,
                natural_defect_control=False,
                decontamination=None,
                provenance={
                    "qid": qid,
                    "suite": suite,
                    "pool_matched": bool(pinfo),
                    "dataset_source": pinfo.get("dataset_source") if pinfo else None,
                    "scoring_method": scoring_method,
                    "n_evaluations": rec["n"],
                    "n_correct": rec["correct"],
                    "pass_rate": pass_rate,
                    "reference_gold_answer": reference_answer,
                    "candidate_recovery_needed": True,
                    "needs_inference_pass": [
                        "candidate_answer_text",
                        "reasoning_module_labels",
                    ],
                },
            )
        )

    stats = {
        "distinct_qids": len(agg),
        "pool_matched": matched,
        "pool_match_fraction": round(matched / len(agg), 4) if agg else 0.0,
        "total_question_result_entries": sum(r["n"] for r in agg.values()),
    }
    return rows, stats


def main() -> int:
    rows, stats = mine()
    out = common.STAGING_DIR / "journals.jsonl"
    n = common.write_jsonl(out, rows)
    print(f"[mine_journals] wrote {n} rows -> {out}  stats={stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
