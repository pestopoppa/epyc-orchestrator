#!/usr/bin/env python3
"""Source 1: c-CRAB miner -> natural code-review defect rows (H4 RC-3).

c-CRAB is a SWE-CARE-derived benchmark of real human review comments with a
staged funnel that attaches *executable* fail-then-pass oracles:

  stage1/preprocess : 410 instances / 595 comments (LLM comment-quality filter)
  stage3            : 339 / 485 (test-gen fail->pass VERIFIED -> executable oracle)
  stage4            : 184 / 234 (agent-resolved -> oracle + confirmed fix)

Row unit = one (instance patch, review comment) pair:
  * candidate  = commit_to_review.patch_to_review  (the diff a reviewer judges)
  * gold cause = the human review-comment text     (reasoning-module label + WHY)
  * executable_oracle = derived from stage membership (stage3/stage4)
  * gold_label = "reject" (a real defect the human caught)  -> natural defect

We also emit one *accept-control* per instance (candidate = merged_patch, the
accepted/merged revision) as an FA-measurable clean example (observation-grade).

Per RC-3 we do NOT drop the solvable-only tail: comments with only the human
oracle (no verified executable test) are kept and flagged ambiguous_tail=true
(single_oracle -> route to human arbitration).

NATURAL defects. Pure stdlib (no inference).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

try:
    from corpus_v1 import common
except ImportError:  # allow running as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from corpus_v1 import common


def _comment_key(instance_id: str, c: dict) -> tuple:
    text = (c.get("text") or "")
    return (
        instance_id,
        c.get("path"),
        c.get("original_line"),
        hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12],
    )


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stage_comment_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    for r in _load_jsonl(path):
        for c in r.get("reference_review_comments") or []:
            keys.add(_comment_key(r["instance_id"], c))
    return keys


def mine() -> list[dict]:
    if not common.CCRAB_PREPROCESS.exists():
        print(f"[mine_ccrab] MISSING {common.CCRAB_PREPROCESS}; skipping", file=sys.stderr)
        return []

    preprocess = _load_jsonl(common.CCRAB_PREPROCESS)
    stage3 = _stage_comment_keys(common.CCRAB_FUNNEL / "stage3_testgen_verified.jsonl")
    stage4 = _stage_comment_keys(common.CCRAB_FUNNEL / "stage4_agent_resolved.jsonl")
    stage3_instances = {key[0] for key in stage3}
    stage4_instances = {key[0] for key in stage4}

    rows: list[dict] = []
    for inst in preprocess:
        instance_id = inst["instance_id"]
        repo = inst.get("repo")
        base_commit = inst.get("base_commit")
        pull_number = inst.get("pull_number")
        created_at = inst.get("created_at")
        language = (inst.get("language") or "").lower() or "python"
        domain = common.map_domain(language)
        problem = inst.get("problem_statement") or ""
        title = inst.get("title") or ""
        task = common.truncate((title + "\n\n" + problem).strip(), 8000)
        candidate = common.truncate(
            (inst.get("commit_to_review") or {}).get("patch_to_review") or "", 20000
        )
        decontam = {
            "repo": repo,
            "base_commit": base_commit,
            "pull_number": pull_number,
            "created_at": created_at,
            "instance_id": instance_id,
        }

        comments = inst.get("reference_review_comments") or []
        for idx, c in enumerate(comments):
            ckey = _comment_key(instance_id, c)
            in_s4 = ckey in stage4
            in_s3 = ckey in stage3
            comment_text = c.get("text") or ""

            if in_s4:
                execu = {
                    "verdict": "fail",
                    "oracle_type": "testgen_fail_then_pass",
                    "resolution": "agent_resolved",
                    "source": "c-crab/stage4_agent_resolved",
                }
                gold_conf = "multi_oracle"
                ambiguous = False
            elif in_s3:
                execu = {
                    "verdict": "fail",
                    "oracle_type": "testgen_fail_then_pass",
                    "source": "c-crab/stage3_testgen_verified",
                }
                gold_conf = "multi_oracle"
                ambiguous = False
            else:
                execu = None
                gold_conf = "single_oracle"  # human comment only
                ambiguous = True  # route to arbitration (RC-3: keep the tail)

            reasoning = {
                "source": "human_review_comment",
                "labels": [comment_text],
                "path": c.get("path"),
                "line": c.get("original_line"),
                "diff_hunk": common.truncate(c.get("diff_hunk"), 2000),
            }
            rows.append(
                common.make_row(
                    source_benchmark="c-crab",
                    source_suite=language,
                    domain=domain,
                    task=task,
                    candidate=candidate,
                    gold_label="reject",
                    gold_source="human_review_comment+testgen_oracle"
                    if execu
                    else "human_review_comment",
                    gold_confidence=gold_conf,
                    defect_origin="natural",
                    row_key=f"ccrab|{instance_id}|{ckey[1]}|{ckey[2]}|{ckey[3]}",
                    executable_oracle=execu,
                    reasoning_module_labels=reasoning,
                    rationale_gold_cause=common.truncate(comment_text, 4000),
                    ambiguous_tail=ambiguous,
                    natural_defect_control=True,  # the natural-defect control slice
                    decontamination=decontam,
                    provenance={
                        "instance_id": instance_id,
                        "comment_index": idx,
                        "n_comments": len(comments),
                        "oracle_stage": "stage4" if in_s4 else ("stage3" if in_s3 else "stage1"),
                        "candidate_is": "patch_to_review",
                        "difficulty": (inst.get("metadata") or {}).get("difficulty"),
                        "problem_domain": (inst.get("metadata") or {}).get("problem_domain"),
                    },
                )
            )

        # One accept-control per instance: the merged/accepted revision.
        merged = common.truncate(inst.get("merged_patch") or "", 20000)
        if merged:
            accept_in_s4 = instance_id in stage4_instances
            accept_in_s3 = instance_id in stage3_instances
            accept_oracle = None
            accept_confidence = "observation"
            if accept_in_s4 or accept_in_s3:
                accept_oracle = {
                    "oracle_type": "testgen_fail_then_pass",
                    "verdict": "pass",
                    "source": "c-crab/stage4_agent_resolved"
                    if accept_in_s4
                    else "c-crab/stage3_testgen_verified",
                    "resolution": "agent_resolved"
                    if accept_in_s4
                    else "testgen_verified",
                }
                accept_confidence = "multi_oracle"
            rows.append(
                common.make_row(
                    source_benchmark="c-crab",
                    source_suite=language,
                    domain=domain,
                    task=task,
                    candidate=merged,
                    gold_label="accept",
                    gold_source="merged_pr_accepted",
                    gold_confidence=accept_confidence,
                    defect_origin="natural",
                    row_key=f"ccrab-accept|{instance_id}",
                    executable_oracle=accept_oracle,
                    reasoning_module_labels={
                        "source": "merged_pr",
                        "note": "final merged/accepted revision (clean control)",
                    },
                    rationale_gold_cause=None,
                    ambiguous_tail=False,
                    natural_defect_control=False,
                    decontamination=decontam,
                    provenance={
                        "instance_id": instance_id,
                        "candidate_is": "merged_patch",
                        "clean_control": True,
                    },
                )
            )
    return rows


def main() -> int:
    rows = mine()
    out = common.STAGING_DIR / "ccrab.jsonl"
    n = common.write_jsonl(out, rows)
    print(f"[mine_ccrab] wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
