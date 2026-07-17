#!/usr/bin/env python3
"""Source 2: SWE-CARE miner -> natural code-review *candidate pool* rows (RC-3).

SWE-CARE test split = 671 instances / 1313 human review comments. c-CRAB's
released subset (410 instances) is drawn from this split, so we DEDUP: rows whose
instance_id is already mined by c-CRAB are dropped here (c-CRAB carries the
richer executable-oracle stage metadata). The residual ~261 test instances are
exactly the ones c-CRAB's comment-quality filter dropped as LOW -> we keep them
as a single-oracle candidate pool flagged ambiguous_tail=true (arbitration).

Row unit mirrors c-CRAB: (patch_to_review as candidate, human review comment as
gold cause). NO executable oracle locally (execution needs Docker, not run) ->
executable_oracle=null, reasoning_module_labels=human comment. NATURAL defects.

pyarrow lives in the inference-research venv, not the orchestrator venv (no new
pip deps). This script self-relocates to a pyarrow-capable interpreter.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    from corpus_v1 import common
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from corpus_v1 import common


def _ensure_pyarrow() -> None:
    """Re-exec under a pyarrow-capable interpreter if the current one lacks it."""
    try:
        import pyarrow  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("_CORPUS_SWECARE_RELOCATED") == "1":
        print(
            "[mine_swecare] pyarrow unavailable and relocation already attempted; "
            "cannot read parquet. Checked: "
            + ", ".join(str(p) for p in common.PYARROW_PYTHONS),
            file=sys.stderr,
        )
        raise SystemExit(3)
    for py in common.PYARROW_PYTHONS:
        if not py.exists():
            continue
        # Probe the candidate for pyarrow before committing to exec.
        import subprocess

        probe = subprocess.run(
            [str(py), "-c", "import pyarrow"], capture_output=True
        )
        if probe.returncode == 0:
            env = dict(os.environ, _CORPUS_SWECARE_RELOCATED="1")
            print(f"[mine_swecare] relocating to pyarrow interpreter: {py}", file=sys.stderr)
            os.execve(str(py), [str(py), os.path.abspath(__file__), *sys.argv[1:]], env)
    print("[mine_swecare] no pyarrow-capable interpreter found", file=sys.stderr)
    raise SystemExit(3)


def _ccrab_instance_ids() -> set[str]:
    ids: set[str] = set()
    if not common.CCRAB_PREPROCESS.exists():
        return ids
    with open(common.CCRAB_PREPROCESS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["instance_id"].lower())
    return ids


def mine() -> list[dict]:
    _ensure_pyarrow()
    import pyarrow.parquet as pq

    if not common.SWECARE_TEST.exists():
        print(f"[mine_swecare] MISSING {common.SWECARE_TEST}; skipping", file=sys.stderr)
        return []

    ccrab_ids = _ccrab_instance_ids()
    cols = [
        "instance_id", "repo", "language", "pull_number", "created_at",
        "problem_statement", "title", "base_commit", "commit_to_review",
        "reference_review_comments", "merged_patch", "metadata",
    ]
    table = pq.read_table(common.SWECARE_TEST, columns=cols)
    data = table.to_pylist()

    rows: list[dict] = []
    for inst in data:
        instance_id = inst["instance_id"]
        if instance_id.lower() in ccrab_ids:
            continue  # dedup: already mined (richer) by c-CRAB
        language = (inst.get("language") or "").lower() or "python"
        domain = common.map_domain(language)
        problem = inst.get("problem_statement") or ""
        title = inst.get("title") or ""
        task = common.truncate((title + "\n\n" + problem).strip(), 8000)
        candidate = common.truncate(
            (inst.get("commit_to_review") or {}).get("patch_to_review") or "", 20000
        )
        decontam = {
            "repo": inst.get("repo"),
            "base_commit": inst.get("base_commit"),
            "pull_number": inst.get("pull_number"),
            "created_at": inst.get("created_at"),
            "instance_id": instance_id,
        }
        comments = inst.get("reference_review_comments") or []
        for idx, c in enumerate(comments):
            comment_text = c.get("text") or ""
            reasoning = {
                "source": "human_review_comment",
                "labels": [comment_text],
                "path": c.get("path"),
                "line": c.get("original_line"),
                "diff_hunk": common.truncate(c.get("diff_hunk"), 2000),
            }
            rows.append(
                common.make_row(
                    source_benchmark="swe-care",
                    source_suite=language,
                    domain=domain,
                    task=task,
                    candidate=candidate,
                    gold_label="reject",
                    gold_source="human_review_comment",
                    gold_confidence="single_oracle",  # human comment only, no local exec
                    defect_origin="natural",
                    row_key=f"swecare|{instance_id}|{idx}",
                    executable_oracle=None,
                    reasoning_module_labels=reasoning,
                    rationale_gold_cause=common.truncate(comment_text, 4000),
                    ambiguous_tail=True,  # single-oracle residual -> arbitration
                    natural_defect_control=True,
                    decontamination=decontam,
                    provenance={
                        "instance_id": instance_id,
                        "comment_index": idx,
                        "n_comments": len(comments),
                        "candidate_is": "patch_to_review",
                        "note": "swe-care-only residual (dropped by c-CRAB comment filter)",
                        "difficulty": (inst.get("metadata") or {}).get("difficulty"),
                        "problem_domain": (inst.get("metadata") or {}).get("problem_domain"),
                    },
                )
            )
        merged = common.truncate(inst.get("merged_patch") or "", 20000)
        if merged:
            rows.append(
                common.make_row(
                    source_benchmark="swe-care",
                    source_suite=language,
                    domain=domain,
                    task=task,
                    candidate=merged,
                    gold_label="accept",
                    gold_source="merged_pr_accepted",
                    gold_confidence="observation",
                    defect_origin="natural",
                    row_key=f"swecare-accept|{instance_id}",
                    executable_oracle=None,
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
    out = common.STAGING_DIR / "swecare.jsonl"
    n = common.write_jsonl(out, rows)
    print(f"[mine_swecare] wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
