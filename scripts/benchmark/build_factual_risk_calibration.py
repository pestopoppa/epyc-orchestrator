#!/usr/bin/env python3
"""Build factual risk calibration dataset from seeding diagnostics + question pool.

Extracts labeled prompts with known factual-risk levels for calibrating the
factual_risk classifier (routing-intelligence Phase 4, RI-1).

Sources:
    1. seeding_diagnostics.jsonl — eval results with pass/fail per question
    2. question_pool.jsonl — full question metadata with tier labels

Risk labeling strategy:
    - SimpleQA failed (3% pass rate) → "high" risk
    - SimpleQA passed → "medium" risk (easy factual, still risky domain)
    - GPQA failed → "high" risk (graduate-level science)
    - GPQA passed → "medium" risk
    - HotpotQA failed → "high" risk (multi-hop reasoning)
    - HotpotQA passed → "medium" risk
    - Non-factual suites (coder, math, debugbench) → "low" risk

Output: calibration_dataset.jsonl with {prompt, risk_label, suite, tier, passed, source}

Usage:
    python build_factual_risk_calibration.py [--min-examples 200] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")

DIAGNOSTICS_PATH = ORCH_ROOT / "logs" / "seeding_diagnostics.jsonl"
QUESTION_POOL_PATH = RESEARCH_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl"
DEFAULT_OUTPUT = ORCH_ROOT / "orchestration" / "factual_risk_calibration.jsonl"

# Factual suites — prompts that test factual knowledge / reasoning
FACTUAL_SUITES = {"simpleqa", "gpqa", "hotpotqa"}

# Risk labeling: (suite_is_factual, passed) → risk_label
RISK_LABELS = {
    # Factual suite, failed → high risk (model got it wrong on factual question)
    (True, False): "high",
    # Factual suite, passed → medium risk (factual domain, model happened to get it right)
    (True, True): "medium",
    # Non-factual suite → low risk (code, math, debugging — not factual)
    (False, True): "low",
    (False, False): "low",
}


def load_question_pool(path: Path) -> dict[str, dict]:
    """Load question pool indexed by question_id."""
    pool = {}
    if not path.exists():
        print(f"WARNING: Question pool not found at {path}", file=sys.stderr)
        return pool
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if q.get("__pool_metadata__"):
                continue  # Skip metadata header
            qid = q.get("id", "")
            if qid:
                pool[qid] = q
    return pool


def load_diagnostics(path: Path) -> list[dict]:
    """Load seeding diagnostic records."""
    records = []
    if not path.exists():
        print(f"WARNING: Diagnostics not found at {path}", file=sys.stderr)
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_calibration_dataset(
    diagnostics: list[dict],
    pool: dict[str, dict],
    include_non_factual: bool = True,
) -> list[dict]:
    """Build labeled calibration dataset from diagnostics + pool metadata."""
    dataset = []
    seen_prompts: set[str] = set()

    for diag in diagnostics:
        qid = diag.get("question_id", "")
        suite = diag.get("suite", "")
        passed = diag.get("passed", False)

        if not qid:
            continue

        # Get prompt from question pool (diagnostics don't store prompts).
        # Diagnostic question_ids may have suite/ prefix: "gpqa/gpqa_..." → "gpqa_..."
        pool_key = qid.split("/", 1)[-1] if "/" in qid else qid
        pool_entry = pool.get(pool_key, pool.get(qid, {}))
        prompt = pool_entry.get("prompt", "")
        if not prompt:
            continue

        # Deduplicate by prompt text (same question may appear in multiple evals)
        prompt_key = prompt[:200]
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)

        is_factual = suite in FACTUAL_SUITES
        if not include_non_factual and not is_factual:
            continue

        risk_label = RISK_LABELS.get((is_factual, passed), "low")

        # Tier from question pool
        tier = pool_entry.get("tier", 0)

        # Tier-based refinement: tier 3 questions are always high risk
        if is_factual and tier == 3:
            risk_label = "high"

        entry = {
            "prompt": prompt,
            "risk_label": risk_label,
            "suite": suite,
            "tier": tier,
            "passed": passed,
            "question_id": qid,
            "source": "seeding_diagnostics",
            "scoring_method": diag.get("scoring_method", ""),
            "error_type": diag.get("error_type", "none"),
        }
        dataset.append(entry)

    # Also add factual questions from pool that WEREN'T in diagnostics
    # (unseen questions — label by suite membership and tier)
    diag_qids = {d.get("question_id") for d in diagnostics}
    for qid, q in pool.items():
        if qid in diag_qids:
            continue
        suite = q.get("suite", "")
        if suite not in FACTUAL_SUITES:
            continue

        prompt = q.get("prompt", "")
        if not prompt:
            continue

        prompt_key = prompt[:200]
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)

        tier = q.get("tier", 0)
        # Unseen factual questions: risk based on tier
        if tier >= 3:
            risk_label = "high"
        elif tier >= 2:
            risk_label = "high"  # medium+ tier factual = high risk
        else:
            risk_label = "medium"

        entry = {
            "prompt": prompt,
            "risk_label": risk_label,
            "suite": suite,
            "tier": tier,
            "passed": None,  # No eval data
            "question_id": qid,
            "source": "question_pool",
            "scoring_method": q.get("scoring_method", ""),
            "error_type": "",
        }
        dataset.append(entry)

        # Stop adding pool questions once we have enough
        if len(dataset) >= 2000:
            break

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build factual risk calibration dataset")
    parser.add_argument("--min-examples", type=int, default=200,
                        help="Minimum examples required (default 200)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output JSONL path")
    parser.add_argument("--include-non-factual", action="store_true",
                        help="Include non-factual suites as 'low' risk examples")
    args = parser.parse_args()

    print(f"Loading question pool from {QUESTION_POOL_PATH}...")
    pool = load_question_pool(QUESTION_POOL_PATH)
    print(f"  Loaded {len(pool)} questions")

    print(f"Loading diagnostics from {DIAGNOSTICS_PATH}...")
    diagnostics = load_diagnostics(DIAGNOSTICS_PATH)
    print(f"  Loaded {len(diagnostics)} diagnostic records")

    print("Building calibration dataset...")
    dataset = build_calibration_dataset(
        diagnostics, pool, include_non_factual=args.include_non_factual
    )

    # Statistics
    label_counts = Counter(d["risk_label"] for d in dataset)
    suite_counts = Counter(d["suite"] for d in dataset)
    source_counts = Counter(d["source"] for d in dataset)

    print(f"\nDataset: {len(dataset)} examples")
    print(f"  By risk label: {dict(label_counts)}")
    print(f"  By suite: {dict(suite_counts)}")
    print(f"  By source: {dict(source_counts)}")

    if len(dataset) < args.min_examples:
        print(f"\nERROR: Only {len(dataset)} examples, need {args.min_examples}")
        sys.exit(1)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
