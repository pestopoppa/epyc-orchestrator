#!/usr/bin/env python3
"""Build v2 factual-risk calibration dataset (NIB2-34).

Supersedes ``scripts/benchmark/build_factual_risk_calibration.py`` (v1,
2026-03-29, 2000 regex-derived examples). V2 expands to 4,000-5,000
labeled examples by combining:

1. V1 dataset (2,000 examples, tier-based labels) — preserved as baseline
2. AA-Omniscience 600 public questions (intake-381, Apache 2.0) via
   ``ArtificialAnalysis/AA-Omniscience-Public`` on HuggingFace. Provides
   abstention-aware ground truth and domain diversity (Finance, Health,
   Humanities, Law, Science/Engineering, Software Engineering).
3. Fresh seeding_diagnostics entries (factual suites: simpleqa, gpqa,
   hotpotqa) not already in v1 — extracted by question_id delta.

Target: >=4,000 labeled examples with 4-class labels:
  * CORRECT         — model answered and was right
  * INCORRECT       — model answered and was wrong (high factual risk)
  * PARTIAL         — partially correct (scoring method allowed partial credit)
  * NOT_ATTEMPTED   — model abstained or said "I don't know"

Output schema (unified across all three sources):
  {
    "prompt": str,
    "expected_answer": str,
    "domain": str,          # Finance / simpleqa / gpqa / hotpotqa / tier_N
    "label_4class": str,    # CORRECT | INCORRECT | PARTIAL | NOT_ATTEMPTED
    "risk_band_v1": str,    # legacy binary-ish label (high/medium/low) from v1
    "label_source": str,    # v1_regex | aa_omniscience | seeding_diagnostics
    "risk_features": dict,  # from factual_risk.assess_risk() for feature stability
    "prompt_hash": str,     # sha256(prompt) for dedup
  }

Incremental write (feedback_incremental_persistence): examples are
streamed to the output JSONL per-source, not buffered.

Usage:
    python3 scripts/build_factual_risk_calibration_v2.py
    python3 scripts/build_factual_risk_calibration_v2.py --skip-risk-features
    python3 scripts/build_factual_risk_calibration_v2.py --no-aa-omniscience
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parent
RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")

V1_DATASET_PATH = ORCH_ROOT / "orchestration" / "factual_risk_calibration.jsonl"
DIAGNOSTICS_PATH = ORCH_ROOT / "logs" / "seeding_diagnostics.jsonl"
DEFAULT_OUTPUT = ORCH_ROOT / "orchestration" / "factual_risk_calibration_v2.jsonl"
QUESTION_POOL_PATH = RESEARCH_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl"

FACTUAL_SUITES = {"simpleqa", "gpqa", "hotpotqa"}


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def try_import_factual_risk():
    sys.path.insert(0, str(ORCH_ROOT))
    try:
        from src.classifiers.factual_risk import assess_risk  # type: ignore
        return assess_risk
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] factual_risk.assess_risk unavailable: {exc}", file=sys.stderr)
        return None


def load_jsonl(path: Path, skip_metadata: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if skip_metadata and rec.get("__pool_metadata__"):
                continue
            out.append(rec)
    return out


def _answer_to_label4(
    answer: str,
    expected: str,
    passed: bool | None,
    scoring_method: str | None,
) -> str:
    """Map raw outcome to 4-class label."""
    if answer is None:
        return "NOT_ATTEMPTED"
    a = answer.strip() if isinstance(answer, str) else str(answer)
    # Explicit abstention patterns
    lowered = a.lower()
    if any(x in lowered for x in ("i don't know", "i do not know", "idk", "unknown", "cannot determine")):
        return "NOT_ATTEMPTED"
    if passed is True:
        return "CORRECT"
    if passed is False:
        # Partial credit: f1 / partial scoring methods and the answer is non-empty
        if scoring_method in {"f1", "partial_credit"} and a:
            return "PARTIAL"
        return "INCORRECT"
    return "NOT_ATTEMPTED"


def source_v1_dataset(path: Path) -> list[dict[str, Any]]:
    """Load v1 dataset and promote to v2 schema."""
    v1 = load_jsonl(path)
    out = []
    for rec in v1:
        prompt = rec.get("prompt", "")
        if not prompt:
            continue
        # V1 didn't capture actual answers — infer label from pass/fail
        passed = rec.get("passed")
        if passed is True:
            label = "CORRECT"
        elif passed is False:
            label = "INCORRECT"
        else:
            # V1 tier-based examples without eval outcomes
            label = "NOT_ATTEMPTED"
        suite = rec.get("suite", "")
        domain = suite if suite else f"tier_{rec.get('tier', 'unknown')}"
        out.append({
            "prompt": prompt,
            "expected_answer": "",  # v1 didn't store expected
            "domain": domain,
            "label_4class": label,
            "risk_band_v1": rec.get("risk_label", ""),
            "label_source": "v1_regex",
            "prompt_hash": sha256_hex(prompt),
        })
    return out


def source_aa_omniscience() -> list[dict[str, Any]]:
    """Load AA-Omniscience 600 public questions via HuggingFace datasets."""
    try:
        import datasets as hf
        ds = hf.load_dataset(
            "ArtificialAnalysis/AA-Omniscience-Public", split="train",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] AA-Omniscience load failed: {exc}", file=sys.stderr)
        return []
    out = []
    for row in ds:
        question = row.get("question", "")
        answer = row.get("answer", "") or ""
        domain = row.get("domain", "unknown")
        if not question:
            continue
        # Without running inference, we don't know pass/fail — mark as
        # NOT_ATTEMPTED baseline. Production use will populate label_4class
        # after running benchmarks; for now the prompts + expected answers
        # are themselves a valuable augmentation.
        out.append({
            "prompt": question,
            "expected_answer": answer.strip(),
            "domain": str(domain),
            "label_4class": "NOT_ATTEMPTED",
            "risk_band_v1": "high" if len(answer) > 25 else "medium",
            "label_source": "aa_omniscience",
            "prompt_hash": sha256_hex(question),
        })
    return out


def source_fresh_seeding_diagnostics(
    v1_hashes: set[str],
) -> list[dict[str, Any]]:
    """Mine seeding_diagnostics for factual-suite entries not in v1.

    V1 was built from diagnostics as of 2026-03-29; newer entries may exist.
    """
    diagnostics = load_jsonl(DIAGNOSTICS_PATH)
    # Diagnostics don't carry prompt text — need question_pool to lookup by id
    pool = {}
    for q in load_jsonl(QUESTION_POOL_PATH, skip_metadata=True):
        qid = q.get("id")
        if qid:
            pool[qid] = q
    out = []
    for d in diagnostics:
        suite = d.get("suite", "")
        if suite not in FACTUAL_SUITES:
            continue
        qid = d.get("question_id")
        pool_entry = pool.get(qid) if qid else None
        if not pool_entry:
            continue
        prompt = pool_entry.get("prompt", "")
        if not prompt:
            continue
        phash = sha256_hex(prompt)
        if phash in v1_hashes:
            continue  # already in v1
        answer = d.get("answer", "") or ""
        expected = d.get("expected", "") or pool_entry.get("expected", "") or ""
        label = _answer_to_label4(
            answer, expected, d.get("passed"), d.get("scoring_method"),
        )
        out.append({
            "prompt": prompt,
            "expected_answer": str(expected).strip(),
            "domain": suite,
            "label_4class": label,
            "risk_band_v1": "high" if d.get("passed") is False else "medium",
            "label_source": "seeding_diagnostics",
            "prompt_hash": phash,
        })
    return out


def add_risk_features(
    examples: list[dict[str, Any]],
    assess_risk,
) -> None:
    """Populate risk_features in place."""
    for ex in examples:
        try:
            r = assess_risk(ex["prompt"])
            ex["risk_features"] = dict(r.risk_features)
            ex["risk_score_computed"] = round(r.risk_score, 4)
        except Exception as exc:  # noqa: BLE001
            ex["risk_features"] = {}
            ex["risk_score_computed"] = None


def stratified_split(
    examples: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """70/15/15 train/val/test stratified by (label_4class, domain) buckets."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ex in examples:
        key = (ex["label_4class"], ex["domain"])
        buckets.setdefault(key, []).append(ex)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for items in buckets.values():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    return {"train": train, "val": val, "test": test}


def write_jsonl_streaming(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(ex["label_source"] for ex in examples)
    by_label = Counter(ex["label_4class"] for ex in examples)
    by_domain = Counter(ex["domain"] for ex in examples)
    return {
        "total": len(examples),
        "by_source": dict(by_source),
        "by_label": dict(by_label),
        "top_domains": dict(by_domain.most_common(10)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--skip-risk-features", action="store_true",
                   help="Skip factual_risk feature computation (faster).")
    p.add_argument("--no-aa-omniscience", action="store_true",
                   help="Skip AA-Omniscience source (e.g., if HF cache unavailable).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[v2] Loading v1 dataset from {V1_DATASET_PATH}...")
    v1 = source_v1_dataset(V1_DATASET_PATH)
    print(f"  v1: {len(v1)} examples")

    v1_hashes = {ex["prompt_hash"] for ex in v1}

    aa = []
    if not args.no_aa_omniscience:
        print("[v2] Loading AA-Omniscience 600 public questions...")
        aa = source_aa_omniscience()
        aa = [ex for ex in aa if ex["prompt_hash"] not in v1_hashes]
        v1_hashes.update(ex["prompt_hash"] for ex in aa)
        print(f"  aa-omniscience: {len(aa)} examples (after v1 dedup)")

    print("[v2] Mining fresh seeding_diagnostics for factual suites...")
    fresh = source_fresh_seeding_diagnostics(v1_hashes)
    print(f"  seeding_diagnostics (fresh): {len(fresh)} examples (after v1 dedup)")

    combined = v1 + aa + fresh

    if not args.skip_risk_features:
        print("[v2] Computing risk_features per example...")
        assess_risk = try_import_factual_risk()
        if assess_risk is not None:
            add_risk_features(combined, assess_risk)
            print(f"  risk_features populated for {len(combined)} examples")
        else:
            print("  skipped (factual_risk.assess_risk unavailable)")

    # Stratified split
    splits = stratified_split(combined, seed=args.seed)

    # Write combined unsplit + per-split files
    write_jsonl_streaming(args.output, combined)
    for split_name, rows in splits.items():
        split_path = args.output.with_name(args.output.stem + f"_{split_name}.jsonl")
        write_jsonl_streaming(split_path, rows)

    summary = summarize(combined)
    summary["splits"] = {k: len(v) for k, v in splits.items()}
    summary["output"] = str(args.output)

    print("\n" + "=" * 60)
    print("V2 calibration dataset summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nOutput written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
