#!/usr/bin/env python3
"""Assembler: merge per-source staging rows -> versioned corpus + manifest (RC-3).

Steps:
  1. (optional --run-miners) run each miner in order as a subprocess. The
     swecare miner self-relocates to a pyarrow interpreter, so all miners can be
     launched with the same interpreter.
  2. Read all _staging/*.jsonl.
  3. Dedup by row_id; validate against common.validate_row (drop + report bad).
  4. Enforce the seeded cap: defect_origin=seeded rows <= 50% of the corpus
     (deterministic downsample if exceeded).
  5. Write rows.jsonl (sorted by row_id -> stable content hash) + manifest.json.

manifest.json carries: corpus_id, schema/build versions, build timestamp,
content sha256, build-config hash, per-source / per-domain / per-defect_origin /
per-gold_label / per-gold_confidence counts, ambiguous_tail (arbitration) count,
natural-defect control-slice count, candidate-recovery-needed count, the
journal<->pool join fraction, input-file provenance hashes, and gap notes.

NO inference. No git operations.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from corpus_v1 import common
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from corpus_v1 import common

MINERS = [
    ("c-crab", "mine_ccrab.py", "ccrab.jsonl"),
    ("swe-care", "mine_swecare.py", "swecare.jsonl"),
    ("autopilot-journal", "mine_journals.py", "journals.jsonl"),
    ("seeded-mutation", "seed_mutations.py", "seeded.jsonl"),
    ("bug-report", "mine_bugreports.py", "bugreports.jsonl"),
]
_HERE = Path(__file__).resolve().parent


def run_miners() -> list[dict]:
    log = []
    for name, script, _out in MINERS:
        print(f"[assemble] running miner: {name} ({script})")
        proc = subprocess.run(
            [sys.executable, str(_HERE / script)],
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
        log.append({"miner": name, "returncode": proc.returncode,
                    "stdout_tail": proc.stdout.strip().splitlines()[-1:] })
    return log


def build_config_hash() -> str:
    cfg = {
        "corpus_id": common.CORPUS_ID,
        "schema_version": common.SCHEMA_VERSION,
        "mutation_rules_version": common.MUTATION_RULES_VERSION,
        "qid_scheme": common.QID_SCHEME,
        "gold_instrument_versions": common.GOLD_INSTRUMENT_VERSIONS,
        "domain_map": common.DOMAIN_MAP,
        "seeded_cap_fraction": 0.5,
        "sources": {
            "c-crab": str(common.CCRAB_PREPROCESS),
            "swe-care": str(common.SWECARE_TEST),
            "question_pool": str(common.QUESTION_POOL),
            "journals": [str(p) for p in common.journal_shards()],
            "bug_report_dirs": [str(p) for p in common.BUGREPORT_DIRS],
        },
    }
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return common.sha256_bytes(blob)


def input_provenance() -> dict:
    files = {
        "ccrab_preprocess": common.CCRAB_PREPROCESS,
        "ccrab_stage3": common.CCRAB_FUNNEL / "stage3_testgen_verified.jsonl",
        "ccrab_stage4": common.CCRAB_FUNNEL / "stage4_agent_resolved.jsonl",
        "swecare_test": common.SWECARE_TEST,
        "question_pool": common.QUESTION_POOL,
    }
    prov = {}
    for k, p in files.items():
        prov[k] = {"path": str(p), "exists": p.exists(),
                   "sha256": common.sha256_file(p) if p.exists() else None,
                   "bytes": p.stat().st_size if p.exists() else None}
    prov["journals"] = [
        {"path": str(p), "sha256": common.sha256_file(p), "bytes": p.stat().st_size}
        for p in common.journal_shards()
    ]
    return prov


def enforce_seeded_cap(rows: list[dict], cap: float = 0.5) -> tuple[list[dict], dict]:
    seeded = [r for r in rows if r["defect_origin"] == "seeded"]
    natural = [r for r in rows if r["defect_origin"] == "natural"]
    total = len(rows)
    info = {"applied": False, "seeded_before": len(seeded),
            "natural": len(natural), "seeded_after": len(seeded)}
    if total and len(seeded) / total > cap:
        # seeded <= natural  <=>  fraction <= 0.5
        keep = len(natural)
        seeded_sorted = sorted(seeded, key=lambda r: r["row_id"])
        seeded = seeded_sorted[:keep]
        info.update(applied=True, seeded_after=len(seeded))
        rows = natural + seeded
    return rows, info


def assemble(output_dir: Path, run: bool) -> dict:
    common.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    miner_log = run_miners() if run else []

    # Also recover the journal join fraction directly for the manifest.
    journal_join = None
    jpath = common.STAGING_DIR / "journals.jsonl"
    if jpath.exists():
        total = matched = 0
        for r in common.read_jsonl(jpath):
            total += 1
            if (r.get("provenance") or {}).get("pool_matched"):
                matched += 1
        journal_join = {"journal_rows": total, "pool_matched": matched,
                        "fraction": round(matched / total, 4) if total else 0.0}

    seen: set[str] = set()
    rows: list[dict] = []
    invalid = 0
    dup = 0
    per_staging = {}
    for _name, _script, out in MINERS:
        p = common.STAGING_DIR / out
        if not p.exists():
            per_staging[out] = 0
            continue
        cnt = 0
        for row in common.read_jsonl(p):
            errs = common.validate_row(row)
            if errs:
                invalid += 1
                if invalid <= 10:
                    print(f"[assemble] INVALID row ({out}): {errs} :: {row.get('row_id')}",
                          file=sys.stderr)
                continue
            rid = row["row_id"]
            if rid in seen:
                dup += 1
                continue
            seen.add(rid)
            rows.append(row)
            cnt += 1
        per_staging[out] = cnt

    rows, cap_info = enforce_seeded_cap(rows)
    rows.sort(key=lambda r: r["row_id"])  # deterministic content hash

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    common.write_jsonl(rows_path, rows)
    content_sha = common.sha256_file(rows_path)

    def count(field):
        return dict(Counter(r[field] for r in rows))

    natural_control = sum(1 for r in rows if r["natural_defect_control"])
    ambiguous = sum(1 for r in rows if r["ambiguous_tail"])
    recovery_needed = sum(
        1 for r in rows if (r.get("provenance") or {}).get("candidate_recovery_needed")
    )
    have_candidate = sum(1 for r in rows if r["candidate"])
    seeded_n = sum(1 for r in rows if r["defect_origin"] == "seeded")

    manifest = {
        "corpus_id": common.CORPUS_ID,
        "schema_version": common.SCHEMA_VERSION,
        "mutation_rules_version": common.MUTATION_RULES_VERSION,
        "qid_scheme": common.QID_SCHEME,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "content_sha256": content_sha,
        "build_config_hash": build_config_hash(),
        "rows_path": str(rows_path),
        "total_rows": len(rows),
        "counts": {
            "per_source": count("source_benchmark"),
            "per_domain": count("domain"),
            "per_defect_origin": count("defect_origin"),
            "per_gold_label": {str(k): v for k, v in count("gold_label").items()},
            "per_gold_confidence": count("gold_confidence"),
        },
        "seeded_fraction": round(seeded_n / len(rows), 4) if rows else 0.0,
        "seeded_cap": {"fraction": 0.5, **cap_info},
        "natural_defect_control_slice": natural_control,
        "ambiguous_tail_arbitration": ambiguous,
        "gate_worthy_multi_oracle": sum(
            1 for r in rows if r["gold_confidence"] == "multi_oracle"),
        "single_oracle_needs_arbitration": sum(
            1 for r in rows if r["gold_confidence"] == "single_oracle"),
        "candidate_present": have_candidate,
        "candidate_recovery_needed": recovery_needed,
        "journal_pool_join": journal_join,
        "gold_instrument_versions": common.GOLD_INSTRUMENT_VERSIONS,
        "input_provenance": input_provenance(),
        "dedup_dropped": dup,
        "invalid_dropped": invalid,
        "per_staging_kept": per_staging,
        "miner_log": miner_log,
        "notes": [
            "Layer-A instrument: reviewer metrics, NOT a T0-T3 model-quality axis.",
            "All pre-P-REV-1 numbers are observations; nothing here gates a decision.",
            "candidate_recovery_needed rows (autopilot-journal): task + reference gold "
            "answer recovered via qid<->question_pool join; the model's candidate "
            "answer text was never persisted -> needs a later NON-inference join to a "
            "captured-answer store OR an eval re-run to become reviewer-judgeable.",
            "reasoning_module_labels for journal rows are null -> a later inference "
            "labeling pass is required for WHY-diagnosis scoring.",
            "seeded 'multi_oracle' == synthetic ground truth (reference gold + "
            "deterministic rule), not two independent executable oracles.",
            "natural code defects (c-crab/swe-care) are the natural-defect control "
            "slice; SWE-Bench-Illusion decontamination metadata preserved per row.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    print(f"[assemble] wrote {len(rows)} rows -> {rows_path}")
    print(f"[assemble] manifest -> {manifest_path}")
    print(f"[assemble] content_sha256={content_sha}")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assemble near-miss corpus v1")
    ap.add_argument("--run-miners", action="store_true",
                    help="run all source miners before assembling")
    ap.add_argument("--output-dir", default=str(common.OUTPUT_DIR))
    args = ap.parse_args(argv)
    m = assemble(Path(args.output_dir), run=args.run_miners)
    print(json.dumps(m["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
