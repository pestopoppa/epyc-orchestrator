#!/usr/bin/env python3
"""P4.5: Extract per-qid per-role soft labels from autopilot journal question_results.

Fugu Stage 1 analog: compute mean correctness per role per question, apply
softmax(τ) to get a soft probability distribution, and save a labeled dataset
for KL-divergence-based LRC retraining.

Usage:
    python3 scripts/graph_router/extract_journal_soft_labels.py \
        [--journal PATH] [--output PATH] [--min-appearances N] [--tau FLOAT]

Outputs:
    soft_labels.jsonl  — per-qid records: {qid, suite, role_correctness, soft_labels}
    routing_analysis.md — per-suite routing diagnostic report

Next step (requires BGE server):
    Run scripts/graph_router/embed_soft_label_dataset.py to add BGE embeddings,
    then scripts/graph_router/train_routing_classifier_kl.py to retrain the MLP
    via KL divergence against the soft label distributions.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_primitives.stat_tests import wilson_interval  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_journal_soft_labels")

DEFAULT_JOURNAL = PROJECT_ROOT / "orchestration/autopilot_journal.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "orchestration/reports/p45_soft_labels"

# Canonical role ordering (must match RoutingClassifier label_map)
CANONICAL_ROLES = [
    "frontdoor",
    "coder_escalation",
    "worker_general",
    "architect_general",
    "ingest_long_context",
    "worker_vision",
]


_ROBUST_MIN_N = 20  # minimum sample size per arm for a routing gap to count


def _softmax(values: list[float], tau: float = 2.0) -> list[float]:
    """Numerically stable softmax with temperature tau."""
    scaled = [v / tau for v in values]
    max_v = max(scaled)
    exp_v = [math.exp(x - max_v) for x in scaled]
    total = sum(exp_v)
    return [x / total for x in exp_v]


def _wilson_lower(correct: int, total: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial proportion.

    Consolidated: delegates to the clean-room
    ``src.llm_primitives.stat_tests.wilson_interval`` (``z=1.96`` passed through
    to preserve the historical constant). ``total == 0`` yields 0.0 as before.
    """
    return wilson_interval(correct, total, z)[0]


def _wilson_upper(correct: int, total: int, z: float = 1.96) -> float:
    """Wilson score interval upper bound for a binomial proportion.

    Consolidated: delegates to the clean-room
    ``src.llm_primitives.stat_tests.wilson_interval`` (``z=1.96`` passed through
    to preserve the historical constant). ``total == 0`` yields 1.0 as before.
    """
    return wilson_interval(correct, total, z)[1]


def extract(
    journal_path: Path,
    output_dir: Path,
    min_appearances: int = 5,
    tau: float = 2.0,
) -> dict:
    """Extract soft labels from autopilot journal.

    Returns summary statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Collect per-qid per-role statistics ---
    logger.info("Loading journal from %s", journal_path)
    qid_role: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    qid_suite: dict[str, str] = {}
    suite_role: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    trials_with_qr = 0

    with open(journal_path) as f:
        for line in f:
            obj = json.loads(line)
            qr = obj.get("eval_details", {}).get("question_results", [])
            if not qr:
                continue
            trials_with_qr += 1
            for q in qr:
                qid = q.get("qid", "")
                suite = q.get("suite", "")
                role = q.get("route", "") or "unknown"
                correct = bool(q.get("correct", False))
                qid_suite[qid] = suite
                qid_role[qid][role]["correct"] += int(correct)
                qid_role[qid][role]["total"] += 1
                suite_role[suite][role]["correct"] += int(correct)
                suite_role[suite][role]["total"] += 1

    logger.info(
        "Loaded %d trials with question_results, %d unique qids",
        trials_with_qr,
        len(qid_role),
    )

    # --- Phase 2: Build per-qid soft labels ---
    all_roles = sorted(set(
        role
        for roles in qid_role.values()
        for role in roles
        if role != "unknown"
    ))
    logger.info("Roles observed: %s", all_roles)

    soft_label_records = []
    skipped_insufficient = 0
    skipped_no_route = 0

    for qid, role_stats in qid_role.items():
        total_appearances = sum(s["total"] for s in role_stats.values())
        has_route = any(r != "unknown" for r in role_stats)

        if total_appearances < min_appearances:
            skipped_insufficient += 1
            continue
        if not has_route:
            skipped_no_route += 1
            continue

        # Correctness per role (exclude 'unknown')
        role_correctness = {
            role: s["correct"] / s["total"]
            for role, s in role_stats.items()
            if role != "unknown" and s["total"] > 0
        }

        if not role_correctness:
            skipped_no_route += 1
            continue

        # Build correctness vector over canonical roles (0.0 for unseen)
        correctness_vec = [role_correctness.get(r, 0.0) for r in CANONICAL_ROLES]

        # Soft labels: softmax over correctness with temperature tau
        soft = _softmax(correctness_vec, tau=tau)

        soft_label_records.append({
            "qid": qid,
            "suite": qid_suite.get(qid, ""),
            "total_appearances": total_appearances,
            "role_correctness": role_correctness,
            "roles_seen": list(role_correctness.keys()),
            "canonical_roles": CANONICAL_ROLES,
            "correctness_vector": correctness_vec,
            "soft_labels_tau": tau,
            "soft_labels": soft,
            # Argmax = recommended single role
            "recommended_role": CANONICAL_ROLES[soft.index(max(soft))],
        })

    logger.info(
        "Soft-label records: %d (skipped: %d insufficient, %d no-route)",
        len(soft_label_records),
        skipped_insufficient,
        skipped_no_route,
    )

    # --- Phase 3: Save soft labels ---
    soft_labels_path = output_dir / "soft_labels.jsonl"
    with open(soft_labels_path, "w") as f:
        for rec in soft_label_records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Saved %d soft-label records to %s", len(soft_label_records), soft_labels_path)

    # --- Phase 4: Per-suite routing analysis ---
    _write_routing_analysis(suite_role, output_dir, tau)

    # Machine-readable robust routing misses (Wilson-CI, both arms n>=20)
    robust_misses = []
    for suite, role_stats in sorted(suite_role.items()):
        fd = role_stats.get("frontdoor", {})
        if fd.get("total", 0) < _ROBUST_MIN_N:
            continue
        fd_hi = _wilson_upper(fd["correct"], fd["total"])
        fd_rate = fd["correct"] / fd["total"]
        for route, s in role_stats.items():
            if route in ("frontdoor", "unknown") or s.get("total", 0) < _ROBUST_MIN_N:
                continue
            r_lo = _wilson_lower(s["correct"], s["total"])
            r_rate = s["correct"] / s["total"]
            if r_lo > fd_hi:
                robust_misses.append({
                    "suite": suite,
                    "better_route": route,
                    "better_rate": round(r_rate, 4),
                    "better_n": s["total"],
                    "better_ci_lo": round(r_lo, 4),
                    "frontdoor_rate": round(fd_rate, 4),
                    "frontdoor_n": fd["total"],
                    "frontdoor_ci_hi": round(fd_hi, 4),
                    "gain": round(r_rate - fd_rate, 4),
                })
    robust_misses.sort(key=lambda m: -m["gain"])

    # --- Phase 5: Suite-level soft label priors ---
    suite_priors = {}
    for suite, role_stats in suite_role.items():
        total = sum(s["total"] for s in role_stats.values())
        if total < 10:
            continue
        role_correctness = {
            r: s["correct"] / s["total"]
            for r, s in role_stats.items()
            if r != "unknown" and s["total"] >= 5
        }
        if not role_correctness:
            continue
        correctness_vec = [role_correctness.get(r, 0.0) for r in CANONICAL_ROLES]
        soft = _softmax(correctness_vec, tau=tau)
        suite_priors[suite] = {
            "role_correctness": role_correctness,
            "soft_labels": {r: p for r, p in zip(CANONICAL_ROLES, soft)},
            "recommended_role": CANONICAL_ROLES[soft.index(max(soft))],
            "total_questions": total,
        }

    suite_priors_path = output_dir / "suite_priors.json"
    with open(suite_priors_path, "w") as f:
        json.dump(suite_priors, f, indent=2)
    logger.info("Saved suite priors to %s", suite_priors_path)

    # Summary
    suite_dist = defaultdict(int)
    for rec in soft_label_records:
        suite_dist[rec["suite"]] += 1

    summary = {
        "trials_with_qr": trials_with_qr,
        "unique_qids": len(qid_role),
        "soft_label_records": len(soft_label_records),
        "skipped_insufficient": skipped_insufficient,
        "skipped_no_route": skipped_no_route,
        "tau": tau,
        "min_appearances": min_appearances,
        "suite_distribution": dict(sorted(suite_dist.items(), key=lambda x: -x[1])),
        "roles_observed": all_roles,
        "output_dir": str(output_dir),
        "soft_labels_path": str(soft_labels_path),
        "suite_priors_path": str(suite_priors_path),
        "robust_routing_misses": robust_misses,
        "robust_min_n": _ROBUST_MIN_N,
    }

    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    return summary


def _write_routing_analysis(
    suite_role: dict,
    output_dir: Path,
    tau: float,
) -> None:
    """Write per-suite routing diagnostic markdown report."""
    report_path = output_dir / "routing_analysis.md"

    roles = [r for r in CANONICAL_ROLES if any(
        r in suite_role[s] for s in suite_role
    )]

    lines = [
        "# P4.5 Journal Routing Analysis",
        "",
        "Per-suite per-role correctness extracted from autopilot journal.",
        f"Temperature τ={tau} for soft labels.",
        "",
        "## Statistically Robust Routing Misses",
        "",
        "Suites where a non-frontdoor route's Wilson 95% lower bound exceeds",
        "frontdoor's Wilson 95% upper bound — BOTH with n>=20. These are genuine",
        "routing gains, not sample-size noise. (Naive max-rate scans surface n=1",
        "flukes like simpleqa's 'architect 100%' which is a single lucky draw —",
        "those are excluded here.)",
        "",
    ]

    robust = []
    for suite, role_stats in sorted(suite_role.items()):
        fd = role_stats.get("frontdoor", {})
        if fd.get("total", 0) < _ROBUST_MIN_N:
            continue
        fd_hi = _wilson_upper(fd["correct"], fd["total"])
        fd_rate = fd["correct"] / fd["total"]
        for route, s in role_stats.items():
            if route in ("frontdoor", "unknown") or s.get("total", 0) < _ROBUST_MIN_N:
                continue
            r_lo = _wilson_lower(s["correct"], s["total"])
            r_rate = s["correct"] / s["total"]
            if r_lo > fd_hi:
                robust.append((suite, route, r_rate, s["total"], r_lo, fd_rate, fd["total"], fd_hi))

    if robust:
        for suite, route, r_rate, r_n, r_lo, fd_rate, fd_n, fd_hi in sorted(
            robust, key=lambda x: -(x[2] - x[5])
        ):
            lines.append(
                f"- **{suite}**: {route} {r_rate:.0%} (n={r_n}, CI_lo={r_lo:.0%}) "
                f"BEATS frontdoor {fd_rate:.0%} (n={fd_n}, CI_hi={fd_hi:.0%}) "
                f"— gain +{(r_rate-fd_rate):.0%}"
            )
    else:
        lines.append("- None found at n>=%d with non-overlapping CIs." % _ROBUST_MIN_N)

    lines += [
        "",
        "**Capability-ceiling suites (NOT routing misses)**: suites where all",
        "routes score similarly low (e.g. simpleqa ~5% across every route) are a",
        "model-capability/benchmark-difficulty ceiling — obscure factual recall",
        "that small quantized local models genuinely cannot do. Re-routing will",
        "not help; only a larger/RAG-augmented model would.",
    ]

    lines += [
        "",
        "## Per-Suite Per-Role Correctness Table",
        "",
        "| Suite | " + " | ".join(r[:12] for r in roles) + " | N total |",
        "|-------|" + "|".join("---" for _ in roles) + "|---------|",
    ]

    for suite in sorted(suite_role, key=lambda s: -sum(v["total"] for v in suite_role[s].values())):
        row_stats = suite_role[suite]
        total = sum(v["total"] for v in row_stats.values())
        if total < 10:
            continue
        cells = []
        for r in roles:
            s = row_stats.get(r, {})
            if s.get("total", 0) >= 5:
                cells.append(f"{s['correct']/s['total']:.0%}")
            elif s.get("total", 0) > 0:
                cells.append(f"({s['correct']/s['total']:.0%})*")
            else:
                cells.append("—")
        lines.append(f"| {suite} | " + " | ".join(cells) + f" | {total} |")

    lines += [
        "",
        "\\* Fewer than 5 examples — low confidence.",
        "",
        "## Routing Recommendations (from soft labels)",
        "",
        "| Suite | Recommended role | Confidence (soft label mass) |",
        "|-------|-----------------|------------------------------|",
    ]

    for suite, role_stats in sorted(suite_role.items()):
        total = sum(v["total"] for v in role_stats.values())
        if total < 20:
            continue
        role_correctness = {
            r: s["correct"] / s["total"]
            for r, s in role_stats.items()
            if r != "unknown" and s.get("total", 0) >= 5
        }
        if not role_correctness:
            continue
        correctness_vec = [role_correctness.get(r, 0.0) for r in CANONICAL_ROLES]
        soft = _softmax(correctness_vec, tau=tau)
        best_role = CANONICAL_ROLES[soft.index(max(soft))]
        best_mass = max(soft)
        lines.append(f"| {suite} | {best_role} | {best_mass:.1%} |")

    lines += [
        "",
        "## Next Steps for MLP Retraining",
        "",
        "The `soft_labels.jsonl` dataset has per-qid soft label distributions.",
        "To complete P4.5 MLP retraining:",
        "",
        "1. **Embed question texts** (requires BGE server at port 8090/8091):",
        "   ```",
        "   python3 scripts/graph_router/embed_soft_label_dataset.py \\",
        "       --soft-labels orchestration/reports/p45_soft_labels/soft_labels.jsonl \\",
        "       --output orchestration/reports/p45_soft_labels/soft_labels_embedded.npz",
        "   ```",
        "",
        "2. **Retrain MLP via KL divergence**:",
        "   ```",
        "   python3 scripts/graph_router/train_routing_classifier_kl.py \\",
        "       --data orchestration/reports/p45_soft_labels/soft_labels_embedded.npz \\",
        "       --output orchestration/repl_memory/routing_classifier_weights_kl.npz",
        "   ```",
        "",
        "3. **A/B against hard-label baseline** (val acc gate: ≥1 pp improvement).",
        "",
        "Alternatively, apply `suite_priors.json` as label smoothing to the",
        "existing episodic memory training set (if suite-type classification is added).",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Routing analysis saved to %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-appearances", type=int, default=5)
    parser.add_argument("--tau", type=float, default=2.0)
    args = parser.parse_args()

    summary = extract(
        journal_path=args.journal,
        output_dir=args.output,
        min_appearances=args.min_appearances,
        tau=args.tau,
    )

    print("\n=== P4.5 Extraction Summary ===")
    print(f"Trials with question_results: {summary['trials_with_qr']}")
    print(f"Unique qids: {summary['unique_qids']}")
    print(f"Soft-label records: {summary['soft_label_records']}")
    print(f"Skipped (insufficient data): {summary['skipped_insufficient']}")
    print(f"Skipped (no route info): {summary['skipped_no_route']}")
    print(f"Temperature τ: {summary['tau']}")
    print("\nSuite distribution of soft-label records:")
    for suite, count in summary["suite_distribution"].items():
        print(f"  {suite:<35} {count:4d}")
    print(f"\nStatistically robust routing misses (Wilson-CI, both arms n>={summary['robust_min_n']}):")
    if summary["robust_routing_misses"]:
        for m in summary["robust_routing_misses"]:
            print(
                f"  {m['suite']:<20} {m['better_route']} {m['better_rate']:.0%} (n={m['better_n']}) "
                f"> frontdoor {m['frontdoor_rate']:.0%} (n={m['frontdoor_n']})  +{m['gain']:.0%}"
            )
    else:
        print("  None.")
    print(f"\nOutput: {summary['output_dir']}")
    print("  soft_labels.jsonl   → embed with BGE for MLP training")
    print("  suite_priors.json   → use as label smoothing prior")
    print("  routing_analysis.md → routing diagnostic report")


if __name__ == "__main__":
    main()
