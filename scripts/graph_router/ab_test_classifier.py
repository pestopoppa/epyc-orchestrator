#!/usr/bin/env python3
"""A/B test harness for routing classifier vs baseline FAISS retrieval.

Runs seeding with classifier enabled vs disabled, compares:
- Pass rate (primary metric)
- Latency (secondary — classifier should reduce avg by skipping retrieval)
- Routing distribution (diagnostic — confirm similar routing patterns)

Uses ORCHESTRATOR_ROUTING_CLASSIFIER feature flag to toggle between arms.
Fisher exact test for significance on pass rate difference.

Usage:
    python3 scripts/graph_router/ab_test_classifier.py \
        --suite thinking --n-per-arm 50 [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ab_test_classifier")

SEEDING_SCRIPT = str(
    Path("/mnt/raid0/llm/epyc-inference-research/scripts/benchmark/seed_specialist_routing.py")
)


def run_arm(
    arm_name: str,
    classifier_enabled: bool,
    suite: str,
    n_questions: int,
    seed: int,
) -> dict:
    """Run one arm of the A/B test.

    Returns:
        {"pass_rate": float, "latencies": list, "routes": Counter, "n": int}
    """
    logger.info("=== Running arm: %s (classifier=%s) ===", arm_name, classifier_enabled)

    env = os.environ.copy()
    env["ORCHESTRATOR_ROUTING_CLASSIFIER"] = "1" if classifier_enabled else "0"

    cmd = [
        sys.executable, SEEDING_SCRIPT,
        "--suite", suite,
        "--max-questions", str(n_questions),
        "--seed", str(seed),
        "--json-output",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("Arm %s timed out after 1h", arm_name)
        return {"pass_rate": 0.0, "latencies": [], "routes": Counter(), "n": 0, "error": "timeout"}

    elapsed = time.time() - t0
    logger.info("Arm %s completed in %.1fs", arm_name, elapsed)

    # Parse JSON output (last line)
    output_lines = result.stdout.strip().split("\n")
    for line in reversed(output_lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                passes = data.get("passed", 0)
                total = data.get("total", 0)
                latencies = data.get("latencies", [])
                routes = Counter(data.get("routes", {}))
                return {
                    "pass_rate": passes / max(total, 1),
                    "latencies": latencies,
                    "routes": routes,
                    "n": total,
                    "passes": passes,
                }
            except json.JSONDecodeError:
                continue

    # Fallback: parse from stderr/stdout
    logger.warning("Could not parse JSON output for arm %s", arm_name)
    return {"pass_rate": 0.0, "latencies": [], "routes": Counter(), "n": 0, "error": "parse_failure"}


def fisher_exact_test(a_pass: int, a_total: int, b_pass: int, b_total: int) -> float:
    """Fisher exact test p-value for 2x2 contingency table.

    Uses scipy if available, otherwise falls back to chi-squared approximation.
    """
    try:
        from scipy.stats import fisher_exact
        table = [[a_pass, a_total - a_pass], [b_pass, b_total - b_pass]]
        _, p_value = fisher_exact(table)
        return p_value
    except ImportError:
        # Chi-squared approximation fallback
        n = a_total + b_total
        if n == 0:
            return 1.0
        p_pool = (a_pass + b_pass) / n
        if p_pool == 0 or p_pool == 1:
            return 1.0
        se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / max(a_total, 1) + 1.0 / max(b_total, 1)))
        if se == 0:
            return 1.0
        z = abs((a_pass / max(a_total, 1)) - (b_pass / max(b_total, 1))) / se
        # Normal CDF approximation
        p_value = 2 * (1 - 0.5 * (1 + np.math.erf(z / np.sqrt(2))))
        return float(p_value)


def main():
    parser = argparse.ArgumentParser(description="A/B test routing classifier")
    parser.add_argument("--suite", type=str, default="thinking", help="Benchmark suite")
    parser.add_argument("--n-per-arm", type=int, default=50, help="Questions per arm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info("=== Routing Classifier A/B Test ===")
    logger.info("Suite: %s, N per arm: %d, Seed: %d", args.suite, args.n_per_arm, args.seed)

    # Verify classifier weights exist
    weights_path = PROJECT_ROOT / "orchestration/repl_memory/routing_classifier_weights.npz"
    if not weights_path.exists():
        logger.error(
            "Classifier weights not found at %s. "
            "Run extract_training_data.py + train_routing_classifier.py first.",
            weights_path,
        )
        sys.exit(1)

    # Run both arms
    control = run_arm("control", classifier_enabled=False, suite=args.suite, n_questions=args.n_per_arm, seed=args.seed)
    treatment = run_arm("treatment", classifier_enabled=True, suite=args.suite, n_questions=args.n_per_arm, seed=args.seed)

    # Report
    logger.info("\n=== A/B Test Results ===")
    logger.info("Control (FAISS only):    pass_rate=%.1f%% (%d/%d)",
                control["pass_rate"] * 100, control.get("passes", 0), control["n"])
    logger.info("Treatment (classifier):  pass_rate=%.1f%% (%d/%d)",
                treatment["pass_rate"] * 100, treatment.get("passes", 0), treatment["n"])

    # Latency comparison
    if control["latencies"] and treatment["latencies"]:
        ctrl_lat = np.array(control["latencies"])
        treat_lat = np.array(treatment["latencies"])
        logger.info("\nLatency (seconds):")
        logger.info("  Control:   mean=%.2f  p50=%.2f  p95=%.2f",
                    ctrl_lat.mean(), np.median(ctrl_lat), np.percentile(ctrl_lat, 95))
        logger.info("  Treatment: mean=%.2f  p50=%.2f  p95=%.2f",
                    treat_lat.mean(), np.median(treat_lat), np.percentile(treat_lat, 95))

    # Routing distribution
    logger.info("\nRouting distribution:")
    all_routes = set(control["routes"].keys()) | set(treatment["routes"].keys())
    for route in sorted(all_routes):
        logger.info("  %-30s  control=%d  treatment=%d",
                    route, control["routes"].get(route, 0), treatment["routes"].get(route, 0))

    # Statistical test
    p_value = fisher_exact_test(
        control.get("passes", 0), control["n"],
        treatment.get("passes", 0), treatment["n"],
    )
    logger.info("\nFisher exact test p-value: %.4f", p_value)
    if p_value < 0.05:
        logger.info("Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        logger.info("Result: Not statistically significant (p >= 0.05)")

    # Decision
    delta = treatment["pass_rate"] - control["pass_rate"]
    logger.info("\nPass rate delta: %+.1f%%", delta * 100)
    if delta >= 0 and p_value < 0.1:
        logger.info("RECOMMENDATION: Enable classifier (non-negative effect, trending significant)")
    elif delta >= -0.02:
        logger.info("RECOMMENDATION: Classifier is neutral — enable for latency benefit")
    else:
        logger.info("RECOMMENDATION: Do NOT enable classifier (regression detected)")


if __name__ == "__main__":
    main()
