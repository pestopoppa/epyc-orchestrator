#!/usr/bin/env python3
"""Feature Validation Battery: systematic A/B testing of disabled production features.

Validates each candidate feature against the production baseline by measuring
quality, latency, throughput, and memory impact. Features are toggled at runtime
via POST /config (no restart needed).

Usage:
    # Offline validation (no servers needed)
    python3 scripts/benchmark/feature_validation.py --offline --tier 0
    python3 scripts/benchmark/feature_validation.py --offline --tier 1

    # Live validation (requires orchestrator stack)
    python3 scripts/benchmark/feature_validation.py --live --tier 1
    python3 scripts/benchmark/feature_validation.py --live --tier 2

    # Generate comparison report
    python3 scripts/benchmark/feature_validation.py --report

    # Validate a single feature
    python3 scripts/benchmark/feature_validation.py --live --feature specialist_routing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

logger = logging.getLogger("feature_validation")

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "runs" / "feature_validation"
MANIFESTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "v1" / "feature_validation"
API_URL = os.environ.get("ORCHESTRATOR_API_URL", "http://localhost:8000")


# Re-export profile registry + validator/report classes so existing imports keep working.
# Sibling modules created during the 2026-05-22 Task-D refactor.
from feature_validation_profiles import (
    _build_profiles,
)
from feature_validation_offline import OfflineValidator
from feature_validation_live import (
    LiveValidator,
    _load_prompt_manifest,
)
from feature_validation_report import ReportGenerator


# Built once at import time so --feature filtering still works
PROFILES = _build_profiles()


# ── CLI ──────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature Validation Battery")
    p.add_argument("--offline", action="store_true", help="Run offline tests (mock + replay)")
    p.add_argument("--live", action="store_true", help="Run live tests (requires stack)")
    p.add_argument("--report", action="store_true", help="Generate comparison report")
    p.add_argument("--tier", type=int, default=None, help="Run only features in this tier")
    p.add_argument("--feature", type=str, default=None, help="Run only this feature")
    p.add_argument("--sample-size", type=int, default=5,
                    help="Prompts per feature (5=fast, 20=final)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter profiles
    targets = dict(PROFILES)
    if args.tier is not None:
        targets = {k: v for k, v in targets.items() if v.tier == args.tier}
    if args.feature:
        targets = {k: v for k, v in targets.items() if k == args.feature}

    if not targets:
        logger.error("No matching feature profiles found")
        sys.exit(1)

    logger.info("Validating %d features (tier=%s, feature=%s)",
                len(targets), args.tier, args.feature)

    if args.offline:
        validator = OfflineValidator()
        for name, profile in sorted(targets.items(), key=lambda x: (x[1].tier, x[0])):
            if profile.tier == 4:
                logger.info("SKIP (deferred): %s", name)
                continue
            logger.info("OFFLINE: %s (tier %d)", name, profile.tier)
            results = validator.validate_feature(profile)
            for r in results:
                status = "PASS" if r.test_pass_rate >= 1.0 and not r.errors else "FAIL"
                logger.info("  %s → %s (pass_rate=%.1f%%)",
                            name, status, r.test_pass_rate * 100)

    if args.live:
        validator = LiveValidator()
        if not validator._check_stack():
            logger.error("Orchestrator stack not reachable at %s", API_URL)
            sys.exit(1)

        # Capture baseline
        baseline_prompts = _load_prompt_manifest("general_5.json")
        if not baseline_prompts:
            logger.warning("No baseline prompts found, using empty set")
            baseline_prompts = []
        baseline = validator.capture_baseline(baseline_prompts)
        logger.info("Baseline captured: p50=%.2fs, tps=%.1f",
                     baseline.latency_p50_s, baseline.predicted_tps)

        for name, profile in sorted(targets.items(), key=lambda x: (x[1].tier, x[0])):
            if profile.tier == 4:
                logger.info("SKIP (deferred): %s", name)
                continue
            if not profile.live_tests:
                logger.info("SKIP (no live tests): %s", name)
                continue
            logger.info("LIVE: %s (tier %d)", name, profile.tier)
            report = validator.validate_feature(profile, baseline)
            logger.info("  %s → %s (qΔ=%+.3f, latΔ=%+.2fs)",
                        name, report.verdict, report.quality_delta,
                        report.latency_delta_s)

    if args.report:
        gen = ReportGenerator()
        md = gen.generate()
        print(md)
        logger.info("Report written to %s", RESULTS_DIR / "report.md")

    if not (args.offline or args.live or args.report):
        logger.error("Specify --offline, --live, or --report")
        sys.exit(1)


if __name__ == "__main__":
    main()
