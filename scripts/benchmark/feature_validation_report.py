"""ReportGenerator — render comparison reports from feature_validation snapshots.

Extracted from scripts/benchmark/feature_validation.py during the 2026-05-22
Task-D refactor.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("feature_validation")

import os
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "runs" / "feature_validation"
MANIFESTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "v1" / "feature_validation"
API_URL = os.environ.get("ORCHESTRATOR_API_URL", "http://localhost:8000")



class ReportGenerator:
    """Generate markdown and CSV reports from validation results."""

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir

    def generate(self) -> str:
        """Generate a markdown summary report from all .jsonl results."""
        lines = [
            "# Feature Validation Battery Report",
            f"\nGenerated: {_now_iso()}\n",
            "## Results Summary\n",
            "| Feature | Tier | Verdict | Quality Δ | Latency Δ (s) | TPS Δ | Mem Δ (MB) |",
            "|---------|------|---------|-----------|---------------|-------|------------|",
        ]
        csv_rows = []

        for subdir in ("offline", "live"):
            result_dir = self.results_dir / subdir
            if not result_dir.exists():
                continue
            for jsonl_file in sorted(result_dir.glob("*.jsonl")):
                last_entry = None
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            last_entry = json.loads(line)
                if not last_entry:
                    continue

                feat = last_entry.get("feature", jsonl_file.stem)
                tier = PROFILES.get(feat, FeatureProfile(feat, -1)).tier
                verdict = last_entry.get("verdict", last_entry.get("test_pass_rate", "?"))
                qd = last_entry.get("quality_delta", 0)
                ld = last_entry.get("latency_delta_s", 0)
                td = last_entry.get("tps_delta", 0)
                md = last_entry.get("memory_delta_mb", 0)

                lines.append(
                    f"| {feat} | {tier} | {verdict} | {qd:+.3f} | {ld:+.2f} | "
                    f"{td:+.1f} | {md:+.1f} |"
                )
                csv_rows.append({
                    "feature": feat, "tier": tier, "verdict": verdict,
                    "quality_delta": qd, "latency_delta_s": ld,
                    "tps_delta": td, "memory_delta_mb": md,
                })

        report_md = "\n".join(lines) + "\n"

        # Write files
        report_path = self.results_dir / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_md)

        if csv_rows:
            csv_path = self.results_dir / "summary.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

        return report_md


# ── CLI ──────────────────────────────────────────────────────────────


