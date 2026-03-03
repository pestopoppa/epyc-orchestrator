#!/usr/bin/env python3
"""Skill Transfer Pass-Rate Analysis.

Reads checkpoint JSONL files, filters to suite=="skill_transfer", joins
against the YAML source to recover skill/domain metadata, and outputs
a skill x domain pass-rate matrix per action key.

Usage:
    python scripts/benchmark/analyze_skill_transfer.py
    python scripts/benchmark/analyze_skill_transfer.py --checkpoint-dir /path/to/eval
    python scripts/benchmark/analyze_skill_transfer.py --format csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.seeding_types import DEBUG_PROMPTS_DIR, EVAL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_yaml_metadata(yaml_path: Path) -> dict[str, dict]:
    """Load skill/domain metadata from skill_transfer.yaml.

    Returns:
        Mapping of question id -> {"skill": ..., "domain": ...}
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    meta = {}
    for q in data.get("questions", []):
        meta[q["id"]] = {
            "skill": q.get("skill", "unknown"),
            "domain": q.get("domain", "unknown"),
        }
    return meta


def load_checkpoints(checkpoint_dir: Path) -> list[dict]:
    """Load all JSONL checkpoint records for suite=skill_transfer."""
    records = []
    if not checkpoint_dir.exists():
        return records

    for jsonl_path in sorted(checkpoint_dir.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("suite") == "skill_transfer":
                    records.append(rec)
    return records


def compute_pass_rates(
    records: list[dict],
    meta: dict[str, dict],
) -> dict[str, dict[tuple[str, str], dict]]:
    """Compute pass rates per action key, grouped by (skill, domain).

    Returns:
        action_key -> {(skill, domain): {"pass": n, "total": n, "rate": float}}
    """
    # Group by action key
    action_cells: dict[str, dict[tuple[str, str], list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rec in records:
        qid = rec.get("question_id", "")
        qmeta = meta.get(qid, {})
        skill = qmeta.get("skill", "unknown")
        domain = qmeta.get("domain", "unknown")

        role_results = rec.get("role_results", {})
        for action_key, result in role_results.items():
            passed = result.get("pass", False)
            action_cells[action_key][(skill, domain)].append(passed)

    # Compute rates
    action_rates: dict[str, dict[tuple[str, str], dict]] = {}
    for action_key, cells in action_cells.items():
        action_rates[action_key] = {}
        for cell_key, outcomes in cells.items():
            n_pass = sum(outcomes)
            n_total = len(outcomes)
            action_rates[action_key][cell_key] = {
                "pass": n_pass,
                "total": n_total,
                "rate": n_pass / n_total if n_total else 0.0,
            }

    return action_rates


def print_matrix(
    action_rates: dict[str, dict[tuple[str, str], dict]],
    output_format: str = "table",
) -> None:
    """Print skill x domain pass-rate matrix per action key."""
    # Collect all skills and domains
    all_skills: set[str] = set()
    all_domains: set[str] = set()
    for cells in action_rates.values():
        for skill, domain in cells:
            all_skills.add(skill)
            all_domains.add(domain)

    skills = sorted(all_skills)
    domains = sorted(all_domains)

    for action_key in sorted(action_rates):
        cells = action_rates[action_key]

        if output_format == "csv":
            print(f"\n# Action: {action_key}")
            print("skill," + ",".join(domains))
            for skill in skills:
                row = [skill]
                for domain in domains:
                    info = cells.get((skill, domain))
                    if info:
                        row.append(f"{info['rate']:.2f} ({info['pass']}/{info['total']})")
                    else:
                        row.append("-")
                print(",".join(row))
        else:
            print(f"\n{'='*70}")
            print(f"ACTION: {action_key}")
            print(f"{'='*70}")

            # Column widths
            skill_w = max(len(s) for s in skills) if skills else 10
            col_w = max(12, max((len(d) for d in domains), default=12))

            header = f"{'Skill':<{skill_w}}"
            for d in domains:
                header += f" | {d:^{col_w}}"
            print(header)
            print("-" * len(header))

            for skill in skills:
                row = f"{skill:<{skill_w}}"
                for domain in domains:
                    info = cells.get((skill, domain))
                    if info:
                        cell = f"{info['rate']:.0%} ({info['pass']}/{info['total']})"
                    else:
                        cell = "-"
                    row += f" | {cell:^{col_w}}"
                print(row)

            # Marginals
            print("-" * len(header))
            row = f"{'ALL':<{skill_w}}"
            for domain in domains:
                n_pass = sum(
                    cells.get((s, domain), {}).get("pass", 0) for s in skills
                )
                n_total = sum(
                    cells.get((s, domain), {}).get("total", 0) for s in skills
                )
                rate = n_pass / n_total if n_total else 0.0
                cell = f"{rate:.0%} ({n_pass}/{n_total})"
                row += f" | {cell:^{col_w}}"
            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Skill Transfer Pass-Rate Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=EVAL_DIR,
        help=f"Directory containing checkpoint JSONL files (default: {EVAL_DIR})",
    )
    parser.add_argument(
        "--yaml", type=Path, default=DEBUG_PROMPTS_DIR / "skill_transfer.yaml",
        help="Path to skill_transfer.yaml",
    )
    parser.add_argument(
        "--format", choices=["table", "csv"], default="table",
        help="Output format (default: table)",
    )

    args = parser.parse_args()

    # Load YAML metadata
    if not args.yaml.exists():
        logger.error(f"YAML file not found: {args.yaml}")
        sys.exit(1)

    meta = load_yaml_metadata(args.yaml)
    logger.info(f"Loaded metadata for {len(meta)} questions from {args.yaml.name}")

    # Load checkpoint data
    records = load_checkpoints(args.checkpoint_dir)
    if not records:
        print("No skill_transfer data found.")
        sys.exit(0)

    logger.info(f"Loaded {len(records)} skill_transfer checkpoint records")

    # Compute and display
    action_rates = compute_pass_rates(records, meta)
    print_matrix(action_rates, output_format=args.format)


if __name__ == "__main__":
    main()
