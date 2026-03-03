#!/usr/bin/env python3
"""Skill Transfer Regression Detection.

Compares two checkpoint sets (before/after a model swap) and flags
skill x domain cells where pass rate regressed beyond a threshold.

Usage:
    python scripts/benchmark/skill_transfer_regression.py --before dir_a --after dir_b
    python scripts/benchmark/skill_transfer_regression.py --before dir_a --after dir_b --threshold 0.15
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

from scripts.benchmark.seeding_types import DEBUG_PROMPTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_yaml_metadata(yaml_path: Path) -> dict[str, dict]:
    """Load skill/domain metadata from skill_transfer.yaml."""
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


def compute_cell_rates(
    records: list[dict],
    meta: dict[str, dict],
) -> dict[str, dict[tuple[str, str], float]]:
    """Compute pass rate per action key per (skill, domain) cell.

    Returns:
        action_key -> {(skill, domain): pass_rate}
    """
    action_cells: dict[str, dict[tuple[str, str], list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rec in records:
        qid = rec.get("question_id", "")
        qmeta = meta.get(qid, {})
        skill = qmeta.get("skill", "unknown")
        domain = qmeta.get("domain", "unknown")

        for action_key, result in rec.get("role_results", {}).items():
            passed = result.get("pass", False)
            action_cells[action_key][(skill, domain)].append(passed)

    rates: dict[str, dict[tuple[str, str], float]] = {}
    for action_key, cells in action_cells.items():
        rates[action_key] = {}
        for cell_key, outcomes in cells.items():
            rates[action_key][cell_key] = sum(outcomes) / len(outcomes) if outcomes else 0.0

    return rates


def compare_and_report(
    before_rates: dict[str, dict[tuple[str, str], float]],
    after_rates: dict[str, dict[tuple[str, str], float]],
    threshold: float,
) -> int:
    """Print diff table and flag regressions.

    Returns:
        Number of regressed cells.
    """
    all_actions = sorted(set(before_rates) | set(after_rates))
    if not all_actions:
        print("No skill_transfer data found in either checkpoint set.")
        return 0

    all_cells: set[tuple[str, str]] = set()
    for rates in (before_rates, after_rates):
        for cells in rates.values():
            all_cells.update(cells.keys())

    skills = sorted({s for s, _ in all_cells})
    domains = sorted({d for _, d in all_cells})

    total_regressions = 0

    for action_key in all_actions:
        b_cells = before_rates.get(action_key, {})
        a_cells = after_rates.get(action_key, {})

        print(f"\n{'='*80}")
        print(f"ACTION: {action_key}")
        print(f"{'='*80}")

        skill_w = max(len(s) for s in skills) if skills else 10
        col_w = 22

        header = f"{'Skill':<{skill_w}}"
        for d in domains:
            header += f" | {d:^{col_w}}"
        print(header)
        print("-" * len(header))

        for skill in skills:
            row = f"{skill:<{skill_w}}"
            for domain in domains:
                key = (skill, domain)
                b_rate = b_cells.get(key)
                a_rate = a_cells.get(key)

                if b_rate is not None and a_rate is not None:
                    delta = a_rate - b_rate
                    flag = ""
                    if delta < -threshold:
                        flag = " REGRESSED"
                        total_regressions += 1
                    cell = f"{b_rate:.0%}->{a_rate:.0%} ({delta:+.0%}){flag}"
                elif b_rate is not None:
                    cell = f"{b_rate:.0%}->N/A"
                elif a_rate is not None:
                    cell = f"N/A->{a_rate:.0%}"
                else:
                    cell = "-"
                row += f" | {cell:<{col_w}}"
            print(row)

    if total_regressions:
        print(f"\n** {total_regressions} cell(s) regressed beyond {threshold:.0%} threshold **")
    else:
        print(f"\nNo regressions beyond {threshold:.0%} threshold.")

    return total_regressions


def main():
    parser = argparse.ArgumentParser(
        description="Skill Transfer Regression Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--before", type=Path, required=True,
        help="Checkpoint directory for the baseline (before) run",
    )
    parser.add_argument(
        "--after", type=Path, required=True,
        help="Checkpoint directory for the comparison (after) run",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="Pass-rate drop threshold to flag as regression (default: 0.10)",
    )
    parser.add_argument(
        "--yaml", type=Path, default=DEBUG_PROMPTS_DIR / "skill_transfer.yaml",
        help="Path to skill_transfer.yaml",
    )

    args = parser.parse_args()

    if not args.yaml.exists():
        logger.error(f"YAML file not found: {args.yaml}")
        sys.exit(1)

    meta = load_yaml_metadata(args.yaml)
    logger.info(f"Loaded metadata for {len(meta)} questions from {args.yaml.name}")

    before_records = load_checkpoints(args.before)
    after_records = load_checkpoints(args.after)

    if not before_records and not after_records:
        print("No skill_transfer data found in either checkpoint set.")
        sys.exit(0)

    logger.info(f"Before: {len(before_records)} records, After: {len(after_records)} records")

    before_rates = compute_cell_rates(before_records, meta)
    after_rates = compute_cell_rates(after_records, meta)

    compare_and_report(before_rates, after_rates, args.threshold)
    sys.exit(0)


if __name__ == "__main__":
    main()
