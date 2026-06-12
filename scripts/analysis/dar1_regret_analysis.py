#!/usr/bin/env python3
"""DAR-1: Offline regret analysis for decision-aware routing.

Analyzes historical routing decisions from progress logs to quantify
how much quality the predict-then-optimize Q-scorer leaves on the table.

Computes:
  - Selection score spread (are alternatives meaningfully different?)
  - Decision alignment (did we pick the top-ranked candidate?)
  - Outcome correlation (does selecting top candidate predict success?)

Source: decision-aware-routing.md DAR-1
Usage:
    python3 scripts/analysis/dar1_regret_analysis.py [--from 2026-04-01] [--to 2026-04-15]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoutingDecision:
    task_id: str
    timestamp: str
    chosen_action: str
    action_topk: list[str]
    strategy: str
    q_topk: list[float]
    selection_score_topk: list[float]
    difficulty_score: float
    difficulty_band: str
    factual_risk_score: float
    factual_risk_band: str
    q_robust_confidence: float


@dataclass
class TaskOutcome:
    task_id: str
    outcome: str  # "success", "failure", "partial"
    reward: float


@dataclass
class RegretReport:
    total_decisions: int = 0
    learned_decisions: int = 0  # strategy == "learned"
    rules_decisions: int = 0  # strategy == "rules" or "classifier"

    # Score spread analysis
    mean_score_spread: float = 0.0
    median_score_spread: float = 0.0
    pct_trivial_spread: float = 0.0  # spread < 0.01

    # Decision alignment
    identifiable_regret_decisions: int = 0
    regret_identifiable_pct: float = 0.0
    regret_gate_pct: float = 0.0
    top_candidate_selected_pct: float = 0.0
    mean_decision_regret: float = 0.0
    max_decision_regret: float = 0.0

    # Outcome correlation
    top_selected_success_rate: float = 0.0
    non_top_selected_success_rate: float = 0.0
    outcome_matched: int = 0

    # Q-value analysis
    mean_q_spread: float = 0.0
    pct_uniform_q: float = 0.0  # all Q-values identical (degenerate)

    # Difficulty band breakdown
    band_counts: dict[str, int] = field(default_factory=dict)
    band_regret: dict[str, float] = field(default_factory=dict)

    # Try-cheap-first denominator and acceptance telemetry.
    cheap_first_total: int = 0
    cheap_first_attempted: int = 0
    cheap_first_passed: int = 0
    cheap_first_reasons: dict[str, int] = field(default_factory=dict)

    # Caveats are first-class so downstream handoffs do not treat proxy metrics
    # as observed oracle regret.
    measurement_notes: list[str] = field(default_factory=list)


def parse_progress_logs(log_dir: Path, from_date: str, to_date: str) -> tuple[
    dict[str, RoutingDecision], dict[str, TaskOutcome], dict[str, dict[str, Any]]
]:
    """Parse progress JSONL logs for routing decisions and task outcomes."""
    decisions: dict[str, RoutingDecision] = {}
    outcomes: dict[str, TaskOutcome] = {}
    cheap_first: dict[str, dict[str, Any]] = {}

    pattern = str(log_dir / "*.jsonl")
    files = sorted(glob.glob(pattern))

    for filepath in files:
        fname = Path(filepath).stem  # YYYY-MM-DD
        if from_date and fname < from_date:
            continue
        if to_date and fname > to_date:
            continue

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = entry.get("event_type", "")
                task_id = entry.get("task_id", "")
                data = entry.get("data", {})

                if event_type == "routing_decision" and task_id:
                    q_topk = data.get("q_topk", [])
                    sel_topk = data.get("selection_score_topk", data.get("posterior_score_topk", []))
                    if q_topk and sel_topk:
                        routing = data.get("routing", [])
                        decisions[task_id] = RoutingDecision(
                            task_id=task_id,
                            timestamp=entry.get("timestamp", ""),
                            chosen_action=data.get(
                                "chosen_action",
                                routing[0] if routing else "unknown",
                            ),
                            action_topk=data.get("action_topk", []),
                            strategy=data.get("strategy", data.get("decision_source", "unknown")),
                            q_topk=q_topk,
                            selection_score_topk=sel_topk,
                            difficulty_score=data.get("difficulty_score", 0.0),
                            difficulty_band=data.get("difficulty_band", "unknown"),
                            factual_risk_score=data.get("factual_risk_score", 0.0),
                            factual_risk_band=data.get("factual_risk_band", ""),
                            q_robust_confidence=data.get("q_robust_confidence", 0.0),
                        )

                elif event_type in ("task_completed", "task_failed") and task_id:
                    outcome = entry.get("outcome") or data.get("outcome")
                    if not outcome:
                        outcome = "failure" if event_type == "task_failed" else "unknown"
                    reward = entry.get("reward", data.get("reward", 0.0))
                    outcomes[task_id] = TaskOutcome(
                        task_id=task_id,
                        outcome=outcome,
                        reward=reward,
                    )

                elif (
                    event_type == "routing_fallback"
                    and task_id
                    and data.get("kind") == "try_cheap_first"
                ):
                    cheap_first[task_id] = data

    return decisions, outcomes, cheap_first


def compute_regret(
    decisions: dict[str, RoutingDecision],
    outcomes: dict[str, TaskOutcome],
    cheap_first: dict[str, dict[str, Any]] | None = None,
) -> RegretReport:
    """Compute regret metrics from routing decisions and outcomes."""
    report = RegretReport()
    report.total_decisions = len(decisions)
    cheap_first = cheap_first or {}

    if not decisions:
        return report

    score_spreads: list[float] = []
    q_spreads: list[float] = []
    decision_regrets: list[float] = []
    top_selected_outcomes: list[bool] = []  # success when top was selected
    non_top_outcomes: list[bool] = []  # success when non-top was selected
    band_regrets: dict[str, list[float]] = defaultdict(list)
    band_counter: Counter = Counter()

    for task_id, d in decisions.items():
        # Strategy breakdown
        if d.strategy in ("learned", "memrl"):
            report.learned_decisions += 1
        else:
            report.rules_decisions += 1

        # Score spread
        if len(d.selection_score_topk) >= 2:
            spread = max(d.selection_score_topk) - min(d.selection_score_topk)
            score_spreads.append(spread)

        # Q-value spread
        if len(d.q_topk) >= 2:
            q_spread = max(d.q_topk) - min(d.q_topk)
            q_spreads.append(q_spread)

        if len(d.selection_score_topk) >= 2:
            top_score = d.selection_score_topk[0]
            regret: float | None = None
            if d.chosen_action and d.action_topk and d.chosen_action in d.action_topk:
                selected_idx = d.action_topk.index(d.chosen_action)
                if selected_idx < len(d.selection_score_topk):
                    regret = max(0.0, top_score - d.selection_score_topk[selected_idx])
            elif d.strategy in ("learned", "memrl"):
                regret = 0.0

            if regret is not None:
                report.identifiable_regret_decisions += 1
                decision_regrets.append(regret)

        # Difficulty band
        band_counter[d.difficulty_band] += 1

        # Outcome correlation
        outcome = outcomes.get(task_id)
        if outcome:
            report.outcome_matched += 1
            success = outcome.outcome == "success"
            if d.strategy in ("learned", "memrl"):
                top_selected_outcomes.append(success)
            else:
                non_top_outcomes.append(success)
            band_regrets[d.difficulty_band].append(
                0.0 if success else max(d.selection_score_topk) - min(d.selection_score_topk)
            )

    # Aggregate
    if score_spreads:
        sorted_spreads = sorted(score_spreads)
        report.mean_score_spread = sum(score_spreads) / len(score_spreads)
        report.median_score_spread = sorted_spreads[len(sorted_spreads) // 2]
        report.pct_trivial_spread = sum(1 for s in score_spreads if s < 0.01) / len(score_spreads) * 100

    if q_spreads:
        report.mean_q_spread = sum(q_spreads) / len(q_spreads)
        report.pct_uniform_q = sum(1 for s in q_spreads if s < 0.001) / len(q_spreads) * 100

    if decision_regrets:
        report.mean_decision_regret = sum(decision_regrets) / len(decision_regrets)
        report.max_decision_regret = max(decision_regrets)
        report.regret_gate_pct = report.mean_decision_regret * 100
        report.top_candidate_selected_pct = (
            sum(1 for r in decision_regrets if r < 0.001) / len(decision_regrets) * 100
        )
    if report.total_decisions:
        report.regret_identifiable_pct = (
            report.identifiable_regret_decisions / report.total_decisions * 100
        )

    if top_selected_outcomes:
        report.top_selected_success_rate = sum(top_selected_outcomes) / len(top_selected_outcomes) * 100
    if non_top_outcomes:
        report.non_top_selected_success_rate = sum(non_top_outcomes) / len(non_top_outcomes) * 100

    report.band_counts = dict(band_counter)
    report.band_regret = {
        band: sum(regrets) / len(regrets) if regrets else 0.0
        for band, regrets in band_regrets.items()
    }
    report.cheap_first_total = len(cheap_first)
    if cheap_first:
        report.cheap_first_attempted = sum(
            1 for row in cheap_first.values()
            if row.get("cheap_first_attempted") is True
        )
        report.cheap_first_passed = sum(
            1 for row in cheap_first.values()
            if row.get("cheap_first_passed") is True
        )
        report.cheap_first_reasons = dict(Counter(
            str(row.get("reason", "unknown")) for row in cheap_first.values()
        ))

    if report.identifiable_regret_decisions < report.total_decisions:
        missing = report.total_decisions - report.identifiable_regret_decisions
        report.measurement_notes.append(
            f"{missing} routing decisions lack enough action_topk telemetry for true selected-vs-best regret."
        )
    if report.regret_gate_pct < 5.0:
        report.measurement_notes.append(
            "DAR-1 gate remains closed: no >=5% mean decision-regret signal was proven."
        )

    return report


def print_report(report: RegretReport) -> None:
    """Print formatted regret analysis report."""
    print("=" * 70)
    print("DAR-1: OFFLINE REGRET ANALYSIS")
    print("=" * 70)

    print(f"\n{'Decisions analyzed:':<40} {report.total_decisions}")
    print(f"{'  Learned (Q-scorer):':<40} {report.learned_decisions}")
    print(f"{'  Rules/classifier:':<40} {report.rules_decisions}")
    print(f"{'  Matched with outcomes:':<40} {report.outcome_matched}")

    print(f"\n--- Score Spread Analysis ---")
    print(f"{'Mean selection score spread:':<40} {report.mean_score_spread:.4f}")
    print(f"{'Median selection score spread:':<40} {report.median_score_spread:.4f}")
    print(f"{'Trivial spread (<0.01):':<40} {report.pct_trivial_spread:.1f}%")
    print(f"{'Mean Q-value spread:':<40} {report.mean_q_spread:.4f}")
    print(f"{'Uniform Q-values (<0.001 spread):':<40} {report.pct_uniform_q:.1f}%")

    print(f"\n--- Decision Alignment ---")
    print(f"{'Regret-identifiable decisions:':<40} {report.identifiable_regret_decisions} ({report.regret_identifiable_pct:.1f}%)")
    print(f"{'Top candidate selected:':<40} {report.top_candidate_selected_pct:.1f}%")
    print(f"{'Mean decision regret:':<40} {report.mean_decision_regret:.4f}")
    print(f"{'DAR-1 gate regret pct:':<40} {report.regret_gate_pct:.2f}%")
    print(f"{'Max decision regret:':<40} {report.max_decision_regret:.4f}")

    print(f"\n--- Outcome Correlation ---")
    print(f"{'Success rate (top selected):':<40} {report.top_selected_success_rate:.1f}%")
    print(f"{'Success rate (non-top selected):':<40} {report.non_top_selected_success_rate:.1f}%")

    print(f"\n--- Difficulty Band Breakdown ---")
    for band in ["easy", "medium", "hard", "unknown"]:
        count = report.band_counts.get(band, 0)
        regret = report.band_regret.get(band, 0.0)
        if count > 0:
            print(f"  {band:<12} n={count:<6} avg_regret={regret:.4f}")

    print(f"\n--- Try-Cheap-First Counters ---")
    print(f"{'Counter rows:':<40} {report.cheap_first_total}")
    print(f"{'Attempts:':<40} {report.cheap_first_attempted}")
    print(f"{'Accepted:':<40} {report.cheap_first_passed}")
    if report.cheap_first_reasons:
        for reason, count in sorted(report.cheap_first_reasons.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {reason:<34} {count}")

    # Interpretation
    print(f"\n--- Interpretation ---")
    if report.pct_uniform_q > 80:
        print("⚠ >80% of decisions have uniform Q-values — Q-scorer has not learned meaningful preferences.")
        print("  DAR-2 contrastive training will have limited data to work with.")
        print("  Consider accumulating more routing memories via seeding before proceeding.")
    elif report.pct_trivial_spread > 50:
        print("⚠ >50% of decisions have trivial score spread (<0.01).")
        print("  Routing alternatives are not meaningfully differentiated.")
        print("  DAR-2 can sharpen boundaries but needs signal to work with.")
    else:
        print("✓ Meaningful score spread detected — decision-aware routing has signal to optimize.")

    if report.top_selected_success_rate > 0 and report.non_top_selected_success_rate > 0:
        delta = report.top_selected_success_rate - report.non_top_selected_success_rate
        if delta > 5:
            print(f"✓ Top-candidate selection predicts +{delta:.1f}pp success rate — Q-scorer has signal.")
        elif delta > -5:
            print(f"~ No significant difference ({delta:+.1f}pp) — Q-scorer ranking is weakly correlated with outcome.")
        else:
            print(f"⚠ Top-candidate selection is WORSE ({delta:+.1f}pp) — Q-scorer is miscalibrated. DAR-2 is HIGH value.")

    if report.measurement_notes:
        print(f"\n--- Measurement Notes ---")
        for note in report.measurement_notes:
            print(f"- {note}")


def main():
    parser = argparse.ArgumentParser(description="DAR-1: Offline routing regret analysis")
    parser.add_argument("--from", dest="from_date", default="", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", default="", help="End date (YYYY-MM-DD)")
    parser.add_argument("--log-dir", default=None, help="Progress log directory")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else (
        Path(__file__).resolve().parents[2] / "logs" / "progress"
    )

    if not log_dir.exists():
        print(f"Error: log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    decisions, outcomes, cheap_first = parse_progress_logs(log_dir, args.from_date, args.to_date)

    if not decisions:
        print("No routing decisions found in specified date range.", file=sys.stderr)
        sys.exit(1)

    report = compute_regret(decisions, outcomes, cheap_first)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(report), indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
