#!/usr/bin/env python3
"""Offline DAR-4b routing preference sweep.

This harness is intentionally read-only. It re-scores frozen progress-log
``routing_decision`` top-k telemetry under request-level performance/cost
preferences and cost tau values. It does not call the live orchestrator, the
router, embeddings, AutoPilot state, or replay DB writers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OMEGA_GRID = ("0.5,0.5", "0.8,0.2", "0.2,0.8")
DEFAULT_TAU_GRID = (0.8, 1.0, 1.2)
DEFAULT_COST_LAMBDA = 0.15


@dataclass(frozen=True)
class FrozenDecision:
    task_id: str
    timestamp: str
    chosen_action: str
    baseline_action: str
    action_topk: tuple[str, ...]
    q_topk: tuple[float, ...]
    normalized_cost_topk: tuple[float, ...]
    task_class: str = "unknown"
    strategy: str = "unknown"


@dataclass(frozen=True)
class SweepPoint:
    omega_perf: float
    omega_cost: float
    tau: float
    eligible_decisions: int
    mean_q: float
    mean_normalized_cost: float
    mean_score: float
    mean_margin: float
    flip_rate_vs_baseline: float
    flip_rate_vs_chosen: float
    action_counts: dict[str, int]
    pareto_frontier: bool = False


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _argmax(values: Iterable[float]) -> int:
    best_idx = 0
    best_value = float("-inf")
    for idx, value in enumerate(values):
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx


def parse_omega(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"omega must be '<perf>,<cost>', got {value!r}"
        )
    try:
        perf = max(0.0, float(parts[0]))
        cost = max(0.0, float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"omega values must be numeric, got {value!r}"
        ) from exc
    total = perf + cost
    if total <= 0.0:
        raise argparse.ArgumentTypeError(
            f"omega must have positive total weight, got {value!r}"
        )
    return perf / total, cost / total


def _progress_paths(log_dir: Path, from_date: str, to_date: str) -> list[Path]:
    paths = sorted(log_dir.glob("*.jsonl"))
    if from_date:
        paths = [path for path in paths if path.stem >= from_date]
    if to_date:
        paths = [path for path in paths if path.stem <= to_date]
    return paths


def _task_class_from(entry: dict[str, Any], data: dict[str, Any]) -> str:
    for value in (
        data.get("task_class"),
        data.get("real_task_class"),
        entry.get("task_class"),
        entry.get("real_task_class"),
    ):
        if value:
            return str(value)
    task_record = data.get("task_record_v1")
    if isinstance(task_record, dict) and task_record.get("class"):
        return str(task_record["class"])
    return "unknown"


def _normalized_costs(
    data: dict[str, Any],
    *,
    n: int,
    cost_lambda: float,
) -> tuple[float, ...]:
    normalized = _float_list(data.get("normalized_cost_topk"))
    if len(normalized) == n:
        return tuple(normalized)

    cost_terms = _float_list(data.get("cost_term_topk"))
    if len(cost_terms) == n and cost_lambda > 0.0:
        return tuple(term / cost_lambda for term in cost_terms)

    expected = _float_list(data.get("expected_cost_s"))
    cold = _float_list(data.get("cold_cost_s"))
    if len(expected) == n and len(cold) == n:
        return tuple(
            expected[idx] / max(cold[idx], 1e-6)
            for idx in range(n)
        )

    return ()


def load_frozen_decisions(
    log_dir: Path,
    *,
    from_date: str = "",
    to_date: str = "",
    cost_lambda: float = DEFAULT_COST_LAMBDA,
) -> tuple[list[FrozenDecision], dict[str, Any]]:
    decisions: list[FrozenDecision] = []
    skipped: Counter[str] = Counter()
    total_events = 0
    paths = _progress_paths(log_dir, from_date, to_date)

    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("event_type") != "routing_decision":
                    continue
                total_events += 1
                data = entry.get("data") or {}
                action_topk = _str_list(data.get("action_topk"))
                q_topk = _float_list(data.get("q_topk"))
                n = min(len(action_topk), len(q_topk))
                if n < 2:
                    skipped["missing_action_or_q_topk"] += 1
                    continue
                action_topk = action_topk[:n]
                q_topk = q_topk[:n]
                normalized_costs = _normalized_costs(
                    data,
                    n=n,
                    cost_lambda=cost_lambda,
                )
                if len(normalized_costs) != n:
                    skipped["missing_cost_topk"] += 1
                    continue
                selection_topk = _float_list(
                    data.get("selection_score_topk", data.get("posterior_score_topk"))
                )
                if len(selection_topk) == n:
                    baseline_action = action_topk[_argmax(selection_topk)]
                else:
                    baseline_action = action_topk[0]

                routing = _str_list(data.get("routing"))
                decisions.append(
                    FrozenDecision(
                        task_id=str(entry.get("task_id") or ""),
                        timestamp=str(entry.get("timestamp") or ""),
                        chosen_action=str(
                            data.get("chosen_action")
                            or (routing[0] if routing else baseline_action)
                        ),
                        baseline_action=baseline_action,
                        action_topk=tuple(action_topk),
                        q_topk=tuple(q_topk),
                        normalized_cost_topk=tuple(normalized_costs),
                        task_class=_task_class_from(entry, data),
                        strategy=str(
                            data.get("strategy", data.get("decision_source", "unknown"))
                        ),
                    )
                )

    meta = {
        "log_dir": str(log_dir),
        "paths": [str(path) for path in paths],
        "total_routing_events": total_events,
        "eligible_decisions": len(decisions),
        "skipped": dict(skipped),
    }
    return decisions, meta


def _score_topk(
    decision: FrozenDecision,
    *,
    omega_perf: float,
    omega_cost: float,
    tau: float,
    cost_lambda: float,
) -> tuple[int, list[float]]:
    scores = [
        2.0 * (
            omega_perf * decision.q_topk[idx]
            - omega_cost * cost_lambda * tau * decision.normalized_cost_topk[idx]
        )
        for idx in range(len(decision.action_topk))
    ]
    return _argmax(scores), scores


def run_sweep(
    decisions: list[FrozenDecision],
    *,
    omega_grid: list[tuple[float, float]],
    tau_grid: list[float],
    cost_lambda: float = DEFAULT_COST_LAMBDA,
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    if not decisions:
        return points

    for omega_perf, omega_cost in omega_grid:
        for tau in tau_grid:
            q_values: list[float] = []
            costs: list[float] = []
            scores: list[float] = []
            margins: list[float] = []
            baseline_flips = 0
            chosen_flips = 0
            action_counts: Counter[str] = Counter()

            for decision in decisions:
                idx, topk_scores = _score_topk(
                    decision,
                    omega_perf=omega_perf,
                    omega_cost=omega_cost,
                    tau=tau,
                    cost_lambda=cost_lambda,
                )
                action = decision.action_topk[idx]
                action_counts[action] += 1
                q_values.append(decision.q_topk[idx])
                costs.append(decision.normalized_cost_topk[idx])
                scores.append(topk_scores[idx])
                sorted_scores = sorted(topk_scores, reverse=True)
                if len(sorted_scores) >= 2:
                    margins.append(sorted_scores[0] - sorted_scores[1])
                if action != decision.baseline_action:
                    baseline_flips += 1
                if action != decision.chosen_action:
                    chosen_flips += 1

            n = len(decisions)
            points.append(
                SweepPoint(
                    omega_perf=omega_perf,
                    omega_cost=omega_cost,
                    tau=tau,
                    eligible_decisions=n,
                    mean_q=sum(q_values) / n,
                    mean_normalized_cost=sum(costs) / n,
                    mean_score=sum(scores) / n,
                    mean_margin=sum(margins) / len(margins) if margins else 0.0,
                    flip_rate_vs_baseline=baseline_flips / n,
                    flip_rate_vs_chosen=chosen_flips / n,
                    action_counts=dict(sorted(action_counts.items())),
                )
            )

    return _mark_pareto(points)


def _mark_pareto(points: list[SweepPoint]) -> list[SweepPoint]:
    out: list[SweepPoint] = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            quality_ok = other.mean_q >= point.mean_q
            cost_ok = other.mean_normalized_cost <= point.mean_normalized_cost
            strict = (
                other.mean_q > point.mean_q
                or other.mean_normalized_cost < point.mean_normalized_cost
            )
            if quality_ok and cost_ok and strict:
                dominated = True
                break
        payload = asdict(point)
        payload["pareto_frontier"] = not dominated
        out.append(SweepPoint(**payload))
    return out


def build_summary(
    *,
    decisions: list[FrozenDecision],
    source_meta: dict[str, Any],
    points: list[SweepPoint],
    omega_grid: list[tuple[float, float]],
    tau_grid: list[float],
    cost_lambda: float,
) -> dict[str, Any]:
    baseline_counts = Counter(decision.baseline_action for decision in decisions)
    pareto = [asdict(point) for point in points if point.pareto_frontier]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "dar4b_offline_routing_preference_sweep_v1",
        "measurement_class": "offline_proxy_observation",
        "source": source_meta,
        "cost_lambda": cost_lambda,
        "omega_grid": [
            {"perf": perf, "cost": cost}
            for perf, cost in omega_grid
        ],
        "tau_grid": tau_grid,
        "baseline": {
            "action_counts": dict(sorted(baseline_counts.items())),
        },
        "sweeps": [asdict(point) for point in points],
        "pareto": pareto,
        "notes": [
            "Uses frozen routing_decision top-k telemetry only.",
            "Mean q and normalized cost are selector proxies, not live quality or latency.",
            "No live inference, embedding, router, AutoPilot state, or replay DB writes are used.",
        ],
    }


def _write_outputs(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "sweep.jsonl").open("w", encoding="utf-8") as fh:
        for point in summary["sweeps"]:
            fh.write(json.dumps(point, sort_keys=True) + "\n")
    (out_dir / "pareto.json").write_text(
        json.dumps(summary["pareto"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: dict[str, Any]) -> str:
    source = summary["source"]
    lines = [
        "# DAR-4b Offline Routing Preference Sweep",
        "",
        f"- Protocol: `{summary['protocol']}`",
        f"- Measurement class: `{summary['measurement_class']}`",
        f"- Eligible decisions: `{source['eligible_decisions']}` / `{source['total_routing_events']}`",
        f"- Cost lambda: `{summary['cost_lambda']}`",
        "",
        "## Pareto Points",
        "",
        "| omega_perf | omega_cost | tau | mean_q | mean_cost | flip_vs_baseline | actions |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for point in summary["pareto"]:
        actions = ", ".join(
            f"{name}:{count}" for name, count in point["action_counts"].items()
        )
        lines.append(
            "| {omega_perf:.3f} | {omega_cost:.3f} | {tau:.3f} | "
            "{mean_q:.4f} | {mean_normalized_cost:.4f} | "
            "{flip_rate_vs_baseline:.2%} | {actions} |".format(
                actions=actions,
                **point,
            )
        )
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in summary["notes"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline DAR-4b routing preference/tau sweep"
    )
    parser.add_argument("--log-dir", "--progress-dir", dest="log_dir", default=None)
    parser.add_argument("--from", dest="from_date", default="")
    parser.add_argument("--to", dest="to_date", default="")
    parser.add_argument("--omega-grid", nargs="+", default=list(DEFAULT_OMEGA_GRID))
    parser.add_argument("--tau-grid", nargs="+", type=float, default=list(DEFAULT_TAU_GRID))
    parser.add_argument("--cost-lambda", type=float, default=DEFAULT_COST_LAMBDA)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--json", action="store_true", help="Print summary JSON")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    log_dir = Path(args.log_dir).expanduser() if args.log_dir else repo_root / "logs" / "progress"
    if not log_dir.exists():
        print(f"Error: log directory not found: {log_dir}", file=sys.stderr)
        return 1

    omega_grid = [parse_omega(item) for item in args.omega_grid]
    decisions, source_meta = load_frozen_decisions(
        log_dir,
        from_date=args.from_date,
        to_date=args.to_date,
        cost_lambda=args.cost_lambda,
    )
    if not decisions:
        print("Error: no eligible routing decisions found.", file=sys.stderr)
        return 1

    points = run_sweep(
        decisions,
        omega_grid=omega_grid,
        tau_grid=list(args.tau_grid),
        cost_lambda=args.cost_lambda,
    )
    summary = build_summary(
        decisions=decisions,
        source_meta=source_meta,
        points=points,
        omega_grid=omega_grid,
        tau_grid=list(args.tau_grid),
        cost_lambda=args.cost_lambda,
    )
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else repo_root / "orchestration" / "reports" / f"dar4b_sweep_{_utc_stamp()}"
    )
    _write_outputs(out_dir, summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Wrote DAR-4b sweep artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
