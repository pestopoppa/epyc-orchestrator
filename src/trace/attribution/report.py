"""Per-rung accuracy and token table, and the aggregated-ranking nDCG@5 table (HIL-17 acceptance shapes).

Accuracy is against each case's consensus map (``metrics.pick_accuracy``): exact = a top-consensus
step, within1, hit = any flagged step; abstentions count as misses and are reported. Agent accuracy is
scored only on cases with a gold agent set. Tokens: total and mean per case over the cases where the
rung RAN (and, for rung 3, the share of cases that escalated to it). Scope fields say what the
numbers are: which judge(s), whether quantised, whether the judge saw the reference answer.
"""

from __future__ import annotations

import json
from typing import Any

from src.trace.attribution.aggregate import AggregateResult
from src.trace.attribution.case import AttributionCase
from src.trace.attribution.ladder import RUNGS, LadderResult
from src.trace.attribution.metrics import mean, pick_accuracy


def _rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _agent_of(case: AttributionCase, step: int | None) -> str | None:
    if step is None:
        return None
    e = case.trajectory.by_step().get(step)
    return e.agent_name if e else None


def _row(
    name: str, picks: list[tuple[AttributionCase, int | None]], ran: int, n_cases: int, tokens: dict
) -> dict:
    acc = [pick_accuracy(p, c.relevance) for c, p in picks]
    agent = [(_agent_of(c, p) in c.gold_agents) for c, p in picks if c.gold_agents]
    return {
        "rung": name,
        "N": len(picks),
        "ran_on": ran,
        "ran_share": round(ran / n_cases, 4) if n_cases else None,
        "exact": _rate([a["exact"] for a in acc]),
        "within1": _rate([a["within1"] for a in acc]),
        "hit": _rate([a["hit"] for a in acc]),
        "abstain": _rate([a["abstain"] for a in acc]),
        "agent_acc": _rate(agent),
        "agent_N": len(agent),
        **tokens,
        "tokens_per_case_ran": round(tokens["total_tokens"] / ran, 1) if ran else None,
        "tokens_per_case_all": round(tokens["total_tokens"] / n_cases, 1) if n_cases else None,
    }


def rung_table(cases: list[AttributionCase], results: list[LadderResult]) -> list[dict]:
    """One row per rung (standalone pick on the cases it ran on) plus the ladder's final answer."""
    by = {c.case_id: c for c in cases}
    scored = [(by[r.case_id], r) for r in results if by[r.case_id].relevance]
    rows = []
    for rung in RUNGS:
        tok: dict[str, int] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls_without_usage": 0,
        }
        picks, ran = [], 0
        for c, r in scored:
            rr = r.rungs[rung]
            for k, v in rr.tokens.as_dict().items():
                tok[k] += v
            if rr.ran:
                ran += 1
                picks.append((c, rr.step))
        rows.append(_row(rung, picks, ran, len(scored), tok))
    tot = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls_without_usage": 0,
    }
    for _, r in scored:
        for k, v in r.tokens_total().as_dict().items():
            tot[k] += v
    rows.append(
        _row("ladder_final", [(c, r.final_step) for c, r in scored], len(scored), len(scored), tot)
    )
    return rows


def ndcg_table(results: list[AggregateResult]) -> dict:
    keys = sorted({k for r in results for k in r.ndcg})
    out: dict[str, Any] = {"N": len(results)}
    for k in keys:
        vals = [r.ndcg.get(k) for r in results]
        m = mean(vals)
        out[k] = None if m is None else round(m, 4)
        out[f"{k}_N"] = sum(v is not None for v in vals)
    ks = sum(r.tokens_ksample.total for r in results)
    t0 = sum(r.tokens_temp0.total for r in results)
    out["tokens_ksample_total"], out["tokens_temp0_total"] = ks, t0
    out["tokens_ksample_per_case"] = round(ks / len(results), 1) if results else None
    out["tokens_temp0_per_case"] = round(t0 / len(results), 1) if results else None
    fam: dict[str, dict] = {}
    for r in results:
        for name, d in r.per_family.items():
            f = fam.setdefault(
                name,
                {
                    "family": d["family"],
                    "quantised": d["quantised"],
                    "samples": 0,
                    "parsed": 0,
                    "parse_failures": 0,
                    "out_of_range_steps": 0,
                },
            )
            for k in ("samples", "parsed", "parse_failures", "out_of_range_steps"):
                f[k] += d[k]
    out["per_family"] = fam
    return out


_COLS = (
    "rung",
    "N",
    "ran_on",
    "exact",
    "within1",
    "hit",
    "abstain",
    "agent_acc",
    "calls",
    "total_tokens",
    "tokens_per_case_ran",
    "tokens_per_case_all",
)


def markdown(report: dict) -> str:
    lines = [f"# Attribution ladder: {report.get('set', '?')}", ""]
    for k in ("scope", "judges", "with_reference", "policy"):
        if k in report:
            lines.append(f"- {k}: `{json.dumps(report[k], sort_keys=True)}`")
    lines += ["", "| " + " | ".join(_COLS) + " |", "|" + "---|" * len(_COLS)]
    for row in report.get("rungs", []):
        lines.append(
            "| " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in _COLS) + " |"
        )
    if report.get("aggregate"):
        a = report["aggregate"]
        lines += [
            "",
            "## Aggregated ranking (nDCG@5)",
            "",
            "| arm | exp | linear |",
            "|---|---|---|",
        ]
        for arm in ("ksample", "temp0", "random_expected", "random_mc"):
            lines.append(f"| {arm} | {a.get(f'{arm}_exp')} | {a.get(f'{arm}_linear')} |")
    return "\n".join(lines) + "\n"
