"""EV-19: scorer self-check on gold answers, per suite (zero inference, read-only).

Runs each suite's own scorer on that suite's own reference answers. A scorer that marks its
own gold wrong has a verifier-validity defect, and that defect bounds every pass rate it
produces (intake-2274#record). Pure functions here; the CLI driver is
``scripts/autopilot/gold_self_check.py``.

Outcome per question:
  not_checkable:<why>   no inference-free self-check exists (llm_judge, rubric, code_execution
                        whose `expected` is an entry-point name rather than a solution, ...)
  ok_bare               the bare gold string is scored correct
  ok_wrapped            bare gold is scored wrong but gold inside a canonical answer wrapper is
                        scored correct (format sensitivity of the extractor, not a defect)
  wrong                 bare gold and every wrapper scored wrong or unparseable: the scorer
                        cannot recognise its own reference answer
  scorer_error          the scorer raised or reported unavailable

`rate` is wrong / checkable. A suite with rate > 0 is flagged; the flag is surfaced in the
AutoPilot quality-delta report text via ``header_text``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

WRAPPERS = (
    "solution = {g}",
    "Answer: {g}",
    "ANSWER: {g}",
    "The answer is {g}",
    "Final answer: {g}",
    "<answer>{g}</answer>",
    "\\boxed{{{g}}}",
    "```\n{g}\n```",
)
NOT_CHECKABLE_METHODS = {
    "llm_judge": "needs a judge model",
    "code_execution": "expected is not a reference solution",
}
DEFAULT_REPORT = Path(__file__).resolve().parents[2] / "data" / "autopilot" / "gold_self_check.json"


def classify(q: Mapping[str, Any], score) -> tuple[str, str]:
    """(outcome, detail) for one pool question. `score(answer, expected, method, config)` ->
    (verdict|None, error|None), i.e. seeding_scoring.score_answer_or_error."""
    method = str(q.get("scoring_method") or "")
    expected = q.get("expected")
    if expected is None or str(expected).strip() == "":
        return "not_checkable", "no_expected"
    if method in NOT_CHECKABLE_METHODS:
        return "not_checkable", method
    if q.get("scoring") not in (None, "", "[]", []) and not method:
        return "not_checkable", "rubric"
    if not method:
        return "not_checkable", "no_scoring_method"
    config = q.get("scoring_config") or {}
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except ValueError:
            return "scorer_error", "scoring_config_unparseable"
    gold = str(expected)
    try:
        verdict, err = score(gold, gold, method, dict(config))
        if err is None and verdict:
            return "ok_bare", ""
        first = err or "scored_false"
        for w in WRAPPERS:
            v, e = score(w.format(g=gold), gold, method, dict(config))
            if e is None and v:
                return "ok_wrapped", w
        return "wrong", first
    except Exception as exc:  # noqa: BLE001 - a raising scorer is itself a finding
        return "scorer_error", f"{type(exc).__name__}: {exc}"[:200]


def summarize(results: Iterable[tuple[str, str, str]]) -> dict[str, Any]:
    """results: (suite, outcome, detail). Per-suite table + flagged suites."""
    per: dict[str, Counter] = {}
    reasons: dict[str, Counter] = {}
    for suite, outcome, detail in results:
        per.setdefault(suite, Counter())[outcome] += 1
        if outcome in ("wrong", "scorer_error", "not_checkable"):
            reasons.setdefault(suite, Counter())[f"{outcome}:{detail}"] += 1
    table = {}
    for suite in sorted(per):
        c = per[suite]
        checkable = c["ok_bare"] + c["ok_wrapped"] + c["wrong"] + c["scorer_error"]
        bad = c["wrong"] + c["scorer_error"]
        table[suite] = {
            "n": sum(c.values()),
            "checkable": checkable,
            "ok_bare": c["ok_bare"],
            "ok_wrapped": c["ok_wrapped"],
            "wrong": c["wrong"],
            "scorer_error": c["scorer_error"],
            "not_checkable": c["not_checkable"],
            "rate": (bad / checkable) if checkable else None,
            "bare_wrong_rate": ((checkable - c["ok_bare"]) / checkable) if checkable else None,
            "top_reasons": dict(reasons.get(suite, Counter()).most_common(3)),
        }
    return {"suites": table, "flagged": sorted(s for s, r in table.items() if r["rate"])}


def header_text(suites: Iterable[str], path: str | Path | None = None) -> str:
    """Report-header text flagging evaluated suites whose scorer fails its own gold; '' when
    no report exists or nothing relevant is flagged (default behaviour unchanged)."""
    try:
        p = Path(path or os.environ.get("AUTOPILOT_GOLD_SELF_CHECK_PATH") or DEFAULT_REPORT)
        if not p.is_file():
            return ""
        rep = json.loads(p.read_text())
        wanted = set(suites)
        bits = []
        for s in rep.get("flagged", []):
            if s in wanted:
                r = rep["suites"][s]
                bits.append(f"{s} {100 * r['rate']:.1f}% ({r['wrong'] + r['scorer_error']}/{r['checkable']})")
        if not bits:
            return ""
        return "SCORER GOLD SELF-CHECK nonzero (scorer fails its own reference answers): " + "; ".join(bits) + ". "
    except Exception:  # noqa: BLE001 - report text only
        return ""
