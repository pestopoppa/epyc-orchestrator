#!/usr/bin/env python3
"""Reviewer decision calibration report (H4 RC-4).

Given review-ledger rows (a ``review_ledger`` SQLite table and/or a decisions
JSONL, optionally joined to a near-miss corpus for gold labels), compute — per
``(reviewer config x grading model x rubric version x corpus version x domain)`` —
the reviewer calibration panel:

  * FA rate, FR rate, FA/FR ratio (first-class — overcorrection dominates)
  * acceptance rate, request-evidence yield, escalation precision (sampled; null-safe)
  * Brier, ECE, AUC   (confidence vs verdict-correctness)
  * Consistency Rate (test-retest >=2 runs) + pass^k for review decisions
  * parse-failure rate
  * Wilson score intervals on every rate

Emits a report JSON + a markdown table. Default outputs are observation-grade.
A caller may provide a run manifest whose measurement protocol is P-REV-1 and
``observation_only`` is false; in that case the report stamps the metrics as
decision-grade for that specific attested run. Directions are stamped in the
output (FA/FR/parse lower-better; acceptance/yield/CR higher-better).

Metric provenance (EV-tier reuse — RC-4 "reuse ... do not duplicate"):
  * ECE / ROC-AUC / Wilson interval are CONSOLIDATED into the single clean-room
    stdlib module ``src.llm_primitives.stat_tests`` (no numpy/sklearn dep); the
    thin ``wilson_interval``/``ece``/``roc_auc`` wrappers below delegate to it.
    That module also subsumes the prior duplicate copies in
    ``scripts/graph_router/*`` and ``scripts/maintenance/analyze_verifier_shadow.py``.
    The eval-tower EV-2 inline ECE/AUC (``scripts/autopilot/eval_tower.py``) is a
    separate Wave-2 swap and is intentionally NOT touched here.
  * Brier stays a local one-liner — it was never the source of drift and is out
    of ``stat_tests`` scope.
  * FA/FR classification is IMPORTED (not duplicated) from
    ``src.trace.review_ledger`` — the single polarity source of truth.

NO inference. Reads data, computes, writes report.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Repo root on path so `src.trace.review_ledger` imports whether this runs as a
# CLI (`python scripts/analysis/reviewer_calibration_report.py`) or is imported
# with `scripts/analysis` on sys.path (tests).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.llm_primitives.stat_tests import (  # noqa: E402
    expected_calibration_error as _stat_ece,
    roc_auc as _stat_roc_auc,
    wilson_interval as _stat_wilson_interval,
)
from src.trace.review_ledger import (  # noqa: E402
    decision_correct,
    gold_is_bad,
    has_gold,
    is_accept_like,
    is_false_accept,
    is_false_reject,
    is_parse_failure,
    is_reject_like,
    is_terminal,
)

OBSERVATION_STAMP = {
    "grade": "observation",
    "protocol": "P-REV-1 (DRAFT — pre-amendment; observation-grade, non-decision-gating)",
    "note": (
        "Pre-P-REV-1: every number here is an observation. It MUST NOT gate any "
        "keep/revert/deploy/promote of a reviewer configuration (RC-6a open)."
    ),
    "directions": {
        "fa_rate": "lower-better",
        "fr_rate": "lower-better",
        "parse_failure_rate": "lower-better",
        "acceptance_rate": "context",
        "request_evidence_yield": "higher-better",
        "escalation_precision": "higher-better",
        "consistency_rate": "higher-better",
        "ece": "lower-better",
        "brier": "lower-better",
        "auc": "higher-better",
    },
}

P_REV1_DECISION_STAMP = {
    "grade": "decision",
    "protocol": "P-REV-1",
    "note": (
        "P-REV-1: metrics are decision-grade for the material inputs and "
        "attestation recorded in the supplied run manifest."
    ),
    "directions": OBSERVATION_STAMP["directions"],
}

# Grouping key: (reviewer config x grading model x rubric version x corpus version x domain).
GROUP_FIELDS = ("reviewer_model_quant", "grading_model", "rubric_version", "corpus_id", "domain")


# --------------------------------------------------------------------------- #
# Metrics (stdlib; origin-noted — see module docstring)
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Consolidated: delegates to ``src.llm_primitives.stat_tests.wilson_interval``.
    ``z=1.96`` (this report's historical constant) is passed through explicitly,
    so the returned bounds are bit-for-bit identical to the prior inline copy.
    """
    return _stat_wilson_interval(successes, n, z)


def rate(successes: int, n: int) -> float | None:
    return (successes / n) if n > 0 else None


def brier(confidences: list[float], correct: list[float]) -> float | None:
    """Brier score = mean((p - y)^2).

    MIRRORS analyze_verifier_shadow._brier (numpy) in stdlib.
    """
    if not confidences:
        return None
    return sum((p - y) ** 2 for p, y in zip(confidences, correct)) / len(confidences)


def ece(confidences: list[float], correct: list[float], n_bins: int = 10) -> float | None:
    """Expected Calibration Error (equal-width bins).

    Consolidated: delegates to
    ``src.llm_primitives.stat_tests.expected_calibration_error``. Identical
    equal-width binning (final bin closed on the right so confidence==1.0 lands
    in it); returns ``None`` on empty input as before.
    """
    return _stat_ece(confidences, correct, n_bins)


def roc_auc(scores: list[float], labels: list[float]) -> float | None:
    """Rank-based ROC-AUC (Mann–Whitney) with tie averaging.

    Consolidated: delegates to ``src.llm_primitives.stat_tests.roc_auc``.
    Returns ``None`` when one class is absent (AUC undefined), as before.
    """
    return _stat_roc_auc(scores, labels)


# --------------------------------------------------------------------------- #
# Row loading
# --------------------------------------------------------------------------- #
def load_ledger_sqlite(db_path: str | Path) -> list[dict[str, Any]]:
    """Read all rows from a ``review_ledger`` table in a SQLite DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT * FROM review_ledger ORDER BY ts, id")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def load_decisions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read decision rows from a JSONL file (ledger-row-shaped dicts)."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def join_corpus_gold(
    rows: list[dict[str, Any]], corpus_path: str | Path
) -> list[dict[str, Any]]:
    """Fill missing gold fields on decision rows from a corpus rows.jsonl.

    Join key: ``candidate_id`` == corpus ``row_id`` (or ``candidate_id``). Only
    fills gold_* fields that are absent/empty on the decision row — never
    overwrites an oracle-resolved gold already on the decision.
    """
    corpus: dict[str, dict[str, Any]] = {}
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cr = json.loads(line)
            key = cr.get("row_id") or cr.get("candidate_id")
            if key:
                corpus[str(key)] = cr
    for row in rows:
        cid = str(row.get("candidate_id") or "")
        cr = corpus.get(cid)
        if not cr:
            continue
        if not row.get("gold_label"):
            row["gold_label"] = cr.get("gold_label")
        if not row.get("gold_source"):
            row["gold_source"] = cr.get("gold_source")
        if not row.get("gold_instrument_version"):
            row["gold_instrument_version"] = cr.get("gold_instrument_version")
        if not row.get("domain"):
            row["domain"] = cr.get("domain")
        if not row.get("corpus_id"):
            row["corpus_id"] = cr.get("corpus_id")
    return rows


def measurement_stamp_from_run_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return dict(OBSERVATION_STAMP)
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"run manifest must be a JSON object: {manifest_path}")
    stamp = dict(P_REV1_DECISION_STAMP if data.get("measurement_protocol") == "p_rev1" and data.get("observation_only") is False else OBSERVATION_STAMP)
    stamp["run_manifest"] = {
        "path": str(manifest_path),
        "measurement_protocol": data.get("measurement_protocol"),
        "observation_only": data.get("observation_only"),
        "protocol_attestation": data.get("protocol_attestation"),
    }
    if stamp["grade"] == "decision" and data.get("protocol_attestation"):
        stamp["note"] = f"{stamp['note']} Attestation: {data['protocol_attestation']}."
    return stamp


# --------------------------------------------------------------------------- #
# Per-group metric computation
# --------------------------------------------------------------------------- #
def _group_key(row: dict[str, Any]) -> tuple:
    return tuple(row.get(f) for f in GROUP_FIELDS)


def _rate_block(successes: int, n: int) -> dict[str, Any]:
    r = rate(successes, n)
    lo, hi = wilson_interval(successes, n)
    return {"rate": r, "successes": successes, "n": n, "wilson95": [lo, hi]}


def compute_group_metrics(rows: list[dict[str, Any]], *, k: int = 2) -> dict[str, Any]:
    """Compute the full calibration panel for one group's rows."""
    n_total = len(rows)
    terminal = [r for r in rows if is_terminal(r)]
    golded_terminal = [r for r in terminal if has_gold(r)]

    n_bad = [r for r in golded_terminal if gold_is_bad(r)]
    n_good = [r for r in golded_terminal if not gold_is_bad(r)]  # gold good
    fa = sum(1 for r in golded_terminal if is_false_accept(r))
    fr = sum(1 for r in golded_terminal if is_false_reject(r))

    fa_block = _rate_block(fa, len(n_bad))  # FA rate = FA / actually-bad
    fr_block = _rate_block(fr, len(n_good))  # FR rate = FR / actually-good
    fa_rate = fa_block["rate"]
    fr_rate = fr_block["rate"]
    if fa_rate is not None and fr_rate not in (None, 0.0):
        fa_fr_ratio: float | None = fa_rate / fr_rate
    else:
        fa_fr_ratio = None

    accepts = sum(1 for r in terminal if is_accept_like(r))
    acceptance_block = _rate_block(accepts, len(terminal))

    # request-evidence yield: of request_evidence verdicts with gold, fraction
    # that correctly targeted a defective (bad) candidate. Null-safe.
    re_rows = [r for r in rows if str(r.get("decision") or "").lower() == "request_evidence"]
    re_golded = [r for r in re_rows if has_gold(r)]
    re_hits = sum(1 for r in re_golded if gold_is_bad(r))
    request_evidence_yield = (
        _rate_block(re_hits, len(re_golded)) if re_golded else {"rate": None, "n": 0}
    )

    # escalation precision (sampled — supports null): of escalate verdicts with
    # gold, fraction where gold was bad (correctly escalated a real problem).
    esc_rows = [r for r in rows if str(r.get("decision") or "").lower() == "escalate"]
    esc_golded = [r for r in esc_rows if has_gold(r)]
    esc_hits = sum(1 for r in esc_golded if gold_is_bad(r))
    escalation_precision = (
        _rate_block(esc_hits, len(esc_golded)) if esc_golded else {"rate": None, "n": 0}
    )

    # Calibration on (confidence, verdict-correctness) over golded terminal rows.
    conf: list[float] = []
    corr: list[float] = []
    for r in golded_terminal:
        c = r.get("confidence")
        dc = decision_correct(r)
        if c is None or dc is None:
            continue
        conf.append(float(c))
        corr.append(1.0 if dc else 0.0)

    # parse-failure rate over all rows.
    parse_fail = sum(1 for r in rows if is_parse_failure(r))
    parse_block = _rate_block(parse_fail, n_total)

    # Consistency Rate + pass^k (test-retest >=2 runs of the same candidate).
    cr, passk = _consistency_and_passk(rows, k=k)

    return {
        "n_total": n_total,
        "n_terminal": len(terminal),
        "n_golded_terminal": len(golded_terminal),
        "n_actually_bad": len(n_bad),
        "n_actually_good": len(n_good),
        "fa_rate": fa_block,
        "fr_rate": fr_block,
        "fa_fr_ratio": fa_fr_ratio,
        "acceptance_rate": acceptance_block,
        "request_evidence_yield": request_evidence_yield,
        "escalation_precision": escalation_precision,
        "brier": brier(conf, corr),
        "ece": ece(conf, corr),
        "auc": roc_auc(conf, corr),
        "calibration_n": len(conf),
        "consistency_rate": cr,
        f"pass_hat_{k}": passk,
        "parse_failure_rate": parse_block,
    }


def _consistency_and_passk(
    rows: list[dict[str, Any]], *, k: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Test-retest Consistency Rate + pass^k over per-candidate run groups.

    Consistency Rate: among candidates with >=2 TERMINAL runs, the fraction whose
    verdicts all agree (a verdict-agreement metric; report accuracy alongside — CR
    can inflate to ~81% at near-random accuracy, intake-837).

    pass^k: among candidates with >=k golded-terminal runs, the fraction where the
    first k runs are ALL decision-correct (consistency-gated correctness).
    """
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cid = r.get("candidate_id")
        if cid is not None:
            by_candidate[str(cid)].append(r)

    multi = 0
    agree = 0
    for runs in by_candidate.values():
        term = [r for r in runs if is_terminal(r)]
        if len(term) >= 2:
            multi += 1
            verdicts = {str(r.get("decision") or "").lower() for r in term}
            if len(verdicts) == 1:
                agree += 1
    cr = {
        "rate": rate(agree, multi),
        "n_candidates_multi_run": multi,
        "n_all_agree": agree,
        "wilson95": list(wilson_interval(agree, multi)) if multi else None,
        "caveat": "CR can reach ~81% at near-random accuracy (intake-837); read with AUC/Brier.",
    }

    eligible = 0
    passed = 0
    for runs in by_candidate.values():
        gold_terminal = [r for r in runs if is_terminal(r) and has_gold(r)]
        correctness = [decision_correct(r) for r in gold_terminal]
        correctness = [c for c in correctness if c is not None]
        if len(correctness) >= k:
            eligible += 1
            if all(correctness[:k]):
                passed += 1
    passk = {
        "k": k,
        "rate": rate(passed, eligible),
        "n_candidates_ge_k": eligible,
        "n_pass": passed,
        "wilson95": list(wilson_interval(passed, eligible)) if eligible else None,
    }
    return cr, passk


def build_report(
    rows: list[dict[str, Any]],
    *,
    k: int = 2,
    instrument: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Group rows and compute per-group + overall metrics."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)

    group_reports = []
    for key, grows in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        group_reports.append(
            {
                "group": dict(zip(GROUP_FIELDS, key)),
                "metrics": compute_group_metrics(grows, k=k),
            }
        )

    return {
        "measurement": measurement or dict(OBSERVATION_STAMP),
        "instrument": instrument or {},
        "group_fields": list(GROUP_FIELDS),
        "n_rows": len(rows),
        "n_groups": len(group_reports),
        "overall": compute_group_metrics(rows, k=k),
        "groups": group_reports,
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        v = v.get("rate")
        if v is None:
            return "—"
    if isinstance(v, float):
        return f"{v * 100:.1f}%" if pct else f"{v:.3f}"
    return str(v)


def render_markdown(report: dict[str, Any], *, k: int = 2) -> str:
    lines: list[str] = []
    grade = str(report.get("measurement", {}).get("grade") or "observation")
    grade_label = "decision-grade" if grade == "decision" else "observation-grade"
    lines.append(f"# Reviewer calibration report ({grade_label})")
    lines.append("")
    lines.append(f"> {report['measurement']['note']}")
    lines.append("")
    lines.append(
        f"- rows: **{report['n_rows']}** · groups: **{report['n_groups']}** · "
        f"protocol: `{report['measurement']['protocol']}`"
    )
    if report.get("instrument"):
        lines.append(f"- instrument: `{json.dumps(report['instrument'], sort_keys=True)}`")
    lines.append("")
    header = (
        "| reviewer | grader | rubric | corpus | domain | n | FA | FR | FA/FR | "
        f"accept | yield | esc.prec | ECE | AUC | Brier | CR | pass^{k} | parse |"
    )
    sep = "|" + "---|" * 19
    lines.append(header)
    lines.append(sep)

    def row_line(label_group: dict[str, Any], m: dict[str, Any]) -> str:
        ratio = m["fa_fr_ratio"]
        return (
            f"| {label_group.get('reviewer_model_quant') or '—'} "
            f"| {label_group.get('grading_model') or '—'} "
            f"| {label_group.get('rubric_version') or '—'} "
            f"| {label_group.get('corpus_id') or '—'} "
            f"| {label_group.get('domain') or '—'} "
            f"| {m['n_total']} "
            f"| {_fmt(m['fa_rate'], pct=True)} "
            f"| {_fmt(m['fr_rate'], pct=True)} "
            f"| {(f'{ratio:.2f}' if ratio is not None else '—')} "
            f"| {_fmt(m['acceptance_rate'], pct=True)} "
            f"| {_fmt(m['request_evidence_yield'], pct=True)} "
            f"| {_fmt(m['escalation_precision'], pct=True)} "
            f"| {_fmt(m['ece'])} "
            f"| {_fmt(m['auc'])} "
            f"| {_fmt(m['brier'])} "
            f"| {_fmt(m['consistency_rate'], pct=True)} "
            f"| {_fmt(m[f'pass_hat_{k}'], pct=True)} "
            f"| {_fmt(m['parse_failure_rate'], pct=True)} |"
        )

    lines.append(row_line({"reviewer_model_quant": "**OVERALL**"}, report["overall"]))
    for g in report["groups"]:
        lines.append(row_line(g["group"], g["metrics"]))
    lines.append("")
    lines.append(
        "Directions: FA/FR/parse **lower-better**; accept=context; "
        "yield/esc.prec/CR/AUC **higher-better**; ECE/Brier **lower-better**. "
        "FA/FR ratio is first-class (overcorrection prior FR≫FA)."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ledger", help="Path to a SQLite DB with a review_ledger table.")
    src.add_argument("--decisions", help="Path to a decisions JSONL (ledger-row-shaped).")
    ap.add_argument("--corpus", help="Optional near-miss corpus rows.jsonl to join gold labels from.")
    ap.add_argument("--k", type=int, default=2, help="k for pass^k (default 2).")
    ap.add_argument("--out-json", help="Write report JSON here.")
    ap.add_argument("--out-md", help="Write markdown table here.")
    ap.add_argument(
        "--run-manifest",
        help=(
            "Optional runner manifest. If it records measurement_protocol=p_rev1 "
            "and observation_only=false, stamp this report decision-grade."
        ),
    )
    ap.add_argument("--print", dest="do_print", action="store_true", help="Print markdown to stdout.")
    args = ap.parse_args(argv)

    if args.ledger:
        rows = load_ledger_sqlite(args.ledger)
        source_desc = {"kind": "ledger_sqlite", "path": str(args.ledger)}
    else:
        rows = load_decisions_jsonl(args.decisions)
        source_desc = {"kind": "decisions_jsonl", "path": str(args.decisions)}
    if args.corpus:
        rows = join_corpus_gold(rows, args.corpus)
        source_desc["corpus"] = str(args.corpus)
    if args.run_manifest:
        source_desc["run_manifest"] = str(args.run_manifest)

    report = build_report(
        rows,
        k=args.k,
        instrument={"source": source_desc},
        measurement=measurement_stamp_from_run_manifest(args.run_manifest),
    )
    md = render_markdown(report, k=args.k)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.out_md:
        Path(args.out_md).write_text(md)
    if args.do_print or not (args.out_json or args.out_md):
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
