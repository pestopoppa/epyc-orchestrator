#!/usr/bin/env python3
"""Zero-inference item analytics over the autopilot journal.

Current journals persist per-suite quality/counts but not per-question vectors.
This script therefore emits suite-level saturation/brokenness diagnostics today
and automatically promotes to per-qid analytics when future rows carry
``question_results`` / ``per_question_results`` from the evidence ledger.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.autopilot_core.journal_reconstruction import fold_supersession_events


DEFAULT_JOURNAL = Path("orchestration/autopilot_journal.jsonl")
DEFAULT_OUT_DIR = Path("orchestration/reports")
PINNED_ZERO_WATCHLIST = (
    "mode_advantage_hard",
    "usaco",
    "vl",
    "instruction_precision",
    "bigcodebench",
)
KNOWN_ARTIFACT_VERDICTS = {
    "usaco": (
        "artifact",
        "Fable5 audit found seed-42 T1 rows with expected='' behind a text-scorer gate; "
        "W4 replaced empty-expected sampled rows for future evals.",
    ),
    "instruction_precision": (
        "artifact",
        "Fable5 audit found fixed-pool expected='' instruction-precision rows; "
        "historical nonzero values are tier/sentinel mixing, not proof of item health.",
    ),
    "vl": (
        "artifact",
        "Fable5 follow-up traced historical VL zeros to routing/OCR plumbing; W4 repaired "
        "image bypass and OCR MTMD fallback, so historical pinned-zero is artifact-contaminated.",
    ),
    "bigcodebench": (
        "artifact",
        "Fable5 audit found the fixed BigCodeBench rows require pandas, which was absent in "
        "the orchestrator environment; W4 added the dependency.",
    ),
}


@dataclass
class SuiteStats:
    suite: str
    observations: int = 0
    question_evals: int = 0
    estimated_correct: float = 0.0
    quality_values: list[float] = field(default_factory=list)
    zero_quality_trials: int = 0
    saturated_trials: int = 0
    error_trials: int = 0

    def add(self, quality: float, count: int, had_error: bool) -> None:
        self.observations += 1
        self.question_evals += count
        self.estimated_correct += (quality / 3.0) * count
        self.quality_values.append(quality)
        if quality <= 0.0:
            self.zero_quality_trials += 1
        if quality >= 3.0:
            self.saturated_trials += 1
        if had_error:
            self.error_trials += 1

    @property
    def p_correct(self) -> float | None:
        if self.question_evals <= 0:
            return None
        return self.estimated_correct / self.question_evals

    @property
    def mean_quality(self) -> float | None:
        if not self.quality_values:
            return None
        return mean(self.quality_values)

    @property
    def quality_std(self) -> float:
        if len(self.quality_values) < 2:
            return 0.0
        return pstdev(self.quality_values)

    def flags(self, min_observations: int) -> list[str]:
        flags: list[str] = []
        p_correct = self.p_correct
        if self.observations >= min_observations and p_correct is not None:
            if p_correct <= 0.05:
                flags.append("pinned_zero_or_broken")
            elif p_correct >= 0.95:
                flags.append("saturated")
            elif self.quality_std >= 1.0:
                flags.append("high_variance")
        if self.error_trials:
            flags.append("errors_present")
        return flags


def _suite_verdict(suite: SuiteStats, min_observations: int) -> dict[str, Any]:
    flags = suite.flags(min_observations)
    if suite.suite in KNOWN_ARTIFACT_VERDICTS:
        verdict, basis = KNOWN_ARTIFACT_VERDICTS[suite.suite]
    elif "pinned_zero_or_broken" in flags:
        verdict = "genuinely_hard_candidate"
        basis = (
            "Pinned near zero with no structural artifact documented in the Fable5 audit; "
            "requires per-qid vectors/core_v2 calibration before treating it as a useful "
            "discriminator."
        )
    elif "saturated" in flags:
        verdict = "saturated_low_discrimination"
        basis = (
            "Suite is near-always correct in this window and contributes little decision signal."
        )
    else:
        verdict = "not_pinned"
        basis = "Suite did not cross pinned-zero or saturation thresholds in this window."
    return {"verdict": verdict, "basis": basis, "flags": flags}


@dataclass
class QuestionStats:
    question_id: str
    suite: str
    observations: int = 0
    correct: int = 0
    errors: int = 0
    kept_correct: int = 0
    kept_total: int = 0
    reverted_correct: int = 0
    reverted_total: int = 0

    def add(self, *, correct: bool, error: bool, decision: str) -> None:
        self.observations += 1
        self.correct += int(correct)
        self.errors += int(error)
        if decision == "keep":
            self.kept_total += 1
            self.kept_correct += int(correct)
        elif decision == "revert":
            self.reverted_total += 1
            self.reverted_correct += int(correct)

    @property
    def p_correct(self) -> float:
        return self.correct / self.observations if self.observations else 0.0

    @property
    def discrimination(self) -> float | None:
        if self.kept_total == 0 or self.reverted_total == 0:
            return None
        return (self.kept_correct / self.kept_total) - (self.reverted_correct / self.reverted_total)

    def flags(self, min_observations: int) -> list[str]:
        flags: list[str] = []
        if self.observations >= min_observations:
            if self.p_correct <= 0.05:
                flags.append("pinned_zero_or_broken")
            elif self.p_correct >= 0.95:
                flags.append("saturated")
            if self.discrimination is not None and self.discrimination < -0.15:
                flags.append("negative_discrimination")
        if self.errors:
            flags.append("errors_present")
        return flags


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) or math.isinf(result) else result


def load_journal(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(
                    {
                        "trial_id": None,
                        "timestamp": None,
                        "parse_error": f"line {line_no}: {exc}",
                    }
                )
                continue
            rows.append(row)
    folded_rows, _ = fold_supersession_events(rows)
    return [
        row for row in folded_rows if row.get("trial_id") is not None
    ] + parse_errors


def _details(row: dict[str, Any]) -> dict[str, Any]:
    details = row.get("eval_details") or {}
    return details if isinstance(details, dict) else {}


def _nested_details(row: dict[str, Any]) -> dict[str, Any]:
    details = _details(row).get("details") or {}
    return details if isinstance(details, dict) else {}


def _per_suite_quality(row: dict[str, Any]) -> dict[str, float]:
    raw = _details(row).get("per_suite_quality") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): _safe_float(v) for k, v in raw.items()}


def _per_suite_counts(row: dict[str, Any]) -> dict[str, int]:
    raw = _nested_details(row).get("per_suite_counts") or {}
    if not isinstance(raw, dict):
        return {}
    counts: dict[str, int] = {}
    for key, value in raw.items():
        try:
            counts[str(key)] = max(0, int(value))
        except (TypeError, ValueError):
            counts[str(key)] = 0
    return counts


def _question_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    details = _details(row)
    nested = _nested_details(row)
    for key in ("question_results", "per_question_results", "per_question"):
        raw = details.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        raw = nested.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def _window_rows(
    rows: list[dict[str, Any]],
    *,
    last_trials: int,
    days: int,
    now: datetime | None = None,
) -> dict[str, list[dict[str, Any]]]:
    usable = [row for row in rows if row.get("trial_id") is not None]
    usable.sort(key=lambda row: int(row.get("trial_id") or -1))
    windows = {
        f"last_{last_trials}_trials": usable[-last_trials:] if last_trials > 0 else usable,
    }
    if days > 0:
        ts_values = [_parse_ts(row.get("timestamp")) for row in usable]
        ts_values = [ts for ts in ts_values if ts is not None]
        end = now or (max(ts_values) if ts_values else datetime.now(timezone.utc))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        start = end - timedelta(days=days)
        windows[f"last_{days}_days"] = [
            row
            for row in usable
            if (ts := _parse_ts(row.get("timestamp"))) is not None and ts >= start
        ]
    return windows


def _summarize_suite_window(
    rows: list[dict[str, Any]],
    *,
    min_observations: int,
) -> list[dict[str, Any]]:
    suites: dict[str, SuiteStats] = {}
    for row in rows:
        qualities = _per_suite_quality(row)
        counts = _per_suite_counts(row)
        had_error = bool(_safe_float(_nested_details(row).get("errors"), 0.0) > 0)
        for suite, quality in qualities.items():
            stats = suites.setdefault(suite, SuiteStats(suite=suite))
            stats.add(quality=quality, count=counts.get(suite, 0), had_error=had_error)

    summary = []
    for suite in sorted(suites.values(), key=lambda s: (s.p_correct or -1.0, s.suite)):
        verdict = _suite_verdict(suite, min_observations)
        summary.append(
            {
                "suite": suite.suite,
                "observations": suite.observations,
                "question_evals": suite.question_evals,
                "p_correct": suite.p_correct,
                "mean_quality": suite.mean_quality,
                "quality_std": suite.quality_std,
                "zero_quality_trials": suite.zero_quality_trials,
                "saturated_trials": suite.saturated_trials,
                "error_trials": suite.error_trials,
                "artifact_verdict": verdict["verdict"],
                "verdict_basis": verdict["basis"],
                "flags": verdict["flags"],
            }
        )
    return summary


def _summarize_question_window(
    rows: list[dict[str, Any]],
    *,
    min_observations: int,
) -> list[dict[str, Any]]:
    questions: dict[tuple[str, str], QuestionStats] = {}
    for row in rows:
        decision = str(row.get("keep_revert_decision") or "")
        for result in _question_results(row):
            qid = str(result.get("question_id") or result.get("id") or "")
            suite = str(result.get("suite") or "unknown")
            if not qid:
                continue
            key = (suite, qid)
            stats = questions.setdefault(key, QuestionStats(question_id=qid, suite=suite))
            stats.add(
                correct=bool(result.get("correct")),
                error=bool(result.get("error")),
                decision=decision,
            )

    summary = []
    for question in sorted(
        questions.values(),
        key=lambda q: (q.p_correct, q.suite, q.question_id),
    ):
        summary.append(
            {
                "question_id": question.question_id,
                "suite": question.suite,
                "observations": question.observations,
                "correct": question.correct,
                "errors": question.errors,
                "p_correct": question.p_correct,
                "discrimination": question.discrimination,
                "flags": question.flags(min_observations),
            }
        )
    return summary


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    last_trials: int = 100,
    days: int = 7,
    min_observations: int = 5,
    now: datetime | None = None,
) -> dict[str, Any]:
    windows = _window_rows(rows, last_trials=last_trials, days=days, now=now)
    window_reports: dict[str, Any] = {}
    for name, window in windows.items():
        question_summary = _summarize_question_window(
            window,
            min_observations=min_observations,
        )
        suite_summary = _summarize_suite_window(
            window,
            min_observations=min_observations,
        )
        suites_by_name = {row["suite"]: row for row in suite_summary}
        window_reports[name] = {
            "trial_count": len(window),
            "trial_id_min": min((row.get("trial_id") for row in window), default=None),
            "trial_id_max": max((row.get("trial_id") for row in window), default=None),
            "per_qid_available": bool(question_summary),
            "per_qid_limitation": (
                None
                if question_summary
                else "journal rows do not persist per-question results yet; N2 ledger must add them"
            ),
            "suite_summary": suite_summary,
            "question_summary": question_summary,
            "flagged_suites": [row for row in suite_summary if row["flags"]],
            "flagged_questions": [row for row in question_summary if row["flags"]],
            "watchlist_verdicts": [
                suites_by_name[suite] for suite in PINNED_ZERO_WATCHLIST if suite in suites_by_name
            ],
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_rows": len(rows),
        "parameters": {
            "last_trials": last_trials,
            "days": days,
            "min_observations": min_observations,
        },
        "windows": window_reports,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Autopilot Item Analytics",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Source rows: `{report['source_rows']}`",
        "",
    ]
    for window_name, window in report["windows"].items():
        lines.extend(
            [
                f"## {window_name}",
                "",
                f"Trials: `{window['trial_count']}` "
                f"(`{window['trial_id_min']}` to `{window['trial_id_max']}`)",
                "",
            ]
        )
        if not window["per_qid_available"]:
            lines.append(f"Per-qid analytics: **unavailable** — {window['per_qid_limitation']}.")
            lines.append("")
        if window["watchlist_verdicts"]:
            lines.extend(
                [
                    "### Pinned-Zero Watchlist Verdicts",
                    "",
                    "| suite | p_correct | verdict | basis |",
                    "|---|---:|---|---|",
                ]
            )
            for suite in window["watchlist_verdicts"]:
                lines.append(
                    "| {suite} | {pc} | {verdict} | {basis} |".format(
                        suite=suite["suite"],
                        pc=_fmt(suite["p_correct"]),
                        verdict=suite["artifact_verdict"],
                        basis=suite["verdict_basis"],
                    )
                )
            lines.append("")
        if window["flagged_suites"]:
            lines.extend(
                [
                    "### Flagged Suites",
                    "",
                    "| suite | obs | q_evals | p_correct | mean_q | std | verdict | flags |",
                    "|---|---:|---:|---:|---:|---:|---|---|",
                ]
            )
            for suite in window["flagged_suites"]:
                lines.append(
                    "| {suite} | {obs} | {qevals} | {pc} | {mean_q} | {std} | "
                    "{verdict} | {flags} |".format(
                        suite=suite["suite"],
                        obs=suite["observations"],
                        qevals=suite["question_evals"],
                        pc=_fmt(suite["p_correct"]),
                        mean_q=_fmt(suite["mean_quality"]),
                        std=_fmt(suite["quality_std"]),
                        verdict=suite["artifact_verdict"],
                        flags=", ".join(suite["flags"]),
                    )
                )
            lines.append("")
        if window["flagged_questions"]:
            lines.extend(
                [
                    "### Flagged Questions",
                    "",
                    "| suite | qid | obs | p_correct | discrimination | flags |",
                    "|---|---|---:|---:|---:|---|",
                ]
            )
            for question in window["flagged_questions"]:
                lines.append(
                    "| {suite} | {qid} | {obs} | {pc} | {disc} | {flags} |".format(
                        suite=question["suite"],
                        qid=question["question_id"],
                        obs=question["observations"],
                        pc=_fmt(question["p_correct"]),
                        disc=_fmt(question["discrimination"]),
                        flags=", ".join(question["flags"]),
                    )
                )
            lines.append("")
        if not window["flagged_suites"] and not window["flagged_questions"]:
            lines.append("No suites or questions crossed the configured flag thresholds.")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze autopilot item/suite stability")
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="item_analytics_latest")
    parser.add_argument("--last-trials", type=int, default=100)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--min-observations", type=int, default=5)
    parser.add_argument("--print-md", action="store_true", help="Print markdown to stdout")
    args = parser.parse_args(argv)

    rows = load_journal(args.journal)
    report = analyze_rows(
        rows,
        last_trials=args.last_trials,
        days=args.days,
        min_observations=args.min_observations,
    )
    json_path, md_path = write_report(report, args.out_dir, args.stem)
    if args.print_md:
        print(render_markdown(report), end="")
    else:
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
