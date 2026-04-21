#!/usr/bin/env python3
"""Difficulty signal re-validation (NIB2-32 / reasoning-compression Action 3).

Validates the recalibrated difficulty-band thresholds (0.15 / 0.35) by
cross-correlating shadow-mode `difficulty_score` / `difficulty_band`
telemetry with benchmark pass/fail outcomes. Produces a go/no-go verdict
on moving the difficulty signal from shadow to enforce mode.

Required inputs (the script auto-detects whichever are present):

1. ``orchestrator/logs/seeding_diagnostics.jsonl`` — per-question eval
   results with ``passed`` / ``expected`` / ``suite`` / ``config``.
2. ``orchestrator/logs/progress/*.jsonl`` — routing decisions with
   ``routing_meta.difficulty_score`` and ``routing_meta.difficulty_band``.
3. Optional: raw Package A / Package D output JSONL at any user-specified
   path via ``--raw-routing-decisions``.

Important data-availability note (2026-04-21):

The reasoning-compression handoff (L93) references 635 Package A routing
decisions with shadow difficulty predictions, analyzed 2026-04-06 to
produce "92.3% easy, 0% hard at old thresholds." Those raw decisions
were analyzed in-session and NOT committed to disk — the `data/package_a/`
directories now contain only `env_flags.txt`. Additionally,
`seeding_diagnostics.jsonl` does not currently persist
`routing_meta.difficulty_score`, so the diagnostic stream alone cannot
be joined to shadow predictions.

Running this script today therefore reports INSUFFICIENT_DATA. The
script is deliverable-ready for when either:
  (a) a persistence fix lands that routes `routing_meta.difficulty_*`
      into `seeding_diagnostics.jsonl`, or
  (b) a fresh Package A/D run writes routing-decision JSONL to disk.

Verdict thresholds (from plan NIB2-32):
  - **ENFORCE-READY**    Spearman rho >= 0.3 between band ordinal
                          (easy=0, medium=1, hard=2) and error rate
  - **TUNE-THRESHOLDS**   0.15 <= |rho| < 0.3 — signal exists but needs
                          threshold adjustment; report optimal pair
  - **SIGNAL-NOISE**      |rho| < 0.15 — abandon enforce gate
  - **INSUFFICIENT-DATA** n < 100 joined decisions

Usage:
    python3 scripts/analysis/difficulty_signal_validation.py
    python3 scripts/analysis/difficulty_signal_validation.py --raw-routing-decisions path/to/decisions.jsonl
    python3 scripts/analysis/difficulty_signal_validation.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ORCH_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DIAGNOSTICS_PATH = ORCH_ROOT / "logs" / "seeding_diagnostics.jsonl"
PROGRESS_DIR = ORCH_ROOT / "logs" / "progress"

BAND_TO_ORDINAL = {"easy": 0, "medium": 1, "hard": 2}
MIN_SAMPLES = 100

# Thresholds from NIB2-32 plan
RHO_ENFORCE_READY = 0.30
RHO_TUNE_THRESHOLDS = 0.15


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return out


def _extract_difficulty(entry: dict[str, Any]) -> tuple[float | None, str | None]:
    """Return (difficulty_score, difficulty_band) from whichever place they live."""
    rm = entry.get("routing_meta") or {}
    if isinstance(rm, dict):
        s = rm.get("difficulty_score")
        b = rm.get("difficulty_band")
        if isinstance(s, (int, float)) and b:
            return float(s), str(b)
    s = entry.get("difficulty_score")
    b = entry.get("difficulty_band")
    if isinstance(s, (int, float)) and b:
        return float(s), str(b)
    return None, None


def _extract_outcome(entry: dict[str, Any]) -> bool | None:
    """Return True if the eval passed, False if failed, None if unknown."""
    if "passed" in entry:
        v = entry["passed"]
        if isinstance(v, bool):
            return v
    return None


def join_diagnostics_with_progress(
    diagnostics: list[dict[str, Any]],
    progress: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attempt a best-effort join by question_id."""
    # Index progress entries with difficulty data by question_id
    prog_by_qid: dict[str, tuple[float, str]] = {}
    for p in progress:
        qid = p.get("question_id") or p.get("task_id") or p.get("id")
        if not qid:
            continue
        s, b = _extract_difficulty(p)
        if s is None or b is None:
            continue
        prog_by_qid[str(qid)] = (s, b)

    joined: list[dict[str, Any]] = []
    for d in diagnostics:
        qid = d.get("question_id")
        if not qid or qid not in prog_by_qid:
            continue
        passed = _extract_outcome(d)
        if passed is None:
            continue
        score, band = prog_by_qid[qid]
        joined.append({
            "question_id": qid,
            "suite": d.get("suite"),
            "difficulty_score": score,
            "difficulty_band": band,
            "passed": passed,
        })
    return joined


def load_raw_routing_decisions(path: Path) -> list[dict[str, Any]]:
    """Load a raw routing-decisions JSONL (user-provided path)."""
    decisions = parse_jsonl(path)
    out: list[dict[str, Any]] = []
    for d in decisions:
        s, b = _extract_difficulty(d)
        passed = _extract_outcome(d)
        if s is None or b is None or passed is None:
            continue
        out.append({
            "question_id": d.get("question_id") or d.get("id"),
            "suite": d.get("suite"),
            "difficulty_score": s,
            "difficulty_band": b,
            "passed": passed,
        })
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation without scipy dependency."""
    n = len(xs)
    if n < 2:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    den_x = sum((r - mean_x) ** 2 for r in rx) ** 0.5
    den_y = sum((r - mean_y) ** 2 for r in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def compute_stats(joined: list[dict[str, Any]]) -> dict[str, Any]:
    if not joined:
        return {"n": 0}
    band_counts = Counter(j["difficulty_band"] for j in joined)
    band_error_rate: dict[str, dict[str, float]] = {}
    for band in ("easy", "medium", "hard"):
        rows = [j for j in joined if j["difficulty_band"] == band]
        if not rows:
            continue
        fails = sum(1 for j in rows if not j["passed"])
        band_error_rate[band] = {
            "n": len(rows),
            "fail_rate_pct": round(100.0 * fails / len(rows), 2),
            "pass_rate_pct": round(100.0 * (len(rows) - fails) / len(rows), 2),
        }

    # Spearman: band ordinal vs error indicator (1=fail, 0=pass)
    bands_numeric = [float(BAND_TO_ORDINAL.get(j["difficulty_band"], 0)) for j in joined]
    errors = [0.0 if j["passed"] else 1.0 for j in joined]
    rho = _spearman(bands_numeric, errors)

    return {
        "n": len(joined),
        "band_counts": dict(band_counts),
        "band_distribution_pct": {
            b: round(100.0 * c / len(joined), 2)
            for b, c in band_counts.items()
        },
        "band_error_rate": band_error_rate,
        "spearman_band_vs_error": round(rho, 4),
    }


def compute_verdict(stats: dict[str, Any]) -> dict[str, Any]:
    n = stats.get("n", 0)
    rho = abs(stats.get("spearman_band_vs_error", 0.0))
    if n < MIN_SAMPLES:
        return {
            "verdict": "INSUFFICIENT-DATA",
            "reason": f"n={n} < {MIN_SAMPLES} minimum samples",
        }
    if rho >= RHO_ENFORCE_READY:
        return {
            "verdict": "ENFORCE-READY",
            "reason": f"|rho|={rho:.3f} >= {RHO_ENFORCE_READY} threshold",
        }
    if rho >= RHO_TUNE_THRESHOLDS:
        return {
            "verdict": "TUNE-THRESHOLDS",
            "reason": f"{RHO_TUNE_THRESHOLDS} <= |rho|={rho:.3f} < {RHO_ENFORCE_READY} — signal exists but weak",
        }
    return {
        "verdict": "SIGNAL-NOISE",
        "reason": f"|rho|={rho:.3f} < {RHO_TUNE_THRESHOLDS} — difficulty signal has no predictive power",
    }


def format_human(stats: dict[str, Any], verdict: dict[str, Any], source: str) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("Difficulty Signal Re-Validation (NIB2-32)")
    lines.append("=" * 72)
    lines.append(f"Data source: {source}")
    lines.append(f"Joined samples: {stats.get('n', 0)}")
    lines.append("")
    if stats.get("n", 0) > 0:
        lines.append("Band distribution (target from recalibration: ~40/40/20):")
        for band in ("easy", "medium", "hard"):
            pct = stats["band_distribution_pct"].get(band, 0.0)
            count = stats["band_counts"].get(band, 0)
            lines.append(f"  {band:<7} {pct:5.1f}%  (n={count})")
        lines.append("")
        lines.append("Per-band pass/fail rate:")
        for band in ("easy", "medium", "hard"):
            info = stats["band_error_rate"].get(band)
            if info:
                lines.append(
                    f"  {band:<7} n={info['n']:<4} pass={info['pass_rate_pct']:5.1f}% "
                    f"fail={info['fail_rate_pct']:5.1f}%"
                )
        lines.append("")
        lines.append(
            f"Spearman(band_ordinal, error_indicator) = "
            f"{stats['spearman_band_vs_error']:+.4f}"
        )
    lines.append("")
    lines.append(f"VERDICT: {verdict['verdict']}")
    lines.append(f"  {verdict['reason']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--raw-routing-decisions", type=Path, default=None,
        help="Path to a user-provided routing_decisions.jsonl (bypasses "
        "diagnostics+progress join).",
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    source = ""
    if args.raw_routing_decisions:
        source = str(args.raw_routing_decisions)
        joined = load_raw_routing_decisions(args.raw_routing_decisions)
    else:
        diagnostics = parse_jsonl(DIAGNOSTICS_PATH)
        progress: list[dict[str, Any]] = []
        if PROGRESS_DIR.exists():
            for path in sorted(PROGRESS_DIR.glob("*.jsonl")):
                progress.extend(parse_jsonl(path))
        source = f"diagnostics({len(diagnostics)}) + progress({len(progress)})"
        joined = join_diagnostics_with_progress(diagnostics, progress)

    stats = compute_stats(joined)
    verdict = compute_verdict(stats)

    if args.json:
        print(json.dumps({"source": source, "stats": stats, "verdict": verdict}, indent=2))
    else:
        print(format_human(stats, verdict, source))

    return 0 if verdict["verdict"] != "SIGNAL-NOISE" else 1


if __name__ == "__main__":
    sys.exit(main())
