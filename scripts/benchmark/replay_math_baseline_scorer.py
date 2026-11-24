#!/usr/bin/env python3
"""Deterministic scorer replay of a banked EV-11 math re-baseline (no inference).

MEASUREMENT.md "Deterministic replay before regeneration": a scorer change is
re-baselined by re-running the FIXED scorer over BANKED model outputs, never by
regenerating. This replays every ``question_result`` row of an
``eval_batch_serving_evaltower_window.py --mode math_rebaseline`` report directory
(``question_results.<label>.jsonl``) through the live ``debug_scorer.score_answer``,
joining each row to its gold from the on-disk GSM8K test split / MATH-500 exactly
as ``dataset_adapter_modules/math_adapter.py`` derives it.

It NEVER edits the source report (historical records are appended to, not
rescored in place). It writes a NEW directory holding:

* ``replay.jsonl`` — one row per question: qid, source, banked verdict, replayed
  verdict (``true``/``false``/``excluded``/``error:<Exc>``), changed flag;
* ``summary.json`` / ``summary.md`` — per label: n, banked vs replayed accuracy over
  the replay-scorable rows, flip counts in each direction, exclusions, plus the
  scorer semantics version and the sha256 of every input.

Usage::

    python scripts/benchmark/replay_math_baseline_scorer.py \\
        --report-dir orchestration/reports/eval_tower_math_rebaseline_E7c \\
        --out-dir orchestration/reports/eval_tower_math_rebaseline_E7c_replay_scorer_v2

Default OFF: an operator/CLI tool; nothing in the runtime imports it.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HUB = Path("/mnt/raid0/llm/hf-home/hub")
_SCORER_KEY = "epyc_orch_debug_scorer"


DEFAULT_SCORER = REPO_ROOT / "scripts" / "benchmark" / "debug_scorer.py"


def _load_scorer(path: Path = DEFAULT_SCORER):
    cached = sys.modules.get(_SCORER_KEY)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_SCORER_KEY, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load debug_scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SCORER_KEY] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gsm8k_gold(answer_text: str) -> str:
    # Mirrors MathAdapter._extract_gsm8k_answer.
    match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL)
    if not match:
        match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def load_golds() -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """``{question_id: (expected, scoring_config extraction_mode)}`` plus input digests."""
    import pyarrow.parquet as pq

    parquet = Path(sorted(glob.glob(str(HUB / "datasets--gsm8k/snapshots/*/main/test-*.parquet")))[0])
    jsonl = Path(sorted(glob.glob(str(HUB / "datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl")))[0])
    golds: dict[str, tuple[str, str]] = {}
    for idx, row in enumerate(pq.read_table(parquet).to_pylist()):
        golds[f"gsm8k_{idx:05d}"] = (_gsm8k_gold(row["answer"]), "expr")
    for idx, line in enumerate(l for l in jsonl.read_text().splitlines() if l.strip()):
        row = json.loads(line)
        golds[f"math500_{row.get('subject', '')}_{idx:05d}"] = (str(row.get("answer", "")), "latex")
    return golds, {"gsm8k_test_parquet": _sha256(parquet), "math500_test_jsonl": _sha256(jsonl)}


def replay_file(path: Path, golds: dict[str, tuple[str, str]], scorer: Any) -> list[dict[str, Any]]:
    unavailable = getattr(scorer, "ScoringUnavailableError", ())
    parse_error = getattr(scorer, "AnswerParseError", ())
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("row_type") != "question_result":
            continue
        res = row.get("result") or {}
        qid = str(res.get("question_id", ""))
        banked = res.get("correct")
        if qid not in golds:
            out.append({"question_id": qid, "label": row.get("label"), "banked": banked, "replayed": "no_gold"})
            continue
        expected, mode = golds[qid]
        method = str(res.get("scoring_method") or "math_verify")
        try:
            verdict: Any = bool(
                scorer.score_answer(str(row.get("answer") or ""), expected, method, {"extraction_mode": mode})
            )
        except unavailable:
            verdict = "excluded"
        except parse_error:
            verdict = "excluded"
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            verdict = f"error:{type(exc).__name__}"
        out.append(
            {
                "question_id": qid,
                "label": row.get("label"),
                "source": qid.split("_", 1)[0],
                "banked": banked,
                "replayed": verdict,
                "changed": isinstance(verdict, bool) and isinstance(banked, bool) and verdict != banked,
            }
        )
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, dict[str, Any]] = {}
    for label in sorted({str(r["label"]) for r in rows}):
        lr = [r for r in rows if str(r["label"]) == label]
        both = [r for r in lr if isinstance(r["banked"], bool) and isinstance(r["replayed"], bool)]
        per_source: dict[str, Any] = {}
        for src in sorted({r.get("source", "") for r in both}):
            sb = [r for r in both if r.get("source") == src]
            per_source[src] = {
                "n_scorable": len(sb),
                "banked_correct": sum(r["banked"] for r in sb),
                "replayed_correct": sum(r["replayed"] for r in sb),
                "flips_true_to_false": sum(r["banked"] and not r["replayed"] for r in sb),
                "flips_false_to_true": sum((not r["banked"]) and r["replayed"] for r in sb),
            }
        by_label[label] = {
            "n_rows": len(lr),
            "n_scorable_both": len(both),
            "banked_accuracy": (sum(r["banked"] for r in both) / len(both)) if both else None,
            "replayed_accuracy": (sum(r["replayed"] for r in both) / len(both)) if both else None,
            "flips_true_to_false": sum(r["banked"] and not r["replayed"] for r in both),
            "flips_false_to_true": sum((not r["banked"]) and r["replayed"] for r in both),
            "replayed_outcomes": dict(Counter(str(r["replayed"]) for r in lr)),
            "banked_outcomes": dict(Counter(str(r["banked"]) for r in lr)),
            "by_source": per_source,
        }
    return by_label


def render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Math re-baseline: deterministic scorer replay",
        "",
        f"- source report: `{summary['source_report_dir']}` (not modified)",
        f"- scorer_semantics_version: `{summary['scorer_semantics_version']}`",
        f"- debug_scorer sha256: `{summary['inputs_sha256']['debug_scorer.py']}`",
        "- no inference: banked answers re-scored by the live scorer",
        "",
        "| label | source | n scorable | banked correct | replayed correct | T->F | F->T |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, s in summary["labels"].items():
        for src, p in s["by_source"].items():
            lines.append(
                f"| {label} | {src} | {p['n_scorable']} | {p['banked_correct']} | {p['replayed_correct']} "
                f"| {p['flips_true_to_false']} | {p['flips_false_to_true']} |"
            )
        ba, ra = s["banked_accuracy"], s["replayed_accuracy"]
        lines.append(
            f"| {label} | **all** | {s['n_scorable_both']} | {ba:.4f} | {ra:.4f} "
            f"| {s['flips_true_to_false']} | {s['flips_false_to_true']} |"
        )
    lines.append("")
    lines.append("Accuracy is over rows scorable under BOTH semantics (banked bool and replayed bool).")
    lines.append("Per-label replayed outcome counts (including exclusions) are in `summary.json`.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument(
        "--dump-golds",
        type=Path,
        default=None,
        help="write the joined gold table (+ dataset digests) as JSON and exit; for hosts where "
        "the interpreter with pyarrow is not the one with math-verify",
    )
    ap.add_argument("--golds-json", type=Path, default=None, help="read the gold table written by --dump-golds")
    ap.add_argument(
        "--scorer-path",
        type=Path,
        default=DEFAULT_SCORER,
        help="debug_scorer.py to replay with (default: this repo's). Point it at a pre-change "
        "copy to prove the join: that replay must reproduce the banked verdicts.",
    )
    args = ap.parse_args(argv)

    if args.dump_golds:
        golds, digests = load_golds()
        args.dump_golds.write_text(json.dumps({"golds": golds, "digests": digests}, sort_keys=True) + "\n")
        print(f"wrote {len(golds)} golds to {args.dump_golds}")
        return 0
    if args.report_dir is None or args.out_dir is None:
        ap.error("--report-dir and --out-dir are required unless --dump-golds is given")
    report_dir = args.report_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == report_dir:
        raise SystemExit("replay refuses to write into the source report directory")
    scorer = _load_scorer(args.scorer_path.resolve())
    if args.golds_json:
        blob = json.loads(args.golds_json.read_text())
        golds = {k: tuple(v) for k, v in blob["golds"].items()}
        digests = dict(blob["digests"])
    else:
        golds, digests = load_golds()
    files = sorted(report_dir.glob("question_results.*.jsonl"))
    if not files:
        raise SystemExit(f"no question_results.*.jsonl under {report_dir}")
    rows: list[dict[str, Any]] = []
    for f in files:
        rows.extend(replay_file(f, golds, scorer))
        digests[f.name] = _sha256(f)
    digests["debug_scorer.py"] = _sha256(args.scorer_path.resolve())
    summary = {
        "schema": "epyc.math_baseline_scorer_replay.v1",
        "source_report_dir": str(report_dir.relative_to(REPO_ROOT)) if report_dir.is_relative_to(REPO_ROOT) else str(report_dir),
        "scorer_semantics_version": getattr(scorer, "SCORER_SEMANTICS_VERSION", None),
        "inputs_sha256": digests,
        "labels": summarise(rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "replay.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "summary.md").write_text(render_md(summary))
    print(render_md(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
