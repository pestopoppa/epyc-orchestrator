#!/usr/bin/env python3
"""Run or plan the F1 real_suite_v1 EvalTower clean-window ledger run.

The source suite is local-private because it carries prompts/expected answers.
This runner writes only compact prompt-free packaged artifacts under
``orchestration/reports``; the one intermediate raw JSONL row defaults to
``/mnt/raid0/llm/tmp``.

Default mode is plan-only. Live evaluation requires both ``--apply`` and
``--confirm-clean-window``. A run collected while AutoPilot is active is allowed
only with ``--allow-autopilot-active`` and is marked non-decision-grade.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import httpx
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = PROJECT_ROOT / "scripts" / "autopilot"
DEFAULT_SUITE_YAML = Path(
    "/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/debug/real_suite_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "orchestration" / "reports"
DEFAULT_RAW_ROOT = Path("/mnt/raid0/llm/tmp")
DEFAULT_API_URL = "http://localhost:8000"
FULL_REAL_SUITE_N = 50

for path in (PROJECT_ROOT, AUTOPILOT_DIR):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from eval_tower import EvalTower  # noqa: E402
from scripts.tasks import package_real_suite_eval as packager  # noqa: E402


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def default_output_dir(stamp: str | None = None) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"real_suite_v1_eval_{stamp or utc_stamp()}"


def default_raw_jsonl(stamp: str | None = None) -> Path:
    return DEFAULT_RAW_ROOT / f"real_suite_v1_eval_{stamp or utc_stamp()}.jsonl"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def autopilot_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "scripts/autopilot/autopilot.py start"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        return [f"pgrep failed rc={result.returncode}: {result.stderr.strip()}"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def api_health(api_url: str, *, timeout_s: float) -> dict[str, Any]:
    url = api_url.rstrip("/") + "/health"
    try:
        response = httpx.get(url, timeout=timeout_s)
        return {
            "ok": response.status_code < 500,
            "status_code": response.status_code,
            "url": url,
        }
    except Exception as exc:  # noqa: BLE001 - report preflight, do not crash
        return {"ok": False, "url": url, "error": str(exc)}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_suite_questions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse as a YAML mapping")
    suite_name = str(data.get("suite") or "real_suite_v1")
    default_scoring = data.get("scoring_default")
    default_method = "exact_match"
    default_config: dict[str, Any] = {}
    if isinstance(default_scoring, dict):
        default_method = str(default_scoring.get("method") or default_method)
        raw_config = default_scoring.get("config")
        default_config = raw_config if isinstance(raw_config, dict) else {}

    raw_questions = data.get("questions") or []
    if not isinstance(raw_questions, list):
        raise ValueError(f"{path} questions field is not a list")

    questions: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            continue
        q = dict(item)
        qid = str(q.get("qid") or q.get("stable_qid") or q.get("id") or f"real_suite_v1_{idx:04d}")
        q["id"] = str(q.get("id") or qid)
        q["qid"] = qid
        q["stable_qid"] = qid
        q["suite"] = suite_name
        q["scoring_method"] = str(q.get("scoring_method") or default_method)
        if not isinstance(q.get("scoring_config"), dict):
            q["scoring_config"] = dict(default_config)
        questions.append(q)

    metadata = {
        "suite": suite_name,
        "version": str(data.get("version") or ""),
        "generated_at": str(data.get("generated_at") or ""),
        "question_count": len(questions),
        "tier_counts": dict(Counter(str(q.get("tier") or "unknown") for q in questions)),
        "task_class_counts": dict(
            Counter(str(q.get("real_task_class") or "unknown") for q in questions)
        ),
        "scoring_methods": dict(Counter(str(q.get("scoring_method") or "") for q in questions)),
    }
    return questions, metadata


def select_questions(questions: list[dict[str, Any]], *, n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or n > len(questions):
        raise ValueError(f"--n must be within 1..{len(questions)}, got {n}")
    if n == len(questions):
        return list(questions)
    rng = random.Random(seed)
    return rng.sample(questions, n)


def _question_metadata_by_qid(questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(q.get("qid") or q.get("id") or ""): q for q in questions}


def enrich_question_results(
    question_results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_qid = _question_metadata_by_qid(questions)
    enriched: list[dict[str, Any]] = []
    for row in question_results:
        item = dict(row)
        meta = by_qid.get(str(item.get("qid") or ""))
        if meta:
            for key in ("real_task_class", "real_task_id", "real_task_outcome"):
                if key in meta and key not in item:
                    item[key] = meta[key]
        enriched.append(item)
    return enriched


def run_evaltower_eval(
    *,
    questions: list[dict[str, Any]],
    api_url: str,
    timeout_s: float,
    tier_label: int,
    log_every: int | None,
) -> Any:
    tower = EvalTower(url=api_url.rstrip("/"), timeout=int(timeout_s))
    with httpx.Client(timeout=timeout_s) as client:
        results = tower._eval_batch(questions, client, log_every=log_every, label="real_suite_v1")
    result = tower._aggregate(results, tier=tier_label)
    result.core_id = "real_suite_v1"
    result.question_results = enrich_question_results(
        list(getattr(result, "question_results", []) or []),
        questions,
    )
    return result


def eval_result_raw_row(
    *,
    result: Any,
    selected_questions: list[dict[str, Any]],
    args: argparse.Namespace,
    started_at: str,
    finished_at: str,
    calibration_id: str,
) -> dict[str, Any]:
    details = getattr(result, "details", {}) or {}
    return {
        "event_type": "real_suite_v1_evaltower_window",
        "calibration_id": calibration_id,
        "core_id": "real_suite_v1",
        "trial_id": None,
        "started_at": started_at,
        "finished_at": finished_at,
        "requested_n": args.n,
        "n_questions": int(getattr(result, "n_questions", 0) or len(selected_questions)),
        "seed": args.seed,
        "tier": args.tier_label,
        "quality": float(getattr(result, "quality", 0.0) or 0.0),
        "speed": float(getattr(result, "speed", 0.0) or 0.0),
        "cost": float(getattr(result, "cost", 0.0) or 0.0),
        "reliability": float(getattr(result, "reliability", 0.0) or 0.0),
        "eval_wall_s": float(getattr(result, "eval_wall_s", 0.0) or 0.0),
        "eval_concurrency": int(getattr(result, "eval_concurrency", 0) or 0),
        "speed_metric_mode": str(getattr(result, "speed_metric_mode", "") or ""),
        "aggregate_speed": float(getattr(result, "aggregate_speed", 0.0) or 0.0),
        "median_request_speed": float(getattr(result, "median_request_speed", 0.0) or 0.0),
        "eval_details": {
            "question_results": list(getattr(result, "question_results", []) or []),
            "details": {
                "correct": _safe_int(details.get("correct")),
                "total": _safe_int(details.get("total"), len(selected_questions)),
                "errors": _safe_int(details.get("errors")),
                "per_suite_counts": details.get("per_suite_counts", {}),
                "task_rate_qph": details.get("task_rate_qph", 0.0),
                "goodput_qph": details.get("goodput_qph", 0.0),
            },
        },
    }


def caveat_for_report(args: argparse.Namespace, *, autopilot_active: bool, full_suite: bool) -> str:
    if not args.apply:
        return "Plan-only report; no model calls were made."
    if autopilot_active:
        return (
            "Run is isolated from AutoPilot journal/state, but was collected while "
            "AutoPilot was live; treat timing and quality as a concurrent-window "
            "observation, not decision-grade acceptance evidence."
        )
    if not full_suite:
        return (
            "Partial real_suite_v1 run; useful for harness smoke only. It is not "
            "the clean full 50-question W3 acceptance run."
        )
    return (
        "Clean-window standalone EvalTower real_suite_v1 run. It is isolated from "
        "AutoPilot journal/state and packaged prompt-free for F1 W3 acceptance review."
    )


def preflight(args: argparse.Namespace, *, questions: list[dict[str, Any]]) -> dict[str, Any]:
    active = autopilot_processes()
    return {
        "api_health": api_health(args.api_url, timeout_s=args.http_timeout_s),
        "autopilot_active": bool(active),
        "autopilot_processes": active,
        "suite_yaml": str(args.suite_yaml),
        "suite_yaml_exists": args.suite_yaml.exists(),
        "available_questions": len(questions),
    }


def pre_apply_blockers(
    args: argparse.Namespace,
    *,
    suite_meta: dict[str, Any],
    preflight_report: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not args.confirm_clean_window:
        blockers.append("--apply requires --confirm-clean-window")
    if not args.suite_yaml.exists():
        blockers.append(f"suite YAML is missing: {args.suite_yaml}")
    if suite_meta.get("question_count", 0) <= 0:
        blockers.append("suite YAML contains no questions")
    if not (preflight_report.get("api_health") or {}).get("ok"):
        blockers.append("orchestrator API health is not OK")
    if preflight_report.get("autopilot_active") and not args.allow_autopilot_active:
        blockers.append("AutoPilot appears active; real-suite run needs a clean window")
    if args.n != suite_meta.get("question_count") and not args.allow_partial:
        blockers.append(
            f"--apply defaults to the full suite; pass --allow-partial for n={args.n}"
        )
    return blockers


def runner_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# real_suite_v1 EvalTower Window Runner",
        "",
        f"- status: `{report['status']}`",
        f"- decision_grade: `{report['decision_grade']}`",
        f"- applied: `{report['applied']}`",
        f"- suite questions: `{report['suite']['question_count']}`",
        f"- selected questions: `{report['selected_n']}`",
        f"- AutoPilot active: `{report['preflight'].get('autopilot_active')}`",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    if report.get("raw_jsonl"):
        lines.extend(["", "## Raw Local Row", "", f"- `{report['raw_jsonl']}`"])
    if report.get("packaged_summary"):
        metrics = report["packaged_summary"].get("metrics", {})
        lines.extend(
            [
                "",
                "## Packaged Metrics",
                "",
                f"- quality_0_3: `{metrics.get('quality_0_3')}`",
                f"- reliability: `{metrics.get('reliability')}`",
                f"- errors: `{metrics.get('errors')}`",
                f"- question_ledger_path: `{report['packaged_summary'].get('question_ledger_path')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def write_runner_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "runner_report.json"
    md_path = output_dir / "runner_report.md"
    write_json(json_path, report)
    md_path.write_text(runner_markdown(report), encoding="utf-8")
    return json_path, md_path


def build_report(args: argparse.Namespace, *, output_dir: Path, stamp: str | None = None) -> tuple[dict[str, Any], int]:
    run_stamp = stamp or utc_stamp()
    raw_jsonl = args.raw_jsonl or default_raw_jsonl(run_stamp)
    questions, suite_meta = load_suite_questions(args.suite_yaml)
    selected = select_questions(questions, n=args.n, seed=args.seed)
    full_suite = args.n == suite_meta["question_count"] == FULL_REAL_SUITE_N
    preflight_report = preflight(args, questions=questions)
    blockers = (
        pre_apply_blockers(args, suite_meta=suite_meta, preflight_report=preflight_report)
        if args.apply
        else []
    )
    status = "plan_only"
    rc = 0
    raw_row: dict[str, Any] | None = None
    packaged_summary: dict[str, Any] | None = None
    started_at = ""
    finished_at = ""

    if args.apply and blockers:
        status = "blocked"
        rc = 75 if any("AutoPilot appears active" in item for item in blockers) else 2
    elif args.apply:
        started_at = utc_now()
        started = time.perf_counter()
        result = run_evaltower_eval(
            questions=selected,
            api_url=args.api_url,
            timeout_s=args.evaltower_timeout_s,
            tier_label=args.tier_label,
            log_every=args.log_every,
        )
        result.question_results = enrich_question_results(
            list(getattr(result, "question_results", []) or []),
            selected,
        )
        finished_at = utc_now()
        raw_row = eval_result_raw_row(
            result=result,
            selected_questions=selected,
            args=args,
            started_at=started_at,
            finished_at=finished_at,
            calibration_id=f"real_suite_v1_eval_{run_stamp}",
        )
        if not raw_row["eval_wall_s"]:
            raw_row["eval_wall_s"] = time.perf_counter() - started
        write_jsonl(raw_jsonl, [raw_row])
        packaged_summary = packager.run(
            argparse.Namespace(
                input=raw_jsonl,
                output_dir=output_dir,
                caveat=caveat_for_report(
                    args,
                    autopilot_active=bool(preflight_report.get("autopilot_active")),
                    full_suite=full_suite,
                ),
            )
        )
        status = "packaged_observation"
        if full_suite and not preflight_report.get("autopilot_active"):
            status = "clean_full_suite_packaged"

    decision_grade = bool(
        args.apply
        and args.confirm_clean_window
        and full_suite
        and not args.allow_autopilot_active
        and not preflight_report.get("autopilot_active")
        and not blockers
        and packaged_summary
    )
    report = {
        "generated_at": utc_now(),
        "status": status,
        "decision_grade": decision_grade,
        "decision_grade_notes": [
            "requires --apply --confirm-clean-window",
            "requires full real_suite_v1 50-question run",
            "requires AutoPilot inactive unless --allow-autopilot-active is set",
            "the run is standalone and does not write AutoPilot journal/state",
        ],
        "applied": bool(args.apply),
        "confirm_clean_window": bool(args.confirm_clean_window),
        "allow_autopilot_active": bool(args.allow_autopilot_active),
        "allow_partial": bool(args.allow_partial),
        "api_url": args.api_url.rstrip("/"),
        "suite_yaml": str(args.suite_yaml),
        "suite": suite_meta,
        "selected_n": args.n,
        "seed": args.seed,
        "tier_label": args.tier_label,
        "blockers": blockers,
        "preflight": preflight_report,
        "raw_jsonl": str(raw_jsonl) if raw_row else "",
        "output_dir": str(output_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "raw_row_metrics": {
            "quality": raw_row.get("quality") if raw_row else None,
            "reliability": raw_row.get("reliability") if raw_row else None,
            "n_questions": raw_row.get("n_questions") if raw_row else None,
            "eval_wall_s": raw_row.get("eval_wall_s") if raw_row else None,
        },
        "packaged_summary": packaged_summary,
    }
    return report, rc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-yaml", type=Path, default=DEFAULT_SUITE_YAML)
    parser.add_argument("--api-url", default=os.environ.get("ORCHESTRATOR_API_URL", DEFAULT_API_URL))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--raw-jsonl", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-clean-window", action="store_true")
    parser.add_argument("--allow-autopilot-active", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--n", type=int, default=FULL_REAL_SUITE_N)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--tier-label",
        type=int,
        default=1,
        help="Report tier label for the standalone EvalResult; does not change suite content.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--http-timeout-s", type=float, default=5.0)
    parser.add_argument("--evaltower-timeout-s", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or default_output_dir()
    report, rc = build_report(args, output_dir=output_dir)
    json_path, md_path = write_runner_report(report, output_dir)
    if not args.summary_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {json_path}")
        print(f"wrote {md_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
