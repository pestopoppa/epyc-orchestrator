#!/usr/bin/env python3
"""Run controlled BSV-2 paired eval artifacts.

Default is plan-only. Pass ``--run`` to apply baseline/candidate params,
evaluate both arms sequentially on the same EvalTower core, write EvalResult-like
JSON artifacts, and build the existing BSV-2 paired behavior report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

try:
    from scripts.autopilot.bsv_paired_report import build_eval_result_pair_report
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from bsv_paired_report import build_eval_result_pair_report  # type: ignore[no-redef]


def _default_apply_params(params: dict[str, Any]) -> dict[str, Any]:
    try:
        from scripts.autopilot.config_applicator import apply_params
    except ModuleNotFoundError:  # pragma: no cover - direct script execution path
        from config_applicator import apply_params  # type: ignore[no-redef]
    return apply_params(params)


def _default_tower() -> Any:
    try:
        from scripts.autopilot.eval_tower import EvalTower
    except ModuleNotFoundError:  # pragma: no cover - direct script execution path
        from eval_tower import EvalTower  # type: ignore[no-redef]
    return EvalTower()


def _load_jsonish(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    text = value
    if value.startswith("@"):
        text = Path(value[1:]).read_text()
    else:
        p = Path(value)
        if p.exists():
            text = p.read_text()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _eval_once(
    tower: Any,
    *,
    mode: str,
    tier: int,
    t1_n: int,
    seed: int,
    trial_id: int | None,
) -> Any:
    if mode == "hybrid":
        return tower.hybrid_eval(seed=seed, t1_n=t1_n, trial_id=trial_id)
    if mode == "t1":
        return tower.eval_t1(n=t1_n, seed=seed, trial_id=trial_id)
    if mode == "t2":
        return tower.eval_t2(seed=seed)
    if mode == "evaluate":
        return tower.evaluate(tier=tier, seed=seed)
    raise ValueError(f"unknown eval mode: {mode}")


def _artifact(
    result: Any,
    *,
    label: str,
    archive_member_id: str,
    params: dict[str, Any],
    apply_result: dict[str, Any],
) -> dict[str, Any]:
    payload = _jsonable(result)
    if not isinstance(payload, dict):
        payload = {"result": payload}
    payload["archive_member_id"] = archive_member_id
    payload["bsv_pair_label"] = label
    payload["applied_params"] = params
    payload["apply_result"] = apply_result
    return payload


def build_plan(
    *,
    baseline_params: dict[str, Any],
    candidate_params: dict[str, Any],
    output_dir: Path,
    eval_mode: str,
    tier: int,
    t1_n: int,
    seed: int,
    restore_baseline: bool,
) -> dict[str, Any]:
    return {
        "bsv_paired_runner_version": "bsv-2-paired-runner-v1",
        "mode": "plan",
        "eval_mode": eval_mode,
        "tier": tier,
        "t1_n": t1_n,
        "seed": seed,
        "restore_baseline": restore_baseline,
        "output_dir": str(output_dir),
        "artifacts": {
            "baseline": str(output_dir / "baseline_eval.json"),
            "candidate": str(output_dir / "candidate_eval.json"),
            "report": str(output_dir / "bsv_paired_report.json"),
        },
        "baseline_param_keys": sorted(baseline_params),
        "candidate_param_keys": sorted(candidate_params),
    }


def run_paired_evaluation(
    *,
    baseline_params: dict[str, Any],
    candidate_params: dict[str, Any],
    output_dir: Path,
    eval_mode: str = "t1",
    tier: int = 1,
    t1_n: int = 50,
    seed: int = 42,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    min_shared_qids: int = 35,
    max_accuracy_regression: float = 0.0,
    restore_baseline: bool = True,
    tower: Any | None = None,
    apply_params_func: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tower = tower or _default_tower()
    apply_params_func = apply_params_func or _default_apply_params
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_apply = apply_params_func(baseline_params) if baseline_params else {"status": "ok", "noop": True}
    if baseline_apply.get("status") == "error":
        raise RuntimeError(f"baseline params failed: {baseline_apply}")
    baseline_result = _eval_once(
        tower,
        mode=eval_mode,
        tier=tier,
        t1_n=t1_n,
        seed=seed,
        trial_id=None,
    )
    baseline_artifact = _artifact(
        baseline_result,
        label=baseline_label,
        archive_member_id=f"paired:{baseline_label}",
        params=baseline_params,
        apply_result=baseline_apply,
    )
    _write_json(output_dir / "baseline_eval.json", baseline_artifact)

    candidate_apply = apply_params_func(candidate_params) if candidate_params else {"status": "ok", "noop": True}
    if candidate_apply.get("status") == "error":
        raise RuntimeError(f"candidate params failed: {candidate_apply}")
    try:
        candidate_result = _eval_once(
            tower,
            mode=eval_mode,
            tier=tier,
            t1_n=t1_n,
            seed=seed,
            trial_id=None,
        )
    finally:
        restore_result = None
        if restore_baseline and baseline_params:
            restore_result = apply_params_func(baseline_params)

    candidate_artifact = _artifact(
        candidate_result,
        label=candidate_label,
        archive_member_id=f"paired:{candidate_label}",
        params=candidate_params,
        apply_result=candidate_apply,
    )
    if restore_result is not None:
        candidate_artifact["restore_baseline_result"] = restore_result
    _write_json(output_dir / "candidate_eval.json", candidate_artifact)

    report = build_eval_result_pair_report(
        baseline_artifact,
        candidate_artifact,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        min_shared_qids=min_shared_qids,
        max_accuracy_regression=max_accuracy_regression,
    )
    report["runner"] = build_plan(
        baseline_params=baseline_params,
        candidate_params=candidate_params,
        output_dir=output_dir,
        eval_mode=eval_mode,
        tier=tier,
        t1_n=t1_n,
        seed=seed,
        restore_baseline=restore_baseline,
    )
    report["runner"]["mode"] = "run"
    _write_json(output_dir / "bsv_paired_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BSV-2 paired eval artifacts")
    parser.add_argument("--run", action="store_true", help="actually apply params and run evals")
    parser.add_argument("--baseline-params", default="{}", help="JSON object, path, or @path")
    parser.add_argument("--candidate-params", default="{}", help="JSON object, path, or @path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-mode", choices=["hybrid", "t1", "t2", "evaluate"], default="t1")
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--t1-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--min-shared-qids", type=int, default=35)
    parser.add_argument("--max-accuracy-regression", type=float, default=0.0)
    parser.add_argument("--no-restore", action="store_true", help="leave candidate params applied")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline_params = _load_jsonish(args.baseline_params)
    candidate_params = _load_jsonish(args.candidate_params)
    output_dir = Path(args.output_dir)
    restore_baseline = not args.no_restore

    if not args.run:
        plan = build_plan(
            baseline_params=baseline_params,
            candidate_params=candidate_params,
            output_dir=output_dir,
            eval_mode=args.eval_mode,
            tier=args.tier,
            t1_n=args.t1_n,
            seed=args.seed,
            restore_baseline=restore_baseline,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    report = run_paired_evaluation(
        baseline_params=baseline_params,
        candidate_params=candidate_params,
        output_dir=output_dir,
        eval_mode=args.eval_mode,
        tier=args.tier,
        t1_n=args.t1_n,
        seed=args.seed,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        min_shared_qids=args.min_shared_qids,
        max_accuracy_regression=args.max_accuracy_regression,
        restore_baseline=restore_baseline,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["gate_decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
