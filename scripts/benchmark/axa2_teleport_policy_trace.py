#!/usr/bin/env python3
"""Dry AXA-2 teleport policy trace.

Reads JSONL request/stream summaries, evaluates the default-off AXA-2
TeleportPolicy for each row, and writes deterministic policy artifacts. This
script never starts inference, never acquires/releases a GPU lease, and never
touches production kernels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TELEPORT_PATH = REPO_ROOT / "src" / "llm_primitives" / "teleport.py"
_TELEPORT_SPEC = importlib.util.spec_from_file_location("axa2_teleport_policy_module", _TELEPORT_PATH)
if _TELEPORT_SPEC is None or _TELEPORT_SPEC.loader is None:
    raise RuntimeError(f"failed to load teleport policy module from {_TELEPORT_PATH}")
_TELEPORT_MODULE = importlib.util.module_from_spec(_TELEPORT_SPEC)
sys.modules[_TELEPORT_SPEC.name] = _TELEPORT_MODULE
_TELEPORT_SPEC.loader.exec_module(_TELEPORT_MODULE)

TeleportInputs = _TELEPORT_MODULE.TeleportInputs
TeleportPolicy = _TELEPORT_MODULE.TeleportPolicy
decide_teleport = _TELEPORT_MODULE.decide_teleport


SCHEMA = "epyc.axa2_teleport_policy_trace.v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def jsonable(value: Any) -> Any:
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def parse_csv_set(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be a JSON object")
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def _int_from(row: dict[str, Any], *names: str, default: int) -> int:
    for name in names:
        if row.get(name) is not None:
            return int(row[name])
    return default


def _float_from(row: dict[str, Any], *names: str, default: float) -> float:
    for name in names:
        if row.get(name) is not None:
            return float(row[name])
    return default


def _bool_from(row: dict[str, Any], *names: str, default: bool) -> bool:
    for name in names:
        if row.get(name) is not None:
            value = row[name]
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
    return default


def build_policy(args: argparse.Namespace) -> TeleportPolicy:
    return TeleportPolicy(
        enabled=args.policy_enabled,
        mode=args.mode,
        quant_policy=args.quant_policy,
        long_running_trigger_tokens=args.long_running_trigger_tokens,
        rate_window_tokens=args.rate_window_tokens,
        min_resident_remaining_tokens=args.min_resident_remaining_tokens,
        min_cold_remaining_tokens=args.min_cold_remaining_tokens,
        min_speedup=args.min_speedup,
        allowed_roles=parse_csv_set(args.role_allowlist),
        allowed_quant_change_roles=parse_csv_set(args.quant_change_role_allowlist),
    )


def build_inputs(row: dict[str, Any], args: argparse.Namespace) -> TeleportInputs:
    return TeleportInputs(
        role=str(row.get("role") or args.role),
        generated_tokens=_int_from(row, "generated_tokens", "generated_so_far", default=args.generated_tokens),
        estimated_remaining_tokens=_int_from(
            row,
            "estimated_remaining_tokens",
            "expected_remaining_tokens",
            default=args.estimated_remaining_tokens,
        ),
        cpu_tps=_float_from(row, "cpu_tps", default=args.cpu_tps),
        gpu_tps=_float_from(row, "gpu_tps", default=args.gpu_tps),
        gpu_available=_bool_from(row, "gpu_available", default=args.gpu_available),
        gpu_resident=_bool_from(row, "gpu_resident", default=args.gpu_resident),
        cpu_quant=str(row.get("cpu_quant") if row.get("cpu_quant") is not None else args.cpu_quant),
        gpu_quant=str(row.get("gpu_quant") if row.get("gpu_quant") is not None else args.gpu_quant),
        catch_up_supported=_bool_from(row, "catch_up_supported", default=args.catch_up_supported),
        metadata={
            "trace_line_no": row.get("_line_no"),
            "trace_id": row.get("trace_id") or row.get("request_id") or row.get("id"),
            "prompt_tokens": row.get("prompt_tokens"),
            "workload_class": row.get("workload_class"),
        },
    )


def economics(inputs: TeleportInputs, args: argparse.Namespace) -> dict[str, Any]:
    if inputs.cpu_tps <= 0 or inputs.gpu_tps <= 0:
        saved_seconds = None
        positive_after_load = None
    else:
        saved_seconds = inputs.estimated_remaining_tokens * (1.0 / inputs.cpu_tps - 1.0 / inputs.gpu_tps)
        positive_after_load = saved_seconds > args.load_seconds
    return {
        "cpu_tps": inputs.cpu_tps,
        "gpu_tps": inputs.gpu_tps,
        "load_seconds": args.load_seconds,
        "estimated_saved_seconds_before_load": saved_seconds,
        "positive_after_load_only": positive_after_load,
    }


def evaluate_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    policy = build_policy(args)
    decisions = []
    for idx, row in enumerate(rows):
        inputs = build_inputs(row, args)
        decision = decide_teleport(policy, inputs)
        decisions.append(
            {
                "idx": idx,
                "trace_line_no": row.get("_line_no"),
                "trace_id": inputs.metadata.get("trace_id"),
                "inputs": jsonable(asdict(inputs)),
                "policy": jsonable(asdict(policy)),
                "decision": jsonable(asdict(decision)),
                "costs": economics(inputs, args),
            }
        )
    return decisions


def collect_environment() -> dict[str, Any]:
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return {
        "captured_at": utc_now(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "repo": {
            "path": str(Path.cwd()),
            "head": git.stdout.strip() if git.returncode == 0 else None,
            "head_error": git.stderr.strip() if git.returncode != 0 else None,
        },
    }


def write_artifacts(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = evaluate_rows(rows, args)
    cutovers = [item for item in decisions if item["decision"]["should_cutover"]]
    summary = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": "dry_policy_trace_only",
        "trace_path": str(Path(args.trace).expanduser().resolve()),
        "output_dir": str(output_dir),
        "row_count": len(decisions),
        "cutover_count": len(cutovers),
        "decision_reasons": sorted({item["decision"]["reason"] for item in decisions}),
        "cutover_reasons": sorted({item["decision"]["reason"] for item in cutovers}),
        "no_inference": True,
        "no_lease_acquire": True,
        "production_v6_touch_authorized": False,
        "environment": collect_environment(),
    }
    (output_dir / "policy_decisions.jsonl").write_text(
        "".join(canonical_json(item) for item in decisions),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(canonical_json(summary), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_markdown(summary, decisions), encoding="utf-8")
    return summary


def render_markdown(summary: dict[str, Any], decisions: list[dict[str, Any]]) -> str:
    rows = [
        f"| {item['idx']} | {item.get('trace_id') or ''} | {item['inputs']['role']} | "
        f"{item['decision']['should_cutover']} | {item['decision']['reason']} | "
        f"{item['decision']['threshold_tokens']} | {item['decision']['estimated_speedup']} |"
        for item in decisions
    ]
    return "\n".join(
        [
            "# AXA-2 Teleport Policy Trace",
            "",
            f"- Schema: `{summary['schema']}`",
            f"- Status: `{summary['status']}`",
            f"- Trace rows: `{summary['row_count']}`",
            f"- Cutover rows: `{summary['cutover_count']}`",
            "- No inference, no lease acquisition, no production-v6 touch.",
            "",
            "| idx | trace_id | role | cutover | reason | threshold_tokens | speedup |",
            "|---:|---|---|---|---|---:|---:|",
            *rows,
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="JSONL trace/request summary input")
    parser.add_argument("--output", required=True, help="Output artifact directory")
    parser.add_argument("--policy-enabled", action="store_true")
    parser.add_argument("--mode", default="v1_reprefill_cutover_only")
    parser.add_argument("--role", default="architect_general")
    parser.add_argument("--role-allowlist", default="")
    parser.add_argument("--quant-policy", default="same_quant_only")
    parser.add_argument("--quant-change-role-allowlist", default="")
    parser.add_argument("--long-running-trigger-tokens", type=int, default=128)
    parser.add_argument("--rate-window-tokens", type=int, default=64)
    parser.add_argument("--min-resident-remaining-tokens", type=int, default=150)
    parser.add_argument("--min-cold-remaining-tokens", type=int, default=350)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--generated-tokens", type=int, default=0)
    parser.add_argument("--estimated-remaining-tokens", type=int, default=0)
    parser.add_argument("--cpu-tps", type=float, default=0.0)
    parser.add_argument("--gpu-tps", type=float, default=0.0)
    parser.add_argument("--load-seconds", type=float, default=0.0)
    parser.add_argument("--gpu-available", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-resident", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu-quant", default="")
    parser.add_argument("--gpu-quant", default="")
    parser.add_argument("--catch-up-supported", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_trace(Path(args.trace).expanduser())
    summary = write_artifacts(rows, args)
    print(summary["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
