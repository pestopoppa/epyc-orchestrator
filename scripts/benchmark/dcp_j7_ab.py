#!/usr/bin/env python3
"""J7/DCP-6 live A/B runner for delegation-context pre-assembly.

The DCP hook is advisory and default-off.  This driver gives the deferred J7
inference half an executable gate: run matched DCP_PRE_ASSEMBLY off/on arms
against delegation-heavy prompts, capture latency/token/delegation telemetry,
and leave production state reverted to OFF.

Default mode is --stub, which performs no inference and only exercises the
artifact schema.  Real inference is refused unless --host-quiet-confirmed is
passed and AutoPilot is not running.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ORCH = Path("/mnt/raid0/llm/epyc-orchestrator")
API_URL = "http://127.0.0.1:8000"
RESULTS_ROOT = ORCH / "benchmarks" / "results" / "runs" / "dcp_j7"
MIN_ROWS_PER_ARM = 3
MIN_LATENCY_IMPROVEMENT = 0.10

PROMPTS: list[dict[str, str]] = [
    {
        "id": "trace_callers",
        "prompt": (
            "In the orchestrator repository, identify every caller that can reach "
            "`src.api.routes.chat_delegation._maybe_dcp_seed_context`, explain the "
            "runtime path in order, and name the tests that should fail if DCP "
            "pre-assembly stopped being advisory."
        ),
    },
    {
        "id": "autopilot_gate_path",
        "prompt": (
            "Trace how an AutoPilot promotion candidate flows from an action handler "
            "through SafetyGate.check and baseline update. Identify which call sites "
            "must receive per-question sequential evidence before enabling authority."
        ),
    },
    {
        "id": "feature_flag_attest",
        "prompt": (
            "Find the runtime feature-flag hot-reload and attestation path for "
            "DCP_PRE_ASSEMBLY. Summarize how a live A/B runner should toggle, verify, "
            "and revert the flag without relying on a server restart."
        ),
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _orch_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ORCH,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _arm_sequence(reps: int) -> list[bool]:
    seq: list[bool] = []
    for i in range(reps):
        seq.extend([False, True] if i % 2 == 0 else [True, False])
    return seq


def _autopilot_pids() -> list[str]:
    r = subprocess.run(["pgrep", "-f", "autopilot.py start"], capture_output=True, text=True)
    return r.stdout.split()


def _hot_reload_dcp(client: Any, enabled: bool, *, api_url: str) -> None:
    resp = client.post(f"{api_url}/config", json={"dcp_pre_assembly": enabled})
    resp.raise_for_status()
    actual = (resp.json().get("features") or {}).get("dcp_pre_assembly")
    if actual is not enabled:
        raise RuntimeError(f"DCP hot-reload mismatch: requested={enabled} actual={actual}")


def _chat(
    client: Any,
    prompt: str,
    *,
    api_url: str,
    timeout_s: int,
    request_id: str,
) -> tuple[dict[str, Any], float, int]:
    payload = {
        "prompt": prompt,
        "force_role": "architect_general",
        "force_mode": "delegated",
        "allow_delegation": True,
        "mock_mode": False,
        "real_mode": True,
        "cache_prompt": False,
        "timeout_s": timeout_s,
        "request_id": request_id,
        "request_priority": "background",
        "max_queue_wait_ms": 90_000,
    }
    t0 = time.monotonic()
    resp = client.post(f"{api_url}/chat", json=payload, timeout=timeout_s + 30)
    elapsed = time.monotonic() - t0
    status = resp.status_code
    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if status >= 400:
        data = {"error": data or resp.text[:500]}
    return data, elapsed, status


def _summarize_response(data: dict[str, Any], elapsed_s: float, status: int) -> dict[str, Any]:
    events = data.get("delegation_events") or []
    diagnostics = data.get("delegation_diagnostics") or {}
    event_tokens = [
        int((event.get("inference_meta") or {}).get("tokens") or event.get("tokens_generated") or 0)
        for event in events
    ]
    event_prompt_ms = [
        float((event.get("inference_meta") or {}).get("prompt_ms") or 0.0)
        for event in events
    ]
    tools = data.get("tools_called") or []
    return {
        "status": status,
        "elapsed_s": round(elapsed_s, 3),
        "api_elapsed_seconds": data.get("elapsed_seconds"),
        "turns": data.get("turns"),
        "tokens_used": data.get("tokens_used"),
        "tokens_generated": data.get("tokens_generated"),
        "prompt_eval_ms": data.get("prompt_eval_ms"),
        "generation_ms": data.get("generation_ms"),
        "predicted_tps": data.get("predicted_tps"),
        "tools_used": data.get("tools_used"),
        "tools_called_count": len(tools),
        "delegation_events_count": len(events),
        "delegation_success": data.get("delegation_success"),
        "delegation_break_reason": diagnostics.get("break_reason"),
        "delegation_report_handles_count": diagnostics.get("report_handles_count"),
        "delegation_inference_hops": diagnostics.get("delegation_inference_hops"),
        "delegation_event_tokens": sum(event_tokens),
        "delegation_event_prompt_ms": round(sum(event_prompt_ms), 3),
        "quality_score": data.get("quality_score"),
        "quality_pass": data.get("quality_pass"),
        "error_code": data.get("error_code"),
        "error_detail": data.get("error_detail"),
        "answer_chars": len(data.get("answer") or ""),
    }


def _stub_row(block: int, arm: str, prompt: dict[str, str]) -> dict[str, Any]:
    return {
        "block": block,
        "arm": arm,
        "prompt_id": prompt["id"],
        "mode": "stub",
        "dcp_pre_assembly": arm == "on",
        "summary": {
            "status": 200,
            "elapsed_s": 0.0,
            "api_elapsed_seconds": 0.0,
            "turns": 0,
            "tokens_used": 0,
            "tokens_generated": 0,
            "prompt_eval_ms": 0.0,
            "generation_ms": 0.0,
            "predicted_tps": 0.0,
            "tools_used": 0,
            "tools_called_count": 0,
            "delegation_events_count": 0,
            "delegation_success": None,
            "delegation_break_reason": "stub",
            "delegation_report_handles_count": 0,
            "delegation_inference_hops": 0,
            "delegation_event_tokens": 0,
            "delegation_event_prompt_ms": 0.0,
            "quality_score": None,
            "quality_pass": None,
            "error_code": None,
            "error_detail": None,
            "answer_chars": 0,
        },
    }


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {"off": [], "on": []}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row["summary"])
    out: dict[str, Any] = {}
    for arm, summaries in by_arm.items():
        if not summaries:
            continue
        latencies = sorted(float(s.get("elapsed_s") or 0.0) for s in summaries)
        out[arm] = {
            "n": len(summaries),
            "p50_elapsed_s": latencies[len(latencies) // 2],
            "avg_tokens_generated": round(
                sum(float(s.get("tokens_generated") or 0.0) for s in summaries) / len(summaries),
                3,
            ),
            "avg_delegation_event_tokens": round(
                sum(float(s.get("delegation_event_tokens") or 0.0) for s in summaries) / len(summaries),
                3,
            ),
            "avg_delegation_events": round(
                sum(float(s.get("delegation_events_count") or 0.0) for s in summaries) / len(summaries),
                3,
            ),
            "delegation_successes": sum(1 for s in summaries if s.get("delegation_success") is True),
            "delegation_failures": sum(1 for s in summaries if s.get("delegation_success") is False),
            "quality_scored": sum(1 for s in summaries if s.get("quality_score") is not None),
            "quality_passes": sum(1 for s in summaries if s.get("quality_pass") is True),
            "quality_failures": sum(1 for s in summaries if s.get("quality_pass") is False),
            "errors": sum(1 for s in summaries if s.get("error_code") or int(s.get("status") or 0) >= 400),
        }
    if out.get("off") and out.get("on") and out["off"]["p50_elapsed_s"]:
        out["delta"] = {
            "p50_elapsed_pct": round(
                (out["off"]["p50_elapsed_s"] - out["on"]["p50_elapsed_s"])
                / out["off"]["p50_elapsed_s"],
                4,
            ),
            "avg_tokens_generated_delta": round(
                out["on"]["avg_tokens_generated"] - out["off"]["avg_tokens_generated"],
                3,
            ),
            "avg_delegation_event_tokens_delta": round(
                out["on"]["avg_delegation_event_tokens"]
                - out["off"]["avg_delegation_event_tokens"],
                3,
            ),
        }
    out["decision"] = _decision(out)
    return out


def _decision(summary: dict[str, Any]) -> dict[str, Any]:
    off = summary.get("off") or {}
    on = summary.get("on") or {}
    delta = summary.get("delta") or {}
    blockers: list[str] = []

    if off.get("n", 0) < MIN_ROWS_PER_ARM or on.get("n", 0) < MIN_ROWS_PER_ARM:
        blockers.append("too_few_rows_per_arm")
    if off.get("errors", 0) or on.get("errors", 0):
        blockers.append("errors_present")

    latency_delta = delta.get("p50_elapsed_pct")
    if latency_delta is None:
        blockers.append("missing_latency_delta")
    elif latency_delta < MIN_LATENCY_IMPROVEMENT:
        blockers.append("latency_not_improved")

    if on.get("avg_delegation_event_tokens", 0.0) > off.get("avg_delegation_event_tokens", 0.0):
        blockers.append("delegation_event_tokens_regressed")

    quality_required = min(off.get("n", 0), on.get("n", 0))
    if off.get("quality_scored", 0) < quality_required or on.get("quality_scored", 0) < quality_required:
        blockers.append("quality_not_scored")
    elif on.get("quality_failures", 0) > off.get("quality_failures", 0):
        blockers.append("quality_failures_regressed")

    if off.get("delegation_failures", 0) or on.get("delegation_failures", 0):
        blockers.append("delegation_failures_present")

    if not blockers:
        status = "promote_advisory"
        recommendation = "keep advisory DCP enabled for a second confirmatory run"
    elif "latency_not_improved" in blockers or "errors_present" in blockers:
        status = "hold"
        recommendation = "keep dcp_pre_assembly default-off"
    else:
        status = "insufficient"
        recommendation = "rerun with enough rows and quality-scored prompts"

    return {
        "schema_version": "dcp_j7_decision.v1",
        "status": status,
        "recommendation": recommendation,
        "blockers": blockers,
        "criteria": {
            "min_rows_per_arm": MIN_ROWS_PER_ARM,
            "min_latency_improvement": MIN_LATENCY_IMPROVEMENT,
            "requires_zero_errors": True,
            "requires_quality_scored_rows": True,
            "requires_no_quality_regression": True,
            "requires_no_delegation_failures": True,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="J7/DCP-6 delegation-context A/B runner")
    ap.add_argument("--reps", type=int, default=1, help="ABBA reps; 1 -> off,on")
    ap.add_argument("--output", default="", help="Output directory; defaults under benchmarks/results")
    ap.add_argument("--api-url", default=API_URL)
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--stub", action="store_true", help="No inference; schema/artifact dry-run")
    ap.add_argument(
        "--host-quiet-confirmed",
        action="store_true",
        help="Required for real inference; operator confirms no contaminating workload",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    real = not args.stub
    if real and not args.host_quiet_confirmed:
        print(
            "REFUSING real J7 run: pass --host-quiet-confirmed after clean DCP attestation "
            "and a quiet host window. Use --stub for no-inference schema validation.",
            file=sys.stderr,
        )
        return 2
    if real:
        pids = _autopilot_pids()
        if pids:
            print(f"REFUSING: autopilot still running (pids {pids}). Stop W4/J6 accrual first.", file=sys.stderr)
            return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output) if args.output else RESULTS_ROOT / ts
    rows_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    meta_path = out_dir / "meta.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_seq = _arm_sequence(args.reps)
    meta = {
        "kind": "dcp-j7-ab-run",
        "created_at": _now(),
        "mode": "real" if real else "stub",
        "api_url": args.api_url,
        "orch_head_before": _orch_head(),
        "reps": args.reps,
        "arm_sequence": ["on" if enabled else "off" for enabled in arm_seq],
        "prompts": [p["id"] for p in PROMPTS],
        "host_quiet_confirmed": bool(args.host_quiet_confirmed),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    rows: list[dict[str, Any]] = []
    print(f"[dcp_j7] mode={'REAL' if real else 'STUB'} out={rows_path}")
    try:
        if real:
            import httpx  # live-only dependency; --stub should run in plain Python
            client_cm: Any = httpx.Client()
        else:
            from contextlib import nullcontext
            client_cm = nullcontext(None)

        with client_cm as client:
            for block, enabled in enumerate(arm_seq):
                arm = "on" if enabled else "off"
                if real:
                    _hot_reload_dcp(client, enabled, api_url=args.api_url)
                    time.sleep(0.5)
                for prompt in PROMPTS:
                    if real:
                        request_id = f"dcp-j7-{prompt['id']}-{arm}-blk{block}"
                        data, elapsed, status = _chat(
                            client,
                            prompt["prompt"],
                            api_url=args.api_url,
                            timeout_s=args.timeout_s,
                            request_id=request_id,
                        )
                        row = {
                            "block": block,
                            "arm": arm,
                            "prompt_id": prompt["id"],
                            "mode": "real",
                            "dcp_pre_assembly": enabled,
                            "request_id": request_id,
                            "summary": _summarize_response(data, elapsed, status),
                        }
                    else:
                        row = _stub_row(block, arm, prompt)
                    rows.append(row)
                    _write_jsonl(rows_path, row)
                    print(
                        f"  block{block} arm={arm} {prompt['id']} "
                        f"lat={row['summary']['elapsed_s']}s "
                        f"deleg={row['summary']['delegation_events_count']} "
                        f"tokens={row['summary']['tokens_generated']}"
                    )
    finally:
        if real:
            try:
                import httpx
                with httpx.Client(timeout=10) as client:
                    _hot_reload_dcp(client, False, api_url=args.api_url)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: failed to revert dcp_pre_assembly=False: {exc}", file=sys.stderr)

    summary = _aggregate(rows)
    meta["orch_head_after"] = _orch_head()
    meta["orch_checkout_unchanged"] = bool(meta["orch_head_after"]) and (
        meta["orch_head_after"] == meta["orch_head_before"]
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[dcp_j7] wrote {len(rows)} rows -> {rows_path}")
    print(f"[dcp_j7] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
