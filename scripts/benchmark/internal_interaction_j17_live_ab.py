#!/usr/bin/env python3
"""J17 live A/B for review_before_commit consult at the edit-transaction seam.

This harness keeps the production API route inert: it calls
``run_edit_transaction`` in-process and uses the live orchestrator API only for
the coder draft and optional architect consult calls. The checked-in BEP slice
has five code-edit tasks, so the default 50-turn run repeats that fixed slice
ten times and records the repetition explicitly in the artifact.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ORCH = Path("/mnt/raid0/llm/epyc-orchestrator")
TASKS_PATH = ORCH / "data" / "bep_sandbox" / "tasks.jsonl"
DEFAULT_API_URL = "http://localhost:8000/chat"
DEFAULT_OUTPUT_ROOT = ORCH / "orchestration" / "reports"
DEFAULT_SCRATCH = Path("/mnt/raid0/llm/tmp/j17_review_before_commit")

sys.path.insert(0, str(ORCH))
from src.edit_transaction import EditResult, run_edit_transaction  # noqa: E402
from src.orchestration.consultation import consult  # noqa: E402


@dataclass
class HTTPPrimitives:
    api_url: str
    timeout_s: float
    calls: list[dict[str, Any]] = field(default_factory=list)

    def request_context(self, **_kwargs: Any):
        return nullcontext()

    def llm_call(
        self,
        prompt: str,
        *,
        role: str,
        n_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        started = time.monotonic()
        payload = {
            "prompt": prompt,
            "force_role": role,
            "force_mode": "direct",
            "max_turns": 1,
            "max_tokens": n_tokens,
            "cache_prompt": False,
            "mock_mode": False,
            "real_mode": True,
            "request_priority": "background",
            "workload_class": "campaign",
            "session_id": f"j17-consult-{int(started * 1000)}",
        }
        if json_schema:
            # /chat does not expose constrained decoding; keep the schema in
            # the prompt via consult() and record that this was advisory JSON.
            payload["context"] = "json_schema_supplied_by_consult_helper"
        resp = httpx.post(self.api_url, json=payload, timeout=self.timeout_s)
        elapsed = time.monotonic() - started
        body = resp.text
        resp.raise_for_status()
        data = resp.json()
        text = data.get("answer") or data.get("response") or data.get("content") or ""
        self.calls.append(
            {
                "role": role,
                "wall_s": round(elapsed, 3),
                "answer_chars": len(text),
                "status_code": resp.status_code,
                "mode": data.get("mode"),
            }
        )
        return str(text)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ORCH, text=True, capture_output=True, timeout=10)
    return r.stdout.strip()


def _autopilot_active() -> bool:
    r = subprocess.run(["pgrep", "-af", "scripts/autopilot/autopilot.py"], text=True, capture_output=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _reset_root(root: Path, files: dict[str, str] | None) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _run_verifier(root: Path, cmd: str) -> bool:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run(cmd, shell=True, cwd=root, text=True, capture_output=True, timeout=60, env=env)
    except Exception:
        return False
    return r.returncode == 0 and "PASS" in (r.stdout + r.stderr)


def _chat_llm(api_url: str, timeout_s: float, role: str, prompt: str, *, session_id: str) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    payload = {
        "prompt": prompt,
        "force_role": role,
        "force_mode": "direct",
        "max_turns": 1,
        "cache_prompt": False,
        "mock_mode": False,
        "real_mode": True,
        "request_priority": "background",
        "workload_class": "campaign",
        "session_id": session_id,
    }
    resp = httpx.post(api_url, json=payload, timeout=timeout_s)
    elapsed = time.monotonic() - started
    resp.raise_for_status()
    data = resp.json()
    text = data.get("answer") or data.get("response") or data.get("content") or ""
    return str(text), {
        "wall_s": round(elapsed, 3),
        "answer_chars": len(str(text)),
        "status_code": resp.status_code,
        "mode": data.get("mode"),
    }


def _classify(result: EditResult | None, raw: str, verifier_ok: bool, error: str) -> str:
    if error:
        return "chat/http error"
    if result is None:
        return "chat/http error"
    if not result.ok:
        err = (result.error or "").lower()
        if "scope too large" in err:
            return "scope-cap reject"
        if "no valid file blocks" in err or not raw.strip():
            return "parse/no blocks"
        if "functional verifier failed" in err:
            return "verifier fail"
        return "rollback/self-check"
    if not verifier_ok:
        return "verifier fail"
    return "pass"


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return round(ordered[idx], 3)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    out: dict[str, Any] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        coder_wall = [float(r["coder_call"]["wall_s"]) for r in arm_rows if r.get("coder_call")]
        consult_wall = [
            float(c["wall_s"])
            for r in arm_rows
            for c in (r.get("consult_calls") or [])
        ]
        consult_events = [e for r in arm_rows for e in (r.get("consult_events") or [])]
        out[arm] = {
            "turns": len(arm_rows),
            "passes": sum(1 for r in arm_rows if r["bucket"] == "pass"),
            "quality": round(sum(1 for r in arm_rows if r["bucket"] == "pass") / max(1, len(arm_rows)), 4),
            "coder_wall_p50_s": _percentile(coder_wall, 0.5),
            "coder_wall_p95_s": _percentile(coder_wall, 0.95),
            "consult_calls": len(consult_wall),
            "consult_wall_p50_s": _percentile(consult_wall, 0.5),
            "consult_wall_p95_s": _percentile(consult_wall, 0.95),
            "consult_successes": sum(1 for e in consult_events if e.get("success")),
            "consult_failures": sum(1 for e in consult_events if not e.get("success")),
            "rerun_requests": sum(1 for e in consult_events if e.get("rerun_requested")),
            "cache_hits": sum(int(e.get("cache_hit", 0) or 0) for e in consult_events),
        }
    if "baseline" in out and "consult" in out:
        b = out["baseline"]
        c = out["consult"]
        out["comparison"] = {
            "quality_delta_pp": round(100 * (c["quality"] - b["quality"]), 3),
            "coder_wall_p50_delta_pct": (
                round(100 * (c["coder_wall_p50_s"] - b["coder_wall_p50_s"]) / b["coder_wall_p50_s"], 3)
                if b.get("coder_wall_p50_s") else None
            ),
            "cache_hit_rate": (
                round(c["cache_hits"] / c["consult_calls"], 4)
                if c.get("consult_calls") else None
            ),
            "gate_notes": [
                "BEP slice has 5 unique tasks repeated to reach 50 turns.",
                "Consult helper currently records cache_ttl_seconds but no cache_hit events.",
                "Wall-clock p50 is used as the live proxy for coder decode p50 in this harness.",
            ],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--confirm-clean-window", action="store_true")
    ap.add_argument("--allow-autopilot-active", action="store_true")
    ap.add_argument("--api-url", default=DEFAULT_API_URL)
    ap.add_argument("--turns", type=int, default=50)
    ap.add_argument("--timeout-s", type=float, default=360)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    args = ap.parse_args(argv)

    if not args.apply:
        print("Plan only. Pass --apply --confirm-clean-window to run live J17 A/B.")
        return 0
    if not args.confirm_clean_window:
        print("REFUSING live inference: pass --confirm-clean-window.", file=sys.stderr)
        return 2
    if _autopilot_active() and not args.allow_autopilot_active:
        print("REFUSING live inference while AutoPilot is active.", file=sys.stderr)
        return 75

    tasks = _load_jsonl(TASKS_PATH)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / f"internal_interaction_j17_ab_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "turns.jsonl"
    rows: list[dict[str, Any]] = []
    task_plan = [tasks[i % len(tasks)] for i in range(args.turns)]

    with rows_path.open("w", encoding="utf-8") as fh:
        for arm in ("baseline", "consult"):
            for idx, task in enumerate(task_plan, start=1):
                root = args.scratch_root / arm / f"{idx:03d}_{task['id']}"
                _reset_root(root, task.get("files"))
                coder_calls: list[dict[str, Any]] = []
                primitives = HTTPPrimitives(args.api_url, args.timeout_s)
                error = ""

                def llm(prompt: str, *, _arm: str = arm, _idx: int = idx, _task: dict[str, Any] = task) -> str:
                    text, meta = _chat_llm(
                        args.api_url,
                        args.timeout_s,
                        "coder_escalation",
                        prompt,
                        session_id=f"j17-{_arm}-{_idx:03d}-{_task['id']}",
                    )
                    coder_calls.append(meta)
                    return text

                review = None
                if arm == "consult":
                    def review(review_context: str) -> tuple[dict[str, Any], dict[str, Any]]:
                        return consult(
                            consultant_role="architect_general",
                            requester_role="coder_escalation",
                            skill="review_before_commit",
                            context=review_context,
                            primitives=primitives,
                            override_priority="background",
                        )

                try:
                    result, raw = run_edit_transaction(
                        llm,
                        task["prompt"],
                        root,
                        target_files=list((task.get("files") or {}).keys()) or None,
                        verify_fn=lambda tx_root, cmd=task["verifier_cmd"]: _run_verifier(tx_root, cmd),
                        review_before_commit=review,
                        enable_review_before_commit=(arm == "consult"),
                    )
                except Exception as exc:
                    result, raw, error = None, "", f"{type(exc).__name__}: {exc}"
                verifier_ok = bool(result and result.ok and _run_verifier(root, task["verifier_cmd"]))
                bucket = _classify(result, raw, verifier_ok, error)
                row = {
                    "run_ts": run_ts,
                    "arm": arm,
                    "turn_index": idx,
                    "task_id": task["id"],
                    "task_kind": task.get("kind"),
                    "bucket": bucket,
                    "txn_ok": bool(result and result.ok),
                    "verifier_ok": verifier_ok,
                    "written": result.written if result else [],
                    "deleted": result.deleted if result else [],
                    "error": error or (result.error if result else ""),
                    "coder_call": coder_calls[0] if coder_calls else {},
                    "coder_calls_total": len(coder_calls),
                    "consult_calls": primitives.calls,
                    "consult_events": result.consult_events if result else [],
                }
                rows.append(row)
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                print(
                    f"[{arm}] {idx}/{args.turns} {task['id']} {bucket} "
                    f"coder_calls={len(coder_calls)} consult_calls={len(primitives.calls)}",
                    flush=True,
                )

    summary = {
        "kind": "internal_interaction_j17_live_ab",
        "run_ts": run_ts,
        "orch_head": _git_head(),
        "api_url": args.api_url,
        "turns_requested_per_arm": args.turns,
        "unique_task_count": len(tasks),
        "fixed_slice_repetitions": args.turns / max(1, len(tasks)),
        "rows_path": str(rows_path),
        "summary": _summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "summary.md").write_text(
        "# J17 Internal Interaction Live A/B\n\n"
        f"- run_ts: `{run_ts}`\n"
        f"- orch_head: `{summary['orch_head']}`\n"
        f"- rows: `{rows_path}`\n"
        f"- unique tasks: `{len(tasks)}` repeated to `{args.turns}` turns per arm\n\n"
        "```json\n" + json.dumps(summary["summary"], indent=2, sort_keys=True) + "\n```\n"
    )
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
