#!/usr/bin/env python3
"""RD-12 + TM-8 — one shared 50-question shadow-reviewer replay harness.

ONE harness, BOTH gates (per reviewer-decision-plane.md RD-12 and
reviewer-trace-materialization.md TM-8):

  * **TM-8 coverage gate**: % of review invocations that produce trace rows over
    the pinned question set (must be ~100% before H4 starts), plus verification
    that the rows carry per-step ``phase`` tags, ``executor_model_id`` and
    PLAN_REMINDER events (plan-compliance metrics, intake-835).
  * **RD-12 baseline**: per-decision ``latency_ms`` + token accounting (prompt AND
    completion), parse-failure fallback count (distinct — never dropped or
    double-counted), enforcement side-effect count (must be 0 in shadow), and the
    reviewer-ON vs reviewer-OFF **overhead-delta scaffold** that hands the H-LB
    baseline.

NO INFERENCE IS STARTED HERE. This script talks ONLY to a live OpenAI-compatible
server you point it at (llama-server on the GPU lane, or the orchestrator stack's
``/v1``). It never spawns a model or a server. It REQUIRES a live server: the
reviewer model answers the 50 questions and the harness records what it does.

Prerequisites (documented for the operator):
  * A live OpenAI-compatible endpoint (e.g. the production llama.cpp kernel at
    ``http://<host>:<port>/v1``) serving the reviewer model, reachable from this
    host.
  * Model artifact: any text model whose chat template tolerates a JSON-only
    instruction (the v9 production kernel set; the reviewer-plane interim target
    is 122B-IQ2; ``architect_general``-tier models are the production default).
  * Env: nothing beyond ``PYTHONPATH=<orchestrator repo root>`` (the script
    inserts it itself). Use the orchestrator's venv if the server is the stack.

Usage (all numbers are observation-grade until P-REV-1/P-AB-1):

  # TM-8 + RD-12 numbers in one run (reviewer-ON, shadow):
  python3 scripts/review/review_replay_50.py \
      --mode shadow \
      --base-url http://127.0.0.1:8080/v1 \
      --model <model-id> \
      --questions data/trace/review_replay_50.json \
      --trace-db data/trace/review_replay_rd12.sqlite \
      --report data/trace/review_replay_report_shadow.json \
      --artifacts-dir data/trace/review_replay_artifacts

  # Reviewer-OFF baseline (the H-LB "reviewer-off" arm):
  python3 scripts/review/review_replay_50.py \
      --mode off \
      --base-url http://127.0.0.1:8080/v1 \
      --model <model-id> \
      --questions data/trace/review_replay_50.json \
      --report data/trace/review_replay_report_off.json

  # Overhead-delta scaffold (H-LB baseline): folds both reports:
  python3 scripts/review/review_replay_50.py \
      --mode delta \
      --on-report data/trace/review_replay_report_shadow.json \
      --off-report data/trace/review_replay_report_off.json \
      --report data/trace/review_replay_report_delta.json

The ``shadow`` run writes one trace DB you can fold with the same coverage helper
the unit tests use (``src.trace.coverage``). Exit code 3 = a gate FAILED
(coverage < 100%, enforcement side-effects != 0, or trace DB not written).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make src/ importable when invoked directly (scripts/trace/cli.py pattern).
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

REVIEW_SESSION_PREFIX = "rr50-"


# ── live-server seam (stdlib only; mirrors production llm_call's str contract) ──


class LiveServerPrimitives:
    """Minimal ``llm_call`` seam over an OpenAI-compatible ``/v1`` endpoint.

    Mirrors the production seam's contract: ``llm_call`` returns a plain ``str``,
    so the service applies the same token-estimation it would in production.
    Server-reported usage is ALSO captured per call and surfaced in the report
    (``usage_log``) so the RD-12 numbers can be cross-checked against the
    endpoint's own counts. Never starts a server.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.usage_log: list[dict[str, Any]] = []

    def llm_call(
        self, prompt: str, role: str | None = None, n_tokens: int | None = None, **_: Any
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if n_tokens is not None:
            payload["max_tokens"] = int(n_tokens)
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - operator-pinned URL
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"server {exc.code}: {exc.read()[:200]!r}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"server unreachable at {self.base_url}: {exc.reason}") from exc
        text = str((data.get("choices") or [{}])[0].get("message", {}).get("content", "") or "")
        usage = data.get("usage") or {}
        self.usage_log.append(
            {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        )
        return text


# ── question-set plumbing ─────────────────────────────────────────────────────


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = data.get("tasks") or []
    if len(tasks) != 50:
        raise SystemExit(f"question set {path} has {len(tasks)} tasks; expected 50 (pinned set)")
    return tasks


def build_sanitized_view(q: dict[str, Any]) -> dict[str, Any]:
    """Project one pinned question onto the reviewer-visible CandidatePackage view."""
    return {
        "task_ref": q["task_id"],
        "objective": q.get("objective", ""),
        "outputs": q.get("outputs", [{"type": "text", "ref": q.get("task_id", "out")}]),
        "acceptance_checks": q.get("acceptance_checks", []),
        "sanitization": {
            "applied": True,
            "removed_fields": ["author_self_assessment", "quality_labels"],
        },
    }


def session_id_for(task_id: str) -> str:
    return f"{REVIEW_SESSION_PREFIX}{task_id}"


def plan_steps_for(q: dict[str, Any]) -> list[dict[str, Any]]:
    """Synthetic plan for reminder-event verification (TM-8): one step per check."""
    executor = q.get("executor_model_id", "worker")
    return [
        {"id": f"S{i + 1}", "actor": executor, "action": c.get("statement", "")[:80]}
        for i, c in enumerate(q.get("acceptance_checks", []))
    ] or [{"id": "S1", "actor": executor, "action": q.get("objective", "")[:80]}]


def latest_decision_detail(trace_db: Path, session_id: str) -> dict[str, Any]:
    """The most recent ``review_decision`` row's detail for ``session_id``.

    The service's own per-decision accounting (``latency_ms`` + ``tokens``) is
    written into the trace row inside ``review_candidate``; the harness mirrors
    THOSE numbers into ``shadow_decide``'s artifact telemetry so the artifact and
    the trace row can never disagree.
    """
    import sqlite3

    if not Path(trace_db).exists():
        return {}
    conn = sqlite3.connect(str(trace_db))
    try:
        row = conn.execute(
            "SELECT detail_json FROM event WHERE session_id = ? "
            "AND category = 'review_decision' ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


# ── runs ──────────────────────────────────────────────────────────────────────


def run_shadow(
    questions: list[dict[str, Any]],
    primitives: LiveServerPrimitives,
    trace_db: Path,
    artifacts_dir: Path | None,
    report_out: Path,
) -> dict[str, Any]:
    """One shadow run: review_candidate + shadow_decide per question (TM-3/8 always-on)."""
    from src.proactive_delegation.review_service import ArchitectReviewService
    from src.trace.coverage import (
        aggregate_decision_metrics,
        enforcement_side_effects,
        review_trace_coverage,
        verify_phase_metadata,
    )

    trace_db = Path(trace_db)
    if trace_db.exists():
        trace_db.unlink()  # fresh run: a stale DB would pollute coverage
    svc = ArchitectReviewService(
        primitives,
        trace_db_path=str(trace_db),
        warn_only=True,
    )
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    sessions: list[str] = []
    decisions: list[dict[str, Any]] = []
    for q in questions:
        task_id = q["task_id"]
        session_id = session_id_for(task_id)
        sessions.append(session_id)
        view = build_sanitized_view(q)
        executor = q.get("executor_model_id")
        t0 = time.perf_counter()
        review = svc.review_candidate(
            view,
            subtask_id=task_id,
            session_id=session_id,
            executor_model_id=executor,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        # Mirror the service's own accounting (written to the trace row) into the
        # artifact telemetry so the two can never disagree (RD-12).
        detail = latest_decision_detail(trace_db, session_id)
        latency_ms = detail.get("latency_ms")
        tokens = detail.get("tokens") or {"tokens_in": 0, "tokens_out": 0, "chars_out": 0}
        artifact = svc.shadow_decide(
            review,
            session_id=session_id,
            latency_ms=latency_ms,
            tokens=tokens,
            executor_model_id=executor,
        )
        if artifacts_dir is not None:
            (artifacts_dir / f"{task_id}.json").write_text(
                json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
            )
        decisions.append(
            {
                "task_id": task_id,
                "session_id": session_id,
                "decision": artifact.get("decision"),
                "shadow": artifact.get("shadow"),
                "latency_ms": latency_ms,
                "wall_ms": wall_ms,
                "tokens_in": (tokens or {}).get("tokens_in", 0),
                "tokens_out": (tokens or {}).get("tokens_out", 0),
            }
        )
        # TM-8: reminder events — exercise the cheap knob with emit=True (intake-835).
        svc.build_plan_reminder(
            plan_steps_for(q),
            cadence_n=5,
            step_index=10,
            emit=True,
            session_id=session_id,
        )

    # Gates (TM-8 / RD-12) over the trace DB the run just wrote.
    coverage = review_trace_coverage(trace_db, sessions)
    metrics = aggregate_decision_metrics(trace_db, sessions)
    enforcement = enforcement_side_effects(trace_db, sessions)
    phase_meta = verify_phase_metadata(trace_db, sessions)
    server_usage = primitives.usage_log

    report = {
        "run": "shadow",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(questions),
        "trace_db": str(trace_db),
        "service": {
            "parse_failure_count": svc.parse_failure_count,
            "model_call_failures": svc.model_call_failures,
        },
        "trace_coverage": coverage,
        "decision_metrics": metrics,
        "enforcement_side_effect_count": len(enforcement),
        "enforcement_side_effects": enforcement,
        "phase_metadata": phase_meta,
        "per_question": decisions,
        "server_usage": server_usage,
    }
    _write_report(report, report_out)
    _gate_check(report)
    return report


def run_off(
    questions: list[dict[str, Any]],
    primitives: LiveServerPrimitives,
    report_out: Path,
) -> dict[str, Any]:
    """Reviewer-OFF baseline: the same question loop WITHOUT any review decision.

    Measures the harness/question-loop cost only (one connectivity probe to the
    same endpoint, then pure loop timing). The H-LB overhead delta is
    ``mean(latency_ms)`` of the shadow run minus this baseline — the scaffold is
    documented in the delta mode; the honest paired measurement is LB-4's job.
    """
    probe = "Reply with the single word OK."
    primitives.llm_call(probe, n_tokens=16)  # proves liveness; not a review call
    t0 = time.perf_counter()
    for q in questions:
        build_sanitized_view(q)
    loop_ms = (time.perf_counter() - t0) * 1000.0
    report = {
        "run": "off",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(questions),
        "loop_ms_total": loop_ms,
        "loop_ms_mean": loop_ms / len(questions) if questions else 0.0,
        "probe_usage": primitives.usage_log,
        "note": "reviewer-off baseline: no review decision performed; connectivity probe only",
    }
    _write_report(report, report_out)
    return report


def run_delta(on_report: Path, off_report: Path, report_out: Path) -> dict[str, Any]:
    """Overhead-delta scaffold for H-LB: reviewer-ON mean latency vs reviewer-OFF."""
    on = json.loads(Path(on_report).read_text(encoding="utf-8"))
    off = json.loads(Path(off_report).read_text(encoding="utf-8"))
    on_lat = (on.get("decision_metrics") or {}).get("latency_ms", {}) or {}
    off_mean = off.get("loop_ms_mean", 0.0)
    on_mean = on_lat.get("mean")
    tokens_in = (on.get("decision_metrics") or {}).get("tokens_in", {}) or {}
    tokens_out = (on.get("decision_metrics") or {}).get("tokens_out", {}) or {}
    report = {
        "run": "delta",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "on_report": str(on_report),
        "off_report": str(off_report),
        "reviewer_on_mean_latency_ms": on_mean,
        "reviewer_off_loop_mean_ms": off_mean,
        "overhead_delta_ms": (on_mean - off_mean) if on_mean is not None else None,
        "overhead_ratio": (on_mean / off_mean) if (on_mean and off_mean) else None,
        "tokens_in_total": tokens_in.get("sum"),
        "tokens_out_total": tokens_out.get("sum"),
        "note": (
            "Observation-grade scaffold (P-AB-1/P-SPEED-OBJ pending). The H-LB "
            "baseline is reviewer-on mean per-decision latency_ms vs reviewer-off "
            "mean loop cost on the same pinned set; paired task-rate A/B is LB-4."
        ),
    }
    _write_report(report, report_out)
    return report


def _write_report(report: dict[str, Any], report_out: Path) -> None:
    out = Path(report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report written: {out}")


def _gate_check(report: dict[str, Any]) -> None:
    """Exit 3 when a gate fails: coverage < 100%, or enforcement side-effects != 0."""
    coverage = report.get("trace_coverage", {}) or {}
    pct = coverage.get("coverage_pct", 0.0)
    enforcement = report.get("enforcement_side_effect_count", -1)
    missing = coverage.get("missing_session_ids", [])
    fails: list[str] = []
    if pct < 100.0:
        fails.append(f"trace coverage {pct}% != 100% (missing={missing})")
    if enforcement != 0:
        fails.append(f"enforcement side-effect count {enforcement} != 0")
    if fails:
        print("GATE FAILED:", "; ".join(fails))
        sys.exit(3)
    print(
        f"GATES PASS: coverage={pct}% enforcement_side_effects={enforcement} "
        f"parse_failures={report.get('service', {}).get('parse_failure_count')}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("shadow", "off", "delta"), required=True)
    parser.add_argument(
        "--base-url", help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8080/v1"
    )
    parser.add_argument("--model", help="model id the endpoint serves (chat completions)")
    parser.add_argument(
        "--questions", default=str(Path(_REPO) / "data" / "trace" / "review_replay_50.json")
    )
    parser.add_argument(
        "--trace-db", default=str(Path(_REPO) / "data" / "trace" / "review_replay_rd12.sqlite")
    )
    parser.add_argument("--report", required=True, help="report JSON to write")
    parser.add_argument("--artifacts-dir", help="optional dir for per-question decision artifacts")
    parser.add_argument("--on-report", help="delta mode: reviewer-ON report JSON")
    parser.add_argument("--off-report", help="delta mode: reviewer-OFF report JSON")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    report_out = Path(args.report)

    if args.mode == "delta":
        if not args.on_report or not args.off_report:
            parser.error("--mode delta requires --on-report and --off-report")
        run_delta(Path(args.on_report), Path(args.off_report), report_out)
        return 0

    if not args.base_url or not args.model:
        parser.error("--mode shadow/off requires --base-url and --model")
    primitives = LiveServerPrimitives(args.base_url, args.model, timeout=args.timeout)
    questions = load_questions(Path(args.questions))

    try:
        if args.mode == "shadow":
            run_shadow(questions, primitives, Path(args.trace_db), args.artifacts_dir, report_out)
        else:
            run_off(questions, primitives, report_out)
    except RuntimeError as exc:
        # The common failure is an unreachable/refused endpoint — fail cleanly
        # with the server address instead of a traceback.
        print(f"replay aborted: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
