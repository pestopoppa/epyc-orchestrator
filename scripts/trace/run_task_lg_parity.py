#!/usr/bin/env python3
"""TM-7 durable-resume PARITY driver: run_task_lg vs run_task decision-chain diff.

What this is
------------
The manifest entry ``BULK-langgraph-tm7-parity`` gates enabling interrupt-based
review gates (H3) on a proof that the LangGraph durable path produces the SAME
decision-chain / trace coverage as the baseline non-durable path — and that a
run interrupted mid-flight and *resumed* from its checkpoint reconstructs the
same chain (TM-7 durable resume, TM-8 coverage).

This driver runs a fixed task set through two "arms" and diffs the resulting
trace decision-chains for parity. The arms are:

    run_task              baseline pydantic_graph execution (src.graph.graph)
    run_task_lg           LangGraph, no checkpointer
    run_task_lg_durable   LangGraph + AsyncSqliteSaver (durable per-super-step)
    run_task_lg_resume    durable run interrupted then resumed from checkpoint

The two arms named on the CLI (``--arms A,B``) are executed over the same task
set/seed and their decision-chains compared step-for-step + by phase coverage.

Two clean layers (why the default needs no model)
-------------------------------------------------
1. **Parity check** (``parity_verdict`` and friends) is a PURE, non-inference
   trace/decision-chain diff. Given two chain artifacts it computes a
   coverage/parity verdict. This is fully unit-testable with fixtures and is
   also directly reachable via ``--parity-a FILE --parity-b FILE`` to diff two
   already-recorded runs with zero model calls.
2. **Execution** (running the arms to *produce* those chains) needs the real
   nodes (frontdoor, architect_general) and is therefore INFERENCE-GATED. It is
   OFF by default: the default invocation validates config, checks the forbidden
   review-gate flag, resolves the task set, and prints the run plan — no server,
   no model. Pass ``--execute`` (or set ``RUN_TASK_LG_PARITY_EXECUTE=1``) to run
   the real arms; that leg needs a live inference window.

Forbidden-flag note
-------------------
The fabricated manifest entry referenced a NONEXISTENT flag
``GRAPH_INTERRUPT_REVIEW_GATES``. The REAL flag that gates interrupt-based
review gates is ``generalized_interrupts`` (env ``ORCHESTRATOR_GENERALIZED_INTERRUPTS``
/ ``ORCHESTRATOR_FEATURE_GENERALIZED_INTERRUPTS``; ``FeatureSpec`` env suffix
``GENERALIZED_INTERRUPTS`` in ``src/features.py``), together with
``approval_gates``. The parity run MUST keep both OFF — that is precisely the
gate this driver clears. This script checks the real flags and warns loudly if
the fabricated env name is set (it does nothing).

Usage (dry-run / plan — no inference):
    python scripts/trace/run_task_lg_parity.py \
        --arms run_task_lg,run_task --fixed-task-set data/trace/parity_task_set.json --seed 42

Usage (pure parity diff of two recorded chains — no inference):
    python scripts/trace/run_task_lg_parity.py \
        --parity-a runA.json --parity-b runB.json --output verdict.json

Usage (live parity leg — needs an inference window):
    python scripts/trace/run_task_lg_parity.py \
        --arms run_task_lg,run_task --fixed-task-set data/trace/parity_task_set.json \
        --seed 42 --execute
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

# Make src/ importable when invoked directly (mirrors scripts/trace/cli.py).
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Flag constants (the fabricated → real substitution)
# ---------------------------------------------------------------------------

#: The nonexistent flag the fabricated manifest entry referenced.
FABRICATED_FLAG = "GRAPH_INTERRUPT_REVIEW_GATES"

#: The REAL feature that gates interrupt-based review gates. ``env_suffix`` is the
#: FeatureSpec env var suffix (prefix ``ORCHESTRATOR_`` or ``ORCHESTRATOR_FEATURE_``).
REAL_FLAG_FIELD = "generalized_interrupts"
REAL_FLAG_ENV_SUFFIX = "GENERALIZED_INTERRUPTS"

#: THE forbidden flag for the parity run — the real substitution for the
#: fabricated ``GRAPH_INTERRUPT_REVIEW_GATES``. Interrupt-based review gates in
#: the pluggable/generalized sense are gated by this; it MUST be OFF (H3 not yet
#: unblocked). ``approval_gates`` is a legitimate production baseline (and a
#: dependency of this flag) so it is reported for context but does NOT fail the
#: gate on its own.
FORBIDDEN_FLAG = "generalized_interrupts"
CONTEXT_FLAGS = ("approval_gates", "resume_tokens")

#: Recognised arms. Value = a short human label; execution wiring lives in
#: :func:`_run_arm` (gated behind ``--execute``).
KNOWN_ARMS = {
    "run_task": "baseline pydantic_graph execution",
    "run_task_lg": "LangGraph, no checkpointer",
    "run_task_lg_durable": "LangGraph + durable AsyncSqliteSaver",
    "run_task_lg_resume": "durable run interrupted then resumed from checkpoint",
}

DEFAULT_OUTPUT = _REPO / "data" / "trace" / "tm7_realnode_parity.json"
DEFAULT_CHECKPOINT = _REPO / "data" / "graph_checkpoints.sqlite"

#: A tiny deterministic built-in corpus so the plan resolves even before a real
#: ``--fixed-task-set`` file exists. Each task: id + prompt + start_role.
DEFAULT_CORPUS: list[dict[str, str]] = [
    {"task_id": "tm7-smoke-1", "prompt": "What is 2 + 2?", "start_role": "frontdoor"},
    {"task_id": "tm7-smoke-2", "prompt": "Summarize the phrase 'hello world'.", "start_role": "frontdoor"},
    {"task_id": "tm7-smoke-3", "prompt": "Name the capital of France.", "start_role": "frontdoor"},
    {"task_id": "tm7-smoke-4", "prompt": "Write a one-line docstring for add(a, b).", "start_role": "frontdoor"},
]


# ---------------------------------------------------------------------------
# Phase mapping (reused from the trace decision-chain replay; local fallback)
# ---------------------------------------------------------------------------

# Canonical fallback copy of src.trace.query's phase map so the parity core is
# importable + testable in isolation and resilient to query.py churn.
_FALLBACK_PHASE_ORDER = (
    "task", "plan", "reminder", "review", "gate", "escalation", "outcome",
)
_FALLBACK_CATEGORY_PHASE = {
    "task_start": "task",
    "candidate_package": "plan",
    "plan_reminder": "reminder",
    "review_decision": "review",
    "verification_report": "gate",
    "safety_verdict": "gate",
    "review_escalation": "escalation",
    "task_end": "outcome",
}


def _load_phase_map() -> tuple[tuple[str, ...], dict[str, str]]:
    """Return (phase_order, category->phase). Prefer the real map from query.py."""
    try:
        from src.trace.query import _CHAIN_PHASE_ORDER, _CATEGORY_PHASE  # type: ignore

        return tuple(_CHAIN_PHASE_ORDER), dict(_CATEGORY_PHASE)
    except Exception:
        return _FALLBACK_PHASE_ORDER, dict(_FALLBACK_CATEGORY_PHASE)


PHASE_ORDER, CATEGORY_PHASE = _load_phase_map()


def phase_for_step(step: dict[str, Any]) -> str | None:
    """Best-effort phase for a chain step.

    Precedence: explicit ``phase`` field -> category lookup -> the category
    string itself (so unknown categories still contribute distinct coverage).
    """
    if step.get("phase"):
        return str(step["phase"])
    cat = step.get("category")
    if cat is None:
        return None
    return CATEGORY_PHASE.get(str(cat), str(cat))


# ---------------------------------------------------------------------------
# Parity core (PURE — no inference, fully unit-testable)
# ---------------------------------------------------------------------------

#: Fields that define a step's identity for parity (order = signature order).
DEFAULT_PARITY_KEYS = ("phase", "category", "role", "status", "decision", "next_node")

#: Volatile / run-specific fields that must never affect a parity verdict.
VOLATILE_KEYS = frozenset({
    "id", "ts_utc", "ts", "source_line", "source_path", "detail_json",
    "latency_ms", "started_at", "ended_at", "duration_ms", "session_id",
    "trial_id", "thread_id", "redacted",
})


def extract_chain(artifact: Any) -> list[dict[str, Any]]:
    """Normalise a trace/decision-chain artifact into a list of step dicts.

    Accepts several on-disk shapes:
      * a bare ``list`` of step dicts / strings
      * ``{"chain": [...]}``       — ``src.trace.query.decision_chain`` output
      * ``{"steps": [...]}``       — a run trace (strings or dicts)
      * ``{"events": [...]}``      — raw event rows
      * ``{"role_history": [...]}``— a ``TaskResult``-shaped dict (synthesised)
    """
    if artifact is None:
        return []
    if isinstance(artifact, list):
        raw = artifact
    elif isinstance(artifact, dict):
        if "chain" in artifact:
            raw = artifact["chain"]
        elif "steps" in artifact:
            raw = artifact["steps"]
        elif "events" in artifact:
            raw = artifact["events"]
        elif "role_history" in artifact:
            raw = [
                {"role": r, "category": "role_step", "phase": "role"}
                for r in (artifact.get("role_history") or [])
            ]
        else:
            raise ValueError(
                "unrecognised parity artifact shape: expected one of "
                "'chain'/'steps'/'events'/'role_history' or a bare list"
            )
    else:
        raise ValueError(f"parity artifact must be a list or dict, got {type(artifact).__name__}")

    steps: list[dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, dict):
            # Pull a couple of common nested fields up to the top level so they
            # can be parity keys (decision often lives in detail_json).
            step = dict(item)
            detail = step.get("detail_json") or step.get("detail")
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except Exception:
                    detail = None
            if isinstance(detail, dict):
                for k in ("decision", "next_node"):
                    if k not in step and k in detail:
                        step[k] = detail[k]
            steps.append(step)
        else:
            steps.append({"category": str(item), "step": str(item)})
    return steps


def normalize_step(step: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Project a step onto the parity ``keys`` (missing -> None), filling ``phase``."""
    out: dict[str, Any] = {}
    for k in keys:
        if k == "phase":
            out[k] = phase_for_step(step)
        else:
            out[k] = step.get(k)
    return out


def chain_signature(chain: list[dict[str, Any]], keys: tuple[str, ...]) -> list[tuple]:
    """Ordered list of per-step signature tuples over the parity ``keys``."""
    return [tuple(normalize_step(s, keys).get(k) for k in keys) for s in chain]


def coverage(chain: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Phase + category coverage sets (sorted lists) for a chain."""
    phases: set[str] = set()
    categories: set[str] = set()
    for s in chain:
        ph = phase_for_step(s)
        if ph is not None:
            phases.add(ph)
        cat = s.get("category")
        if cat is not None:
            categories.add(str(cat))
    return {"phases": sorted(phases), "categories": sorted(categories)}


def parity_verdict(
    artifact_a: Any,
    artifact_b: Any,
    *,
    keys: tuple[str, ...] = DEFAULT_PARITY_KEYS,
    label_a: str = "A",
    label_b: str = "B",
) -> dict[str, Any]:
    """Compute a coverage/parity verdict between two chain artifacts.

    Parity holds iff the two chains have:
      * equal length,
      * identical per-step signatures over ``keys`` (in order), AND
      * identical phase + category coverage.

    Returns a JSON-serialisable verdict dict (see keys below).
    """
    keys = tuple(keys)
    chain_a = extract_chain(artifact_a)
    chain_b = extract_chain(artifact_b)

    sig_a = chain_signature(chain_a, keys)
    sig_b = chain_signature(chain_b, keys)

    cov_a = coverage(chain_a)
    cov_b = coverage(chain_b)
    phases_only_a = sorted(set(cov_a["phases"]) - set(cov_b["phases"]))
    phases_only_b = sorted(set(cov_b["phases"]) - set(cov_a["phases"]))
    cats_only_a = sorted(set(cov_a["categories"]) - set(cov_b["categories"]))
    cats_only_b = sorted(set(cov_b["categories"]) - set(cov_a["categories"]))
    coverage_match = not (phases_only_a or phases_only_b or cats_only_a or cats_only_b)

    step_diffs: list[dict[str, Any]] = []
    for i in range(max(len(sig_a), len(sig_b))):
        a = sig_a[i] if i < len(sig_a) else None
        b = sig_b[i] if i < len(sig_b) else None
        if a != b:
            step_diffs.append({
                "index": i,
                label_a: dict(zip(keys, a)) if a is not None else None,
                label_b: dict(zip(keys, b)) if b is not None else None,
            })

    parity = (
        len(chain_a) == len(chain_b)
        and coverage_match
        and not step_diffs
    )

    return {
        "parity": parity,
        "verdict": "PASS" if parity else "FAIL",
        "keys": list(keys),
        "labels": {"a": label_a, "b": label_b},
        "len_a": len(chain_a),
        "len_b": len(chain_b),
        "length_match": len(chain_a) == len(chain_b),
        "coverage_match": coverage_match,
        "coverage_a": cov_a,
        "coverage_b": cov_b,
        "phases_only_in_a": phases_only_a,
        "phases_only_in_b": phases_only_b,
        "categories_only_in_a": cats_only_a,
        "categories_only_in_b": cats_only_b,
        "n_step_diffs": len(step_diffs),
        "step_diffs": step_diffs,
    }


def load_artifact(path: str | Path) -> Any:
    """Load a JSON chain/trace artifact from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"parity artifact not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Forbidden-flag check (real flag, not the fabricated name)
# ---------------------------------------------------------------------------


def check_review_gate_flags() -> dict[str, Any]:
    """Assert the interrupt-based review-gate flag is OFF for the parity run.

    Returns a structured report. ``ok`` is True iff :data:`FORBIDDEN_FLAG`
    (``generalized_interrupts`` — the real substitution for the fabricated
    ``GRAPH_INTERRUPT_REVIEW_GATES``) is disabled. ``CONTEXT_FLAGS`` (e.g.
    ``approval_gates``, a legitimate prod baseline) are reported but do NOT fail
    the gate. Also flags the case where the *fabricated* env var name is set (it
    is a no-op and should be renamed in the manifest).
    """
    report: dict[str, Any] = {
        "real_flag": REAL_FLAG_FIELD,
        "real_flag_env": f"ORCHESTRATOR_{REAL_FLAG_ENV_SUFFIX}",
        "fabricated_flag": FABRICATED_FLAG,
        "flags": {},
        "context_flags": {},
        "warnings": [],
        "ok": True,
    }

    # A set fabricated env var does nothing — surface it so the manifest gets fixed.
    for env_name in (
        FABRICATED_FLAG,
        f"ORCHESTRATOR_{FABRICATED_FLAG}",
        f"ORCHESTRATOR_FEATURE_{FABRICATED_FLAG}",
    ):
        if os.environ.get(env_name) not in (None, ""):
            report["warnings"].append(
                f"env {env_name!r} is set but is a NO-OP — the real flag is "
                f"{REAL_FLAG_FIELD!r} (env ORCHESTRATOR_{REAL_FLAG_ENV_SUFFIX})."
            )

    try:
        from src.features import features as _features

        f = _features()
        val = bool(getattr(f, FORBIDDEN_FLAG, False))
        report["flags"][FORBIDDEN_FLAG] = val
        if val:
            report["ok"] = False
            report["warnings"].append(
                f"forbidden flag {FORBIDDEN_FLAG!r} is ON — must be OFF for the "
                "TM-7 parity run (H3 interrupt review gates not yet unblocked)."
            )
        for name in CONTEXT_FLAGS:
            report["context_flags"][name] = bool(getattr(f, name, False))
    except Exception as exc:  # pragma: no cover - defensive
        report["warnings"].append(f"could not read features: {exc!r}")
        report["flags_readable"] = False
    return report


# ---------------------------------------------------------------------------
# Task-set resolution
# ---------------------------------------------------------------------------


def resolve_task_set(
    *,
    fixed_task_set: str | None,
    corpus: str | None,
    n: int | None,
    seed: int,
    allow_missing: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve the task set to run through both arms.

    Precedence: an existing ``--fixed-task-set`` file wins; otherwise the
    built-in :data:`DEFAULT_CORPUS`. ``--n`` caps the count; ``--seed`` makes the
    (optional) subsample deterministic. Returns (tasks, provenance).
    """
    provenance: dict[str, Any] = {"seed": seed, "requested_n": n}
    tasks: list[dict[str, Any]]

    if fixed_task_set:
        p = Path(fixed_task_set)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            tasks = data["tasks"] if isinstance(data, dict) and "tasks" in data else data
            provenance["source"] = f"fixed-task-set:{p}"
        elif allow_missing:
            tasks = list(DEFAULT_CORPUS)
            provenance["source"] = f"builtin (missing {p}, dry-run fallback)"
            provenance["warning"] = f"fixed-task-set {p} not found; using built-in corpus"
        else:
            raise FileNotFoundError(
                f"--fixed-task-set {p} not found (required under --execute)"
            )
    else:
        tasks = list(DEFAULT_CORPUS)
        provenance["source"] = f"builtin corpus:{corpus or 'tm7-smoke'}"

    if not isinstance(tasks, list) or not tasks:
        raise ValueError("resolved task set is empty or not a list")

    # Deterministic subsample when n < len.
    if n is not None and 0 < n < len(tasks):
        rng = random.Random(seed)
        tasks = rng.sample(tasks, n)
        provenance["subsampled"] = True
    provenance["resolved_n"] = len(tasks)
    return tasks, provenance


# ---------------------------------------------------------------------------
# Execution (INFERENCE-GATED — never runs without --execute)
# ---------------------------------------------------------------------------


async def _run_arm(arm: str, task: dict[str, Any], *, checkpoint_path: Path, seed: int) -> dict[str, Any]:
    """Execute one arm for one task and return its recorded decision-chain.

    INFERENCE-GATED: this calls the real graph nodes (models). Only reached from
    :func:`execute_parity` under ``--execute``. Returns
    ``{"task_id", "arm", "result", "chain"}``.
    """
    from src.graph.state import TaskDeps, TaskState
    from src.trace.query import decision_chain

    task_id = str(task["task_id"])
    prompt = str(task["prompt"])
    start_role = task.get("start_role", "frontdoor")
    thread_id = f"{task_id}:{arm}"

    state = TaskState(task_id=task_id, prompt=prompt)
    deps = TaskDeps()

    if arm == "run_task":
        from src.graph.graph import run_task

        result = await run_task(state, deps, start_role=start_role)
    elif arm == "run_task_lg":
        from src.graph.langgraph.graph import run_task_lg

        result = await run_task_lg(state, deps, start_role=start_role, thread_id=thread_id)
    elif arm == "run_task_lg_durable":
        from src.graph.langgraph.graph import run_task_lg_durable

        result = await run_task_lg_durable(
            state, deps, start_role=start_role,
            checkpoint_path=checkpoint_path, thread_id=thread_id,
        )
    elif arm == "run_task_lg_resume":
        # durable run that pauses at a review interrupt, then resume from the
        # checkpoint across a simulated restart. (In a real live leg the pause is
        # driven by an enabled gate; parity compares the reconstructed chain.)
        from src.graph.langgraph.graph import resume_task_lg, run_task_lg_durable

        await run_task_lg_durable(
            state, deps, start_role=start_role,
            checkpoint_path=checkpoint_path, thread_id=thread_id,
        )
        result = await resume_task_lg(
            deps, thread_id=thread_id, checkpoint_path=checkpoint_path,
            resume_value="APPROVE", state=state,
        )
    else:
        raise ValueError(f"unknown arm: {arm}")

    chain = decision_chain(session_id=thread_id)
    return {
        "task_id": task_id,
        "arm": arm,
        "thread_id": thread_id,
        "result": {
            "answer": getattr(result, "answer", None),
            "success": getattr(result, "success", None),
            "role_history": getattr(result, "role_history", None),
            "turns": getattr(result, "turns", None),
        },
        "chain": chain,
    }


def execute_parity(
    tasks: list[dict[str, Any]],
    arms: list[str],
    *,
    checkpoint_path: Path,
    seed: int,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    """Run both arms over the task set and diff their chains. INFERENCE-GATED."""
    import asyncio

    if len(arms) != 2:
        raise ValueError(f"parity needs exactly 2 arms, got {arms}")
    arm_a, arm_b = arms

    async def _run_all() -> list[dict[str, Any]]:
        per_task: list[dict[str, Any]] = []
        for task in tasks:
            run_a = await _run_arm(arm_a, task, checkpoint_path=checkpoint_path, seed=seed)
            run_b = await _run_arm(arm_b, task, checkpoint_path=checkpoint_path, seed=seed)
            verdict = parity_verdict(
                run_a["chain"], run_b["chain"], keys=keys, label_a=arm_a, label_b=arm_b,
            )
            per_task.append({
                "task_id": task["task_id"],
                arm_a: run_a["result"],
                arm_b: run_b["result"],
                "verdict": verdict,
            })
        return per_task

    per_task = asyncio.run(_run_all())
    n_pass = sum(1 for t in per_task if t["verdict"]["parity"])
    return {
        "arms": arms,
        "seed": seed,
        "n_tasks": len(tasks),
        "n_pass": n_pass,
        "n_fail": len(tasks) - n_pass,
        "overall": "PASS" if n_pass == len(tasks) else "FAIL",
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_task_lg_parity.py",
        description="TM-7 durable-resume parity driver (run_task_lg vs run_task decision-chain diff).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Arm / task selection.
    p.add_argument(
        "--arms", default="run_task_lg,run_task",
        help="comma-separated pair of arms to compare (default: run_task_lg,run_task). "
             f"Known: {', '.join(KNOWN_ARMS)}",
    )
    p.add_argument(
        "--fixed-task-set", default=None,
        help="path to a JSON task set ({'tasks':[...]} or a bare list of {task_id,prompt,start_role}).",
    )
    p.add_argument(
        "--corpus", default=None,
        help="named built-in corpus (default: tm7-smoke) when no --fixed-task-set is given.",
    )
    p.add_argument("--n", type=int, default=None, help="cap the number of tasks (deterministic subsample under --seed).")
    p.add_argument("--seed", type=int, default=42, help="sampling seed (default: 42).")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"report output path (default: {DEFAULT_OUTPUT}).")
    p.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT), help="durable checkpoint sqlite file.")
    p.add_argument(
        "--parity-keys", default=",".join(DEFAULT_PARITY_KEYS),
        help="comma-separated step fields that define parity (default: %(default)s).",
    )
    # Pure parity-diff mode (no inference).
    p.add_argument("--parity-a", default=None, help="diff mode: path to recorded chain artifact A.")
    p.add_argument("--parity-b", default=None, help="diff mode: path to recorded chain artifact B.")
    # Execution gate.
    p.add_argument(
        "--execute", action="store_true",
        help="ACTUALLY run the arms (inference). Default is dry-run/plan. Also honoured: "
             "env RUN_TASK_LG_PARITY_EXECUTE=1.",
    )
    return p


def _parse_arms(raw: str) -> list[str]:
    arms = [a.strip() for a in raw.split(",") if a.strip()]
    unknown = [a for a in arms if a not in KNOWN_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; known: {sorted(KNOWN_ARMS)}")
    if len(arms) != 2:
        raise SystemExit(f"--arms must name exactly 2 arms, got {arms}")
    return arms


def _write_output(obj: dict[str, Any], output: str | None) -> None:
    if not output:
        return
    outp = Path(output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    keys = tuple(k.strip() for k in args.parity_keys.split(",") if k.strip())

    # --- Pure parity-diff mode: diff two recorded chains, no inference. -------
    if args.parity_a or args.parity_b:
        if not (args.parity_a and args.parity_b):
            raise SystemExit("--parity-a and --parity-b must be given together.")
        verdict = parity_verdict(
            load_artifact(args.parity_a), load_artifact(args.parity_b), keys=keys,
            label_a=Path(args.parity_a).stem, label_b=Path(args.parity_b).stem,
        )
        print(json.dumps(verdict, indent=2, default=str))
        _write_output(verdict, args.output if args.output != str(DEFAULT_OUTPUT) else None)
        return 0 if verdict["parity"] else 3

    arms = _parse_arms(args.arms)
    execute = args.execute or os.environ.get("RUN_TASK_LG_PARITY_EXECUTE") in ("1", "true", "True")

    flag_report = check_review_gate_flags()
    tasks, provenance = resolve_task_set(
        fixed_task_set=args.fixed_task_set, corpus=args.corpus,
        n=args.n, seed=args.seed, allow_missing=not execute,
    )

    plan = {
        "mode": "execute" if execute else "dry-run",
        "arms": arms,
        "arm_labels": {a: KNOWN_ARMS[a] for a in arms},
        "seed": args.seed,
        "parity_keys": list(keys),
        "output": args.output,
        "checkpoint_path": args.checkpoint_path,
        "task_set": provenance,
        "n_tasks": len(tasks),
        "task_ids": [t.get("task_id") for t in tasks],
        "flag_check": flag_report,
    }

    # --- Dry-run / plan (default): validate + print, no model. ---------------
    if not execute:
        print(json.dumps(plan, indent=2, default=str))
        if not flag_report["ok"]:
            print("\nFLAG CHECK FAILED: review-gate flags must be OFF for the parity run.", file=sys.stderr)
            return 4
        print(
            f"\n[dry-run] resolved {len(tasks)} task(s); arms={arms}. "
            "Pass --execute (or RUN_TASK_LG_PARITY_EXECUTE=1) to run the inference leg.",
            file=sys.stderr,
        )
        return 0

    # --- Execute (inference-gated). ------------------------------------------
    if not flag_report["ok"]:
        print(json.dumps(plan, indent=2, default=str))
        print("\nREFUSING TO EXECUTE: review-gate flags must be OFF (H3 gate).", file=sys.stderr)
        return 4

    report = execute_parity(
        tasks, arms, checkpoint_path=Path(args.checkpoint_path), seed=args.seed, keys=keys,
    )
    report["plan"] = plan
    _write_output(report, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=2, default=str))
    return 0 if report["overall"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
