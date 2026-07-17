#!/usr/bin/env python3
"""Screening/placement-queue executor for the H5 RM-3 reviewer screening tier (B3).

This is the standalone executor that fills the declared inference gap in
``scripts/autopilot/actions.py`` (``_action_screening_tier_driver``, the
``NotImplementedError`` around L2275). ``actions.py`` enumerates a screening plan
via ``review_policy_trials.plan_screening_tier`` and stashes it on
``ctx.state["_screening_tier_plan"]``; this module turns that plan (+ the
``reviewer_pool_gen.py`` output and the near-miss corpus manifest) into a
concrete, placement-queue-dispatched screening RUN.

Two responsibilities, cleanly split:

  1. **Plan resolution** (pure, inference-free — this is all the tests exercise):
     join the screening plan's per-pairing queue to the full ``reviewer_pool_gen``
     pairings, expand each (architect, reviewer, grader) pairing x corpus-slice
     into a concrete eval-tower T0/T1 *job spec* carrying **placement-queue
     transport** (``request_priority=background`` + ``workload_class=eval_batch``,
     NEVER a foreground ``/chat`` call — the RM-3 discipline). Dedup by pairing_id,
     drop coresidency-unfit pairings, order by pool priority, and cap N per
     pairing / number of pairings.

  2. **Execution bridge** (env-flag-gated ``AUTOPILOT_SCREENING_TIER_INFERENCE=1``,
     DEFAULT OFF): with the flag OFF the resolved queue is returned as a dry-run
     plan and NO inference happens. With the flag ON, drive the eval-tower over the
     queue via the placement queue, collect per-pairing FA/FR/CR estimates, and
     emit them as JSONL. The execution path is modeled on
     ``bsv_paired_runner.py`` (deferred ``EvalTower`` import, autopilot-stopped
     assumption) and is intentionally NEVER reached by the tests.

Constraints honored (see CLAUDE.md + reviewer-control-plane handoffs):
  * The autopilot daemon may be RUNNING (a parallel session owns its lifecycle).
    This runner NEVER starts/stops it, NEVER writes autopilot_state.json / journals
    / runtime_flags.json, and NEVER edits actions.py, review_policy_trials.py, or
    eval_tower.py (all imported READ-ONLY).
  * All numbers produced here are pre-P-REV-1 OBSERVATIONS (MEASUREMENT.md); they
    never gate a keep/revert/promote decision on their own — the B5 verdict layer
    (or the operator loop) consumes the queue/results.

Wiring (documented, NOT done here — actions.py stays untouched): a future
``actions.py`` edit replaces the ``NotImplementedError`` at
``_action_screening_tier_driver`` with, roughly::

    from scripts.autopilot.screening_tier_runner import run_screening_tier
    result = run_screening_tier(
        plan=ctx.state["_screening_tier_plan"],
        pool_gen_output=pool_gen_output,          # already loaded above
        corpus_manifest=manifest,                 # already loaded above
        output_path=Path(action["results_path"]),
    )
    # env flag drives dry-run vs execute; result["mode"] in {"dry_run","execute"}.
    # The action returns result["resolved_queue"] as its plan and (when executed)
    # result["results"] as the per-pairing FA/FR/CR rows. The runner does NOT
    # write the batch ledger — B5 / the loop does.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(SCRIPT_DIR), str(ORCH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Env flag name mirrors actions._SCREENING_TIER_INFERENCE_ENV exactly so the
# runner and the (future) action handler gate on the SAME variable.
SCREENING_TIER_INFERENCE_ENV = "AUTOPILOT_SCREENING_TIER_INFERENCE"

RUNNER_VERSION = "screening-tier-runner-v1"

# Placement-queue transport constants (RM-3 discipline). These mirror the kwargs
# eval_tower._eval_question already sets on its orchestrator calls, so a screening
# trial rides the SAME background/eval_batch placement path a normal autopilot eval
# fan-out uses — it is never a foreground /chat request.
PLACEMENT_QUEUE_TRANSPORT = "placement_queue"
PLACEMENT_REQUEST_PRIORITY = "background"
PLACEMENT_WORKLOAD_CLASS = "eval_batch"

DEFAULT_CORPUS_MANIFEST = Path(
    "/mnt/raid0/llm/datasets/nearmiss-corpus-v1/manifest.json"
)

# Gold labels that can score a reviewer decision (conclusive ground truth). An
# "observation" gold_confidence is NOT gate-worthy, so those rows are excluded
# from FA/FR/CR just like an inconclusive objective gate is in
# review_policy_trials.reviewer_calibration_from_decisions.
_CONCLUSIVE_GOLD_LABELS = {"accept", "reject"}
_GATE_WORTHY_CONFIDENCE = {"multi_oracle", "single_oracle"}

_APPROVE_DECISIONS = {"approve", "accept"}
_REJECT_DECISIONS = {"reject", "reject_to_empty"}


# ── deferred, defensive cross-module imports ─────────────────────────────────
# review_policy_trials is pure (stdlib only) and safe to import eagerly-ish, but
# we still go through a loader so the runner degrades to a clear error rather than
# an ImportError traceback if the layout changes. EvalTower / call_orchestrator
# are imported ONLY in the execution bridge (they pull httpx + the server client).


def _load_review_policy_trials():
    try:
        from scripts.autopilot import review_policy_trials as rpt
    except Exception:  # noqa: BLE001
        import review_policy_trials as rpt  # type: ignore[no-redef]
    return rpt


def _env_flag_enabled(name: str) -> bool:
    """True iff env var ``name`` is a truthy flag (matches actions._env_flag_enabled)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ══════════════════════════════════════════════════════════════════════════════
# Job spec + resolved queue dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrialJobSpec:
    """One concrete screening trial: a pairing x corpus-slice eval-tower T0/T1 job.

    The ``transport``/``request_priority``/``workload_class`` fields pin the
    placement-queue path — asserting them (and the absence of any ``/chat``
    target) is how the RM-3 no-/chat discipline is enforced/tested.
    """

    pairing_id: str
    architect: str | None
    reviewer: str | None
    grader: str | None
    anchor_arm: str | None
    self_review: bool
    cross_family: bool
    staged_involved: bool
    n: int
    eval_tier: str
    corpus_id: str
    domain: str
    corpus_content_sha256: str
    corpus_n_rows: int
    coresidency_fits: bool | None
    priority_rank: int
    # transport (placement queue, never /chat)
    transport: str = PLACEMENT_QUEUE_TRANSPORT
    request_priority: str = PLACEMENT_REQUEST_PRIORITY
    workload_class: str = PLACEMENT_WORKLOAD_CLASS

    def force_bindings(self) -> dict[str, Any]:
        """eval-tower ``force_role`` bindings that pin the reviewer under test.

        The architect produced the candidate already (corpus rows carry it); the
        screening trial pins the REVIEWER role so the placement queue routes the
        judgement to the model being screened, with the grader as the rubric
        scorer when one is configured.
        """
        return {
            "force_role": self.reviewer or "",
            "grader_role": self.grader or "",
        }

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["force_bindings"] = self.force_bindings()
        d["kind"] = "screening_trial_job"
        return d


@dataclass
class ResolvedScreeningQueue:
    """The concrete, placement-queue-dispatched screening queue (dry-run plan)."""

    jobs: list[TrialJobSpec]
    corpus_slice: dict[str, Any]
    eval_tier: str
    per_pairing_n: int
    pairings_considered: int
    n_deduped: int
    n_pruned_unfit: int
    n_truncated: int
    provenance: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    inference_required: bool = True

    def transport_summary(self) -> dict[str, Any]:
        return {
            "transport": PLACEMENT_QUEUE_TRANSPORT,
            "request_priority": PLACEMENT_REQUEST_PRIORITY,
            "workload_class": PLACEMENT_WORKLOAD_CLASS,
            "uses_chat_endpoint": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "resolved_screening_queue",
            "runner_version": RUNNER_VERSION,
            "eval_tier": self.eval_tier,
            "per_pairing_n": self.per_pairing_n,
            "pairings_considered": self.pairings_considered,
            "n_jobs": len(self.jobs),
            "n_deduped": self.n_deduped,
            "n_pruned_unfit": self.n_pruned_unfit,
            "n_truncated": self.n_truncated,
            "corpus_slice": dict(self.corpus_slice),
            "transport": self.transport_summary(),
            "jobs": [j.to_dict() for j in self.jobs],
            "provenance": dict(self.provenance),
            "notes": list(self.notes),
            "inference_required": self.inference_required,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Plan resolution (pure, inference-free)
# ══════════════════════════════════════════════════════════════════════════════


def _pairing_index(pool_gen_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the full reviewer_pool_gen pairings by pairing_id (last wins is fine —
    dedup happens on the plan side)."""
    idx: dict[str, dict[str, Any]] = {}
    for p in (pool_gen_output or {}).get("pairings") or []:
        if isinstance(p, dict) and p.get("pairing_id") is not None:
            idx[str(p["pairing_id"])] = p
    return idx


def _priority_rank(entry: dict[str, Any]) -> tuple[int, int, int, int, str]:
    """Sort key: most-informative screening trials first (lower tuple = earlier).

    Order: anchor arms (guaranteed baselines) -> staged-candidate pairings (the
    reason the tournament exists) -> cross-family pairings (the collusion-control
    axis) -> everything else; self-review pairings sink last (status-quo alias).
    Ties break deterministically on pairing_id.
    """
    anchor = 0 if entry.get("anchor_arm") else 1
    staged = 0 if entry.get("staged_involved") else 1
    cross = 0 if entry.get("cross_family") else 1
    selfrev = 1 if entry.get("self_review") else 0  # self-review LAST
    return (anchor, staged, cross, selfrev, str(entry.get("pairing_id") or ""))


def _enrich_entry(plan_entry: dict[str, Any], pool_pairing: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a plan queue entry with its full pool-gen pairing (pool-gen wins for
    coresidency/staged/family facts the stripped plan entry lacks)."""
    out = dict(plan_entry)
    if pool_pairing:
        out.setdefault("architect", pool_pairing.get("architect"))
        out.setdefault("reviewer", pool_pairing.get("reviewer"))
        out.setdefault("grader", pool_pairing.get("grader"))
        out["anchor_arm"] = plan_entry.get("anchor_arm") or pool_pairing.get("anchor_arm")
        out["self_review"] = bool(
            plan_entry.get("self_review") or pool_pairing.get("self_review")
        )
        out["cross_family"] = bool(
            plan_entry.get("cross_family") or pool_pairing.get("cross_family_preferred")
        )
        out["staged_involved"] = bool(pool_pairing.get("staged_involved"))
        cores = pool_pairing.get("coresidency") or {}
        out["coresidency_fits"] = cores.get("fits") if isinstance(cores, dict) else None
    else:
        # No evidence in pool-gen -> never silently prune (mirror pool-gen policy).
        out["staged_involved"] = bool(plan_entry.get("staged_involved"))
        out["coresidency_fits"] = None
    return out


def resolve_screening_queue(
    plan: dict[str, Any],
    pool_gen_output: dict[str, Any],
    *,
    cap_per_pairing: int = 0,
    max_pairings: int = 0,
    prune_unfit: bool = True,
    priority: bool = True,
) -> ResolvedScreeningQueue:
    """Expand a ``_screening_tier_plan`` into a concrete placement-queue job queue.

    ``plan`` is the dict shape produced by
    ``review_policy_trials.ScreeningTierPlan.to_dict()`` (i.e. what
    ``actions._action_screening_tier_driver`` stashes on
    ``ctx.state["_screening_tier_plan"]``). ``pool_gen_output`` is the full
    ``reviewer_pool_gen.py`` output (the source of truth for per-pairing
    coresidency/staged/family facts the stripped plan queue does not carry).

    Pure: no inference, no I/O beyond the in-memory dicts. Steps, in order:
    join -> dedup (pairing_id) -> prune coresidency-unfit -> priority sort ->
    truncate to max_pairings -> materialize TrialJobSpec (n capped per pairing).
    """
    if not isinstance(plan, dict):
        raise TypeError("plan must be a screening_tier_plan dict")
    queue = plan.get("queue") or []
    corpus_slice = dict(plan.get("corpus_slice") or {})
    eval_tier = str(plan.get("eval_tier", "T0"))
    per_pairing_n = int(plan.get("per_pairing_n", 12))

    idx = _pairing_index(pool_gen_output)

    # 1. join + 2. dedup by pairing_id (first occurrence wins — stable).
    seen: set[str] = set()
    enriched: list[dict[str, Any]] = []
    n_dupes = 0
    for entry in queue:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("pairing_id"))
        if pid in seen:
            n_dupes += 1
            continue
        seen.add(pid)
        enriched.append(_enrich_entry(entry, idx.get(pid)))

    # 3. prune coresidency-unfit (only when there is POSITIVE evidence of unfit;
    #    unknown fit is kept, never pruned on missing data).
    n_pruned = 0
    if prune_unfit:
        kept: list[dict[str, Any]] = []
        for e in enriched:
            if e.get("coresidency_fits") is False:
                n_pruned += 1
                continue
            kept.append(e)
        enriched = kept

    # 4. priority ordering.
    if priority:
        enriched.sort(key=_priority_rank)

    # 5. truncate to max_pairings.
    n_truncated = 0
    if max_pairings and len(enriched) > max_pairings:
        n_truncated = len(enriched) - max_pairings
        enriched = enriched[:max_pairings]

    # 6. materialize job specs (cap n per pairing).
    effective_n = per_pairing_n
    if cap_per_pairing and cap_per_pairing > 0:
        effective_n = min(per_pairing_n, cap_per_pairing)

    corpus_id = str(corpus_slice.get("corpus_id", "unknown"))
    domain = str(corpus_slice.get("domain", "all"))
    content_sha = str(corpus_slice.get("content_sha256", ""))
    n_rows = int(corpus_slice.get("n_rows", 0) or 0)

    jobs: list[TrialJobSpec] = []
    for rank, e in enumerate(enriched):
        jobs.append(
            TrialJobSpec(
                pairing_id=str(e.get("pairing_id")),
                architect=e.get("architect"),
                reviewer=e.get("reviewer"),
                grader=e.get("grader"),
                anchor_arm=e.get("anchor_arm"),
                self_review=bool(e.get("self_review")),
                cross_family=bool(e.get("cross_family")),
                staged_involved=bool(e.get("staged_involved")),
                n=effective_n,
                eval_tier=eval_tier,
                corpus_id=corpus_id,
                domain=domain,
                corpus_content_sha256=content_sha,
                corpus_n_rows=n_rows,
                coresidency_fits=e.get("coresidency_fits"),
                priority_rank=rank,
            )
        )

    provenance = dict(plan.get("provenance") or {})
    provenance.update(
        {
            "pool_gen_schema_version": (pool_gen_output.get("provenance", {}) or {}).get(
                "schema_version"
            ),
            "pool_gen_registry_sha256": (pool_gen_output.get("provenance", {}) or {}).get(
                "registry_sha256"
            ),
            "resolver_config": {
                "cap_per_pairing": cap_per_pairing,
                "max_pairings": max_pairings,
                "prune_unfit": prune_unfit,
                "priority": priority,
            },
        }
    )

    notes = list(plan.get("notes") or [])
    notes.append(
        "resolved into concrete placement-queue job specs; NEVER /chat (RM-3)."
    )
    notes.append(
        "all FA/FR/CR produced by execution are pre-P-REV-1 observations, not "
        "decision-gating numbers (MEASUREMENT.md)."
    )

    return ResolvedScreeningQueue(
        jobs=jobs,
        corpus_slice=corpus_slice,
        eval_tier=eval_tier,
        per_pairing_n=effective_n,
        pairings_considered=len(queue),
        n_deduped=n_dupes,
        n_pruned_unfit=n_pruned,
        n_truncated=n_truncated,
        provenance=provenance,
        notes=notes,
    )


def build_and_resolve(
    pool_gen_output: dict[str, Any],
    *,
    corpus_manifest: dict[str, Any] | None = None,
    per_pairing_n: int = 12,
    eval_tier: str = "T0",
    max_pairings: int = 0,
    domain: str | None = None,
    cap_per_pairing: int = 0,
    prune_unfit: bool = True,
    priority: bool = True,
) -> tuple[ResolvedScreeningQueue | None, str | None]:
    """Convenience: build a screening plan from pool-gen output, then resolve it.

    Delegates plan generation to ``review_policy_trials.plan_screening_tier``
    (READ-ONLY) so the runner and ``actions.py`` share one planner. Returns
    ``(resolved_queue, None)`` or ``(None, error)``.
    """
    rpt = _load_review_policy_trials()
    plan, error = rpt.plan_screening_tier(
        pool_gen_output,
        corpus_manifest=corpus_manifest,
        per_pairing_n=per_pairing_n,
        eval_tier=eval_tier,
        max_pairings=max_pairings,
        domain=domain,
    )
    if error is not None or plan is None:
        return None, error or "screening plan generation returned no plan"
    resolved = resolve_screening_queue(
        plan.to_dict(),
        pool_gen_output,
        cap_per_pairing=cap_per_pairing,
        max_pairings=0,  # plan already applied max_pairings; don't double-truncate
        prune_unfit=prune_unfit,
        priority=priority,
    )
    return resolved, None


# ══════════════════════════════════════════════════════════════════════════════
# Scoring helpers (pure — corpus-row -> gate, FA/FR/CR)
# ══════════════════════════════════════════════════════════════════════════════


def is_judgeable_row(row: dict[str, Any]) -> bool:
    """True iff a corpus row can score a reviewer decision without more inference.

    Requires a persisted candidate answer AND a conclusive, gate-worthy gold label.
    Excludes the ``candidate_recovery_needed`` / observation-only rows the manifest
    flags as not-yet-judgeable (they need a later non-inference join or a re-run).
    """
    if not isinstance(row, dict):
        return False
    candidate = row.get("candidate")
    if candidate in (None, "", "None"):
        return False
    gold = str(row.get("gold_label", "")).strip().lower()
    if gold not in _CONCLUSIVE_GOLD_LABELS:
        return False
    conf = str(row.get("gold_confidence", "")).strip().lower()
    return conf in _GATE_WORTHY_CONFIDENCE


def gate_from_gold_label(gold_label: Any) -> str | None:
    """Map a corpus ``gold_label`` to the objective-gate outcome the reviewer is
    scored against: an acceptable candidate == gate PASS, a defective one == FAIL."""
    g = str(gold_label or "").strip().lower()
    if g == "accept":
        return "pass"
    if g == "reject":
        return "fail"
    return None


def consistency_rate(decisions: Iterable[dict[str, Any]]) -> float | None:
    """Reviewer/gold agreement rate on conclusive rows (CR).

    CR complements FA/FR: it is the fraction of gate-scored rows on which the
    reviewer's approve/reject matched the objective gate (approve&pass or
    reject&fail). ``None`` when no row carries a conclusive gate. Pre-P-REV-1
    observation only.
    """
    agree = total = 0
    for row in decisions:
        decision = str(row.get("decision", "")).strip().lower()
        gate = row.get("gate")
        gate_s = str(gate).strip().lower() if gate is not None else None
        if gate_s in {"pass", "true", "1"}:
            total += 1
            if decision in _APPROVE_DECISIONS:
                agree += 1
        elif gate_s in {"fail", "false", "0"}:
            total += 1
            if decision in _REJECT_DECISIONS:
                agree += 1
    return (agree / total) if total else None


def summarize_pairing(job: TrialJobSpec, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll decision rows for one pairing into an FA/FR/CR result row (pure)."""
    rpt = _load_review_policy_trials()
    axes = rpt.reviewer_calibration_from_decisions(decisions)
    n_conclusive = sum(1 for d in decisions if d.get("gate") is not None)
    return {
        "kind": "screening_tier_result",
        "runner_version": RUNNER_VERSION,
        "pairing_id": job.pairing_id,
        "architect": job.architect,
        "reviewer": job.reviewer,
        "grader": job.grader,
        "anchor_arm": job.anchor_arm,
        "self_review": job.self_review,
        "cross_family": job.cross_family,
        "eval_tier": job.eval_tier,
        "corpus_id": job.corpus_id,
        "domain": job.domain,
        "corpus_content_sha256": job.corpus_content_sha256,
        "transport": PLACEMENT_QUEUE_TRANSPORT,
        "n_requested": job.n,
        "n_scored": len(decisions),
        "n_conclusive": n_conclusive,
        "reviewer_fa_rate": axes.get("reviewer_fa_rate"),
        "reviewer_fr_rate": axes.get("reviewer_fr_rate"),
        "reviewer_fa_fr_ratio": axes.get("reviewer_fa_fr_ratio"),
        "review_decision_latency_ms": axes.get("review_decision_latency_ms"),
        "consistency_rate": consistency_rate(decisions),
        "observation_only": True,  # pre-P-REV-1 (MEASUREMENT.md)
    }


# ══════════════════════════════════════════════════════════════════════════════
# Corpus row selection (pure — deterministic, lazy JSONL read)
# ══════════════════════════════════════════════════════════════════════════════


def iter_judgeable_rows(
    rows_path: Path,
    *,
    domain: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Lazily yield judgeable corpus rows (optionally filtered to one domain)."""
    with rows_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if domain and domain != "all" and str(row.get("domain")) != domain:
                continue
            if is_judgeable_row(row):
                yield row


def select_rows_for_job(
    rows: list[dict[str, Any]],
    *,
    n: int,
    seed_key: str,
) -> list[dict[str, Any]]:
    """Deterministically pick ``n`` rows for a pairing (stable across runs).

    Sort by a per-(seed_key,row_id) hash and take the first ``n`` — same corpus +
    same pairing => same slice, so a re-run is reproducible without persisting the
    sample. Pure; no RNG global state touched.
    """
    if n <= 0 or not rows:
        return []

    def _h(row: dict[str, Any]) -> str:
        rid = str(row.get("row_id") or row.get("qid") or "")
        return hashlib.sha1(f"{seed_key}\x00{rid}".encode("utf-8")).hexdigest()

    return sorted(rows, key=_h)[:n]


# ══════════════════════════════════════════════════════════════════════════════
# Execution bridge (env-gated; models bsv_paired_runner; NEVER run in tests)
# ══════════════════════════════════════════════════════════════════════════════


def _default_tower() -> Any:  # pragma: no cover - inference path
    """Import + construct EvalTower (deferred, exactly like bsv_paired_runner)."""
    try:
        from scripts.autopilot.eval_tower import EvalTower
    except Exception:  # noqa: BLE001
        from eval_tower import EvalTower  # type: ignore[no-redef]
    return EvalTower()


def _default_reviewer_probe(
    job: TrialJobSpec,
    row: dict[str, Any],
    tower: Any,
) -> dict[str, Any]:  # pragma: no cover - inference path
    """Send ONE (task, candidate) reviewer judgement over the placement queue.

    This is the real inference seam. It reuses the SAME transport eval_tower uses
    internally — ``call_orchestrator_forced`` with ``request_priority=background``
    and ``workload_class=eval_batch`` (the placement-queue path), pinning
    ``force_role`` to the reviewer under test — so a screening judgement is never a
    foreground ``/chat`` request. Returns a decision row shaped for
    ``review_policy_trials.reviewer_calibration_from_decisions`` +
    ``consistency_rate``: ``{"decision","gate","latency_ms"}``.

    Never exercised by the unit tests (the whole execution bridge is env-gated and
    unreached under the zero-inference constraint).
    """
    import time as _time

    # Deferred import of the orchestrator client (same source eval_tower uses).
    _research = Path("/mnt/raid0/llm/epyc-inference-research")
    _bench = str(_research / "scripts" / "benchmark")
    if _bench not in sys.path:
        sys.path.insert(0, _bench)
    from seeding_orchestrator import call_orchestrator_forced  # type: ignore

    task = str(row.get("task") or "")
    candidate = str(row.get("candidate") or "")
    prompt = (
        "You are a strict reviewer. Decide whether the CANDIDATE answer to the "
        "TASK is acceptable. Reply with a single token: APPROVE or REJECT.\n\n"
        f"TASK:\n{task}\n\nCANDIDATE:\n{candidate}\n\nDECISION:"
    )
    start = _time.time()
    resp = call_orchestrator_forced(
        prompt=prompt,
        force_role=job.reviewer or "",
        force_mode="",
        url=getattr(tower, "url", "http://localhost:8000"),
        timeout=getattr(tower, "timeout", 300),
        request_priority=PLACEMENT_REQUEST_PRIORITY,   # placement queue, not /chat
        workload_class=PLACEMENT_WORKLOAD_CLASS,
    )
    latency_ms = (_time.time() - start) * 1000.0
    answer = str(resp.get("answer") or "").strip().lower()
    if "approve" in answer or "accept" in answer:
        decision = "approve"
    elif "reject" in answer:
        decision = "reject"
    else:
        decision = "abstain"  # unparseable -> excluded from FA/FR by the calibrator
    return {
        "decision": decision,
        "gate": gate_from_gold_label(row.get("gold_label")),
        "latency_ms": latency_ms,
        "row_id": row.get("row_id"),
    }


def execute_screening_queue(
    resolved: ResolvedScreeningQueue,
    *,
    output_path: Path | None = None,
    corpus_rows_path: Path | None = None,
    tower: Any | None = None,
    tower_factory: Callable[[], Any] | None = None,
    reviewer_probe: Callable[[TrialJobSpec, dict[str, Any], Any], dict[str, Any]] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:  # pragma: no cover - inference path
    """Drive the resolved queue over the placement queue and collect FA/FR/CR.

    Reached ONLY when ``AUTOPILOT_SCREENING_TIER_INFERENCE=1`` (via
    ``run_screening_tier``) or called directly by a future caller that owns the
    inference decision. Autopilot-stopped assumption (bsv pattern): the caller is
    responsible for the no-concurrent-inference window; this function never touches
    autopilot lifecycle/state. Emits one JSONL row per pairing to ``output_path``
    (append) and returns the same rows. Does NOT write the batch ledger.
    """
    tower = tower or (tower_factory or _default_tower)()
    probe = reviewer_probe or _default_reviewer_probe

    rows_path = corpus_rows_path
    if rows_path is None:
        rp = resolved.corpus_slice.get("rows_path") or (
            resolved.provenance.get("rows_path")
        )
        rows_path = Path(rp) if rp else None
    if rows_path is None or not Path(rows_path).exists():
        # Fall back to the manifest's canonical rows path if the slice omitted it.
        default_rows = DEFAULT_CORPUS_MANIFEST.parent / "rows.jsonl"
        rows_path = default_rows

    domain = str(resolved.corpus_slice.get("domain", "all"))
    pool = list(iter_judgeable_rows(Path(rows_path), domain=domain))

    results: list[dict[str, Any]] = []
    for job in resolved.jobs:
        sample = select_rows_for_job(pool, n=job.n, seed_key=f"{seed}:{job.pairing_id}")
        decisions = [probe(job, row, tower) for row in sample]
        result = summarize_pairing(job, decisions)
        results.append(result)
        if output_path is not None:
            _append_jsonl(Path(output_path), result)
    return results


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level orchestration (env-gated dry-run vs execute)
# ══════════════════════════════════════════════════════════════════════════════


def run_screening_tier(
    plan: dict[str, Any],
    pool_gen_output: dict[str, Any],
    *,
    corpus_manifest: dict[str, Any] | None = None,
    output_path: Path | None = None,
    corpus_rows_path: Path | None = None,
    cap_per_pairing: int = 0,
    max_pairings: int = 0,
    prune_unfit: bool = True,
    priority: bool = True,
    seed: int = 42,
    tower: Any | None = None,
    tower_factory: Callable[[], Any] | None = None,
    reviewer_probe: Callable[[TrialJobSpec, dict[str, Any], Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the queue, then dry-run OR execute depending on the inference flag.

    DEFAULT (``AUTOPILOT_SCREENING_TIER_INFERENCE`` unset/false): returns the
    resolved queue as a dry-run plan and runs NO inference — this is the entire
    surface the unit tests exercise. When the flag is set the resolved queue is
    driven over the placement queue via :func:`execute_screening_queue`.

    ``plan`` may be a ``_screening_tier_plan`` dict (already generated by
    ``actions.py`` / ``plan_screening_tier``); the resolver joins it to
    ``pool_gen_output``. (Building a plan from scratch is :func:`build_and_resolve`.)
    """
    resolved = resolve_screening_queue(
        plan,
        pool_gen_output,
        cap_per_pairing=cap_per_pairing,
        max_pairings=max_pairings,
        prune_unfit=prune_unfit,
        priority=priority,
    )

    if not _env_flag_enabled(SCREENING_TIER_INFERENCE_ENV):
        return {
            "mode": "dry_run",
            "runner_version": RUNNER_VERSION,
            "inference_ran": False,
            "reason": (
                f"{SCREENING_TIER_INFERENCE_ENV} not set; resolved queue returned as "
                "a dry-run plan (no inference, RM-3 placement-queue transport)."
            ),
            "n_jobs": len(resolved.jobs),
            "resolved_queue": resolved.to_dict(),
        }

    results = execute_screening_queue(
        resolved,
        output_path=output_path,
        corpus_rows_path=corpus_rows_path,
        tower=tower,
        tower_factory=tower_factory,
        reviewer_probe=reviewer_probe,
        seed=seed,
    )
    return {
        "mode": "execute",
        "runner_version": RUNNER_VERSION,
        "inference_ran": True,
        "n_jobs": len(resolved.jobs),
        "output_path": str(output_path) if output_path else None,
        "resolved_queue": resolved.to_dict(),
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI (__main__)
# ══════════════════════════════════════════════════════════════════════════════


def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Resolve + (optionally, env-gated) execute the H5 RM-3 reviewer "
            "screening tier over the placement queue. Default is a pure dry-run "
            "that prints the resolved queue and runs NO inference."
        )
    )
    p.add_argument(
        "--pool-gen",
        required=True,
        help="path to reviewer_pool_gen.py output JSON (source of pairings)",
    )
    p.add_argument(
        "--plan",
        default=None,
        help="optional path to a prebuilt _screening_tier_plan JSON; if omitted "
        "one is built from --pool-gen via review_policy_trials.plan_screening_tier",
    )
    p.add_argument(
        "--corpus-manifest",
        default=str(DEFAULT_CORPUS_MANIFEST),
        help="near-miss corpus manifest.json (metadata only when resolving)",
    )
    p.add_argument("--per-pairing-n", type=int, default=12)
    p.add_argument("--tier", default="T0", help="eval tier tag (T0 or T1)")
    p.add_argument("--domain", default=None, help="restrict to one corpus domain slice")
    p.add_argument("--max-pairings", type=int, default=0, help="cap number of pairings (0=all)")
    p.add_argument("--cap-per-pairing", type=int, default=0, help="cap N trials per pairing (0=plan n)")
    p.add_argument("--no-prune", action="store_true", help="keep coresidency-unfit pairings")
    p.add_argument("--no-priority", action="store_true", help="preserve plan order (no priority sort)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        default=None,
        help="JSONL path for per-pairing FA/FR/CR results (execute path only)",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="attempt execution (STILL env-gated by "
        f"{SCREENING_TIER_INFERENCE_ENV}=1; otherwise falls back to dry-run)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rpt = _load_review_policy_trials()

    pool_gen_output = rpt.load_pool_gen_output(Path(args.pool_gen))
    if not pool_gen_output.get("pairings"):
        print(
            json.dumps(
                {"error": f"no pairings loadable from {args.pool_gen!r}"},
                indent=2,
            )
        )
        return 2

    manifest = rpt.load_corpus_manifest(Path(args.corpus_manifest))

    if args.plan:
        plan = _load_json_file(Path(args.plan))
    else:
        plan_obj, error = rpt.plan_screening_tier(
            pool_gen_output,
            corpus_manifest=manifest,
            per_pairing_n=args.per_pairing_n,
            eval_tier=args.tier,
            max_pairings=args.max_pairings,
            domain=args.domain,
        )
        if error is not None or plan_obj is None:
            print(json.dumps({"error": error or "no plan"}, indent=2))
            return 2
        plan = plan_obj.to_dict()

    if not args.run:
        # Pure resolution — no inference, whatever the env flag says.
        resolved = resolve_screening_queue(
            plan,
            pool_gen_output,
            cap_per_pairing=args.cap_per_pairing,
            # plan already applied max_pairings when built here; still honor an
            # explicit CLI cap when a prebuilt plan was passed in.
            max_pairings=args.max_pairings if args.plan else 0,
            prune_unfit=not args.no_prune,
            priority=not args.no_priority,
        )
        print(json.dumps(resolved.to_dict(), indent=2, sort_keys=True, default=str))
        return 0

    result = run_screening_tier(
        plan,
        pool_gen_output,
        corpus_manifest=manifest,
        output_path=Path(args.output) if args.output else None,
        cap_per_pairing=args.cap_per_pairing,
        max_pairings=args.max_pairings if args.plan else 0,
        prune_unfit=not args.no_prune,
        priority=not args.no_priority,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
