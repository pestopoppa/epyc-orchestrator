"""Reviewer control-plane autopilot helpers (H8 AP-1/4/5/6/7).

Pure, dependency-light helpers behind the new autopilot actions and the codex
dogfooding path. Everything here is inference-free and unit-testable WITHOUT any
model/server call — the live eval-tower execution is inference-gated and lives in
the action handlers (``actions.py``), which call the planners defined here to
enumerate a trial plan first.

Sections:
  * AP-4  reviewer-calibration Pareto axes (extract + compute + era-row fixture)
  * AP-6  codex-critic -> typed ReviewDecision dogfooding (map + validate + count)
  * AP-5  review_policy_trial + screening_tier_driver PLAN generators
  * AP-7  journal event-type constants/schema + checkpoint-compat state defaults
  * AP-1  strategy-store SEED fixture (NOT written to the live store)

The autopilot may be running or intentionally stopped; nothing in this module
writes to autopilot_state.json, the journals, runtime_flags.json, or the strategy
store. Seeding + era registration are emitted as fixtures for the operator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ORCH_ROOT = Path(__file__).resolve().parents[2]


# ── Defensive cross-module imports ────────────────────────────────────────────
# These modules are importable both as bare names (autopilot dir on sys.path, the
# way autopilot.py loads) and as ``scripts.autopilot.*`` / ``src.*`` (package /
# pytest mode). Try both; degrade gracefully so a missing optional dep never
# breaks the pure helpers.


def _load_review_grammar():
    try:
        from src.proactive_delegation import review_grammar as rg
    except Exception:  # noqa: BLE001
        try:
            import review_grammar as rg  # type: ignore
        except Exception:  # noqa: BLE001
            return None
    return rg


def _load_knob_specs() -> dict[str, Any]:
    try:
        from scripts.autopilot.config_applicator import REVIEW_PLANE_KNOB_SPECS
    except Exception:  # noqa: BLE001
        try:
            from config_applicator import REVIEW_PLANE_KNOB_SPECS  # type: ignore
        except Exception:  # noqa: BLE001
            return {}
    return dict(REVIEW_PLANE_KNOB_SPECS)


# ══════════════════════════════════════════════════════════════════════════════
# AP-4 — reviewer-calibration Pareto axes
# ══════════════════════════════════════════════════════════════════════════════

# Optional quality axes (see EvalResult in safety_gate.py). Order is stable so a
# fixed-position era row and any downstream vector stay aligned.
REVIEW_PARETO_AXES: tuple[str, ...] = (
    "reviewer_fa_rate",
    "reviewer_fr_rate",
    "reviewer_fa_fr_ratio",
    "review_decision_latency_ms",
)

# Optimization direction per axis (lower_is_better): a reviewer wants fewer false
# accepts / false rejects and lower latency. The ratio is diagnostic (balance of
# the two error modes), reported not raw-optimized.
REVIEW_AXIS_LOWER_IS_BETTER: dict[str, bool] = {
    "reviewer_fa_rate": True,
    "reviewer_fr_rate": True,
    "reviewer_fa_fr_ratio": True,
    "review_decision_latency_ms": True,
}


def reviewer_quality_axes(result: Any) -> dict[str, float]:
    """Extract the PRESENT reviewer-calibration axes from an EvalResult.

    Absent (NaN / missing) axes are dropped, so a result with no reviewer in the
    loop returns ``{}`` and no caller ever sees a placeholder value. This is the
    additive read side of AP-4: existing Pareto/quality bookkeeping is untouched
    unless a reviewer actually populated an axis.
    """
    out: dict[str, float] = {}
    for axis in REVIEW_PARETO_AXES:
        val = getattr(result, axis, None)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isnan(fval):
            out[axis] = fval
    return out


def reviewer_calibration_from_decisions(
    decisions: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute FA/FR rates + ratio + mean latency from decision/outcome rows.

    Each row is ``{"decision": <approve|reject|...>, "gate": <pass|fail|None>,
    "latency_ms": <float?>}`` — the reviewer verdict paired with the conclusive
    objective-gate outcome (verifier-precedence, RD-3). Rows with no conclusive
    gate ("None"/inconclusive) are excluded from FA/FR (they cannot be scored)
    but still count toward latency. Returns only the axes it can compute.

      * false-accept (FA): reviewer approved but the objective gate FAILED.
      * false-reject (FR): reviewer rejected but the objective gate PASSED.
    """
    approve = {"approve"}
    reject = {"reject", "reject_to_empty"}
    fa = fr = n_gate_fail = n_gate_pass = 0
    latencies: list[float] = []
    for row in decisions:
        decision = str(row.get("decision", "")).lower()
        gate = row.get("gate")
        gate_s = str(gate).lower() if gate is not None else None
        lat = row.get("latency_ms")
        if isinstance(lat, (int, float)) and not math.isnan(float(lat)):
            latencies.append(float(lat))
        if gate_s in {"fail", "false", "0"}:
            n_gate_fail += 1
            if decision in approve:
                fa += 1
        elif gate_s in {"pass", "true", "1"}:
            n_gate_pass += 1
            if decision in reject:
                fr += 1
    out: dict[str, float] = {}
    if n_gate_fail:
        out["reviewer_fa_rate"] = fa / n_gate_fail
    if n_gate_pass:
        out["reviewer_fr_rate"] = fr / n_gate_pass
    if "reviewer_fa_rate" in out and "reviewer_fr_rate" in out:
        fr_rate = out["reviewer_fr_rate"]
        out["reviewer_fa_fr_ratio"] = (
            out["reviewer_fa_rate"] / fr_rate if fr_rate > 0 else float("inf")
        )
    if latencies:
        out["review_decision_latency_ms"] = sum(latencies) / len(latencies)
    return out


def instrument_era_row(
    *,
    era_id: str = "reviewer-calibration-axes-v1",
    effective_date: str = "2026-07-17",
    protocol_id: str = "P-REV-1",
) -> dict[str, Any]:
    """Era-registration row CONTENT for the new AP-4 axes (coordination artifact).

    Emitted as a fixture for the operator to append to
    ``epyc-orchestrator/orchestration/instrument_eras.yaml`` — this module does
    NOT write that file (the measurement trust boundary is human-amendment-only,
    MEASUREMENT.md). Until P-REV-1 certifies them, any value on these axes is an
    observation, never a decision-gating number.
    """
    return {
        "era_id": era_id,
        "effective_date": effective_date,
        "protocol_id": protocol_id,
        "axes": list(REVIEW_PARETO_AXES),
        "lower_is_better": dict(REVIEW_AXIS_LOWER_IS_BETTER),
        "status": "observation-only until P-REV-1 certified",
        "source_handoff": "reviewer-calibration-accounting.md (H4)",
        "note": (
            "Additive optional axes on EvalResult; NaN when no reviewer in loop. "
            "Registering here does NOT change objectives 4-tuple or SafetyGate."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AP-6 — codex-critic -> typed ReviewDecision dogfooding
# ══════════════════════════════════════════════════════════════════════════════

# Native autopilot_critique decision -> ReviewDecision enum (schema-valid).
_CRITIQUE_DECISION_MAP: dict[str, str] = {
    "approve": "approve",
    "revise": "request_changes",
    "reject": "reject",
}


@dataclass
class CritiqueEmissionStats:
    """In-memory counters for the passive dogfooding emission (AP-6)."""

    emitted: int = 0
    parse_failures: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)

    def record_success(self) -> None:
        self.emitted += 1

    def record_failure(self, reason: str) -> None:
        self.parse_failures += 1
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "emitted": self.emitted,
            "parse_failures": self.parse_failures,
            "failure_reasons": dict(self.failure_reasons),
        }


def _extract_critique_block(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the ```json:autopilot_critique payload.

    Self-contained (does NOT import planner_coordinator — that would be a circular
    import, planner_coordinator -> planner_providers -> here). Falls back to the
    first balanced JSON object in the text.
    """
    marker = "```json:autopilot_critique"
    idx = text.find(marker)
    body = text[idx + len(marker):] if idx != -1 else text
    rg = _load_review_grammar()
    if rg is not None:
        candidate = rg._extract_json_object(body)
    else:
        candidate = _fallback_extract_json_object(body)
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _fallback_extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _critique_to_review_decision_dict(critique: dict[str, Any]) -> dict[str, Any] | None:
    """Map a native autopilot_critique object to a review_decision dict.

    The codex critic is an ADVISORY planner reviewer (not an objective verifier),
    so ``blocking.tripwire`` is always False — a plan critique never hard-stops on
    a violated invariant. Issues become advisory non_blocking_issues.
    """
    decision = str(critique.get("decision", "")).strip().lower()
    mapped = _CRITIQUE_DECISION_MAP.get(decision)
    if mapped is None:
        return None
    try:
        confidence = float(critique.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    issues = critique.get("issues") or []
    non_blocking = [
        {"summary": str(issue).strip()[:2000]}
        for issue in issues
        if isinstance(issue, (str, int, float)) and str(issue).strip()
    ]
    out: dict[str, Any] = {
        "decision": mapped,
        "confidence": confidence,
        "blocking": {"tripwire": False},
        "provenance": {"role": "verifier", "source": "codex_critic"},
    }
    advisory: dict[str, Any] = {}
    if non_blocking:
        advisory["non_blocking_issues"] = non_blocking
    if advisory:
        out["advisory"] = advisory
    return out


def derive_review_decision_from_critique(
    text: str,
) -> tuple[dict[str, Any] | None, Any | None]:
    """Derive a schema-valid ReviewDecision from codex-critic output text (AP-6).

    Returns ``(review_decision_obj, None)`` on success or ``(None, failure)`` on
    failure, where ``failure`` is a review_grammar.ParseFailure (or a lightweight
    stand-in when review_grammar is unavailable) — the accounting hook. Strategy:

      1. If the text already carries a schema-valid review_decision, use it
         (round-trips through review_grammar.parse_review_decision).
      2. Otherwise map the native autopilot_critique block -> a review_decision
         dict and re-validate it through the SAME parser (dogfoods the schema).

    Never raises: any unexpected error becomes a counted parse failure so the
    caller falls back to current behavior.
    """
    rg = _load_review_grammar()

    class _MiniFailure:
        def __init__(self, reason: str, detail: str = ""):
            self.reason = reason
            self.detail = detail

        def to_dict(self) -> dict[str, Any]:
            return {"reason": self.reason, "detail": self.detail}

    def _fail(reason: str, detail: str = "") -> tuple[None, Any]:
        if rg is not None:
            try:
                return None, rg.ParseFailure(
                    rg.ParseFailureReason(reason)
                    if reason in {r.value for r in rg.ParseFailureReason}
                    else rg.ParseFailureReason.NO_JSON,
                    detail,
                )
            except Exception:  # noqa: BLE001
                pass
        return None, _MiniFailure(reason, detail)

    try:
        # Path 1: native review_decision already present.
        if rg is not None:
            obj, failure = rg.parse_review_decision(text)
            if obj is not None:
                return obj, None

        # Path 2: map the critique block, then re-validate through the parser.
        critique = _extract_critique_block(text)
        if critique is None:
            return _fail("no_json", "no autopilot_critique / JSON object in text")
        mapped = _critique_to_review_decision_dict(critique)
        if mapped is None:
            return _fail(
                "schema_invalid",
                f"unmappable critique decision {critique.get('decision')!r}",
            )
        if rg is None:
            # Cannot validate without the grammar module; treat as unavailable.
            return _fail("validator_unavailable", "review_grammar not importable")
        obj, failure = rg.parse_review_decision(json.dumps(mapped))
        if obj is not None:
            return obj, None
        return None, failure
    except Exception as exc:  # noqa: BLE001 — passive emission must never raise
        return _fail("json_decode_error", f"{type(exc).__name__}: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# AP-5 — trial-plan generators (inference-free; execution is gated in actions.py)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CORPUS_MANIFEST = Path(
    "/mnt/raid0/llm/datasets/nearmiss-corpus-v1/manifest.json"
)


def load_corpus_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the near-miss corpus manifest; ``{}`` (graceful) when absent."""
    resolved = path or DEFAULT_CORPUS_MANIFEST
    try:
        if not resolved.exists():
            return {}
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def corpus_slice_summary(
    manifest: dict[str, Any],
    *,
    domain: str | None = None,
) -> dict[str, Any]:
    """Summarize a corpus slice (whole corpus, or one ``per_domain`` slice)."""
    counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
    per_domain = counts.get("per_domain", {}) if isinstance(counts, dict) else {}
    if domain is not None:
        n = int(per_domain.get(domain, 0))
    else:
        n = int(manifest.get("total_rows", 0) or 0)
    return {
        "corpus_id": manifest.get("corpus_id", "unknown"),
        "domain": domain or "all",
        "n_rows": n,
        "content_sha256": manifest.get("content_sha256", ""),
        "schema_version": manifest.get("schema_version", ""),
        "gate_worthy_multi_oracle": int(manifest.get("gate_worthy_multi_oracle", 0) or 0),
    }


def _knob_grid(knob_names: list[str], points: int) -> list[dict[str, Any]]:
    """Enumerate a bounded param grid for the requested knobs.

    ``points`` evenly-spaced values per knob within [lo, hi] (bool -> {False,
    True}). Purely combinatorial and inference-free — this is the trial plan an
    inference run WOULD execute, not the execution itself.
    """
    specs = _load_knob_specs()
    axes: list[list[tuple[str, Any]]] = []
    for name in knob_names:
        spec = specs.get(name)
        if spec is None:
            continue
        if spec.kind == "bool":
            values: list[Any] = [False, True]
        else:
            lo = spec.lo if spec.lo is not None else 0.0
            hi = spec.hi if spec.hi is not None else lo + 1.0
            k = max(2, points)
            step = (hi - lo) / (k - 1)
            raw = [lo + step * i for i in range(k)]
            values = [int(round(v)) for v in raw] if spec.kind == "int" else [
                round(v, 6) for v in raw
            ]
            # de-dup int collapses (small ranges quantize to the same value)
            seen: list[Any] = []
            for v in values:
                if v not in seen:
                    seen.append(v)
            values = seen
        axes.append([(spec.apply_key, v) for v in values])

    grid: list[dict[str, Any]] = [{}]
    for axis in axes:
        grid = [{**combo, key: val} for combo in grid for (key, val) in axis]
    return grid


@dataclass
class ReviewPolicyTrialPlan:
    """Dry-run plan for a class-1 knob sweep over a corpus slice (AP-5)."""

    surface: str
    knobs: list[str]
    corpus_slice: dict[str, Any]
    grid: list[dict[str, Any]]
    eval_tier: int
    n_trials: int
    inference_required: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "review_policy_trial_plan",
            "surface": self.surface,
            "knobs": list(self.knobs),
            "corpus_slice": dict(self.corpus_slice),
            "grid": list(self.grid),
            "eval_tier": self.eval_tier,
            "n_trials": self.n_trials,
            "inference_required": self.inference_required,
            "notes": list(self.notes),
        }


def plan_review_policy_trial(
    action: dict[str, Any],
    *,
    corpus_manifest: dict[str, Any] | None = None,
) -> tuple[ReviewPolicyTrialPlan | None, str | None]:
    """Enumerate a review_policy_trial plan WITHOUT running any eval/backend.

    Returns ``(plan, None)`` or ``(None, error)``. Validates that requested knobs
    are registered class-1 knobs and any explicit values are in-bounds.
    """
    specs = _load_knob_specs()
    requested = action.get("knobs") or list(specs.keys())
    knobs = [k for k in requested if k in specs]
    unknown = [k for k in requested if k not in specs]
    if not knobs:
        return None, (
            "review_policy_trial requires at least one registered class-1 knob; "
            f"got unknown knobs {unknown}"
        )

    # Validate any explicit param overrides against bounds.
    overrides = action.get("params") or {}
    for name, value in overrides.items():
        spec = specs.get(name)
        if spec is None:
            return None, f"review_policy_trial: unknown knob {name!r}"
        err = spec.validate(value)
        if err:
            return None, f"review_policy_trial: {err}"

    points = int(action.get("grid_points", 3))
    grid = (
        [{specs[n].apply_key: overrides[n] for n in overrides if n in specs}]
        if overrides
        else _knob_grid(knobs, points)
    )

    manifest = corpus_manifest if corpus_manifest is not None else load_corpus_manifest()
    corpus_slice = corpus_slice_summary(manifest, domain=action.get("domain"))
    eval_tier = int(action.get("tier", 0))
    notes = [
        "class-1 governance knob sweep (RD-11); observation-only until P-REV-1.",
        "execution is inference-gated: dry-run enumerates the plan only.",
    ]
    if unknown:
        notes.append(f"ignored unregistered knobs: {unknown}")
    plan = ReviewPolicyTrialPlan(
        surface="review_plane",
        knobs=knobs,
        corpus_slice=corpus_slice,
        grid=grid,
        eval_tier=eval_tier,
        n_trials=len(grid),
        notes=notes,
    )
    return plan, None


def load_pool_gen_output(path: Path) -> dict[str, Any]:
    """Load reviewer_pool_gen.py output (deterministic JSON); ``{}`` on failure."""
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@dataclass
class ScreeningTierPlan:
    """Screening trial queue for RM-3 built from pool-gen pairings (AP-5)."""

    corpus_slice: dict[str, Any]
    eval_tier: str
    per_pairing_n: int
    pairings_considered: int
    queue: list[dict[str, Any]]
    provenance: dict[str, Any]
    inference_required: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "screening_tier_plan",
            "corpus_slice": dict(self.corpus_slice),
            "eval_tier": self.eval_tier,
            "per_pairing_n": self.per_pairing_n,
            "pairings_considered": self.pairings_considered,
            "n_queued": len(self.queue),
            "queue": list(self.queue),
            "provenance": dict(self.provenance),
            "inference_required": self.inference_required,
            "notes": list(self.notes),
        }


def plan_screening_tier(
    pool_gen_output: dict[str, Any],
    *,
    corpus_manifest: dict[str, Any] | None = None,
    per_pairing_n: int = 12,
    eval_tier: str = "T0",
    max_pairings: int = 0,
    domain: str | None = None,
) -> tuple[ScreeningTierPlan | None, str | None]:
    """Build the RM-3 screening queue from pool-gen output + corpus manifest.

    Plan-generation only: each queue entry is one (pairing, corpus_slice, n)
    screening trial to be dispatched later via the eval-tower T0/T1 path, on the
    placement queue (NOT /chat), inside operator-approved no-concurrent-inference
    windows. This function runs NO inference.
    """
    pairings = pool_gen_output.get("pairings") if isinstance(pool_gen_output, dict) else None
    if not pairings:
        return None, "screening_tier_driver: pool-gen output has no pairings"

    manifest = corpus_manifest if corpus_manifest is not None else load_corpus_manifest()
    corpus_slice = corpus_slice_summary(manifest, domain=domain)
    if corpus_slice["n_rows"] <= 0:
        return None, (
            "screening_tier_driver: corpus slice is empty "
            f"(domain={domain!r}); refusing to queue trials with no items"
        )

    ordered = list(pairings)
    if max_pairings and len(ordered) > max_pairings:
        ordered = ordered[:max_pairings]

    queue: list[dict[str, Any]] = []
    for pairing in ordered:
        if not isinstance(pairing, dict):
            continue
        queue.append(
            {
                "pairing_id": pairing.get("pairing_id"),
                "architect": pairing.get("architect"),
                "reviewer": pairing.get("reviewer"),
                "grader": pairing.get("grader"),
                "anchor_arm": pairing.get("anchor_arm"),
                "self_review": bool(pairing.get("self_review")),
                "cross_family": bool(pairing.get("cross_family_preferred")),
                "n": per_pairing_n,
                "eval_tier": eval_tier,
                "corpus_id": corpus_slice["corpus_id"],
                "domain": corpus_slice["domain"],
                "dispatch": "placement_queue",  # NOT /chat (RM-3 discipline)
            }
        )

    provenance = {
        "pool_gen_schema_version": (pool_gen_output.get("provenance", {}) or {}).get(
            "schema_version"
        ),
        "registry_sha256": (pool_gen_output.get("provenance", {}) or {}).get(
            "registry_sha256"
        ),
        "prune_config_sha256": (pool_gen_output.get("provenance", {}) or {}).get(
            "prune_config_sha256"
        ),
        "corpus_content_sha256": corpus_slice.get("content_sha256"),
    }
    notes = [
        "RM-3 screening tier: small-n per-pairing FA/FR/CR with wide CIs.",
        "respects no-concurrent-inference + placement-queue-not-/chat discipline.",
        "confirmation tier (RM-4, N>=100) runs only for Pareto-promising pairs.",
    ]
    plan = ScreeningTierPlan(
        corpus_slice=corpus_slice,
        eval_tier=eval_tier,
        per_pairing_n=per_pairing_n,
        pairings_considered=len(ordered),
        queue=queue,
        provenance=provenance,
        notes=notes,
    )
    return plan, None


# ══════════════════════════════════════════════════════════════════════════════
# AP-7 — journal event-type constants + checkpoint-compat state defaults
# ══════════════════════════════════════════════════════════════════════════════
#
# Additive event types for the append-only experiment journal ledger. These are
# the SCHEMA/CONSTANTS (the documented shape a future journal-writer edit adopts);
# this module does NOT write live journals. They mirror the existing
# experiment_journal.py ledger-event convention (a ``type`` discriminator + a flat
# payload appended via append_ledger_event), so they fold in without a schema bump
# and old readers that filter on known ``type`` values simply ignore them.

REVIEW_DECISION_EVENT_TYPE = "review_decision"
REVIEW_POLICY_TRIAL_EVENT_TYPE = "review_policy_trial"

REVIEW_JOURNAL_EVENT_TYPES: tuple[str, ...] = (
    REVIEW_DECISION_EVENT_TYPE,
    REVIEW_POLICY_TRIAL_EVENT_TYPE,
)


@dataclass
class ReviewDecisionEvent:
    """Append-only ledger event for one shadow/dogfooded review decision (AP-7)."""

    decision: str
    confidence: float
    tripwire: bool
    source: str  # e.g. "codex_critic"
    subtask_id: str = ""
    gate_outcome: str = ""  # pass | fail | inconclusive | ""
    latency_ms: float = math.nan
    trial_id: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    type: str = REVIEW_DECISION_EVENT_TYPE

    def to_event(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "decision": self.decision,
            "confidence": self.confidence,
            "tripwire": self.tripwire,
            "source": self.source,
            "subtask_id": self.subtask_id,
            "gate_outcome": self.gate_outcome,
            "trial_id": self.trial_id,
            "provenance": dict(self.provenance),
        }
        if not math.isnan(self.latency_ms):
            payload["latency_ms"] = self.latency_ms
        return payload


@dataclass
class ReviewPolicyTrialEvent:
    """Append-only ledger event recording a class-1 knob-sweep trial (AP-7)."""

    surface: str
    knobs: list[str]
    grid_size: int
    corpus_slice: dict[str, Any]
    eval_tier: int
    inference_required: bool
    trial_id: int | None = None
    type: str = REVIEW_POLICY_TRIAL_EVENT_TYPE

    def to_event(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "surface": self.surface,
            "knobs": list(self.knobs),
            "grid_size": self.grid_size,
            "corpus_slice": dict(self.corpus_slice),
            "eval_tier": self.eval_tier,
            "inference_required": self.inference_required,
            "trial_id": self.trial_id,
        }


# Checkpoint compatibility: new autopilot_state.json keys, optional-with-defaults
# so an OLD checkpoint (that predates the review plane) still loads. A future
# state-loader edit calls ensure_review_state_defaults() after json.load; here we
# only define the additive keys + a pure merge helper (no live-state write).
REVIEW_STATE_DEFAULTS: dict[str, Any] = {
    "review_plane_knobs": {},          # last-applied class-1 knob values
    "review_decision_shadow_count": 0,  # cumulative shadow/dogfooded decisions
    "review_policy_trial_count": 0,     # cumulative review_policy_trial dispatches
    "last_screening_plan": None,        # most recent screening queue summary
}


def ensure_review_state_defaults(state: dict[str, Any]) -> dict[str, Any]:
    """Inject review-plane state keys if absent (checkpoint back-compat, AP-7).

    Mutates + returns ``state``. Existing keys are never overwritten, so a
    checkpoint that already carries review-plane state round-trips unchanged and a
    pre-review-plane checkpoint gains the defaults. Safe on a minimal ``{}``.
    """
    for key, default in REVIEW_STATE_DEFAULTS.items():
        state.setdefault(key, default.copy() if isinstance(default, dict) else default)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# AP-1 — strategy-store SEED fixture (NOT written to the live store)
# ══════════════════════════════════════════════════════════════════════════════


def review_plane_seed_strategies() -> list[dict[str, Any]]:
    """Strategy-store seeding entries for the class-1 knobs (fixture only).

    Follows operator_seed_strategies.yaml conventions. NOT written to the live
    strategy store — flag-gated seeding is an operator action (per seeding
    discipline). The operator can append these to operator_seed_strategies.yaml
    (or seed via the flag-gated path) to make NumericSwarm aware of the knobs.
    """
    entries: list[dict[str, Any]] = []
    for name, spec in _load_knob_specs().items():
        entries.append(
            {
                "slug": f"review-plane-{name.replace('_', '-')}",
                "tranche": "green",
                "species": "numeric_swarm",
                "entry_type": "pattern",
                "title": f"Tune class-1 review knob {name}",
                "description": f"reviewer control-plane governance knob {name}",
                "insight": spec.doc
                or f"Sweep {name} within its declared bounds via review_policy_trial.",
                "evidence_trial_ids": [],
                "source_handoff": "reviewer-decision-plane",
                "seeded_reason": "Expose the class-1 review knob to NumericSwarm",
                "confidence": "medium",
                "bind_status": "future",
                "bind_identifiers": [spec.apply_key],
            }
        )
    return entries
