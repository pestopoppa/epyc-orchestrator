#!/usr/bin/env python3
"""ROUTE-A1 shape-keyed Step-2 live-smoke planner + placement-queue driver.

Standalone smoke driver for **Step 2** of the shape-keyed-contention-gating work
(`ORCHESTRATOR_SHAPE_AWARE_CONTENTION=1`, still default-off). It plans — and, only
under an explicit ``--execute`` + env flag, drives — the two live observations that
gate Step-2 promotion, WITHOUT modifying any routing/serving-path module (the
dispatcher, `contention_gate.py`, `contention.py`, `placement.py` are all imported
READ-ONLY or not at all; the shape-keyed admission core stays FROZEN):

  1. **admit-overlap probe** — the disjoint-admit / overlap-queue smoke bracket.
     Over the canonical CPU-region model (`src/runtime/instance_topology.py`,
     READ-ONLY), enumerate candidate placements against a held "anchor" placement
     and classify each by *region-set overlap* (never by a shape's ``full`` label —
     see invariant #1 in the handoff): a **disjoint** candidate is EXPECTED to
     ADMIT, an **overlapping** candidate is EXPECTED to QUEUE (fail closed). Live
     Step-2 routing would report the gate's actual decision; the pure aggregator
     scores observed-vs-expected. This is the "disjoint quarters admit while
     q-overlaps queue" bracket the handoff owes before flag-on.

  2. **vision_escalation re-bench** — the higher-sample (>=8) within-role re-bench
     that the within-role-placement handoff still owes ("ratify clean allow;
     current 5/8 pairs cv>5%"). Enumerate the role's **disjoint** instance-shape
     pairs, carry the J5 prior ratio/cv, and (pure) aggregate co-run samples into
     mean/CV/verdict. Results are **model/quant-indexed, NEVER role-indexed** (a
     bench is keyed by the model+quant under test — `Qwen3-VL-30B-A3B-Instruct` /
     `Q4_K_M` — with the role kept only as provenance).

Two responsibilities, cleanly split (mirrors ``screening_tier_runner.py``):

  * **Plan construction + result aggregation** (pure, inference-free — this is the
    entire surface the fixture test exercises): build the probe/re-bench specs from
    a topology region map, and roll *synthetic* routing outcomes into a smoke
    verdict. Every spec pins **placement-queue transport**
    (``request_priority=background`` + ``workload_class=eval_batch``) and NEVER a
    foreground ``/chat`` target — the routing discipline these handoffs require.

  * **Execution bridge** (``--execute`` AND ``AUTOPILOT_SHAPEKEYED_STEP2_SMOKE=1``,
    BOTH default OFF): with either gate closed the resolved plan is returned as a
    dry-run and NO inference happens. With both open, drive the probe/re-bench over
    the **placement queue** (eval_batch), collect outcomes, and hand them to the
    pure aggregators. The bridge is intentionally NEVER reached by the tests and
    never touches autopilot lifecycle/state or the serving path.

All numbers produced here are pre-promotion OBSERVATIONS (MEASUREMENT.md): they
feed the operator's Step-2 flag-on decision, they never gate it on their own.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(SCRIPT_DIR), str(ORCH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Env flag gating the (never-in-tests) execution bridge. BOTH this flag AND the
# CLI --execute must be set for any inference to happen.
SHAPEKEYED_STEP2_INFERENCE_ENV = "AUTOPILOT_SHAPEKEYED_STEP2_SMOKE"

RUNNER_VERSION = "shapekeyed-step2-smoke-v1"

# Placement-queue transport constants — a Step-2 smoke request rides the SAME
# background/eval_batch placement path a normal autopilot eval fan-out uses; it is
# never a foreground /chat request. (The within-role handoff's J1 finding F3: the
# placement-SM queue is only reachable via the eval-batch path, not rate-limited
# external /chat.)
PLACEMENT_QUEUE_TRANSPORT = "placement_queue"
PLACEMENT_REQUEST_PRIORITY = "background"
PLACEMENT_WORKLOAD_CLASS = "eval_batch"

# Shape-keyed Step-2 gate contract: a placement whose region set is DISJOINT from
# the held set is EXPECTED to admit; an OVERLAPPING one is EXPECTED to queue.
DECISION_ADMIT = "admit"
DECISION_QUEUE = "queue"

# Contention-matrix default floor (orchestration/contention_matrix.yaml
# default_floor: 0.85). ratio >= 1.0 => allow; floor <= ratio < 1.0 => borderline
# (net-positive but sub-linear); ratio < floor => block.
DEFAULT_MATRIX_FLOOR = 0.85
# A "clean" (ratifiable) allow requires CV at/under the same 5% gate J5 used.
DEFAULT_CV_THRESHOLD = 0.05
# The re-bench the within-role handoff owes: >=8 samples to beat 5/8 pairs cv>5%.
DEFAULT_TARGET_SAMPLES = 8

# vision_escalation model/quant — registry-sourced
# (orchestration/model_registry.yaml roles.vision_escalation.model, ~L1770). Bench
# results are indexed by THIS (model, quant), never by the role name.
VISION_DEFAULT_MODEL = "Qwen3-VL-30B-A3B-Instruct"
VISION_DEFAULT_QUANT = "Q4_K_M"
VISION_DEFAULT_ROLE = "vision_escalation"

# Default probe anchor (the classic handoff example): ingest_long_context's full
# node0-half instance holds {q0,q1}; disjoint {q2,q3} candidates must admit while
# {q0,q1}/full candidates must queue.
DEFAULT_ANCHOR_ROLE = "ingest_long_context"
DEFAULT_ANCHOR_IDX = 0
DEFAULT_PROBE_ROLES = (
    "frontdoor",
    "ingest_long_context",
    "vision_escalation",
    "worker_general",
)

# J5 within-role vision priors (within-role-placement-state-machine.md, J5 -t48
# re-bench table). Keyed by the canonical region-set-pair label so a "full" shape
# is expressed as its region set ({q2,q3} -> "q2q3"), never the human label.
_VISION_J5_PRIORS: dict[str, dict[str, float]] = {
    "q0+q1": {"ratio": 1.154, "cv": 0.067},
    "q0+q2": {"ratio": 1.188, "cv": 0.088},
    "q0+q3": {"ratio": 1.266, "cv": 0.004},
    "q1+q2": {"ratio": 1.233, "cv": 0.007},
    "q1+q3": {"ratio": 0.963, "cv": 0.420},
    "q2+q3": {"ratio": 1.140, "cv": 0.076},
    "q0+q2q3": {"ratio": 0.580, "cv": 0.024},  # full+q0 (full = node1-half {q2,q3})
    "q1+q2q3": {"ratio": 0.619, "cv": 0.124},  # full+q1
}


def _env_flag_enabled(name: str) -> bool:
    """True iff env var ``name`` is a truthy flag."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ── canonical region helpers (region sets are the ONLY overlap authority) ─────


def region_label(regions: Iterable[str]) -> str:
    """Stable label for a region set: sorted quarters concatenated.

    e.g. {"q0"} -> "q0"; {"q2","q3"} (a node-half "full") -> "q2q3". NEVER emits a
    human shape label like ``full`` — invariant #1: overlap is a region-set fact.
    """
    return "".join(sorted(regions))


def pair_label(a: Iterable[str], b: Iterable[str]) -> str:
    """Order-independent label for a region-set pair, e.g. "q0+q2q3"."""
    la, lb = region_label(a), region_label(b)
    return "+".join(sorted((la, lb)))


def regions_disjoint(a: Iterable[str], b: Iterable[str]) -> bool:
    """True iff two region sets share no atomic quarter (can co-reside)."""
    return not (set(a) & set(b))


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses (plan specs + resolved plan)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Placement:
    """A concrete (role, instance_idx) placement and the region set it occupies."""

    role: str
    instance_idx: int
    regions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "instance_idx": self.instance_idx,
            "regions": list(self.regions),
            "region_label": region_label(self.regions),
        }


@dataclass
class AdmitOverlapProbeSpec:
    """One admit-overlap probe: does ``candidate`` co-admit with the held ``active``?

    ``expected_decision`` is derived purely from region-set overlap — disjoint =>
    admit, overlapping => queue. Live Step-2 routing reports the gate's real
    decision, which the aggregator compares against this expectation.
    """

    probe_id: str
    active: Placement
    candidate: Placement
    disjoint: bool
    expected_decision: str
    # transport (placement queue, never /chat)
    transport: str = PLACEMENT_QUEUE_TRANSPORT
    request_priority: str = PLACEMENT_REQUEST_PRIORITY
    workload_class: str = PLACEMENT_WORKLOAD_CLASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "admit_overlap_probe",
            "probe_id": self.probe_id,
            "active": self.active.to_dict(),
            "candidate": self.candidate.to_dict(),
            "disjoint": self.disjoint,
            "expected_decision": self.expected_decision,
            "transport": self.transport,
            "request_priority": self.request_priority,
            "workload_class": self.workload_class,
        }


@dataclass
class RebenchPairSpec:
    """One within-role disjoint instance-shape pair to re-bench at higher samples.

    Bench-indexed by (model, quant); ``role`` is provenance only. Carries the J5
    prior so the aggregated re-bench can report the delta.
    """

    pair_id: str
    role: str
    model: str
    quant: str
    instance_a_idx: int
    instance_b_idx: int
    region_a: tuple[str, ...]
    region_b: tuple[str, ...]
    target_samples: int
    prior_ratio: float | None
    prior_cv: float | None
    # transport (placement queue, never /chat)
    transport: str = PLACEMENT_QUEUE_TRANSPORT
    request_priority: str = PLACEMENT_REQUEST_PRIORITY
    workload_class: str = PLACEMENT_WORKLOAD_CLASS

    @property
    def model_quant_key(self) -> str:
        return f"{self.model}::{self.quant}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "rebench_pair",
            "pair_id": self.pair_id,
            # bench index = model/quant, NEVER role
            "model": self.model,
            "quant": self.quant,
            "model_quant_key": self.model_quant_key,
            "role_provenance": self.role,
            "instance_a_idx": self.instance_a_idx,
            "instance_b_idx": self.instance_b_idx,
            "region_a": list(self.region_a),
            "region_b": list(self.region_b),
            "target_samples": self.target_samples,
            "prior_ratio": self.prior_ratio,
            "prior_cv": self.prior_cv,
            "transport": self.transport,
            "request_priority": self.request_priority,
            "workload_class": self.workload_class,
        }


@dataclass
class Step2SmokePlan:
    """The resolved shape-keyed Step-2 smoke plan (dry-run surface)."""

    probes: list[AdmitOverlapProbeSpec]
    rebench_pairs: list[RebenchPairSpec]
    anchor: Placement
    floor: float
    cv_threshold: float
    target_samples: int
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
            "kind": "shapekeyed_step2_smoke_plan",
            "runner_version": RUNNER_VERSION,
            "anchor": self.anchor.to_dict(),
            "floor": self.floor,
            "cv_threshold": self.cv_threshold,
            "target_samples": self.target_samples,
            "n_probes": len(self.probes),
            "n_admit_expected": sum(
                1 for p in self.probes if p.expected_decision == DECISION_ADMIT
            ),
            "n_queue_expected": sum(
                1 for p in self.probes if p.expected_decision == DECISION_QUEUE
            ),
            "n_rebench_pairs": len(self.rebench_pairs),
            "transport": self.transport_summary(),
            "probes": [p.to_dict() for p in self.probes],
            "rebench_pairs": [r.to_dict() for r in self.rebench_pairs],
            "provenance": dict(self.provenance),
            "notes": list(self.notes),
            "inference_required": self.inference_required,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Plan construction (pure, inference-free)
# ══════════════════════════════════════════════════════════════════════════════


def _regions_of(
    instance_regions: dict[tuple[str, int], frozenset[str]],
    key: tuple[str, int],
) -> tuple[str, ...]:
    return tuple(sorted(instance_regions.get(key, frozenset())))


def build_admit_overlap_probes(
    instance_regions: dict[tuple[str, int], frozenset[str]],
    *,
    anchor: tuple[str, int] = (DEFAULT_ANCHOR_ROLE, DEFAULT_ANCHOR_IDX),
    probe_roles: Iterable[str] = DEFAULT_PROBE_ROLES,
) -> tuple[Placement, list[AdmitOverlapProbeSpec]]:
    """Enumerate admit-overlap probes for every probe-role instance vs the anchor.

    For each ``(role, idx)`` of a ``probe_roles`` role (excluding the anchor
    placement itself), classify it against the anchor's held region set and emit a
    probe whose ``expected_decision`` is ADMIT (disjoint) or QUEUE (overlap).
    Deterministic: probes are ordered by (role, instance_idx).
    """
    probe_role_set = set(probe_roles)
    anchor_regions = _regions_of(instance_regions, anchor)
    anchor_placement = Placement(anchor[0], anchor[1], anchor_regions)

    keys = sorted(
        k for k in instance_regions if k[0] in probe_role_set and k != anchor
    )
    probes: list[AdmitOverlapProbeSpec] = []
    for role, idx in keys:
        cand_regions = _regions_of(instance_regions, (role, idx))
        disjoint = regions_disjoint(anchor_regions, cand_regions)
        expected = DECISION_ADMIT if disjoint else DECISION_QUEUE
        candidate = Placement(role, idx, cand_regions)
        probe_id = (
            f"{anchor[0]}#{anchor[1]}[{region_label(anchor_regions)}]"
            f"__vs__{role}#{idx}[{region_label(cand_regions)}]"
        )
        probes.append(
            AdmitOverlapProbeSpec(
                probe_id=probe_id,
                active=anchor_placement,
                candidate=candidate,
                disjoint=disjoint,
                expected_decision=expected,
            )
        )
    return anchor_placement, probes


def build_rebench_pairs(
    instance_regions: dict[tuple[str, int], frozenset[str]],
    *,
    role: str = VISION_DEFAULT_ROLE,
    model: str = VISION_DEFAULT_MODEL,
    quant: str = VISION_DEFAULT_QUANT,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    priors: dict[str, dict[str, float]] | None = None,
) -> list[RebenchPairSpec]:
    """Enumerate the role's DISJOINT within-role instance-shape pairs to re-bench.

    Overlapping pairs (e.g. vision full{q2,q3}+q2) cannot co-reside and are
    excluded — only disjoint pairs are co-run-benchable. Deterministic: pairs
    ordered by (instance_a_idx, instance_b_idx). Model/quant indexed.
    """
    priors = _VISION_J5_PRIORS if priors is None else priors
    idxs = sorted(idx for (r, idx) in instance_regions if r == role)
    pairs: list[RebenchPairSpec] = []
    for i, a in enumerate(idxs):
        ra = _regions_of(instance_regions, (role, a))
        for b in idxs[i + 1 :]:
            rb = _regions_of(instance_regions, (role, b))
            if not regions_disjoint(ra, rb):
                continue  # overlapping pair — not co-run-benchable
            label = pair_label(ra, rb)
            prior = priors.get(label) or {}
            pairs.append(
                RebenchPairSpec(
                    pair_id=label,
                    role=role,
                    model=model,
                    quant=quant,
                    instance_a_idx=a,
                    instance_b_idx=b,
                    region_a=ra,
                    region_b=rb,
                    target_samples=target_samples,
                    prior_ratio=prior.get("ratio"),
                    prior_cv=prior.get("cv"),
                )
            )
    return pairs


def build_step2_smoke_plan(
    instance_regions: dict[tuple[str, int], frozenset[str]],
    *,
    anchor: tuple[str, int] = (DEFAULT_ANCHOR_ROLE, DEFAULT_ANCHOR_IDX),
    probe_roles: Iterable[str] = DEFAULT_PROBE_ROLES,
    rebench_role: str = VISION_DEFAULT_ROLE,
    rebench_model: str = VISION_DEFAULT_MODEL,
    rebench_quant: str = VISION_DEFAULT_QUANT,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    floor: float = DEFAULT_MATRIX_FLOOR,
    cv_threshold: float = DEFAULT_CV_THRESHOLD,
    topology_hash: str | None = None,
    priors: dict[str, dict[str, float]] | None = None,
) -> Step2SmokePlan:
    """Build the full shape-keyed Step-2 smoke plan (pure — no inference/I/O)."""
    anchor_placement, probes = build_admit_overlap_probes(
        instance_regions, anchor=anchor, probe_roles=probe_roles
    )
    rebench_pairs = build_rebench_pairs(
        instance_regions,
        role=rebench_role,
        model=rebench_model,
        quant=rebench_quant,
        target_samples=target_samples,
        priors=priors,
    )
    provenance = {
        "runner_version": RUNNER_VERSION,
        "topology_hash": topology_hash,
        "anchor": {"role": anchor[0], "instance_idx": anchor[1]},
        "probe_roles": sorted(set(probe_roles)),
        "rebench": {
            "role": rebench_role,
            "model": rebench_model,
            "quant": rebench_quant,
            "model_quant_key": f"{rebench_model}::{rebench_quant}",
        },
        "step2_flag": "ORCHESTRATOR_SHAPE_AWARE_CONTENTION",
    }
    notes = [
        "shape-keyed Step-2 smoke: disjoint region sets EXPECT admit, overlapping "
        "EXPECT queue (region-set overlap is the only authority; never the 'full' "
        "label).",
        "re-bench results are indexed by (model, quant), NEVER by role.",
        "all specs ride the placement queue (background/eval_batch); NEVER /chat.",
        "all produced numbers are pre-promotion OBSERVATIONS (MEASUREMENT.md); they "
        "do not gate the Step-2 flag-on decision on their own.",
    ]
    return Step2SmokePlan(
        probes=probes,
        rebench_pairs=rebench_pairs,
        anchor=anchor_placement,
        floor=floor,
        cv_threshold=cv_threshold,
        target_samples=target_samples,
        provenance=provenance,
        notes=notes,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Result aggregation (pure — synthetic routing outcomes -> smoke verdict)
# ══════════════════════════════════════════════════════════════════════════════


def _normalize_observed(observed: Any) -> dict[str, str]:
    """Accept either {probe_id: decision} or [{probe_id, decision}, ...]."""
    if isinstance(observed, dict):
        return {str(k): str(v) for k, v in observed.items()}
    out: dict[str, str] = {}
    for row in observed or []:
        if isinstance(row, dict) and row.get("probe_id") is not None:
            out[str(row["probe_id"])] = str(row.get("decision"))
    return out


def aggregate_admit_overlap(
    probes: list[AdmitOverlapProbeSpec],
    observed: Any,
) -> dict[str, Any]:
    """Score observed admit/queue decisions against the region-derived expectation.

    ``observed`` maps probe_id -> "admit"|"queue" (a probe with no observation is
    scored ``pass=None`` and excluded from the pass/fail tally). Pure.
    """
    obs = _normalize_observed(observed)
    rows: list[dict[str, Any]] = []
    n_pass = 0
    n_eval = 0
    for p in probes:
        got = obs.get(p.probe_id)
        if got is None:
            ok: bool | None = None
        else:
            ok = got == p.expected_decision
            n_eval += 1
            if ok:
                n_pass += 1
        rows.append(
            {
                "probe_id": p.probe_id,
                "disjoint": p.disjoint,
                "expected": p.expected_decision,
                "observed": got,
                "pass": ok,
            }
        )
    return {
        "kind": "admit_overlap_probe_summary",
        "runner_version": RUNNER_VERSION,
        "n_probes": len(probes),
        "n_admit_expected": sum(
            1 for p in probes if p.expected_decision == DECISION_ADMIT
        ),
        "n_queue_expected": sum(
            1 for p in probes if p.expected_decision == DECISION_QUEUE
        ),
        "n_evaluated": n_eval,
        "n_pass": n_pass,
        "n_fail": n_eval - n_pass,
        "all_pass": (n_eval > 0 and n_pass == n_eval),
        "rows": rows,
        "observation_only": True,
    }


def rebench_verdict(mean_ratio: float | None, floor: float) -> str:
    """ratio >= 1.0 => allow; floor <= ratio < 1.0 => borderline; else block."""
    if mean_ratio is None:
        return "unknown"
    if mean_ratio >= 1.0:
        return "allow"
    if mean_ratio >= floor:
        return "borderline"
    return "block"


def summarize_rebench_pair(
    pair: RebenchPairSpec,
    samples: list[float],
    *,
    floor: float = DEFAULT_MATRIX_FLOOR,
    cv_threshold: float = DEFAULT_CV_THRESHOLD,
) -> dict[str, Any]:
    """Roll co-run ratio samples for one pair into a model/quant-indexed result row.

    mean = arithmetic mean; cv = sample stdev / mean (None for <2 samples or a
    zero mean); ``clean`` = cv at/under the threshold; ``ratified_allow`` = an
    ``allow`` verdict that is also clean (the exact bar the re-bench must clear).
    Pure — no inference. All numbers are pre-promotion observations.
    """
    vals = [float(s) for s in (samples or [])]
    mean = statistics.mean(vals) if vals else None
    cv: float | None = None
    if len(vals) >= 2 and mean:
        cv = statistics.stdev(vals) / mean
    verdict = rebench_verdict(mean, floor)
    clean = cv is not None and cv <= cv_threshold
    ratified_allow = verdict == "allow" and clean
    ratio_delta = (
        mean - pair.prior_ratio
        if (mean is not None and pair.prior_ratio is not None)
        else None
    )
    return {
        "kind": "shapekeyed_rebench_result",
        "runner_version": RUNNER_VERSION,
        # bench index = model/quant, NEVER role
        "model": pair.model,
        "quant": pair.quant,
        "model_quant_key": pair.model_quant_key,
        "role_provenance": pair.role,
        "pair_id": pair.pair_id,
        "region_a": list(pair.region_a),
        "region_b": list(pair.region_b),
        "n_samples": len(vals),
        "target_samples": pair.target_samples,
        "met_sample_target": len(vals) >= pair.target_samples,
        "mean_ratio": mean,
        "cv": cv,
        "clean": clean,
        "verdict": verdict,
        "ratified_allow": ratified_allow,
        "prior_ratio": pair.prior_ratio,
        "prior_cv": pair.prior_cv,
        "ratio_delta_vs_prior": ratio_delta,
        "transport": PLACEMENT_QUEUE_TRANSPORT,
        "observation_only": True,
    }


def aggregate_rebench(
    pairs: list[RebenchPairSpec],
    samples_by_pair: dict[str, list[float]],
    *,
    floor: float = DEFAULT_MATRIX_FLOOR,
    cv_threshold: float = DEFAULT_CV_THRESHOLD,
) -> list[dict[str, Any]]:
    """Aggregate every re-bench pair from its samples (pure). Order preserved."""
    return [
        summarize_rebench_pair(
            p,
            list(samples_by_pair.get(p.pair_id) or []),
            floor=floor,
            cv_threshold=cv_threshold,
        )
        for p in pairs
    ]


def aggregate_smoke(
    plan: Step2SmokePlan,
    *,
    observed_decisions: Any,
    rebench_samples: dict[str, list[float]],
) -> dict[str, Any]:
    """Combine both aggregations into one smoke report (pure)."""
    probe_summary = aggregate_admit_overlap(plan.probes, observed_decisions)
    rebench_rows = aggregate_rebench(
        plan.rebench_pairs,
        rebench_samples,
        floor=plan.floor,
        cv_threshold=plan.cv_threshold,
    )
    n_ratified = sum(1 for r in rebench_rows if r.get("ratified_allow"))
    return {
        "kind": "shapekeyed_step2_smoke_report",
        "runner_version": RUNNER_VERSION,
        "admit_overlap": probe_summary,
        "rebench": rebench_rows,
        "n_rebench_ratified_allow": n_ratified,
        "smoke_pass": bool(probe_summary["all_pass"]),
        "observation_only": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Topology loading (READ-ONLY derive from NUMA_CONFIG; tests pass synthetic)
# ══════════════════════════════════════════════════════════════════════════════


def load_instance_regions(
    numa_config: dict[str, Any] | None = None,
) -> dict[tuple[str, int], frozenset[str]]:
    """Derive the (role, idx) -> region-set map from a NUMA_CONFIG (READ-ONLY).

    ``numa_config`` None -> live ``scripts.server.stack_numa.NUMA_CONFIG`` (imported
    read-only, no inference). Delegates to the production-pure
    ``src.runtime.instance_topology.build_instance_regions`` so the smoke and the
    dispatcher share ONE region model.
    """
    try:
        from src.runtime.instance_topology import build_instance_regions
    except Exception:  # noqa: BLE001
        from runtime.instance_topology import build_instance_regions  # type: ignore
    if numa_config is None:
        try:
            from scripts.server.stack_numa import NUMA_CONFIG
        except Exception:  # noqa: BLE001
            from server.stack_numa import NUMA_CONFIG  # type: ignore
        numa_config = NUMA_CONFIG
    return build_instance_regions(numa_config)


# ══════════════════════════════════════════════════════════════════════════════
# Execution bridge (--execute + env-gated; placement queue; NEVER run in tests)
# ══════════════════════════════════════════════════════════════════════════════


def execute_step2_smoke(
    plan: Step2SmokePlan,
    *,
    output_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:  # pragma: no cover - inference path
    """Drive the plan over the PLACEMENT QUEUE, collect outcomes, aggregate.

    Reached ONLY when both ``--execute`` and ``AUTOPILOT_SHAPEKEYED_STEP2_SMOKE=1``
    are set. Autopilot-stopped assumption: the caller owns the
    no-concurrent-inference window; this function never touches autopilot
    lifecycle/state and never modifies the routing/serving path (the dispatcher and
    gate are consulted through the normal eval_batch placement path only).

    Never exercised by the unit tests — the whole bridge is double-gated and unread
    under the zero-inference constraint. The real seams:

      * admit-overlap: submit each probe's candidate placement as an eval_batch
        (background) placement request while the anchor is held, and record the
        dispatcher's admit/queue outcome for that ``candidate_topology_idx``.
      * re-bench: run each disjoint pair's co-run bench via the eval_batch lane
        (``bench_canonical`` recipe), collecting ``target_samples`` ratios.

    Both feed the pure aggregators; the batch ledger is written by the operator
    loop, not here.
    """
    observed_decisions = _drive_admit_overlap_probes(plan, seed=seed)
    rebench_samples = _drive_rebench_pairs(plan, seed=seed)
    report = aggregate_smoke(
        plan,
        observed_decisions=observed_decisions,
        rebench_samples=rebench_samples,
    )
    if output_path is not None:
        _write_report(Path(output_path), report)
    return report


def _drive_admit_overlap_probes(
    plan: Step2SmokePlan, *, seed: int
) -> dict[str, str]:  # pragma: no cover - inference path
    raise NotImplementedError(
        "live admit-overlap probing needs a quiesce window + a single-worker API "
        "(the multi-worker confound in the within-role handoff's J2/J3 finding). "
        "Wire it to the eval_batch placement lane; do NOT touch the serving path."
    )


def _drive_rebench_pairs(
    plan: Step2SmokePlan, *, seed: int
) -> dict[str, list[float]]:  # pragma: no cover - inference path
    raise NotImplementedError(
        "live vision re-bench needs operator-approved inference via the codified "
        "bench_canonical recipe over the eval_batch lane; not run here."
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# Top-level orchestration (--execute + env-gated dry-run vs execute)
# ══════════════════════════════════════════════════════════════════════════════


def run_shapekeyed_step2_smoke(
    plan: Step2SmokePlan,
    *,
    execute: bool = False,
    output_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Return the plan as a dry-run, or (double-gated) drive it over the queue.

    DEFAULT (``execute`` False OR ``AUTOPILOT_SHAPEKEYED_STEP2_SMOKE`` unset/false):
    returns the resolved plan as a dry-run and runs NO inference — the entire
    surface the fixture test exercises. Real execution requires BOTH ``execute``
    (the CLI ``--execute``) AND the env flag.
    """
    env_on = _env_flag_enabled(SHAPEKEYED_STEP2_INFERENCE_ENV)
    if not (execute and env_on):
        reason_bits = []
        if not execute:
            reason_bits.append("--execute not passed")
        if not env_on:
            reason_bits.append(f"{SHAPEKEYED_STEP2_INFERENCE_ENV} not set")
        return {
            "mode": "dry_run",
            "runner_version": RUNNER_VERSION,
            "inference_ran": False,
            "reason": (
                "; ".join(reason_bits)
                + " — plan returned as a dry-run (no inference, placement-queue "
                "transport, serving path untouched)."
            ),
            "n_probes": len(plan.probes),
            "n_rebench_pairs": len(plan.rebench_pairs),
            "plan": plan.to_dict(),
        }
    report = execute_step2_smoke(plan, output_path=output_path, seed=seed)
    return {
        "mode": "execute",
        "runner_version": RUNNER_VERSION,
        "inference_ran": True,
        "output_path": str(output_path) if output_path else None,
        "plan": plan.to_dict(),
        "report": report,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI (__main__)
# ══════════════════════════════════════════════════════════════════════════════


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plan (and, only under --execute + "
            f"{SHAPEKEYED_STEP2_INFERENCE_ENV}=1, drive over the placement queue) "
            "the shape-keyed Step-2 admit-overlap probe + vision re-bench. Default "
            "is a pure dry-run that prints the plan and runs NO inference."
        )
    )
    p.add_argument(
        "--numa-config",
        default=None,
        help="optional path to a NUMA_CONFIG JSON; default derives from the live "
        "scripts.server.stack_numa.NUMA_CONFIG (READ-ONLY)",
    )
    p.add_argument("--anchor-role", default=DEFAULT_ANCHOR_ROLE)
    p.add_argument("--anchor-idx", type=int, default=DEFAULT_ANCHOR_IDX)
    p.add_argument(
        "--probe-roles",
        default=",".join(DEFAULT_PROBE_ROLES),
        help="comma-separated roles to probe against the anchor",
    )
    p.add_argument("--rebench-role", default=VISION_DEFAULT_ROLE)
    p.add_argument("--rebench-model", default=VISION_DEFAULT_MODEL)
    p.add_argument("--rebench-quant", default=VISION_DEFAULT_QUANT)
    p.add_argument("--target-samples", type=int, default=DEFAULT_TARGET_SAMPLES)
    p.add_argument("--floor", type=float, default=DEFAULT_MATRIX_FLOOR)
    p.add_argument("--cv-threshold", type=float, default=DEFAULT_CV_THRESHOLD)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        default=None,
        help="JSON path for the smoke report (execute path only)",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="attempt execution (STILL env-gated by "
        f"{SHAPEKEYED_STEP2_INFERENCE_ENV}=1; otherwise falls back to dry-run)",
    )
    return p


def _load_numa_config(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # A JSON NUMA config stores instances as lists; build_instance_regions only
    # reads entry[0] (the cpu_list), so list-of-lists is fine as-is.
    return raw


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    numa_config = _load_numa_config(args.numa_config)
    try:
        instance_regions = load_instance_regions(numa_config)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"failed to load topology: {exc}"}, indent=2))
        return 2
    if not instance_regions:
        print(json.dumps({"error": "empty instance-region topology"}, indent=2))
        return 2

    probe_roles = [r.strip() for r in args.probe_roles.split(",") if r.strip()]
    plan = build_step2_smoke_plan(
        instance_regions,
        anchor=(args.anchor_role, args.anchor_idx),
        probe_roles=probe_roles,
        rebench_role=args.rebench_role,
        rebench_model=args.rebench_model,
        rebench_quant=args.rebench_quant,
        target_samples=args.target_samples,
        floor=args.floor,
        cv_threshold=args.cv_threshold,
    )

    if not args.execute:
        # Pure dry-run — print the plan; no inference regardless of the env flag.
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True, default=str))
        return 0

    result = run_shapekeyed_step2_smoke(
        plan,
        execute=True,
        output_path=Path(args.output) if args.output else None,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
