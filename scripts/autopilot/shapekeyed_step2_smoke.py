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
     see invariant #1 in the handoff). The expectation model is selectable
     (``--expectation``): **"seam"** keeps the ORIGINAL flag-on expectation
     (disjoint -> ADMIT, overlapping -> QUEUE/fail-closed), while the standing
     **"replacement"** default restates the expectation against the fleet layer's
     ACTUAL overlap handling — the placement machine re-places forced eval_batch
     requests onto a disjoint instance, so an overlapping candidate is EXPECTED
     to be admitted via re-placement and the safety invariant is *never co-place*
     (an admit whose echoed ``contention_gate.candidate_topology_idx`` still
     overlaps the held anchor is a failure; an unexpected queue means the flag-on
     seam armed). Live Step-2 routing reports the gate's actual decision; the
     pure aggregator scores observed-vs-expected.

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
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

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
# the held set is EXPECTED to admit; an OVERLAPPING one is EXPECTED to queue
# (only in "seam" mode — see EXPECTATION_* below).
DECISION_ADMIT = "admit"
DECISION_QUEUE = "queue"

# Expectation modes for the admit-overlap bracket (operator decision 2026-08-24,
# ROUTE-A1 step-2 smoke: the overlap-queue premise was falsified for the live
# fleet layer — the placement machine RE-PLACES forced eval_batch requests onto a
# disjoint instance and the pair matrix allows (borderline -> allow), so all 5
# overlapping probes admitted instead of queuing):
#   * "replacement" (STANDING default) — restates the smoke's expectation against
#     the fleet layer's actual overlap handling: an overlapping candidate is
#     EXPECTED to be admitted via re-placement. The safety invariant is "never
#     co-place": an admit whose echoed contention_gate.candidate_topology_idx
#     still OVERLAPS the held anchor is a co-placement (always a failure); an
#     unexpected queue (the flag-on seam engaging) is a distinct failure outcome
#     ("queued_unexpected") that still goes red.
#   * "seam" — the ORIGINAL flag-on expectation: overlap -> QUEUE (fail closed),
#     disjoint -> admit. Report output stays byte-compatible with the 2026-08-24
#     route-a1 artifact so the flag-on exercise compares directly.
EXPECTATION_REPLACEMENT = "replacement"
EXPECTATION_SEAM = "seam"
DEFAULT_EXPECTATION = EXPECTATION_REPLACEMENT

# Re-placement-aware verdicts (aggregation, "replacement" mode only). The plain
# "admit"/"queue" outcomes stay as the fallback classification (responses without
# contention_gate topology evidence, queue/503 paths, pre-classified strings).
DECISION_ADMIT_DISJOINT = "admit_disjoint"
DECISION_ADMIT_OVERLAP = "admit_overlap"
DECISION_QUEUED_UNEXPECTED = "queued_unexpected"
# Marker on a co-placement row: the admitted candidate_topology_idx overlaps the
# held anchor — the safety invariant violation. ALWAYS a failure.
CO_PLACEMENT_MARKER = "CO-PLACEMENT"

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

# Default probe anchor (the classic handoff example): ingest_long_context's node0
# HALF instance (48t, cores 0-47,96-143 — region set {q0,q1}). Quarters do not
# exist in production (retired 2026-07-30); the topology is the FULL 96t shape
# plus two 48t HALVES per role. Disjoint (node1-half) candidates must admit while
# overlapping (node0-half/full) candidates must queue.
DEFAULT_ANCHOR_ROLE = "ingest_long_context"
DEFAULT_ANCHOR_IDX = 1  # ingest node0 HALF (0-47,96-143); idx 0 = full 0-95
DEFAULT_PROBE_ROLES = (
    "frontdoor",
    "ingest_long_context",
    "worker_general",
    # NOTE: vision_escalation deliberately absent — it serves on GPU (ROCm0) and
    # has no CPU-region instance in build_instance_regions (2026-08-23 ground
    # truth); its within-role re-bench section below is keyed by (model, quant),
    # not by a CPU instance.
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

    ``expected_decision`` is derived from region-set overlap AND the plan's
    expectation mode: in "seam" mode disjoint => admit and overlapping => queue
    (the original flag-on model); in "replacement" mode (standing default) every
    candidate is expected to admit, because the placement machine re-places a
    forced eval_batch request onto a disjoint instance — the safety invariant is
    then "never co-place", judged by the echoed ``candidate_topology_idx`` at
    aggregation, not by the requested candidate. Live Step-2 routing reports the
    gate's real decision, which the aggregator compares against this expectation.
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
    # expectation mode ("replacement" standing default / "seam" flag-on model)
    expectation: str = DEFAULT_EXPECTATION
    # (role, idx) -> region-set map the re-placement evidence is resolved against
    # (the probe response's contention_gate.candidate_topology_idx names an
    # instance of the probe role in THIS map). Pure data, never mutated.
    instance_regions: dict[tuple[str, int], frozenset[str]] = field(
        default_factory=dict
    )

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
            "expectation": self.expectation,
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
            "instance_regions": [
                {
                    "role": role,
                    "instance_idx": idx,
                    "regions": sorted(regions),
                    "region_label": region_label(regions),
                }
                for (role, idx), regions in sorted(self.instance_regions.items())
            ],
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
    expectation: str = DEFAULT_EXPECTATION,
) -> tuple[Placement, list[AdmitOverlapProbeSpec]]:
    """Enumerate admit-overlap probes for every probe-role instance vs the anchor.

    For each ``(role, idx)`` of a ``probe_roles`` role (excluding the anchor
    placement itself), classify it against the anchor's held region set and emit a
    probe. ``expected_decision`` depends on ``expectation``:

      * "seam" — disjoint => ADMIT, overlapping => QUEUE (the original flag-on
        overlap-queue model).
      * "replacement" (standing default) — EVERY candidate is expected to ADMIT:
        the fleet layer re-places a forced eval_batch request onto a disjoint
        instance, so admission is the expectation for overlapping candidates too;
        the safety invariant ("never co-place") is then judged at aggregation on
        the OBSERVED ``contention_gate.candidate_topology_idx``, not on the
        requested candidate.

    Deterministic: probes are ordered by (role, instance_idx). ``disjoint`` (the
    requested candidate vs the anchor) is always recorded as a fact regardless of
    mode — it describes the REQUEST, not the expectation.
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
        if expectation == EXPECTATION_SEAM:
            expected = DECISION_ADMIT if disjoint else DECISION_QUEUE
        else:
            expected = DECISION_ADMIT  # replacement: re-placed onto a disjoint instance
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
    expectation: str = DEFAULT_EXPECTATION,
) -> Step2SmokePlan:
    """Build the full shape-keyed Step-2 smoke plan (pure — no inference/I/O)."""
    anchor_placement, probes = build_admit_overlap_probes(
        instance_regions,
        anchor=anchor,
        probe_roles=probe_roles,
        expectation=expectation,
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
        "expectation": expectation,
    }
    if expectation == EXPECTATION_SEAM:
        bracket_note = (
            "seam-mode bracket: disjoint region sets EXPECT admit, overlapping "
            "EXPECT queue (region-set overlap is the only authority; never the "
            "'full' label)."
        )
    else:
        bracket_note = (
            "replacement-mode bracket (standing default): the placement machine "
            "re-places forced eval_batch requests onto a disjoint instance, so "
            "EVERY candidate EXPECTS admit; the safety invariant is never "
            "co-place — judged on the echoed contention_gate.candidate_topology_"
            "idx, and an unexpected queue (seam armed) is a failure."
        )
    notes = [
        bracket_note,
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
        expectation=expectation,
        instance_regions=instance_regions,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Result aggregation (pure — synthetic routing outcomes -> smoke verdict)
# ══════════════════════════════════════════════════════════════════════════════


def _normalize_observed(observed: Any) -> dict[str, Any]:
    """Accept either {probe_id: decision} or [{probe_id, decision}, ...].

    Values stay as the driver produced them: a plain "admit"/"queue" string, or —
    for a response whose contention_gate echoed topology evidence — a dict
    {"decision": "admit", "candidate_topology_idx": int, "role": str,
    "regions": [...]}. Dict values are preserved, never coerced.
    """
    if isinstance(observed, dict):
        return {str(k): v for k, v in observed.items()}
    out: dict[str, Any] = {}
    for row in observed or []:
        if isinstance(row, dict) and row.get("probe_id") is not None:
            out[str(row["probe_id"])] = row.get("decision")
    return out


def aggregate_admit_overlap(
    probes: list[AdmitOverlapProbeSpec],
    observed: Any,
    *,
    expectation: str = DEFAULT_EXPECTATION,
) -> dict[str, Any]:
    """Score observed decisions against the expectation model.

    ``observed`` maps probe_id -> "admit"|"queue" (plain) or -> a dict carrying
    the echoed contention-gate evidence (see ``_classify_probe_outcome``); a probe
    with no observation is scored ``pass=None`` and excluded from the pass/fail
    tally. Pure.

    ``expectation`` semantics:

      * "seam" — the original flag-on model: observed "admit"/"queue" is compared
        directly to the region-derived ``expected_decision`` (disjoint=>admit,
        overlap=>queue). Rows and summary are byte-compatible with the 2026-08-24
        route-a1 artifact: exactly ``probe_id/disjoint/expected/observed/pass``
        per row, no verdict keys.
      * "replacement" (standing default) — every probe expects "admit". The
        verdict is computed from the observed placement against the anchor's
        region set (``probe.active.regions``):
          - "admit_disjoint" — admit whose echoed candidate_topology_idx is
            DISJOINT from the anchor  => PASS (re-placement, the fleet behavior).
          - "admit_overlap" — admit whose echoed candidate_topology_idx OVERLAPS
            the anchor => FAIL, row marked "CO-PLACEMENT" (the safety invariant:
            never co-place). ALWAYS a failure.
          - "admit" — plain admit without topology evidence (fallback
            classification, e.g. non-instrumented path) => PASS; the row's
            verdict makes the missing evidence visible.
          - "queued_unexpected" — observed "queue" while the standing expectation
            is replacement (the flag-on seam engaged) => FAIL (distinct outcome).
    """
    obs = _normalize_observed(observed)
    replacement = expectation == EXPECTATION_REPLACEMENT
    rows: list[dict[str, Any]] = []
    n_pass = 0
    n_eval = 0
    n_co_placement = 0
    n_queued_unexpected = 0
    for p in probes:
        got = obs.get(p.probe_id)
        if got is None:
            ok: bool | None = None
            verdict: str | None = None
            observed_display: str | None = None
        else:
            n_eval += 1
            if isinstance(got, dict):
                # topology-evidenced admit (driver's re-placement extraction)
                decision = str(got.get("decision") or "")
                observed_display = decision
                if replacement:
                    cand_regions = frozenset(got.get("regions") or ())
                    disjoint_observed = regions_disjoint(p.active.regions, cand_regions)
                    if decision == DECISION_ADMIT:
                        verdict = (
                            DECISION_ADMIT_DISJOINT
                            if disjoint_observed
                            else DECISION_ADMIT_OVERLAP
                        )
                    else:
                        verdict = DECISION_QUEUED_UNEXPECTED
                    ok = verdict == DECISION_ADMIT_DISJOINT
                    if verdict == DECISION_ADMIT_OVERLAP:
                        n_co_placement += 1
                    elif verdict == DECISION_QUEUED_UNEXPECTED:
                        n_queued_unexpected += 1
                    if ok:
                        n_pass += 1
                else:
                    # seam mode: evidence is collapsed to its decision string so
                    # the 2026-08-24 report shape is preserved byte-for-byte.
                    ok = decision == p.expected_decision
                    if ok:
                        n_pass += 1
            else:
                observed_display = str(got)
                if replacement:
                    if observed_display == DECISION_ADMIT:
                        verdict = DECISION_ADMIT
                        ok = True
                        n_pass += 1
                    else:
                        verdict = DECISION_QUEUED_UNEXPECTED
                        ok = False
                        n_queued_unexpected += 1
                else:
                    ok = observed_display == p.expected_decision
                    if ok:
                        n_pass += 1
        row: dict[str, Any] = {
            "probe_id": p.probe_id,
            "disjoint": p.disjoint,
            "expected": p.expected_decision,
            "observed": observed_display,
            "pass": ok,
        }
        if replacement:
            row["verdict"] = verdict
            if verdict == DECISION_ADMIT_OVERLAP:
                row["marker"] = CO_PLACEMENT_MARKER
        rows.append(row)
    summary: dict[str, Any] = {
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
    if replacement:
        # replacement-only counts (seam mode stays byte-compatible with 2026-08-24)
        summary["expectation"] = EXPECTATION_REPLACEMENT
        summary["n_co_placement"] = n_co_placement
        summary["n_queued_unexpected"] = n_queued_unexpected
    return summary


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
    probe_summary = aggregate_admit_overlap(
        plan.probes, observed_decisions, expectation=plan.expectation
    )
    rebench_rows = aggregate_rebench(
        plan.rebench_pairs,
        rebench_samples,
        floor=plan.floor,
        cv_threshold=plan.cv_threshold,
    )
    n_ratified = sum(1 for r in rebench_rows if r.get("ratified_allow"))
    report: dict[str, Any] = {
        "kind": "shapekeyed_step2_smoke_report",
        "runner_version": RUNNER_VERSION,
        "admit_overlap": probe_summary,
        "rebench": rebench_rows,
        "n_rebench_ratified_allow": n_ratified,
        "smoke_pass": bool(probe_summary["all_pass"]),
        "observation_only": True,
    }
    if plan.expectation == EXPECTATION_REPLACEMENT:
        # replacement mode announces itself at the report level; "seam" mode stays
        # byte-compatible with the 2026-08-24 route-a1 artifact (no added keys).
        report["expectation"] = EXPECTATION_REPLACEMENT
    return report


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
#
# Admit-vs-queue detection — the SIGNAL that decides measurement validity
# --------------------------------------------------------------------------
# A Step-2 admit-overlap probe rides the SAME seam ``run_paired_ab`` uses:
# ``seeding_orchestrator.call_orchestrator_forced`` with
# ``request_priority=background`` + ``workload_class=eval_batch`` (POST /chat over
# the placement queue, never a foreground /chat). The orchestrator does NOT return
# an explicit "admitted vs queued" field, so the bridge INFERS the gate decision
# from the response the placement path actually produces:
#
#   * ADMIT  — the request got a real ``answer`` (no error). When the shape-aware
#     gate admits a candidate ``candidate_topology_idx`` (disjoint from the held
#     anchor), the dispatch poll loop in
#     ``src/backends/concurrency_aware.py::_dispatch`` (~L1078-1102) acquires it and
#     inference proceeds → a normal ChatResponse.
#   * QUEUE  — the shape-aware gate fails closed on an overlapping candidate; the
#     dispatch poll loop exhausts ``max_queue_wait_ms`` and raises
#     ``ContentionDenied`` ("placement timeout …", concurrency_aware.py:1115). The
#     API's exception handler (``src/api/__init__.py`` ~L359-374) maps that to
#     HTTP 503 ``{"error":"contention_denied", ...}``. Through
#     ``call_orchestrator_forced`` that 503 surfaces as ``{"answer":"",
#     "error":"…503…"}`` (its ``response.raise_for_status()`` branch,
#     seeding_orchestrator.py:757-780).
#
# ``_classify_admit_queue`` encodes exactly that mapping and, crucially, returns
# ``None`` (unscored) for any response that is NOT clean gate evidence — a
# backend 500/502, a circuit-open 503 (which arrives as an ``[ERROR: … unavailable]``
# answer + ``error_code=503``, distinct from the contention 503), or an empty
# non-error body. See the two honest limitations documented on
# ``_drive_admit_overlap_probes``.

# Orchestrator API + probe knobs (execute path only; never used in tests/dry-run).
DEFAULT_ORCH_URL = "http://localhost:8000"
# Probe timeout drives call_orchestrator_forced's eval_batch max_queue_wait_ms
# (=min(timeout*500, 300_000)); 120 s ⇒ a 60 s queue window, so an OVERLAPPING
# candidate genuinely exhausts it (→ 503 → QUEUE) rather than slow-admitting.
# The operator must hold the anchor for at least this long.
PROBE_TIMEOUT_S = 120
PROBE_MAX_TOKENS = 4  # a minimal generation — we score routing, not the answer

# Response markers that unambiguously identify the contention gate's fail-closed
# path (case-insensitive substring match over error/error_detail).
_QUEUE_SIGNAL_MARKERS = (
    "contention",          # 503 body {"error":"contention_denied", ...}
    "contention_denied",
    "placement timeout",   # ContentionDenied reason string from _dispatch
    "retry_after",
    "backpressure",
    "queued",
)
_BACKEND_ERROR_PREFIXES = ("[error", "[failed")


def _classify_admit_queue(outcome: Any) -> str | None:
    """Map one probe outcome to ``"admit"`` / ``"queue"`` / ``None`` (unscored).

    ``outcome`` is either a raw ``call_orchestrator_forced`` response dict OR a
    pre-classified decision string (so an injected ``probe_fn`` may return either).
    ``None`` means "not clean gate evidence" — the aggregator excludes it from the
    pass/fail tally rather than counting a backend failure as a gate decision. Pure.
    """
    if isinstance(outcome, str):
        v = outcome.strip().lower()
        return v if v in (DECISION_ADMIT, DECISION_QUEUE) else None
    if not isinstance(outcome, dict):
        return None

    answer = outcome.get("answer")
    answer_s = answer.strip() if isinstance(answer, str) else ""
    answer_is_backend_error = answer_s.lower().startswith(_BACKEND_ERROR_PREFIXES)
    error = str(outcome.get("error") or "").strip()
    error_detail = str(outcome.get("error_detail") or "").strip()
    error_code = outcome.get("error_code")
    blob = f"{error} {error_detail}".lower()

    # (1) Explicit contention-gate fail-closed signal → QUEUE.
    if any(m in blob for m in _QUEUE_SIGNAL_MARKERS):
        return DECISION_QUEUE
    # A bare HTTP-503 error with NO backend "[ERROR:/…]" answer is the contention
    # handler's JSONResponse surfaced through raise_for_status (empty answer).
    # A circuit-open 503 instead arrives as an "[ERROR: … unavailable]" answer +
    # error_code and is NOT gate evidence (falls through to (2)).
    if ("503" in blob or str(error_code) == "503") and not answer_is_backend_error:
        if not answer_s:
            return DECISION_QUEUE
    # (2) Any OTHER error (backend 500/502, circuit-open, connection drop, an
    #     "[ERROR:/…]" answer) is not a gate decision → leave the probe unscored.
    if error or error_detail or error_code or answer_is_backend_error:
        return None
    # (3) A real answer with no error ⇒ the request admitted (immediately OR after
    #     a bounded queue-wait the /chat body does not expose — see the limitation
    #     note on the driver).
    if answer_s:
        return DECISION_ADMIT
    # (4) Empty answer, no error: ambiguous → unscored.
    return None


def _classify_probe_outcome(
    outcome: Any,
    spec: AdmitOverlapProbeSpec,
    instance_regions: dict[tuple[str, int], frozenset[str]],
) -> str | dict[str, Any] | None:
    """Classify one probe outcome, enriching an admit with re-placement evidence.

    Starts from the existing ``_classify_admit_queue`` ("admit"/"queue"/None) and,
    ONLY when the outcome is a clean admit AND the response carries the echoed
    contention-gate verdict (BRIDGE RESIDUAL 1 — landed 2026-08-13 as
    ``c61b8184``), enriches it into a dict:

        {"decision": "admit", "candidate_topology_idx": <int>, "role": <str>,
         "regions": [...]}

    The echoed ``contention_gate`` block has the shape
    (``src/scheduling/gate_observation.py::record``, stamped by
    ``src/api/routes/chat.py`` into ``response.contention_gate``):

        {"admitted": bool, "decision": str, "waited_s": float,
         "candidate_topology_idx": int|None, "queued_then_admitted": bool,
         "reason": str?, "role": str?}

    ``candidate_topology_idx`` is the topology index of the instance the
    placement machine actually dispatched to (e.g. 2 for frontdoor — the node1
    half, disjoint from a held node0-half anchor, in the 2026-08-24 run); its
    region set is resolved against ``instance_regions`` from the PLAN. The
    aggregator then judges that placement against the anchor's region set
    (admit_disjoint vs admit_overlap). When any evidence is missing — no
    contention_gate (queue/503 paths, non-instrumented admits), a non-int idx, an
    unknown (role, idx) — the classification FALLS BACK to the plain
    ``_classify_admit_queue`` result, so a missing echo never fabricates or drops
    a verdict. Pure.
    """
    base = _classify_admit_queue(outcome)
    if base != DECISION_ADMIT or not isinstance(outcome, dict):
        return base
    gate = outcome.get("contention_gate")
    if not isinstance(gate, dict):
        return base
    if gate.get("admitted") is not True:
        return base
    idx = gate.get("candidate_topology_idx")
    if not isinstance(idx, int) or isinstance(idx, bool):
        return base
    role = gate.get("role") or spec.candidate.role
    if not isinstance(role, str) or not role:
        return base
    regions = instance_regions.get((role, idx))
    if regions is None:
        return base
    return {
        "decision": DECISION_ADMIT,
        "candidate_topology_idx": idx,
        "role": role,
        "regions": tuple(sorted(regions)),
    }


def _load_call_orchestrator_forced() -> Callable[..., dict[str, Any]]:  # pragma: no cover - inference path
    """Import ``call_orchestrator_forced`` (the SAME seam ``run_paired_ab`` uses).

    Tries the research benchmark dir first (parity with
    ``run_paired_ab._default_arm_probe``), then this repo's ``scripts/benchmark``.
    """
    _research_bench = "/mnt/raid0/llm/epyc-inference-research/scripts/benchmark"
    _orch_bench = str(ORCH_ROOT / "scripts" / "benchmark")
    for _p in (_research_bench, _orch_bench):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from scripts.benchmark.seeding_orchestrator import (  # type: ignore
            call_orchestrator_forced,
        )
    except Exception:  # noqa: BLE001
        from seeding_orchestrator import call_orchestrator_forced  # type: ignore
    return call_orchestrator_forced


def _default_admit_overlap_probe(
    spec: AdmitOverlapProbeSpec, *, seed: int
) -> dict[str, Any]:  # pragma: no cover - inference path
    """Dispatch ONE small generation for this probe's candidate over the placement
    queue and return the raw orchestrator response dict.

    Mirrors ``run_paired_ab._default_arm_probe``: ``call_orchestrator_forced`` with
    ``force_role`` pinned to the candidate placement's role, ``request_priority=
    background`` + ``workload_class=eval_batch`` (placement queue — NEVER /chat).
    The anchor must be held live for ~``PROBE_TIMEOUT_S`` (operator quiesce window)
    so an overlapping candidate actually times out rather than slow-admitting.
    """
    call_orchestrator_forced = _load_call_orchestrator_forced()
    prompt = f"Reply with OK. (shapekeyed step2 admit probe {spec.probe_id} seed={seed})"
    return call_orchestrator_forced(
        prompt=prompt,
        force_role=spec.candidate.role,
        force_mode="",
        url=DEFAULT_ORCH_URL,
        timeout=PROBE_TIMEOUT_S,
        request_priority=spec.request_priority,   # background → placement queue
        workload_class=spec.workload_class,        # eval_batch (not /chat)
        max_tokens=PROBE_MAX_TOKENS,
    )


def _default_anchor_holder_fn() -> dict[str, list[int]]:  # pragma: no cover - inference path
    """Live seam: currently-held (role, instance_idx) CPU-region locks.

    Read-only filesystem scan of the region-lock layer
    (``src.runtime.cpu_region_lock.active_region_holders``) — no API call, no
    network, no inference. Returns {} when region locks are disabled or nothing
    is dispatching; the driver fails closed on an unverifiable scan. Never
    exercised under test (tests inject ``anchor_hold_fn``).
    """
    try:
        from src.runtime.cpu_region_lock import active_region_holders
    except Exception:  # noqa: BLE001
        from runtime.cpu_region_lock import active_region_holders  # type: ignore
    return active_region_holders() or {}


def _verify_anchor_held(
    plan: Step2SmokePlan,
    holders: dict[str, list[int]],
) -> None:
    """Fail closed unless the anchor placement is verifiably held live.

    The admit-vs-queue signal is only meaningful while the operator holds the
    anchor placement: an overlapping candidate queues (503) only because the
    anchor occupies its region locks. An unreadable/empty holder scan, or the
    anchor instance absent from it, is NOT verification — fail closed with an
    actionable message instead of firing probes that would measure nothing.
    Pure (no I/O): the caller supplies the holder map.
    """
    if not holders:
        raise RuntimeError(
            "anchor-hold precondition cannot be verified: no CPU-region "
            "holders reported (empty region-lock scan — is "
            "PER_REGION_LOCKS enabled and is the stack live?). The operator "
            f"MUST hold the anchor placement {plan.anchor.role}"
            f"#{plan.anchor.instance_idx} (regions "
            f"{region_label(plan.anchor.regions)}) for ~{PROBE_TIMEOUT_S}s "
            "before driving the smoke."
        )
    held_idx = {int(i) for i in holders.get(plan.anchor.role, [])}
    if plan.anchor.instance_idx not in held_idx:
        raise RuntimeError(
            "anchor-hold precondition cannot be verified: "
            f"{plan.anchor.role} currently holds instances "
            f"{sorted(held_idx)!r}, not the anchor instance "
            f"#{plan.anchor.instance_idx} (regions "
            f"{region_label(plan.anchor.regions)}). The operator MUST hold "
            f"{plan.anchor.role}#{plan.anchor.instance_idx} for ~"
            f"{PROBE_TIMEOUT_S}s so probe outcomes are measured against a "
            "verifiably-held anchor (seam mode: overlapping candidates genuinely "
            "queue (503); replacement mode: the echoed candidate_topology_idx is "
            "judged against regions that are truly held)."
        )


def _verify_probe_signal(plan: Step2SmokePlan) -> None:
    """Fail closed when the plan cannot produce the signal contrast.

    The smoke exists to observe BOTH sides of the bracket — a disjoint-requested
    candidate against an overlapping-requested one (seam mode: disjoint admits
    while overlapping queues; replacement mode: both admit, but only the disjoint
    placement is the safe outcome — the overlapping REQUEST is the one whose
    re-placement evidence is measured). The contrast is a property of the
    REQUESTED candidates (region-set overlap vs the anchor), NOT of the
    expectation mode: in replacement mode every probe expects "admit", so an
    expected-decision-based check would refuse every valid plan. A plan with zero
    probes, or every candidate on the same side of the anchor (e.g. a FULL anchor
    against which every candidate overlaps), cannot measure the contrast and
    would merely re-report its own expectation. Refuse with an actionable message
    rather than "still reporting" (handoff 2026-08-12). Pure.
    """
    n_disjoint = sum(1 for p in plan.probes if p.disjoint)
    n_overlap = len(plan.probes) - n_disjoint
    if n_disjoint == 0 or n_overlap == 0:
        raise RuntimeError(
            "admit-vs-queue signal structurally unobtainable from this plan: "
            f"{len(plan.probes)} probes ({n_disjoint} disjoint-requested, "
            f"{n_overlap} overlapping-requested) — the smoke must contain BOTH a "
            "disjoint and an overlapping candidate vs the anchor. With the "
            "default anchor this usually means the anchor is the FULL instance "
            "(ingest_long_context idx 0 is NUMA_FULL 0-95); pass --anchor-idx 1 "
            "or 2 for a 48t HALF anchor so disjoint candidates exist. Quarters "
            "do not exist in production (retired 2026-07-30)."
        )


def _drive_admit_overlap_probes(
    plan: Step2SmokePlan,
    *,
    seed: int,
    probe_fn: Callable[..., Any] | None = None,
    anchor_hold_fn: Callable[[], dict[str, list[int]]] | None = None,
) -> dict[str, str | dict[str, Any]]:
    """Drive every admit-overlap probe and collect its observed gate decision.

    For each probe spec, ``probe_fn(spec, seed=seed)`` returns a
    ``call_orchestrator_forced`` response dict (or a pre-classified decision
    string); ``_classify_probe_outcome`` maps it to ``"admit"``/``"queue"`` — or,
    for a clean admit whose response echoes the gate verdict, to a dict carrying
    the re-placement evidence (``candidate_topology_idx`` + resolved regions).
    Probes whose outcome is not clean gate evidence are OMITTED (the aggregator
    then scores them ``pass=None`` and excludes them). Returns
    ``{probe_id: decision_or_evidence}`` in the exact shape
    ``aggregate_admit_overlap`` consumes (mode-aware: it applies the standing
    "replacement" semantics when the plan's expectation is replacement, and the
    byte-compatible seam semantics otherwise).

    **Fail-closed preconditions, checked BEFORE any probe fires**: the anchor
    placement must be verifiably held (``_verify_anchor_held`` — the operator
    must hold the anchor for ~``PROBE_TIMEOUT_S``) and the plan must contain
    both a disjoint- and an overlapping-requested probe
    (``_verify_probe_signal`` — the contrast is on the REQUESTED candidates, not
    the expectation mode). Either unmet ⇒ the drive raises with an actionable
    message and NO inference happens.

    ``probe_fn``/``anchor_hold_fn`` are injected in tests (like
    ``run_paired_ab``'s ``arm_probe``); the defaults hit the live placement
    queue + region-lock scan and are never exercised under test.

    Measurement-validity notes (flagged, not faked):

      1. **Instance-granular evidence via the echo.** ``call_orchestrator_forced``
         pins ``force_role`` only (no ``candidate_topology_idx``), so the REQUEST
         is role-granular; since 2026-08-13 (bridge residual 1) the response's
         ``contention_gate.candidate_topology_idx`` echoes the instance the
         placement machine actually dispatched to, which is what the
         re-placement verdict is computed from. A response without the echo falls
         back to the role-granular admit/queue classification.
      2. **Admit hides queue-then-admit.** The /chat body carries no ``waited_s``
         (that lands only in ``ContentionGate._metrics``), so a candidate that
         QUEUED then admitted within budget is indistinguishable from an immediate
         admit. Only a fail-closed **timeout** (503) is observably a QUEUE.
    """
    _verify_probe_signal(plan)
    _verify_anchor_held(plan, (anchor_hold_fn or _default_anchor_holder_fn)())
    probe = probe_fn or _default_admit_overlap_probe
    observed: dict[str, str | dict[str, Any]] = {}
    for spec in plan.probes:
        decision = _classify_probe_outcome(
            probe(spec, seed=seed), spec, plan.instance_regions
        )
        if decision is not None:
            observed[spec.probe_id] = decision
    return observed


def _default_rebench_pair_samples(
    pair: RebenchPairSpec, *, seed: int, target_samples: int
) -> list[float]:  # pragma: no cover - inference path
    """Real co-run re-bench seam for one within-role disjoint pair.

    Unlike the admit-overlap probe (which is role-granular and works over the
    ``force_role`` eval lane), a within-role co-run ratio requires the pair's TWO
    same-role instances to be pinned to their specific CPU regions
    (``region_a`` vs ``region_b``) and co-run — which ``call_orchestrator_forced``
    (``force_role`` only) cannot express. The J5 priors were measured with the
    codified ``bench_canonical`` recipe (``taskset``-pinned, solo + co-run), and a
    ``ratio_delta_vs_prior`` is only valid if these samples use the SAME protocol.

    So the module refuses to fabricate a ratio: the operator loop MUST pass a
    ``sample_fn`` bound to the codified within-role co-run recipe (protocol parity
    with J5) — see ``scripts/lib/canonical_recipe.py`` /
    ``scripts/benchmark/bench_canonical.sh``. The driver loop, sample counting,
    aggregation and artifact write are all implemented and injection-tested.
    """
    raise NotImplementedError(
        "live vision within-role co-run re-bench needs a sample_fn bound to the "
        "codified bench_canonical recipe (taskset-pinned solo+co-run, protocol "
        "parity with the J5 priors so ratio_delta_vs_prior stays valid). "
        "call_orchestrator_forced pins force_role only and cannot co-run two "
        f"same-role instances on {list(pair.region_a)} vs {list(pair.region_b)}. "
        "Pass sample_fn=<codified recipe sampler> to execute_step2_smoke."
    )


def _drive_rebench_pairs(
    plan: Step2SmokePlan,
    *,
    seed: int,
    sample_fn: Callable[..., Iterable[float]] | None = None,
) -> dict[str, list[float]]:
    """Collect co-run ratio samples for every re-bench pair.

    For each pair, ``sample_fn(pair, seed=seed, target_samples=pair.target_samples)``
    returns that pair's ratio samples; results are keyed by ``pair_id`` in the exact
    shape ``aggregate_rebench`` consumes. ``sample_fn`` is injected in tests (same
    pattern as the admit-overlap ``probe_fn``); the default is the codified-recipe
    seam and is never exercised under test.
    """
    sampler = sample_fn or _default_rebench_pair_samples
    samples_by_pair: dict[str, list[float]] = {}
    for pair in plan.rebench_pairs:
        raw = sampler(pair, seed=seed, target_samples=pair.target_samples)
        samples_by_pair[pair.pair_id] = [float(s) for s in (raw or [])]
    return samples_by_pair


def execute_step2_smoke(
    plan: Step2SmokePlan,
    *,
    output_path: Path | None = None,
    seed: int = 42,
    probe_fn: Callable[..., Any] | None = None,
    sample_fn: Callable[..., Iterable[float]] | None = None,
    anchor_hold_fn: Callable[[], dict[str, list[int]]] | None = None,
) -> dict[str, Any]:
    """Drive the plan over the PLACEMENT QUEUE, collect outcomes, aggregate.

    Reached (with ``probe_fn``/``sample_fn``/``anchor_hold_fn`` defaulted to the
    live seams) ONLY when both ``--execute`` and
    ``AUTOPILOT_SHAPEKEYED_STEP2_SMOKE=1`` are set —
    ``run_shapekeyed_step2_smoke`` enforces the double gate. Autopilot-stopped
    assumption: the caller owns the no-concurrent-inference window; this function
    never touches autopilot lifecycle/state and never modifies the routing/serving
    path (the dispatcher + gate are consulted only through the normal eval_batch
    placement lane). The admit-overlap driver first fails closed unless the
    anchor placement is verifiably held (operator anchor-hold procedure).

    Tests exercise this function DIRECTLY with injected ``probe_fn``/``sample_fn``/
    ``anchor_hold_fn`` (never the env gate, never a network) to cover aggregation
    + artifact write; the live-seam defaults stay unread under test.
    """
    observed_decisions = _drive_admit_overlap_probes(
        plan, seed=seed, probe_fn=probe_fn, anchor_hold_fn=anchor_hold_fn
    )
    rebench_samples = _drive_rebench_pairs(plan, seed=seed, sample_fn=sample_fn)
    report = aggregate_smoke(
        plan,
        observed_decisions=observed_decisions,
        rebench_samples=rebench_samples,
    )
    if output_path is not None:
        _write_report(Path(output_path), report)
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
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
    p.add_argument(
        "--expectation",
        choices=(EXPECTATION_REPLACEMENT, EXPECTATION_SEAM),
        default=DEFAULT_EXPECTATION,
        help="admit-overlap expectation model: 'replacement' (default) restates "
        "the expectation against the fleet layer's actual overlap handling "
        "(overlapping candidates are re-placed onto a disjoint instance and "
        "admitted; the invariant is never co-place); 'seam' keeps the original "
        "flag-on overlap->queue model with 2026-08-24-compatible report output",
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
        expectation=args.expectation,
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
