#!/usr/bin/env python3
"""GLM-5.2 single-model reviewer-capability probe runner (P5 GC-1/2/3).

Companion to ``scripts/autopilot/screening_tier_runner.py`` — but where the
screening runner scores reviewer *pairings* (architect x reviewer x grader) for
the H5 tournament, this runner scores a SINGLE reviewer model's *capability* in
isolation. It is the executor the ``coordination/inference-batch`` P5 entries
(``P5-GC1-strict-if-typed-emission`` / ``P5-GC2-rubric-authoring-quality`` /
``P5-GC3-why-diagnosis``) were left un-authored for: their ``command`` blocks say
"single-model probe runner does not exist … must be authored once GLM-5.2 is
admitted (COORD-glm52-admission)". This is that runner.

Three probes, selected by ``--probe``:

  * ``strict_if``          — GC-1. Schema-valid typed ``review_decision`` emission
                             rate, GBNF-constrained vs free-parse-with-retry, scored
                             as a K-of-M pass gate. Motivation: 122B-IQ2 scored 2/11
                             on strict instruction-following; grammar constraint is the
                             expected mitigation, so we measure both lanes.
  * ``rubric_authoring``   — GC-2. GLM-5.2 authors a rubric per task; graded
                             DETERMINISTICALLY against a frontier-authored reference
                             on criteria-count / axis-coverage / grounding.
  * ``why_diagnosis``      — GC-3. Rationale-vs-gold-cause match on a corpus-v1
                             sample; captures the detect-THAT vs detect-WHY gap.

Two responsibilities, cleanly split (mirrors the screening runner):

  1. **Plan resolution + scoring** (pure, inference-free — ALL the tests exercise
     this): parse a probe task set, resolve it into concrete placement-queue job
     specs pinned to the GLM model/quant identity (``request_priority=background`` +
     ``workload_class=eval_batch``, NEVER a foreground ``/chat`` call), and score
     synthetic reviewer outputs deterministically. Results are indexed by
     **model/quant** (``glm_52_ud_iq2m`` / ``UD-IQ2_M``), NEVER by role.

  2. **Execution bridge** (env-flag-gated AND ``--execute``, DEFAULT OFF): with the
     gate closed the resolved plan is printed as a dry-run and NO inference happens.
     With it open, the single-model reviewer is driven over the placement queue and
     each raw output is fed through the SAME pure scorers. The execution path is
     modeled on ``screening_tier_runner._default_reviewer_probe`` /
     ``bsv_paired_runner`` (deferred client import, autopilot-stopped assumption) and
     is intentionally NEVER reached by the tests.

Gating (defense in depth — the runner will NOT touch a model unless all hold):
  * ``--execute`` CLI flag present, AND
  * ``AUTOPILOT_GLM_REVIEWER_CAPABILITY_INFERENCE=1`` (this runner's inference flag), AND
  * ``AUTOPILOT_GLM52_ADMITTED=1`` (the COORD-glm52-admission checkpoint the parallel
    GLM session owns; consumed here, never re-run/re-signed by this runner).
  Any missing => the runner falls back to a pure dry-run and exits 0.

Constraints honored (CLAUDE.md + reviewer-control-plane handoffs):
  * The serving path is FROZEN — this runner imports NOTHING from review_service /
    delegator / backends / features and never edits actions.py / eval_tower.py.
  * The autopilot daemon lifecycle is owned elsewhere; this runner never starts/stops
    it and never writes autopilot_state.json / journals / runtime_flags.json.
  * Every number produced is a pre-P-REV-1 SMOKE **observation** (MEASUREMENT.md);
    it never gates a keep/revert/promote decision on its own.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(SCRIPT_DIR), str(ORCH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNNER_VERSION = "glm-reviewer-capability-probe-v1"

# ── gating ────────────────────────────────────────────────────────────────────
# Two independent env gates PLUS the --execute flag must all be set for inference.
INFERENCE_ENV = "AUTOPILOT_GLM_REVIEWER_CAPABILITY_INFERENCE"
GLM_ADMISSION_ENV = "AUTOPILOT_GLM52_ADMITTED"  # COORD-glm52-admission checkpoint

# ── placement-queue transport constants (RM-3 discipline) ──────────────────────
# Identical to the screening runner: a capability probe rides the SAME
# background/eval_batch placement-queue path a normal autopilot eval fan-out uses,
# and is NEVER a foreground /chat request.
PLACEMENT_QUEUE_TRANSPORT = "placement_queue"
PLACEMENT_REQUEST_PRIORITY = "background"
PLACEMENT_WORKLOAD_CLASS = "eval_batch"

PROBE_MODES = ("strict_if", "rubric_authoring", "why_diagnosis")

# ── the single model under test (model/quant identity, NEVER a role) ───────────
# Mirrors the research registry ``glm_52_ud_iq2m`` entry. Results are indexed by
# these fields so a capability number is attributable to a model+quant, per the
# feedback_model_not_role_indexing discipline.
GLM52_MODEL_IDENTITY: dict[str, str] = {
    "model_key": "glm_52_ud_iq2m",
    "model_name": "GLM-5.2-UD-IQ2_M",
    "quant": "UD-IQ2_M",
    "architecture": "glm_moe_dsa",
}

# ── review_decision schema constants (subset validated here) ───────────────────
_DECISION_ENUM = {
    "approve",
    "reject",
    "reject_to_empty",
    "request_changes",
    "request_evidence",
    "abstain",
    "escalate",
}
_RUBRIC_WEIGHTS = {1, 2, 3}
_RUBRIC_ID_RE = re.compile(r"^R[0-9]+$")
_RUBRIC_AXIS_ALIASES = {
    "question-alignment": {
        "alignment",
        "answer relevance",
        "prompt alignment",
        "question alignment",
        "relevance",
        "task alignment",
    },
    "grounding": {
        "accuracy",
        "correctness",
        "evidence",
        "evidence grounding",
        "factuality",
        "source grounding",
    },
    "integrity": {
        "faithfulness",
        "hallucination control",
        "honesty",
        "integrity",
        "non hallucination",
        "truthfulness",
    },
    "completeness": {
        "coverage",
        "completeness",
        "completeness coverage",
        "sufficiency",
    },
}
_RUBRIC_AXIS_CANONICAL = {
    re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip(): canonical
    for canonical, aliases in _RUBRIC_AXIS_ALIASES.items()
    for alias in (aliases | {canonical})
}
# defect-detection keywords for the why-diagnosis "detect-THAT" signal.
_DEFECT_KEYWORDS = (
    "reject",
    "defect",
    "bug",
    "incorrect",
    "wrong",
    "error",
    "flaw",
    "fails",
    "failure",
    "mistake",
)

# Default K-of-M gate for strict_if when the caller does not pin one.
DEFAULT_STRICT_IF_M = 11
DEFAULT_STRICT_IF_K = 8


def _env_flag_enabled(name: str) -> bool:
    """True iff env var ``name`` is a truthy flag (matches actions._env_flag_enabled)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# strict_if (GC-1) — typed review_decision emission, GBNF vs free-parse
# ══════════════════════════════════════════════════════════════════════════════


def parse_typed_emission(text: str, *, grammar_constrained: bool) -> dict[str, Any] | None:
    """Parse a reviewer output string into a candidate ``review_decision`` object.

    ``grammar_constrained`` (GBNF lane): the whole output MUST be valid JSON — a
    GBNF grammar emits nothing but the object, so any surrounding prose is a
    failure. ``free-parse-with-retry`` lane: try whole-string JSON first, then a
    single retry that carves out the first ``{`` .. last ``}`` span (the common
    "here is the decision: {…}" wrapper). Returns the dict or ``None``; never raises.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    if grammar_constrained:
        return None  # GBNF lane does not get the prose-stripping retry
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def validate_review_decision(obj: Any) -> tuple[bool, list[str]]:
    """Validate the required core of a ``review_decision`` (schema subset).

    Enforces exactly the invariants a downstream policy reducer relies on:
    ``decision`` present + in enum, ``confidence`` a number in [0,1], and a
    ``blocking`` object carrying a boolean ``tripwire`` (the structurally-separate
    hard-stop channel). Returns ``(is_valid, errors)``.
    """
    errors: list[str] = []
    if not isinstance(obj, dict):
        return False, ["not_an_object"]
    decision = obj.get("decision")
    if decision is None:
        errors.append("missing:decision")
    elif decision not in _DECISION_ENUM:
        errors.append(f"decision_not_in_enum:{decision!r}")
    conf = obj.get("confidence")
    if conf is None:
        errors.append("missing:confidence")
    elif isinstance(conf, bool) or not isinstance(conf, (int, float)):
        errors.append("confidence_not_a_number")
    elif not (0.0 <= float(conf) <= 1.0):
        errors.append("confidence_out_of_range")
    blocking = obj.get("blocking")
    if blocking is None:
        errors.append("missing:blocking")
    elif not isinstance(blocking, dict):
        errors.append("blocking_not_an_object")
    elif not isinstance(blocking.get("tripwire"), bool):
        errors.append("missing_or_nonbool:blocking.tripwire")
    return (not errors), errors


def score_strict_if_item(text: str, *, grammar_constrained: bool) -> dict[str, Any]:
    """Score ONE strict_if output: parse -> validate. Pure, deterministic."""
    parsed = parse_typed_emission(text, grammar_constrained=grammar_constrained)
    valid, errors = validate_review_decision(parsed)
    return {
        "parsed": parsed is not None,
        "schema_valid": valid,
        "errors": errors,
        "decision": (parsed or {}).get("decision") if parsed else None,
    }


def score_strict_if(
    outputs: list[str],
    *,
    grammar_constrained: bool,
    k: int | None = None,
    m: int | None = None,
) -> dict[str, Any]:
    """Score a strict_if output batch into a K-of-M schema-valid emission gate.

    ``m`` defaults to ``len(outputs)`` (the batch size actually seen); ``k`` to
    :data:`DEFAULT_STRICT_IF_K`. ``passed`` is ``n_valid >= k`` — the K-of-M floor.
    ``emission_rate`` is ``n_valid / m``. Pure; asserts nothing about a model.
    """
    m_eff = int(m) if m is not None else len(outputs)
    k_eff = int(k) if k is not None else DEFAULT_STRICT_IF_K
    items = [score_strict_if_item(o, grammar_constrained=grammar_constrained) for o in outputs]
    n_valid = sum(1 for it in items if it["schema_valid"])
    n_parsed = sum(1 for it in items if it["parsed"])
    emission_rate = (n_valid / m_eff) if m_eff else 0.0
    return {
        "probe": "strict_if",
        "grammar_constrained": bool(grammar_constrained),
        "n": len(outputs),
        "m": m_eff,
        "k": k_eff,
        "n_parsed": n_parsed,
        "n_valid": n_valid,
        "emission_rate": emission_rate,
        "passed": n_valid >= k_eff,
        "per_item": items,
    }


# ══════════════════════════════════════════════════════════════════════════════
# rubric_authoring (GC-2) — authored rubric vs frontier reference
# ══════════════════════════════════════════════════════════════════════════════


def _valid_rubric_items(rubric: Any) -> list[dict[str, Any]]:
    """Return the schema-valid ``items`` of a rubric object (id/text/axis/weight)."""
    if not isinstance(rubric, dict):
        return []
    out: list[dict[str, Any]] = []
    for it in rubric.get("items") or []:
        if not isinstance(it, dict):
            continue
        rid = it.get("id")
        text = it.get("text")
        axis = it.get("axis")
        weight = it.get("weight")
        if not (isinstance(rid, str) and _RUBRIC_ID_RE.match(rid)):
            continue
        if not (isinstance(text, str) and text.strip()):
            continue
        if not (isinstance(axis, str) and axis.strip()):
            continue
        if isinstance(weight, bool) or weight not in _RUBRIC_WEIGHTS:
            continue
        out.append(it)
    return out


def _canonical_rubric_axis(axis: Any) -> str:
    """Normalize equivalent rubric-axis labels for deterministic scoring.

    GLM repair smokes showed that schema-valid rubrics often use conventional
    review-axis synonyms (for example ``accuracy`` for ``grounding``). The scorer
    should measure semantic axis coverage, not exact wording drift.
    """
    norm = _normalize(str(axis))
    return _RUBRIC_AXIS_CANONICAL.get(norm, norm.replace(" ", "-"))


def score_rubric_authoring(
    authored: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Grade an authored rubric against a frontier reference (deterministic).

    Three sub-scores in [0,1], each concrete:
      * ``count_ratio``   = min(1, |valid authored items| / |reference items|)
      * ``axis_coverage`` = |authored axes ∩ reference axes| / |reference axes|
      * ``grounding_rate``= fraction of valid authored items phrased as a checkable
                            question (``text`` ends with '?'), the schema's grounding
                            convention ("phrased as a checkable question").
    ``composite`` is their unweighted mean. Pure.
    """
    a_items = _valid_rubric_items(authored)
    r_items = _valid_rubric_items(reference)
    a_axes = {_canonical_rubric_axis(it["axis"]) for it in a_items}
    r_axes = {_canonical_rubric_axis(it["axis"]) for it in r_items}

    ref_count = len(r_items)
    count_ratio = min(1.0, len(a_items) / ref_count) if ref_count else 0.0
    axis_coverage = (len(a_axes & r_axes) / len(r_axes)) if r_axes else 0.0
    grounded = sum(1 for it in a_items if str(it["text"]).strip().endswith("?"))
    grounding_rate = (grounded / len(a_items)) if a_items else 0.0
    composite = (count_ratio + axis_coverage + grounding_rate) / 3.0

    return {
        "probe": "rubric_authoring",
        "criteria_count": len(a_items),
        "reference_criteria_count": ref_count,
        "count_ratio": count_ratio,
        "axis_coverage": axis_coverage,
        "axes_covered": sorted(a_axes & r_axes),
        "reference_axes": sorted(r_axes),
        "grounding_rate": grounding_rate,
        "composite": composite,
    }


# ══════════════════════════════════════════════════════════════════════════════
# why_diagnosis (GC-3) — rationale vs gold cause (detect-THAT vs detect-WHY)
# ══════════════════════════════════════════════════════════════════════════════


def _normalize(text: str) -> str:
    """Lowercase, collapse non-alphanumerics to single spaces (token-match ready)."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def detect_that(rationale: str) -> bool:
    """True iff the rationale asserts that SOME defect exists (detect-THAT signal)."""
    norm = _normalize(rationale)
    return any(kw in norm for kw in _DEFECT_KEYWORDS)


def cause_matched(rationale: str, gold_aliases: Iterable[str]) -> bool:
    """True iff the rationale names the gold failure CAUSE (detect-WHY signal).

    A gold cause is given as one or more alias phrases; a match is a normalized
    substring hit for ANY alias (so ``"off-by-one"`` matches ``off by one``).
    """
    norm = _normalize(rationale)
    for alias in gold_aliases or []:
        a = _normalize(alias)
        if a and a in norm:
            return True
    return False


def score_why_diagnosis_item(rationale: str, gold_aliases: Iterable[str]) -> dict[str, Any]:
    """Score ONE why-diagnosis output. Pure, deterministic."""
    return {
        "that_detected": detect_that(rationale),
        "why_matched": cause_matched(rationale, gold_aliases),
    }


def score_why_diagnosis(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scored why-diagnosis items into that-rate / why-rate (the gap)."""
    n = len(items)
    that = sum(1 for it in items if it.get("that_detected"))
    why = sum(1 for it in items if it.get("why_matched"))
    return {
        "probe": "why_diagnosis",
        "n": n,
        "n_that_detected": that,
        "n_why_matched": why,
        "that_detection_rate": (that / n) if n else 0.0,
        "why_match_rate": (why / n) if n else 0.0,
        # the whole point of GC-3: detect-THAT is easy, detect-WHY is the gap.
        "that_minus_why_gap": ((that - why) / n) if n else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Task-set parsing + validation (pure)
# ══════════════════════════════════════════════════════════════════════════════


def parse_task_set(raw: dict[str, Any], *, probe: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate + normalize a probe task set. Returns ``(tasks, errors)``.

    Per-probe required fields on each task:
      * strict_if        — ``task_id``, ``prompt``
      * rubric_authoring — ``task_id``, ``prompt``, ``reference_rubric`` (with items)
      * why_diagnosis    — ``task_id``, ``prompt``, ``gold_cause_aliases`` (non-empty)
    """
    if probe not in PROBE_MODES:
        return [], [f"unknown_probe:{probe!r}"]
    if not isinstance(raw, dict):
        return [], ["task_set_not_an_object"]
    tasks_in = raw.get("tasks")
    if not isinstance(tasks_in, list) or not tasks_in:
        return [], ["task_set_missing_tasks"]

    errors: list[str] = []
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, t in enumerate(tasks_in):
        if not isinstance(t, dict):
            errors.append(f"task[{i}]:not_an_object")
            continue
        tid = t.get("task_id")
        if not (isinstance(tid, str) and tid):
            errors.append(f"task[{i}]:missing_task_id")
            continue
        if tid in seen:
            errors.append(f"task[{i}]:duplicate_task_id:{tid}")
            continue
        prompt = t.get("prompt")
        if not (isinstance(prompt, str) and prompt.strip()):
            errors.append(f"task[{tid}]:missing_prompt")
            continue
        if probe == "rubric_authoring":
            ref = t.get("reference_rubric")
            if not _valid_rubric_items(ref):
                errors.append(f"task[{tid}]:missing_or_empty_reference_rubric")
                continue
        elif probe == "why_diagnosis":
            aliases = t.get("gold_cause_aliases")
            if not (isinstance(aliases, list) and any(str(a).strip() for a in aliases)):
                errors.append(f"task[{tid}]:missing_gold_cause_aliases")
                continue
        seen.add(tid)
        tasks.append(dict(t))
    return tasks, errors


def _builtin_task_set(probe: str, *, m: int) -> dict[str, Any]:
    """A tiny self-contained task set so a dry-run works with NO input file.

    Deliberately synthetic — real probe corpora come via ``--task-set`` (GC-1 fixed
    probe set / GC-2 frontier references / GC-3 nearmiss-v1 sample). This only lets
    the default invocation validate+resolve+print a plan with zero arguments.
    """
    if probe == "strict_if":
        return {
            "probe": "strict_if",
            "tasks": [
                {
                    "task_id": f"si-{i:02d}",
                    "prompt": (
                        "Review the CANDIDATE and emit a single JSON review_decision "
                        "object with fields decision, confidence, blocking.tripwire."
                    ),
                }
                for i in range(m)
            ],
        }
    if probe == "rubric_authoring":
        ref = {
            "items": [
                {"id": "R1", "text": "Does the answer address the question?", "axis": "question-alignment", "weight": 3},
                {"id": "R2", "text": "Is every claim grounded in the source?", "axis": "grounding", "weight": 3},
                {"id": "R3", "text": "Are there fabricated facts?", "axis": "integrity", "weight": 2},
                {"id": "R4", "text": "Is the answer complete?", "axis": "completeness", "weight": 1},
            ]
        }
        return {
            "probe": "rubric_authoring",
            "tasks": [
                {
                    "task_id": f"ra-{i:02d}",
                    "prompt": "Author a review rubric for the following QA task.",
                    "reference_rubric": ref,
                }
                for i in range(3)
            ],
        }
    # why_diagnosis
    return {
        "probe": "why_diagnosis",
        "tasks": [
            {
                "task_id": "wd-00",
                "prompt": (
                    "CANDIDATE: In binary_search, the branch for arr[mid] < target "
                    "sets left = mid. Symptom: with adjacent bounds, mid repeats and "
                    "the loop never advances when the target is larger than arr[mid]."
                ),
                "gold_cause_aliases": [
                    "off-by-one",
                    "missing mid plus one",
                    "left equals mid",
                    "left mid",
                    "incorrect midpoint update",
                ],
            },
            {
                "task_id": "wd-01",
                "prompt": (
                    "CANDIDATE: render_user(user) returns user.name.upper() without "
                    "checking user. Symptom: AttributeError/NoneType crash when the "
                    "caller passes user=None."
                ),
                "gold_cause_aliases": [
                    "null dereference",
                    "none dereference",
                    "missing null check",
                    "missing none guard",
                    "attribute access on unvalidated user",
                ],
            },
            {
                "task_id": "wd-02",
                "prompt": (
                    "CANDIDATE: distance(dx, dy) computes sqrt(dx * dx - dy * dy). "
                    "Symptom: valid inputs can produce a negative value under sqrt "
                    "and the distance is mathematically wrong."
                ),
                "gold_cause_aliases": [
                    "wrong formula",
                    "subtraction instead of addition",
                    "sign error",
                    "wrong arithmetic operation",
                    "subtracting dy dy",
                ],
            }
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Job spec + resolved plan dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CapabilityJobSpec:
    """One concrete single-model capability trial (a probe task for GLM-5.2).

    Model/quant-indexed — carries ``model_key``/``model_name``/``quant`` and NO role
    field, so a result attributes to a model+quant, never a role. The transport
    fields pin the placement-queue path (never /chat).
    """

    probe: str
    task_id: str
    prompt_sha256: str
    model_key: str
    model_name: str
    quant: str
    architecture: str
    grammar_constrained: bool | None
    per_task_n: int
    transport: str = PLACEMENT_QUEUE_TRANSPORT
    request_priority: str = PLACEMENT_REQUEST_PRIORITY
    workload_class: str = PLACEMENT_WORKLOAD_CLASS

    def target_binding(self) -> dict[str, Any]:
        """Placement-queue routing binding: pin the MODEL (by key), never a role."""
        return {"force_model": self.model_key, "workload_class": self.workload_class}

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["kind"] = "glm_capability_job"
        d["target_binding"] = self.target_binding()
        return d


@dataclass
class ResolvedCapabilityProbe:
    """The concrete, placement-queue-dispatched capability probe (dry-run plan)."""

    probe: str
    model_identity: dict[str, str]
    jobs: list[CapabilityJobSpec]
    grammar_constrained: bool | None
    scoring_config: dict[str, Any]
    n_tasks: int
    parse_errors: list[str]
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
            "kind": "resolved_glm_capability_probe",
            "runner_version": RUNNER_VERSION,
            "probe": self.probe,
            "model": dict(self.model_identity),
            "grammar_constrained": self.grammar_constrained,
            "scoring_config": dict(self.scoring_config),
            "n_tasks": self.n_tasks,
            "n_jobs": len(self.jobs),
            "parse_errors": list(self.parse_errors),
            "transport": self.transport_summary(),
            "jobs": [j.to_dict() for j in self.jobs],
            "provenance": dict(self.provenance),
            "notes": list(self.notes),
            "inference_required": self.inference_required,
        }


def resolve_capability_probe(
    task_set: dict[str, Any],
    *,
    probe: str,
    model_identity: dict[str, str] | None = None,
    grammar_constrained: bool | None = None,
    per_task_n: int = 1,
    k: int | None = None,
    m: int | None = None,
) -> ResolvedCapabilityProbe:
    """Parse + resolve a task set into a concrete placement-queue probe plan (pure).

    Steps: validate probe -> parse/validate task set -> materialize one
    :class:`CapabilityJobSpec` per task, pinned to the GLM model/quant identity and
    the placement-queue transport. No inference, no I/O. For ``strict_if`` the K-of-M
    gate is recorded in ``scoring_config`` (``m`` defaults to the task count).
    """
    if probe not in PROBE_MODES:
        raise ValueError(f"unknown probe {probe!r}; must be one of {PROBE_MODES}")
    ident = dict(model_identity or GLM52_MODEL_IDENTITY)
    gc = grammar_constrained if probe == "strict_if" else None

    tasks, parse_errors = parse_task_set(task_set, probe=probe)

    jobs: list[CapabilityJobSpec] = []
    for t in tasks:
        jobs.append(
            CapabilityJobSpec(
                probe=probe,
                task_id=str(t["task_id"]),
                prompt_sha256=_sha256(str(t.get("prompt", ""))),
                model_key=str(ident.get("model_key", "")),
                model_name=str(ident.get("model_name", "")),
                quant=str(ident.get("quant", "")),
                architecture=str(ident.get("architecture", "")),
                grammar_constrained=gc,
                per_task_n=int(per_task_n),
            )
        )

    scoring_config: dict[str, Any] = {"per_task_n": int(per_task_n)}
    if probe == "strict_if":
        m_eff = int(m) if m is not None else len(jobs)
        scoring_config.update(
            {
                "gate": "K_of_M",
                "k": int(k) if k is not None else DEFAULT_STRICT_IF_K,
                "m": m_eff,
                "grammar_constrained": bool(gc),
            }
        )
    elif probe == "rubric_authoring":
        scoring_config.update({"axes": ["count_ratio", "axis_coverage", "grounding_rate"]})
    else:  # why_diagnosis
        scoring_config.update({"axes": ["that_detection_rate", "why_match_rate"]})

    provenance = {
        "task_set_probe": task_set.get("probe") if isinstance(task_set, dict) else None,
        "task_set_id": task_set.get("task_set_id") if isinstance(task_set, dict) else None,
        "source_handoff": "glm52-reviewer-capability-gates.md",
        "admission_gate": GLM_ADMISSION_ENV,
        "coordination_task_ids": {
            "strict_if": "P5-GC1-strict-if-typed-emission",
            "rubric_authoring": "P5-GC2-rubric-authoring-quality",
            "why_diagnosis": "P5-GC3-why-diagnosis",
        }[probe],
    }

    notes = [
        "single-model capability probe (NOT a pairing screen); model/quant-indexed.",
        "resolved into placement-queue job specs; NEVER /chat (RM-3 discipline).",
        "all scores are pre-P-REV-1 SMOKE observations, not decision-gating numbers "
        "(MEASUREMENT.md).",
    ]
    if parse_errors:
        notes.append(f"{len(parse_errors)} task(s) rejected during parse; see parse_errors.")

    return ResolvedCapabilityProbe(
        probe=probe,
        model_identity=ident,
        jobs=jobs,
        grammar_constrained=gc,
        scoring_config=scoring_config,
        n_tasks=len(tasks),
        parse_errors=parse_errors,
        provenance=provenance,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pure scoring dispatcher (the "stub the probe" seam — NO inference)
# ══════════════════════════════════════════════════════════════════════════════


def score_probe(
    probe: str,
    tasks: list[dict[str, Any]],
    outputs: list[Any],
    *,
    grammar_constrained: bool = True,
    k: int | None = None,
    m: int | None = None,
    model_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score a probe over synthetic reviewer ``outputs`` (aligned to ``tasks``).

    This is the pure seam the tests drive: pass raw outputs (a stubbed probe's
    return values) and get a model/quant-indexed summary back with NO inference.
    ``outputs[i]`` corresponds to ``tasks[i]``. Shapes:
      * strict_if        — ``outputs[i]`` is the raw emission string.
      * rubric_authoring — ``outputs[i]`` is the authored rubric dict (or JSON str).
      * why_diagnosis    — ``outputs[i]`` is the rationale string.
    """
    ident = dict(model_identity or GLM52_MODEL_IDENTITY)
    base = {
        "kind": "glm_capability_result",
        "runner_version": RUNNER_VERSION,
        "probe": probe,
        # model/quant indexed — NEVER role
        "model_key": ident.get("model_key"),
        "model_name": ident.get("model_name"),
        "quant": ident.get("quant"),
        "architecture": ident.get("architecture"),
        "n_tasks": len(tasks),
        "observation_only": True,  # pre-P-REV-1 (MEASUREMENT.md)
    }

    if probe == "strict_if":
        summary = score_strict_if(
            [str(o) for o in outputs], grammar_constrained=grammar_constrained, k=k, m=m
        )
        base.update(summary)
        return base

    if probe == "rubric_authoring":
        per_task: list[dict[str, Any]] = []
        for t, out in zip(tasks, outputs):
            authored = out if isinstance(out, dict) else _coerce_json(out)
            row = score_rubric_authoring(authored or {}, t.get("reference_rubric") or {})
            row["task_id"] = t.get("task_id")
            per_task.append(row)
        base.update(_aggregate_rubric(per_task))
        base["per_task"] = per_task
        return base

    if probe == "why_diagnosis":
        scored: list[dict[str, Any]] = []
        per_task = []
        for t, out in zip(tasks, outputs):
            item = score_why_diagnosis_item(str(out), t.get("gold_cause_aliases") or [])
            item["task_id"] = t.get("task_id")
            scored.append(item)
            per_task.append(item)
        base.update(score_why_diagnosis(scored))
        base["n_tasks"] = len(tasks)
        base["per_task"] = per_task
        return base

    raise ValueError(f"unknown probe {probe!r}")


def _coerce_json(out: Any) -> dict[str, Any] | None:
    if isinstance(out, dict):
        return out
    if isinstance(out, str):
        try:
            obj = json.loads(out)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _aggregate_rubric(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_task)
    if not n:
        return {
            "mean_count_ratio": 0.0,
            "mean_axis_coverage": 0.0,
            "mean_grounding_rate": 0.0,
            "mean_composite": 0.0,
        }
    return {
        "mean_count_ratio": sum(r["count_ratio"] for r in per_task) / n,
        "mean_axis_coverage": sum(r["axis_coverage"] for r in per_task) / n,
        "mean_grounding_rate": sum(r["grounding_rate"] for r in per_task) / n,
        "mean_composite": sum(r["composite"] for r in per_task) / n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Execution bridge (env+flag gated; deferred client import; NEVER run in tests)
# ══════════════════════════════════════════════════════════════════════════════


def _default_capability_probe(
    job: CapabilityJobSpec,
    task: dict[str, Any],
) -> Any:  # pragma: no cover - inference path
    """Send ONE probe prompt for the single model over the placement queue.

    Reuses the SAME transport eval_tower / screening_tier_runner use internally —
    ``call_orchestrator_forced`` with ``request_priority=background`` and
    ``workload_class=eval_batch`` — pinning the MODEL (by key), so a capability
    probe is never a foreground /chat request. Returns the raw model output for the
    pure scorers. Never exercised by the tests (whole bridge is gated OFF).
    """
    _research = Path("/mnt/raid0/llm/epyc-inference-research")
    _bench = str(_research / "scripts" / "benchmark")
    if _bench not in sys.path:
        sys.path.insert(0, _bench)
    from seeding_orchestrator import call_orchestrator_forced  # type: ignore

    resp = call_orchestrator_forced(
        prompt=str(task.get("prompt") or ""),
        force_role="",
        force_mode="",
        force_model=job.model_key,
        request_priority=PLACEMENT_REQUEST_PRIORITY,
        workload_class=PLACEMENT_WORKLOAD_CLASS,
    )
    return resp.get("answer") if isinstance(resp, dict) else resp


def execute_capability_probe(
    resolved: ResolvedCapabilityProbe,
    tasks: list[dict[str, Any]],
    *,
    grammar_constrained: bool = True,
    k: int | None = None,
    m: int | None = None,
    probe_fn: Callable[[CapabilityJobSpec, dict[str, Any]], Any] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:  # pragma: no cover - inference path
    """Drive the resolved plan over the placement queue, then pure-score the outputs.

    Reached ONLY when the gate is fully open (see :func:`run_capability_probe`). The
    caller owns the no-concurrent-inference window (bsv/autopilot-stopped pattern);
    this function never touches autopilot lifecycle/state. Emits ONE model/quant-
    indexed JSONL summary to ``output_path`` (append). Does NOT write any ledger.
    """
    probe = probe_fn or _default_capability_probe
    tasks_by_id = {str(t["task_id"]): t for t in tasks}
    outputs: list[Any] = []
    ordered_tasks: list[dict[str, Any]] = []
    for job in resolved.jobs:
        t = tasks_by_id.get(job.task_id, {})
        ordered_tasks.append(t)
        outputs.append(probe(job, t))

    summary = score_probe(
        resolved.probe,
        ordered_tasks,
        outputs,
        grammar_constrained=grammar_constrained,
        k=k,
        m=m,
        model_identity=resolved.model_identity,
    )
    if output_path is not None:
        _append_jsonl(Path(output_path), summary)
    return summary


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level orchestration (dry-run by default; execute triple-gated)
# ══════════════════════════════════════════════════════════════════════════════


def execution_gate_status(*, execute_flag: bool) -> dict[str, Any]:
    """Report the three execution gates. Execution needs ALL three True."""
    inf = _env_flag_enabled(INFERENCE_ENV)
    admit = _env_flag_enabled(GLM_ADMISSION_ENV)
    return {
        "execute_flag": bool(execute_flag),
        "inference_env": inf,
        "inference_env_name": INFERENCE_ENV,
        "admission_env": admit,
        "admission_env_name": GLM_ADMISSION_ENV,
        "open": bool(execute_flag) and inf and admit,
    }


def run_capability_probe(
    task_set: dict[str, Any],
    *,
    probe: str,
    execute: bool = False,
    grammar_constrained: bool | None = None,
    per_task_n: int = 1,
    k: int | None = None,
    m: int | None = None,
    model_identity: dict[str, str] | None = None,
    output_path: Path | None = None,
    probe_fn: Callable[[CapabilityJobSpec, dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Resolve the probe plan, then dry-run OR execute depending on the gate.

    DEFAULT (``execute=False`` OR either env gate closed): returns the resolved plan
    as a dry-run and runs NO inference — the ENTIRE surface the tests exercise. When
    all three gates are open the plan is driven over the placement queue via
    :func:`execute_capability_probe` (never in tests).
    """
    gc = grammar_constrained if grammar_constrained is not None else (probe == "strict_if")
    resolved = resolve_capability_probe(
        task_set,
        probe=probe,
        model_identity=model_identity,
        grammar_constrained=gc,
        per_task_n=per_task_n,
        k=k,
        m=m,
    )
    gate = execution_gate_status(execute_flag=execute)

    if not gate["open"]:
        reason_bits = []
        if not gate["execute_flag"]:
            reason_bits.append("--execute not passed")
        if not gate["inference_env"]:
            reason_bits.append(f"{INFERENCE_ENV} not set")
        if not gate["admission_env"]:
            reason_bits.append(f"{GLM_ADMISSION_ENV} not set (COORD-glm52-admission)")
        return {
            "mode": "dry_run",
            "runner_version": RUNNER_VERSION,
            "inference_ran": False,
            "gate": gate,
            "reason": "; ".join(reason_bits)
            + " — resolved plan returned as a dry-run (no inference, RM-3 placement-"
            "queue transport).",
            "n_jobs": len(resolved.jobs),
            "resolved_plan": resolved.to_dict(),
        }

    tasks, _ = parse_task_set(task_set, probe=probe)
    summary = execute_capability_probe(
        resolved,
        tasks,
        grammar_constrained=bool(gc),
        k=k,
        m=m,
        probe_fn=probe_fn,
        output_path=output_path,
    )
    return {
        "mode": "execute",
        "runner_version": RUNNER_VERSION,
        "inference_ran": True,
        "gate": gate,
        "n_jobs": len(resolved.jobs),
        "output_path": str(output_path) if output_path else None,
        "resolved_plan": resolved.to_dict(),
        "result": summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI (__main__)
# ══════════════════════════════════════════════════════════════════════════════


def _load_task_set(path: Path | None, *, probe: str, m: int) -> dict[str, Any]:
    if path is None:
        return _builtin_task_set(probe, m=m)
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "GLM-5.2 single-model reviewer-capability probe (P5 GC-1/2/3). Default is "
            "a pure dry-run that validates + resolves + prints the placement-queue plan "
            "and runs NO inference. Execution is triple-gated: --execute + "
            f"{INFERENCE_ENV}=1 + {GLM_ADMISSION_ENV}=1."
        )
    )
    p.add_argument("--probe", required=True, choices=PROBE_MODES, help="which capability to probe")
    p.add_argument("--task-set", default=None, help="path to a probe task-set JSON (default: builtin synthetic)")
    p.add_argument("--per-task-n", type=int, default=1, help="samples per task (execute path)")
    p.add_argument("--k", type=int, default=None, help="strict_if K-of-M pass floor (default 8)")
    p.add_argument("--m", type=int, default=None, help="strict_if K-of-M denominator (default: task count)")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--grammar", dest="grammar", action="store_true", help="strict_if: GBNF-constrained lane (default)")
    grp.add_argument("--no-grammar", dest="grammar", action="store_false", help="strict_if: free-parse-with-retry lane")
    p.set_defaults(grammar=None)
    p.add_argument("--model-key", default=GLM52_MODEL_IDENTITY["model_key"], help="model/quant registry key under test")
    p.add_argument("--output", default=None, help="JSONL path for the model/quant-indexed result (execute path only)")
    p.add_argument(
        "--execute",
        action="store_true",
        help="attempt execution (STILL gated by "
        f"{INFERENCE_ENV}=1 AND {GLM_ADMISSION_ENV}=1; otherwise falls back to dry-run)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    m_for_builtin = args.m if args.m is not None else DEFAULT_STRICT_IF_M
    try:
        task_set = _load_task_set(
            Path(args.task_set) if args.task_set else None,
            probe=args.probe,
            m=m_for_builtin,
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": f"failed to load task set: {exc}"}, indent=2))
        return 2

    ident = dict(GLM52_MODEL_IDENTITY)
    ident["model_key"] = args.model_key

    grammar_constrained = args.grammar if args.probe == "strict_if" else None

    result = run_capability_probe(
        task_set,
        probe=args.probe,
        execute=args.execute,
        grammar_constrained=grammar_constrained,
        per_task_n=args.per_task_n,
        k=args.k,
        m=args.m,
        model_identity=ident,
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
