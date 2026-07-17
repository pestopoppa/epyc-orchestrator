"""CP1 — deterministic policy reducer (spec §5.3 / §8).

The reducer is the SOLE enforcement authority of the reviewer plane. A model
never decides enforcement: the reviewer emits a typed *recommendation* + findings
and the verifiers emit criterion-scoped *evidence*; this pure function combines
criterion severity, evidence authority (``authority.py``), logical/execution
status, the reviewer recommendation, cohort calibration, assurance-profile policy,
and operational health into the actual system action (§5.3 deterministic-policy
invariant). It is pure and replayable: identical inputs yield an identical
``PolicyResult`` (§12.4 replay requirement), it reads no clock / RNG / I/O, and its
loops are bounded (§5.6 — max review/evidence rounds terminate).

================================================================================
Authority boundary — THREE distinct enforcement authorities, one per plane
================================================================================
This reducer is ONE of three non-overlapping authorities in the stack. It
SUBSUMES the reviewer-plane precedence that used to live inline in
``review_service.ArchitectReviewService`` (the RD-3 ``fa_candidate``/``fr_candidate``
mechanic); that mechanic is now the ``verifier_precedence_recommendation`` /
``conclusive_verdict`` primitives here and ``review_service`` delegates to them.
The three authorities MUST NOT be conflated:

  1. **PolicyReducer (this module) — reviewer-plane semantics.**
     Decides, for ONE CandidatePackage at a review gate, what the system does with
     a reviewer recommendation + verification evidence (continue / replan / rework /
     defer / escalate / abort / collect-evidence / advisory). ``review_service``
     emits recommendations + findings ONLY; it does not enforce. Enforcement is
     still separately flag-gated (``review_decision_enforce``); until then the
     reducer output is recorded in shadow.

  2. **SafetyGate (``src/safety_gate.py``) — autopilot experiment-admission.**
     Decides whether a Pareto candidate is ADMITTED into the autopilot frontier
     (quality-floor / diversity / per-suite-regression). It governs the evidence
     plane, not a single candidate's review, and it is unchanged by CP1. The
     reducer never admits experiments; SafetyGate never reviews a CandidatePackage.

  3. **Batch manifest fork-tables (``epyc-root`` inference-batch system) —
     batch-execution scope.** Pre-decided forks for inference-gated batch entries,
     executed by the operator's long-horizon loop. Batch-execution routing, not
     reviewer semantics; out of scope for the reducer.

Landed behavioral levers (plan-reminders-over-re-review, ``reject_to_empty``,
the sticky decision cache, complexity gating) are RETAINED and sit ABOVE the
reducer: they shape WHETHER/WHEN a review happens and how iterations are bounded;
the reducer decides the OUTCOME once a review + verification exist.

================================================================================
Schema ownership (CP2 interface points)
================================================================================
The sibling CP2 work owns the canonical on-disk schemas. This module *references*
them by the field names in spec §6 and reads the already-landed artifacts:

  * ``assurance_profile.schema.json`` -> ``AssuranceProfile`` / ``ReducerPolicy``
    (``profile_id, domain, risk_class, criteria{severity,mandatory},
    verifier_registry, policy{unknown_on_critical, reviewer_timeout, schema_error,
    no_reviewer_available, conflict, evidence_budget_exhausted, max_review_rounds,
    max_evidence_rounds, max_reviewer_risk}, calibration_cohort``).
  * ``verification_report.schema.json`` v1.1 check fields
    (``criterion_id, severity, logical_status, execution_status,
    authority{class,valid_for,may_block,may_approve}, scope``) and the v1.1
    summary rollups (``unknown, operational_error, conflicts,
    mandatory_criteria{satisfied,unresolved,failed}``) -> ``VerificationView``.
  * ``review_decision.schema.json`` v1.1 (``decision`` incl. ``abstain``,
    telemetry-only ``raw_model_confidence`` which the reducer IGNORES per §5.4,
    ``blocking.blocking_issues[].{criterion_id,evidence_ref}``) -> ``ReviewView``.
  * ``decision_envelope.schema.json`` ``policy_result.{action,blocking_reason_codes}``
    (action enum matches ``PolicyAction``) + ``governance.policy_hash`` (fed by
    ``ReducerPolicy.policy_hash``) + ``calibration.{cohort_id, upper_risk_bound}``
    -> ``PolicyResult`` / ``CalibrationSnapshot``. CP2 emits the envelope; CP1
    produces the ``policy_result`` payload via ``PolicyResult.to_policy_result_dict``.

The dataclasses below are in-memory adapters/stubs over those artifacts. CP2 owns
the canonical schema; when CP2 exposes typed loaders these ``from_dict`` bridges
should be pointed at them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from src.proactive_delegation.authority import (
    Authority,
    ExecutionStatus,
    LogicalStatus,
    Severity,
    coerce_execution,
    coerce_logical,
    coerce_severity,
    is_conclusive_failure,
    is_conclusive_pass,
    is_operational_error,
    severity_at_least,
)

# ── RD-3 precedence constants (subsumed from review_service) ─────────────────
# Disagreement classes logged as FA/FR *candidates* for the calibration ledger.
FA_CANDIDATE = "fa_candidate"  # reviewer APPROVED but a conclusive gate FAILED
FR_CANDIDATE = "fr_candidate"  # reviewer REJECTED but a conclusive gate PASSED

# RD-8 objective-evidence kinds that make a blocking finding *grounded* (§8 step 5:
# an ungrounded reject is downgraded to advisory). Canonical here; re-exported by
# review_service so its RD-8 admissibility check reads one source of truth.
OBJECTIVE_EVIDENCE_KINDS = frozenset({"gate_result", "test_result", "scorer_result"})


# ══════════════════════════════════════════════════════════════════════════════
# RD-3 shadow-equivalence primitives (pure) — review_service delegates here.
# These reproduce the exact three-valued mechanic that used to live inline in
# ArchitectReviewService so the shadow path stays byte-identical.
# ══════════════════════════════════════════════════════════════════════════════


def conclusive_verdict(report: Mapping[str, Any] | None) -> str:
    """Aggregate a VerificationReport-shaped dict -> ``pass|fail|inconclusive``.

    Precedence over the reviewer applies ONLY to conclusive (pass/fail) verdicts;
    ``inconclusive`` hands control back to the reviewer. Uses
    ``summary.conclusive_verdict`` when it is one of pass/fail/inconclusive, else
    derives from required checks. Byte-identical to the former
    ``ArchitectReviewService._conclusive_verdict``.
    """
    report = report or {}
    summary = report.get("summary") or {}
    v = summary.get("conclusive_verdict")
    if v in ("pass", "fail", "inconclusive"):
        return v
    checks = report.get("checks") or []
    required = [c for c in checks if c.get("required", True)]
    pool = required or checks
    if not pool:
        return "inconclusive"
    outcomes = [c.get("outcome") for c in pool]
    if any(o == "inconclusive" for o in outcomes):
        return "inconclusive"
    if any(o == "fail" for o in outcomes):
        return "fail"
    if outcomes and all(o == "pass" for o in outcomes):
        return "pass"
    return "inconclusive"


def fail_certificates(report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Collect failing checks' certificates — the request_evidence payload.

    Byte-identical to the former ``ArchitectReviewService._fail_certificates``.
    """
    out: list[dict[str, Any]] = []
    for c in (report or {}).get("checks") or []:
        if c.get("outcome") == "fail" and c.get("certificate"):
            out.append(
                {
                    "check_id": c.get("check_id"),
                    "kind": c.get("kind"),
                    "certificate": c.get("certificate"),
                }
            )
    return out


def verifier_precedence_recommendation(
    decision_value: str, verdict: str
) -> tuple[str | None, str | None]:
    """RD-3 mechanical verifier-precedence CORE (pure).

    Given a reviewer ``decision`` string and a conclusive ``verdict``
    (``pass|fail|inconclusive``), return
    ``(adjusted_decision | None, disagreement_class | None)``:

      * reviewer ``approve`` + conclusive ``fail`` -> (``request_evidence``, FA_CANDIDATE)
      * reviewer ``reject``/``reject_to_empty`` + conclusive ``pass``
                                                -> (``request_evidence``, FR_CANDIDATE)
      * ``inconclusive`` verdict, or agreement   -> (None, None)  (reviewer stands)

    ``review_service.apply_verifier_precedence`` composes this with the
    ``dataclasses.replace`` + trace-emission it always did, so its return value +
    emitted event are byte-identical to the pre-refactor behavior.
    """
    if verdict == "inconclusive":
        return None, None
    if verdict == "fail" and decision_value == "approve":
        return "request_evidence", FA_CANDIDATE
    if verdict == "pass" and decision_value in ("reject", "reject_to_empty"):
        return "request_evidence", FR_CANDIDATE
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Versioned policy + profile + calibration (CP2-owned canonical schema; stubs here)
# ══════════════════════════════════════════════════════════════════════════════


def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ReducerPolicy:
    """The VERSIONED policy object (§8: "thresholds and actions belong in versioned
    policy, not hard-coded branches"). Mirrors ``assurance_profile.schema.json``
    ``policy`` block; CP1-local knobs (``block_min_severity``,
    ``operational_error_action``, ``escalation_action``, ``default_action``) default
    conservatively and are covered by ``policy_hash`` so the whole enforcement
    policy is content-addressed for the DecisionEnvelope ``governance.policy_hash``.

    Every terminal action string is one of:
    ``continue | replan | rework | defer | escalate | abort | advisory | abstain |
    fail_closed | fail_open`` (the last three map to envelope actions in
    ``_POLICY_TOKEN_TO_ACTION``). Per §10.2 NO field may be unspecified — hence the
    required ``evidence_budget_exhausted`` default is fail-closed.
    """

    policy_version: str = "1.0.0"
    # CP2 assurance_profile.schema.json policy fields:
    unknown_on_critical: str = "escalate"
    reviewer_timeout: str = "abstain"
    schema_error: str = "abstain"
    no_reviewer_available: str = "defer"
    conflict: str = "escalate"
    evidence_budget_exhausted: str = "fail_closed"  # REQUIRED terminal (§10.2)
    max_review_rounds: int = 2
    max_evidence_rounds: int = 2
    max_reviewer_risk: float | None = None  # absent -> reviewer never authorized
    # CP1-local reducer knobs (documented; not in CP2's schema yet):
    block_min_severity: str = "high"  # §8 step 1 default
    operational_error_action: str = "defer"  # §8 step 3 profile-failure policy
    escalation_action: str = "escalate"  # §8 step 5 abstain/escalate handling
    default_action: str = "defer"  # §8 fallthrough (fail-closed default)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ReducerPolicy":
        data = data or {}
        known = {
            "unknown_on_critical",
            "reviewer_timeout",
            "schema_error",
            "no_reviewer_available",
            "conflict",
            "evidence_budget_exhausted",
            "max_review_rounds",
            "max_evidence_rounds",
            "max_reviewer_risk",
            "policy_version",
            "block_min_severity",
            "operational_error_action",
            "escalation_action",
            "default_action",
        }
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    def _content(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "unknown_on_critical": self.unknown_on_critical,
            "reviewer_timeout": self.reviewer_timeout,
            "schema_error": self.schema_error,
            "no_reviewer_available": self.no_reviewer_available,
            "conflict": self.conflict,
            "evidence_budget_exhausted": self.evidence_budget_exhausted,
            "max_review_rounds": self.max_review_rounds,
            "max_evidence_rounds": self.max_evidence_rounds,
            "max_reviewer_risk": self.max_reviewer_risk,
            "block_min_severity": self.block_min_severity,
            "operational_error_action": self.operational_error_action,
            "escalation_action": self.escalation_action,
            "default_action": self.default_action,
        }

    @property
    def policy_hash(self) -> str:
        """Content address for DecisionEnvelope.governance.policy_hash (§12.3)."""
        return _stable_hash(self._content())


@dataclass(frozen=True)
class CriterionSpec:
    severity: Severity = Severity.MEDIUM
    mandatory: bool = False


@dataclass(frozen=True)
class AssuranceProfile:
    """In-memory view of ``assurance_profile.schema.json`` (CP2 owns the schema).

    Domain-specific parameterization of the domain-general control plane (§16):
    criteria + severity/mandatoriness, the criterion->verifier registry, the
    versioned ``policy``, and the calibration cohort.
    """

    profile_id: str = "default"
    domain: str = "general"
    risk_class: str = "high"
    criteria: Mapping[str, CriterionSpec] = field(default_factory=dict)
    verifier_registry: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    policy: ReducerPolicy = field(default_factory=ReducerPolicy)
    calibration_cohort: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AssuranceProfile":
        data = data or {}
        criteria = {
            cid: CriterionSpec(
                severity=coerce_severity((spec or {}).get("severity"), Severity.MEDIUM),
                mandatory=bool((spec or {}).get("mandatory", False)),
            )
            for cid, spec in (data.get("criteria") or {}).items()
        }
        registry = {
            cid: tuple(str(v) for v in (vs or []))
            for cid, vs in (data.get("verifier_registry") or {}).items()
        }
        return cls(
            profile_id=str(data.get("profile_id", "default")),
            domain=str(data.get("domain", "general")),
            risk_class=str(data.get("risk_class", "high")),
            criteria=criteria,
            verifier_registry=registry,
            policy=ReducerPolicy.from_dict(data.get("policy")),
            calibration_cohort=dict(data.get("calibration_cohort") or {}),
        )

    @property
    def max_reviewer_risk(self) -> float:
        """Threshold the cohort ``upper_risk_bound`` must be <= for blocking authority
        (§8 step 5 / §11.4). Absent in the profile -> 0.0 (reviewer never authorized
        to block; advisory only)."""
        r = self.policy.max_reviewer_risk
        return float(r) if r is not None else 0.0

    @property
    def profile_hash(self) -> str:
        return _stable_hash(
            {
                "profile_id": self.profile_id,
                "domain": self.domain,
                "risk_class": self.risk_class,
                "criteria": {
                    cid: {"severity": s.severity.value, "mandatory": s.mandatory}
                    for cid, s in sorted(self.criteria.items())
                },
                "verifier_registry": {
                    cid: list(v) for cid, v in sorted(self.verifier_registry.items())
                },
                "policy_hash": self.policy.policy_hash,
            }
        )

    @property
    def verifier_registry_hash(self) -> str:
        return _stable_hash(
            {cid: list(v) for cid, v in sorted(self.verifier_registry.items())}
        )


DEFAULT_PROFILE = AssuranceProfile()


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Cohort-scoped EMPIRICAL calibration (§11 / DecisionEnvelope.calibration).

    NOT raw model confidence (§5.4): the reducer reads ``upper_risk_bound`` only.
    Conservative default (upper_risk_bound=1.0) means "no calibration data" -> the
    reviewer is never authorized to block until a real cohort snapshot is supplied.
    """

    cohort_id: str = ""
    sample_count: int = 0
    estimated_error_rate: float = 1.0
    upper_risk_bound: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CalibrationSnapshot":
        data = data or {}
        return cls(
            cohort_id=str(data.get("cohort_id", "")),
            sample_count=int(data.get("sample_count", 0) or 0),
            estimated_error_rate=float(
                data.get("estimated_error_rate", data.get("empirical_error_rate", 1.0)) or 1.0
            ),
            upper_risk_bound=float(data.get("upper_risk_bound", 1.0) or 1.0),
        )


DEFAULT_CALIBRATION = CalibrationSnapshot()


# ══════════════════════════════════════════════════════════════════════════════
# Views over VerificationReport (v1.1) and ReviewDecision
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DerivedCheck:
    criterion_id: str
    severity: Severity
    logical: LogicalStatus
    execution: ExecutionStatus
    authority: Authority
    required: bool
    mandatory: bool


_OUTCOME_TO_LOGICAL = {
    "pass": LogicalStatus.PASS,
    "fail": LogicalStatus.FAIL,
    "inconclusive": LogicalStatus.UNKNOWN,
    "conflict": LogicalStatus.CONFLICT,
}


class VerificationView:
    """Adapter over a VerificationReport dict, resolving criterion severity/authority
    from the report's v1.1 fields (falling back to the profile + v1.0 ``outcome``).

    Exposes exactly the predicates the §8 reducer calls. Summary rollups
    (``mandatory_criteria``, ``conflicts``, ``operational_error``) are trusted when
    present (CP2 computes them); per-check computation is the fallback and the
    authority-aware cross-check.
    """

    def __init__(
        self, report: Mapping[str, Any] | None, profile: AssuranceProfile | None = None
    ):
        self.report: Mapping[str, Any] = report or {}
        self.profile = profile or DEFAULT_PROFILE
        self.summary: Mapping[str, Any] = self.report.get("summary") or {}
        self.checks: list[DerivedCheck] = [
            self._derive(c) for c in (self.report.get("checks") or [])
        ]

    def _derive(self, check: Mapping[str, Any]) -> DerivedCheck:
        cid = str(check.get("criterion_id") or check.get("check_id") or "")
        spec = self.profile.criteria.get(cid)
        sev = check.get("severity")
        if sev is None and spec is not None:
            sev = spec.severity
        severity = coerce_severity(sev, Severity.MEDIUM)
        logical_raw = check.get("logical_status")
        if logical_raw is None:
            logical = _OUTCOME_TO_LOGICAL.get(check.get("outcome"), LogicalStatus.UNKNOWN)
        else:
            logical = coerce_logical(logical_raw)
        execution = coerce_execution(check.get("execution_status", "ok"))
        authority = Authority.from_dict(check.get("authority"))
        mandatory = spec.mandatory if spec is not None else bool(check.get("required", True))
        required = bool(check.get("required", True))
        return DerivedCheck(cid, severity, logical, execution, authority, required, mandatory)

    # ── §8 predicates ────────────────────────────────────────────────────────

    def has_conclusive_failure(
        self,
        *,
        minimum_severity: str | Severity = "high",
        mandatory_only: bool = True,
        policy_grant: bool = False,
    ) -> bool:
        """Any criterion-scoped, execution-clean FAIL whose authority may block, at
        or above ``minimum_severity`` (and mandatory when ``mandatory_only``)."""
        minsev = coerce_severity(minimum_severity, Severity.HIGH)
        for dc in self.checks:
            if mandatory_only and not dc.mandatory:
                continue
            if not severity_at_least(dc.severity, minsev):
                continue
            if is_conclusive_failure(
                dc.authority, dc.logical, dc.execution, policy_grant=policy_grant
            ):
                return True
        # Trust CP2's rollup of mandatory conclusive failures when no check carried it.
        for cid in self.summary.get("mandatory_criteria", {}).get("failed", []) or []:
            spec = self.profile.criteria.get(cid)
            sev = spec.severity if spec is not None else Severity.CRITICAL
            if severity_at_least(sev, minsev):
                return True
        return False

    def has_conflict(self) -> bool:
        """Conflicting conclusive evidence (§7.3 rule 4) — never model-voted away."""
        if int(self.summary.get("conflicts") or 0) > 0:
            return True
        if self.summary.get("conclusive_verdict") == "conflict":
            return True
        if any(dc.logical is LogicalStatus.CONFLICT for dc in self.checks):
            return True
        # Two conclusive checks on the same criterion disagreeing.
        by_crit: dict[str, set[str]] = {}
        for dc in self.checks:
            if dc.execution is not ExecutionStatus.OK:
                continue
            if is_conclusive_pass(dc.authority, dc.logical, dc.execution):
                by_crit.setdefault(dc.criterion_id, set()).add("pass")
            elif is_conclusive_failure(dc.authority, dc.logical, dc.execution):
                by_crit.setdefault(dc.criterion_id, set()).add("fail")
        return any({"pass", "fail"} <= s for s in by_crit.values())

    def has_required_operational_error(self) -> bool:
        """A REQUIRED verifier crashed/timed-out/was unavailable (§7.2/§7.3 rule 5).
        Tool failure is NOT logical failure — the profile's failure policy applies."""
        for dc in self.checks:
            if dc.required and is_operational_error(dc.execution):
                return True
        if not self.checks and int(self.summary.get("operational_error") or 0) > 0:
            return True
        return False

    def has_unknown_critical(self) -> bool:
        """A critical-severity criterion is logically unknown with a clean execution."""
        for dc in self.checks:
            if (
                dc.logical is LogicalStatus.UNKNOWN
                and dc.execution is ExecutionStatus.OK
                and dc.severity is Severity.CRITICAL
            ):
                return True
        for cid in self.summary.get("mandatory_criteria", {}).get("unresolved", []) or []:
            spec = self.profile.criteria.get(cid)
            if spec is not None and spec.severity is Severity.CRITICAL:
                return True
        return False

    def mandatory_criteria_satisfied(self) -> bool:
        """Every mandatory criterion has a conclusive pass and no fail/unknown/conflict.

        Trusts ``summary.mandatory_criteria.satisfied`` when present (CP2's rollup);
        otherwise computes it authority-aware. Vacuously True when the profile
        declares no mandatory criteria."""
        mc = self.summary.get("mandatory_criteria") or {}
        if "satisfied" in mc:
            return bool(mc["satisfied"])
        if self.profile.criteria:
            mandatory = [cid for cid, s in self.profile.criteria.items() if s.mandatory]
        else:
            mandatory = sorted({dc.criterion_id for dc in self.checks if dc.mandatory})
        if not mandatory:
            return True
        for cid in mandatory:
            passed = any(
                dc.criterion_id == cid
                and is_conclusive_pass(dc.authority, dc.logical, dc.execution)
                for dc in self.checks
            )
            if not passed:
                return False
            if any(
                dc.criterion_id == cid
                and dc.logical
                in (LogicalStatus.FAIL, LogicalStatus.UNKNOWN, LogicalStatus.CONFLICT)
                for dc in self.checks
            ):
                return False
        return True


def _normalize_recommendation(value: str) -> str:
    v = (value or "").strip().lower()
    if v in ("reject_to_empty",):
        return "reject"
    return v


@dataclass(frozen=True)
class ReviewView:
    """Adapter over the reviewer's typed recommendation (ArchitectReview, a
    ReviewDecision dict per §6.4, or a raw recommendation string).

    ``recommendation`` is normalized to the §8 set (``reject_to_empty`` folds into
    ``reject``; ``request_changes`` is advisory and falls through the reducer to the
    mandatory-criteria check). Raw model confidence is deliberately NOT surfaced —
    the reducer must never read it (§5.4)."""

    recommendation: str
    requested_evidence: tuple[Any, ...] = ()
    grounded_blocking: bool = False
    raw: Any = None

    @classmethod
    def from_review(cls, review: Any) -> "ReviewView | None":
        if review is None:
            return None
        if isinstance(review, ReviewView):
            return review
        if isinstance(review, str):
            return cls(_normalize_recommendation(review), (), False, review)
        decision = getattr(review, "decision", None)
        if decision is not None and not isinstance(review, Mapping):
            rec = _normalize_recommendation(
                decision.value if hasattr(decision, "value") else str(decision)
            )
            reqs = tuple(getattr(review, "verifier_requests", ()) or ())
            return cls(rec, reqs, _grounded_from_review_obj(review), review)
        if isinstance(review, Mapping):
            rec = _normalize_recommendation(
                str(review.get("recommendation") or review.get("decision") or "")
            )
            reqs = tuple(
                review.get("requested_evidence") or review.get("verifier_requests") or ()
            )
            return cls(rec, reqs, _grounded_from_dict(review), review)
        return cls("", (), False, review)


def _grounded_from_review_obj(review: Any) -> bool:
    """A blocking finding is grounded iff it carries objective evidence (RD-8 kinds)
    or a hard tripwire backed by evidence. Aligns with review_service's
    ``check_reject_admissibility`` so an ungrounded reject -> advisory (§8 step 5)."""
    evidence = getattr(review, "evidence", None) or []
    if any((e or {}).get("kind") in OBJECTIVE_EVIDENCE_KINDS for e in evidence):
        return True
    return False


def _grounded_from_dict(review: Mapping[str, Any]) -> bool:
    evidence = review.get("evidence") or []
    if any((e or {}).get("kind") in OBJECTIVE_EVIDENCE_KINDS for e in evidence):
        return True
    # spec §6.4 blocking_findings[].evidence_refs
    for bf in review.get("blocking_findings") or []:
        if (bf or {}).get("evidence_refs"):
            return True
    # candidate_package/ArchitectReview.to_dict() blocking.blocking_issues[].evidence_ref
    for bi in (review.get("blocking") or {}).get("blocking_issues") or []:
        if (bi or {}).get("evidence_ref"):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PolicyResult + reduce_decision (§8)
# ══════════════════════════════════════════════════════════════════════════════


class PolicyAction(str, Enum):
    """Enforced system action. Values match
    ``decision_envelope.schema.json`` ``policy_result.action`` exactly (CP2 interop)."""

    CONTINUE = "continue"
    REPLAN = "replan"
    REWORK = "rework"
    DEFER = "defer"
    ESCALATE = "escalate"
    ABORT = "abort"
    COLLECT_EVIDENCE = "collect_evidence"  # non-terminal loop-back (§8 evidence branch)
    ADVISORY = "advisory"  # record finding, do not block (§8 step 5 downgrade)


# Terminal policy tokens (profile policy strings) -> envelope-valid actions.
# abstain / fail_closed => DEFER (hold, do not proceed); fail_open => CONTINUE.
_POLICY_TOKEN_TO_ACTION: dict[str, PolicyAction] = {
    "continue": PolicyAction.CONTINUE,
    "replan": PolicyAction.REPLAN,
    "rework": PolicyAction.REWORK,
    "defer": PolicyAction.DEFER,
    "escalate": PolicyAction.ESCALATE,
    "abort": PolicyAction.ABORT,
    "advisory": PolicyAction.ADVISORY,
    "abstain": PolicyAction.DEFER,
    "fail_closed": PolicyAction.DEFER,
    "fail_open": PolicyAction.CONTINUE,
}


@dataclass(frozen=True)
class PolicyResult:
    """The reducer's output — the enforced decision (§8). Frozen + value-equal so a
    replay of identical inputs yields an ``==`` result (§12.4).

    ``action`` is envelope-valid; the original policy token (e.g. ``abstain``,
    ``fail_closed``) is preserved in ``reason_codes`` as ``TERMINAL_<TOKEN>`` so no
    semantics are lost when a token collapses onto a shared action.
    """

    action: PolicyAction
    reason_codes: tuple[str, ...] = ()
    requested_evidence: tuple[Any, ...] = ()
    terminal: bool = True

    @property
    def is_advisory(self) -> bool:
        return self.action is PolicyAction.ADVISORY

    @property
    def blocks(self) -> bool:
        """True when the reducer stops the fast-path (anything but continue/advisory)."""
        return self.action not in (PolicyAction.CONTINUE, PolicyAction.ADVISORY)

    def to_policy_result_dict(self) -> dict[str, Any]:
        """The ``policy_result`` payload CP2 folds into a DecisionEnvelope."""
        return {
            "action": self.action.value,
            "blocking_reason_codes": list(self.reason_codes),
        }

    # ── factories ──
    @classmethod
    def continue_(cls, *reason_codes: str) -> "PolicyResult":
        return cls(PolicyAction.CONTINUE, tuple(reason_codes))

    @classmethod
    def replan(cls, *reason_codes: str) -> "PolicyResult":
        return cls(PolicyAction.REPLAN, tuple(reason_codes))

    @classmethod
    def rework(cls, *reason_codes: str) -> "PolicyResult":
        return cls(PolicyAction.REWORK, tuple(reason_codes))

    @classmethod
    def defer(cls, *reason_codes: str) -> "PolicyResult":
        return cls(PolicyAction.DEFER, tuple(reason_codes))

    @classmethod
    def escalate(cls, *reason_codes: str) -> "PolicyResult":
        return cls(PolicyAction.ESCALATE, tuple(reason_codes))

    @classmethod
    def abort(cls, *reason_codes: str) -> "PolicyResult":
        return cls(PolicyAction.ABORT, tuple(reason_codes))

    @classmethod
    def advisory(cls, *reason_codes: str) -> "PolicyResult":
        return cls(PolicyAction.ADVISORY, tuple(reason_codes))

    @classmethod
    def collect_evidence(
        cls, requested_evidence: Iterable[Any], *reason_codes: str
    ) -> "PolicyResult":
        return cls(
            PolicyAction.COLLECT_EVIDENCE,
            tuple(reason_codes),
            tuple(requested_evidence or ()),
            terminal=False,
        )


@dataclass
class LoopState:
    """Bounded-loop counters (§5.6). ``*_round`` = rounds ALREADY consumed for this
    decision. Passed into ``reduce_decision`` so a runaway review/evidence handshake
    reaches a terminal outcome instead of looping forever."""

    review_round: int = 0
    evidence_round: int = 0


def _terminal(token: str, reason: str, *, default: PolicyAction) -> PolicyResult:
    action = _POLICY_TOKEN_TO_ACTION.get((token or "").strip().lower(), default)
    codes = [reason]
    if token:
        codes.append("TERMINAL_" + token.strip().upper())
    return PolicyResult(action=action, reason_codes=tuple(codes), terminal=True)


def _collect_or_terminal(
    policy: ReducerPolicy,
    requested: Iterable[Any],
    loop_state: LoopState | None,
    *,
    reason: str,
) -> PolicyResult:
    """Return a bounded collect-evidence loop-back, or the profile's terminal action
    once the evidence/review budget is exhausted (§5.6 / §10.2)."""
    if loop_state is not None:
        if loop_state.evidence_round >= policy.max_evidence_rounds:
            return _terminal(
                policy.evidence_budget_exhausted,
                "EVIDENCE_BUDGET_EXHAUSTED",
                default=PolicyAction.DEFER,
            )
        if loop_state.review_round >= policy.max_review_rounds:
            return _terminal(
                policy.evidence_budget_exhausted,
                "REVIEW_BUDGET_EXHAUSTED",
                default=PolicyAction.DEFER,
            )
    return PolicyResult.collect_evidence(requested, reason)


def _as_profile(profile: Any) -> AssuranceProfile:
    if isinstance(profile, AssuranceProfile):
        return profile
    if isinstance(profile, Mapping):
        return AssuranceProfile.from_dict(profile)
    return DEFAULT_PROFILE


def _as_calibration(calibration: Any) -> CalibrationSnapshot:
    if isinstance(calibration, CalibrationSnapshot):
        return calibration
    if isinstance(calibration, Mapping):
        return CalibrationSnapshot.from_dict(calibration)
    return DEFAULT_CALIBRATION


def _as_report(verification: Any) -> Mapping[str, Any]:
    if isinstance(verification, Mapping):
        return verification
    to_report = getattr(verification, "to_report", None)  # e.g. VerdictResult
    if callable(to_report):
        return to_report()
    return {}


def reduce_decision(
    package: Any,
    verification: Any,
    review: Any,
    profile: Any,
    calibration: Any,
    *,
    loop_state: LoopState | None = None,
) -> PolicyResult:
    """Pure deterministic policy reduction (spec §8).

    Combines conclusive criterion-scoped evidence, evidence conflicts, operational
    health, critical unknowns, reviewer recommendation, and cohort calibration into
    one enforced ``PolicyResult``. The model NEVER decides — this function does
    (§5.3). Same inputs -> same output (§12.4). ``loop_state`` bounds the evidence/
    review loops (§5.6); omit it to skip loop bounding (single-shot reduction).

    ``package`` is accepted for signature-completeness + future criterion scoping
    (the CandidatePackage's declared scope); the current reduction is driven by the
    verification evidence + reviewer recommendation, so it is not yet dereferenced.
    """
    prof = _as_profile(profile)
    policy = prof.policy
    vv = (
        verification
        if isinstance(verification, VerificationView)
        else VerificationView(_as_report(verification), prof)
    )
    rv = ReviewView.from_review(review)
    cal = _as_calibration(calibration)

    # 1. Conclusive criterion-scoped failures dominate (§7.3 rule 1).
    if vv.has_conclusive_failure(
        minimum_severity=policy.block_min_severity, mandatory_only=True
    ):
        return PolicyResult.replan("CONCLUSIVE_HIGH_SEVERITY_FAILURE")

    # 2. Evidence conflicts are not silently resolved by a model vote (§7.3 rule 4).
    if vv.has_conflict():
        return _terminal(
            policy.conflict,
            "CONFLICTING_AUTHORITATIVE_EVIDENCE",
            default=PolicyAction.ESCALATE,
        )

    # 3. Operational health follows explicit profile policy (§7.3 rule 5).
    if vv.has_required_operational_error():
        return _terminal(
            policy.operational_error_action,
            "VERIFIER_OPERATIONAL_ERROR",
            default=PolicyAction.DEFER,
        )

    # 4. Critical unknowns require evidence or escalation (§7.3 rule 6).
    if vv.has_unknown_critical():
        if rv is not None and rv.recommendation == "request_evidence":
            return _collect_or_terminal(
                policy, rv.requested_evidence, loop_state, reason="UNKNOWN_ON_CRITICAL"
            )
        return _terminal(
            policy.unknown_on_critical, "UNKNOWN_ON_CRITICAL", default=PolicyAction.ESCALATE
        )

    # 5. Reviewer authority is conditional on calibration (§8 step 5 / §11.4).
    if rv is not None:
        reviewer_is_authorized = cal.upper_risk_bound <= prof.max_reviewer_risk
        rec = rv.recommendation
        if rec == "reject":
            if reviewer_is_authorized and rv.grounded_blocking:
                return PolicyResult.replan("CALIBRATED_REVIEWER_REJECTION")
            return PolicyResult.advisory("UNAUTHORIZED_OR_UNGROUNDED_REJECTION")
        if rec == "request_evidence":
            return _collect_or_terminal(
                policy, rv.requested_evidence, loop_state, reason="REVIEWER_REQUEST_EVIDENCE"
            )
        if rec in ("abstain", "escalate"):
            return _terminal(
                policy.escalation_action,
                "REVIEWER_ABSTAIN" if rec == "abstain" else "REVIEWER_ESCALATE",
                default=PolicyAction.ESCALATE,
            )

    # 6. Mandatory evidence passes; approval can continue (§8 step 6).
    if vv.mandatory_criteria_satisfied():
        return PolicyResult.continue_("MANDATORY_CRITERIA_SATISFIED")

    return _terminal(
        policy.default_action, "MANDATORY_CRITERIA_UNRESOLVED", default=PolicyAction.DEFER
    )


__all__ = [
    # RD-3 subsumed primitives
    "FA_CANDIDATE",
    "FR_CANDIDATE",
    "OBJECTIVE_EVIDENCE_KINDS",
    "conclusive_verdict",
    "fail_certificates",
    "verifier_precedence_recommendation",
    # policy + profile + calibration
    "ReducerPolicy",
    "CriterionSpec",
    "AssuranceProfile",
    "CalibrationSnapshot",
    "DEFAULT_PROFILE",
    "DEFAULT_CALIBRATION",
    # views
    "VerificationView",
    "ReviewView",
    "DerivedCheck",
    # reducer
    "PolicyAction",
    "PolicyResult",
    "LoopState",
    "reduce_decision",
]
