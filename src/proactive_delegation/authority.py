"""CP1 — criterion-scoped evidence-authority model (spec §5.2 / §7).

The atom of the deterministic control plane: *authority is scoped to a criterion,
a coverage, a set of assumptions, and a direction of implication* — never global
merely because a signal is labelled "objective" (§5.2 criterion-scoped-authority
invariant). This module encodes:

  * the nine authority classes and their approve/block grants (§7.1 table);
  * ``logical_status`` × ``execution_status`` kept STRICTLY SEPARATE (§7.2) — a tool
    crash is ``logical=unknown / execution=error``, never a proof of logical failure;
  * the precedence primitives (§7.3) the policy reducer composes.

Authority boundary (see ``policy_reducer`` for the full statement): this module and
the reducer decide *reviewer-plane* precedence over a CandidatePackage. It is NOT
``SafetyGate`` (autopilot experiment-admission authority) and NOT the batch
manifest fork-tables (batch-execution scope). Those remain separate authorities.

Canonical schema ownership: the on-disk field names read here
(``evidence_item.schema.json`` / the v1.1 ``verification_report.schema.json`` check:
``criterion_id``, ``severity``, ``logical_status``, ``execution_status``,
``authority.{class,valid_for,may_block,may_approve}``, ``scope.assumptions``) are
owned by the sibling CP2 schema/ledger work. This module *reads* them; it does not
define or mutate them. The dataclasses here are in-memory adapters over those
artifacts, not a competing schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


# ── logical vs execution status (spec §7.2 — kept SEPARATE) ──────────────────


class LogicalStatus(str, Enum):
    """Epistemic verdict of a check. Distinct from operational health.

    solver returns UNKNOWN  -> logical=unknown  (execution=ok)
    verifier process crashes -> logical=unknown  (execution=error)
    two sound verifiers disagree -> logical=conflict (execution=ok)
    property test finds a counterexample -> logical=fail (execution=ok)
    """

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class ExecutionStatus(str, Enum):
    """Operational health of a verifier run. A non-``ok`` value is a tool failure,
    NOT proof of logical failure (§7.3 rule 5)."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


_OPERATIONAL_ERROR = frozenset(
    {ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT, ExecutionStatus.UNAVAILABLE}
)

# v1.0 back-compat: a report ``outcome`` maps to a logical_status when the check
# does not carry the v1.1 ``logical_status`` field (schema note: pass↔pass,
# fail↔fail, inconclusive↔unknown).
_OUTCOME_TO_LOGICAL = {
    "pass": LogicalStatus.PASS,
    "fail": LogicalStatus.FAIL,
    "inconclusive": LogicalStatus.UNKNOWN,
    "conflict": LogicalStatus.CONFLICT,
}


def coerce_logical(value: Any) -> LogicalStatus:
    if isinstance(value, LogicalStatus):
        return value
    try:
        return LogicalStatus(str(value))
    except ValueError:
        return LogicalStatus.UNKNOWN


def coerce_execution(value: Any) -> ExecutionStatus:
    if isinstance(value, ExecutionStatus):
        return value
    try:
        return ExecutionStatus(str(value))
    except ValueError:
        return ExecutionStatus.OK


# ── severity (mirror of AssuranceProfile.criteria[*].severity) ───────────────


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_SEVERITY_RANK = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def coerce_severity(value: Any, default: Severity = Severity.MEDIUM) -> Severity:
    if isinstance(value, Severity):
        return value
    try:
        return Severity(str(value))
    except ValueError:
        return default


def severity_at_least(value: Severity, minimum: Severity) -> bool:
    return _SEVERITY_RANK[value] >= _SEVERITY_RANK[minimum]


# ── authority classes (spec §7.1 table) ──────────────────────────────────────


class Grant(str, Enum):
    """Whether a class carries an approve/block grant.

    ``YES`` / ``NO`` are fixed by the class's soundness semantics; ``POLICY`` means
    the grant is decided by calibrated policy (§7.1 "policy-dependent" / §11).
    """

    YES = "yes"
    NO = "no"
    POLICY = "policy"


class AuthorityClass(str, Enum):
    PROOF = "proof"
    COMPLETE_DECIDER = "complete_decider"
    SOUND_REFUTATION = "sound_refutation"
    SOUND_ACCEPTANCE = "sound_acceptance"
    BOUNDED_TEST = "bounded_test"
    STATISTICAL_EVIDENCE = "statistical_evidence"
    HEURISTIC_STATIC = "heuristic_static"
    LLM_JUDGMENT = "llm_judgment"
    HUMAN_ATTESTATION = "human_attestation"


# (approve_grant, block_grant) exactly per the §7.1 table.
#   proof / complete_decider  -> approve YES, block YES
#   sound_refutation          -> approve NO,  block YES   (a pass proves nothing)
#   sound_acceptance          -> approve YES, block NO    (a fail may be incomplete)
#   bounded_test / statistical_evidence / human_attestation -> policy-dependent both
#   heuristic_static          -> approve NO, block "normally no" -> NO in the reducer
#   llm_judgment              -> both only after calibration + policy grant -> POLICY
_AUTHORITY_TABLE: dict[AuthorityClass, tuple[Grant, Grant]] = {
    AuthorityClass.PROOF: (Grant.YES, Grant.YES),
    AuthorityClass.COMPLETE_DECIDER: (Grant.YES, Grant.YES),
    AuthorityClass.SOUND_REFUTATION: (Grant.NO, Grant.YES),
    AuthorityClass.SOUND_ACCEPTANCE: (Grant.YES, Grant.NO),
    AuthorityClass.BOUNDED_TEST: (Grant.POLICY, Grant.POLICY),
    AuthorityClass.STATISTICAL_EVIDENCE: (Grant.POLICY, Grant.POLICY),
    AuthorityClass.HEURISTIC_STATIC: (Grant.NO, Grant.NO),
    AuthorityClass.LLM_JUDGMENT: (Grant.POLICY, Grant.POLICY),
    AuthorityClass.HUMAN_ATTESTATION: (Grant.POLICY, Grant.POLICY),
}


def coerce_authority_class(value: Any) -> AuthorityClass:
    if isinstance(value, AuthorityClass):
        return value
    try:
        return AuthorityClass(str(value))
    except ValueError:
        # Unknown/absent authority defaults to the weakest safe class: a heuristic
        # that can neither approve nor block. The reducer therefore never auto-acts
        # on unclassified evidence (fail-safe toward advisory).
        return AuthorityClass.HEURISTIC_STATIC


def _resolve_grant(grant: Grant, explicit: bool | None, policy_grant: bool) -> bool:
    """Resolve one grant to a concrete bool.

    Trust model (anti-authority-laundering, §13.2): an explicit per-evidence
    boolean (set by a TRUSTED producer/registry) may only *narrow* a YES class and
    may *set* a POLICY class; it can never *widen* a NO class. A ``NO`` grant is
    fixed by soundness (e.g. a ``sound_refutation`` pass can never approve; a
    ``heuristic_static`` can never block conclusive evidence, §7.3 rule 3), so no
    downstream field can escalate it.
    """
    if grant is Grant.NO:
        return False
    if grant is Grant.YES:
        return True if explicit is None else bool(explicit)  # narrow-only
    # POLICY: a trusted explicit declaration wins; else the calibrated policy grant.
    if explicit is not None:
        return bool(explicit)
    return bool(policy_grant)


@dataclass(frozen=True)
class Authority:
    """Criterion-scoped authority carried by one evidence item / check.

    ``cls`` fixes the soundness semantics; ``explicit_may_*`` are the optional
    trusted per-evidence booleans from ``evidence_item.schema.json`` /
    ``verification_report`` check ``authority.{may_block,may_approve}``.
    """

    cls: AuthorityClass = AuthorityClass.HEURISTIC_STATIC
    valid_for: tuple[str, ...] = ()
    explicit_may_approve: bool | None = None
    explicit_may_block: bool | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Authority":
        data = data or {}
        vf = data.get("valid_for") or ()
        return cls(
            cls=coerce_authority_class(data.get("class")),
            valid_for=tuple(str(x) for x in vf),
            explicit_may_approve=_opt_bool(data.get("may_approve")),
            explicit_may_block=_opt_bool(data.get("may_block")),
        )

    def may_approve(self, *, policy_grant: bool = False) -> bool:
        approve_grant, _ = _AUTHORITY_TABLE[self.cls]
        return _resolve_grant(approve_grant, self.explicit_may_approve, policy_grant)

    def may_block(self, *, policy_grant: bool = False) -> bool:
        _, block_grant = _AUTHORITY_TABLE[self.cls]
        return _resolve_grant(block_grant, self.explicit_may_block, policy_grant)

    def scopes_criterion(self, criterion_id: str | None) -> bool:
        """Authority applies only to its own criterion unless ``valid_for`` lists more."""
        if not self.valid_for:
            return True
        return criterion_id in self.valid_for


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


# ── conclusive-status primitives (§7.3 building blocks) ──────────────────────


def is_operational_error(execution: ExecutionStatus) -> bool:
    return execution in _OPERATIONAL_ERROR


def is_conclusive_failure(
    authority: Authority,
    logical: LogicalStatus,
    execution: ExecutionStatus,
    *,
    policy_grant: bool = False,
) -> bool:
    """A criterion-scoped, execution-clean FAIL whose authority may block.

    Encodes §7.3 rule 1's "conclusive failure" and §7.1's block column:
    sound_refutation / proof / complete_decider block unconditionally; bounded_test
    / statistical_evidence / llm_judgment block only under a policy grant (or a
    trusted explicit ``may_block``); heuristic_static / sound_acceptance never block.
    An operational error is NOT a failure (§7.3 rule 5).
    """
    if execution is not ExecutionStatus.OK:
        return False
    if logical is not LogicalStatus.FAIL:
        return False
    return authority.may_block(policy_grant=policy_grant)


def is_conclusive_pass(
    authority: Authority,
    logical: LogicalStatus,
    execution: ExecutionStatus,
    *,
    policy_grant: bool = False,
) -> bool:
    """A criterion-scoped, execution-clean PASS whose authority may approve.

    proof / complete_decider / sound_acceptance establish; sound_refutation PASS
    proves nothing (approve NO); others policy-dependent.
    """
    if execution is not ExecutionStatus.OK:
        return False
    if logical is not LogicalStatus.PASS:
        return False
    return authority.may_approve(policy_grant=policy_grant)


__all__ = [
    "LogicalStatus",
    "ExecutionStatus",
    "Severity",
    "Grant",
    "AuthorityClass",
    "Authority",
    "coerce_logical",
    "coerce_execution",
    "coerce_severity",
    "coerce_authority_class",
    "severity_at_least",
    "is_operational_error",
    "is_conclusive_failure",
    "is_conclusive_pass",
]
