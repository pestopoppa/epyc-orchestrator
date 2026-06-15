"""Trinity tri-role classifier (TR-3.1 of tri-role-coordinator-architecture.md).

Rule-based classifier that maps a request to one of {THINKER, WORKER, VERIFIER}.
Deterministic, regex-only, no model inference — designed to run on the routing
hot path without measurable latency cost.

The output is the per-call Trinity role axis ORTHOGONAL to model selection.
TR-2 plumbed the field end-to-end through `RoutingResult.assigned_role` +
`RoleResult.assigned_role` + `episodic.db.assigned_role`. TR-3 (this module)
populates it. TR-4 will gate prompt-template selection on it. TR-5 A/B will
flip the `ORCHESTRATOR_ROLE_AWARE_ROUTING` feature flag.

The classifier ALWAYS runs in shadow mode — i.e. `RoutingResult.assigned_role`
is always populated and logged, but only acted on when the feature flag is ON
(see `role_taxonomy.role_aware_routing_enabled()`).

Heuristic rules (priority order — first match wins):

1. **VERIFIER**: prompt mentions review / verify / check / audit / correctness
   semantics AND has prior-content cues (the request is a quality gate over
   something already produced).
2. **THINKER**: routing chose an architect_* role, or force_role names an
   architect, or thinking_budget > 0, or prompt has plan/decompose/design/
   strategy keywords. Plan-class work is THINKER even on a non-architect model.
3. **WORKER**: default. Direct execution of the task on whatever input.

Why this ordering: VERIFIER is the most specific signal (review-trigger is
discrete); THINKER is the next most specific (architect routing or explicit
plan request); WORKER is the catch-all. Reverse-ordering would drown VERIFIER
hits in plan-keyword false-positives.

Per the TR-1 taxonomy decision, role distribution should be non-degenerate
(NOT 99% Worker). TR-3.4 will check this empirically against shadow telemetry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.classifiers.role_taxonomy import TrinityRole
from src.roles import Role

# ── Keyword patterns ─────────────────────────────────────────────────
# Compiled once at import. Word-boundary anchored to avoid substring drift
# (e.g. "checkmate" should not match "check"); lowercased input expected.

_VERIFIER_TRIGGER_RE = re.compile(
    r"\b("
    r"review|reviewing|reviewed|"
    r"verify|verifying|verified|verification|"
    r"audit|auditing|audited|"
    r"check\s+(?:that|whether|if|the|this|my|our)|"
    r"correctness|correct(?:ness)?(?:\s+of)?|"
    r"validate|validation|validating|"
    r"critique|critic|"
    r"is\s+this\s+(?:right|correct|valid|accurate)|"
    r"does\s+(?:this|the\s+\w+)\s+(?:work|pass|satisfy)|"
    r"sanity[-\s]?check|"
    r"double[-\s]?check"
    r")\b"
)

# Verifier needs SOMETHING TO VERIFY — a cue that the prompt references prior
# content (code snippet, answer, draft). Without this, "verify the formula"
# might be a Worker-class derivation request.
_PRIOR_CONTENT_RE = re.compile(
    r"\b("
    r"this\s+(?:answer|code|solution|output|response|draft|result|claim|proof|argument|plan|design)|"
    r"the\s+(?:above|previous|preceding|prior|attached|following)|"
    r"my\s+\w+\s+above|"  # "my plan above", "my code above", etc.
    r"my\s+(?:answer|code|solution|attempt|plan|draft|proof|design|proposal|reasoning|approach)|"
    r"```|"  # Fenced code block in prompt → there's content to verify
    r"i\s+(?:wrote|tried|computed|got|believe)|"
    r"following\s+(?:answer|code|solution|response|draft)"
    r")\b"
)

_THINKER_TRIGGER_RE = re.compile(
    r"\b("
    r"plan(?:ning)?(?:\s+for)?|"
    r"decompose|decomposition|break(?:ing)?\s+(?:this|it|the\s+\w+)\s+down|"
    r"design(?:ing)?(?:\s+a)?|architect(?:ure)?|"
    r"strateg(?:y|ies|ic)|"
    r"approach(?:\s+to)?|"
    r"outline(?:\s+a)?|"
    r"high[-\s]?level|"
    r"trade[-\s]?off|"
    r"pros\s+and\s+cons|"
    r"how\s+(?:should|would|do)\s+(?:i|we|you)\s+(?:approach|tackle|structure)|"
    r"propose\s+a"
    r")\b"
)

# Architect-class roles that are inherently planning-tier.
_ARCHITECT_ROLE_PREFIXES = ("architect_",)


@dataclass(frozen=True)
class RoleClassification:
    """Outcome of classify_role().

    Returns the chosen role plus a short reason code for telemetry. Reason
    codes are stable strings — TR-3.3 telemetry will aggregate by reason.
    """

    role: str  # one of TrinityRole values
    reason: str  # short code, e.g. "verifier_review_trigger", "thinker_architect_role"


def _has_verifier_signal(prompt_lc: str) -> bool:
    """Verifier requires BOTH a trigger keyword AND a prior-content cue."""
    if not _VERIFIER_TRIGGER_RE.search(prompt_lc):
        return False
    return _PRIOR_CONTENT_RE.search(prompt_lc) is not None


def _is_architect_role_name(role: str) -> bool:
    canonical = Role.from_string(role)
    if canonical is not None:
        return canonical == Role.ARCHITECT_GENERAL or canonical.value.startswith("architect_")
    return role.startswith(_ARCHITECT_ROLE_PREFIXES)


def _has_architect_role(routing_decision: list | None, force_role: str | None) -> bool:
    """True if routing or force_role names an architect-class model."""
    if force_role and _is_architect_role_name(str(force_role)):
        return True
    if routing_decision:
        head = str(routing_decision[0])
        if _is_architect_role_name(head):
            return True
    return False


def classify_role(
    prompt: str,
    *,
    routing_decision: list | None = None,
    force_role: str | None = None,
    thinking_budget: int = 0,
    context: str = "",
) -> RoleClassification:
    """Map a request to a Trinity tri-role.

    Args:
        prompt: User's input prompt.
        routing_decision: Routed model role list (head is primary). May be empty.
        force_role: Explicit role override from the request (None if not set).
        thinking_budget: Token budget for internal reasoning (0 = no explicit
            reasoning request).
        context: Optional pre-prompt context (e.g. attached document text).
            Combined with prompt for keyword scanning when present.

    Returns:
        RoleClassification with `.role` ∈ {thinker, worker, verifier} and a
        short reason code for telemetry.
    """
    text = (prompt + " " + (context or "")).lower()

    # Rule 1: Verifier — explicit review/verify trigger over prior content.
    if _has_verifier_signal(text):
        return RoleClassification(
            role=TrinityRole.VERIFIER.value,
            reason="verifier_review_trigger",
        )

    # Rule 2a: Thinker — architect-class routing (model already chosen for plan-tier).
    if _has_architect_role(routing_decision, force_role):
        return RoleClassification(
            role=TrinityRole.THINKER.value,
            reason="thinker_architect_role",
        )

    # Rule 2b: Thinker — explicit reasoning budget requested.
    if thinking_budget > 0:
        return RoleClassification(
            role=TrinityRole.THINKER.value,
            reason="thinker_thinking_budget",
        )

    # Rule 2c: Thinker — plan/decompose/design keywords in prompt.
    if _THINKER_TRIGGER_RE.search(text):
        return RoleClassification(
            role=TrinityRole.THINKER.value,
            reason="thinker_plan_keyword",
        )

    # Rule 3: Worker — default. Direct execution.
    return RoleClassification(
        role=TrinityRole.WORKER.value,
        reason="worker_default",
    )
