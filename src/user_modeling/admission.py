"""Profile admission schema (HS-PROF-1; pattern: Honcho peer-card rule, intake-2237#record).

Opt-in write filter for :class:`ProfileStore`. Off by default: with no policy the
store behaves byte-for-byte as before. A fact is admitted only if

  1. its kind is one of identity / attribute / relationship / instruction,
  2. it passes the six-month stability test (no transient markers; if the
     caller supplies ``evidence_span_days`` it must be >= 180),
  3. it is not behavioural or inferred (hedged/habitual wording, or flagged
     ``inferred``), and
  4. the profile is below the entry cap.

The rules are deterministic; they judge text and flags, never a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ALLOWED_KINDS = ("identity", "attribute", "relationship", "instruction")
DEFAULT_MAX_ENTRIES = 24
STABILITY_DAYS = 180.0

# Legacy category -> kind when the caller supplies no explicit kind.
# "general" is deliberately unmapped: an unclassifiable fact is not admitted.
CATEGORY_TO_KIND = {
    "domain": "attribute",
    "format": "instruction",
    "workflow": "instruction",
    "style": "instruction",
}

_TRANSIENT_RE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|right now|at the moment|for now|currently|"
    r"this (?:session|week|month|morning|afternoon|evening)|next (?:week|month)|"
    r"temporar(?:y|ily)|just (?:now|started)|these days|lately|recently)\b",
    re.I,
)
_BEHAVIOURAL_RE = re.compile(
    r"\b(tends? to|seems? to|appears? to|probably|likely|presumably|might|"
    r"often|usually|frequently|typically|sometimes|apparently|inferred|"
    r"suggests? that|(?:was|were) (?:observed|seen) (?:to )?)\b",
    re.I,
)


@dataclass(frozen=True)
class AdmissionPolicy:
    allowed_kinds: tuple[str, ...] = ALLOWED_KINDS
    max_entries: int = DEFAULT_MAX_ENTRIES
    stability_days: float = STABILITY_DAYS


@dataclass(frozen=True)
class AdmissionDecision:
    admit: bool
    reason: str = "ok"
    kind: str = ""


def resolve_kind(kind: str, category: str) -> str:
    return (kind or CATEGORY_TO_KIND.get(category, "")).lower()


def check_admission(
    fact_text: str,
    category: str,
    *,
    kind: str = "",
    inferred: bool = False,
    evidence_span_days: float | None = None,
    current_entries: int = 0,
    policy: AdmissionPolicy | None = None,
) -> AdmissionDecision:
    """Decide admission for one fact. Pure function; first failing rule wins."""
    p = policy or AdmissionPolicy()
    k = resolve_kind(kind, category)
    if k not in p.allowed_kinds:
        return AdmissionDecision(False, f"kind_not_allowed:{k or 'unclassified'}", k)
    if inferred:
        return AdmissionDecision(False, "inferred_flag", k)
    if _BEHAVIOURAL_RE.search(fact_text):
        return AdmissionDecision(False, "behavioural_or_hedged", k)
    if _TRANSIENT_RE.search(fact_text):
        return AdmissionDecision(False, "unstable_transient_marker", k)
    if evidence_span_days is not None and evidence_span_days < p.stability_days:
        return AdmissionDecision(False, "unstable_short_evidence_span", k)
    if current_entries >= p.max_entries:
        return AdmissionDecision(False, "entry_cap", k)
    return AdmissionDecision(True, "ok", k)
