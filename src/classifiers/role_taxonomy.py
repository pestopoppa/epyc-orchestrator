"""Trinity tri-role taxonomy constants (TR-2.0 of tri-role-coordinator-architecture.md).

Trinity-style per-call role axis (Thinker / Worker / Verifier), orthogonal to
model selection. Each orchestrator dispatch carries an assigned role that
constrains the prompt template and the verification expectation.

This module is the canonical source of role-name constants. Importers should
use `Role.WORKER` etc. rather than hardcoding string literals — keeps the
typo blast radius small if we ever rename.

Per `agents/shared/ENGINEERING_STANDARDS.md` § Code Invariants: enums and
constants, not ad hoc strings.

Per the TR-1 deliverable in handoffs/active/tri-role-coordinator-architecture.md:
- Roles are NOT model-permanent. The same model takes different roles on
  different calls within the same session.
- Action space encoding is decoupled `(L + 3)` flat logits — model and role
  are independently softmaxed (TR-1.3).
- Default role is `WORKER` for backward compat (legacy memories pre-TR-2 lack
  the field; treat as Worker on read).
- Feature flag `ROLE_AWARE_ROUTING` (env var `ORCHESTRATOR_ROLE_AWARE_ROUTING`)
  defaults OFF — role assignment is logged in shadow mode until the TR-5 A/B
  promotes it to default-on.
"""

from __future__ import annotations

import os
from enum import Enum


class TrinityRole(str, Enum):
    """Per-call tri-role axis. String enum so values flow through SQLite + JSON."""

    THINKER = "thinker"   # Plan / decompose / critique / abstract reasoning
    WORKER = "worker"     # Execute the task on whatever input was assigned
    VERIFIER = "verifier"  # Check correctness / completeness of prior output


# Default for legacy / unannotated memories. Per TR-1.5: default is WORKER for
# backward compatibility — legacy memories from before this column existed are
# overwhelmingly direct-execute calls.
DEFAULT_TRINITY_ROLE: str = TrinityRole.WORKER.value


# All valid role values, for validation in writers + classifier dispatch.
VALID_TRINITY_ROLES: frozenset[str] = frozenset(r.value for r in TrinityRole)


def role_aware_routing_enabled() -> bool:
    """Feature flag for the tri-role routing axis (TR-1.5).

    OFF by default — the role classifier (TR-3) runs in shadow mode and logs
    decisions but does not act. Flipped to ON after the TR-5 A/B promotes
    role-awareness to default.

    Toggle via env var `ORCHESTRATOR_ROLE_AWARE_ROUTING` (1 / true / yes).
    """
    raw = os.environ.get("ORCHESTRATOR_ROLE_AWARE_ROUTING", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def normalise_role(value: str | None) -> str:
    """Coerce an arbitrary input string to a valid Trinity role value.

    Returns DEFAULT_TRINITY_ROLE if the input is None, empty, or not a known
    role. Tolerant casing (case-insensitive). Keeps the writer + read paths
    robust against legacy / pre-TR-2 memories that store NULL or stale strings.
    """
    if not value:
        return DEFAULT_TRINITY_ROLE
    candidate = value.strip().lower()
    if candidate in VALID_TRINITY_ROLES:
        return candidate
    return DEFAULT_TRINITY_ROLE
