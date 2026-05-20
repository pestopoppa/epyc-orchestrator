"""Orchestration sub-decision taxonomy (intake-548, arxiv:2605.02801).

The 5-sub-decision schema from Zhang et al.'s "RL for LLM-based MAS through
Orchestration Traces" survey. Every orchestration episode involves five
discrete decision points; labelling traces by which one a given event represents
turns the episodic store into a sliceable substrate for any future
stopping-policy / delegation-policy learned head.

This is companion-axis to `role_taxonomy.py`. The Trinity role
(thinker/worker/verifier) says *what kind of call* a memory was; the
sub-decision says *where in the orchestration lifecycle* the call sat.

- `when-to-spawn`     → SPAWN        — a parent decides to call a child agent at all
- `whom-to-delegate`  → DELEGATE     — picking which agent / model / role to call
- `how-to-communicate`→ COMMUNICATE  — formatting / routing the inter-agent message
- `how-to-aggregate`  → AGGREGATE    — folding child output back into parent context
- `when-to-stop`      → STOP         — terminating recursion or the whole episode

Per the RAO+ReDel substrate spike (handoffs/active/rao-redel-substrate-spike.md
Step 3 "sub-decision labelling" subtask), this column adds a labelling axis to
the episodic store. Default is NULL — most events are not sub-decisions, and
forcing a value on every row would dilute the axis.

Feature flag `SUBDECISION_LABELLING` (env var
`ORCHESTRATOR_SUBDECISION_LABELLING`) defaults OFF — until the writer side is
wired in, the column accepts new labels but does not enforce or auto-populate.
"""

from __future__ import annotations

import os
from enum import Enum


class OrchestrationSubDecision(str, Enum):
    """Per-event sub-decision axis (intake-548, 5-class)."""

    SPAWN = "spawn"
    DELEGATE = "delegate"
    COMMUNICATE = "communicate"
    AGGREGATE = "aggregate"
    STOP = "stop"


# Default for events that are NOT sub-decisions (most rows). Stored as NULL
# in SQLite. Readers should NOT coerce NULL to a sentinel — a NULL row means
# "this event is not a sub-decision", not "this event is a SPAWN we couldn't
# classify". This is the opposite polarity to `assigned_role`, where NULL is
# treated as the default WORKER role.
DEFAULT_SUBDECISION: str | None = None


# All valid sub-decision values, for writer validation + classifier dispatch.
VALID_SUBDECISIONS: frozenset[str] = frozenset(d.value for d in OrchestrationSubDecision)


def subdecision_labelling_enabled() -> bool:
    """Feature flag for sub-decision labelling.

    OFF by default — the column accepts writes but no production code path is
    required to set it yet. Flip ON once a writer is wired (RAO+ReDel Step 3
    Phase A) and the heuristic backfill has run at least once.

    Toggle via env var `ORCHESTRATOR_SUBDECISION_LABELLING` (1 / true / yes / on).
    """
    raw = os.environ.get("ORCHESTRATOR_SUBDECISION_LABELLING", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def normalise_subdecision(value: str | None) -> str | None:
    """Coerce an arbitrary input string to a valid sub-decision value or None.

    Unlike `normalise_role`, this returns None for unknown / empty inputs
    rather than falling back to a default. The polarity is "this event is
    not a labelled sub-decision" rather than "we don't know, assume X".

    Returns:
        - One of {"spawn","delegate","communicate","aggregate","stop"} for
          recognised inputs (case-insensitive).
        - None for None, empty, whitespace, or unrecognised inputs.
    """
    if not value:
        return None
    candidate = value.strip().lower()
    if candidate in VALID_SUBDECISIONS:
        return candidate
    return None
