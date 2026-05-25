"""WP-5 scaffold (pre-WP-3): per-role placement policy enum + accessor.

This module exists to stabilize the dispatcher's policy-lookup call sites
BEFORE WP-3 (forward migration trigger) lands. Without it, WP-3 would either
hard-code role-table fallbacks or branch on role-name strings, both of which
WP-5 would later refactor — exactly the kind of churn the 2026-05-25
operator-approved sequencing decision in within-role-placement-state-machine.md
avoids.

What ships here (WP-5 scaffold):
  - The `RolePlacementPolicy` enum with four canonical values.
  - `get_placement_policy(role)` — reads `NUMA_CONFIG[role]['placement_policy']`
    if present, otherwise returns the conservative `SOLO_PREFER_FULL` default.

What does NOT ship here (deferred to full WP-5):
  - Per-role policy values in NUMA_CONFIG (full ratification gated on
    autopilot observability + cross-role concurrency audit).
  - Dispatcher behavior changes that depend on the policy value beyond the
    existing full-first preference (that's WP-2 + WP-3).

Conservative default justification: `SOLO_PREFER_FULL` mirrors the current
dispatcher's behavior (try full first, then disjoint quarters). Every role
keeps its observed-2026-05-25 placement until WP-5 ratifies a different
value. No live behavior changes when this module is added.

Cross-ref: handoffs/active/within-role-placement-state-machine.md § Phase 5
"""

from __future__ import annotations

import enum
from typing import Optional


class RolePlacementPolicy(str, enum.Enum):
    """Per-role placement strategy for the dispatcher (WP-5).

    Values are strings (rather than auto-int) so NUMA_CONFIG entries can use
    bare strings without an enum import — the accessor converts them back.

    Semantics (consumed by WP-2 placement state machine + WP-3 migration
    trigger; not yet acted on as of WP-5 scaffold):
      * SOLO_PREFER_FULL — current behavior. Solo requests prefer the full
        instance for peak per-request latency; concurrent requests spill
        to NUMA-disjoint quarters. Migration may evict full→quarter under
        load (WP-3). This is the conservative default.
      * BURST_PREFER_QUARTERS — at N≥2, prefer quarters even when full is
        free. Avoids paying the migration cost on every load transition.
        Single requests still go to full when load is 0.
      * FULL_DISABLED — never place on full; quarters only. Reclaims the
        full instance's mlock at the cost of solo per-request latency.
        Useful for roles whose full overlaps all quarters (worker_general).
      * QUEUE_ONLY — single-flight per role; no concurrent placement at
        all. Effectively N=1 with a queue. Diagnostic / fallback mode.
    """

    SOLO_PREFER_FULL = "solo_prefer_full"
    BURST_PREFER_QUARTERS = "burst_prefer_quarters"
    FULL_DISABLED = "full_disabled"
    QUEUE_ONLY = "queue_only"


# Conservative default — preserves 2026-05-25 dispatcher behavior for every
# role until WP-5 ratifies per-role values.
DEFAULT_PLACEMENT_POLICY = RolePlacementPolicy.SOLO_PREFER_FULL


def _coerce(raw: object) -> Optional[RolePlacementPolicy]:
    """Best-effort coercion of a NUMA_CONFIG-supplied value to the enum.
    Returns None if the value can't be mapped (caller substitutes default)."""
    if isinstance(raw, RolePlacementPolicy):
        return raw
    if isinstance(raw, str):
        try:
            return RolePlacementPolicy(raw.strip().lower())
        except ValueError:
            return None
    return None


def get_placement_policy(role: str, numa_config: Optional[dict] = None) -> RolePlacementPolicy:
    """Return the placement policy for `role`.

    Resolution order:
      1. `numa_config[role]['placement_policy']` if present (caller-supplied
         config, e.g. test fixture).
      2. Live `scripts.server.stack_numa.NUMA_CONFIG[role]['placement_policy']`
         if importable.
      3. `DEFAULT_PLACEMENT_POLICY` (= SOLO_PREFER_FULL).

    Unknown role → default. Malformed value → default + (callers can log if
    they care).
    """
    if numa_config is not None:
        cfg = numa_config.get(role) if numa_config else None
    else:
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
            cfg = NUMA_CONFIG.get(role)
        except Exception:
            cfg = None

    if not cfg:
        return DEFAULT_PLACEMENT_POLICY
    raw = cfg.get("placement_policy")
    if raw is None:
        return DEFAULT_PLACEMENT_POLICY
    coerced = _coerce(raw)
    return coerced if coerced is not None else DEFAULT_PLACEMENT_POLICY
