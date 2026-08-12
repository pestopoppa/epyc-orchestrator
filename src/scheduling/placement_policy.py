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
dispatcher's behavior (try full first, then disjoint split regions). Every role
keeps its observed-2026-05-25 placement until WP-5 ratifies a different
value. No live behavior changes when this module is added.

Cross-ref: handoffs/active/within-role-placement-state-machine.md § Phase 5
"""

from __future__ import annotations

import enum
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# One-shot guard: the live-config import is attempted on every dispatch, so an
# unconditional warning would spam. We only need the first one — it is the
# report that the fleet is running defaults it was never configured with.
_live_config_warned = False


class RolePlacementPolicy(str, enum.Enum):
    """Per-role placement strategy for the dispatcher (WP-5).

    Values are strings (rather than auto-int) so NUMA_CONFIG entries can use
    bare strings without an enum import — the accessor converts them back.

    Semantics (consumed by WP-2 placement state machine + WP-3 migration
    trigger; not yet acted on as of WP-5 scaffold):
      * SOLO_PREFER_FULL — current behavior. Solo requests prefer the full
        instance for peak per-request latency; concurrent requests spill to
        the topology's NUMA-disjoint split instances (halves as of the
        2026-07-30 cutover; quarters historically). Migration may evict
        full→split under load (WP-3). This is the conservative default.
      * BURST_PREFER_SPLIT — at N≥2, prefer the topology's sub-full instances
        even when full is free. Avoids paying the migration cost on every load
        transition. Single requests still go to full when load is 0.
      * FULL_DISABLED — never place on full; split instances only. Reclaims the
        full instance's mlock at the cost of solo per-request latency.
        Useful for roles whose full overlaps every split region (worker_general).
      * QUEUE_ONLY — single-flight per role; no concurrent placement at
        all. Effectively N=1 with a queue. Diagnostic / fallback mode.
    """

    SOLO_PREFER_FULL = "solo_prefer_full"
    BURST_PREFER_SPLIT = "burst_prefer_split"
    FULL_DISABLED = "full_disabled"
    QUEUE_ONLY = "queue_only"


class BatchPlacementMode(str, enum.Enum):
    """Request-scoped placement intent for burst/eval admission.

    ``AUTO`` preserves ordinary workload-sensitive placement. A known
    homogeneous cohort may share the full server's certified native slots.
    A routed pipeline cohort uses split CPU instances from its first dispatch
    so downstream roles can occupy disjoint regions without mid-decode moves.
    """

    AUTO = "auto"
    HOMOGENEOUS_NATIVE_BATCH = "homogeneous_native_batch"
    MIXED_ROLE_SPLIT = "mixed_role_split"


def coerce_batch_placement_mode(raw: object) -> BatchPlacementMode:
    """Validate a request-scoped batch placement mode.

    Invalid internal values fail closed to split placement. Public API callers
    are rejected earlier by Pydantic, but internal callers must never turn a
    typo into an all-region shared lease.
    """
    if isinstance(raw, BatchPlacementMode):
        return raw
    text = str(raw or BatchPlacementMode.AUTO.value).strip().lower()
    try:
        return BatchPlacementMode(text)
    except ValueError:
        return BatchPlacementMode.MIXED_ROLE_SPLIT


# Conservative default — preserves 2026-05-25 dispatcher behavior for every
# role until WP-5 ratifies per-role values.
DEFAULT_PLACEMENT_POLICY = RolePlacementPolicy.SOLO_PREFER_FULL

# Read compatibility only.  The deployed fleet retired quarter instances in
# favor of halves on 2026-07-30, so the canonical policy vocabulary must not
# describe a physical shape that no longer exists.  Keeping the old spelling
# out of the enum and runtime configuration makes every newly rendered/logged
# value accurate while still allowing an older external config to boot safely.
_LEGACY_POLICY_ALIASES = {
    "burst_prefer_quarters": RolePlacementPolicy.BURST_PREFER_SPLIT,
}


def _coerce(raw: object) -> Optional[RolePlacementPolicy]:
    """Coerce a NUMA_CONFIG-supplied value to the enum.

    Returns None ONLY when nothing was configured. A value that WAS configured
    but cannot be mapped now raises, because the previous behaviour was a silent
    downgrade with real consequences: an unrecognised string returned None and
    the caller substituted DEFAULT_PLACEMENT_POLICY (= SOLO_PREFER_FULL), which
    is not "no policy" — it is a DIFFERENT policy, and the one that lets a solo
    request acquire a full instance's every region lock and serialize the
    machine (the DISPATCH-A shape, 2026-07-21).

    The deployed topology is 1 full + 2 halves, so the canonical vocabulary is
    shape-agnostic. The explicit legacy alias above prevents an older external
    config from degrading to SOLO_PREFER_FULL during the rename.
    """
    if isinstance(raw, RolePlacementPolicy):
        return raw
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().lower()
        if not text:
            return None
        if text in _LEGACY_POLICY_ALIASES:
            return _LEGACY_POLICY_ALIASES[text]
        try:
            return RolePlacementPolicy(text)
        except ValueError as exc:
            raise ValueError(
                f"placement_policy {raw!r} is not a RolePlacementPolicy. "
                f"Valid: {sorted(p.value for p in RolePlacementPolicy)}. "
                "Refusing to fall back to the default — that would silently "
                "substitute a DIFFERENT policy, not an absent one."
            ) from exc
    raise TypeError(
        f"placement_policy must be a str or RolePlacementPolicy, got {type(raw).__name__}: {raw!r}"
    )


def get_placement_policy(role: str, numa_config: Optional[dict] = None) -> RolePlacementPolicy:
    """Return the placement policy for `role`.

    Resolution order:
      1. `numa_config[role]['placement_policy']` if present (caller-supplied
         config, e.g. test fixture).
      2. Live `scripts.server.stack_numa.NUMA_CONFIG[role]['placement_policy']`
         if importable.
      3. `DEFAULT_PLACEMENT_POLICY` (= SOLO_PREFER_FULL).

    An ABSENT policy (unknown role, missing key, None, empty string) resolves
    to `DEFAULT_PLACEMENT_POLICY`. A value that WAS configured but cannot be
    mapped RAISES — see `_coerce`. Do not reintroduce a fallback here: the
    default is not "no policy", it is a different one, and substituting it for
    a typo is the fail-open shape this module exists to prevent.
    """
    if numa_config is not None:
        cfg = numa_config.get(role) if numa_config else None
    else:
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
            cfg = NUMA_CONFIG.get(role)
        except Exception as exc:
            # Not fatal: an absent live config is legitimate (tests, tooling
            # imported outside the server tree). But it is NOT silent — if this
            # import breaks in production, EVERY role resolves to
            # DEFAULT_PLACEMENT_POLICY, which is the same fail-open degradation
            # `_coerce` refuses, just applied fleet-wide instead of per-role.
            global _live_config_warned
            if not _live_config_warned:
                _live_config_warned = True
                logger.warning(
                    "placement_policy: live NUMA_CONFIG unavailable (%s: %s); "
                    "every role now resolves to the default %r until this is "
                    "fixed. Configured per-role policies are NOT in effect.",
                    type(exc).__name__, exc, DEFAULT_PLACEMENT_POLICY.value,
                )
            cfg = None

    if not cfg:
        return DEFAULT_PLACEMENT_POLICY
    raw = cfg.get("placement_policy")
    if raw is None:
        return DEFAULT_PLACEMENT_POLICY
    coerced = _coerce(raw)
    return coerced if coerced is not None else DEFAULT_PLACEMENT_POLICY
