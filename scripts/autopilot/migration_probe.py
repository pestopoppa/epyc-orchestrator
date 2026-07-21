#!/usr/bin/env python3
"""ROUTE-A3 J2/J3 under-traffic KV-migration probe (WP-3 forward / WP-4 reverse).

Standalone driver for the *genuinely-live under-traffic* migration probe that the
within-role full<->quarter placement handoff still lists as open:

    handoffs/active/within-role-placement-state-machine.md
      "[ ] WP-3/WP-4 genuinely-live under-traffic migration probe - needs
       single-worker API (multi-worker --workers 6 confounds session affinity);
       SM trigger logic verified + regression-protected but live-traffic
       observation still open"

The migration STATE MACHINE is already unit-verified in-process
(``tests/unit/test_concurrency_aware_migration_sm.py``). What is missing is a
live observation that, *under real placement-queue traffic*, a reused session is
(J2) displaced from ``full`` onto a disjoint quarter when load bursts past the
role's safe-slot count, and (J3) migrated back to ``full`` when load drops to 1
and the reverse-migration guards (full-idle >= cooldown, session warm within the
recency window, under the per-session cap) are satisfied. Per J1 finding F3 and
the J4 ratification note, that observation is impossible on external ``/chat``
(rate-limited, distinct sessions) and on multi-worker APIs (per-worker session
affinity) -- it needs the autopilot eval-concurrency *placement-queue* path with
a **reused session** + **oscillating load** + a **single-worker** API.

Two responsibilities, cleanly split (mirrors ``screening_tier_runner.py``):

  1. **Plan construction + migration-event analysis** (pure, inference-free --
     this is all the tests exercise):
       * ``expected_migration_schedule`` runs the documented WP-3/WP-4 trigger
         model over a load-oscillation profile and emits the exact sequence of
         *expected* forward/reverse migration events + reverse-skip reasons.
       * ``plan_migration_probe`` turns that schedule into a concrete per-step
         placement-queue request plan (reused primary session + ephemeral
         interferers), with **placement-queue transport** on every request
         (``request_priority=background`` + ``workload_class=eval_batch``, NEVER a
         foreground ``/chat`` call) and single-worker/oscillation validation
         warnings. Results are **model/quant-indexed, never role-indexed**.
       * ``analyze_migration_events`` rolls a set of *observed* migration
         outcomes into forward/reverse direction totals, thrash-skip totals,
         per-session counts, cap violations, and a J2/J3 observation verdict --
         matching the ``kv_migration_direction_total`` /
         ``kv_migration_thrash_skipped_total`` counters named in the handoff.

  2. **Execution bridge** (env-flag-gated ``AUTOPILOT_MIGRATION_PROBE_INFERENCE=1``
     AND ``--execute``, DEFAULT OFF): with the gate closed the resolved plan is
     returned as a dry-run and NO inference happens. With it open, drive the plan
     over the placement queue (reused ``session_id`` across turns; oscillating
     concurrency), snapshotting migration counters between steps to reconstruct
     the observed event stream, and emit a model/quant-indexed JSONL analysis
     row. The execution path is modeled on ``bsv_paired_runner.py`` (deferred
     client import, autopilot-stopped/single-worker assumption owned by the
     caller) and is intentionally NEVER reached by the tests.

Constraints honored (CLAUDE.md + within-role handoff):
  * The serving path is FROZEN. This runner NEVER edits, imports-for-write, or
    otherwise touches ``review_service.py``, ``delegator.py``,
    ``src/backends/*``, ``features.py``, ``eval_tower.py``, or the placement
    state machine. It is a NEW standalone driver.
  * It NEVER starts/stops a llama-server or the autopilot daemon and NEVER writes
    autopilot_state.json / journals / runtime_flags.json. The caller owns the
    single-worker, no-concurrent-inference window for ``--execute``.
  * Every number produced here is a pre-gating OBSERVATION (MEASUREMENT.md); the
    J2/J3 verdict never gates a keep/revert/promote decision on its own.

Wiring (documented, NOT done here): a future caller reuses ``run_migration_probe``
the same way ``actions.py`` will reuse ``run_screening_tier`` -- env flag drives
dry-run vs execute; the driver returns the plan and (when executed) the
per-session analysis rows. This driver does NOT write any ledger.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(SCRIPT_DIR), str(ORCH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Env flag gating the inference bridge. DEFAULT OFF -> dry-run/plan only.
MIGRATION_PROBE_INFERENCE_ENV = "AUTOPILOT_MIGRATION_PROBE_INFERENCE"

RUNNER_VERSION = "migration-probe-v1"

# Placement-queue transport constants. These mirror the kwargs eval_tower's
# eval fan-out already sets (request_priority=background + workload_class=eval_batch,
# see eval_tower._eval_question) so a probe request rides the SAME background/
# eval_batch placement path the placement state machine was built for (handoff
# J1 finding F3) -- it is never a foreground /chat request.
PLACEMENT_QUEUE_TRANSPORT = "placement_queue"
PLACEMENT_REQUEST_PRIORITY = "background"
PLACEMENT_WORKLOAD_CLASS = "eval_batch"

# WP-4 reverse-migration guard defaults (mirror the handoff Phase 4 defaults).
DEFAULT_COOLDOWN_MS = 2000
DEFAULT_WINDOW_MS = 30000
DEFAULT_PER_SESSION_CAP = 5
DEFAULT_DWELL_MS = 1500
DEFAULT_COUNTER_SETTLE_MS = 250
DEFAULT_BURST_STAGGER_MS = 50
# frontdoor safe-slot count (full + q2 + q3) per the handoff safe-placement table.
DEFAULT_SAFE_SLOTS = 3
DEFAULT_ROLE = "frontdoor"
DEFAULT_SESSION_ID = "migprobe-primary"
DEFAULT_OSCILLATION = (1, 4, 4, 1, 1, 1)

DEFAULT_REGISTRY = ORCH_ROOT / "orchestration" / "model_registry.yaml"

# Reverse-skip reason tags (match kv_migration_thrash_skipped_total{...} labels
# where they overlap; "cooldown" and "session_cap" are the wired Prometheus
# thrash labels).
SKIP_COOLDOWN = "cooldown"
SKIP_SESSION_CAP = "session_cap"
SKIP_STALE_WINDOW = "stale_window"


def _env_flag_enabled(name: str) -> bool:
    """True iff env var ``name`` is a truthy flag (matches actions._env_flag_enabled)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ══════════════════════════════════════════════════════════════════════════════
# Pure helpers: oscillation parsing + role->model/quant resolution
# ══════════════════════════════════════════════════════════════════════════════


def parse_oscillation(value: str | Sequence[int] | None) -> list[int]:
    """Parse a load-oscillation profile into a list of positive concurrency ints.

    Accepts a comma/space-separated string (``"1,4,4,1"``) or an iterable of
    ints. Empty/blank tokens are dropped; every level is coerced to ``>= 1``
    (there is always at least the reused primary session in flight).
    """
    if value is None:
        return list(DEFAULT_OSCILLATION)
    if isinstance(value, str):
        toks = [t for t in value.replace(",", " ").split() if t.strip()]
        raw = [int(t) for t in toks]
    else:
        raw = [int(t) for t in value]
    return [max(1, n) for n in raw]


def resolve_model_quant_for_role(
    registry: dict[str, Any], role: str
) -> tuple[str | None, str | None]:
    """Pull ``(model_name, quant)`` for ``role`` from a lean-registry-shaped dict.

    Pure (operates on an already-loaded dict). Reads ``roles[role].model.name`` /
    ``.quant``. Returns ``(None, None)`` when the role or model block is absent so
    the caller can fall back to an explicit ``--model``/``--quant`` or "unknown".
    """
    roles = (registry or {}).get("roles") or {}
    block = roles.get(role) or {}
    model = block.get("model") or {}
    if not isinstance(model, dict):
        return None, None
    name = model.get("name")
    quant = model.get("quant")
    return (str(name) if name is not None else None,
            str(quant) if quant is not None else None)


# ══════════════════════════════════════════════════════════════════════════════
# Migration-event model (pure state machine over the load-oscillation profile)
# ══════════════════════════════════════════════════════════════════════════════


def expected_migration_schedule(
    oscillation: Sequence[int],
    *,
    safe_slots: int,
    cooldown_ms: int = DEFAULT_COOLDOWN_MS,
    window_ms: int = DEFAULT_WINDOW_MS,
    per_session_cap: int = DEFAULT_PER_SESSION_CAP,
    dwell_ms: int = DEFAULT_DWELL_MS,
    session_id: str = DEFAULT_SESSION_ID,
) -> dict[str, Any]:
    """Run the documented WP-3/WP-4 trigger model over a load profile.

    The reused *primary* session starts on ``full``. At step ``i`` (time
    ``t = i * dwell_ms``) with concurrency ``c``:

      * **Forward (J2, WP-3)** when ``c > safe_slots`` and the primary is still on
        ``full``: an interfering session takes over ``full`` and displaces the
        primary onto a disjoint quarter (one migration; primary now quartered).
      * **Reverse (J3, WP-4)** when ``c == 1`` and the primary is quartered
        (``full`` therefore idle this step): migrate back to ``full`` iff
        ``full`` has been idle >= ``cooldown_ms``, the primary is warm (issued a
        request within ``window_ms``), and the per-session migration count is
        below ``per_session_cap``. Otherwise the reverse is **skipped** with a
        reason (``cooldown`` / ``stale_window`` / ``session_cap``).

    ``full`` is considered held on any step with ``c > 1`` (an interferer holds
    it) or while the primary sits on it. Pure and deterministic. Returns a dict
    with per-step trace, the forward/reverse event lists, the skip list, and the
    total per-session migration count.
    """
    steps: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []

    primary_on_full = True
    full_last_active_ms: int | None = None
    session_migrations = 0
    last_primary_request_ms: int | None = None

    for i, c in enumerate(oscillation):
        c = max(1, int(c))
        t = i * dwell_ms
        loc_before = "full" if primary_on_full else "quarter"
        event: dict[str, Any] | None = None
        skip: dict[str, Any] | None = None

        if c > safe_slots and primary_on_full:
            # J2 forward: interferer seizes full; primary displaced to a quarter.
            event = {
                "step": i,
                "t_ms": t,
                "direction": "forward",
                "session_id": session_id,
                "from_instance": "full",
                "to_instance": "quarter",
                "concurrency": c,
            }
            primary_on_full = False
            session_migrations += 1
        elif c == 1 and not primary_on_full:
            # full is idle this step -> reverse candidate (J3).
            idle_ms = t if full_last_active_ms is None else (t - full_last_active_ms)
            warm = (
                last_primary_request_ms is not None
                and (t - last_primary_request_ms) <= window_ms
            )
            if idle_ms < cooldown_ms:
                skip = {"step": i, "t_ms": t, "reason": SKIP_COOLDOWN,
                        "session_id": session_id, "idle_ms": idle_ms}
            elif not warm:
                skip = {"step": i, "t_ms": t, "reason": SKIP_STALE_WINDOW,
                        "session_id": session_id, "idle_ms": idle_ms}
            elif session_migrations >= per_session_cap:
                skip = {"step": i, "t_ms": t, "reason": SKIP_SESSION_CAP,
                        "session_id": session_id, "idle_ms": idle_ms}
            else:
                event = {
                    "step": i,
                    "t_ms": t,
                    "direction": "reverse",
                    "session_id": session_id,
                    "from_instance": "quarter",
                    "to_instance": "full",
                    "concurrency": c,
                }
                primary_on_full = True
                session_migrations += 1

        # bookkeeping AFTER the decision: is full held on this step?
        full_held = (c > 1) or primary_on_full
        if full_held:
            full_last_active_ms = t
        last_primary_request_ms = t

        loc_after = "full" if primary_on_full else "quarter"
        steps.append({
            "step": i,
            "t_ms": t,
            "concurrency": c,
            "full_held": full_held,
            "primary_location_before": loc_before,
            "primary_location_after": loc_after,
            "event": event,
            "reverse_skip": skip,
        })
        if event is not None:
            events.append(event)
        if skip is not None:
            skips.append(skip)

    forward = [e for e in events if e["direction"] == "forward"]
    reverse = [e for e in events if e["direction"] == "reverse"]
    return {
        "steps": steps,
        "events": events,
        "forward_events": forward,
        "reverse_events": reverse,
        "reverse_skips": skips,
        "session_migrations": session_migrations,
        "expected_forward": len(forward),
        "expected_reverse": len(reverse),
        "expected_skips": len(skips),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plan dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProbeRequestSpec:
    """One placement-queue request in the probe schedule.

    ``transport``/``request_priority``/``workload_class`` pin the placement-queue
    path; ``force_role`` + ``allow_delegation=False`` pin placement to the role
    under probe so the reused session lands on the role's instances (never a
    foreground /chat request).
    """

    step: int
    t_ms: int
    session_id: str
    is_primary: bool
    role: str
    model: str | None
    quant: str | None
    transport: str = PLACEMENT_QUEUE_TRANSPORT
    request_priority: str = PLACEMENT_REQUEST_PRIORITY
    workload_class: str = PLACEMENT_WORKLOAD_CLASS

    def force_bindings(self) -> dict[str, Any]:
        return {"force_role": self.role, "allow_delegation": False}

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["force_bindings"] = self.force_bindings()
        d["kind"] = "migration_probe_request"
        return d


@dataclass
class MigrationProbePlan:
    """A concrete, placement-queue-dispatched J2/J3 migration probe (dry-run plan).

    Model/quant-indexed (never role-indexed): ``model``/``quant`` are the top-level
    identity; ``role`` is retained only as a placement detail.
    """

    role: str
    model: str | None
    quant: str | None
    session_id: str
    safe_slots: int
    oscillation: list[int]
    cooldown_ms: int
    window_ms: int
    per_session_cap: int
    dwell_ms: int
    requests: list[ProbeRequestSpec]
    schedule: dict[str, Any]
    validation_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    worker_count: int | None = None
    requires_single_worker: bool = True
    inference_required: bool = True

    def transport_summary(self) -> dict[str, Any]:
        return {
            "transport": PLACEMENT_QUEUE_TRANSPORT,
            "request_priority": PLACEMENT_REQUEST_PRIORITY,
            "workload_class": PLACEMENT_WORKLOAD_CLASS,
            "uses_chat_endpoint": False,
        }

    def model_index(self) -> dict[str, Any]:
        """The model/quant identity results are keyed on (never role)."""
        return {
            "model": self.model or "unknown",
            "quant": self.quant or "unknown",
            "model_quant_key": f"{self.model or 'unknown'}::{self.quant or 'unknown'}",
        }

    def to_dict(self) -> dict[str, Any]:
        sched = self.schedule
        return {
            "kind": "migration_probe_plan",
            "runner_version": RUNNER_VERSION,
            "route": "ROUTE-A3-j2j3-single-worker",
            "model_index": self.model_index(),
            "placement_role": self.role,
            "session_id": self.session_id,
            "safe_slots": self.safe_slots,
            "oscillation": list(self.oscillation),
            "reverse_guards": {
                "cooldown_ms": self.cooldown_ms,
                "window_ms": self.window_ms,
                "per_session_cap": self.per_session_cap,
                "dwell_ms": self.dwell_ms,
            },
            "transport": self.transport_summary(),
            "requires_single_worker": self.requires_single_worker,
            "worker_count": self.worker_count,
            "inference_required": self.inference_required,
            "n_requests": len(self.requests),
            "expected_forward": sched.get("expected_forward"),
            "expected_reverse": sched.get("expected_reverse"),
            "expected_skips": sched.get("expected_skips"),
            "expected_forward_events": sched.get("forward_events"),
            "expected_reverse_events": sched.get("reverse_events"),
            "expected_reverse_skips": sched.get("reverse_skips"),
            "schedule_steps": sched.get("steps"),
            "requests": [r.to_dict() for r in self.requests],
            "validation_warnings": list(self.validation_warnings),
            "notes": list(self.notes),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Plan construction (pure, inference-free)
# ══════════════════════════════════════════════════════════════════════════════


def plan_migration_probe(
    *,
    role: str = DEFAULT_ROLE,
    model: str | None = None,
    quant: str | None = None,
    session_id: str = DEFAULT_SESSION_ID,
    safe_slots: int = DEFAULT_SAFE_SLOTS,
    oscillation: str | Sequence[int] | None = None,
    cooldown_ms: int = DEFAULT_COOLDOWN_MS,
    window_ms: int = DEFAULT_WINDOW_MS,
    per_session_cap: int = DEFAULT_PER_SESSION_CAP,
    dwell_ms: int = DEFAULT_DWELL_MS,
    worker_count: int | None = None,
) -> MigrationProbePlan:
    """Build a concrete J2/J3 migration probe plan (pure; no inference, no I/O).

    Expands the load-oscillation profile into per-step placement-queue requests
    (one reused primary session + ``c-1`` ephemeral interferers per step), runs
    the WP-3/WP-4 trigger model to attach the expected migration schedule, and
    collects single-worker / oscillation-adequacy validation warnings.
    """
    osc = parse_oscillation(oscillation)
    if safe_slots < 1:
        raise ValueError("safe_slots must be >= 1")

    schedule = expected_migration_schedule(
        osc,
        safe_slots=safe_slots,
        cooldown_ms=cooldown_ms,
        window_ms=window_ms,
        per_session_cap=per_session_cap,
        dwell_ms=dwell_ms,
        session_id=session_id,
    )

    requests: list[ProbeRequestSpec] = []
    for i, c in enumerate(osc):
        t = i * dwell_ms
        # The reused primary session is always request 0 of the step.
        requests.append(ProbeRequestSpec(
            step=i, t_ms=t, session_id=session_id, is_primary=True,
            role=role, model=model, quant=quant,
        ))
        # Ephemeral interferers create the handover/burst pressure.
        for k in range(1, c):
            requests.append(ProbeRequestSpec(
                step=i, t_ms=t,
                session_id=f"{session_id}-intf-{i}-{k}",
                is_primary=False, role=role, model=model, quant=quant,
            ))

    warnings: list[str] = []
    if worker_count is not None and worker_count > 1:
        warnings.append(
            f"multi-worker API (workers={worker_count}) confounds per-worker "
            "session affinity; probe REQUIRES a single-worker API (--workers 1) "
            "before --execute (within-role handoff multi-worker confound)."
        )
    max_c = max(osc) if osc else 0
    if schedule["expected_forward"] == 0:
        warnings.append(
            f"oscillation never triggers J2 forward migration (max concurrency "
            f"{max_c} <= safe_slots {safe_slots} while primary on full); add a "
            "burst step with concurrency > safe_slots."
        )
    if schedule["expected_reverse"] == 0:
        warnings.append(
            "oscillation never triggers J3 reverse migration (no drop-to-1 step "
            f"after a burst with full idle >= cooldown_ms {cooldown_ms} at "
            f"dwell_ms {dwell_ms}); add >=2 consecutive concurrency-1 steps after "
            "a burst, or raise dwell_ms."
        )

    notes = [
        "reused primary session across turns forces WP-3 session-handover; "
        "ephemeral interferers create the burst that displaces it (J2).",
        "drop-to-1 after burst + cooldown drives WP-4 reverse migration (J3).",
        "all requests ride the placement queue (request_priority=background, "
        "workload_class=eval_batch); NEVER a foreground /chat request.",
        "requires a SINGLE-WORKER API; the caller owns the no-concurrent-"
        "inference window for --execute.",
        "all migration counts produced by --execute are pre-gating OBSERVATIONS "
        "(MEASUREMENT.md), never decision-gating on their own.",
    ]

    return MigrationProbePlan(
        role=role, model=model, quant=quant, session_id=session_id,
        safe_slots=safe_slots, oscillation=osc, cooldown_ms=cooldown_ms,
        window_ms=window_ms, per_session_cap=per_session_cap, dwell_ms=dwell_ms,
        requests=requests, schedule=schedule, validation_warnings=warnings,
        notes=notes, worker_count=worker_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Migration-event analysis (pure — observed outcomes -> J2/J3 verdict)
# ══════════════════════════════════════════════════════════════════════════════


def analyze_migration_events(
    observed: Sequence[dict[str, Any]],
    *,
    per_session_cap: int = DEFAULT_PER_SESSION_CAP,
    expected: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Roll observed migration outcomes into direction totals + a J2/J3 verdict.

    Each observed record is either a migration
    (``{"direction": "forward"|"reverse", "session_id", "committed": bool, ...}``)
    or a thrash-skip (``{"skipped": "<reason>", "session_id", ...}``). Aborted
    migrations (``committed=False``) count toward per-session aborts, not the
    committed direction totals. Mirrors the ``kv_migration_direction_total`` /
    ``kv_migration_thrash_skipped_total`` counters. Pure; the verdict is an
    OBSERVATION (MEASUREMENT.md), never gating on its own.
    """
    direction_total = {"forward": 0, "reverse": 0}
    thrash_skipped_total: dict[str, int] = {}
    per_session: dict[str, dict[str, int]] = {}
    n_aborted = 0

    def _ps(sid: str) -> dict[str, int]:
        return per_session.setdefault(
            sid, {"forward": 0, "reverse": 0, "total": 0, "aborted": 0}
        )

    for e in observed:
        if not isinstance(e, dict):
            continue
        if e.get("skipped"):
            reason = str(e.get("skipped"))
            thrash_skipped_total[reason] = thrash_skipped_total.get(reason, 0) + 1
            continue
        d = e.get("direction")
        if d not in ("forward", "reverse"):
            continue
        sid = str(e.get("session_id", "?"))
        ps = _ps(sid)
        if bool(e.get("committed", True)):
            direction_total[d] += 1
            ps[d] += 1
            ps["total"] += 1
        else:
            n_aborted += 1
            ps["aborted"] += 1

    sessions_over_cap = sorted(
        sid for sid, ps in per_session.items() if ps["total"] > per_session_cap
    )
    j2_forward_observed = direction_total["forward"] >= 1
    j3_reverse_observed = direction_total["reverse"] >= 1

    reasons: list[str] = []
    if not j2_forward_observed:
        reasons.append("no forward (J2) migration observed")
    if not j3_reverse_observed:
        reasons.append("no reverse (J3) migration observed")
    if sessions_over_cap:
        reasons.append(
            f"{len(sessions_over_cap)} session(s) exceeded per-session cap "
            f"{per_session_cap}"
        )
    if n_aborted:
        reasons.append(f"{n_aborted} migration transaction(s) aborted")

    verdict = "PASS" if not reasons else "INCONCLUSIVE"

    expected_match: dict[str, Any] | None = None
    if expected is not None:
        exp_fwd = sum(1 for x in expected if x.get("direction") == "forward")
        exp_rev = sum(1 for x in expected if x.get("direction") == "reverse")
        expected_match = {
            "expected_forward": exp_fwd,
            "expected_reverse": exp_rev,
            "forward_matches": exp_fwd == direction_total["forward"],
            "reverse_matches": exp_rev == direction_total["reverse"],
        }

    return {
        "kind": "migration_probe_analysis",
        "runner_version": RUNNER_VERSION,
        "direction_total": direction_total,
        "thrash_skipped_total": thrash_skipped_total,
        "n_committed": direction_total["forward"] + direction_total["reverse"],
        "n_aborted": n_aborted,
        "per_session": per_session,
        "sessions_over_cap": sessions_over_cap,
        "j2_forward_observed": j2_forward_observed,
        "j3_reverse_observed": j3_reverse_observed,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "expected_match": expected_match,
        "observation_only": True,  # MEASUREMENT.md — never decision-gating alone
    }


# ══════════════════════════════════════════════════════════════════════════════
# Execution bridge (env-gated; models bsv_paired_runner; NEVER run in tests)
# ══════════════════════════════════════════════════════════════════════════════


def _default_request(  # pragma: no cover - inference path
    spec: ProbeRequestSpec,
    client: Any,
    *,
    url: str,
    timeout: int,
) -> dict[str, Any]:
    """Send ONE probe request over the placement queue (reused session_id).

    Reuses the SAME transport eval_tower's fan-out uses --
    ``call_orchestrator_forced`` with ``request_priority=background`` and
    ``workload_class=eval_batch`` (the placement-queue path), pinning
    ``force_role`` to the probed role and ``allow_delegation=False`` -- so a probe
    request is never a foreground /chat request. Never exercised by the tests.
    """
    _bench = str(ORCH_ROOT / "scripts" / "benchmark")
    if _bench not in sys.path:
        sys.path.insert(0, _bench)
    from seeding_orchestrator import call_orchestrator_forced  # type: ignore

    return call_orchestrator_forced(
        prompt="[migration-probe] warm turn; reply with a single token.",
        force_role=spec.role,
        force_mode="direct",
        url=url,
        timeout=timeout,
        client=client,
        allow_delegation=False,
        session_id=spec.session_id,               # reused across turns (WP-3)
        request_priority=PLACEMENT_REQUEST_PRIORITY,  # placement queue, not /chat
        workload_class=PLACEMENT_WORKLOAD_CLASS,
        max_tokens=1,
    )


def _default_counter_probe(
    url: str,
    *,
    role: str | None = None,
) -> dict[str, Any]:  # pragma: no cover
    """Snapshot live migration counters (forward/reverse/thrash) for diffing.

    Best-effort read of the single-worker API's migration counters; the caller
    guarantees a single worker so the counters are attributable. Prefer the
    committed ``src.metrics.migration_counters`` surface when exposed by the
    contention endpoint. Fall back to the per-role scheduling counters that
    already back the dashboard (`migrations_started` / `reverse_migrations`)
    because older API builds did not expose the Prometheus-shaped counters on
    region locks. Returns a dict like
    ``{"forward": int, "reverse": int, "thrash": {reason: int}}``. Any read
    failure yields zeros (the step contributes no observed events).
    """
    import httpx  # deferred; only in the inference path

    try:
        resp = httpx.get(f"{url}/dashboard/api/contention", timeout=10)
        data = resp.json()
        counters = (data or {}).get("migration_counters") or {}
        direction = dict(counters.get("kv_migration_direction_total", {}) or {})
        thrash = dict(counters.get("kv_migration_thrash_skipped_total", {}) or {})
        events = list(counters.get("kv_migration_recent_events", []) or [])
        per_role = (data or {}).get("per_role_scheduling") or {}
        records: list[dict[str, Any]] = []
        if role and isinstance(per_role.get(role), dict):
            records = [per_role[role]]
        elif isinstance(per_role, dict):
            records = [v for v in per_role.values() if isinstance(v, dict)]
        failures = sum(int(r.get("migration_failures", 0) or 0) for r in records)
        if direction or thrash:
            return {
                "forward": int(direction.get("forward", 0) or 0),
                "reverse": int(direction.get("reverse", 0) or 0),
                "thrash": thrash,
                "failures": failures,
                "events": events,
            }
        return {
            "forward": sum(int(r.get("migrations_started", 0) or 0) for r in records),
            "reverse": sum(int(r.get("reverse_migrations", 0) or 0) for r in records),
            "thrash": {},
            "failures": failures,
            "events": events,
        }
    except Exception:  # noqa: BLE001
        return {"forward": 0, "reverse": 0, "thrash": {}}


def _execution_order_for_step(
    specs: Sequence[ProbeRequestSpec],
    *,
    safe_slots: int,
) -> tuple[list[ProbeRequestSpec], bool]:
    """Return per-step request order plus whether to stagger burst handover.

    The first burst request must be an interferer, not the reused primary:
    WP-3 migration triggers when a new session claims the idle full instance
    while the previous primary session is the last full owner. The whole step is
    still run concurrently; this only makes the handover deterministic.
    """
    ordered = list(specs)
    if len(ordered) <= safe_slots:
        return ordered, False
    interferers = [s for s in ordered if not s.is_primary]
    primaries = [s for s in ordered if s.is_primary]
    if not interferers or not primaries:
        return ordered, False
    return [interferers[0], *primaries, *interferers[1:]], True


def execute_migration_probe(  # pragma: no cover - inference path
    plan: MigrationProbePlan,
    *,
    url: str = "http://localhost:8000",
    timeout: int = 300,
    output_path: Path | None = None,
    client: Any | None = None,
    request_fn: Callable[..., dict[str, Any]] | None = None,
    counter_probe: Callable[[str], dict[str, Any]] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Drive the plan over the placement queue and reconstruct observed events.

    Reached ONLY when ``AUTOPILOT_MIGRATION_PROBE_INFERENCE=1`` (via
    ``run_migration_probe``). Single-worker / no-concurrent-inference window is
    the caller's responsibility (bsv pattern). Fires each step's requests over the
    placement queue, diffs the migration counters between steps to reconstruct the
    observed forward/reverse/thrash events, then rolls them up with
    ``analyze_migration_events``. Emits a model/quant-indexed JSONL analysis row.
    Never touches autopilot lifecycle/state; never writes a ledger.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _time

    sleep = sleep_fn or _time.sleep
    probe = counter_probe or (lambda u: _default_counter_probe(u, role=plan.role))
    send = request_fn or (lambda spec: _default_request(spec, client, url=url, timeout=timeout))

    steps = plan.schedule.get("steps") or []
    by_step: dict[int, list[ProbeRequestSpec]] = {}
    for r in plan.requests:
        by_step.setdefault(r.step, []).append(r)

    observed: list[dict[str, Any]] = []
    prev = probe(url)
    for i in range(len(steps)):
        step_specs, stagger_burst = _execution_order_for_step(
            by_step.get(i, []),
            safe_slots=plan.safe_slots,
        )
        if step_specs:
            with ThreadPoolExecutor(max_workers=len(step_specs)) as ex:
                futures = []
                if stagger_burst and len(step_specs) > 1:
                    futures.append(ex.submit(send, step_specs[0]))
                    sleep(DEFAULT_BURST_STAGGER_MS / 1000.0)
                    step_specs = step_specs[1:]
                futures.extend(ex.submit(send, spec) for spec in step_specs)
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception:  # noqa: BLE001 - one bad turn must not abort the probe
                        pass
        sleep(DEFAULT_COUNTER_SETTLE_MS / 1000.0)
        cur = probe(url)
        cur_events = [e for e in (cur.get("events", []) or []) if isinstance(e, dict)]
        prev_seq = max(
            [int(e.get("seq", 0) or 0) for e in (prev.get("events", []) or []) if isinstance(e, dict)]
            or [0]
        )
        new_committed_events = [
            e for e in cur_events if int(e.get("seq", 0) or 0) > prev_seq
        ]
        use_session_events = bool(cur_events or (prev.get("events", []) or []))
        d_fwd = 0 if use_session_events else max(0, int(cur.get("forward", 0)) - int(prev.get("forward", 0)))
        d_rev = 0 if use_session_events else max(0, int(cur.get("reverse", 0)) - int(prev.get("reverse", 0)))
        d_fail = max(0, int(cur.get("failures", 0)) - int(prev.get("failures", 0)))
        t_ms = i * plan.dwell_ms
        for event in new_committed_events:
            direction = str(event.get("direction") or "")
            if direction not in {"forward", "reverse"}:
                continue
            observed.append({
                "direction": direction,
                "session_id": str(event.get("session_id") or "?"),
                "committed": bool(event.get("committed", True)),
                "t_ms": t_ms,
            })
        for _ in range(d_fwd):
            observed.append({"direction": "forward", "session_id": plan.session_id,
                             "committed": True, "t_ms": t_ms})
        for _ in range(d_rev):
            observed.append({"direction": "reverse", "session_id": plan.session_id,
                             "committed": True, "t_ms": t_ms})
        for _ in range(d_fail):
            observed.append({"direction": "forward", "session_id": plan.session_id,
                             "committed": False, "t_ms": t_ms})
        cur_thrash = cur.get("thrash", {}) or {}
        prev_thrash = prev.get("thrash", {}) or {}
        for reason, n in cur_thrash.items():
            delta = max(0, int(n) - int(prev_thrash.get(reason, 0)))
            for _ in range(delta):
                observed.append({"skipped": reason, "session_id": plan.session_id,
                                 "t_ms": t_ms})
        prev = cur
        if i + 1 < len(steps):
            sleep(plan.dwell_ms / 1000.0)

    analysis = analyze_migration_events(
        observed,
        per_session_cap=plan.per_session_cap,
        expected=plan.schedule.get("events"),
    )
    # Model/quant-indexed result row (never role-indexed).
    row = {
        **plan.model_index(),
        "route": "ROUTE-A3-j2j3-single-worker",
        "placement_role": plan.role,
        "transport": PLACEMENT_QUEUE_TRANSPORT,
        "observed_events": observed,
        "analysis": analysis,
    }
    if output_path is not None:
        _append_jsonl(Path(output_path), row)
    return row


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level orchestration (env-gated dry-run vs execute)
# ══════════════════════════════════════════════════════════════════════════════


def run_migration_probe(
    plan: MigrationProbePlan,
    *,
    url: str = "http://localhost:8000",
    timeout: int = 300,
    output_path: Path | None = None,
    client: Any | None = None,
    request_fn: Callable[..., dict[str, Any]] | None = None,
    counter_probe: Callable[[str], dict[str, Any]] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Return the plan as a dry-run OR execute it, gated on the inference flag.

    DEFAULT (``AUTOPILOT_MIGRATION_PROBE_INFERENCE`` unset/false): returns the
    plan as a dry-run and runs NO inference -- the entire surface the tests touch.
    When the flag is set the plan is driven over the placement queue via
    :func:`execute_migration_probe`.
    """
    if not _env_flag_enabled(MIGRATION_PROBE_INFERENCE_ENV):
        return {
            "mode": "dry_run",
            "runner_version": RUNNER_VERSION,
            "inference_ran": False,
            "reason": (
                f"{MIGRATION_PROBE_INFERENCE_ENV} not set; plan returned as a "
                "dry-run (no inference, placement-queue transport)."
            ),
            "n_requests": len(plan.requests),
            "plan": plan.to_dict(),
        }

    row = execute_migration_probe(
        plan, url=url, timeout=timeout, output_path=output_path, client=client,
        request_fn=request_fn, counter_probe=counter_probe, sleep_fn=sleep_fn,
    )
    return {
        "mode": "execute",
        "runner_version": RUNNER_VERSION,
        "inference_ran": True,
        "n_requests": len(plan.requests),
        "output_path": str(output_path) if output_path else None,
        "plan": plan.to_dict(),
        "result": row,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI (__main__) — default is a pure dry-run/plan (validate + resolve + print)
# ══════════════════════════════════════════════════════════════════════════════


def _load_registry(path: Path) -> dict[str, Any]:
    """Load the lean model registry (read-only) for role->model/quant resolution."""
    try:
        import yaml  # deferred; only needed for the CLI resolver
    except Exception:  # noqa: BLE001
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plan (and, gated, execute) the ROUTE-A3 J2/J3 under-traffic KV "
            "migration probe over the placement queue. DEFAULT is a pure dry-run "
            "that validates, resolves, and prints the plan and runs NO inference."
        )
    )
    p.add_argument("--role", default=DEFAULT_ROLE, help="placement role under probe")
    p.add_argument("--session-id", default=DEFAULT_SESSION_ID,
                   help="reused primary session id (forces WP-3 handover)")
    p.add_argument("--oscillation", default=",".join(str(n) for n in DEFAULT_OSCILLATION),
                   help="load profile, comma-separated concurrency per step")
    p.add_argument("--safe-slots", type=int, default=DEFAULT_SAFE_SLOTS,
                   help="role's max safe disjoint concurrent placements")
    p.add_argument("--cooldown-ms", type=int, default=DEFAULT_COOLDOWN_MS)
    p.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS)
    p.add_argument("--per-session-cap", type=int, default=DEFAULT_PER_SESSION_CAP)
    p.add_argument("--dwell-ms", type=int, default=DEFAULT_DWELL_MS)
    p.add_argument("--model", default=None, help="model name (default: resolve from registry)")
    p.add_argument("--quant", default=None, help="quant (default: resolve from registry)")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY),
                   help="lean model registry for role->model/quant resolution")
    p.add_argument("--worker-count", type=int, default=None,
                   help="API worker count (a value >1 raises a single-worker warning)")
    p.add_argument("--url", default="http://localhost:8000", help="orchestrator URL (execute only)")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--output", default=None,
                   help="JSONL path for the model/quant-indexed analysis row (execute only)")
    p.add_argument("--execute", action="store_true",
                   help="attempt execution (STILL env-gated by "
                   f"{MIGRATION_PROBE_INFERENCE_ENV}=1; otherwise falls back to dry-run)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    model, quant = args.model, args.quant
    if model is None or quant is None:
        registry = _load_registry(Path(args.registry))
        r_model, r_quant = resolve_model_quant_for_role(registry, args.role)
        model = model or r_model
        quant = quant or r_quant

    plan = plan_migration_probe(
        role=args.role, model=model, quant=quant, session_id=args.session_id,
        safe_slots=args.safe_slots, oscillation=args.oscillation,
        cooldown_ms=args.cooldown_ms, window_ms=args.window_ms,
        per_session_cap=args.per_session_cap, dwell_ms=args.dwell_ms,
        worker_count=args.worker_count,
    )

    if not args.execute:
        # DEFAULT: pure resolution — no inference, no model, whatever the env says.
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True, default=str))
        return 0

    result = run_migration_probe(
        plan, url=args.url, timeout=args.timeout,
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
