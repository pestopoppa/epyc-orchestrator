"""RTG-DUR-1: when is a per-role wall-clock duration baseline still TRUE?

`orchestration/derived/duration_baselines_by_role.json` prices the reward's speed
axis (RTG-09). It is a MEASURED distribution of how long a role's tasks took on the
model that served the role at the time. It was static: generated once, and nothing
noticed when a role's model changed (`architect_general` Flash-Next since 2026-09-27
was priced against the pre-swap distribution) or when the kernel / reward era moved.

This module is the refusal rule, pure and read-only: given the artifact, the CURRENT
role->model binding (derived `stack_priors.yaml`) and the era registry, it says per
role whether the entry may price a reward and, if not, why. It never regenerates
anything: regeneration changes the reward, which is an era decision for the operator
(`instrument_eras.yaml` is human-amendment-only).

Refusal reasons (a role is USABLE only with none):
  no_model_stamp        entry carries no `bound_model_id`: a v1 artifact cannot prove
                        which model its numbers describe, so it is unverifiable
  model_changed         the role's current model differs from the stamped one
  role_not_bound        the role is absent from the live stack (retired / never bound)
  window_before_era     the entry's first observation predates the latest era boundary
                        (routing_reward / autopilot_speed / cpu_bench): the distribution
                        straddles a reward or kernel change and prices neither side
  expired               the artifact is older than `max_age_days`
  low_n                 fewer than `min_n` observations

Why per ROLE and not per model: progress logs record `producer_role` only, never the
model, so a duration cannot be attributed to a model after the fact. The model stamp is
therefore a BINDING PROOF (the role was bound to this model when the window was
derived, and still is), not a re-keying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

# Era scopes whose boundary invalidates a duration distribution: E18 changed what the
# reward does with duration, a kernel cutover changes how long a token takes, and an
# autopilot_speed boundary is the relaunch fence. Eval-quality eras do not move wall
# clock, so they are deliberately absent.
ERA_FENCE_SCOPES = ("routing_reward", "autopilot_speed", "cpu_bench")

DEFAULT_MAX_AGE_DAYS = 60
DEFAULT_MIN_N = 30

REASONS = (
    "no_model_stamp",
    "model_changed",
    "role_not_bound",
    "window_before_era",
    "expired",
    "low_n",
)


@dataclass
class RoleVerdict:
    role: str
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return not self.reasons


@dataclass
class FreshnessReport:
    verdicts: dict[str, RoleVerdict]
    era_floor_utc: str | None
    era_floor_id: str | None
    unbaselined_live_roles: list[str]

    @property
    def usable_roles(self) -> set[str]:
        return {r for r, v in self.verdicts.items() if v.usable}

    @property
    def stale_roles(self) -> dict[str, list[str]]:
        return {r: v.reasons for r, v in self.verdicts.items() if not v.usable}

    @property
    def ok(self) -> bool:
        """True only if every baselined role is usable AND no live role lacks a baseline."""
        return not self.stale_roles and not self.unbaselined_live_roles

    def lines(self) -> list[str]:
        out = []
        for role, reasons in sorted(self.stale_roles.items()):
            out.append(f"duration baseline for {role}: STALE ({', '.join(reasons)})")
        for role in self.unbaselined_live_roles:
            out.append(f"duration baseline for live role {role}: MISSING (duration term skipped)")
        return out


def parse_ts(value: Any) -> datetime | None:
    """Parse an ISO timestamp (Z or offset, naive = UTC) or a bare date; None if not one."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        # YAML may have loaded an unquoted date as datetime.date
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def current_role_bindings(stack_priors_path: Path) -> dict[str, str]:
    """{role: model_id} for every LIVE-STACK role in the derived stack priors."""
    doc = yaml.safe_load(Path(stack_priors_path).read_text(encoding="utf-8")) or {}
    roles = doc.get("roles") or {}
    out: dict[str, str] = {}
    for name, rec in roles.items():
        if not isinstance(rec, Mapping):
            continue
        if rec.get("deployment_status") != "live_stack":
            continue
        model_id = rec.get("model_id")
        if isinstance(model_id, str) and model_id:
            out[str(rec.get("role") or name)] = model_id
    return out


def era_floor(
    eras_path: Path,
    *,
    now: datetime | None = None,
    scopes: Iterable[str] = ERA_FENCE_SCOPES,
) -> tuple[datetime | None, str | None]:
    """Latest era boundary (`from` <= now) among `scopes`; (None, None) if there is none.

    Rows without `from` (candidate-binary bindings) carry no cutover claim and are
    skipped, same as the registry's own consumers.
    """
    now = now or datetime.now(timezone.utc)
    doc = yaml.safe_load(Path(eras_path).read_text(encoding="utf-8")) or {}
    scopes = set(scopes)
    best: tuple[datetime, str] | None = None
    for era in doc.get("eras") or []:
        if not isinstance(era, Mapping) or era.get("scope") not in scopes:
            continue
        start = parse_ts(era.get("from"))
        if start is None or start > now:
            continue
        if best is None or start >= best[0]:
            best = (start, str(era.get("id")))
    return (best[0], best[1]) if best else (None, None)


def evaluate(
    doc: Mapping[str, Any],
    bindings: Mapping[str, str],
    *,
    floor: datetime | None,
    floor_id: str | None = None,
    now: datetime | None = None,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    min_n: int = DEFAULT_MIN_N,
) -> FreshnessReport:
    """Verdict per baselined role. Pure: no I/O, no clock unless `now` is omitted."""
    now = now or datetime.now(timezone.utc)
    verdicts: dict[str, RoleVerdict] = {}
    generated = parse_ts(doc.get("generated_at_utc"))
    expired = (
        generated is None or (now - generated).total_seconds() > max_age_days * 86400.0
    )
    roles = doc.get("roles") if isinstance(doc.get("roles"), Mapping) else {}
    for role, stats in roles.items():
        verdict = RoleVerdict(role=str(role))
        stats = stats if isinstance(stats, Mapping) else {}
        stamp = stats.get("bound_model_id")
        if not isinstance(stamp, str) or not stamp:
            verdict.reasons.append("no_model_stamp")
        if role not in bindings:
            verdict.reasons.append("role_not_bound")
        elif isinstance(stamp, str) and stamp and stamp != bindings[role]:
            verdict.reasons.append("model_changed")
            verdict.detail["stamped_model"] = stamp
            verdict.detail["current_model"] = bindings[role]
        first = parse_ts(stats.get("first_ts"))
        if floor is not None and (first is None or first < floor):
            verdict.reasons.append("window_before_era")
            verdict.detail["era_floor"] = floor_id or floor.isoformat()
            verdict.detail["first_ts"] = stats.get("first_ts")
        if expired:
            verdict.reasons.append("expired")
        n = stats.get("n")
        if not isinstance(n, int) or n < min_n:
            verdict.reasons.append("low_n")
        verdicts[str(role)] = verdict
    unbaselined = sorted(r for r in bindings if r not in verdicts)
    return FreshnessReport(
        verdicts=verdicts,
        era_floor_utc=floor.isoformat() if floor else None,
        era_floor_id=floor_id,
        unbaselined_live_roles=unbaselined,
    )


def check_files(
    baselines_path: Path,
    stack_priors_path: Path,
    eras_path: Path,
    *,
    now: datetime | None = None,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    min_n: int = DEFAULT_MIN_N,
) -> FreshnessReport:
    """File-level wrapper. A missing/unreadable artifact is every role stale, not an exception."""
    import json

    now = now or datetime.now(timezone.utc)
    bindings = current_role_bindings(stack_priors_path)
    floor, floor_id = era_floor(eras_path, now=now)
    try:
        doc = json.loads(Path(baselines_path).read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            doc = {}
    except (OSError, ValueError):
        doc = {}
    return evaluate(
        doc, bindings, floor=floor, floor_id=floor_id, now=now,
        max_age_days=max_age_days, min_n=min_n,
    )
