"""BSV-1/BSV-2 core: compute behavior signatures and classify their diffs.

Per `handoffs/active/autopilot-continuous-optimization.md` BSV-1/BSV-2/BSV-3 (intake-607 §5.2.3/§5.2.4).
A behavior signature fingerprints *how* an archive member behaves (not just its scalar score), so
the autopilot can catch a mutation that silently breaks a prior Pareto win even when the aggregate
score looks fine.

Pure functions (no disk/model/autopilot imports). The autopilot accept-path extracts the inputs
from its journal and calls these; storage uses `src.trace.BehaviorSignature` /
`insert_behavior_signature`. Diff severity classes drive BSV-2's accept gate.

Conventions:
- `sentinel_outcomes`: {sentinel_id: outcome}; outcomes 'pass' | 'fail' | 'error' | 'skip' |
  'pass_via_shortcut' (a forbidden shortcut, e.g. web-searched the answer — audit #1/#2).
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from src.trace.harness_schema import BehaviorSignature

PASS_LIKE = {"pass"}
FAIL_LIKE = {"fail", "error"}
SHORTCUT = "pass_via_shortcut"

# Ordered bucket thresholds (upper bound inclusive); index = position.
_LATENCY_MS = [("<1s", 1_000), ("1-5s", 5_000), ("5-30s", 30_000),
               ("30-120s", 120_000), (">120s", float("inf"))]
_TOKENS = [("<1k", 1_000), ("1-4k", 4_000), ("4-16k", 16_000),
           ("16-64k", 64_000), (">64k", float("inf"))]


def _bucket(value: float | None, table: list[tuple[str, float]]) -> tuple[int, str]:
    if value is None:
        return (-1, "unknown")
    for i, (label, hi) in enumerate(table):
        if value <= hi:
            return (i, label)
    return (len(table) - 1, table[-1][0])


def latency_bucket(ms: float | None) -> str:
    return _bucket(ms, _LATENCY_MS)[1]


def token_bucket(n: float | None) -> str:
    return _bucket(n, _TOKENS)[1]


def _hash_seq(seq: Sequence[str] | None) -> str | None:
    if seq is None:
        return None
    return hashlib.sha256("\x1f".join(seq).encode("utf-8")).hexdigest()[:16]


def _hash_obj(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def compute_behavior_signature(
    *,
    archive_member_id: str,
    trial_id: int | None = None,
    sentinel_outcomes: dict[str, str] | None = None,
    answer_text: str | None = None,
    route_path: Sequence[str] | None = None,
    tool_sequence: Sequence[str] | None = None,
    escalation_path: Sequence[str] | None = None,
    latency_ms: float | None = None,
    total_tokens: float | None = None,
    harness_metrics_id: int | None = None,
    oracle_adequacy_version: int | None = None,
    signature_confidence: str = "full",
) -> BehaviorSignature:
    """Build a BehaviorSignature (BSV-1) from a trial's observed behavior.

    `signature_confidence='partial'` should be passed for journal-backfilled members that lack
    some fields, so BSV-2 does not treat them as full-evidence comparisons (audit #4).
    """
    outcomes = sentinel_outcomes or {}
    lat_b = latency_bucket(latency_ms)
    tok_b = token_bucket(total_tokens)
    route_h = _hash_seq(list(route_path) if route_path else None)
    tool_h = _hash_seq(list(tool_sequence) if tool_sequence else None)
    esc_h = _hash_seq(list(escalation_path) if escalation_path else None)
    answer_h = None if answer_text is None else hashlib.sha256(
        answer_text.strip().encode("utf-8")).hexdigest()[:16]

    signature_hash = _hash_obj({
        "outcomes": outcomes, "route": route_h, "tool": tool_h, "esc": esc_h,
        "answer": answer_h, "lat": lat_b, "tok": tok_b,
    })

    return BehaviorSignature(
        archive_member_id=archive_member_id,
        trial_id=trial_id,
        sentinel_outcomes=outcomes,
        answer_hash=answer_h,
        route_path_hash=route_h,
        tool_sequence_hash=tool_h,
        escalation_path_hash=esc_h,
        latency_bucket=lat_b,
        token_bucket=tok_b,
        harness_metrics_id=harness_metrics_id,
        oracle_adequacy_version=oracle_adequacy_version,
        signature_hash=signature_hash,
        signature_confidence=signature_confidence,
    )


# ─── BSV-2 diff severity ─────────────────────────────────────────────────────────


class DiffSeverity:
    BENIGN = "benign"     # format-only / unchanged buckets — safe to auto-accept
    WATCH = "watch"       # route/tool/escalation path changed but outcomes equal — log + accept
    BLOCKING = "blocking"  # prior-pass sentinel regressed, shortcut appeared, or cost guardrail crossed

    RANK = {BENIGN: 0, WATCH: 1, BLOCKING: 2}


def _as_dict(sig) -> dict:
    if isinstance(sig, BehaviorSignature):
        return {
            "sentinel_outcomes": sig.sentinel_outcomes or {},
            "route_path_hash": sig.route_path_hash,
            "tool_sequence_hash": sig.tool_sequence_hash,
            "escalation_path_hash": sig.escalation_path_hash,
            "latency_bucket": sig.latency_bucket,
            "token_bucket": sig.token_bucket,
            "signature_confidence": sig.signature_confidence,
        }
    d = dict(sig)
    out = d.get("sentinel_outcomes") or {}
    if isinstance(out, str):
        out = json.loads(out)
    d["sentinel_outcomes"] = out
    return d


def diff_signatures(old, new, *, cost_guardrail_buckets: int = 2) -> tuple[str, list[str]]:
    """Classify the behavioral delta old→new. Returns (severity, reasons).

    BLOCKING wins over WATCH wins over BENIGN. `cost_guardrail_buckets` = how many buckets a
    latency/token regression may move before it is treated as blocking (default 2).
    """
    o, n = _as_dict(old), _as_dict(new)
    reasons: list[str] = []
    severity = DiffSeverity.BENIGN

    def bump(level: str, reason: str) -> None:
        nonlocal severity
        reasons.append(reason)
        if DiffSeverity.RANK[level] > DiffSeverity.RANK[severity]:
            severity = level

    o_out, n_out = o["sentinel_outcomes"], n["sentinel_outcomes"]

    # 1. regressions + shortcuts (blocking)
    for sid, old_outcome in o_out.items():
        new_outcome = n_out.get(sid)
        if new_outcome is None:
            bump(DiffSeverity.WATCH, f"sentinel {sid} disappeared")
            continue
        if old_outcome in PASS_LIKE and new_outcome in FAIL_LIKE:
            bump(DiffSeverity.BLOCKING, f"sentinel {sid} regressed {old_outcome}->{new_outcome}")
        if new_outcome == SHORTCUT and old_outcome != SHORTCUT:
            bump(DiffSeverity.BLOCKING, f"sentinel {sid} now passes via forbidden shortcut")
    for sid in n_out:
        if sid not in o_out:
            bump(DiffSeverity.WATCH, f"new sentinel {sid}")

    # 2. cost guardrail (blocking if worse by >= cost_guardrail_buckets)
    for field, table in (("latency_bucket", _LATENCY_MS), ("token_bucket", _TOKENS)):
        oi = _label_index(o.get(field), table)
        ni = _label_index(n.get(field), table)
        if oi >= 0 and ni >= 0 and ni > oi:
            delta = ni - oi
            if delta >= cost_guardrail_buckets:
                bump(DiffSeverity.BLOCKING, f"{field} regressed {o.get(field)}->{n.get(field)} ({delta} buckets)")
            else:
                bump(DiffSeverity.WATCH, f"{field} worsened {o.get(field)}->{n.get(field)}")

    # 3. path changes with equal outcomes (watch)
    for field in ("route_path_hash", "tool_sequence_hash", "escalation_path_hash"):
        if o.get(field) != n.get(field):
            bump(DiffSeverity.WATCH, f"{field} changed")

    # partial-confidence comparisons cannot certify BENIGN (audit #4)
    if severity == DiffSeverity.BENIGN and (
        o.get("signature_confidence") == "partial" or n.get("signature_confidence") == "partial"
    ):
        bump(DiffSeverity.WATCH, "partial-confidence signature — cannot certify benign")

    if not reasons:
        reasons.append("no material behavioral change")
    return severity, reasons


def _label_index(label: str | None, table: list[tuple[str, float]]) -> int:
    if not label:
        return -1
    for i, (lbl, _hi) in enumerate(table):
        if lbl == label:
            return i
    return -1
