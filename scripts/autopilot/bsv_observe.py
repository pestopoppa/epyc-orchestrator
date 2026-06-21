"""J11 / BSV-2 observe-only behavior-signature diff for autopilot trials.

Mirrors `hle_metrics.py`: a pure helper that computes a diagnostic payload only. It does NOT feed
SafetyGate, ParetoArchive, routing, or baseline mutation. The autopilot's observe-only J11 run
decides later whether the differential accept-test has enough signal to ever gate acceptance.

Scope honesty (review 2026-05-27, finding #2): an autopilot trial's `EvalResult` is a *trial-level
aggregate*. It exposes `routing_distribution`, `per_suite_quality`, cost/token stats and the HLE
ids, but NOT per-request `tool_sequence` / `escalation_path` / true per-sentinel outcomes (those
live in the URE trace store — a richer follow-up). So the signature is built from what is reliably
present and tagged `signature_confidence="partial"`; `diff_signatures` then refuses to certify a
partial comparison as BENIGN (bumps to WATCH), which is the correct conservative behaviour here.

What the coarse signature captures + what the diff therefore flags:
  - route_path  = sorted roles, each tagged with a coarse weight quartile -> WATCH on a role-set
                  change OR a major weight shift across the same roles (finding #3)
  - sentinel_outcomes = per_suite pass/fail proxy (>=2.0/3)  -> BLOCKING on a suite pass->fail
  - token_bucket = avg_prompt_tokens                         -> WATCH/BLOCKING on cost regression
"""
from __future__ import annotations

from typing import Any

BSV_METRIC_VERSION = "bsv-2-observe-v1"

# per_suite_quality is on a 0-3 scale; >= this is treated as a suite-level "pass" proxy. This is a
# coarse stand-in for true per-sentinel outcomes (which need the URE trace store) and is the reason
# the signature is emitted at "partial" confidence.
SUITE_PASS_QUALITY = 2.0


def _suite_outcomes(per_suite_quality: dict[str, float] | None) -> dict[str, str]:
    """Coarse suite-level pass/fail proxy for `sentinel_outcomes`, using the BSV vocab
    ('pass'/'fail'). True per-sentinel outcomes are a URE-trace follow-up."""
    out: dict[str, str] = {}
    for suite, q in (per_suite_quality or {}).items():
        try:
            out[str(suite)] = "pass" if float(q) >= SUITE_PASS_QUALITY else "fail"
        except (TypeError, ValueError):
            continue
    return out


def _weight_bucket(w: float) -> str:
    """Coarse quartile of a routing-weight fraction, so a major distribution shift across the
    SAME roles still changes the route-path hash (finding #3). q1<.25 q2<.5 q3<.75 q4>=.75."""
    try:
        w = float(w)
    except (TypeError, ValueError):
        return "q?"
    return "q1" if w < 0.25 else "q2" if w < 0.5 else "q3" if w < 0.75 else "q4"


def _route_path(routing_distribution: dict[str, float] | None) -> list[str] | None:
    """Routing fingerprint = each role tagged with its coarse weight quartile (sorted), so BOTH a
    role-set change AND a major weight shift across the same roles register in route_path_hash."""
    items = sorted((routing_distribution or {}).items())
    return [f"{role}:{_weight_bucket(w)}" for role, w in items] or None


def compute_bsv_observe_payload(
    eval_result: Any,
    *,
    species_name: str,
    trial_id: int,
    incumbent_signature: dict | None,
    archive_member_id: str | None = None,
    incumbent_archive_member_id: str | None = None,
) -> dict[str, Any]:
    """Build this trial's coarse behavior signature and, if an incumbent exists, its diff severity.

    Pure + observe-only. Returns a JSON-serialisable dict suitable for journaling and for storing
    back as the next incumbent. Any input gap degrades gracefully (fields default to None/{}).
    """
    from src.behavior_signature import (  # local import keeps this module dependency-light
        compute_behavior_signature,
        diff_signatures,
        _as_dict,
    )

    routing = getattr(eval_result, "routing_distribution", {}) or {}
    per_suite = getattr(eval_result, "per_suite_quality", {}) or {}
    oracle = getattr(eval_result, "oracle_adequacy", {}) or {}
    avg_tokens = float(getattr(eval_result, "avg_prompt_tokens", 0.0) or 0.0)

    sig = compute_behavior_signature(
        archive_member_id=str(archive_member_id or species_name or "?"),
        trial_id=trial_id,
        sentinel_outcomes=_suite_outcomes(per_suite),
        route_path=_route_path(routing),
        total_tokens=avg_tokens or None,
        harness_metrics_id=None,        # no real trace-store ID yet (finding #2); see diagnostics below
        oracle_adequacy_version=None,   # len(oracle) is a count, not a version (finding #2)
        signature_confidence="partial",  # trial-level aggregate, not per-request evidence
    )
    sig_dict = _as_dict(sig)

    payload: dict[str, Any] = {
        "bsv_metric_version": BSV_METRIC_VERSION,
        "archive_member_id": sig.archive_member_id,
        "incumbent_archive_member_id": incumbent_archive_member_id,
        "signature_hash": sig.signature_hash,
        "signature": sig_dict,
        "signature_confidence": "partial",
        "severity": None,
        "reasons": [],
        "compared_to_incumbent": incumbent_signature is not None,
        # explicitly-named diagnostics — NOT signature IDs (finding #2). Journaled here, not folded
        # into the signature hash (which strips IDs anyway).
        "metric_schema_version": int(getattr(eval_result, "metric_schema_version", 1) or 1),
        "oracle_adequacy_count": len(oracle),
    }
    if incumbent_signature is not None:
        severity, reasons = diff_signatures(incumbent_signature, sig_dict)
        payload["severity"] = severity
        payload["reasons"] = reasons[:8]
    return payload
