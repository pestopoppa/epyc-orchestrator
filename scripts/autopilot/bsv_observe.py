"""J11 / BSV-2 observe-only behavior-signature diff for autopilot trials.

Mirrors `hle_metrics.py`: a pure helper that computes a diagnostic payload only. It does NOT feed
SafetyGate, ParetoArchive, routing, or baseline mutation. The autopilot's observe-only J11 run
decides later whether the differential accept-test has enough signal to ever gate acceptance.

Scope honesty (review 2026-05-27, finding #2): an autopilot trial's `EvalResult` is still a
*trial-level aggregate*. It exposes `routing_distribution`, `per_suite_quality`, compact
`question_results`, cost/token stats and the HLE ids, but NOT full answer traces. The signature is
built from what is reliably present and tagged `signature_confidence="partial"`; `diff_signatures`
then refuses to certify a partial comparison as BENIGN (bumps to WATCH), which is the correct
conservative behaviour here.

What the coarse signature captures + what the diff therefore flags:
  - route_path  = sorted roles, each tagged with a coarse weight quartile -> WATCH on a role-set
                  change OR a major weight shift across the same roles (finding #3)
  - tool_sequence = coarse aggregate tool-call/rate/name buckets when present -> WATCH on tool drift
  - escalation_path = compact per-question route aggregate when present      -> WATCH on path drift
  - sentinel_outcomes = per-question pass/fail when available, else per-suite proxy (>=2.0/3)
                                                               -> BLOCKING on prior pass->fail
  - latency_bucket/token_bucket = mean request latency + avg prompt tokens  -> WATCH/BLOCKING on
                                                                                cost regression
"""
from __future__ import annotations

from typing import Any

BSV_METRIC_VERSION = "bsv-2-observe-v1"
BSV_CONFLICT_VERSION = "bsv-3-conflict-ledger-v1"

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


def _question_outcomes(question_results: Any) -> dict[str, str]:
    """Normalize compact per-question eval rows into BSV sentinel outcomes."""
    if not isinstance(question_results, list):
        return {}

    out: dict[str, str] = {}
    for item in question_results:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("qid") or item.get("question_id") or "").strip()
        if not qid:
            continue
        out[qid] = "pass" if bool(item.get("correct")) else "fail"
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


def _count_bucket(n: float | int | None) -> str:
    try:
        value = float(n)
    except (TypeError, ValueError):
        return "n?"
    if value <= 0:
        return "n0"
    if value <= 1:
        return "n1"
    if value <= 4:
        return "n2-4"
    if value <= 9:
        return "n5-9"
    if value <= 24:
        return "n10-24"
    return "n25+"


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _details(eval_result: Any) -> dict[str, Any]:
    details = getattr(eval_result, "details", {}) or {}
    return details if isinstance(details, dict) else {}


def _question_rows(eval_result: Any) -> list[dict[str, Any]]:
    rows = getattr(eval_result, "question_results", None)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _tool_sequence(eval_result: Any, question_results: list[dict[str, Any]]) -> tuple[list[str] | None, str]:
    """Coarse, stable tool-use aggregate for the BSV tool sequence hash.

    The observe path has per-question counts/names, not a globally ordered call trace. Hashing coarse
    counts is intentional: it catches "tools disabled vs used" and large mix changes without making
    every single-call wobble look like a distinct behavior class.
    """
    tool_name_counts: dict[str, int] = {}
    total_tools = 0
    tool_questions = 0
    saw_signal = False

    for row in question_results:
        tools_used = row.get("tools_used")
        if tools_used is not None:
            saw_signal = True
            try:
                used = max(0, int(tools_used))
            except (TypeError, ValueError):
                used = 0
            total_tools += used
            if used > 0:
                tool_questions += 1
        tools_called = row.get("tools_called")
        if isinstance(tools_called, list):
            saw_signal = True
            for name in tools_called:
                label = str(name).strip()
                if label:
                    tool_name_counts[label] = tool_name_counts.get(label, 0) + 1

    source = "question_results.tools_used"
    details = _details(eval_result)
    if not saw_signal:
        raw_counts = details.get("tool_name_counts")
        if isinstance(raw_counts, dict):
            for name, count in raw_counts.items():
                label = str(name).strip()
                if not label:
                    continue
                try:
                    tool_name_counts[label] = int(count)
                except (TypeError, ValueError):
                    continue
            if tool_name_counts:
                saw_signal = True
                total_tools = sum(tool_name_counts.values())
                source = "details.tool_name_counts"

    if not saw_signal:
        raw_total = getattr(eval_result, "total_tool_calls", None)
        if raw_total is None:
            raw_total = details.get("total_tool_calls")
        if raw_total is not None:
            saw_signal = True
            try:
                total_tools = max(0, int(raw_total))
            except (TypeError, ValueError):
                total_tools = 0
            source = "eval_result.total_tool_calls"

    if not saw_signal:
        return None, "none"

    raw_rate = _float_or_none(getattr(eval_result, "tool_use_rate", None))
    if raw_rate is None:
        raw_rate = _float_or_none(details.get("tool_use_rate"))
    if raw_rate is None and question_results:
        raw_rate = tool_questions / len(question_results)

    seq = [f"tool_total:{_count_bucket(total_tools)}"]
    if raw_rate is not None:
        seq.append(f"tool_rate:{_weight_bucket(raw_rate)}")
    for name, count in sorted(tool_name_counts.items()):
        seq.append(f"tool:{name}:{_count_bucket(count)}")
    return seq, source


def _escalation_path(question_results: list[dict[str, Any]]) -> tuple[list[str] | None, str]:
    route_counts: dict[str, int] = {}
    for row in question_results:
        route = str(row.get("route") or "").strip()
        if route:
            route_counts[route] = route_counts.get(route, 0) + 1
    if not route_counts:
        return None, "none"
    return [f"route:{route}:{_count_bucket(count)}" for route, count in sorted(route_counts.items())], (
        "question_results.route"
    )


def _latency_ms(eval_result: Any, question_results: list[dict[str, Any]]) -> tuple[float | None, str]:
    latencies = [
        value
        for value in (_float_or_none(row.get("latency_ms")) for row in question_results)
        if value is not None and value >= 0
    ]
    if latencies:
        return sum(latencies) / len(latencies), "question_results.latency_ms_mean"

    details = _details(eval_result)
    sum_request_elapsed_s = _float_or_none(getattr(eval_result, "sum_request_elapsed_s", None))
    if sum_request_elapsed_s is None:
        sum_request_elapsed_s = _float_or_none(details.get("sum_request_elapsed_s"))
    n_questions = _float_or_none(getattr(eval_result, "n_questions", None))
    if not n_questions and question_results:
        n_questions = float(len(question_results))
    if sum_request_elapsed_s is not None and n_questions and n_questions > 0:
        return (sum_request_elapsed_s / n_questions) * 1000.0, "eval_result.sum_request_elapsed_s_mean"

    eval_wall_s = _float_or_none(getattr(eval_result, "eval_wall_s", None))
    if eval_wall_s is None:
        eval_wall_s = _float_or_none(details.get("eval_wall_s"))
    if eval_wall_s is not None and eval_wall_s >= 0:
        return eval_wall_s * 1000.0, "eval_result.eval_wall_s"

    return None, "none"


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
    question_rows = _question_rows(eval_result)
    question_outcomes = _question_outcomes(question_rows)
    suite_outcomes = _suite_outcomes(per_suite)
    sentinel_outcomes = question_outcomes or suite_outcomes
    sentinel_outcome_source = (
        "question_results"
        if question_outcomes
        else "suite_quality_proxy"
        if suite_outcomes
        else "none"
    )
    oracle = getattr(eval_result, "oracle_adequacy", {}) or {}
    avg_tokens = float(getattr(eval_result, "avg_prompt_tokens", 0.0) or 0.0)
    tool_sequence, tool_sequence_source = _tool_sequence(eval_result, question_rows)
    escalation_path, escalation_path_source = _escalation_path(question_rows)
    latency_ms, latency_source = _latency_ms(eval_result, question_rows)

    sig = compute_behavior_signature(
        archive_member_id=str(archive_member_id or species_name or "?"),
        trial_id=trial_id,
        sentinel_outcomes=sentinel_outcomes,
        route_path=_route_path(routing),
        tool_sequence=tool_sequence,
        escalation_path=escalation_path,
        latency_ms=latency_ms,
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
        "sentinel_outcome_source": sentinel_outcome_source,
        "sentinel_outcome_count": len(sentinel_outcomes),
        "process_signal_sources": {
            "tool_sequence": tool_sequence_source,
            "escalation_path": escalation_path_source,
            "latency_ms": latency_source,
        },
    }
    if incumbent_signature is not None:
        severity, reasons = diff_signatures(incumbent_signature, sig_dict)
        payload["severity"] = severity
        payload["reasons"] = reasons[:8]
    return payload


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        return [str(k) for k in sorted(value)]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    return [str(value)]


def _path_subsystem(path: str) -> str:
    p = path.replace("\\", "/").lower()
    if "prompt" in p or p.endswith(".md"):
        return "prompt"
    if "routing" in p or "router" in p:
        return "routing"
    if "stack" in p or "model_registry" in p or "orchestration/" in p:
        return "stack"
    if "eval" in p or "benchmark" in p:
        return "eval"
    if "memory" in p or "repl_memory" in p:
        return "memory"
    return "code"


def _action_subsystem(action: dict[str, Any], files: list[str]) -> str:
    explicit = action.get("subsystem") or action.get("surface") or action.get("target")
    if explicit:
        text = str(explicit).lower()
        if "routing" in text or "router" in text:
            return "routing"
        if "prompt" in text or "forge" in text:
            return "prompt"
        if "stack" in text or "model" in text:
            return "stack"
        if "eval" in text or "tower" in text:
            return "eval"
        if "memory" in text:
            return "memory"
        return text
    if files:
        subsystems = sorted({_path_subsystem(path) for path in files})
        if len(subsystems) == 1:
            return subsystems[0]
        return "+".join(subsystems)
    action_type = str(action.get("type") or "unknown")
    if action_type == "seed_batch":
        return "seeding"
    if action_type == "structural_experiment":
        return "feature_flags"
    if action_type == "numeric_trial":
        return "numeric"
    return action_type


def _extract_files(action: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for key in ("file", "target_file", "path"):
        files.extend(_listify(action.get(key)))
    files.extend(_listify(action.get("files")))
    return sorted({f for f in files if f and f not in {"None", "null"}})


def _extract_prompt_sections(action: dict[str, Any]) -> list[str]:
    sections: list[str] = []
    for key in ("section", "sections", "prompt_section", "prompt_sections"):
        sections.extend(_listify(action.get(key)))
    return sorted(set(sections))


def _extract_feature_flags(action: dict[str, Any]) -> dict[str, Any]:
    flags = action.get("flags")
    if isinstance(flags, dict):
        return {str(k): v for k, v in sorted(flags.items())}
    return {}


def _signature_delta(
    bsv_payload: dict[str, Any],
    incumbent_signature: dict[str, Any] | None,
) -> dict[str, Any]:
    signature = bsv_payload.get("signature") or {}
    old_outcomes = (incumbent_signature or {}).get("sentinel_outcomes") or {}
    new_outcomes = signature.get("sentinel_outcomes") or {}
    improved: list[str] = []
    regressed: list[str] = []
    for sid, old in old_outcomes.items():
        new = new_outcomes.get(sid)
        if old == "fail" and new == "pass":
            improved.append(str(sid))
        elif old == "pass" and new in {"fail", "error"}:
            regressed.append(str(sid))

    changed_fields = []
    if incumbent_signature:
        for field in (
            "route_path_hash",
            "tool_sequence_hash",
            "escalation_path_hash",
            "latency_bucket",
            "token_bucket",
        ):
            if (incumbent_signature or {}).get(field) != signature.get(field):
                changed_fields.append(field)

    return {
        "severity": bsv_payload.get("severity"),
        "reasons": list(bsv_payload.get("reasons") or [])[:8],
        "signature_hash": bsv_payload.get("signature_hash"),
        "signature_confidence": bsv_payload.get("signature_confidence"),
        "changed_fields": changed_fields,
        "improved_sentinels": sorted(improved),
        "regressed_sentinels": sorted(regressed),
    }


def build_mutation_dependency_entry(
    *,
    trial_id: int,
    action: dict[str, Any],
    parent_trial: int | None,
    bsv_payload: dict[str, Any],
    incumbent_signature: dict[str, Any] | None,
    pareto_status: str,
) -> dict[str, Any]:
    """Build a BSV-3 mutation-dependency ledger row for an accepted mutation."""
    files = _extract_files(action)
    return {
        "version": BSV_CONFLICT_VERSION,
        "trial_id": int(trial_id),
        "action_type": str(action.get("type") or ""),
        "subsystem": _action_subsystem(action, files),
        "files_touched": files,
        "prompt_sections_touched": _extract_prompt_sections(action),
        "feature_flags": _extract_feature_flags(action),
        "behavior_signature_delta": _signature_delta(bsv_payload, incumbent_signature),
        "parent_trial": parent_trial,
        "pareto_status": pareto_status,
        "archive_member_id": bsv_payload.get("archive_member_id"),
    }


def _entry_overlap(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if a.get("subsystem") and a.get("subsystem") == b.get("subsystem"):
        reasons.append(f"same subsystem {a['subsystem']}")
    for key, label in (
        ("files_touched", "file"),
        ("prompt_sections_touched", "prompt section"),
    ):
        shared = sorted(set(a.get(key) or []) & set(b.get(key) or []))
        if shared:
            reasons.append(f"shared {label}: {', '.join(shared[:4])}")
    shared_flags = sorted(set((a.get("feature_flags") or {}).keys()) & set((b.get("feature_flags") or {}).keys()))
    if shared_flags:
        reasons.append(f"shared feature flag: {', '.join(shared_flags[:4])}")
    return reasons


def _conflict_severity(new_entry: dict[str, Any], prior: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = _entry_overlap(new_entry, prior)
    if not reasons:
        return "none", []

    severity = "watch"
    new_delta = new_entry.get("behavior_signature_delta") or {}
    old_delta = prior.get("behavior_signature_delta") or {}
    if new_delta.get("severity") == "blocking" or old_delta.get("severity") == "blocking":
        severity = "blocking"

    new_changed = set(new_delta.get("changed_fields") or [])
    old_changed = set(old_delta.get("changed_fields") or [])
    if new_changed and old_changed and new_changed != old_changed:
        reasons.append(
            "different behavior surfaces changed: "
            f"new={sorted(new_changed)}, prior={sorted(old_changed)}"
        )

    new_improved = set(new_delta.get("improved_sentinels") or [])
    old_improved = set(old_delta.get("improved_sentinels") or [])
    new_regressed = set(new_delta.get("regressed_sentinels") or [])
    old_regressed = set(old_delta.get("regressed_sentinels") or [])
    if (new_improved and old_improved and new_improved.isdisjoint(old_improved)) or (
        new_regressed & old_improved
    ) or (old_regressed & new_improved):
        severity = "blocking"
        reasons.append("opposing or disjoint sentinel movement across accepted mutations")

    return severity, reasons


def build_conflict_report(
    new_entry: dict[str, Any],
    existing_ledger: list[dict[str, Any]] | None,
    *,
    max_conflicts: int = 8,
) -> dict[str, Any]:
    """Compare a new accepted mutation against the existing dependency ledger."""
    conflicts: list[dict[str, Any]] = []
    worst = "none"
    rank = {"none": 0, "watch": 1, "blocking": 2}
    for prior in existing_ledger or []:
        if not isinstance(prior, dict):
            continue
        severity, reasons = _conflict_severity(new_entry, prior)
        if severity == "none":
            continue
        if rank[severity] > rank[worst]:
            worst = severity
        conflicts.append(
            {
                "prior_trial": prior.get("trial_id"),
                "prior_action_type": prior.get("action_type"),
                "prior_subsystem": prior.get("subsystem"),
                "severity": severity,
                "reasons": reasons[:8],
            }
        )

    return {
        "version": BSV_CONFLICT_VERSION,
        "severity": worst,
        "conflicts": conflicts[:max_conflicts],
        "conflict_count": len(conflicts),
    }
