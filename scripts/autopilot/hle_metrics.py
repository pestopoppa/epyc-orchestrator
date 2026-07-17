"""HLE-1/HLE-2 observe-only metrics for autopilot trials.

These helpers intentionally compute diagnostic payloads only. They do not feed
SafetyGate, ParetoArchive, routing, or baseline mutation. J9's observe-only run
decides later whether any axis has enough signal to promote.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

METRIC_SCHEMA_VERSION = 1
HARNESS_METRIC_VERSION = "hle-1-rule-v1"
ORACLE_METRIC_VERSION = "hle-2-oracle-defaults-v1"
CONTROL_ATTESTATION_VERSION = "w5-control-pair-report-v1"
CONTROL_ATTESTATION_ENV = "AUTOPILOT_ORACLE_CONTROL_ATTESTATION"

_KEYWORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{3,}")
_STOPWORDS = {
    "action",
    "analysis",
    "because",
    "config",
    "failure",
    "quality",
    "should",
    "trial",
    "unknown",
    "without",
}
_MEMORY_FAILURE_TERMS = (
    "forgot",
    "lost context",
    "missing context",
    "nameerror",
    "undefined",
    "not defined",
    "no such file",
    "file not found",
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _axis(
    *,
    score: float | None,
    confidence: float,
    evidence_keys: list[str],
    missing: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "score": None if score is None else round(_clamp(score), 4),
        "confidence": round(_clamp(confidence), 4),
        # Event ids are empty until trace-store ingestion assigns stable ids.
        # This keeps observe-only metrics out of Pareto promotion until J9
        # analysis can prove enough evidence linkage.
        "evidence_event_ids": [],
        "evidence_keys": evidence_keys,
        "missing": missing,
        "reason": reason,
    }


def _keywords(text: str) -> set[str]:
    return {tok.lower() for tok in _KEYWORD_RE.findall(text or "") if tok.lower() not in _STOPWORDS}


def _keyword_overlap(reference: str, candidate: str) -> float:
    ref = _keywords(reference)
    if not ref:
        return 0.0
    cand = _keywords(candidate)
    return len(ref & cand) / len(ref)


def _trace_memory_score(recent_traces: str) -> tuple[float | None, float, str]:
    if not recent_traces.strip():
        return None, 0.0, "no recent inference trace captured"
    lower = recent_traces.lower()
    hits = sum(lower.count(term) for term in _MEMORY_FAILURE_TERMS)
    score = 1.0 - min(0.7, hits * 0.15)
    structured_hint = any(marker in recent_traces for marker in ("ROLE", "PROMPT", "RESPONSE"))
    confidence = 0.55 if structured_hint else 0.35
    reason = "" if hits == 0 else f"{hits} context-loss/error marker(s) in recent traces"
    return score, confidence, reason


def compute_harness_metrics(
    eval_result: Any,
    *,
    action: dict[str, Any] | None = None,
    verdict: Any | None = None,
    failure_analysis: str = "",
    prior_criticism: str = "",
    recent_traces: str = "",
) -> dict[str, Any]:
    """Compute rule-based HLE-1 axes from current trial evidence.

    The output is deliberately evidence-rich and policy-inert. Missing axes stay
    explicit instead of being imputed, so the later J9 analysis can measure
    missingness honestly.
    """
    details = getattr(eval_result, "details", {}) or {}
    total = _safe_int(details.get("total"), _safe_int(getattr(eval_result, "n_questions", 0)))
    errors = _safe_int(details.get("errors"), 0)
    partial = _safe_int(getattr(eval_result, "partial_count", 0), 0)
    degraded = _safe_int(getattr(eval_result, "degraded_count", 0), 0)
    quality_norm = _clamp(_safe_float(getattr(eval_result, "quality", 0.0)) / 3.0)
    reliability = _clamp(_safe_float(getattr(eval_result, "reliability", 0.0)))

    axes: dict[str, dict[str, Any]] = {}
    if total <= 0:
        axes["execution_fidelity"] = _axis(
            score=None,
            confidence=0.0,
            evidence_keys=[],
            missing=True,
            reason="no evaluated questions",
        )
    else:
        clean_rate = 1.0 - _clamp((errors + partial + degraded) / total)
        axes["execution_fidelity"] = _axis(
            score=0.45 * quality_norm + 0.35 * reliability + 0.20 * clean_rate,
            confidence=0.75 if total >= 10 else 0.55,
            evidence_keys=[
                "EvalResult.quality",
                "EvalResult.reliability",
                "EvalResult.details.errors",
                "EvalResult.partial_count",
                "EvalResult.degraded_count",
            ],
        )

    action_text = json.dumps(action or {}, sort_keys=True, default=str)
    if prior_criticism and not prior_criticism.startswith("(first trial"):
        overlap = _keyword_overlap(prior_criticism, action_text)
        axes["feedback_interpretation"] = _axis(
            score=0.25 + 0.75 * overlap,
            confidence=0.35,
            evidence_keys=["prior_self_criticism", "action_json"],
            reason="keyword-overlap proxy; observe-only until validated",
        )
    else:
        axes["feedback_interpretation"] = _axis(
            score=None,
            confidence=0.0,
            evidence_keys=[],
            missing=True,
            reason="no prior criticism available",
        )

    violations = len(getattr(verdict, "violations", []) or [])
    warnings = len(getattr(verdict, "warnings", []) or [])
    branching = _safe_float(getattr(eval_result, "branching_density", 0.0), 0.0)
    axes["planning_stability"] = _axis(
        score=1.0 - min(0.6, violations * 0.2 + warnings * 0.1) - min(0.3, branching),
        confidence=0.6 if verdict is not None else 0.35,
        evidence_keys=[
            "SafetyVerdict.violations",
            "SafetyVerdict.warnings",
            "EvalResult.branching_density",
            "failure_analysis",
        ],
        reason="" if not failure_analysis else failure_analysis[:160],
    )

    memory_score, memory_confidence, memory_reason = _trace_memory_score(recent_traces)
    axes["memory_coherence"] = _axis(
        score=memory_score,
        confidence=memory_confidence,
        evidence_keys=["inference_tap.recent_traces"] if recent_traces else [],
        missing=memory_score is None,
        reason=memory_reason,
    )

    recovered = _safe_int(getattr(eval_result, "n_exogenous_recovered", 0), 0)
    unrecovered = _safe_int(getattr(eval_result, "n_exogenous_unrecovered", 0), 0)
    if recovered + unrecovered:
        axes["recovery_rate"] = _axis(
            score=recovered / (recovered + unrecovered),
            confidence=0.8,
            evidence_keys=[
                "EvalResult.n_exogenous_recovered",
                "EvalResult.n_exogenous_unrecovered",
            ],
        )
    else:
        axes["recovery_rate"] = _axis(
            score=None,
            confidence=0.0,
            evidence_keys=[],
            missing=True,
            reason="no recovery event observed",
        )

    missing_axes = [name for name, payload in axes.items() if payload.get("missing")]
    return {
        "metric_version": HARNESS_METRIC_VERSION,
        "schema_version": METRIC_SCHEMA_VERSION,
        "observe_only": True,
        "axes": axes,
        "summary": {
            "missing_axes": missing_axes,
            "missingness": round(len(missing_axes) / len(axes), 4),
            "question_count": total,
            "error_count": errors,
            "partial_count": partial,
            "degraded_count": degraded,
        },
    }


def infer_oracle_adequacy(eval_result: Any) -> dict[str, Any]:
    """Register HLE-2 oracle-adequacy defaults for every observed suite."""
    suites = sorted((getattr(eval_result, "per_suite_quality", {}) or {}).keys())
    if not suites:
        suites = ["unknown"]
    return {
        "metric_version": ORACLE_METRIC_VERSION,
        "schema_version": METRIC_SCHEMA_VERSION,
        "observe_only": True,
        "suites": {suite: _oracle_profile(suite) for suite in suites},
        "control_attestation": infer_control_attestation(eval_result),
    }


def infer_control_attestation(eval_result: Any) -> dict[str, Any]:
    """Report-only W5 control-pair attestation for eval-axis trust.

    This does not gate SafetyGate, Pareto admission, blacklists, or learning
    exclusion. It only records whether a caller supplied known-good and
    known-bad controls whose scorer outcomes match their expected polarity.
    """
    payload: dict[str, Any] = {
        "metric_version": CONTROL_ATTESTATION_VERSION,
        "schema_version": METRIC_SCHEMA_VERSION,
        "observe_only": True,
        "env_flag": CONTROL_ATTESTATION_ENV,
        "enabled": _env_truthy(CONTROL_ATTESTATION_ENV),
        "status": "disabled",
        "eligible_for_evidence": False,
        "controls_seen": {"known_good": 0, "known_bad": 0},
        "failures": [],
        "suites": [],
    }
    if not payload["enabled"]:
        payload["reason"] = "control-pair attestation disabled"
        return payload

    controls = _control_rows(eval_result)
    if not controls:
        payload["status"] = "no_controls"
        payload["reason"] = "no oracle control-pair rows supplied"
        return payload

    failures: list[dict[str, Any]] = []
    suites: set[str] = set()
    counts = {"known_good": 0, "known_bad": 0}
    for row in controls:
        kind = _control_kind(row)
        if kind not in counts:
            failures.append({"kind": kind or "unknown", "reason": "unsupported control kind"})
            continue
        counts[kind] += 1
        suite = str(row.get("suite") or row.get("axis") or "unknown")
        suites.add(suite)
        expected_accept = kind == "known_good"
        observed_accept = _control_observed_accept(row)
        if observed_accept is None:
            failures.append(
                {
                    "kind": kind,
                    "suite": suite,
                    "reason": "missing observed acceptance boolean",
                }
            )
        elif bool(observed_accept) != expected_accept:
            failures.append(
                {
                    "kind": kind,
                    "suite": suite,
                    "expected_accept": expected_accept,
                    "observed_accept": bool(observed_accept),
                }
            )

    payload["controls_seen"] = counts
    payload["failures"] = failures
    payload["suites"] = sorted(suites)
    if counts["known_good"] == 0 or counts["known_bad"] == 0:
        payload["status"] = "incomplete"
        payload["reason"] = "both known_good and known_bad controls are required"
    elif failures:
        payload["status"] = "failed"
        payload["reason"] = "one or more control rows disagreed with expected polarity"
    else:
        payload["status"] = "passed"
        payload["reason"] = "known-good and known-bad controls matched expected polarity"
    return payload


def _control_rows(eval_result: Any) -> list[dict[str, Any]]:
    details = getattr(eval_result, "details", {}) or {}
    if not isinstance(details, dict):
        return []
    raw = (
        details.get("oracle_control_pairs")
        or details.get("control_attestation")
        or details.get("control_pair_results")
    )
    if isinstance(raw, dict):
        rows: list[dict[str, Any]] = []
        for kind in ("known_good", "known_bad"):
            for item in raw.get(kind) or []:
                if isinstance(item, dict):
                    rows.append({"kind": kind, **item})
        if rows:
            return rows
        raw_rows = raw.get("controls") or raw.get("rows")
        if isinstance(raw_rows, list):
            return [row for row in raw_rows if isinstance(row, dict)]
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    return []


def _control_kind(row: dict[str, Any]) -> str:
    raw = row.get("kind") or row.get("control_kind") or row.get("type")
    return str(raw or "").strip().lower().replace("-", "_")


def _control_observed_accept(row: dict[str, Any]) -> bool | None:
    for key in ("observed_accept", "accepted", "scorer_passed", "passed", "correct"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
    return None


def _oracle_profile(suite: str) -> dict[str, Any]:
    s = suite.lower()
    if any(tok in s for tok in ("humaneval", "mbpp", "code", "coding", "usaco")):
        return {
            "oracle_type": "unit_test",
            "coverage_claim": "functional tests check executable behavior for the supplied cases",
            "known_blind_spots": [
                "hidden edge cases",
                "style/maintainability",
                "overfitting visible tests",
            ],
            "shortcut_risk": "medium",
            "requires_external_answer": False,
            "deterministic": True,
            "reviewed_by": "rule:hle-2-oracle-defaults-v1",
        }
    if any(tok in s for tok in ("math", "gsm", "aime", "mmlu")):
        return {
            "oracle_type": "exact_or_multiple_choice",
            "coverage_claim": "final answer match covers answer correctness only",
            "known_blind_spots": ["reasoning shortcut", "format-equivalent answer rejection"],
            "shortcut_risk": "medium",
            "requires_external_answer": False,
            "deterministic": True,
            "reviewed_by": "rule:hle-2-oracle-defaults-v1",
        }
    if any(tok in s for tok in ("hotpot", "trivia", "qa", "squad")):
        return {
            "oracle_type": "short_answer_f1_or_exact",
            "coverage_claim": "short-answer overlap approximates factual answer correctness",
            "known_blind_spots": [
                "web/search leakage",
                "unsupported answer",
                "semantic paraphrase miss",
            ],
            "shortcut_risk": "high",
            "requires_external_answer": False,
            "deterministic": True,
            "reviewed_by": "rule:hle-2-oracle-defaults-v1",
        }
    if "sentinel" in s or "repl" in s:
        return {
            "oracle_type": "deterministic_sentinel",
            "coverage_claim": "hand-authored sentinel covers the named regression mode",
            "known_blind_spots": ["narrow coverage", "prompt-specific shortcut"],
            "shortcut_risk": "medium",
            "requires_external_answer": False,
            "deterministic": True,
            "reviewed_by": "rule:hle-2-oracle-defaults-v1",
        }
    return {
        "oracle_type": "unknown_deterministic_scorer",
        "coverage_claim": "suite scorer exists, but adequacy has not been manually characterized",
        "known_blind_spots": ["unknown coverage", "unknown shortcut risk"],
        "shortcut_risk": "unknown",
        "requires_external_answer": None,
        "deterministic": None,
        "reviewed_by": "rule:hle-2-oracle-defaults-v1",
    }


def compute_hle_observe_payload(
    eval_result: Any,
    *,
    action: dict[str, Any] | None = None,
    verdict: Any | None = None,
    failure_analysis: str = "",
    prior_criticism: str = "",
    recent_traces: str = "",
) -> dict[str, Any]:
    """Return the full J9 observe-only payload for EvalResult/journal fields."""
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "harness_metrics": compute_harness_metrics(
            eval_result,
            action=action,
            verdict=verdict,
            failure_analysis=failure_analysis,
            prior_criticism=prior_criticism,
            recent_traces=recent_traces,
        ),
        "oracle_adequacy": infer_oracle_adequacy(eval_result),
    }
