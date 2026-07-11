"""SEAGym-style evaluation views for EvalTower result payloads.

This module is observe-only: it classifies already-recorded EvalTower question
results into train/validation/test/replay/OOD views without changing scoring,
SafetyGate promotion, Pareto admission, or baseline calibration.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

SEAGYM_VIEW_SCHEMA_VERSION = "seagym_eval_views.v1"
SEAGYM_VIEWS: tuple[str, ...] = ("train", "validation", "test", "replay", "ood")

_EXPLICIT_VIEW_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "dev": "validation",
    "test": "test",
    "heldout": "test",
    "held_out": "test",
    "promotion": "test",
    "replay": "replay",
    "sentinel": "replay",
    "tool_sentinel": "replay",
    "ood": "ood",
    "out_of_distribution": "ood",
    "audit": "ood",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _details_from_payload(payload: Any) -> Mapping[str, Any]:
    if hasattr(payload, "details"):
        return _mapping(getattr(payload, "details"))
    if isinstance(payload, Mapping):
        details = payload.get("details")
        if isinstance(details, Mapping):
            return details
        eval_details = payload.get("eval_details")
        if isinstance(eval_details, Mapping):
            nested = eval_details.get("details")
            return _mapping(nested) if isinstance(nested, Mapping) else eval_details
    return {}


def _tier_from_payload(payload: Any) -> int | None:
    raw = getattr(payload, "tier", None)
    if raw is None and isinstance(payload, Mapping):
        raw = payload.get("tier")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _question_results_from_payload(payload: Any) -> list[Mapping[str, Any]]:
    raw = getattr(payload, "question_results", None)
    if raw is None and isinstance(payload, Mapping):
        raw = payload.get("question_results")
        if raw is None:
            eval_details = payload.get("eval_details")
            if isinstance(eval_details, Mapping):
                raw = eval_details.get("question_results")
                nested = eval_details.get("details")
                if raw is None and isinstance(nested, Mapping):
                    raw = nested.get("question_results")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _explicit_view(value: Any) -> str | None:
    key = str(value or "").strip().lower().replace("-", "_")
    return _EXPLICIT_VIEW_ALIASES.get(key)


def seagym_view_for_question(
    question: Mapping[str, Any],
    *,
    tier: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> str:
    """Classify one compact EvalTower question result into a SEAGym view."""
    details = details or {}
    for key in ("seagym_view", "eval_view", "view", "partition"):
        view = _explicit_view(question.get(key))
        if view:
            if (
                key == "partition"
                and view in {"validation", "test"}
                and question.get(key) == "core"
            ):
                break
            return view

    partition = str(question.get("partition") or "").strip().lower()
    suite = str(question.get("suite") or "").strip().lower()
    if partition == "core":
        promotion_policy = _mapping(details.get("promotion_eval_policy"))
        if promotion_policy.get("enabled"):
            return "test"
        if tier == 0:
            return "replay"
        if tier == 1:
            return "validation"
        if tier == 3:
            return "ood"
        if tier is not None and tier >= 2:
            return "test"
        return "validation"
    if partition in {"audit", "w6_audit"}:
        return "ood"
    if partition in {"tool_sentinel", "sentinel"} or suite.startswith("sentinel_"):
        return "replay"
    if "ood" in suite or "out_of_distribution" in suite:
        return "ood"
    if suite.endswith("_replay") or "replay" in suite:
        return "replay"
    if tier == 3:
        return "ood"
    if tier is not None and tier >= 2:
        return "test"
    return "validation"


def build_seagym_view_summary(payload: Any) -> dict[str, Any]:
    """Build observe-only train/validation/test/replay/OOD accounting.

    Accepts an EvalResult instance, a serialized EvalResult-shaped mapping, or a
    journal row with nested eval details.
    """
    details = _details_from_payload(payload)
    tier = _tier_from_payload(payload)
    questions = _question_results_from_payload(payload)
    by_view: dict[str, dict[str, Any]] = {
        view: {
            "n": 0,
            "correct": 0,
            "quality": 0.0,
            "suite_counts": {},
            "qid_sample": [],
        }
        for view in SEAGYM_VIEWS
    }

    suite_counts: dict[str, Counter[str]] = {view: Counter() for view in SEAGYM_VIEWS}
    for question in questions:
        view = seagym_view_for_question(question, tier=tier, details=details)
        if view not in by_view:
            view = "validation"
        bucket = by_view[view]
        bucket["n"] += 1
        if bool(question.get("correct")):
            bucket["correct"] += 1
        suite = str(question.get("suite") or "unknown")
        suite_counts[view][suite] += 1
        qid = str(question.get("qid") or question.get("question_id") or "").strip()
        if qid and len(bucket["qid_sample"]) < 10:
            bucket["qid_sample"].append(qid)

    for view, bucket in by_view.items():
        n = int(bucket["n"])
        bucket["quality"] = (float(bucket["correct"]) / n) * 3.0 if n else 0.0
        bucket["suite_counts"] = dict(sorted(suite_counts[view].items()))

    return {
        "schema_version": SEAGYM_VIEW_SCHEMA_VERSION,
        "views": by_view,
        "view_counts": {view: by_view[view]["n"] for view in SEAGYM_VIEWS},
        "total_questions": len(questions),
        "tier": tier,
        "observe_only": True,
        "scoring_effect": "none",
    }


def render_seagym_view_summary(summary: Mapping[str, Any]) -> str:
    views = _mapping(summary.get("views"))
    lines = [
        "# EvalTower SEAGym View Summary",
        "",
        f"- schema: `{summary.get('schema_version', SEAGYM_VIEW_SCHEMA_VERSION)}`",
        f"- observe_only: `{bool(summary.get('observe_only', True))}`",
        f"- total_questions: `{int(summary.get('total_questions') or 0)}`",
        "",
        "| view | n | correct | quality | suites |",
        "|---|---:|---:|---:|---|",
    ]
    for view in SEAGYM_VIEWS:
        bucket = _mapping(views.get(view))
        suites = _mapping(bucket.get("suite_counts"))
        suite_text = ", ".join(f"{k}={v}" for k, v in suites.items()) or "-"
        lines.append(
            f"| {view} | {int(bucket.get('n') or 0)} | "
            f"{int(bucket.get('correct') or 0)} | "
            f"{float(bucket.get('quality') or 0.0):.3f} | {suite_text} |"
        )
    return "\n".join(lines)
