"""Targeted gate for review-before-commit edit consults.

The first broad J17 consult A/B was latency-negative. A targeted slice showed
quality lift on parser/data-contract edges, so production wiring should consult
only when the edit shape has hidden verifier or contract risk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ReviewConsultGateDecision:
    enabled: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsultIntent:
    """Structured intent for consult-policy decisions."""

    skill: str
    task_prompt: str = ""
    current_paths: tuple[str, ...] = ()
    draft_paths: tuple[str, ...] = ()
    delete_paths: tuple[str, ...] = ()
    raw_model_output: str = ""


@dataclass(frozen=True)
class ConsultSignals:
    """Optional routing/MemRL signals for consult-policy decisions."""

    tier: int | None = None
    difficulty_band: str = ""
    factual_risk_band: str = ""
    factual_risk_score: float = 0.0
    touched_symbol_blast_radius: str = ""
    recent_failure_count: int = 0
    benchmark_class: str = ""
    latency_budget_remaining_s: float | None = None
    memrl_hints: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


_DATA_CONTRACT_RE = re.compile(
    r"\b("
    r"parser|parse|parsing|comments?|quotes?|escaping|schema|contract|json|yaml|toml|csv|"
    r"serialization|deserialize|normalize|validator|migration|config|configuration|"
    r"compatibility|backwards?\s+compatible|shim|fallback|optional\s+dependenc(?:y|ies)|"
    r"missing\s+(?:module|package|dependency)|importerror|plugin|registry"
    r")\b",
    re.IGNORECASE,
)
_HIDDEN_VERIFIER_RE = re.compile(
    r"\b(strict|preserve|edge\s+case|hidden\s+test|verifier|round-?trip|idempotent|"
    r"stable\s+order|casefold|case-insensitive|path\s+traversal|unsafe\s+path|"
    r"rollback|transaction|atomic|race|concurrent|async)\b",
    re.IGNORECASE,
)
_PUBLIC_SURFACE_PARTS = {
    "api",
    "routes",
    "models",
    "features.py",
    "config",
    "registry",
    "schemas",
    "migration",
    "plugins",
    "orchestration",
}


def _path_parts(path: str) -> set[str]:
    pure = PurePosixPath(path.replace("\\", "/"))
    return {part.lower() for part in pure.parts if part and part != "."}


def review_before_commit_targeted_gate(
    *,
    task_prompt: str,
    current_paths: list[str] | tuple[str, ...],
    draft_paths: list[str] | tuple[str, ...],
    delete_paths: list[str] | tuple[str, ...],
    raw_model_output: str,
) -> ReviewConsultGateDecision:
    """Return whether an edit draft should receive an architect consult.

    This intentionally uses transparent lexical/shape signals rather than model
    judgment. It is a conservative gate around a costly consult, not an
    authority decision about whether an edit is safe.
    """
    reasons: list[str] = []
    text = "\n".join(
        [
            task_prompt,
            " ".join(current_paths),
            " ".join(draft_paths),
            " ".join(delete_paths),
            raw_model_output[:4000],
        ]
    )
    touched = sorted(set(draft_paths) | set(delete_paths))
    if not touched:
        return ReviewConsultGateDecision(False, ("no_parsed_file_blocks",))

    if _DATA_CONTRACT_RE.search(text):
        reasons.append("parser_data_contract_or_compatibility")
    if _HIDDEN_VERIFIER_RE.search(text):
        reasons.append("hidden_verifier_or_transaction_risk")
    if delete_paths:
        reasons.append("delete_or_removal")
    if len(touched) >= 3:
        reasons.append("multi_file_edit_surface")

    path_parts: set[str] = set()
    for path in [*current_paths, *touched]:
        path_parts.update(_path_parts(path))
    if path_parts & _PUBLIC_SURFACE_PARTS:
        reasons.append("public_api_registry_or_config_surface")

    return ReviewConsultGateDecision(bool(reasons), tuple(dict.fromkeys(reasons)))


def _band_is_high(value: str) -> bool:
    return str(value or "").strip().lower() in {"high", "critical", "hard"}


def _tier_is_hard(tier: int | None) -> bool:
    try:
        return int(tier) >= 2
    except (TypeError, ValueError):
        return False


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _float_or_zero(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int_or_zero(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def should_consult(
    interaction_intent: ConsultIntent,
    signals: ConsultSignals | None = None,
) -> ReviewConsultGateDecision:
    """Return whether an interaction should call its configured consultant.

    The policy is additive over the original edit-shape gate. With no routing
    or MemRL signals, `review_before_commit` behaves like
    `review_before_commit_targeted_gate`. When signals are available, hard-tier
    and high-risk task evidence can trigger consultation even if lexical shape
    alone is not decisive; tight latency budgets can suppress low-risk calls.
    """
    signals = signals or ConsultSignals()
    skill = str(interaction_intent.skill or "").strip()
    if skill != "review_before_commit":
        return ReviewConsultGateDecision(False, ("unsupported_consult_skill",))

    lexical = review_before_commit_targeted_gate(
        task_prompt=interaction_intent.task_prompt,
        current_paths=list(interaction_intent.current_paths),
        draft_paths=list(interaction_intent.draft_paths),
        delete_paths=list(interaction_intent.delete_paths),
        raw_model_output=interaction_intent.raw_model_output,
    )
    reasons = list(lexical.reasons)
    if lexical.reasons == ("no_parsed_file_blocks",):
        return lexical

    hard_tier = _tier_is_hard(signals.tier)
    high_risk = (
        _band_is_high(signals.factual_risk_band)
        or _float_or_zero(signals.factual_risk_score) >= 0.75
        or str(signals.touched_symbol_blast_radius or "").strip().upper() in {"HIGH", "CRITICAL"}
    )
    hard_task = _band_is_high(signals.difficulty_band)
    failure_pressure = _int_or_zero(signals.recent_failure_count) >= 2
    benchmark = str(signals.benchmark_class or "").strip().lower()
    hard_benchmark = benchmark in {"t2", "t3", "hard", "workflow", "real_suite_v1"}
    hint_text = " ".join(signals.memrl_hints).lower()
    memrl_support = any(
        token in hint_text
        for token in (
            "consult",
            "review_before_commit",
            "parser_data_contract",
            "hidden_verifier",
        )
    )

    if hard_tier:
        reasons.append(f"tier_{int(signals.tier)}_hard_workflow")
    if high_risk:
        reasons.append("routing_or_blast_radius_risk")
    if hard_task:
        reasons.append("difficulty_hard")
    if failure_pressure:
        reasons.append("recent_failure_pressure")
    if hard_benchmark:
        reasons.append("hard_benchmark_class")
    if memrl_support:
        reasons.append("memrl_consult_hint")

    try:
        latency_budget = (
            None
            if signals.latency_budget_remaining_s is None
            else float(signals.latency_budget_remaining_s)
        )
    except (TypeError, ValueError):
        latency_budget = None
    if latency_budget is not None and latency_budget < 5.0 and not (high_risk or lexical.enabled):
        return ReviewConsultGateDecision(False, ("latency_budget_too_low",))

    if lexical.enabled:
        return ReviewConsultGateDecision(True, tuple(dict.fromkeys(reasons)))
    if (hard_tier and (high_risk or hard_task or hard_benchmark or memrl_support)) or failure_pressure:
        return ReviewConsultGateDecision(True, tuple(dict.fromkeys(reasons)))
    return ReviewConsultGateDecision(False, ())


def review_before_commit_gate_from_context(context: dict[str, object]) -> ReviewConsultGateDecision:
    """Adapter from run_edit_transaction gate context to `should_consult`."""
    metadata = context.get("signals")
    if not isinstance(metadata, dict):
        metadata = {}
    signals = ConsultSignals(
        tier=metadata.get("tier"),  # type: ignore[arg-type]
        difficulty_band=str(metadata.get("difficulty_band") or ""),
        factual_risk_band=str(metadata.get("factual_risk_band") or ""),
        factual_risk_score=_float_or_zero(metadata.get("factual_risk_score")),
        touched_symbol_blast_radius=str(metadata.get("touched_symbol_blast_radius") or ""),
        recent_failure_count=_int_or_zero(metadata.get("recent_failure_count")),
        benchmark_class=str(metadata.get("benchmark_class") or ""),
        latency_budget_remaining_s=metadata.get("latency_budget_remaining_s"),  # type: ignore[arg-type]
        memrl_hints=_string_tuple(metadata.get("memrl_hints")),
        metadata=dict(metadata),
    )
    return should_consult(
        ConsultIntent(
            skill="review_before_commit",
            task_prompt=str(context.get("task_prompt") or ""),
            current_paths=tuple(str(p) for p in context.get("current_paths") or ()),
            draft_paths=tuple(str(p) for p in context.get("draft_paths") or ()),
            delete_paths=tuple(str(p) for p in context.get("delete_paths") or ()),
            raw_model_output=str(context.get("raw_model_output") or ""),
        ),
        signals,
    )
