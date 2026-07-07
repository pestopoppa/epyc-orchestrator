"""Targeted gate for review-before-commit edit consults.

The first broad J17 consult A/B was latency-negative. A targeted slice showed
quality lift on parser/data-contract edges, so production wiring should consult
only when the edit shape has hidden verifier or contract risk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ReviewConsultGateDecision:
    enabled: bool
    reasons: tuple[str, ...] = ()


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
