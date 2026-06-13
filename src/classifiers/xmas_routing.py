"""X-MAS-style domain/function routing scaffolding.

This module is intentionally side-effect free: it classifies a prompt into a
coarse (domain, function) cell and loads a per-stack winner table, but it does
not alter production routing by itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.roles import Role

XMAS_DOMAINS: tuple[str, ...] = (
    "math",
    "code",
    "knowledge",
    "long_context",
    "reasoning",
)

XMAS_FUNCTIONS: tuple[str, ...] = (
    "solve",
    "verify",
    "plan",
    "refine",
    "extract",
)

DEFAULT_DOMAIN = "knowledge"
DEFAULT_FUNCTION = "solve"
DEFAULT_FALLBACK_ROLE = str(Role.FRONTDOOR)

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "math": (
        "math",
        "algebra",
        "geometry",
        "calculus",
        "integral",
        "derivative",
        "equation",
        "proof",
        "theorem",
        "aime",
        "gsm8k",
        "olympiad",
    ),
    "code": (
        "code",
        "python",
        "javascript",
        "typescript",
        "rust",
        "c++",
        "function",
        "class",
        "method",
        "bug",
        "debug",
        "refactor",
        "unit test",
        "api",
        "sql",
    ),
    "knowledge": (
        "who",
        "what",
        "when",
        "where",
        "which",
        "define",
        "explain",
        "history",
        "fact",
        "facts",
        "source",
        "citation",
    ),
    "long_context": (
        "document",
        "paper",
        "report",
        "transcript",
        "chapter",
        "long context",
        "context",
        "summarize",
        "summarise",
        "extract from",
        "needle",
    ),
    "reasoning": (
        "reason",
        "reasoning",
        "logic",
        "deduce",
        "infer",
        "analyze",
        "analyse",
        "evaluate",
        "compare",
        "gpqa",
        "multiple choice",
        "mcq",
    ),
}

_FUNCTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "solve": (
        "solve",
        "answer",
        "calculate",
        "compute",
        "find",
        "derive",
        "implement",
        "write",
    ),
    "verify": (
        "verify",
        "check",
        "prove",
        "validate",
        "audit",
        "review",
        "test",
        "confirm",
        "is this correct",
    ),
    "plan": (
        "plan",
        "design",
        "architecture",
        "strategy",
        "roadmap",
        "steps",
        "approach",
        "decompose",
    ),
    "refine": (
        "refine",
        "improve",
        "optimize",
        "optimise",
        "rewrite",
        "refactor",
        "edit",
        "polish",
        "fix",
    ),
    "extract": (
        "extract",
        "summarize",
        "summarise",
        "list",
        "identify",
        "classify",
        "parse",
        "pull out",
        "key points",
    ),
}


@dataclass(frozen=True, order=True)
class XmasCell:
    """A single X-MAS domain/function cell."""

    domain: str
    function: str

    def __post_init__(self) -> None:
        _validate_domain(self.domain)
        _validate_function(self.function)

    @property
    def key(self) -> str:
        """Stable string key for logs and table exports."""
        return f"{self.domain}:{self.function}"


@dataclass(frozen=True)
class XmasClassification:
    """Deterministic classification into an X-MAS cell."""

    cell: XmasCell
    confidence: float
    domain_confidence: float
    function_confidence: float
    matched_terms: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return self.cell.domain

    @property
    def function(self) -> str:
        return self.cell.function

    def is_confident(self, threshold: float = 0.55) -> bool:
        """Return whether both axes are strong enough for a routing override."""
        return self.confidence >= threshold


@dataclass(frozen=True)
class WinnerTable:
    """Per-stack X-MAS winner lookup table."""

    cells: dict[XmasCell, str]
    fallback_role: str = DEFAULT_FALLBACK_ROLE
    source: Path | None = None
    version: str = "xmas-v1"

    def __post_init__(self) -> None:
        _validate_role(self.fallback_role)
        for cell, role in self.cells.items():
            if not isinstance(cell, XmasCell):
                raise TypeError(f"winner table key must be XmasCell, got {type(cell)!r}")
            _validate_role(role)

    def winner_for(
        self,
        domain: str,
        function: str,
        *,
        default: str | None = None,
    ) -> str:
        """Return the winning role for a cell, or the configured fallback."""
        cell = XmasCell(domain=domain, function=function)
        role = self.cells.get(cell, default if default is not None else self.fallback_role)
        _validate_role(role)
        return role

    def missing_cells(self) -> list[XmasCell]:
        """Return required 5x5 cells not present in this table."""
        return [
            XmasCell(domain=domain, function=function)
            for domain in XMAS_DOMAINS
            for function in XMAS_FUNCTIONS
            if XmasCell(domain=domain, function=function) not in self.cells
        ]

    def require_complete(self) -> None:
        """Raise if any 5x5 cell is missing."""
        missing = self.missing_cells()
        if missing:
            keys = ", ".join(cell.key for cell in missing)
            raise ValueError(f"winner table is missing {len(missing)} cells: {keys}")

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        source: Path | None = None,
        require_complete: bool = False,
    ) -> "WinnerTable":
        """Build a winner table from a parsed JSON/YAML payload."""
        if not isinstance(payload, dict):
            raise TypeError("winner table payload must be a mapping")

        version = str(payload.get("version") or "xmas-v1")
        fallback_role = str(payload.get("fallback_role") or DEFAULT_FALLBACK_ROLE)
        raw_cells = payload.get("cells")
        if not isinstance(raw_cells, dict):
            raise ValueError("winner table must contain a mapping field named 'cells'")

        cells: dict[XmasCell, str] = {}
        for raw_domain, function_map in raw_cells.items():
            domain = str(raw_domain)
            _validate_domain(domain)
            if not isinstance(function_map, dict):
                raise ValueError(f"cells.{domain} must be a mapping")
            for raw_function, raw_role in function_map.items():
                function = str(raw_function)
                role = str(raw_role)
                _validate_function(function)
                _validate_role(role)
                cells[XmasCell(domain=domain, function=function)] = role

        table = cls(
            cells=cells,
            fallback_role=fallback_role,
            source=source,
            version=version,
        )
        if require_complete:
            table.require_complete()
        return table


def classify_xmas_cell(prompt: str, context: str = "") -> XmasClassification:
    """Classify text into a coarse X-MAS domain/function cell.

    The classifier is lexical by design so it can run during inference-free
    windows. It is a scaffold for the future embedding/bench-populated router,
    not a final capability claim.
    """
    text = _normalize_text(f"{prompt}\n{context}")
    domain, domain_terms, domain_score = _best_axis(
        text,
        _DOMAIN_KEYWORDS,
        default=DEFAULT_DOMAIN,
    )
    function, function_terms, function_score = _best_axis(
        text,
        _FUNCTION_KEYWORDS,
        default=DEFAULT_FUNCTION,
    )

    context_bonus = 1 if len(context) >= 8000 and domain == "long_context" else 0
    domain_score += context_bonus
    domain_confidence = _score_to_confidence(domain_score)
    function_confidence = _score_to_confidence(function_score)
    confidence = round((domain_confidence + function_confidence) / 2.0, 3)

    return XmasClassification(
        cell=XmasCell(domain=domain, function=function),
        confidence=confidence,
        domain_confidence=domain_confidence,
        function_confidence=function_confidence,
        matched_terms={
            "domain": tuple(domain_terms),
            "function": tuple(function_terms),
        },
    )


def load_winner_table(path: str | Path, *, require_complete: bool = False) -> WinnerTable:
    """Load an X-MAS winner table from JSON or YAML."""
    table_path = Path(path)
    try:
        payload = _load_mapping(table_path)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"failed to load X-MAS winner table {table_path}: {exc}") from exc
    return WinnerTable.from_mapping(
        payload,
        source=table_path,
        require_complete=require_complete,
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
    elif suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text())
    else:
        raise ValueError("winner table path must end in .json, .yaml, or .yml")
    if not isinstance(data, dict):
        raise ValueError("winner table file must contain a mapping")
    return data


def _best_axis(
    text: str,
    keywords: dict[str, tuple[str, ...]],
    *,
    default: str,
) -> tuple[str, list[str], int]:
    best_name = default
    best_terms: list[str] = []
    best_score = 0

    for name, terms in keywords.items():
        matched = [term for term in terms if _contains_term(text, term)]
        score = len(matched)
        if score > best_score:
            best_name = name
            best_terms = matched
            best_score = score

    return best_name, best_terms, best_score


def _contains_term(text: str, term: str) -> bool:
    normalized = _normalize_text(term)
    if not normalized:
        return False
    if re.search(r"[^a-z0-9_ ]", normalized) or " " in normalized:
        return normalized in text
    return re.search(rf"\b{re.escape(normalized)}\b", text) is not None


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _score_to_confidence(score: int) -> float:
    if score <= 0:
        return 0.0
    return round(min(0.95, 0.45 + 0.12 * min(score, 4)), 3)


def _validate_domain(domain: str) -> None:
    if domain not in XMAS_DOMAINS:
        raise ValueError(f"unknown X-MAS domain: {domain!r}")


def _validate_function(function: str) -> None:
    if function not in XMAS_FUNCTIONS:
        raise ValueError(f"unknown X-MAS function: {function!r}")


def _validate_role(role: str) -> None:
    if not Role.is_valid(role):
        raise ValueError(f"unknown orchestrator role in X-MAS winner table: {role!r}")
