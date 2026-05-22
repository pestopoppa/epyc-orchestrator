"""Procedure data models — StepResult, ProcedureResult, ProcedureInput,
ProcedureStep, Procedure + exception types.

Extracted from orchestration/procedure_registry.py during the 2026-05-22
Task-J refactor. procedure_registry.py re-exports every name so existing
imports keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """Result of executing a single procedure step."""

    step_id: str
    success: bool
    output: str = ""
    error: str | None = None
    elapsed_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class ProcedureResult:
    """Result of executing a complete procedure."""

    procedure_id: str
    success: bool
    step_results: list[StepResult] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_seconds: float = 0.0
    checkpoint_id: str | None = None
    rolled_back: bool = False


@dataclass
class ProcedureInput:
    """Definition of a procedure input parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    validation: dict[str, Any] | None = None


@dataclass
class ProcedureStep:
    """Definition of a procedure step."""

    id: str
    name: str
    action: dict[str, Any]
    description: str | None = None
    condition: str | None = None
    on_failure: str = "abort"
    max_retries: int = 0
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Procedure:
    """Complete procedure definition."""

    id: str
    name: str
    version: str
    description: str
    category: str
    steps: list[ProcedureStep]
    verification: dict[str, Any]
    inputs: list[ProcedureInput] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    permissions: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 350


class ProcedureValidationError(Exception):
    """Error validating a procedure definition."""

    pass


class ProcedureExecutionError(Exception):
    """Error executing a procedure."""

    pass
