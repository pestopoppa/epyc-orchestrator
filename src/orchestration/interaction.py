"""Internal interaction lifecycle primitives.

The first revision is an additive substrate for the existing delegation loop.
It intentionally does not dispatch work or change scheduling behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


INTERACTION_POLICY_VERSION = "1.0"

InteractionKind = Literal["delegate", "consult", "verify", "route"]
InteractionState = Literal[
    "created",
    "working",
    "input_required",
    "completed",
    "failed",
    "cancelled",
]


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an artifact produced by an interaction."""

    kind: str
    ref: str
    role: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulerPolicy:
    """Thin wrapper over existing request admission fields."""

    priority: str = "interactive"
    max_queue_wait_ms: int | None = None
    migration_budget_ms: int | None = None
    cancellable: bool = False

    @classmethod
    def from_primitives(
        cls,
        primitives: Any,
        *,
        default_priority: str = "interactive",
        cancellable: bool = False,
    ) -> "SchedulerPolicy":
        get_priority = getattr(primitives, "get_request_priority", None)
        get_max_queue_wait_ms = getattr(primitives, "get_max_queue_wait_ms", None)
        get_migration_budget_ms = getattr(primitives, "get_migration_budget_ms", None)
        priority = get_priority() if callable(get_priority) else default_priority
        max_queue_wait_ms = (
            get_max_queue_wait_ms() if callable(get_max_queue_wait_ms) else None
        )
        migration_budget_ms = (
            get_migration_budget_ms() if callable(get_migration_budget_ms) else None
        )
        return cls(
            priority=str(priority or default_priority),
            max_queue_wait_ms=max_queue_wait_ms,
            migration_budget_ms=migration_budget_ms,
            cancellable=cancellable,
        )

    def request_context_kwargs(self) -> dict[str, Any]:
        """Return kwargs compatible with ``LLMPrimitives.request_context``."""
        kwargs: dict[str, Any] = {"priority": self.priority}
        if self.max_queue_wait_ms is not None:
            kwargs["max_queue_wait_ms"] = self.max_queue_wait_ms
        if self.migration_budget_ms is not None:
            kwargs["migration_budget_ms"] = self.migration_budget_ms
        return kwargs


@dataclass(frozen=True)
class InteractionTelemetry:
    """Telemetry attributes shared by delegation and future consult flows."""

    interaction_type: InteractionKind = "delegate"
    skill: str = ""
    context_hash: str = ""
    policy_version: str = INTERACTION_POLICY_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "interaction_type": self.interaction_type,
            "skill": self.skill,
            "context_hash": self.context_hash,
            "interaction_policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class InteractionEvent:
    """Wire-compatible interaction event.

    The field names intentionally mirror ``DelegationEvent`` so existing
    ``delegation_events`` consumers keep working while ``interaction_type`` is
    added.
    """

    from_role: str
    to_role: str
    task_summary: str = ""
    interaction_type: InteractionKind = "delegate"
    success: bool | None = None
    elapsed_ms: float = 0.0
    tokens_generated: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_delegation_event_payload(self) -> dict[str, Any]:
        payload = {
            "from_role": self.from_role,
            "to_role": self.to_role,
            "task_summary": self.task_summary,
            "interaction_type": self.interaction_type,
            "success": self.success,
            "elapsed_ms": self.elapsed_ms,
            "tokens_generated": self.tokens_generated,
        }
        payload.update(self.metadata)
        return payload


@dataclass
class Interaction:
    """Internal lifecycle object for delegated, consult, verify, or route work."""

    kind: InteractionKind
    owner_role: str
    callee_role: str
    skill: str = ""
    state: InteractionState = "created"
    artifacts: list[ArtifactRef] = field(default_factory=list)
    events: list[InteractionEvent] = field(default_factory=list)
    token_budget: int = 0
    deadline: float | None = None
    scheduler_policy: SchedulerPolicy = field(default_factory=SchedulerPolicy)
    telemetry: InteractionTelemetry = field(default_factory=InteractionTelemetry)

    def start(self) -> None:
        self.state = "working"

    def complete(self, *, artifact: ArtifactRef | None = None) -> None:
        if artifact is not None:
            self.artifacts.append(artifact)
        self.state = "completed"

    def fail(self) -> None:
        self.state = "failed"

    def cancel(self) -> None:
        self.state = "cancelled"

    def add_event(self, event: InteractionEvent) -> InteractionEvent:
        self.events.append(event)
        return event

    def emit_event(
        self,
        *,
        to_role: str | None = None,
        task_summary: str = "",
        success: bool | None = None,
        elapsed_ms: float = 0.0,
        tokens_generated: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> InteractionEvent:
        event = InteractionEvent(
            from_role=self.owner_role,
            to_role=to_role or self.callee_role,
            task_summary=task_summary,
            interaction_type=self.kind,
            success=success,
            elapsed_ms=elapsed_ms,
            tokens_generated=tokens_generated,
            metadata=dict(metadata or {}),
        )
        return self.add_event(event)
