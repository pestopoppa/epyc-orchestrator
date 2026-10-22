"""AgentDebugX-compatible trajectory schema (field-level mirror, no package dependency).

Pinned to ``agentdebugx`` 0.5.2 (PyPI wheel ``agentdebugx-0.5.2-py3-none-any.whl``, sha256
``4fdb90ae19167deb49cbaacce3fbc925220041408aee31c48e59c03bbf7ec82b``, uploaded 2026-09-04, MIT),
module ``agentdebug/schema/models.py``, cross-checked against arXiv:2607.18754v1 Appendix A
("Trace schema") and Appendix D. Only the two record types an importer needs are mirrored:
``AgentEvent`` and ``AgentTrajectory`` (field names, defaults and the ``EventType`` string values are
the package's). Nothing is imported from the package and no code is copied: the package needs
pydantic and httpx, and its only subprocess calls are ``git`` (provenance, hub) and its own Python UI
server; it ships no Node helper. ``to_dict`` output is meant to load with the package's
``trajectory_from_json``; ``ADX_EVENT_FIELDS`` / ``ADX_TRAJECTORY_FIELDS`` pin the key sets a test
checks against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

ADX_PACKAGE = "agentdebugx"
ADX_VERSION = "0.5.2"
ADX_WHEEL_SHA256 = "4fdb90ae19167deb49cbaacce3fbc925220041408aee31c48e59c03bbf7ec82b"
ADX_PAPER = "arXiv:2607.18754v1 (HTML sha256 68707a7fe94f76613ecd56de99ef17010f46f7bf3165b62be04ee55d4655c859)"
SCHEMA_PIN = {
    "package": ADX_PACKAGE,
    "version": ADX_VERSION,
    "wheel_sha256": ADX_WHEEL_SHA256,
    "module": "agentdebug/schema/models.py",
    "paper": ADX_PAPER,
    "mirrored": ["AgentEvent", "AgentTrajectory", "EventType"],
}

# The package's field sets (pydantic models, 0.5.2). Artifact objects are carried as plain dicts.
ADX_EVENT_FIELDS = (
    "event_id",
    "trace_id",
    "parent_event_id",
    "agent_name",
    "event_type",
    "module",
    "step_index",
    "timestamp",
    "input",
    "output",
    "error",
    "duration_ms",
    "metadata",
    "artifacts",
)
ADX_TRAJECTORY_FIELDS = (
    "trace_id",
    "task_id",
    "goal",
    "framework",
    "started_at",
    "ended_at",
    "metadata",
    "events",
)

# Deterministic placeholder for imported records without a clock reading. The package defaults to
# ``now()``; an importer must not invent a time, so this fixed epoch marks "not recorded".
EPOCH = "1970-01-01T00:00:00+00:00"


class EventType(str, Enum):
    RUN_START = "run.start"
    RUN_END = "run.end"
    AGENT_STEP = "agent.step"
    LLM_CALL = "llm.call"
    LLM_RESPONSE = "llm.response"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    REFLECTION = "reflection"
    PLAN = "plan"
    HANDOFF = "handoff"
    GUARDRAIL = "guardrail"
    OBSERVATION = "observation"
    ERROR = "error"
    HUMAN_FEEDBACK = "human.feedback"


@dataclass
class AgentEvent:
    event_id: str
    trace_id: str
    event_type: EventType = EventType.AGENT_STEP
    agent_name: str = "agent"
    parent_event_id: str | None = None
    module: str | None = None
    step_index: int | None = None  # None = not attributable (run start/end, task text)
    timestamp: str = EPOCH
    input: Any = None
    output: Any = None
    error: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in ADX_EVENT_FIELDS}
        d["event_type"] = self.event_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentEvent":
        kw = {k: d[k] for k in ADX_EVENT_FIELDS if k in d}
        kw["event_type"] = EventType(kw.get("event_type", EventType.AGENT_STEP.value))
        return cls(**kw)


@dataclass
class AgentTrajectory:
    trace_id: str
    task_id: str | None = None
    goal: str | None = None
    framework: str | None = None
    started_at: str = EPOCH
    ended_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[AgentEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in ADX_TRAJECTORY_FIELDS if k != "events"}
        d["events"] = [e.to_dict() for e in self.events]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentTrajectory":
        kw = {k: d[k] for k in ADX_TRAJECTORY_FIELDS if k in d and k != "events"}
        return cls(events=[AgentEvent.from_dict(e) for e in d.get("events", [])], **kw)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def steps(self) -> list[AgentEvent]:
        """Attributable events (``step_index`` set), ordered by step index."""
        return sorted(
            (e for e in self.events if e.step_index is not None), key=lambda e: e.step_index
        )

    def step_indices(self) -> list[int]:
        return [e.step_index for e in self.steps()]  # type: ignore[misc]

    def by_step(self) -> dict[int, AgentEvent]:
        return {e.step_index: e for e in self.steps()}  # type: ignore[misc]


def write_jsonl(trajs: Iterable[AgentTrajectory], path: Path | str) -> int:
    """Append-only JSONL, one trajectory per line (the package's JSONL store layout)."""
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for t in trajs:
            fh.write(t.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> Iterator[AgentTrajectory]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield AgentTrajectory.from_dict(json.loads(line))
