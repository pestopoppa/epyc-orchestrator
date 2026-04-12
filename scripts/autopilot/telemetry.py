"""Agent Lightning-style telemetry for AutoPilot (MH-5).

Decomposes orchestrator sessions into per-step (input, output, reward)
transitions. OTLP-compatible JSON span format (no collector required).
Exported as JSONL alongside existing journal files.

Source: intake-338 (Agent Lightning, Microsoft Research), intake-344 (LightningRL).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TELEMETRY_DIR = Path(__file__).resolve().parents[2] / "orchestration"


@dataclass
class TransitionRecord:
    """A single (input, output, reward) transition from an orchestrator session.

    Maps to an OTLP-compatible span with attributes for
    per-step credit assignment (LightningRL pattern).
    """
    # Identity
    trace_id: str = ""  # Session-level trace ID
    span_id: str = ""  # Per-transition span ID
    parent_span_id: str = ""  # Parent (trial-level) span

    # Timing
    timestamp: str = ""  # ISO 8601
    duration_ms: float = 0.0

    # Content
    input_text: str = ""  # What went into this step
    output_text: str = ""  # What came out
    reward: float = 0.0  # Per-step reward signal (0-3 quality, or binary)

    # Context
    trial_id: int = 0
    species: str = ""
    role: str = ""  # Which orchestrator role handled this step
    step_type: str = ""  # "controller_reasoning" | "eval" | "mutation" | "safety_gate"
    action_type: str = ""  # From autopilot action

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_otlp_span(self) -> dict[str, Any]:
        """Format as OTLP-compatible JSON span."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id,
            "name": f"{self.species}/{self.step_type}",
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": int(
                datetime.fromisoformat(self.timestamp).timestamp() * 1e9
            ) if self.timestamp else 0,
            "endTimeUnixNano": int(
                datetime.fromisoformat(self.timestamp).timestamp() * 1e9
                + self.duration_ms * 1e6
            ) if self.timestamp else 0,
            "attributes": [
                {"key": "trial_id", "value": {"intValue": self.trial_id}},
                {"key": "species", "value": {"stringValue": self.species}},
                {"key": "role", "value": {"stringValue": self.role}},
                {"key": "action_type", "value": {"stringValue": self.action_type}},
                {"key": "reward", "value": {"doubleValue": self.reward}},
            ],
            "status": {"code": "STATUS_CODE_OK" if self.reward > 0 else "STATUS_CODE_UNSET"},
        }


class TelemetryCollector:
    """Collects transition records for autopilot trials.

    Writes JSONL to orchestration/autopilot_telemetry.jsonl alongside
    the existing journal files.
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or DEFAULT_TELEMETRY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / "autopilot_telemetry.jsonl"
        self._trace_id = uuid.uuid4().hex[:32]  # Session-level trace ID
        self._records: list[TransitionRecord] = []

    def record_transition(
        self,
        trial_id: int,
        species: str,
        step_type: str,
        input_text: str,
        output_text: str,
        reward: float = 0.0,
        role: str = "",
        action_type: str = "",
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> TransitionRecord:
        """Record a single transition and persist to JSONL."""
        record = TransitionRecord(
            trace_id=self._trace_id,
            span_id=uuid.uuid4().hex[:16],
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            trial_id=trial_id,
            species=species,
            step_type=step_type,
            input_text=input_text[:2000],  # Cap for storage
            output_text=output_text[:2000],
            reward=reward,
            role=role,
            action_type=action_type,
            metadata=metadata or {},
        )
        self._records.append(record)
        self._append_jsonl(record)
        return record

    def record_trial(
        self,
        trial_id: int,
        species: str,
        action: dict[str, Any],
        eval_quality: float,
        eval_speed: float,
        passed: bool,
        controller_prompt: str = "",
        controller_response: str = "",
        failure_analysis: str = "",
    ) -> list[TransitionRecord]:
        """Record a complete trial as multiple transitions.

        Decomposes a trial into: reasoning → action → evaluation → outcome.
        Returns all recorded transitions.
        """
        records = []
        action_type = action.get("type", "")

        # 1. Controller reasoning step
        if controller_prompt:
            records.append(self.record_transition(
                trial_id=trial_id,
                species=species,
                step_type="controller_reasoning",
                input_text=controller_prompt[-2000:],
                output_text=controller_response[:2000],
                action_type=action_type,
            ))

        # 2. Action execution step
        records.append(self.record_transition(
            trial_id=trial_id,
            species=species,
            step_type="action_execution",
            input_text=json.dumps(action)[:2000],
            output_text=f"q={eval_quality:.3f} s={eval_speed:.1f}",
            reward=eval_quality,
            action_type=action_type,
        ))

        # 3. Safety gate evaluation
        records.append(self.record_transition(
            trial_id=trial_id,
            species=species,
            step_type="safety_gate",
            input_text=f"q={eval_quality:.3f} s={eval_speed:.1f}",
            output_text="passed" if passed else f"failed: {failure_analysis[:200]}",
            reward=eval_quality if passed else 0.0,
            action_type=action_type,
        ))

        return records

    def _append_jsonl(self, record: TransitionRecord) -> None:
        """Append a single record to the JSONL file."""
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def record_count(self) -> int:
        return len(self._records)
