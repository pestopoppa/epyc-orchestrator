"""EnvSynth species coordinator + journal events + solvability gate.

Wires the sub-modules (``ETDAgent`` / ``TaskSynthesizer`` / ``VerifierBuilder``
/ ``MCPToolRegistry``) into a single 5th autopilot species alongside
Seeder / NumericSwarm / PromptForge / StructuralLab.

Phase 1 scope (NIB2-44 AW-2, AW-4):

  EnvSynth.discover_and_synthesize  one discovery → tasks pipeline run
  EnvSynth.propose_actions          autopilot-visible mutation emissions
  EnvSynthAction                    journaled controller action record
  SolvabilityGate                   reference-model solvability check
                                    (AW-4 — rejects tasks a reference
                                    model cannot solve)

AW-3 (gap diagnosis rollup), AW-5 (EvalTower T1 integration), and
AW-7 (MCP tool adoption) are scaffolded in Week 2-3 of NIB2-44.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from scripts.autopilot.species.env_synth.etd_agent import (
    ETDAgent,
    EnvironmentDiscovery,
)
from scripts.autopilot.species.env_synth.mcp_tool_registry import (
    MCPToolEntry,
    MCPToolRegistry,
)
from scripts.autopilot.species.env_synth.task_synthesizer import (
    DifficultyBand,
    SynthesizedTask,
    TaskSynthesizer,
)
from scripts.autopilot.species.env_synth.verifier_builder import VerifierBuilder

log = logging.getLogger("autopilot.env_synth.species")

ORCH_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REGISTRY_PATH = ORCH_ROOT / "orchestration" / "autopilot_env_synth_registry.jsonl"
DEFAULT_ARENA_PATH = ORCH_ROOT / "orchestration" / "autopilot_env_synth_arena.jsonl"
DEFAULT_JOURNAL_PATH = ORCH_ROOT / "orchestration" / "autopilot_env_synth_journal.jsonl"


# Reference-model solvability: the gate calls this callback with
# ``(task_prompt, tool_set)`` and expects a ``(solved: bool, confidence:
# float, reason: str)`` tuple. In production this is wired to the
# architect_general worker; tests inject a mock.
ReferenceSolver = Callable[[str, list[str]], Awaitable[tuple[bool, float, str]]]


@dataclass
class EnvSynthAction:
    """Journaled controller event emitted per EnvSynth cycle."""

    timestamp: str
    environment_id: str
    tool_set: list[str]
    synthesized_tasks: list[str]                 # SynthesizedTask prompt hashes / ids
    rejected_task_count: int
    difficulty_band: str
    gap_descriptor: str = ""
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class SolvabilityGate:
    """AW-4: reject synthesized tasks a reference model cannot solve.

    Rejection rate should stay below ``max_rejection_rate`` at steady
    state; higher rates indicate difficulty-band miscalibration and
    feed back into autopilot gap diagnosis (AW-3).
    """

    reference_solver: ReferenceSolver
    min_confidence: float = 0.6
    max_rejection_rate: float = 0.20

    async def evaluate(
        self,
        task: SynthesizedTask,
    ) -> tuple[bool, float, str]:
        try:
            solved, confidence, reason = await self.reference_solver(
                task.prompt, task.tool_set,
            )
        except Exception as e:
            log.warning("solvability check raised: %s", e)
            return False, 0.0, f"solver_error: {e}"
        if not solved:
            return False, confidence, reason or "reference_model_failed"
        if confidence < self.min_confidence:
            return False, confidence, "low_confidence_below_threshold"
        return True, confidence, "ok"


@dataclass
class EnvSynth:
    """5th autopilot species — coordinates discovery + synthesis + gating."""

    etd_agent: ETDAgent
    task_synthesizer: TaskSynthesizer
    registry: MCPToolRegistry
    solvability_gate: Optional[SolvabilityGate] = None
    arena_path: Path = field(default_factory=lambda: DEFAULT_ARENA_PATH)
    journal_path: Path = field(default_factory=lambda: DEFAULT_JOURNAL_PATH)

    def __post_init__(self) -> None:
        self.arena_path.parent.mkdir(parents=True, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────

    async def discover_and_synthesize(
        self,
        theme: str,
        *,
        gap_descriptor: str = "",
        tasks_per_env: int = 3,
        band: DifficultyBand = DifficultyBand.MEDIUM,
    ) -> list[SynthesizedTask]:
        """Full pipeline: ETD → synthesis → solvability gate → arena persist.

        Returns the accepted tasks. Rejected tasks are counted in the
        journaled ``EnvSynthAction`` for gap diagnosis (AW-3).
        """
        discoveries = await self.etd_agent.discover(
            theme=theme,
            gap_descriptor=gap_descriptor,
            max_environments=3,
        )
        accepted: list[SynthesizedTask] = []
        rejected = 0

        for disc in discoveries:
            if not disc.tools:
                continue
            for _ in range(tasks_per_env):
                task = await self.task_synthesizer.synthesize(
                    disc.environment_id, disc.tools, band,
                )
                if task is None:
                    rejected += 1
                    continue
                if self.solvability_gate is not None:
                    ok, conf, reason = await self.solvability_gate.evaluate(task)
                    if not ok:
                        log.debug(
                            "solvability gate rejected env=%s reason=%s",
                            disc.environment_id, reason,
                        )
                        rejected += 1
                        continue
                accepted.append(task)
                self._append_arena(task)

            self._journal(EnvSynthAction(
                timestamp=datetime.now(timezone.utc).isoformat(),
                environment_id=disc.environment_id,
                tool_set=[t.tool_id for t in disc.tools],
                synthesized_tasks=[_task_id(t) for t in accepted if t.environment_id == disc.environment_id],
                rejected_task_count=rejected,
                difficulty_band=band.value,
                gap_descriptor=gap_descriptor,
            ))

        return accepted

    def propose_actions(
        self,
        theme: str,
        band: DifficultyBand = DifficultyBand.MEDIUM,
        gap_descriptor: str = "",
    ) -> dict[str, Any]:
        """Return an autopilot-style action dict describing what this
        species would do on the next cycle. Matches the shape of
        ``{"type": "structural_experiment", ...}`` etc. so the
        controller dispatch layer can treat species uniformly.
        """
        return {
            "type": "env_synth_cycle",
            "theme": theme,
            "difficulty_band": band.value,
            "gap_descriptor": gap_descriptor,
        }

    # ── persistence helpers ────────────────────────────────────────

    def _append_arena(self, task: SynthesizedTask) -> None:
        record = {
            "task_id": _task_id(task),
            "environment_id": task.environment_id,
            "tool_set": task.tool_set,
            "prompt": task.prompt,
            "difficulty_band": task.difficulty_band.value,
            "verifier": asdict(task.verifier),
            "ground_truth_hint": task.ground_truth_hint,
            "expected_tool_calls": list(task.expected_tool_calls),
            "metadata": task.metadata,
            "persisted_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.arena_path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _journal(self, action: EnvSynthAction) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a") as f:
            f.write(action.to_json() + "\n")


# ── helpers ─────────────────────────────────────────────────────────


def _task_id(task: SynthesizedTask) -> str:
    """Stable identifier for a synthesized task — env + prompt hash."""
    import hashlib
    h = hashlib.sha256(
        f"{task.environment_id}::{task.prompt}".encode()
    ).hexdigest()[:16]
    return f"envsynth_{h}"
