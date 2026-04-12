"""Short-term memory for AutoPilot controller (AP-22).

Accumulates structured learnings across trials as a markdown file.
Read by the controller before generating each proposal.
Written after each trial evaluation.

Source: MiniMax M2.7 3-component self-evolution harness (intake-328/329).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MEMORY_PATH = Path(__file__).resolve().parent / "short_term_memory.md"
MAX_LINES = 120  # ~2000 tokens budget


@dataclass
class TrialOutcome:
    """Minimal trial outcome for memory update."""
    trial_id: int
    species: str
    action_type: str
    quality: float
    speed: float
    passed: bool
    hypothesis: str
    failure_analysis: str
    self_criticism: str
    optimization_directions: str
    keep_revert: str
    per_suite_quality: dict[str, float]


class ShortTermMemory:
    """Persistent per-session memory accumulator for the autopilot controller.

    Sections:
    - Running Hypotheses: active beliefs about what works, revised after each trial
    - Optimization Directions: forward-looking guidance for next trials
    - Failure Patterns: recurring failure signatures to avoid
    - Working Context: key running statistics
    """

    def __init__(self, path: Path | None = None):
        self.path = path or MEMORY_PATH
        self._hypotheses: list[str] = []
        self._directions: list[str] = []
        self._failure_patterns: list[str] = []
        self._context: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load existing memory from disk."""
        if not self.path.exists():
            return
        text = self.path.read_text()
        current_section = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## Running Hypotheses"):
                current_section = "hypotheses"
            elif stripped.startswith("## Optimization Directions"):
                current_section = "directions"
            elif stripped.startswith("## Failure Patterns"):
                current_section = "failures"
            elif stripped.startswith("## Working Context"):
                current_section = "context"
            elif stripped.startswith("## ") or stripped.startswith("# "):
                current_section = None
            elif stripped.startswith("- ") and current_section:
                entry = stripped[2:]
                if current_section == "hypotheses":
                    self._hypotheses.append(entry)
                elif current_section == "directions":
                    self._directions.append(entry)
                elif current_section == "failures":
                    self._failure_patterns.append(entry)
                elif current_section == "context":
                    self._context[entry.split(":")[0].strip()] = entry

    def update(self, outcome: TrialOutcome) -> None:
        """Update memory with a new trial outcome."""
        tag = f"[t{outcome.trial_id}]"

        # Update hypotheses: add outcome of current trial
        if outcome.hypothesis:
            status = "confirmed" if outcome.passed else "rejected"
            entry = f"{tag} {outcome.hypothesis} -- {status} (q={outcome.quality:.2f})"
            self._hypotheses.append(entry)

        # Update optimization directions from self-criticism
        if outcome.optimization_directions:
            for d in outcome.optimization_directions.split(";"):
                d = d.strip()
                if d:
                    self._directions.append(f"{tag} {d}")

        # Track failure patterns
        if not outcome.passed and outcome.failure_analysis:
            pattern = f"{tag} {outcome.species}/{outcome.action_type}: {outcome.failure_analysis[:120]}"
            self._failure_patterns.append(pattern)

        # Update working context
        self._context["Last trial"] = f"Last trial: {outcome.trial_id} ({outcome.species}/{outcome.action_type}, q={outcome.quality:.2f}, {outcome.keep_revert or 'n/a'})"
        self._context["Best quality"] = f"Best quality: {max(outcome.quality, float(self._context.get('Best quality', 'Best quality: 0').split(': ')[-1])):.2f}"

        # Track declining suites
        if outcome.per_suite_quality:
            declining = [
                f"{suite}={q:.2f}"
                for suite, q in outcome.per_suite_quality.items()
                if q < 1.5
            ]
            if declining:
                self._context["Weak suites"] = f"Weak suites: {', '.join(declining)}"

        # Enforce budget: trim oldest entries
        self._trim()
        self._save()

    def _trim(self) -> None:
        """Trim to MAX_LINES budget, keeping most recent entries."""
        max_per_section = MAX_LINES // 4
        self._hypotheses = self._hypotheses[-max_per_section:]
        self._directions = self._directions[-max_per_section:]
        self._failure_patterns = self._failure_patterns[-max_per_section:]

    def _save(self) -> None:
        """Write memory to disk."""
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines = [
            "# AutoPilot Short-Term Memory",
            f"<!-- Auto-generated. Last updated: {ts} -->",
            "",
            "## Running Hypotheses",
        ]
        for h in self._hypotheses:
            lines.append(f"- {h}")

        lines.append("")
        lines.append("## Optimization Directions")
        for d in self._directions:
            lines.append(f"- {d}")

        lines.append("")
        lines.append("## Failure Patterns")
        for f in self._failure_patterns:
            lines.append(f"- {f}")

        lines.append("")
        lines.append("## Working Context")
        for v in self._context.values():
            lines.append(f"- {v}")
        lines.append("")

        self.path.write_text("\n".join(lines))

    def to_text(self) -> str:
        """Return memory content for controller prompt injection."""
        if not self.path.exists():
            return "(no memory yet — first trial)"
        text = self.path.read_text()
        # Strip markdown header and HTML comments for prompt injection
        lines = [
            ln for ln in text.splitlines()
            if not ln.startswith("<!--") and not ln.startswith("# AutoPilot Short")
        ]
        return "\n".join(lines).strip() or "(empty memory)"

    def clear(self) -> None:
        """Reset memory (e.g., on session restart or CLI command)."""
        self._hypotheses.clear()
        self._directions.clear()
        self._failure_patterns.clear()
        self._context.clear()
        if self.path.exists():
            self.path.unlink()
