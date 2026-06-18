"""Short-term memory for AutoPilot controller (AP-22).

Maintains the journal-derived generated view as a markdown file.
Read by the controller before generating each proposal.
Written from the folded append-only journal view.

Source: MiniMax M2.7 3-component self-evolution harness (intake-328/329).
"""

from __future__ import annotations

from os import replace
from pathlib import Path
from typing import Any

MEMORY_PATH = Path(__file__).resolve().parent / "short_term_memory.md"


class ShortTermMemory:
    """Persistent generated-view store for the autopilot controller.

    Sections:
    - Running Hypotheses: active beliefs about what works, revised after each trial
    - Optimization Directions: forward-looking guidance for next trials
    - Failure Patterns: recurring failure signatures to avoid
    - Working Context: key running statistics
    """

    def __init__(self, path: Path | None = None):
        self.path = path or MEMORY_PATH
        self._generated_text: str | None = None

    def refresh_from_journal(
        self,
        journal: Any,
        *,
        last_n: int = 30,
        budget_tokens: int = 2000,
    ) -> str:
        """Rebuild memory from the folded append-only journal view."""
        from stm_generated_view import render_generated_stm

        entries = (
            journal.entries_with_supersessions()
            if hasattr(journal, "entries_with_supersessions")
            else journal.all_entries()
        )
        text = render_generated_stm(
            entries,
            last_n=last_n,
            budget_tokens=budget_tokens,
        )
        self._generated_text = text
        self._write_generated_text(text)
        return text

    def _write_generated_text(self, text: str) -> None:
        """Atomically persist the ledger-derived memory projection."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.tmp")
        tmp_path.write_text(text)
        replace(tmp_path, self.path)

    def to_text(self) -> str:
        """Return memory content for controller prompt injection."""
        text: str
        if self._generated_text is not None:
            text = self._generated_text
        elif self.path.exists():
            text = self.path.read_text()
        else:
            return "(no memory yet — first trial)"
        # Strip markdown header and HTML comments for prompt injection
        lines = [
            ln for ln in text.splitlines()
            if not ln.startswith("<!--") and not ln.startswith("# AutoPilot Short")
        ]
        return "\n".join(lines).strip() or "(empty memory)"

    def clear(self) -> None:
        """Reset memory (e.g., on session restart or CLI command)."""
        self._generated_text = None
        if self.path.exists():
            self.path.unlink()
