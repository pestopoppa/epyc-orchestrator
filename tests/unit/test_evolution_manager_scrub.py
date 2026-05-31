"""EvolutionManager must not distill legacy corrupt-baseline text."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

from experiment_journal import JournalEntry  # noqa: E402
from species.evolution_manager import EvolutionManager  # noqa: E402


class CapturingEvolutionManager(EvolutionManager):
    def __init__(self) -> None:
        super().__init__()
        self.prompt = ""

    def _invoke_llm(self, prompt: str) -> str:
        self.prompt = prompt
        return """```json:insights
[
  {
    "description": "Still says baseline 9.900",
    "insight": "Suite 'coder' regression: -6.900",
    "species": "seeder",
    "confidence": "high"
  }
]
```"""


class FakeStrategyStore:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def store(self, **kwargs):
        self.rows.append(kwargs)
        return "stored"


def test_distill_scrubs_legacy_scale_input_and_output() -> None:
    manager = CapturingEvolutionManager()
    store = FakeStrategyStore()
    entry = JournalEntry(
        trial_id=28,
        timestamp="2026-05-31T00:00:00Z",
        species="seeder",
        action_type="seed_batch",
        tier=1,
        quality=0.0,
        speed=1.0,
        cost=0.5,
        reliability=0.0,
        pareto_status="dominated",
        failure_analysis=(
            "Quality regression: 0.000 vs baseline 9.900 (-100.0%); "
            "Suite 'coder' regression: -9.900"
        ),
    )

    result = manager.distill([entry], store, last_n=1, trial_id=183)

    assert result["status"] == "success"
    assert "baseline 9.900" not in manager.prompt
    assert "-9.900" not in manager.prompt
    assert "legacy-scale failure_analysis omitted" in manager.prompt
    assert len(store.rows) == 1
    stored = store.rows[0]
    assert "9.900" not in stored["description"]
    assert "-6.900" not in stored["insight"]
    assert "legacy-scale" in stored["description"]
    assert "legacy-scale" in stored["insight"]
