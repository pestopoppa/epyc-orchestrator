"""Meta-optimizer budget credit should follow realized information gain."""

from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

experiment_journal = importlib.import_module("experiment_journal")
meta_optimizer = importlib.import_module("meta_optimizer")

ExperimentJournal = experiment_journal.ExperimentJournal
JournalEntry = experiment_journal.JournalEntry
MetaOptimizer = meta_optimizer.MetaOptimizer


def _entry(trial_id: int, species: str, **kw) -> JournalEntry:
    return JournalEntry(
        trial_id=trial_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        species=species,
        action_type="seed_batch",
        tier=kw.pop("tier", 1),
        quality=kw.pop("quality", 1.0),
        speed=10.0,
        cost=0.1,
        reliability=1.0,
        pareto_status=kw.pop("pareto_status", "dominated"),
        surprise_score=kw.pop("surprise_score", None),
        bug_corrupted_by=kw.pop("bug_corrupted_by", ""),
        outcome_status=kw.pop("outcome_status", "ok"),
        **kw,
    )


def test_species_effectiveness_prefers_peaf_information_for_budget_rate(
    tmp_path: Path,
) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(1, "seeder", surprise_score=0.8))
    journal.record(_entry(2, "seeder", surprise_score=0.2))
    journal.record(_entry(3, "prompt_forge", pareto_status="frontier"))

    stats = journal.species_effectiveness()

    assert stats["seeder"]["rate"] == 0.0
    assert stats["seeder"]["information_rate"] == 0.5
    assert stats["seeder"]["budget_rate"] == 0.5
    assert stats["prompt_forge"]["rate"] == 1.0
    assert stats["prompt_forge"]["information_count"] == 0
    assert stats["prompt_forge"]["budget_rate"] == 1.0


def test_species_effectiveness_excludes_untrusted_surprise_from_budget_rate(
    tmp_path: Path,
) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(1, "seeder", surprise_score=0.9))
    journal.record(
        _entry(2, "seeder", surprise_score=1.0, bug_corrupted_by="badc0de")
    )
    journal.record(_entry(3, "seeder", surprise_score=1.0, outcome_status="invalid"))

    stats = journal.species_effectiveness()

    assert stats["seeder"]["information_count"] == 1
    assert stats["seeder"]["information_rate"] == 0.9
    assert stats["seeder"]["budget_rate"] == 0.9


def test_species_effectiveness_adds_clipped_higher_tier_quality_credit(
    tmp_path: Path,
) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(1, "structural_lab", tier=3, quality=2.4))
    journal.record(_entry(2, "numeric_swarm", pareto_status="frontier"))

    stats = journal.species_effectiveness()

    assert stats["structural_lab"]["rate"] == 0.0
    assert stats["structural_lab"]["higher_tier_quality_count"] == 1
    assert stats["structural_lab"]["higher_tier_quality_rate"] == pytest.approx(0.8)
    assert stats["structural_lab"]["budget_rate"] == pytest.approx(0.12)
    assert stats["numeric_swarm"]["budget_rate"] == 1.0


def test_species_effectiveness_excludes_failed_higher_tier_quality_credit(
    tmp_path: Path,
) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(
        _entry(
            1,
            "seeder",
            tier=3,
            quality=2.4,
            failure_analysis="Quality floor violation",
        )
    )

    stats = journal.species_effectiveness()

    assert stats["seeder"]["higher_tier_quality_count"] == 0
    assert stats["seeder"]["higher_tier_quality_rate"] == 0.0
    assert stats["seeder"]["budget_rate"] == 0.0


def test_rebalance_uses_budget_rate_instead_of_frontier_rate() -> None:
    optimizer = MetaOptimizer()

    budget = optimizer.rebalance(
        species_effectiveness={
            "numeric_swarm": {
                "total": 10,
                "pareto": 0,
                "rate": 0.0,
                "budget_rate": 0.5,
            },
            "prompt_forge": {
                "total": 10,
                "pareto": 10,
                "rate": 1.0,
                "budget_rate": 0.0,
            },
        },
        hv_slope=0.01,
        memory_count=600,
        is_converged=False,
    )

    assert budget.numeric_swarm > budget.prompt_forge
