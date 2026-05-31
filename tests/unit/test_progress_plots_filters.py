from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import progress_plots  # noqa: E402


@dataclass
class _Entry:
    trial_id: int
    tier: int
    quality: float
    speed: float = 10.0
    cost: float = 0.1
    reliability: float = 1.0
    species: str = "seeder"
    bug_corrupted_by: str = ""
    eval_details: dict = field(default_factory=dict)


@dataclass
class _ArchiveEntry:
    trial_id: int
    eval_tier: int
    objectives: tuple[float, float, float, float]
    species: str = "seeder"

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "eval_tier": self.eval_tier,
            "objectives": list(self.objectives),
            "species": self.species,
        }


class _Archive:
    def __init__(self) -> None:
        self._frontier = [_ArchiveEntry(2, 1, (1.5, 20.0, -0.1, 1.0))]
        self._all_entries = [
            _ArchiveEntry(1, 0, (2.4, 30.0, -0.1, 1.0)),
            self._frontier[0],
            _ArchiveEntry(3, 1, (1.0, 10.0, -0.1, 1.0)),
        ]

    def hypervolume_trend(self):
        return [(2, 1.0)]

    def frontier(self):
        return list(self._frontier)

    def is_frontier_eligible(self, entry) -> bool:
        return entry.eval_tier > 0


class _Journal:
    def __init__(self) -> None:
        self._entries = [
            _Entry(1, 0, 2.4, eval_details={"per_suite_quality": {"sentinel": 2.4}}),
            _Entry(2, 1, 1.5, eval_details={"per_suite_quality": {"t1": 1.5}}),
            _Entry(
                3,
                1,
                2.4,
                bug_corrupted_by="ec9622d",
                eval_details={"per_suite_quality": {"stale": 2.4}},
            ),
        ]

    def all_entries(self):
        return list(self._entries)

    def species_effectiveness(self):
        return {}


def test_generate_all_plots_filters_t0_and_bug_corrupted_points(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        progress_plots,
        "plot_hypervolume_trend",
        lambda history, output_dir=None: captured.setdefault("hv", history) or tmp_path / "hv.png",
    )

    def _frontier(frontier, dominated, output_dir=None):
        captured["frontier"] = frontier
        captured["dominated"] = dominated
        return tmp_path / "frontier.png"

    monkeypatch.setattr(progress_plots, "plot_pareto_frontier_2d", _frontier)
    monkeypatch.setattr(
        progress_plots,
        "plot_species_effectiveness",
        lambda eff, output_dir=None: tmp_path / "species.png",
    )

    def _suite(data, output_dir=None):
        captured["suite_data"] = data
        return tmp_path / "suite.png"

    monkeypatch.setattr(progress_plots, "plot_per_suite_quality", _suite)
    monkeypatch.setattr(
        progress_plots,
        "plot_memory_convergence",
        lambda td_errors, output_dir=None: tmp_path / "memory.png",
    )

    def _timeline(data, output_dir=None):
        captured["timeline"] = data
        return tmp_path / "timeline.png"

    monkeypatch.setattr(progress_plots, "plot_trial_timeline", _timeline)

    progress_plots.generate_all_plots(_Archive(), _Journal(), output_dir=tmp_path)

    assert [p["trial_id"] for p in captured["frontier"]] == [2]
    assert [p["trial_id"] for p in captured["dominated"]] == [3]
    assert [p["trial_id"] for p in captured["suite_data"]] == [2]
    assert [p["trial_id"] for p in captured["timeline"]] == [2]
