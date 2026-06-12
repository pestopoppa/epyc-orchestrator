"""Hybrid eval should default to the decision-grade T1 instrument."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

from eval_tower import EvalTower  # type: ignore[import-not-found]
from safety_gate import EvalResult  # type: ignore[import-not-found]


class FakeTower(EvalTower):
    def __init__(self, t0_quality: float = 0.0) -> None:
        self.calls: list[tuple[str, int | None, int | None]] = []
        self.t0_quality = t0_quality

    def eval_t0(self) -> EvalResult:
        self.calls.append(("t0", None, None))
        return EvalResult(tier=0, quality=self.t0_quality, speed=10.0, cost=0.5, reliability=1.0)

    def eval_t1(self, n: int = 100, seed: int = 42) -> EvalResult:
        self.calls.append(("t1", n, seed))
        return EvalResult(tier=1, quality=1.7, speed=50.0, cost=0.2, reliability=0.98)


def test_hybrid_eval_skips_t0_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_HYBRID_T0_GATE", raising=False)
    tower = FakeTower(t0_quality=0.0)

    result = tower.hybrid_eval(seed=7, t1_n=43)

    assert result.tier == 1
    assert tower.calls == [("t1", 43, 7)]


def test_hybrid_eval_legacy_t0_gate_is_env_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_HYBRID_T0_GATE", "1")
    tower = FakeTower(t0_quality=0.0)

    result = tower.hybrid_eval(seed=7, t1_n=43)

    assert result.tier == 0
    assert tower.calls == [("t0", None, None)]
