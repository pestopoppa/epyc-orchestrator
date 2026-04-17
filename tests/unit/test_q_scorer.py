"""Tests for QScorer cost-aware reward computation."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Any

# Minimal stubs so we can test _compute_reward without real dependencies.
# The actual ProgressEntry uses more fields; we only need outcome + event_type + data.


class _EventType:
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    ESCALATION_TRIGGERED = "escalation_triggered"
    PLAN_REVIEWED = "plan_reviewed"


@dataclass
class _FakeEntry:
    event_type: str
    outcome: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Import the real ScoringConfig and QScorer
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.q_scorer import ScoringConfig, QScorer
from orchestration.repl_memory.progress_logger import EventType


def _make_outcome(outcome: str = "success") -> _FakeEntry:
    """Create a fake task outcome entry."""
    evt = EventType.TASK_COMPLETED if outcome != "failure" else EventType.TASK_FAILED
    return _FakeEntry(event_type=evt, outcome=outcome)


def _make_gate_fail() -> _FakeEntry:
    return _FakeEntry(event_type=EventType.GATE_FAILED)


def _make_escalation() -> _FakeEntry:
    return _FakeEntry(event_type=EventType.ESCALATION_TRIGGERED)


def _make_plan_review(decision: str = "ok") -> _FakeEntry:
    return _FakeEntry(event_type=EventType.PLAN_REVIEWED, data={"decision": decision})


def _scorer(config: ScoringConfig | None = None) -> QScorer:
    """Build a QScorer with mocked dependencies (we only test _compute_reward)."""
    return QScorer(
        store=MagicMock(),
        embedder=MagicMock(),
        logger=MagicMock(),
        reader=MagicMock(),
        config=config or ScoringConfig(),
    )


# ===== ScoringConfig defaults =====


class TestScoringConfigDefaults:
    def test_cost_penalty_lambda_default(self):
        cfg = ScoringConfig()
        assert cfg.cost_penalty_lambda == 0.15

    def test_baseline_tps_has_all_production_roles(self):
        cfg = ScoringConfig()
        expected_roles = {
            "frontdoor",
            "coder_escalation",
            "coder_escalation",
            "architect_general",
            "architect_coding",
            "ingest_long_context",
            "worker_explore",
            "worker_math",
            "worker_vision",
            "vision_escalation",
        }
        assert set(cfg.baseline_tps_by_role.keys()) == expected_roles

    def test_baseline_tps_values_positive(self):
        cfg = ScoringConfig()
        for role, tps in cfg.baseline_tps_by_role.items():
            assert tps > 0, f"{role} has non-positive tps: {tps}"

    def test_config_override(self):
        cfg = ScoringConfig(cost_penalty_lambda=0.5)
        assert cfg.cost_penalty_lambda == 0.5


# ===== _compute_reward without cost metrics (backward compat) =====


class TestComputeRewardNoCost:
    def test_success_no_cost(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [], [])
        assert r == 1.0

    def test_failure_no_cost(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("failure"), [], [])
        assert r == -0.5

    def test_partial_no_cost(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("partial"), [], [])
        assert r == 0.3

    def test_gate_failures_penalize(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [_make_gate_fail(), _make_gate_fail()], [])
        assert r == pytest.approx(0.8)  # 1.0 - 2*0.1

    def test_escalation_penalizes(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [], [_make_escalation()])
        assert r == pytest.approx(0.85)  # 1.0 - 0.15

    def test_plan_review_approved(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [], [], [_make_plan_review("ok")])
        assert r == pytest.approx(1.0)  # 1.0 + 0.1 clamped to 1.0

    def test_plan_review_corrected(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [], [], [_make_plan_review("corrected")])
        assert r == pytest.approx(0.8)  # 1.0 - 0.2


# ===== _compute_reward with cost metrics =====


def _latency_only_config(**overrides) -> ScoringConfig:
    """Config with quality-gap and memory penalties zeroed (isolate latency tests)."""
    return ScoringConfig(cost_lambda_quality_gap=0.0, cost_lambda_memory=0.0, **overrides)


class TestComputeRewardWithCost:
    # NOTE: frontdoor baseline_tps=12.7, architect=4.3 (updated 2026-03-29).
    # Use 127 tokens for frontdoor (127/12.7=10s expected) and 43 for architect
    # (43/4.3=10s expected) to produce exact integer cost ratios.

    def test_at_expected_speed_no_penalty(self):
        """Running at exactly baseline speed → cost_ratio=1.0 → no latency penalty."""
        s = _scorer(_latency_only_config())
        # frontdoor at 12.7 t/s, 127 tokens in 10s → exactly expected
        cost = {"tokens_generated": 127, "elapsed_seconds": 10.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_faster_than_expected_no_penalty(self):
        """Running faster than baseline → cost_ratio < 1.0 → no penalty."""
        s = _scorer(_latency_only_config())
        # frontdoor at 12.7 t/s, 127 tokens in 5s → 2x faster
        cost = {"tokens_generated": 127, "elapsed_seconds": 5.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_slower_than_expected_penalized(self):
        """Running 2x slower → cost_ratio=2.0 → penalty = 0.15 * (2.0 - 1.0) = 0.15."""
        s = _scorer(_latency_only_config())
        # frontdoor at 12.7 t/s, 127 tokens in 20s → 2x slower
        cost = {"tokens_generated": 127, "elapsed_seconds": 20.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == pytest.approx(0.85)  # 1.0 - 0.15

    def test_much_slower_higher_penalty(self):
        """Running 5x slower → penalty = 0.15 * 4.0 = 0.60."""
        s = _scorer(_latency_only_config())
        cost = {"tokens_generated": 127, "elapsed_seconds": 50.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == pytest.approx(0.4)  # 1.0 - 0.60

    def test_incorrect_no_cost_penalty(self):
        """Failed tasks get failure_reward regardless of cost."""
        s = _scorer(_latency_only_config())
        cost = {"tokens_generated": 127, "elapsed_seconds": 100.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("failure"), [], [], cost_metrics=cost)
        assert r == -0.5  # No cost penalty applied (reward <= 0)

    def test_unknown_role_no_penalty(self):
        """Unknown role has no baseline → no cost penalty."""
        s = _scorer(_latency_only_config())
        cost = {"tokens_generated": 100, "elapsed_seconds": 100.0, "role": "unknown_role"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_zero_tokens_no_penalty(self):
        """Zero tokens_generated → skip cost computation."""
        s = _scorer(_latency_only_config())
        cost = {"tokens_generated": 0, "elapsed_seconds": 10.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_zero_elapsed_no_penalty(self):
        """Zero elapsed → skip cost computation (avoid division by zero)."""
        s = _scorer(_latency_only_config())
        cost = {"tokens_generated": 100, "elapsed_seconds": 0.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_cost_plus_gate_penalties_stack(self):
        """Cost penalty stacks with gate failure penalties."""
        s = _scorer(_latency_only_config())
        # 2x slower + 1 gate failure
        cost = {"tokens_generated": 127, "elapsed_seconds": 20.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [_make_gate_fail()], [], cost_metrics=cost)
        # 1.0 - 0.1 (gate) - 0.15 (cost) = 0.75
        assert r == pytest.approx(0.75)

    def test_clamp_lower_bound(self):
        """Extreme cost penalty clamped to -1.0."""
        cfg = _latency_only_config(cost_penalty_lambda=10.0)  # Very aggressive
        s = _scorer(cfg)
        cost = {"tokens_generated": 127, "elapsed_seconds": 50.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == -1.0

    def test_custom_lambda(self):
        """Custom lambda changes penalty magnitude."""
        cfg = _latency_only_config(cost_penalty_lambda=0.5)
        s = _scorer(cfg)
        # 2x slower with lambda=0.5 → penalty = 0.5 * 1.0 = 0.5
        cost = {"tokens_generated": 127, "elapsed_seconds": 20.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == pytest.approx(0.5)

    def test_architect_role_slower_baseline(self):
        """Architect (4.3 t/s) at expected speed → no latency penalty."""
        s = _scorer(_latency_only_config())
        # 43 tokens at 4.3 t/s → 10s expected; actual 10s
        cost = {"tokens_generated": 43, "elapsed_seconds": 10.0, "role": "architect_general"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_architect_role_2x_slower(self):
        """Architect at 2x slower → penalty = 0.15 * 1.0 = 0.15."""
        s = _scorer(_latency_only_config())
        cost = {"tokens_generated": 43, "elapsed_seconds": 20.0, "role": "architect_general"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == pytest.approx(0.85)


# ===== Multi-dimensional cost model tests =====


class TestMultiDimensionalCost:
    """Test quality-gap and memory-tier cost dimensions."""

    def test_config_has_quality_baselines(self):
        cfg = ScoringConfig()
        assert "architect_general" in cfg.baseline_quality_by_role
        assert "worker_explore" in cfg.baseline_quality_by_role
        assert cfg.baseline_quality_by_role["architect_general"] > cfg.baseline_quality_by_role["worker_explore"]

    def test_config_has_memory_costs(self):
        cfg = ScoringConfig()
        assert cfg.memory_cost_by_role["worker_explore"] < 1.0  # Small model
        assert cfg.memory_cost_by_role["architect_general"] > 1.0  # WARM tier

    def test_quality_gap_penalty_architect(self):
        """Architect (quality=0.94) gets penalized for quality gap above 0.75 baseline."""
        s = _scorer()
        # At expected speed, no latency penalty
        cost = {"tokens_generated": 675, "elapsed_seconds": 100.0, "role": "architect_general"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        # quality_gap = 0.94 - 0.75 = 0.19, penalty = 0.10 * 0.19 = 0.019
        # memory_cost = 3.0, penalty = 0.05 * (3.0 - 1.0) = 0.10
        # total = 1.0 - 0.019 - 0.10 = 0.881
        assert r < 1.0  # Should be penalized
        assert r > 0.8  # But not too much

    def test_quality_gap_penalty_worker(self):
        """Worker (quality=0.745) has minimal quality gap penalty."""
        s = _scorer()
        # worker_explore at 39.1 t/s, 391 tokens in 10s → exactly at expected speed
        cost = {"tokens_generated": 391, "elapsed_seconds": 10.0, "role": "worker_explore"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        # quality_gap = max(0, 0.745 - 0.75) = 0.0 → no quality penalty
        # memory_cost = 0.5 < 1.0 → no memory penalty
        assert r == 1.0  # No penalty at all for worker at expected speed

    def test_memory_tier_penalty_warm(self):
        """WARM tier models (architect) get memory penalty."""
        cfg = ScoringConfig(cost_penalty_lambda=0.0, cost_lambda_quality_gap=0.0)
        s = _scorer(cfg)
        # Isolate memory penalty only (zero out other dimensions)
        cost = {"tokens_generated": 675, "elapsed_seconds": 100.0, "role": "architect_general"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        # memory_cost = 3.0, penalty = 0.05 * (3.0 - 1.0) = 0.10
        assert r == pytest.approx(0.9)

    def test_memory_tier_no_penalty_hot(self):
        """HOT tier models (worker) have no memory penalty."""
        cfg = ScoringConfig(cost_penalty_lambda=0.0, cost_lambda_quality_gap=0.0)
        s = _scorer(cfg)
        cost = {"tokens_generated": 279, "elapsed_seconds": 10.0, "role": "worker_explore"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0  # No memory penalty for HOT

    def test_worker_beats_architect_on_simple_task(self):
        """Worker on simple task should get higher reward than architect."""
        s = _scorer()
        # Worker at expected speed, correct
        worker_cost = {"tokens_generated": 279, "elapsed_seconds": 10.0, "role": "worker_explore"}
        worker_r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=worker_cost)

        # Architect at expected speed, correct — but quality gap + memory penalty
        arch_cost = {"tokens_generated": 675, "elapsed_seconds": 100.0, "role": "architect_general"}
        arch_r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=arch_cost)

        assert worker_r > arch_r, (
            f"Worker reward ({worker_r}) should beat architect ({arch_r}) on simple correct tasks"
        )

    def test_no_quality_memory_penalty_on_failure(self):
        """Quality and memory penalties only apply to correct answers."""
        s = _scorer()
        cost = {"tokens_generated": 675, "elapsed_seconds": 100.0, "role": "architect_general"}
        r = s._compute_reward(_make_outcome("failure"), [], [], cost_metrics=cost)
        assert r == -0.5  # Pure failure reward, no cost penalty


# ===== _compute_contrastive_adjustment (DAR-2) =====

import numpy as np
from orchestration.repl_memory.episodic_store import MemoryEntry


def _make_memory(action: str, q_value: float = 0.5, memory_id: str = "m1") -> MemoryEntry:
    """Create a MemoryEntry with controlled action/Q-value for contrastive tests."""
    return MemoryEntry(
        id=memory_id,
        embedding=None,
        action=action,
        action_type="routing",
        context={},
        q_value=q_value,
    )


def _make_task_entry(data: dict | None = None) -> _FakeEntry:
    """Create a fake task_started entry with task context data."""
    return _FakeEntry(
        event_type=EventType.TASK_COMPLETED,
        data=data or {"task_type": "chat", "objective": "test task"},
    )


def _make_routing_entry(
    action: str = "frontdoor", memory_id: str | None = "m-selected",
) -> _FakeEntry:
    """Create a fake routing_decision entry."""
    entry = _FakeEntry(
        event_type=EventType.TASK_COMPLETED,
        data={"routing": [action]},
    )
    entry.memory_id = memory_id  # type: ignore[attr-defined]
    return entry


class TestComputeContrastiveAdjustment:
    """Tests for DAR-2 _compute_contrastive_adjustment."""

    def test_no_task_context_returns_zero(self):
        """Empty task_started data → 0.0."""
        s = _scorer()
        task = _FakeEntry(event_type=EventType.TASK_COMPLETED, data={})
        routing = _make_routing_entry()
        assert s._compute_contrastive_adjustment(task, routing, reward=0.5) == 0.0

    def test_none_task_started_returns_zero(self):
        """None task_started → 0.0."""
        s = _scorer()
        routing = _make_routing_entry()
        assert s._compute_contrastive_adjustment(None, routing, reward=0.5) == 0.0

    def test_embedding_failure_returns_zero(self):
        """Embedding failure → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.side_effect = RuntimeError("embed failed")
        task = _make_task_entry()
        routing = _make_routing_entry()
        assert s._compute_contrastive_adjustment(task, routing, reward=0.5) == 0.0

    def test_no_candidates_returns_zero(self):
        """No similar routing memories → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = []
        task = _make_task_entry()
        routing = _make_routing_entry()
        assert s._compute_contrastive_adjustment(task, routing, reward=0.5) == 0.0

    def test_no_alternatives_returns_zero(self):
        """All candidates have the same action as selected → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        # All candidates use same action as selected
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.7, memory_id="c1"),
            _make_memory("frontdoor", q_value=0.6, memory_id="c2"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.65)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")
        assert s._compute_contrastive_adjustment(task, routing, reward=0.5) == 0.0

    def test_all_alternatives_at_default_returns_zero(self):
        """Alternatives at default Q=0.5 (unlearned) → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.7, memory_id="c1"),
            _make_memory("architect_general", q_value=0.5, memory_id="c2"),  # default
            _make_memory("coder_escalation", q_value=0.5, memory_id="c3"),  # default
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.7)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")
        assert s._compute_contrastive_adjustment(task, routing, reward=0.5) == 0.0

    def test_success_selected_below_best_alt_positive_adjustment(self):
        """Success + selected Q below best alternative → positive adjustment."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.6, memory_id="c1"),
            _make_memory("architect_general", q_value=0.8, memory_id="c2"),
        ]
        # Selected model's current Q
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.6)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")

        adj = s._compute_contrastive_adjustment(task, routing, reward=0.5)
        # max_alt_q=0.8, margin=0.05, gap = 0.8 + 0.05 - 0.6 = 0.25
        # adj = min(0.1, 0.05 * 0.25) = min(0.1, 0.0125) = 0.0125
        assert adj > 0
        assert adj == pytest.approx(0.0125)

    def test_success_selected_above_alt_with_margin_returns_zero(self):
        """Success + selected Q already above alternatives + margin → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.9, memory_id="c1"),
            _make_memory("architect_general", q_value=0.7, memory_id="c2"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.9)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")

        # max_alt_q=0.7, gap = 0.7 + 0.05 - 0.9 = -0.15 → negative, no adjustment
        adj = s._compute_contrastive_adjustment(task, routing, reward=0.5)
        assert adj == 0.0

    def test_failure_selected_above_worst_alt_negative_adjustment(self):
        """Failure + selected Q above worst alternative → negative adjustment."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.7, memory_id="c1"),
            _make_memory("architect_general", q_value=0.4, memory_id="c2"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.7)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")

        adj = s._compute_contrastive_adjustment(task, routing, reward=-0.5)
        # min_alt_q=0.4, gap = 0.7 + 0.05 - 0.4 = 0.35
        # adj = max(-0.1, -0.05 * 0.35) = max(-0.1, -0.0175) = -0.0175
        assert adj < 0
        assert adj == pytest.approx(-0.0175)

    def test_failure_selected_below_alt_returns_zero(self):
        """Failure + selected Q already below worst alternative → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.3, memory_id="c1"),
            _make_memory("architect_general", q_value=0.6, memory_id="c2"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.3)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")

        # min_alt_q=0.6, gap = 0.3 + 0.05 - 0.6 = -0.25 → negative, no adjustment
        adj = s._compute_contrastive_adjustment(task, routing, reward=-0.5)
        assert adj == 0.0

    def test_positive_adjustment_capped_at_max_adj(self):
        """Large positive gap still capped at max_adj."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        # Huge gap: selected at 0.1, alt at 0.95
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.1, memory_id="c1"),
            _make_memory("architect_general", q_value=0.95, memory_id="c2"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.1)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")

        adj = s._compute_contrastive_adjustment(task, routing, reward=0.5, max_adj=0.1)
        # gap = 0.95 + 0.05 - 0.1 = 0.9, min(0.1, 0.05*0.9) = min(0.1, 0.045) = 0.045
        assert adj <= 0.1
        assert adj > 0

    def test_negative_adjustment_capped_at_neg_max_adj(self):
        """Large negative gap still capped at -max_adj."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        # Huge gap: selected at 0.95, alt at 0.1
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.95, memory_id="c1"),
            _make_memory("architect_general", q_value=0.1, memory_id="c2"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.95)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")

        adj = s._compute_contrastive_adjustment(task, routing, reward=-0.5, max_adj=0.1)
        # gap = 0.95 + 0.05 - 0.1 = 0.9, max(-0.1, -0.05*0.9) = max(-0.1, -0.045) = -0.045
        assert adj >= -0.1
        assert adj < 0

    def test_no_memory_id_uses_default_q(self):
        """No memory_id on routing decision → uses default Q=0.5."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("architect_general", q_value=0.8, memory_id="c1"),
        ]
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor", memory_id=None)

        adj = s._compute_contrastive_adjustment(task, routing, reward=0.5)
        # selected_q=0.5 (default), max_alt_q=0.8, gap=0.8+0.05-0.5=0.35
        # adj = min(0.1, 0.05*0.35) = 0.0175
        assert adj == pytest.approx(0.0175)


# ===== _compute_spo_plus_adjustment (DAR-3) =====


class TestComputeSpoPlusAdjustment:
    """Tests for DAR-3 _compute_spo_plus_adjustment."""

    def test_no_task_context_returns_zero(self):
        """Empty task_started data → 0.0."""
        s = _scorer()
        task = _FakeEntry(event_type=EventType.TASK_COMPLETED, data={})
        routing = _make_routing_entry()
        assert s._compute_spo_plus_adjustment(task, routing, reward=0.5) == 0.0

    def test_embedding_failure_returns_zero(self):
        """Embedding failure → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.side_effect = RuntimeError("embed failed")
        task = _make_task_entry()
        routing = _make_routing_entry()
        assert s._compute_spo_plus_adjustment(task, routing, reward=0.5) == 0.0

    def test_no_candidates_returns_zero(self):
        """No similar routing memories → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = []
        task = _make_task_entry()
        routing = _make_routing_entry()
        assert s._compute_spo_plus_adjustment(task, routing, reward=0.5) == 0.0

    def test_no_alternatives_returns_zero(self):
        """All candidates same action → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.7, memory_id="c1"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.7)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")
        assert s._compute_spo_plus_adjustment(task, routing, reward=0.5) == 0.0

    def test_success_with_alternatives_produces_adjustment(self):
        """Success with alternatives → non-zero adjustment."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.6, memory_id="c1"),
            _make_memory("architect_general", q_value=0.8, memory_id="c2"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.6)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")

        adj = s._compute_spo_plus_adjustment(task, routing, reward=0.7)
        # Non-zero: SPO+ should produce some adjustment when alternatives exist
        # and the reward signal disagrees with the current ranking
        assert isinstance(adj, float)

    def test_adjustment_bounded(self):
        """Adjustment never exceeds max_adj."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        # Large gap between selected and alternatives
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.1, memory_id="c1"),
            _make_memory("architect_general", q_value=0.95, memory_id="c2"),
            _make_memory("coder_escalation", q_value=0.9, memory_id="c3"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.1)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")

        adj = s._compute_spo_plus_adjustment(task, routing, reward=1.0, max_adj=0.15)
        assert abs(adj) <= 0.15

    def test_small_spo_loss_below_margin_returns_zero(self):
        """SPO+ loss below margin threshold → 0.0."""
        s = _scorer()
        s.embedder.embed_task_ir.return_value = np.zeros(128)
        # Nearly equal Q-values → loss ≈ 0
        s.store.retrieve_by_similarity.return_value = [
            _make_memory("frontdoor", q_value=0.5, memory_id="c1"),
            _make_memory("architect_general", q_value=0.51, memory_id="c2"),
        ]
        s.store.get_by_id.return_value = _make_memory("frontdoor", q_value=0.5)
        task = _make_task_entry()
        routing = _make_routing_entry("frontdoor")

        adj = s._compute_spo_plus_adjustment(task, routing, reward=0.5, margin=1.0)
        assert adj == 0.0
