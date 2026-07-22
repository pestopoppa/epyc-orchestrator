"""DAR-L491 live write-path fix (ORCHESTRATOR_Q_TD_WRITE).

The production routing scorer blind-appended a fresh row per observation because
_update_routing_memory only TD-updates when routing_decision.memory_id is
pre-linked — which the sole ROUTING_DECISION emitter never sets. These tests
pin:
  * flag OFF  -> byte-identical legacy append (a fresh row per observation)
  * flag ON   -> find-or-update the (objective, action) row in place (TD)
  * flag ON, distinct objective -> not merged
  * the in-place update matches episodic_store.apply_td_update math
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from orchestration.repl_memory import q_scorer as q_scorer_mod
from orchestration.repl_memory.episodic_store import EpisodicStore, apply_td_update
from orchestration.repl_memory.progress_logger import EventType, ProgressEntry
from orchestration.repl_memory.q_scorer import QScorer, ScoringConfig


class _FakeEmbedder:
    """Deterministic embeddings keyed by objective text: identical objective ->
    identical vector, so the FAISS similarity lookup finds the prior row."""

    def embed_task_ir(self, context):
        obj = (context or {}).get("objective") or ""
        seed = int.from_bytes(hashlib.sha256(obj.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(1024).astype(np.float32)

    def embed_failure_context(self, context):  # pragma: no cover - unused here
        return self.embed_task_ir(context)


class _FakeLogger:
    def __init__(self):
        self.memory_updates = []
        self.logged = []

    def log_memory_update(self, memory_id, old_q, new_q, reward, task_id):
        self.memory_updates.append((memory_id, old_q, new_q, reward, task_id))

    def log(self, entry):
        self.logged.append(entry)


@pytest.fixture
def scorer(tmp_path):
    store = EpisodicStore(db_path=tmp_path / "sessions", use_faiss=True)
    # decay disabled so wall-clock between observations does not perturb TD math
    config = ScoringConfig(learning_rate=0.1, temporal_decay_rate=None)
    sc = QScorer(
        store=store,
        embedder=_FakeEmbedder(),
        logger=_FakeLogger(),
        reader=None,
        config=config,
    )
    yield sc
    store.close()


def _task_started(objective="solve X", task_type="chat"):
    return ProgressEntry(
        event_type=EventType.TASK_STARTED,
        task_id="t",
        data={"task_type": task_type, "objective": objective, "priority": "normal"},
    )


def _routing_decision(action="worker_general"):
    # memory_id defaults to None — exactly what log_task_started emits in prod.
    return ProgressEntry(
        event_type=EventType.ROUTING_DECISION,
        task_id="t",
        data={"routing": [action]},
    )


def _observe(sc, task_id, objective, action, reward):
    return sc._update_routing_memory(
        task_id, _task_started(objective), _routing_decision(action), reward,
    )


def test_flag_off_appends_a_fresh_row_per_observation(scorer, monkeypatch):
    monkeypatch.setattr(q_scorer_mod, "Q_TD_WRITE", False)
    r1 = _observe(scorer, "t1", "same objective", "worker_general", 0.4)
    r2 = _observe(scorer, "t2", "same objective", "worker_general", 0.4)
    assert r1["memories_created"] == 1 and r1["memories_updated"] == 0
    assert r2["memories_created"] == 1 and r2["memories_updated"] == 0
    mems = scorer.store.get_all_memories(action_type="routing")
    assert len(mems) == 2  # append-only: two rows for the same (objective, action)
    assert all(m.update_count == 0 for m in mems)


def test_flag_on_updates_in_place(scorer, monkeypatch):
    monkeypatch.setattr(q_scorer_mod, "Q_TD_WRITE", True)
    r1 = _observe(scorer, "t1", "same objective", "worker_general", 0.4)
    r2 = _observe(scorer, "t2", "same objective", "worker_general", 0.4)
    assert r1["memories_created"] == 1 and r1["memories_updated"] == 0
    assert r2["memories_created"] == 0 and r2["memories_updated"] == 1  # in-place TD
    mems = scorer.store.get_all_memories(action_type="routing")
    assert len(mems) == 1  # single consolidated row
    assert mems[0].update_count == 1


def test_flag_on_matches_apply_td_update_math(scorer, monkeypatch):
    monkeypatch.setattr(q_scorer_mod, "Q_TD_WRITE", True)
    _observe(scorer, "t1", "same objective", "worker_general", 0.4)
    _observe(scorer, "t2", "same objective", "worker_general", 0.4)
    mems = scorer.store.get_all_memories(action_type="routing")
    initial_q = 0.5 + 0.4 * 0.5  # first observation store()
    expected = apply_td_update(initial_q, 0.4, 0.1, temporal_decay_rate=None)
    assert mems[0].q_value == pytest.approx(expected)
    # the logger saw exactly one in-place update carrying old->new
    assert scorer.logger.memory_updates
    _, old_q, new_q, reward, _ = scorer.logger.memory_updates[-1]
    assert old_q == pytest.approx(initial_q)
    assert new_q == pytest.approx(expected)


def test_flag_on_does_not_merge_distinct_objectives(scorer, monkeypatch):
    monkeypatch.setattr(q_scorer_mod, "Q_TD_WRITE", True)
    _observe(scorer, "t1", "objective A", "worker_general", 0.4)
    _observe(scorer, "t2", "objective B", "worker_general", 0.4)
    mems = scorer.store.get_all_memories(action_type="routing")
    assert len(mems) == 2  # distinct objectives -> distinct rows


def test_flag_on_does_not_merge_distinct_actions(scorer, monkeypatch):
    monkeypatch.setattr(q_scorer_mod, "Q_TD_WRITE", True)
    _observe(scorer, "t1", "same objective", "worker_general", 0.4)
    _observe(scorer, "t2", "same objective", "architect_general", 0.4)
    mems = scorer.store.get_all_memories(action_type="routing")
    assert len(mems) == 2  # same objective, different action -> distinct rows


def test_prelinked_memory_id_still_updates_regardless_of_flag(scorer, monkeypatch):
    """The original (pre-linked) update branch is unchanged by the flag."""
    monkeypatch.setattr(q_scorer_mod, "Q_TD_WRITE", False)
    _observe(scorer, "t1", "same objective", "worker_general", 0.4)
    existing = scorer.store.get_all_memories(action_type="routing")[0]
    rd = _routing_decision("worker_general")
    rd.memory_id = existing.id
    res = scorer._update_routing_memory("t2", _task_started("same objective"), rd, 0.4)
    assert res["memories_updated"] == 1 and res["memories_created"] == 0
    assert len(scorer.store.get_all_memories(action_type="routing")) == 1
