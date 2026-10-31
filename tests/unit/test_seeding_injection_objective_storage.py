"""EPD-3-R7: eval rewards must store the full source objective.

The test uses both production eval payload builders, the native RewardRequest
and endpoint, QScorer, and a private SQLite EpisodicStore. It checks the stored
database contexts so payload construction alone cannot mask another truncation.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError


_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK = _ROOT / "scripts" / "benchmark"
sys.path.insert(0, str(_BENCHMARK))
_SPEC = importlib.util.spec_from_file_location(
    "seeding_injection_objective_storage", _BENCHMARK / "seeding_injection.py"
)
_INJECTION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _INJECTION
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_INJECTION)


class _LocalRewardClient:
    """Route the writer's real HTTP request into the real API handler."""

    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def post(self, _url, *, json, timeout):
        from src.api.models import RewardRequest
        from src.api.routes.chat import inject_reward

        request = RewardRequest.model_validate(json)
        result = asyncio.run(inject_reward(request, state=self.state))
        return SimpleNamespace(status_code=200, json=lambda: result)


class _EmbedderClient:
    """Observe the real canonical text sent to the embedder endpoint."""

    def __init__(self, inputs, embedding):
        self.inputs = inputs
        self.embedding = embedding

    def post(self, _url, *, json, timeout):
        self.inputs.append(json["content"])
        return SimpleNamespace(status_code=200, json=lambda: {"embedding": self.embedding})


class _ImmediateExecutor:
    """Run the actual single-reward sender synchronously for a private DB."""

    def submit(self, fn, url, payload, action_key):
        future = Future()
        try:
            future.set_result(fn(url, payload, action_key))
        except Exception as exc:  # surface sender errors through normal accounting
            future.set_exception(exc)
        return future


class _QuietLogger:
    def log(self, _entry):
        pass

    def log_memory_update(self, *_args):
        pass


def _embedding():
    # A deterministic non-degenerate vector, intentionally shared across
    # objectives with the same canonical bounded embedding view.
    return np.random.default_rng(4321).standard_normal(1024).astype(np.float32).tolist()


def _stored_contexts(store):
    with sqlite3.connect(store.sqlite_path) as conn:
        rows = conn.execute("SELECT context FROM memories ORDER BY id").fetchall()
    return [json.loads(row[0]) for row in rows]


def test_eval_writers_round_trip_full_objectives_to_private_store(tmp_path, monkeypatch):
    from orchestration.repl_memory.episodic_store import EpisodicStore
    from orchestration.repl_memory.q_scorer import QScorer, ScoringConfig
    from src.api.services import memrl
    from src.api.state import AppState

    store = EpisodicStore(db_path=tmp_path / "sessions", use_faiss=True)
    scorer = QScorer(
        store=store,
        embedder=Mock(),
        logger=_QuietLogger(),
        reader=None,
        config=ScoringConfig(learning_rate=0.1, temporal_decay_rate=None),
    )
    state = AppState(q_scorer=scorer, episodic_store=store)
    monkeypatch.setattr(memrl, "features", lambda: SimpleNamespace(memrl=True))
    monkeypatch.setattr(_INJECTION, "_get_reward_executor", lambda: _ImmediateExecutor())

    long_a = "a" * 2100 + " suffix-A: preserve every character"
    long_b = "a" * 2100 + " suffix-B: these are different objectives"
    unicode_short = "計算 λe\u0301 — résumé"
    embedding_inputs = []
    vector = _embedding()

    monkeypatch.setattr(_INJECTION.httpx, "Client", lambda: _LocalRewardClient(state))

    try:
        three_way = _INJECTION._inject_3way_rewards_http(
            prompt=long_a,
            suite="math",
            question_id="long-a",
            rewards={"SELF:direct": 1.0},
            metadata={},
            url="http://private-test/chat/reward",
            client=_EmbedderClient(embedding_inputs, vector),
        )
        per_role = _INJECTION._inject_per_role_rewards_http(
            prompt=long_b,
            suite="math",
            question_id="long-b",
            rewards={"frontdoor": 1.0},
            metadata={},
            url="http://private-test/chat/reward",
            client=_EmbedderClient(embedding_inputs, vector),
        )
        unicode_result = _INJECTION._inject_3way_rewards_http(
            prompt=unicode_short,
            suite="general",
            question_id="unicode-short",
            rewards={"SELF:direct": 0.5},
            metadata={},
            url="http://private-test/chat/reward",
            client=_EmbedderClient(embedding_inputs, vector),
        )

        assert [
            three_way["acknowledged"],
            per_role["acknowledged"],
            unicode_result["acknowledged"],
        ] == [1, 1, 1]
        from orchestration.repl_memory.memory_record import embedding_text_for

        expected_embedding_inputs = [
            embedding_text_for(objective=long_a, task_type="math"),
            embedding_text_for(objective=long_b, task_type="math"),
            embedding_text_for(objective=unicode_short, task_type="general"),
        ]
        assert embedding_inputs == expected_embedding_inputs
        assert len(embedding_inputs[0]) == 2000
        assert embedding_inputs[0] == embedding_inputs[1]
        assert embedding_inputs[0] != long_a

        contexts = _stored_contexts(store)
        objectives = {
            context["metrics"]["question_id"]: context["objective"] for context in contexts
        }
        assert objectives == {
            "long-a": long_a,
            "long-b": long_b,
            "unicode-short": unicode_short,
        }
        assert objectives["long-a"][:200] == objectives["long-b"][:200]
        assert objectives["long-a"] != objectives["long-b"]

        # Native request semantics are retained: an empty supplied objective is
        # accepted as empty, while omitting the required field is rejected.
        from src.api.models import RewardRequest

        assert RewardRequest(task_description="", action="a", reward=0).task_description == ""
        with pytest.raises(ValidationError):
            RewardRequest.model_validate({"action": "a", "reward": 0})
    finally:
        store.close()
