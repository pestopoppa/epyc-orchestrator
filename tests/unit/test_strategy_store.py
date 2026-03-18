"""Tests for StrategyStore — FAISS + SQLite strategy memory."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator")


class MockEmbedder:
    """Deterministic hash-based embedder for testing (no model needed)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec


@pytest.fixture
def store(tmp_path):
    from orchestration.repl_memory.strategy_store import StrategyStore
    s = StrategyStore(path=tmp_path / "strategies", embedding_dim=1024, embedder=MockEmbedder())
    yield s
    s.close()


class TestStrategyStore:

    def test_store_and_count(self, store):
        sid = store.store(
            description="Disable self-speculation for dense models",
            insight="HSD net-negative on hybrid",
            source_trial_id=1,
            species="config_tuner",
        )
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID
        assert store.count() == 1

    def test_store_multiple(self, store):
        for i in range(5):
            store.store(
                description=f"Strategy {i}",
                insight=f"Insight {i}",
                source_trial_id=i,
                species="explorer",
            )
        assert store.count() == 5

    def test_retrieve_returns_results(self, store):
        store.store("Enable caching for read-heavy workloads", "Cache hit rate 90%",
                     source_trial_id=1, species="perf_tuner")
        store.store("Increase batch size for throughput", "2x throughput at batch=8",
                     source_trial_id=2, species="perf_tuner")
        results = store.retrieve("caching performance", k=5)
        assert len(results) >= 1
        assert results[0].similarity_score > 0

    def test_retrieve_empty_store(self, store):
        results = store.retrieve("anything", k=5)
        assert results == []

    def test_retrieve_with_species_filter(self, store):
        store.store("Strategy A", "Insight A", source_trial_id=1, species="alpha")
        store.store("Strategy B", "Insight B", source_trial_id=2, species="beta")
        store.store("Strategy C", "Insight C", source_trial_id=3, species="alpha")

        results = store.retrieve("Strategy", k=10, species="alpha")
        assert all(r.species == "alpha" for r in results)

    def test_metadata_roundtrip(self, store):
        meta = {"key": "value", "nested": {"a": 1}}
        store.store("Test", "Test insight", source_trial_id=1, species="test",
                     metadata=meta)
        results = store.retrieve("Test", k=1)
        assert len(results) == 1
        assert results[0].metadata == meta

    def test_to_dict_serialization(self, store):
        store.store("Serialize me", "Check dict", source_trial_id=7, species="serializer")
        results = store.retrieve("Serialize me", k=1)
        d = results[0].to_dict()
        assert isinstance(d, dict)
        assert d["species"] == "serializer"
        assert d["source_trial_id"] == 7
        assert "id" in d
        assert "created_at" in d

    def test_close_is_safe(self, store):
        store.close()
        # Double close should not raise
        store.close()
