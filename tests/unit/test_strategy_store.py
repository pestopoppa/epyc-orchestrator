"""Tests for StrategyStore — FAISS + SQLite strategy memory."""

from __future__ import annotations

import hashlib
import sys

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

    def test_retrieve_excludes_source_trial_ids(self, store):
        store.store("Strategy A", "Insight A", source_trial_id=1, species="alpha")
        store.store("Strategy B", "Insight B", source_trial_id=2, species="alpha")

        results = store.retrieve("Strategy", k=10, excluded_trial_ids={2})

        assert results
        assert all(r.source_trial_id != 2 for r in results)

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


class TestAP28HybridRetrieval:
    """AP-28: FTS5 + RRF fusion, content-hash staleness, validity weighting."""

    def test_fts5_index_present(self, store):
        # FTS5 virtual table should exist after _init_schema
        rows = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='strategies_fts'"
        ).fetchall()
        assert len(rows) == 1
        assert getattr(store, "_fts_enabled", False) is True

    def test_entry_type_default_raw(self, store):
        sid = store.store("desc", "insight", source_trial_id=1, species="alpha")
        row = store._conn.execute(
            "SELECT entry_type FROM strategies WHERE id=?", (sid,)
        ).fetchone()
        assert row["entry_type"] == "raw"

    def test_entry_type_explicit(self, store):
        sid = store.store(
            "desc", "insight", source_trial_id=1, species="alpha", entry_type="pattern"
        )
        row = store._conn.execute(
            "SELECT entry_type FROM strategies WHERE id=?", (sid,)
        ).fetchone()
        assert row["entry_type"] == "pattern"

    def test_context_hash_recorded(self, store):
        sid = store.store("desc", "insight", source_trial_id=1, species="alpha")
        row = store._conn.execute(
            "SELECT context_hash FROM strategies WHERE id=?", (sid,)
        ).fetchone()
        # context files may not exist in tests → hash of empty input is fine,
        # we just need a non-NULL string value (incl. empty).
        assert row["context_hash"] is not None

    def test_bm25_exact_term_match(self, store):
        # FAISS via MockEmbedder is hash-based and has zero semantic fidelity.
        # BM25 must surface the entry that contains the exact query term.
        store.store("Speculation tuning for Qwen3.5", "Disable HSD",
                     source_trial_id=1, species="config_tuner")
        store.store("Increase ubatch size", "Throughput +20%",
                     source_trial_id=2, species="config_tuner")
        store.store("Cache hit rate optimisation", "Use prefix cache",
                     source_trial_id=3, species="config_tuner")

        results = store.retrieve("Qwen3.5 speculation", k=1)
        assert len(results) == 1
        assert "Qwen3.5" in results[0].description

    def test_rrf_fuses_both_signals(self, store):
        # Even when FAISS+BM25 disagree, RRF should produce a deterministic
        # ordering and never error.
        for i in range(8):
            store.store(f"Strategy {i}", f"Insight {i}",
                        source_trial_id=i, species="explorer")
        results = store.retrieve("Strategy 3", k=3)
        assert 1 <= len(results) <= 3
        # All returned entries must carry the diagnostic fields.
        for r in results:
            assert r.entry_type == "raw"
            assert r.staleness == 1.0
            assert 0.0 <= r.validity_score <= 1.0

    def test_quarantined_entries_excluded(self, store):
        sid = store.store("Quarantine me", "should be hidden",
                          source_trial_id=1, species="alpha")
        # Force quarantine via repeated failures (NIB2-41 pathway)
        for _ in range(20):
            store.update_validity(sid, failure=True)
        results = store.retrieve("Quarantine", k=5)
        assert all(r.id != sid for r in results)
        # And re-includable when explicitly requested
        results_full = store.retrieve("Quarantine", k=5, include_quarantined=True)
        assert any(r.id == sid for r in results_full)

    def test_staleness_penalises_old_entries(self, store):
        # Insert with fixed hash, then pretend the world moved on by
        # rewriting compute_context_hash to return a different string.
        store.store("Old entry", "from epoch A", source_trial_id=1, species="alpha")
        # Confirm it's currently fresh
        results = store.retrieve("Old entry", k=1)
        assert results[0].staleness == 1.0
        # Force a different epoch by monkeypatching the helper
        store.compute_context_hash = lambda *a, **kw: "DIFFERENTHASH0001"
        results = store.retrieve("Old entry", k=1)
        assert results[0].staleness == 0.5

    def test_backfill_fts_idempotent(self, store):
        # Insert directly into ``strategies`` bypassing the store() FTS path,
        # then call backfill_fts to populate the index.
        store._conn.execute(
            "INSERT INTO strategies(id, description, insight, source_trial_id, species, "
            "created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("legacy-1", "legacy desc", "legacy insight", 1, "old", "now", "{}"),
        )
        store._conn.commit()
        n1 = store.backfill_fts()
        assert n1 >= 1
        # Idempotent: second call should not double-insert.
        n2 = store.backfill_fts()
        assert n2 == 0

    def test_bm25_handles_punctuation_safely(self, store):
        # FTS5 MATCH chokes on raw punctuation; sanitiser must handle it.
        store.store("Mutation type=targeted_fix", "boost q",
                    source_trial_id=1, species="prompt_forge")
        # Punctuation-heavy query should not raise.
        results = store.retrieve("type=targeted_fix?!", k=2)
        assert isinstance(results, list)
