"""Tests for KnowledgeDistiller — L1->L2->L3 strategy memory consolidation (AP-29)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator")


class ClusterableEmbedder:
    """Embedder that produces tightly-clustered vectors for entries that share
    a substring tag.

    Each entry embeds to a base vector (one of the registered tags) plus a
    small per-entry noise vector. The cosine similarity between entries
    sharing a tag is far higher than the 0.75 cluster threshold, while
    entries with different tags are roughly orthogonal.
    """

    def __init__(self, dim: int = 32, tags: tuple[str, ...] = ("alpha", "beta", "gamma", "delta")):
        self.dim = dim
        rng = np.random.RandomState(0)
        self._base = {t: rng.randn(dim).astype(np.float32) for t in tags}
        for k, v in self._base.items():
            self._base[k] = v / (np.linalg.norm(v) + 1e-9)

    def embed_text(self, text: str) -> np.ndarray:
        # Pick the first registered tag that appears in the text.
        for tag, base in self._base.items():
            if tag in text:
                # Tiny per-text noise so different entries with the same tag
                # are not literally identical, while still > 0.99 cosine.
                rng = np.random.RandomState(
                    int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
                )
                noise = 0.01 * rng.randn(self.dim).astype(np.float32)
                v = base + noise
                return v / (np.linalg.norm(v) + 1e-9)
        # Fallback for unseen tags
        rng = np.random.RandomState(
            int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        )
        v = rng.randn(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


@pytest.fixture
def store(tmp_path):
    from orchestration.repl_memory.strategy_store import StrategyStore

    s = StrategyStore(
        path=tmp_path / "strategies",
        embedding_dim=32,
        embedder=ClusterableEmbedder(dim=32),
    )
    yield s
    s.close()


@pytest.fixture
def distiller(store):
    from orchestration.repl_memory.knowledge_distiller import KnowledgeDistiller

    return KnowledgeDistiller(store)


class TestKnowledgeDistiller:
    def test_no_action_below_mdl(self, store, distiller):
        # Below MDL_MIN_CLUSTER_SIZE — nothing should promote.
        store.store("alpha tag entry one", "rich detailed insight one",
                     source_trial_id=1, species="prompt_forge")
        store.store("alpha tag entry two", "rich detailed insight two",
                     source_trial_id=2, species="prompt_forge")
        stats = distiller.distill(trial_id=10)
        assert stats.patterns_created == 0
        assert stats.conventions_created == 0

    def test_l1_to_l2_pattern_promotion(self, store, distiller):
        # Three+ similar entries within a species → one pattern.
        for i in range(4):
            store.store(
                description=f"alpha tag detailed strategy entry index {i} long version",
                insight=f"alpha tag insight entry {i} with extra explanatory detail",
                source_trial_id=i,
                species="prompt_forge",
            )
        stats = distiller.distill(trial_id=20)
        assert stats.patterns_created == 1
        assert stats.raw_entries_consolidated == 4

        # Source rows must be quarantined → retrieve() should not return them.
        results = store.retrieve("alpha", k=10)
        # The pattern is the only thing left visible.
        types = [r.entry_type for r in results]
        assert "pattern" in types
        assert "raw" not in types

    def test_pattern_skipped_when_not_compressible(self, store, distiller):
        # Cluster has one large seed and two tiny entries; the pattern row
        # would be larger than the cluster total → MDL check skips it.
        store.store(
            description="alpha tag " + ("very long detailed text " * 30),
            insight="alpha tag " + ("explanatory paragraph " * 30),
            source_trial_id=1,
            species="prompt_forge",
        )
        for i in range(2, 4):
            store.store(
                description=f"alpha {i}",
                insight="a",
                source_trial_id=i,
                species="prompt_forge",
            )
        stats = distiller.distill(trial_id=30)
        assert stats.patterns_created == 0
        assert stats.skipped_below_mdl >= 1

    def test_l2_to_l3_convention_cross_species(self, store, distiller):
        # Three species each with three+ similar raw entries → three patterns
        # → cross-species convention.
        for species in ("prompt_forge", "numeric_swarm", "structural_lab"):
            for i in range(3):
                store.store(
                    description=f"alpha tag long shared description for {species} run {i}",
                    insight=f"alpha tag matching insight describing {species} trial {i}",
                    source_trial_id=i,
                    species=species,
                )
        stats = distiller.distill(trial_id=40)
        assert stats.patterns_created >= 3
        assert stats.conventions_created >= 1

        # Convention must be stored under species='all'
        rows = store._conn.execute(
            "SELECT id, species FROM strategies WHERE entry_type = 'convention'"
        ).fetchall()
        assert len(rows) >= 1
        assert all(r["species"] == "all" for r in rows)

    def test_idempotent(self, store, distiller):
        # Running distill twice must not double-promote.
        for i in range(4):
            store.store(
                description=f"beta tag detailed strategy entry index {i} long version",
                insight=f"beta tag insight entry {i} with extra explanatory detail",
                source_trial_id=i,
                species="seeder",
            )
        stats_a = distiller.distill(trial_id=50)
        stats_b = distiller.distill(trial_id=51)
        assert stats_a.patterns_created == 1
        assert stats_b.patterns_created == 0

    def test_distillation_stats_to_dict(self):
        from orchestration.repl_memory.knowledge_distiller import DistillationStats

        s = DistillationStats(patterns_created=2, conventions_created=1)
        d = s.to_dict()
        assert d["patterns_created"] == 2
        assert d["conventions_created"] == 1
        assert d["raw_entries_consolidated"] == 0
        assert d["notes"] == []
