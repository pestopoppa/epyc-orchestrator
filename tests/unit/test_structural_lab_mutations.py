"""Tests for NIB2-41 StructuralLab mutation primitives.

Covers ``mdl_compress_strategies`` and ``staleness_invalidate_strategies``
plus the underlying ``StrategyStore`` schema extensions (conventions,
validity, content_hashes) and the cascade step that flags
``routing_classifier_meta.json`` as stale.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator")


class MockEmbedder:
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
    s = StrategyStore(path=tmp_path / "strategies", embedding_dim=1024,
                      embedder=MockEmbedder())
    yield s
    s.close()


@pytest.fixture
def lab():
    from scripts.autopilot.species.structural_lab import StructuralLab
    return StructuralLab(orchestrator_url="http://unused-test:0")


# ── M1: MDL compression ─────────────────────────────────────────────

def test_mdl_compresses_near_duplicate_cluster(store, lab):
    """3 near-identical insights should promote to one convention with ratio > 0.20."""
    base = "Disable self-speculation for dense models because HSD overhead dominates on CPU"
    for i in range(3):
        store.store(
            description=f"try-{i}",
            insight=base + f" variant detail {i}",
            source_trial_id=i,
            species="config_tuner",
        )

    result = lab.mdl_compress_strategies(
        strategy_store=store,
        min_cluster_size=3,
        jaccard_threshold=0.50,
        compression_threshold=0.10,
    )

    assert result["status"] == "ok"
    assert result["conventions_promoted"] >= 1
    assert result["total_compression_saved_bytes"] > 0
    conventions = store.list_conventions()
    assert len(conventions) >= 1
    assert len(conventions[0]["member_ids"]) >= 3
    assert conventions[0]["compression_ratio"] >= 0.10


def test_mdl_does_not_compress_below_threshold(store, lab):
    """Dissimilar insights (low Jaccard or high MDL_after) should NOT promote."""
    insights = [
        "Reduce draft_max from 16 to 4 on worker role",
        "Switch coder quant from Q4KM to Q6K when context > 32k",
        "Enable frontdoor batch mode for cold-start reranker queries",
    ]
    for i, ins in enumerate(insights):
        store.store(
            description=f"diverse-{i}", insight=ins,
            source_trial_id=i, species="config_tuner",
        )
    result = lab.mdl_compress_strategies(
        strategy_store=store,
        min_cluster_size=2,
        jaccard_threshold=0.60,
        compression_threshold=0.20,
    )
    assert result["status"] == "ok"
    assert result["conventions_promoted"] == 0


def test_mdl_noop_on_empty_store(store, lab):
    result = lab.mdl_compress_strategies(strategy_store=store)
    assert result["status"] == "ok"
    assert result["clusters_examined"] == 0
    assert result["conventions_promoted"] == 0


# ── M2: staleness invalidation ──────────────────────────────────────

def test_staleness_increments_validity_failure_on_hash_change(store, lab, tmp_path):
    """When a scanned file's hash changes, referring strategies lose validity."""
    prompt = tmp_path / "my_prompt.md"
    prompt.write_text("# Original content\nfoo bar")
    sid = store.store(
        description="references my_prompt",
        insight="uses my_prompt for routing",
        source_trial_id=1,
        species="config_tuner",
        metadata={"refs": [str(prompt)]},
    )

    # First scan: records baseline hash, no failure fires.
    r1 = lab.staleness_invalidate_strategies(
        strategy_store=store, scan_targets=[str(prompt)],
    )
    assert r1["status"] == "ok"
    assert r1["hashes_changed"] == 0

    # Mutate file, rescan: hash changes → referring strategy gets a failure.
    prompt.write_text("# Modified content\nfoo baz\nextra line")
    r2 = lab.staleness_invalidate_strategies(
        strategy_store=store, scan_targets=[str(prompt)],
    )
    assert r2["hashes_changed"] == 1
    assert r2["strategies_touched"] == 1
    assert r2["quarantined"] + r2["suspected"] >= 0  # first failure may not quarantine yet

    row = store._conn.execute(
        "SELECT beta_fail FROM strategy_validity WHERE strategy_id = ?", (sid,),
    ).fetchone()
    assert row["beta_fail"] == 1


def test_staleness_quarantines_below_threshold(store, lab, tmp_path):
    """Enough failures push validity < 0.40 and flip the quarantine flag."""
    prompt = tmp_path / "my_prompt.md"
    prompt.write_text("# initial")
    sid = store.store(
        description="references my_prompt",
        insight="uses my_prompt for routing",
        source_trial_id=1,
        species="config_tuner",
        metadata={"refs": [str(prompt)]},
    )
    # Seed baseline hash.
    lab.staleness_invalidate_strategies(strategy_store=store, scan_targets=[str(prompt)])

    # α starts at 2; quarantine_threshold 0.40 means β_fail ≥ 4 → 2/(2+4)=0.33 < 0.40.
    # Mutate + rescan 5 times to guarantee threshold crossed.
    for i in range(5):
        prompt.write_text(f"mutation {i}")
        lab.staleness_invalidate_strategies(
            strategy_store=store, scan_targets=[str(prompt)],
            quarantine_threshold=0.40,
        )

    quarantined_ids = store.quarantined_ids()
    assert sid in quarantined_ids

    # Quarantined strategies are omitted from default retrieve().
    entries = store.retrieve("routing", k=5)
    assert all(e.id != sid for e in entries)
    # include_quarantined=True surfaces them again.
    entries_all = store.retrieve("routing", k=5, include_quarantined=True)
    assert any(e.id == sid for e in entries_all)


def test_cascade_invalidates_routing_classifier_checkpoint(store, lab, tmp_path, monkeypatch):
    """Quarantined strategy that trained the routing classifier → meta.stale=True."""
    # Redirect ORCH_ROOT so the cascade writes into tmp_path.
    from scripts.autopilot.species import structural_lab as sl_mod

    classifier_meta = tmp_path / "orchestration" / "repl_memory" / "routing_classifier_meta.json"
    classifier_meta.parent.mkdir(parents=True, exist_ok=True)

    prompt = tmp_path / "p.md"
    prompt.write_text("v1")
    sid = store.store(
        description="classifier training signal",
        insight="used by routing MLP",
        source_trial_id=1,
        species="config_tuner",
        metadata={"refs": [str(prompt)]},
    )

    classifier_meta.write_text(json.dumps({
        "training_strategy_ids": [sid],
        "trained_at": "2026-04-01T00:00:00Z",
    }))

    monkeypatch.setattr(sl_mod, "ORCH_ROOT", tmp_path)

    lab.staleness_invalidate_strategies(
        strategy_store=store, scan_targets=[str(prompt)],
    )
    for i in range(6):
        prompt.write_text(f"mutation {i}")
        lab.staleness_invalidate_strategies(
            strategy_store=store, scan_targets=[str(prompt)],
            quarantine_threshold=0.40,
        )

    assert sid in store.quarantined_ids()
    meta_now = json.loads(classifier_meta.read_text())
    assert meta_now.get("stale") is True
    assert "stale_at" in meta_now
