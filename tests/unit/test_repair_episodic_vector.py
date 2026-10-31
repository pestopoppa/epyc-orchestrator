"""Position-preserving content repair against isolated on-disk stores."""

import json
import sqlite3

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")
from scripts.maintenance.repair_episodic_vector import repair_vector


@pytest.fixture
def store(tmp_path):
    vectors = np.eye(3, dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    faiss.write_index(index, str(tmp_path / "embeddings.faiss"))
    np.save(tmp_path / "id_map.npy", np.array(["a", "b", "c"], dtype=object))
    with sqlite3.connect(tmp_path / "episodic.db") as conn:
        conn.execute("CREATE TABLE memories (id TEXT, embedding_idx INTEGER, context TEXT)")
        conn.executemany(
            "INSERT INTO memories VALUES (?, ?, ?)",
            [
                (mid, i, json.dumps({"task_description": "repair this task"}))
                for i, mid in enumerate(["a", "b", "c"])
            ],
        )
    return tmp_path, vectors


def test_replaces_only_requested_position_with_backup(store):
    path, original = store
    before = {
        name: (path / name).read_bytes()
        for name in ["episodic.db", "id_map.npy", "embeddings.faiss"]
    }
    seen = []

    def embed(text):
        seen.append(text)
        return np.array([3, 4, 0], dtype=np.float32)

    result = repair_vector(path, "b", apply=True, embed=embed)
    actual = faiss.read_index(str(path / "embeddings.faiss")).reconstruct_n(0, 3)
    assert result["applied"] and seen and "repair this task" in seen[0]
    np.testing.assert_array_equal(actual[[0, 2]], original[[0, 2]])
    np.testing.assert_allclose(actual[1], [0.6, 0.8, 0])
    from pathlib import Path

    assert Path(result["backup"]).read_bytes() == before["embeddings.faiss"]
    for name in ["episodic.db", "id_map.npy"]:
        assert (path / name).read_bytes() == before[name]


def test_dry_run_does_not_embed_or_change_index(store):
    path, _ = store
    before = (path / "embeddings.faiss").read_bytes()

    def forbidden(_):
        pytest.fail("dry run contacted embedder")

    assert not repair_vector(path, "b", embed=forbidden)["applied"]
    assert (path / "embeddings.faiss").read_bytes() == before
    assert not list(path.glob("*.pre-vector-repair-*"))


@pytest.mark.parametrize("vector", [[0, 0, 0], [float("nan"), 1, 0], [1, 0]])
def test_invalid_embedding_never_publishes(store, vector):
    path, _ = store
    before = (path / "embeddings.faiss").read_bytes()
    with pytest.raises(ValueError, match="invalid vector"):
        repair_vector(path, "b", apply=True, embed=lambda _: vector)
    assert (path / "embeddings.faiss").read_bytes() == before


def test_mismatched_position_fails_before_embed(store):
    path, _ = store
    with sqlite3.connect(path / "episodic.db") as conn:
        conn.execute("UPDATE memories SET embedding_idx=2 WHERE id='b'")
    with pytest.raises(ValueError, match="embedding_idx"):
        repair_vector(path, "b", apply=True, embed=lambda _: pytest.fail("embed called"))


def test_embedder_outage_preserves_index(store):
    path, _ = store
    before = (path / "embeddings.faiss").read_bytes()

    def unavailable(_):
        raise RuntimeError("offline")

    with pytest.raises(RuntimeError, match="offline"):
        repair_vector(path, "b", apply=True, embed=unavailable)
    assert (path / "embeddings.faiss").read_bytes() == before


def test_atomic_replace_failure_preserves_original(store, monkeypatch):
    from pathlib import Path

    path, _ = store
    before = (path / "embeddings.faiss").read_bytes()

    def fail_replace(self, target):
        raise OSError("injected publish failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="publish failure"):
        repair_vector(path, "b", apply=True, embed=lambda _: [1, 0, 0])
    assert (path / "embeddings.faiss").read_bytes() == before
    assert not list(path.glob(".vector-repair-*"))


def test_other_row_binding_mismatch_refuses_repair(store):
    path, _ = store
    with sqlite3.connect(path / "episodic.db") as conn:
        conn.execute("UPDATE memories SET embedding_idx=0 WHERE id='c'")
    with pytest.raises(ValueError, match="cross-store"):
        repair_vector(path, "b", apply=True, embed=lambda _: pytest.fail("embed called"))


def test_strict_embed_disables_parallel_hash_fallback(monkeypatch):
    from orchestration.repl_memory import embedder
    from scripts.maintenance.repair_episodic_vector import strict_embed

    seen = []

    class FakeEmbedder:
        def __init__(self, config):
            assert not config.use_fallback
            assert not config.use_parallel
            assert not config.allow_subprocess

        def embed_text(self, text, side):
            seen.append((text, side))
            raise RuntimeError("offline")

    monkeypatch.setattr(embedder, "TaskEmbedder", FakeEmbedder)
    with pytest.raises(RuntimeError, match="offline"):
        strict_embed("test")
    assert seen == [("test", "document")]


def test_real_strict_embedder_outage_never_uses_hash_or_parallel(monkeypatch):
    from orchestration.repl_memory.embedder import TaskEmbedder
    from scripts.maintenance.repair_episodic_vector import strict_embed

    monkeypatch.setattr(TaskEmbedder, "_check_server", lambda self: False)

    def forbidden(*args, **kwargs):
        pytest.fail("strict repair attempted a fallback path")

    monkeypatch.setattr(TaskEmbedder, "_get_parallel_client", forbidden)
    monkeypatch.setattr(TaskEmbedder, "_generate_embedding_fallback", forbidden)
    monkeypatch.setattr(TaskEmbedder, "_generate_embedding_llama", forbidden)
    with pytest.raises(RuntimeError, match="hash fallback disabled"):
        strict_embed("repair this task")


@pytest.mark.parametrize("position", [1, 9, -1])
def test_extra_db_row_cannot_alias_or_escape_index(store, position):
    path, _ = store
    before = (path / "embeddings.faiss").read_bytes()
    with sqlite3.connect(path / "episodic.db") as conn:
        conn.execute("INSERT INTO memories VALUES ('extra', ?, '{}')", (position,))
    with pytest.raises(ValueError, match="reverse cross-store"):
        repair_vector(path, "b", apply=True, embed=lambda _: pytest.fail("embed called"))
    assert (path / "embeddings.faiss").read_bytes() == before


@pytest.mark.parametrize(
    "value", [np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1))]
)
def test_extreme_finite_vectors_are_normalized_before_float32_cast(store, value):
    path, original = store
    result = repair_vector(path, "b", apply=True, embed=lambda _: [value, value, value])
    assert result["applied"]
    vectors = faiss.read_index(str(path / "embeddings.faiss")).reconstruct_n(0, 3)
    assert np.isfinite(vectors[1]).all()
    assert np.linalg.norm(vectors[1].astype(np.float64)) == pytest.approx(1.0)
    np.testing.assert_array_equal(vectors[[0, 2]], original[[0, 2]])


def test_null_unindexed_db_row_remains_allowed(store):
    path, _ = store
    with sqlite3.connect(path / "episodic.db") as conn:
        conn.execute("INSERT INTO memories VALUES ('unindexed', NULL, '{}')")
    assert not repair_vector(path, "b")["applied"]
