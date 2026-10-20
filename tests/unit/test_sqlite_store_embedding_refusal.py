"""SQLiteSessionStore must REFUSE an unusable embeddings file, never destroy it.

REPL-EMB-2.2 follow-on M12: an embedding-dimension change (1024-d BGE ->
256-d EmbeddingGemma 2) must never destroy a session store.  The old
``_load_embeddings`` unlinked ``session_embeddings.npy`` on a dimension mismatch
(and on a parse failure) and silently recreated it empty.  The contract now:

* the existing file stays byte-identical (never unlinked, truncated, or
  overwritten);
* the store still opens (session metadata lives in SQLite) with embeddings
  disabled, a loud ERROR naming stored dim, expected dim and path;
* ``store_embedding`` raises ``SessionEmbeddingsRefused`` (no silent drop);
  ``search_by_embedding`` returns ``[]`` with a warning;
* matching dimension, a missing file and a zero-byte file behave as before.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from src.session import Session, SessionEmbeddingsRefused, SQLiteSessionStore

SQLITE_LOGGER = "src.session.sqlite_store"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return _digest(path), st.st_size, st.st_mtime_ns


def _write_npy(path: Path, rows: int, dim: int) -> None:
    rng = np.random.default_rng(1234)
    np.save(path, rng.standard_normal((rows, dim)).astype(np.float32))


@pytest.fixture
def no_deletes(monkeypatch):
    """Any unlink/remove while a refused store loads is a test failure."""

    def _boom(*_a, **_k):
        raise AssertionError("embeddings file deletion attempted")

    monkeypatch.setattr(Path, "unlink", _boom)
    monkeypatch.setattr(os, "unlink", _boom)
    monkeypatch.setattr(os, "remove", _boom)


def _open(tmp_path: Path, emb_path: Path, dim: int) -> SQLiteSessionStore:
    return SQLiteSessionStore(
        db_path=tmp_path / "sessions.db", embeddings_path=emb_path, embedding_dim=dim
    )


class TestDimensionMismatchIsRefused:
    @pytest.mark.parametrize(
        ("stored_dim", "expected_dim"),
        [(1024, 256), (256, 1024)],
        ids=["bge1024_store_vs_gemma256_config", "gemma256_store_vs_bge1024_config"],
    )
    def test_file_kept_byte_identical(self, tmp_path, no_deletes, caplog, stored_dim, expected_dim):
        emb = tmp_path / "session_embeddings.npy"
        _write_npy(emb, rows=5, dim=stored_dim)
        before = _snapshot(emb)

        with caplog.at_level(logging.ERROR, logger=SQLITE_LOGGER):
            store = _open(tmp_path, emb, expected_dim)

        try:
            assert emb.exists()
            assert _snapshot(emb) == before  # bytes, size AND mtime: never even rewritten
            assert not store.embeddings_available
            refusal = store.embeddings_refusal
            assert isinstance(refusal, SessionEmbeddingsRefused)
            assert refusal.reason == "dim_mismatch"
            assert refusal.stored_dim == stored_dim
            assert refusal.expected_dim == expected_dim
            assert refusal.path == emb

            # Loud: ERROR naming stored dim, expected dim and the file path.
            errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
            assert errors, "refusal must be logged at ERROR"
            joined = "\n".join(errors)
            assert str(emb) in joined
            assert f"stored dimension {stored_dim}" in joined
            assert f"expected dimension {expected_dim}" in joined
        finally:
            store.close()
        assert _snapshot(emb) == before  # close() must not touch it either

    def test_load_embeddings_raises_typed_error_and_keeps_file(self, tmp_path, no_deletes):
        emb = tmp_path / "session_embeddings.npy"
        _write_npy(emb, rows=3, dim=1024)
        before = _snapshot(emb)
        store = _open(tmp_path, emb, 256)
        try:
            with pytest.raises(SessionEmbeddingsRefused) as excinfo:
                store._load_embeddings()
            assert excinfo.value.stored_dim == 1024
            assert excinfo.value.expected_dim == 256
            assert str(emb) in str(excinfo.value)
            assert _snapshot(emb) == before
        finally:
            store.close()

    def test_store_embedding_refuses_without_writing(self, tmp_path, no_deletes):
        emb = tmp_path / "session_embeddings.npy"
        _write_npy(emb, rows=4, dim=1024)
        before = _snapshot(emb)
        store = _open(tmp_path, emb, 256)
        try:
            session = Session.create(name="S")
            store.create_session(session)
            with pytest.raises(SessionEmbeddingsRefused):
                store.store_embedding(session.id, np.ones(256, dtype=np.float32))
            # no metadata row was inserted for a vector that was never stored
            with store._get_connection() as conn:
                assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
            assert _snapshot(emb) == before
        finally:
            store.close()

    def test_search_returns_empty_and_warns(self, tmp_path, no_deletes, caplog):
        emb = tmp_path / "session_embeddings.npy"
        _write_npy(emb, rows=4, dim=1024)
        before = _snapshot(emb)
        store = _open(tmp_path, emb, 256)
        try:
            with caplog.at_level(logging.WARNING, logger=SQLITE_LOGGER):
                assert store.search_by_embedding(np.ones(256, dtype=np.float32)) == []
            assert any("embeddings disabled" in r.getMessage() for r in caplog.records)
            assert _snapshot(emb) == before
        finally:
            store.close()

    def test_session_metadata_still_works(self, tmp_path, no_deletes):
        emb = tmp_path / "session_embeddings.npy"
        _write_npy(emb, rows=2, dim=1024)
        store = _open(tmp_path, emb, 256)
        try:
            session = Session.create(name="still usable")
            store.create_session(session)
            fetched = store.get_session(session.id)
            assert fetched is not None and fetched.name == "still usable"
        finally:
            store.close()

    def test_reopening_with_the_stored_dim_recovers_every_vector(self, tmp_path, no_deletes):
        """The refusal lost nothing: the matching config reads the original rows."""
        emb = tmp_path / "session_embeddings.npy"
        _write_npy(emb, rows=6, dim=256)
        original = np.load(emb)

        refused = _open(tmp_path, emb, 1024)
        refused.close()
        assert not refused.embeddings_available

        ok = _open(tmp_path, emb, 256)
        try:
            assert ok.embeddings_available
            assert ok.embeddings_refusal is None
            assert ok._next_embedding_idx == 6
            np.testing.assert_array_equal(np.asarray(ok._embeddings), original)
        finally:
            ok.close()


class TestUnreadableFileIsRefused:
    """The other branch that used to unlink: a file np.load cannot parse."""

    def test_garbage_bytes_kept(self, tmp_path, no_deletes, caplog):
        emb = tmp_path / "session_embeddings.npy"
        emb.write_bytes(b"corrupt data")
        before = _snapshot(emb)
        with caplog.at_level(logging.ERROR, logger=SQLITE_LOGGER):
            store = _open(tmp_path, emb, 1024)
        try:
            assert _snapshot(emb) == before
            assert not store.embeddings_available
            assert store.embeddings_refusal.reason == "unreadable"
            assert store.embeddings_refusal.stored_dim is None
            assert store.embeddings_refusal.expected_dim == 1024
            joined = "\n".join(r.getMessage() for r in caplog.records)
            assert str(emb) in joined and "expected dimension 1024" in joined
        finally:
            store.close()

    def test_truncated_npy_kept(self, tmp_path, no_deletes):
        """A valid header with cut-short data (crash mid-np.save) is recoverable
        by hand, so it must survive."""
        emb = tmp_path / "session_embeddings.npy"
        _write_npy(emb, rows=20, dim=1024)
        raw = emb.read_bytes()
        emb.write_bytes(raw[: len(raw) // 2])
        before = _snapshot(emb)
        store = _open(tmp_path, emb, 1024)
        try:
            assert _snapshot(emb) == before
            assert store.embeddings_refusal.reason == "unreadable"
            with pytest.raises(SessionEmbeddingsRefused):
                store.store_embedding("s", np.ones(1024, dtype=np.float32))
        finally:
            store.close()

    def test_one_dimensional_array_kept(self, tmp_path, no_deletes):
        emb = tmp_path / "session_embeddings.npy"
        np.save(emb, np.arange(10, dtype=np.float32))
        before = _snapshot(emb)
        store = _open(tmp_path, emb, 1024)
        try:
            assert _snapshot(emb) == before
            assert store.embeddings_refusal.reason == "unreadable"
        finally:
            store.close()


class TestUnchangedBehaviour:
    def test_matching_dimension_loads(self, tmp_path):
        emb = tmp_path / "session_embeddings.npy"
        _write_npy(emb, rows=7, dim=64)
        original = np.load(emb)
        store = _open(tmp_path, emb, 64)
        try:
            assert store.embeddings_available
            assert store.embeddings_refusal is None
            assert store._next_embedding_idx == 7
            np.testing.assert_array_equal(np.asarray(store._embeddings), original)

            session = Session.create(name="S")
            store.create_session(session)
            vec = original[3]
            idx = store.store_embedding(session.id, vec)
            assert idx == 7
            hits = store.search_by_embedding(vec, limit=3)
            assert hits and hits[0][0] == session.id and hits[0][1] > 0.99
        finally:
            store.close()

    def test_missing_file_is_created(self, tmp_path):
        emb = tmp_path / "nested_not_needed.npy"
        assert not emb.exists()
        store = _open(tmp_path, emb, 64)
        try:
            assert store.embeddings_available
            assert emb.exists()
            assert np.load(emb).shape == (1000, 64)
            assert store._next_embedding_idx == 0
        finally:
            store.close()

    def test_zero_byte_file_is_recreated(self, tmp_path):
        """No data to lose in an empty file; recreating it is the old behaviour."""
        emb = tmp_path / "session_embeddings.npy"
        emb.write_bytes(b"")
        store = _open(tmp_path, emb, 64)
        try:
            assert store.embeddings_available
            assert np.load(emb).shape == (1000, 64)
        finally:
            store.close()

    def test_close_without_embeddings_is_safe(self, tmp_path):
        """close() on a store that never loaded embeddings must not raise."""
        emb = tmp_path / "session_embeddings.npy"
        _write_npy(emb, rows=2, dim=32)
        store = _open(tmp_path, emb, 64)
        assert store._embeddings is None
        store.close()
        store.close()
