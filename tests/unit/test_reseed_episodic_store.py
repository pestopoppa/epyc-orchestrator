from __future__ import annotations

import inspect

import numpy as np
import pytest

from scripts.maintenance import reseed_episodic_store as reseed


class _Embedder:
    def __init__(self, values: np.ndarray):
        self.values = values

    def embed_batch(self, _texts: list[str]) -> np.ndarray:
        return self.values


def test_checked_batch_normalizes_and_rejects_invalid_vectors() -> None:
    values = np.ones((2, 1024), dtype=np.float32)
    result = reseed._checked_batch(_Embedder(values), ["a", "b"])
    assert result.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    with pytest.raises(reseed.ReseedVerificationError, match="zero vector"):
        reseed._checked_batch(_Embedder(np.zeros((1, 1024), dtype=np.float32)), ["a"])
    with pytest.raises(reseed.ReseedVerificationError, match="shape"):
        reseed._checked_batch(_Embedder(np.ones((1, 3), dtype=np.float32)), ["a"])


def test_strict_embedder_disables_every_semantic_fallback() -> None:
    embedder = reseed._strict_embedder()
    try:
        assert embedder.config.use_fallback is False
        assert embedder.config.allow_subprocess is False
        assert embedder._parallel_client.config.use_fallback is False
    finally:
        embedder.close()


def test_strict_embedder_fails_when_every_server_path_fails(monkeypatch) -> None:
    embedder = reseed._strict_embedder()
    monkeypatch.setattr(
        embedder._parallel_client,
        "embed_sync",
        lambda _text: (_ for _ in ()).throw(RuntimeError("down")),
    )
    monkeypatch.setattr(embedder, "_check_server", lambda: False)
    try:
        with pytest.raises(RuntimeError, match="fallback disabled"):
            embedder.embed_text("must not become a hash embedding")
    finally:
        embedder.close()


def test_faiss_publication_precedes_sqlite_commit() -> None:
    source = inspect.getsource(reseed.reseed)
    assert source.index('ix_tmp.replace(sessions / "embeddings.faiss")') < source.index(
        "con.commit()"
    )
    assert source.index("con.commit()") < source.index("verify_persisted")


def test_apply_rejects_partial_limit_before_any_embedding(tmp_path) -> None:
    with pytest.raises(ValueError, match="--limit is unsafe"):
        reseed.reseed(tmp_path, apply=True, limit=1)
