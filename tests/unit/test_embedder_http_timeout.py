"""Regression for long valid embeddings; constructing HTTP clients sends no requests."""

import pytest

from orchestration.repl_memory.embedder import EmbeddingConfig, TaskEmbedder


@pytest.mark.parametrize("configured,expected", [(None, 30.0), (47, 47.0)])
def test_single_http_client_honors_embedding_timeout(configured, expected):
    args = {} if configured is None else {"timeout": configured}
    embedder = TaskEmbedder(
        EmbeddingConfig(use_parallel=False, use_fallback=False, allow_subprocess=False, **args)
    )
    client = embedder._get_http_client()
    try:
        assert client.timeout.connect == expected
        assert client.timeout.read == expected
        assert client.timeout.write == expected
        assert client.timeout.pool == expected
    finally:
        client.close()
