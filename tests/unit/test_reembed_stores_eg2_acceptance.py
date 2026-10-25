"""Offline migration acceptance: no live stores, model, endpoint, or deployment changes."""

from __future__ import annotations

from contextlib import nullcontext
import json

import numpy as np
import pytest

from tests.unit.test_reembed_stores_eg2 import (
    FakeEG2,
    apply_args,
    cli,
    eg2 as eg2,
    layout as layout,
    require_faiss as require_faiss,
    template as template,
    tool,
)


@pytest.mark.parametrize("indices", [[0, 0], [1, 2], [0, None], [True, 0], [0, "1"]])
def test_response_indices_cannot_silently_reassign_rows(indices):
    client = tool.EmbedClient("http://unused.invalid", tool.target_spec())
    rows = [{"embedding": FakeEG2.base_vec(str(i)).tolist()} for i in range(2)]
    for row, index in zip(rows, indices):
        if index is not None:
            row["index"] = index
    with pytest.raises(tool.EmbedError, match="complete, unique permutation"):
        client._raw_matrix({"data": rows}, 2)


def test_reversed_response_indices_restore_input_order():
    client = tool.EmbedClient("http://unused.invalid", tool.target_spec())
    vectors = [FakeEG2.base_vec(str(i)) for i in range(2)]
    rows = [{"index": i, "embedding": vectors[i].tolist()} for i in [1, 0]]
    np.testing.assert_array_equal(client._raw_matrix({"data": rows}, 2), vectors)


@pytest.mark.parametrize("key,bad", [("model_file", "wrong.gguf"), ("gguf_sha256", "0" * 64)])
def test_plan_refuses_conflicting_gguf_identity(layout, capsys, eg2, key, bad):
    assert cli(capsys, *apply_args(layout, eg2, "identity"))[0] == 0
    path = layout.ep / tool.MARKER_NAME
    marker = json.loads(path.read_text())
    marker[key] = bad
    path.write_text(json.dumps(marker))
    before = layout.snapshot()
    rc, out, _ = cli(capsys, "plan", "--episodic-dir", str(layout.ep), "--json")
    assert rc == 3 and "different embedding identity" in out
    assert layout.snapshot() == before


@pytest.mark.parametrize("cosine,expected_rc", [(0.99995, 4), (0.999995, 0)])
def test_semantic_acceptance_uses_owner_threshold(
    layout, capsys, eg2, monkeypatch, cosine, expected_rc
):
    original = tool.EmbedClient.embed_documents
    seen = set()

    def drift_on_validation(client, texts):
        result = original(client, texts)
        for i, text in enumerate(texts):
            if text in seen and "probe" not in text:
                v = result[i].copy()
                orthogonal = np.roll(v, 1)
                orthogonal -= np.dot(orthogonal, v) * v
                orthogonal /= np.linalg.norm(orthogonal)
                result[i] = cosine * v + np.sqrt(1 - cosine**2) * orthogonal
            seen.add(text)
        return result

    monkeypatch.setattr(tool.EmbedClient, "embed_documents", drift_on_validation)
    before = layout.snapshot()
    rc, _, err = cli(capsys, *apply_args(layout, eg2, "threshold"))
    assert rc == expected_rc, err
    if expected_rc:
        assert "0.99999" in err and layout.snapshot() == before


def test_document_vectors_match_production_client_path_without_a_model(monkeypatch):
    from orchestration.repl_memory import embedder as production

    # Capture the real production side-prefix and HTTP response handling, without acquiring
    # an inference lock or making a request. The same deterministic fake vectors feed both.
    monkeypatch.setattr(production, "inference_lock", lambda *a, **kw: nullcontext())
    monkeypatch.delenv("EPYC_EMBEDDING_TRUNCATE_DIM", raising=False)
    sent = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": [FakeEG2.base_vec(sent[-1]).tolist()]}

    class Transport:
        def post(self, url, json):
            sent.append(json["content"])
            return Response()

    embedder = production.TaskEmbedder(
        production.EmbeddingConfig(
            use_parallel=False,
            allow_subprocess=False,
            use_fallback=False,
            truncate_dim=256,
        )
    )
    embedder._http_client = Transport()
    embedder._server_available = True
    client = tool.EmbedClient("http://unused.invalid", tool.target_spec())
    monkeypatch.setattr(
        client,
        "_post",
        lambda texts: {
            "data": [
                {"index": i, "embedding": FakeEG2.base_vec(text).tolist()}
                for i, text in enumerate(texts)
            ]
        },
    )
    texts = [
        "type:code | objective:fix caching | priority:normal",
        "strategy title description insight",
    ]
    migrated = client.embed_documents(texts)
    for i, text in enumerate(texts):
        doc = embedder.embed_text(text, side="document")
        assert float(np.dot(doc, migrated[i])) >= 0.99999
        assert sent[-1] == tool.target_spec().doc_prompt + text
        query = embedder.embed_text(text, side="query")
        assert sent[-1] != tool.target_spec().doc_prompt + text
        assert not np.allclose(query, doc)
