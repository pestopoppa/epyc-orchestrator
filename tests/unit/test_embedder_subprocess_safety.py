"""Embedding outages must not load models; explicit subprocess use is store-only."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration.repl_memory import embedder


@pytest.fixture(autouse=True)
def no_live_boundaries(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("test crossed a live embedding boundary")

    monkeypatch.delenv("EPYC_EMBEDDING_TRUNCATE_DIM", raising=False)
    monkeypatch.setattr(embedder, "inference_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(embedder.subprocess, "run", forbidden)
    monkeypatch.setattr(embedder.TaskEmbedder, "_get_http_client", forbidden)
    monkeypatch.setattr(embedder.TaskEmbedder, "_get_parallel_client", forbidden)
    monkeypatch.setattr(embedder.TaskEmbedder, "_check_server", lambda self: False)
    monkeypatch.setattr(embedder.TaskEmbedder, "_check_model", lambda self: True)


@pytest.fixture
def fake_store(monkeypatch, tmp_path):
    physical = tmp_path / "physical" / "bin"
    physical.mkdir(parents=True)
    binary = physical / "llama-embedding"
    binary.touch()
    binary.chmod(0o700)
    store = tmp_path / "cpu"
    store.symlink_to(physical, target_is_directory=True)
    model = tmp_path / "fake-model.gguf"
    model.touch()
    monkeypatch.setattr(embedder, "KERNEL_STORE_CPU", store)
    return store, binary, model


def configured(fake_store, **kwargs):
    store, _, model = fake_store
    return embedder.TaskEmbedder(
        embedder.EmbeddingConfig(
            embedding_binary=store / "llama-embedding",
            model_path=model,
            use_parallel=False,
            **kwargs,
        )
    )


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("request_fails", [False, True])
def test_default_outage_never_spawns_model(fake_store, monkeypatch, fallback, request_fails):
    task = configured(fake_store, use_fallback=fallback)
    assert task._model_available  # Model/binary availability cannot authorize a process.
    if request_fails:
        monkeypatch.setattr(task, "_check_server", lambda: True)

        def fail_http(text):
            raise RuntimeError("fake request failed")

        monkeypatch.setattr(task, "_generate_embedding_http", fail_http)
    sentinel = np.ones(1024, dtype=np.float32)
    monkeypatch.setattr(task, "_generate_embedding_fallback", lambda text: sentinel)
    if fallback:
        assert task.embed_text("outage") is sentinel
    else:
        with pytest.raises(RuntimeError, match="subprocess and hash fallback disabled"):
            task.embed_text("outage")


def test_server_success_does_not_require_store(fake_store, monkeypatch):
    task = configured(fake_store, use_fallback=False)
    fake_store[1].unlink()
    sentinel = np.ones(1024, dtype=np.float32)
    monkeypatch.setattr(task, "_check_server", lambda: True)
    monkeypatch.setattr(task, "_generate_embedding_http", lambda text: sentinel)
    assert task.embed_text("served") is sentinel


@pytest.mark.parametrize("failure", ["binary", "model", "nonexecutable", "outside"])
def test_invalid_explicit_opt_in_raises_before_hash(fake_store, monkeypatch, tmp_path, failure):
    task = configured(fake_store, allow_subprocess=True, use_fallback=True)
    if failure == "binary":
        fake_store[1].unlink()
    elif failure == "model":
        fake_store[2].unlink()
    elif failure == "nonexecutable":
        fake_store[1].chmod(0o600)
    else:
        outside = tmp_path / "llama.cpp" / "build" / "bin" / "llama-embedding"
        outside.parent.mkdir(parents=True)
        outside.touch()
        task.config.embedding_binary = outside

    def no_hash(text):
        pytest.fail("invalid subprocess admission was hidden by hash fallback")

    monkeypatch.setattr(task, "_generate_embedding_fallback", no_hash)
    with pytest.raises(RuntimeError, match="kernel.store|model missing"):
        task.embed_text("explicit")


def test_direct_subprocess_method_requires_opt_in(fake_store):
    task = configured(fake_store)
    with pytest.raises(RuntimeError, match="subprocess disabled"):
        task._generate_embedding_llama("direct")


def test_opt_in_uses_store_binary_and_matching_libraries(fake_store, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout='[{"embedding": [3, 4]}]', stderr="")

    monkeypatch.setattr(embedder.subprocess, "run", fake_run)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/wrong/ggml")
    task = configured(fake_store, allow_subprocess=True, use_fallback=False)
    out = task.embed_text("explicit")
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[0] == str(fake_store[1])
    assert cmd[cmd.index("-m") + 1] == str(fake_store[2])
    assert kwargs["env"]["LD_LIBRARY_PATH"] == str(fake_store[1].parent)
    assert np.allclose(out, [0.6, 0.8])


def test_parallel_outage_cannot_authorize_subprocess(fake_store, monkeypatch):
    task = configured(fake_store, use_fallback=False)
    task.config.use_parallel = True

    def unavailable():
        raise RuntimeError("fake pool unavailable")

    monkeypatch.setattr(task, "_get_parallel_client", unavailable)
    with pytest.raises(RuntimeError, match="subprocess and hash fallback disabled"):
        task.embed_text("parallel outage")


@pytest.mark.parametrize("fallback", [False, True])
def test_admitted_subprocess_failure_keeps_existing_hash_policy(fake_store, monkeypatch, fallback):
    monkeypatch.setattr(
        embedder.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="fake failure"),
    )
    task = configured(fake_store, allow_subprocess=True, use_fallback=fallback)
    sentinel = np.ones(1024, dtype=np.float32)
    monkeypatch.setattr(task, "_generate_embedding_fallback", lambda text: sentinel)
    if fallback:
        assert task.embed_text("admitted failure") is sentinel
    else:
        with pytest.raises(RuntimeError, match="llama-embedding failed: fake failure"):
            task.embed_text("admitted failure")
