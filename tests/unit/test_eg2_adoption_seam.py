"""REPL-EMB-2.9 / 2.10 (B-4, B-5): the EG2 adoption code seam. EVERY FLAG IS OFF by default and the
default behaviour must be byte-identical; with the flag on the 768-d EG2 reply is sliced to 256 and
L2 re-normalised through ONE shared helper. Offline: no server, no model, no process."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from src import embedding_dims as dims
from src.embedding_pool.client import (
    EmbeddingUnavailable,
    PooledEmbeddingClient,
    normalise_vector,
)
from src.embedding_pool.policy import POLICY_PATH, EmbeddingPoolPolicy, load_policy, parse_policy
from src.embedding_pool.scheduler import EmbeddingScheduler
from tests.unit.test_embedding_pool_scheduler import make_topology

REPO = Path(__file__).resolve().parents[2]


# ── helper: flag-off equivalence ─────────────────────────────────────────────────────────


def legacy_normalise(embedding: np.ndarray) -> np.ndarray:
    """The pre-change inline code, verbatim, in all three raw-vector paths."""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


@pytest.mark.parametrize("n", [1024, 768, 8])
def test_flag_off_normalise_is_bitwise_the_legacy_step(n):
    rng = np.random.default_rng(n)
    raw = rng.standard_normal(n).astype(np.float32)
    got = dims.normalise_embedding(raw.copy(), None)
    want = legacy_normalise(raw.copy())
    assert got.dtype == want.dtype and got.tobytes() == want.tobytes()
    zero = np.zeros(n, dtype=np.float32)
    assert dims.normalise_embedding(zero.copy(), None).tobytes() == legacy_normalise(zero).tobytes()


def test_flag_on_is_slice_then_l2_and_reuses_the_frozen_harness_definition():
    rng = np.random.default_rng(1)
    raw = rng.standard_normal(768).astype(np.float32)
    got = dims.normalise_embedding(raw.copy(), 256)
    arms = dims._arms()
    assert got.shape == (256,) and abs(float(np.linalg.norm(got)) - 1.0) < 1e-6
    np.testing.assert_array_equal(got, np.asarray(arms.truncate_renormalise(raw, 256), np.float32))
    with pytest.raises(ValueError):
        dims.normalise_embedding(raw[:100], 256)


def test_default_truncate_dim_is_off_and_env_turns_it_on(monkeypatch):
    monkeypatch.delenv(dims.TRUNCATE_DIM_ENV, raising=False)
    assert dims.default_truncate_dim() is None
    assert dims.effective_embedding_dim() == dims.LEGACY_EMBEDDING_DIM == 1024
    monkeypatch.setenv(dims.TRUNCATE_DIM_ENV, "0")
    assert dims.default_truncate_dim() is None
    monkeypatch.setenv(dims.TRUNCATE_DIM_ENV, "256")
    assert dims.default_truncate_dim() == 256 == dims.EG2_SERVED_DIM
    assert dims.effective_embedding_dim() == 256
    monkeypatch.setenv(dims.TRUNCATE_DIM_ENV, "banana")
    with pytest.raises(ValueError):
        dims.default_truncate_dim()


# ── pooled client ────────────────────────────────────────────────────────────────────────


def test_normalise_vector_flag_off_signature_and_result_unchanged():
    raw = np.arange(1, 33, dtype=np.float32)
    a = normalise_vector(raw, dim=32, min_norm=1e-6)
    b = normalise_vector(raw, dim=32, min_norm=1e-6, truncate_dim=None)
    assert a.tobytes() == b.tobytes()


def test_normalise_vector_truncates_before_the_dimension_check():
    raw = np.random.default_rng(2).standard_normal(768).astype(np.float32)
    out = normalise_vector(raw, dim=256, min_norm=1e-6, truncate_dim=256)
    assert out.shape == (256,) and abs(float(np.linalg.norm(out)) - 1.0) < 1e-6
    with pytest.raises(Exception, match="dimension 768"):
        normalise_vector(raw, dim=256, min_norm=1e-6)  # flag off: a 768-d reply is degenerate


def _pool(handler, **client):
    base = dict(failure_backoff_s=60.0)
    base.update(client)
    pol = EmbeddingPoolPolicy().with_client(**base).with_cap(busy_sources=("slots",))
    topo = make_topology()
    transport = httpx.MockTransport(handler)
    return PooledEmbeddingClient(EmbeddingScheduler(topo, pol), policy=pol, transport=transport)


def _embed_reply(texts, dim):
    rng = np.random.default_rng(3)
    return [
        {"index": i, "embedding": [rng.standard_normal(dim).astype(float).tolist()]}
        for i in range(len(texts))
    ]


def test_client_slices_768_to_256_and_renormalises_when_on():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/slots"):
            return httpx.Response(200, json=[{"is_processing": False}])
        texts = json.loads(request.content)["content"]
        return httpx.Response(200, json=_embed_reply(texts, 768))

    client = _pool(handler, embedding_dim=256, truncate_dim=256)

    async def go():
        async with client:
            return await client.embed_many(["a", "b", "c"], use_cache=False)

    vecs = asyncio.run(go())
    assert vecs.shape == (3, 256)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_client_flag_off_rejects_a_768_reply_as_before():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/slots"):
            return httpx.Response(200, json=[{"is_processing": False}])
        texts = json.loads(request.content)["content"]
        return httpx.Response(200, json=_embed_reply(texts, 768))

    client = _pool(handler, embedding_dim=1024)

    async def go():
        async with client:
            return await client.embed_many(["a"], use_cache=False)

    with pytest.raises(EmbeddingUnavailable) as err:
        asyncio.run(go())
    assert err.value.reason == "degenerate"


# ── oversize policy ──────────────────────────────────────────────────────────────────────

OVERSIZE = {"error": {"type": "exceed_context_size_error", "message": "input is too large"}}


def _oversize_handler(limit_chars, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/slots"):
            return httpx.Response(200, json=[{"is_processing": False}])
        texts = json.loads(request.content)["content"]
        seen.append([len(t) for t in texts])
        if any(len(t) > limit_chars for t in texts):
            return httpx.Response(400, json=OVERSIZE)
        return httpx.Response(200, json=_embed_reply(texts, 32))

    return handler


def test_oversize_flag_on_clips_per_text_without_penalising_the_instance():
    seen: list[list[int]] = []
    client = _pool(_oversize_handler(100, seen), embedding_dim=32, oversize_retry_clipped=True)
    texts = ["x" * 10, "y" * 150, "z" * 10, "w" * 10]

    async def go():
        async with client:
            return await client.embed_many(texts, use_cache=False)

    vecs = asyncio.run(go())
    assert vecs.shape == (4, 32)
    counters = client.scheduler.stats()["counters"]
    assert counters["oversize_clipped"] == 1
    assert counters.get("retries", 0) == 0  # no instance fault
    # the instance that saw the oversize text is not in failure backoff
    assert client.scheduler.healthy_ports() == sorted(client.scheduler.topology.ports)


def test_oversize_that_never_fits_is_input_too_long_not_a_backoff():
    seen: list[list[int]] = []
    client = _pool(
        _oversize_handler(5, seen),
        embedding_dim=32,
        oversize_retry_clipped=True,
        oversize_max_halvings=2,
    )

    async def go():
        async with client:
            return await client.embed_many(["q" * 400], use_cache=False)

    with pytest.raises(EmbeddingUnavailable) as err:
        asyncio.run(go())
    assert err.value.reason == "input_too_long"
    assert client.scheduler.healthy_ports() == sorted(client.scheduler.topology.ports)


def test_oversize_flag_off_is_the_legacy_failure_backoff():
    seen: list[list[int]] = []
    client = _pool(_oversize_handler(100, seen), embedding_dim=32)

    async def go():
        async with client:
            return await client.embed_many(["y" * 150, "x"], use_cache=False)

    with pytest.raises(EmbeddingUnavailable):
        asyncio.run(go())
    assert client.scheduler.stats()["counters"].get("oversize_clipped", 0) == 0
    assert client.scheduler.stats()["counters"]["retries"] >= 1


# ── policy files ─────────────────────────────────────────────────────────────────────────


def test_live_policy_is_unchanged_off():
    p = load_policy()
    c = p.client
    assert (c.embedding_dim, c.truncate_dim, c.oversize_retry_clipped) == (1024, None, False)
    assert POLICY_PATH.name == "embedding_pool_policy.yaml"


def test_eg2_overlay_parses_and_agrees_with_the_live_file_on_every_other_key():
    overlay = REPO / "orchestration" / "embedding_pool_policy.eg2.yaml"
    p = load_policy(overlay)
    assert (p.client.embedding_dim, p.client.truncate_dim) == (256, 256)
    assert p.client.oversize_retry_clipped is True
    live = load_policy().as_dict()
    eg2 = p.as_dict()
    assert live["neighbour_cap"] == eg2["neighbour_cap"]
    differing = {k for k in live["client"] if live["client"][k] != eg2["client"][k]}
    assert differing == {"embedding_dim", "truncate_dim", "oversize_retry_clipped"}


def test_policy_rejects_a_truncation_that_disagrees_with_the_expected_dimension():
    doc = yaml.safe_load(POLICY_PATH.read_text())
    doc["client"]["truncate_dim"] = 256  # embedding_dim stays 1024
    with pytest.raises(ValueError, match="truncate_dim"):
        parse_policy(doc)
    doc["client"]["truncate_dim"] = -1
    with pytest.raises(ValueError, match="truncate_dim"):
        parse_policy(doc)


# ── raw-vector paths: TaskEmbedder, ParallelEmbedderClient ───────────────────────────────


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _Http:
    def __init__(self, vec):
        self.vec = vec
        self.posted = []

    def post(self, url, json):
        self.posted.append(json["content"])
        return _Resp({"embedding": [self.vec.tolist()]})


def _task_embedder(monkeypatch, vec, **cfg):
    from orchestration.repl_memory.embedder import EmbeddingConfig, TaskEmbedder

    monkeypatch.delenv(dims.TRUNCATE_DIM_ENV, raising=False)
    emb = TaskEmbedder(EmbeddingConfig(use_parallel=False, allow_subprocess=False, **cfg))
    http = _Http(vec)
    emb._http_client = http
    emb._server_available = True
    return emb, http


def test_task_embedder_http_path_flag_off_is_byte_identical(monkeypatch):
    raw = np.random.default_rng(5).standard_normal(1024).astype(np.float32)
    emb, http = _task_embedder(monkeypatch, raw)
    assert emb.config.truncate_dim is None and emb.embedding_dim == 1024
    out = emb.embed_text("hello")
    assert out.tobytes() == legacy_normalise(raw.astype(np.float32).copy()).tobytes()
    assert http.posted == ["hello"]  # no prefix, no mutation of the text
    # `side` is accepted and ignored with the seam off
    assert emb.embed_text("hello", side="query").tobytes() == out.tobytes()
    assert http.posted == ["hello", "hello"]
    with pytest.raises(ValueError):
        emb.embed_text("hello", side="bogus")


def test_task_embedder_flag_on_truncates_prefixes_and_demands_a_side(monkeypatch):
    raw = np.random.default_rng(6).standard_normal(768).astype(np.float32)
    emb, http = _task_embedder(monkeypatch, raw, truncate_dim=256)
    assert emb.embedding_dim == 256
    q = emb.embed_text("find me", side="query")
    d = emb.embed_text("stored text", side="document")
    assert q.shape == d.shape == (256,)
    assert http.posted == [
        "task: search result | query: find me",
        "title: none | text: stored text",
    ]
    with pytest.raises(dims.SideRequiredError):
        emb.embed_text("no side given")
    with pytest.raises(dims.SideRequiredError):
        emb.embed_task_ir({"objective": "x"})
    # every embed_* helper threads `side` through
    emb.embed_failure_context({"error": "e"}, side="query")
    emb.embed_exploration("q", "preview", side="query")
    emb.embed_classification_prompt("p", side="query")
    emb.embed_task_ir({"objective": "x"}, side="document")
    assert emb.embed_batch(["a", "b"], side="document").shape == (2, 256)


def test_retriever_callsite_uses_the_real_query_prefix(monkeypatch):
    from orchestration.repl_memory.retriever import TwoPhaseRetriever

    raw = np.random.default_rng(12).standard_normal(768).astype(np.float32)
    emb, http = _task_embedder(monkeypatch, raw, truncate_dim=256)
    retriever = TwoPhaseRetriever.__new__(TwoPhaseRetriever)
    retriever.embedder = emb
    retriever._retrieve = lambda vector, **_kwargs: [vector]

    result = retriever.retrieve_for_routing({"objective": "find a file"})

    assert len(result) == 1 and result[0].shape == (256,)
    assert len(http.posted) == 1
    assert http.posted[0].startswith("task: search result | query: ")


def test_generic_llm_embedding_tool_forwards_optional_side_and_search_roles(monkeypatch):
    from orchestration.repl_memory import embedder as embedder_mod
    from orchestration.tools import llm

    calls = []

    class FakeTaskEmbedder:
        def embed_text(self, text, *, side=None):
            calls.append((text, side))
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(embedder_mod, "TaskEmbedder", FakeTaskEmbedder)

    assert llm.embed_text("generic") == [1.0, 0.0]
    assert llm.embed_text("stored", side="document") == [1.0, 0.0]
    results = llm.similarity_search("lookup", ["candidate"], top_k=1)

    assert calls == [
        ("generic", None),
        ("stored", "document"),
        ("lookup", "query"),
        ("candidate", "document"),
    ]
    assert results[0]["item"] == "candidate"


def test_env_flag_makes_the_config_and_parallel_client_follow(monkeypatch):
    from orchestration.repl_memory.embedder import EmbeddingConfig
    from orchestration.repl_memory.parallel_embedder import EmbedderPoolConfig

    monkeypatch.setenv(dims.TRUNCATE_DIM_ENV, "256")
    assert EmbeddingConfig().embedding_dim == 256 and EmbedderPoolConfig().embedding_dim == 256
    monkeypatch.delenv(dims.TRUNCATE_DIM_ENV)
    assert EmbeddingConfig().embedding_dim == 1024 and EmbedderPoolConfig().embedding_dim == 1024


def _parallel_client(cfg, vec):
    from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

    client = ParallelEmbedderClient(cfg)
    posted = []

    class _AClient:
        async def post(self, url, json):
            posted.append(json["content"])
            return _Resp({"embedding": [vec.tolist()]})

    return client, _AClient(), posted


def test_parallel_client_flag_off_identical_and_flag_on_truncates(monkeypatch):
    from orchestration.repl_memory.parallel_embedder import EmbedderPoolConfig

    monkeypatch.delenv(dims.TRUNCATE_DIM_ENV, raising=False)
    raw = np.random.default_rng(8).standard_normal(1024).astype(np.float32)
    client, http, posted = _parallel_client(EmbedderPoolConfig(), raw)
    out = asyncio.run(client._embed_single_server(http, "http://x", "t"))
    assert out.tobytes() == legacy_normalise(raw.copy()).tobytes() and posted == ["t"]

    raw768 = np.random.default_rng(9).standard_normal(768).astype(np.float32)
    client, http, posted = _parallel_client(EmbedderPoolConfig(truncate_dim=256), raw768)
    out = asyncio.run(client._embed_single_server(http, "http://x", "t"))
    assert out.shape == (256,)
    assert client._with_side("t", "query") == "task: search result | query: t"
    with pytest.raises(dims.SideRequiredError):
        client._with_side("t", None)


# ── llama-embedding resolution + subprocess argv ─────────────────────────────────────────


def test_llama_embedding_resolves_only_through_the_kernel_store(tmp_path):
    from orchestration.repl_memory import embedder

    store = tmp_path / "kernels" / "production" / "cpu"
    assert embedder.resolve_embedding_binary(store) == store / "llama-embedding"
    assert not store.exists()  # Resolution stays safe on server-only hosts.
    assert "build/bin" not in str(embedder.KERNEL_STORE_CPU)


def test_subprocess_argv_flag_off_unchanged_flag_on_adds_mean_pooling(monkeypatch, tmp_path):
    import subprocess

    from orchestration.repl_memory.embedder import EmbeddingConfig, TaskEmbedder

    monkeypatch.delenv(dims.TRUNCATE_DIM_ENV, raising=False)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        vec = np.ones(768, dtype=np.float32).tolist()
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps([{"embedding": vec}]), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    from orchestration.repl_memory import embedder

    store = tmp_path / "store"
    store.mkdir()
    binary = store / "llama-embedding"
    binary.touch()
    binary.chmod(0o700)
    model = tmp_path / "fake.gguf"
    model.touch()
    monkeypatch.setattr(embedder, "KERNEL_STORE_CPU", store)
    config = dict(embedding_binary=binary, model_path=model, allow_subprocess=True)
    off = TaskEmbedder(EmbeddingConfig(**config))
    off._generate_embedding_llama("t")
    assert "--pooling" not in seen["cmd"]
    on = TaskEmbedder(EmbeddingConfig(**config, truncate_dim=256))
    out = on._generate_embedding_llama("t")
    assert seen["cmd"][seen["cmd"].index("--pooling") + 1] == "mean" and out.shape == (256,)
    assert seen["kw"]["env"]["LD_LIBRARY_PATH"] == str(store.resolve())


# ── launch recipe / affinity / gate ──────────────────────────────────────────────────────


def test_embedding_command_flag_off_has_no_new_argv_and_keys_are_emitted_when_set():
    from scripts.server import orchestrator_stack as stack
    from scripts.server.stack_manifest import EMBEDDER_PORTS, EMBEDDING_SERVER_RECIPES

    port = EMBEDDER_PORTS[0]
    live = stack._build_embedding_command(port)
    for flag in ("-b", "-ub", "--embd-normalize"):
        assert flag not in live
    assert not {"batch_tokens", "ubatch_tokens", "embd_normalize"} & set(
        EMBEDDING_SERVER_RECIPES[port]
    )
    original = EMBEDDING_SERVER_RECIPES[port]
    try:
        EMBEDDING_SERVER_RECIPES[port] = {
            **original,
            "context_tokens": 8192,
            "pooling": "mean",
            "batch_tokens": 2048,
            "ubatch_tokens": 2048,
            "embd_normalize": 2,
        }
        cmd = stack._build_embedding_command(port)
    finally:
        EMBEDDING_SERVER_RECIPES[port] = original
    assert cmd[cmd.index("-c") + 1] == "8192" and cmd[cmd.index("--pooling") + 1] == "mean"
    assert cmd[cmd.index("-b") + 1] == "2048" and cmd[cmd.index("-ub") + 1] == "2048"
    assert cmd[cmd.index("--embd-normalize") + 1] == "2"


def test_placement_gate_text_option_default_is_the_phase0_text():
    from scripts.server import embedder_placement_gate as gate

    assert gate.make_emb_text(None) is gate.EMB_TEXT
    for n in (200, len(gate.EMB_TEXT) * 3 + 7):
        assert len(gate.make_emb_text(n)) == n
    with pytest.raises(ValueError):
        gate.make_emb_text(0)
    assert gate._Load([8090], 1).text is gate.EMB_TEXT
    assert gate._Load([8090], 1, text="zz").text == "zz"
    # raw load factory without a text/pace is still the bare class (Phase-0 call shape)
    assert gate._load_for("raw") is gate._Load


# ── store consumers: B-4 ─────────────────────────────────────────────────────────────────


def _marker(d: Path, dim: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / dims.MARKER_NAME).write_text(json.dumps({"dim": dim, "model": "eg2"}))


def test_check_store_marker_contract(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    dims.check_store_marker(d, 1024)  # no marker: nothing to contradict
    dims.check_store_marker(d, 8)
    _marker(d, 256)
    dims.check_store_marker(d, 256)
    with pytest.raises(dims.StoreDimensionMismatch, match="reembed_stores_eg2.py"):
        dims.check_store_marker(d, 1024, consumer="X")
    assert dims.marker_dim(tmp_path / "nowhere") is None


def test_faiss_store_refuses_a_marked_store_at_the_wrong_dim_and_touches_nothing(tmp_path):
    pytest.importorskip("faiss")
    from orchestration.repl_memory.faiss_store import FAISSEmbeddingStore, NumpyEmbeddingStore

    d = tmp_path / "f"
    _marker(d, 256)
    before = sorted(p.name for p in d.iterdir())
    with pytest.raises(dims.StoreDimensionMismatch):
        FAISSEmbeddingStore(path=d, dim=1024)
    with pytest.raises(dims.StoreDimensionMismatch):
        NumpyEmbeddingStore(path=d, dim=1024)
    assert sorted(p.name for p in d.iterdir()) == before  # nothing created
    FAISSEmbeddingStore(path=d, dim=256)  # the matching config opens
    # and the reverse: a 1024-d marker refuses a 256-d consumer
    e = tmp_path / "g"
    _marker(e, 1024)
    with pytest.raises(dims.StoreDimensionMismatch):
        FAISSEmbeddingStore(path=e, dim=256)


def test_unmarked_legacy_store_opens_exactly_as_before(tmp_path):
    pytest.importorskip("faiss")
    from orchestration.repl_memory.faiss_store import FAISSEmbeddingStore

    store = FAISSEmbeddingStore(path=tmp_path / "legacy", dim=16)
    assert store.dim == 16


def test_session_store_refusal_names_the_migration_and_keeps_the_file(tmp_path):
    from src.session import SQLiteSessionStore

    d = tmp_path / "sess"
    _marker(d, 256)
    npy = d / "session_embeddings.npy"
    np.save(npy, np.zeros((4, 256), dtype=np.float32))
    h = npy.read_bytes()
    store = SQLiteSessionStore(db_path=d / "sessions.db", embeddings_path=npy)  # default 1024
    assert not store.embeddings_available
    assert "reembed_stores_eg2" in str(store.embeddings_refusal)
    assert npy.read_bytes() == h
    store.close()


def test_default_embedding_binary_is_kernel_store_path(monkeypatch):
    from orchestration.repl_memory import embedder

    monkeypatch.setenv("EPYC_EMBED_USE_KERNEL_STORE", "0")
    binary = embedder.KERNEL_STORE_CPU / "llama-embedding"
    assert embedder.resolve_embedding_binary() == binary
    assert embedder.DEFAULT_EMBEDDING_BINARY == binary
    assert embedder.EmbeddingConfig().embedding_binary == binary
    assert embedder.EmbeddingConfig().allow_subprocess is False
