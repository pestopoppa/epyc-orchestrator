"""STACKCHG-EMBEDPLACE-20261118: the GPU pool member (embedder_gpu :8099).

Offline. Covers: the member is DERIVED from its aux service (slots from -np, model from -m),
the pool refuses an incoherent declaration, a GPU member is never capped by a busy neighbour,
`prefer_device: gpu` sends equal-tier slots to it first and the CPU pool is the overflow and the
fail-open, and `none` restores the legacy order.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.embedding_pool.policy import ClientPolicy, EmbeddingPoolPolicy, parse_policy
from src.embedding_pool.scheduler import EmbeddingScheduler
from src.embedding_pool.topology import (
    GPU_NODE,
    EmbedderInstance,
    GuardedInstance,
    PoolTopology,
    gpu_members_from_declarations,
    live_topology,
)

MODEL = "/m/bge-large-en-v1.5-f16.gguf"


def aux(**over):
    base = dict(
        port=8099,
        backend="gpu",
        cpuset="184-191",
        argv=("llama-server", "-m", MODEL, "--embeddings", "-np", "4", "--port", "8099"),
    )
    base.update(over)
    return SimpleNamespace(**base)


def topo(gpu=True):
    cpu = [
        EmbedderInstance(port=p, url=f"http://127.0.0.1:{p}", numa_node=n, slots=4, model_id="bge-large-en-v1.5-f16.gguf")
        for p, n in ((8090, 1), (8091, 1), (8094, 3))
    ]
    members = gpu_members_from_declarations({8099: "embedder_gpu"}, {"embedder_gpu": aux()}) if gpu else []
    guarded = [
        GuardedInstance(role="frontdoor", instance_idx=0, port=8070, url="http://127.0.0.1:8070", numa_nodes=frozenset({0, 1, 2, 3}))
    ]
    return PoolTopology.build(cpu + members, guarded)


def sched(prefer, busy=True):
    from src.embedding_pool.busy import BusySnapshot, CompositeBusySource, StaticBusySource

    policy = EmbeddingPoolPolicy(client=ClientPolicy(prefer_device=prefer))
    b = (8070,) if busy else ()
    s = EmbeddingScheduler(topo(), policy, busy_source=CompositeBusySource([StaticBusySource(b)]))
    s.set_busy_snapshot(BusySnapshot(busy=frozenset(b), unknown=frozenset(), taken_at=1e18))
    return s


def test_member_is_derived_from_its_aux_service():
    (m,) = gpu_members_from_declarations({8099: "embedder_gpu"}, {"embedder_gpu": aux()})
    assert (m.port, m.slots, m.device, m.numa_node) == (8099, 4, "gpu", GPU_NODE)
    assert m.model_id == "bge-large-en-v1.5-f16.gguf"


def test_incoherent_declaration_lists_every_problem():
    bad = aux(port=9, backend="cpu", argv=("llama-server",))
    with pytest.raises(ValueError) as e:
        gpu_members_from_declarations({8099: "embedder_gpu", 8100: "nope"}, {"embedder_gpu": bad})
    msg = str(e.value)
    for needle in ("listens on :9", "not 'gpu'", "--embeddings", "-np", "-m", "'nope'"):
        assert needle in msg


def test_pool_refuses_a_different_model():
    other = aux(argv=("llama-server", "-m", "/m/other.gguf", "--embeddings", "-np", "4"))
    members = gpu_members_from_declarations({8099: "embedder_gpu"}, {"embedder_gpu": other})
    cpu = [EmbedderInstance(port=8090, url="u", numa_node=1, slots=4, model_id="bge-large-en-v1.5-f16.gguf")]
    with pytest.raises(ValueError, match="ONE embedding model"):
        PoolTopology.build(cpu + members, [])


def test_gpu_member_has_no_neighbour_and_is_never_capped():
    t = topo()
    assert t.neighbours(8099) == ()
    s = sched("gpu", busy=True)
    grants = s.try_reserve(4)
    assert [(g.port, g.n, g.capped) for g in grants] == [(8099, 4, False)]


def test_prefer_gpu_then_overflow_to_cpu():
    s = sched("gpu", busy=False)
    ports = {g.port: g.n for g in s.try_reserve(6)}
    assert ports[8099] == 4 and sum(ports.values()) == 6


def test_none_restores_legacy_order():
    s = sched("none", busy=False)
    ports = {g.port for g in s.try_reserve(1)}
    assert ports == {8090}


def test_failed_gpu_member_fails_open_to_cpu():
    s = sched("gpu", busy=False)
    s.mark_failure(8099)
    ports = {g.port for g in s.try_reserve(2)}
    assert 8099 not in ports and ports


def test_policy_rejects_bad_prefer_device():
    with pytest.raises(ValueError, match="prefer_device"):
        parse_policy({"version": 1, "client": {"prefer_device": "tpu"}})


def test_live_manifest_declares_the_gpu_member():
    t = live_topology()
    m = t.embedder(8099)
    assert m.device == "gpu" and m.slots == 4 and t.model_id == "bge-large-en-v1.5-f16.gguf"
    assert {e.port for e in t.embedders if e.device == "cpu"} == {8090, 8091, 8092, 8093, 8094, 8095}


def _embed_with_gpu_member_absent(prefer: str):
    """End to end through the real client: the LIVE manifest topology (7 members, :8099 included),
    the real policy file with ``prefer_device`` overridden, and a transport on which nothing
    listens at :8099 (connection refused) while the six CPU ports answer. Returns (vectors, ports
    that got a connection attempt, ports that served an /embedding request)."""
    import asyncio
    import dataclasses

    import httpx

    from src.embedding_pool.client import PooledEmbeddingClient
    from src.embedding_pool.fake import FakeEmbeddingServers
    from src.embedding_pool.policy import load_policy

    base = load_policy()
    policy = dataclasses.replace(base, client=dataclasses.replace(base.client, prefer_device=prefer))
    topology = live_topology()
    cpu_ports = {e.port for e in topology.embedders if e.device == "cpu"}
    servers = FakeEmbeddingServers(cpu_ports, dim=1024)  # bge-large is 1024-d; the policy pins it
    attempts: list[int] = []

    async def handler(request):
        port = request.url.port
        attempts.append(port)
        if port == 8099:  # embedder_gpu is NOT running
            raise httpx.ConnectError("All connection attempts failed", request=request)
        return await servers.handle(request)

    async def go():
        sched = EmbeddingScheduler(topology, policy)
        async with PooledEmbeddingClient(sched, transport=httpx.MockTransport(handler)) as client:
            return await client.embed_many(["tool picker query one", "tool picker query two"])

    vecs = asyncio.run(go())
    return vecs, attempts, {p for p, _ in servers.requests}


def test_gpu_preferred_but_absent_falls_back_to_the_cpu_pool():
    """Safe-while-down proof: prefer_device=gpu with :8099 refusing connections still embeds,
    served by a CPU pool member (the unchanged fail-open), after the GPU member was tried first."""
    vecs, attempts, served = _embed_with_gpu_member_absent("gpu")
    assert len(vecs) == 2
    assert 8099 in attempts, "gpu is preferred, so the absent member must have been tried first"
    assert served and 8099 not in served and served <= {8090, 8091, 8092, 8093, 8094, 8095}


def test_none_never_touches_the_gpu_member():
    vecs, attempts, served = _embed_with_gpu_member_absent("none")
    assert len(vecs) == 2 and 8099 not in attempts and served
