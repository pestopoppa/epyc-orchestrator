"""Pool topology: embedder instances, the serving instances they sit next to, and who shares
which NUMA node.

Placement is DATA. Every fact here is derived from the stack's own declarations:

* embedder port -> cpuset -> NUMA node: ``stack_manifest.EMBEDDING_PLACEMENT`` (from
  ``launch_manifest.yaml`` ``embedding.placement``; the node is derived from the cpuset);
* embedder port -> slots (``-np``) and model path: ``stack_manifest.EMBEDDING_SERVER_RECIPES``;
* guarded serving instance -> port -> NUMA nodes: ``stack_numa.NUMA_CONFIG`` (from
  ``stack_topology.yaml``), nodes derived from each instance's cpuset.

No port, cpuset or node literal lives in this module.

Invariant 4 of the handoff: an index belongs to ONE embedding model, so a pool refuses to be
built over instances that serve different models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_HOST = "127.0.0.1"


@dataclass(frozen=True)
class EmbedderInstance:
    port: int
    url: str
    numa_node: int
    slots: int
    model_id: str
    cpuset: str = ""
    # STACKCHG-EMBEDPLACE-20261118: "cpu" (a pinned CPU port) or "gpu" (an aux service on the
    # MI210). A GPU member has numa_node GPU_NODE (-1): it belongs to no NPS4 node, so it has no
    # guarded neighbour and the neighbour cap never applies to it.
    device: str = "cpu"


GPU_NODE = -1


@dataclass(frozen=True)
class GuardedInstance:
    """A serving instance the pool must not slow down (e.g. a frontdoor full or half)."""

    role: str
    instance_idx: int
    port: int
    url: str
    numa_nodes: frozenset[int]


@dataclass(frozen=True)
class PoolTopology:
    embedders: tuple[EmbedderInstance, ...]
    guarded: tuple[GuardedInstance, ...]
    model_id: str

    @classmethod
    def build(
        cls, embedders: Iterable[EmbedderInstance], guarded: Iterable[GuardedInstance] = ()
    ) -> "PoolTopology":
        emb = tuple(sorted(embedders, key=lambda e: e.port))
        if not emb:
            raise ValueError("embedding pool: no embedder instances")
        models = sorted({e.model_id for e in emb})
        if len(models) != 1:
            raise ValueError(
                f"embedding pool: instances serve {len(models)} models {models}; an index belongs "
                "to ONE embedding model, so a pool may only span instances of one model"
            )
        ports = [e.port for e in emb]
        if len(set(ports)) != len(ports):
            raise ValueError(f"embedding pool: duplicate embedder ports {ports}")
        for e in emb:
            if e.slots <= 0:
                raise ValueError(f"embedding pool: :{e.port} declares {e.slots} slots")
        return cls(embedders=emb, guarded=tuple(guarded), model_id=models[0])

    # -- derived views --------------------------------------------------------------------
    @property
    def ports(self) -> tuple[int, ...]:
        return tuple(e.port for e in self.embedders)

    def embedder(self, port: int) -> EmbedderInstance:
        for e in self.embedders:
            if e.port == port:
                return e
        raise KeyError(port)

    def neighbours(self, embedder_port: int) -> tuple[GuardedInstance, ...]:
        """Guarded serving instances that occupy the embedder's NUMA node."""
        node = self.embedder(embedder_port).numa_node
        return tuple(g for g in self.guarded if node in g.numa_nodes)

    def guarded_by_port(self, port: int) -> GuardedInstance | None:
        for g in self.guarded:
            if g.port == port:
                return g
        return None

    def nodes_of_port(self, port: int | None) -> frozenset[int]:
        """NUMA nodes of a serving port (the requester's hardware); empty if unknown."""
        if port is None:
            return frozenset()
        g = self.guarded_by_port(port)
        if g is not None:
            return g.numa_nodes
        for e in self.embedders:
            if e.port == port:
                return frozenset({e.numa_node})
        return frozenset()

    def restricted_to(self, ports: Sequence[int]) -> "PoolTopology":
        wanted = set(int(p) for p in ports)
        missing = wanted - set(self.ports)
        if missing:
            raise ValueError(f"embedding pool: ports {sorted(missing)} are not pool instances")
        return PoolTopology.build([e for e in self.embedders if e.port in wanted], self.guarded)

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "embedders": [
                {
                    "port": e.port,
                    "numa_node": e.numa_node,
                    "slots": e.slots,
                    "cpuset": e.cpuset,
                    "device": e.device,
                    "neighbours": [g.port for g in self.neighbours(e.port)],
                }
                for e in self.embedders
            ],
            "guarded": [
                {
                    "role": g.role,
                    "instance_idx": g.instance_idx,
                    "port": g.port,
                    "numa_nodes": sorted(g.numa_nodes),
                }
                for g in self.guarded
            ],
        }


def _url(host: str, port: int) -> str:
    return f"http://{host}:{int(port)}"


def _argv_value(argv: Sequence[str], flag: str) -> str | None:
    for i, tok in enumerate(argv[:-1]):
        if tok == flag:
            return str(argv[i + 1])
    return None


def gpu_members_from_declarations(
    declared: Mapping[object, object] | None,
    aux_services: Mapping[str, object],
    *,
    host: str = DEFAULT_HOST,
) -> list[EmbedderInstance]:
    """GPU pool members: ``launch_manifest embedding.gpu_members`` (port -> aux service name).

    Slots and model are RECOMPUTED from the aux service's own argv (``-np``, ``-m``), never
    restated; every violation is raised at once. A member must be a ``backend: gpu`` aux
    service listening on the declared port with ``--embeddings`` and a ``-np``.
    """
    out: list[EmbedderInstance] = []
    problems: list[str] = []
    for raw_port, name in (declared or {}).items():
        port = int(raw_port)
        svc = aux_services.get(str(name))
        if svc is None:
            problems.append(f":{port} names aux service {name!r}, which is not declared")
            continue
        argv = list(getattr(svc, "argv", ()))
        np_raw, model = _argv_value(argv, "-np"), _argv_value(argv, "-m")
        local: list[str] = []
        if getattr(svc, "port", None) != port:
            local.append(f":{port} aux {name!r} listens on :{getattr(svc, 'port', None)}")
        if getattr(svc, "backend", None) != "gpu":
            local.append(f":{port} aux {name!r} has backend {getattr(svc, 'backend', None)!r}, not 'gpu'")
        if "--embeddings" not in argv:
            local.append(f":{port} aux {name!r} argv has no --embeddings")
        if not (np_raw or "").isdigit() or int(np_raw or 0) <= 0:
            local.append(f":{port} aux {name!r} argv declares no positive -np")
        if not model:
            local.append(f":{port} aux {name!r} argv declares no -m")
        problems.extend(local)
        if local:
            continue
        out.append(
            EmbedderInstance(
                port=port,
                url=_url(host, port),
                numa_node=GPU_NODE,
                slots=int(np_raw or 0),
                model_id=Path(model or "").name,
                cpuset=str(getattr(svc, "cpuset", "") or ""),
                device="gpu",
            )
        )
    if problems:
        raise ValueError("embedding pool: gpu_members incoherent:\n  " + "\n  ".join(problems))
    return out


def topology_from_declarations(
    *,
    placement: Mapping[int, tuple[str, int]],
    recipes: Mapping[int, Mapping[str, object]],
    pool_ports: Sequence[int],
    numa_config: Mapping[str, Mapping[str, object]],
    nodes_of_cpuset,
    guarded_roles: Sequence[str],
    host: str = DEFAULT_HOST,
    gpu_members: Sequence[EmbedderInstance] = (),
) -> PoolTopology:
    """Pure builder. ``placement`` maps port -> (cpuset, numa_node); ``numa_config`` has the
    ``stack_numa.NUMA_CONFIG`` shape; ``nodes_of_cpuset(spec) -> list[int]``. ``gpu_members``
    are appended to the CPU pool (same-model rule enforced by ``PoolTopology.build``)."""
    embedders = []
    for port in pool_ports:
        port = int(port)
        if port not in placement:
            raise ValueError(f"embedding pool: :{port} has no declared placement")
        cpuset, node = placement[port]
        recipe = recipes.get(port) or {}
        slots = recipe.get("slots")
        if not isinstance(slots, int) or isinstance(slots, bool) or slots <= 0:
            raise ValueError(f"embedding pool: :{port} recipe declares no positive `slots`")
        model = str(recipe.get("model_path") or recipe.get("model_name") or "")
        if not model:
            raise ValueError(f"embedding pool: :{port} recipe declares no model")
        embedders.append(
            EmbedderInstance(
                port=port,
                url=_url(host, port),
                numa_node=int(node),
                slots=slots,
                model_id=Path(model).name,
                cpuset=str(cpuset),
            )
        )
    guarded = []
    for role in guarded_roles:
        cfg = numa_config.get(role)
        if not cfg:
            raise ValueError(f"embedding pool: guarded role {role!r} is not in NUMA_CONFIG")
        for idx, inst in enumerate(cfg.get("instances") or []):
            cpus, port, _threads = inst
            guarded.append(
                GuardedInstance(
                    role=role,
                    instance_idx=idx,
                    port=int(port),
                    url=_url(host, port),
                    numa_nodes=frozenset(int(n) for n in nodes_of_cpuset(cpus)),
                )
            )
    embedders.extend(gpu_members)
    return PoolTopology.build(embedders, guarded)


def live_topology(
    guarded_roles: Sequence[str] = ("frontdoor",), host: str = DEFAULT_HOST
) -> PoolTopology:
    """The pool as the stack declares it (no network; reads the manifests at import)."""
    from scripts.server.stack_manifest import (  # type: ignore[import-not-found]
        AUX_SERVICES,
        EMBEDDER_PORTS,
        EMBEDDING_PLACEMENT,
        EMBEDDING_SERVER_RECIPES,
        _EMBEDDING,
    )
    from scripts.server.stack_numa import NUMA_CONFIG, _nodes_touched  # type: ignore[import-not-found]

    gpu_members = gpu_members_from_declarations(_EMBEDDING.get("gpu_members"), AUX_SERVICES, host=host)

    return topology_from_declarations(
        placement={p: (pl.cpuset, pl.numa_node) for p, pl in EMBEDDING_PLACEMENT.items()},
        recipes=EMBEDDING_SERVER_RECIPES,
        pool_ports=EMBEDDER_PORTS,
        numa_config=NUMA_CONFIG,
        nodes_of_cpuset=_nodes_touched,
        guarded_roles=guarded_roles,
        host=host,
        gpu_members=gpu_members,
    )


__all__ = [
    "DEFAULT_HOST",
    "GPU_NODE",
    "EmbedderInstance",
    "GuardedInstance",
    "gpu_members_from_declarations",
    "PoolTopology",
    "live_topology",
    "topology_from_declarations",
]
