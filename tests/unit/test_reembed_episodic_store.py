from __future__ import annotations

from scripts.graph_router.reembed_episodic_store import retry_port_order
from scripts.maintenance import repair_episodic_embeddings as repair
from scripts.server.stack_manifest import EMBEDDER_PORTS


def test_retry_port_order_rotates_from_primary() -> None:
    assert retry_port_order(8102, [8100, 8101, 8102, 8103]) == [
        8102,
        8103,
        8100,
        8101,
    ]


def test_retry_port_order_handles_unknown_primary() -> None:
    assert retry_port_order(9000, [8100, 8101]) == [9000, 8100, 8101]


def test_repair_defaults_follow_stack_embedder_ports() -> None:
    assert repair.DEFAULT_EMBEDDER_SERVERS == len(EMBEDDER_PORTS)
    assert repair.DEFAULT_EMBEDDER_BASE_PORT == min(EMBEDDER_PORTS)
