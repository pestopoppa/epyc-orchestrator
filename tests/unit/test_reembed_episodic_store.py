from __future__ import annotations

from scripts.graph_router.reembed_episodic_store import retry_port_order


def test_retry_port_order_rotates_from_primary() -> None:
    assert retry_port_order(8102, [8100, 8101, 8102, 8103]) == [
        8102,
        8103,
        8100,
        8101,
    ]


def test_retry_port_order_handles_unknown_primary() -> None:
    assert retry_port_order(9000, [8100, 8101]) == [9000, 8100, 8101]
