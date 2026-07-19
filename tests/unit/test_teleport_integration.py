"""Decision-only AXA-2 integration tests for LLMPrimitives."""

from unittest.mock import Mock

from src.llm_primitives import LLMPrimitives
from src.llm_primitives.teleport import TeleportInputs, TeleportPolicy


def _inputs(**overrides):
    values = {
        "role": "architect_general",
        "generated_tokens": 200,
        "estimated_remaining_tokens": 500,
        "cpu_tps": 20.0,
        "gpu_tps": 44.0,
        "gpu_available": True,
        "gpu_resident": True,
    }
    values.update(overrides)
    return TeleportInputs(**values)


def test_teleport_integration_is_default_off_and_does_not_touch_lease():
    lease = Mock()
    primitives = LLMPrimitives(mock_mode=True, gpu_lease_manager=lease)

    decision = primitives.evaluate_teleport_decision(_inputs(), lease_owner="request-1")

    assert decision.should_cutover is False
    assert decision.reason == "disabled"
    lease.status.assert_not_called()
    lease.acquire.assert_not_called()


def test_teleport_integration_rejects_lease_owned_by_another_request():
    lease = Mock()
    lease.status.return_value = Mock(acquired=True, owner="request-2")
    policy = TeleportPolicy(enabled=True, allowed_roles=frozenset({"architect_general"}))
    primitives = LLMPrimitives(
        mock_mode=True,
        gpu_lease_manager=lease,
        teleport_policy=policy,
    )

    decision = primitives.evaluate_teleport_decision(_inputs(), lease_owner="request-1")

    assert decision.should_cutover is False
    assert decision.reason == "gpu_unavailable"
    lease.status.assert_called_once_with()
    lease.acquire.assert_not_called()


def test_teleport_integration_only_evaluates_decision_for_available_lease():
    lease = Mock()
    lease.status.return_value = Mock(acquired=False, owner=None)
    policy = TeleportPolicy(enabled=True, allowed_roles=frozenset({"architect_general"}))
    primitives = LLMPrimitives(
        mock_mode=True,
        gpu_lease_manager=lease,
        teleport_policy=policy,
    )

    decision = primitives.evaluate_teleport_decision(_inputs(), lease_owner="request-1")

    assert decision.should_cutover is True
    assert decision.reason == "cutover"
    lease.acquire.assert_not_called()
