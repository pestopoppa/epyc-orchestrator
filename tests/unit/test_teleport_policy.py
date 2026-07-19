from src.llm_primitives.teleport import (
    TeleportInputs,
    TeleportPolicy,
    decide_teleport,
)


def _inputs(**overrides):
    base = {
        "role": "architect_general",
        "generated_tokens": 200,
        "estimated_remaining_tokens": 500,
        "cpu_tps": 20.0,
        "gpu_tps": 44.0,
        "gpu_available": True,
        "gpu_resident": True,
    }
    base.update(overrides)
    return TeleportInputs(**base)


def test_teleport_policy_is_default_off():
    decision = decide_teleport(TeleportPolicy(), _inputs())

    assert decision.should_cutover is False
    assert decision.reason == "disabled"
    assert decision.catch_up_supported is False
    assert decision.catch_up_reason == "llama_server_verify_api_unavailable"


def test_teleport_policy_accepts_resident_break_even():
    policy = TeleportPolicy(enabled=True, allowed_roles=frozenset({"architect_general"}))

    decision = decide_teleport(policy, _inputs(estimated_remaining_tokens=160))

    assert decision.should_cutover is True
    assert decision.reason == "cutover"
    assert decision.threshold_tokens == 150
    assert decision.estimated_speedup == 2.2


def test_teleport_policy_uses_cold_break_even():
    policy = TeleportPolicy(enabled=True)

    decision = decide_teleport(
        policy,
        _inputs(gpu_resident=False, estimated_remaining_tokens=300),
    )

    assert decision.should_cutover is False
    assert decision.reason == "below_break_even_tokens"
    assert decision.threshold_tokens == 350


def test_teleport_policy_rejects_unavailable_gpu():
    policy = TeleportPolicy(enabled=True)

    decision = decide_teleport(policy, _inputs(gpu_available=False))

    assert decision.should_cutover is False
    assert decision.reason == "gpu_unavailable"


def test_teleport_policy_rejects_role_not_allowed():
    policy = TeleportPolicy(enabled=True, allowed_roles=frozenset({"frontdoor"}))

    decision = decide_teleport(policy, _inputs(role="architect_general"))

    assert decision.should_cutover is False
    assert decision.reason == "role_not_allowed"


def test_teleport_policy_rejects_weak_speedup():
    policy = TeleportPolicy(enabled=True, min_speedup=1.5)

    decision = decide_teleport(policy, _inputs(cpu_tps=40.0, gpu_tps=44.0))

    assert decision.should_cutover is False
    assert decision.reason == "below_speedup_threshold"
