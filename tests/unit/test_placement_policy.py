"""WP-5 scaffold tests: RolePlacementPolicy enum + get_placement_policy accessor.

The scaffold ships with conservative defaults and a stable accessor API so
WP-3 can call `get_placement_policy(role)` without WP-5 needing to refactor
call sites later.
"""

from __future__ import annotations

import pytest

from src.scheduling.placement_policy import (
    DEFAULT_PLACEMENT_POLICY,
    RolePlacementPolicy,
    get_placement_policy,
)


def test_default_is_solo_prefer_full() -> None:
    assert DEFAULT_PLACEMENT_POLICY is RolePlacementPolicy.SOLO_PREFER_FULL


def test_enum_values_are_kebab_strings() -> None:
    """Values must be string-comparable so NUMA_CONFIG entries can use bare
    strings without an enum import."""
    assert RolePlacementPolicy.SOLO_PREFER_FULL.value == "solo_prefer_full"
    assert RolePlacementPolicy.BURST_PREFER_QUARTERS.value == "burst_prefer_quarters"
    assert RolePlacementPolicy.FULL_DISABLED.value == "full_disabled"
    assert RolePlacementPolicy.QUEUE_ONLY.value == "queue_only"


def test_unknown_role_returns_default() -> None:
    assert get_placement_policy("nonexistent", {}) is DEFAULT_PLACEMENT_POLICY
    assert get_placement_policy("nonexistent", {"other": {"placement_policy": "queue_only"}}) is DEFAULT_PLACEMENT_POLICY


def test_missing_policy_field_returns_default() -> None:
    cfg = {"frontdoor": {"instances": []}}
    assert get_placement_policy("frontdoor", cfg) is DEFAULT_PLACEMENT_POLICY


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("solo_prefer_full", RolePlacementPolicy.SOLO_PREFER_FULL),
        ("burst_prefer_quarters", RolePlacementPolicy.BURST_PREFER_QUARTERS),
        ("full_disabled", RolePlacementPolicy.FULL_DISABLED),
        ("queue_only", RolePlacementPolicy.QUEUE_ONLY),
        ("  burst_prefer_quarters  ", RolePlacementPolicy.BURST_PREFER_QUARTERS),  # whitespace
        ("FULL_DISABLED", RolePlacementPolicy.FULL_DISABLED),  # case
    ],
)
def test_valid_string_values_coerce(raw: str, expected: RolePlacementPolicy) -> None:
    cfg = {"frontdoor": {"placement_policy": raw}}
    assert get_placement_policy("frontdoor", cfg) is expected


def test_enum_value_passes_through() -> None:
    cfg = {"r": {"placement_policy": RolePlacementPolicy.QUEUE_ONLY}}
    assert get_placement_policy("r", cfg) is RolePlacementPolicy.QUEUE_ONLY


@pytest.mark.parametrize("bad", ["garbage", "", "solo-prefer-full", 42, None, {}])
def test_malformed_value_falls_back_to_default(bad: object) -> None:
    cfg = {"r": {"placement_policy": bad}}
    assert get_placement_policy("r", cfg) is DEFAULT_PLACEMENT_POLICY


def test_live_call_with_no_arg_does_not_crash() -> None:
    """Live wrapper (no numa_config arg) reads production NUMA_CONFIG; every
    role lacks `placement_policy` field as of WP-5 scaffold, so all return
    the default — verifying the path is stable for WP-3 consumers."""
    for role in ("frontdoor", "worker_general", "ingest_long_context", "made_up_role"):
        assert get_placement_policy(role) is DEFAULT_PLACEMENT_POLICY
