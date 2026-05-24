"""Per-instance thread-count resolution for the orchestrator stack launcher.

Regression test for the 2026-05-24 launcher bug: `_resolve_thread_count` used
to always return `NUMA_CONFIG[role]["instances"][0][2]` regardless of which
instance was being launched. Quarters got `-t 96` instead of `-t 48`. This
test asserts that each instance's thread count matches its NUMA_CONFIG entry.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT / "scripts" / "server"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVER_DIR))


orchestrator_stack = importlib.import_module("orchestrator_stack")
stack_numa = importlib.import_module("stack_numa")


def _instances(role: str) -> list[tuple[str, int, int]]:
    return stack_numa.NUMA_CONFIG.get(role, {}).get("instances", [])


def test_frontdoor_per_instance_thread_count() -> None:
    instances = _instances("frontdoor")
    assert len(instances) >= 2, "frontdoor should have a full instance + at least 1 quarter"
    for idx, (_cpus, _port, expected_threads) in enumerate(instances):
        resolved = orchestrator_stack._resolve_thread_count("frontdoor", idx)
        assert int(resolved) == expected_threads, (
            f"frontdoor instance[{idx}] expected -t {expected_threads}, got {resolved}"
        )


def test_worker_general_per_instance_thread_count() -> None:
    """worker_general previously had a manual port-matching workaround; with the
    generic fix it should give identical results without the workaround."""
    instances = _instances("worker_general")
    assert len(instances) >= 2
    for idx, (_cpus, _port, expected_threads) in enumerate(instances):
        resolved = orchestrator_stack._resolve_thread_count("worker_general", idx)
        assert int(resolved) == expected_threads, (
            f"worker_general instance[{idx}] expected -t {expected_threads}, got {resolved}"
        )


def test_single_instance_roles_use_first_entry() -> None:
    """Roles with only one instance (architect_general only post-2026-05-24 fix)
    should resolve to that instance's thread count regardless of numa_instance
    argument (defensive fallback)."""
    for role in ("architect_general",):
        instances = _instances(role)
        if not instances:
            continue
        expected = instances[0][2]
        assert int(orchestrator_stack._resolve_thread_count(role)) == expected
        assert int(orchestrator_stack._resolve_thread_count(role, 0)) == expected
        # out-of-range falls back defensively
        assert int(orchestrator_stack._resolve_thread_count(role, 99)) == expected


def test_quartered_vision_roles_per_instance_thread_count() -> None:
    """Phase 1b: vision_escalation + worker_vision + ingest_long_context now
    have full + 4 quarters. Per-instance thread count must resolve correctly."""
    for role in ("vision_escalation", "worker_vision", "ingest_long_context"):
        instances = _instances(role)
        assert len(instances) >= 2, (
            f"{role} should have full + at least 1 quarter post-Phase-1b "
            f"(got {len(instances)} instances)"
        )
        for idx, (_cpus, _port, expected_threads) in enumerate(instances):
            resolved = orchestrator_stack._resolve_thread_count(role, idx)
            assert int(resolved) == expected_threads, (
                f"{role} instance[{idx}] expected -t {expected_threads}, got {resolved}"
            )


def test_unknown_role_falls_back_to_96() -> None:
    assert orchestrator_stack._resolve_thread_count("not_a_real_role") == "96"
    assert orchestrator_stack._resolve_thread_count("not_a_real_role", 3) == "96"


def test_pre_fix_quarter_bug_is_fixed() -> None:
    """The exact regression: pre-fix, frontdoor quarter 1 (port 8080) reported
    96 threads when its NUMA_CONFIG entry specifies 48. Post-fix it must
    report 48."""
    resolved = orchestrator_stack._resolve_thread_count("frontdoor", 1)
    assert int(resolved) == 48, (
        f"frontdoor quarter 0 should be -t 48 (NPS4 single-quarter spec), "
        f"got {resolved} — this is the 2026-05-24 regression returning"
    )
