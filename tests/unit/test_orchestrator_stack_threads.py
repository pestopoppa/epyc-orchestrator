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

import pytest


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


def test_quartered_roles_per_instance_thread_count() -> None:
    """Per-instance thread count must resolve correctly for the quartered roles.

    Scope corrected 2026-07-30: this previously asserted `vision_escalation`
    had "full + at least 1 quarter post-Phase-1b" and had been failing against
    the live config, which gives it a SINGLE instance on `72-95,168-191` @24t
    (NUMA_Q1B, = NPS4 node3). The Phase-1b claim in the old docstring does not
    match `stack_numa.NUMA_CONFIG`; the config is the source of truth, so the
    role moved to its own shape assertion below.
    """
    for role in ("ingest_long_context",):
        instances = _instances(role)
        assert len(instances) >= 2, (
            f"{role} should have full + at least 1 quarter (got {len(instances)})"
        )
        for idx, (_cpus, _port, expected_threads) in enumerate(instances):
            resolved = orchestrator_stack._resolve_thread_count(role, idx)
            assert int(resolved) == expected_threads, (
                f"{role} instance[{idx}] expected -t {expected_threads}, got {resolved}"
            )


def test_vision_escalation_stays_single_instance() -> None:
    """`vision_escalation` is single-instance on NUMA_Q1B (NPS4 node3).

    Added 2026-07-30 to pin the actual shape, replacing the stale expectation in
    `test_quartered_roles_per_instance_thread_count`. Node-aligned, so it is not
    exposed to the straddling-cpuset defect — but it does share one VL GGUF with
    `worker_vision` on a different node, so under shared mmap only one of the two
    can hold node-local weights. See handoffs/active/numa-placement-defect-20260730.md.
    """
    instances = _instances("vision_escalation")
    assert len(instances) == 1, (
        f"vision_escalation should be single-instance (got {len(instances)})"
    )
    cpus, port, threads = instances[0]
    assert port == 8087
    assert threads == 24
    assert cpus == "72-95,168-191"  # NUMA_Q1B == NPS4 node3
    assert _nodes_spanned(cpus) == {3}, "should be node-aligned, not straddling"


def test_worker_vision_stays_single_instance() -> None:
    """Phase 0.5 bench showed Qwen2.5-VL-7B is flat between 24t/48t and not
    worth quartering. Stays single instance on NUMA_Q0B."""
    instances = _instances("worker_vision")
    assert len(instances) == 1, "worker_vision should be single-instance (too small to quarter)"
    cpus, port, threads = instances[0]
    assert port == 8086
    assert threads == 24
    assert cpus == "24-47,120-143"  # NUMA_Q0B


# Live NPS4 topology, verified 2026-07-30 via `numactl --hardware`.
NPS4_NODES = {
    0: "0-23,96-119",
    1: "24-47,120-143",
    2: "48-71,144-167",
    3: "72-95,168-191",
}


def _parse_cpuset(cpuset: str) -> set[int]:
    cpus: set[int] = set()
    for part in cpuset.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _nodes_spanned(cpuset: str) -> set[int]:
    cpus = _parse_cpuset(cpuset)
    return {n for n, cs in NPS4_NODES.items() if cpus & _parse_cpuset(cs)}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT, fix not yet authorised — see "
        "handoffs/active/numa-placement-defect-20260730.md. `frontdoor` (8070) and "
        "`ingest_long_context` (8085) sit on NUMA_NODE0 '0-47,96-143', which spans "
        "NPS4 nodes 0+1, with no numactl policy. Measured cost 2.16x and 1.85x. "
        "strict=True on purpose: when the wiring is corrected this test starts "
        "passing and the strict xfail FAILS the suite, forcing this marker to be "
        "removed rather than silently outliving the defect."
    ),
)
def test_straddling_cpusets_declare_a_numa_policy() -> None:
    """Any instance whose cpuset spans more than one NPS4 node must declare a
    memory policy.

    Replaces `test_frontdoor_full_uses_numa_node0_not_full`, removed 2026-07-30.
    That test asserted `full[0] == "0-47,96-143"` AND that frontdoor carries no
    `numactl_policy` — so it did not merely document the defect, it *forbade the
    fix*: adding `interleave=all` to frontdoor would have failed it. Its stated
    authority was an April 2026-04-17 head-to-head that is invalid twice over —
    it predates the 2026-04-24 NPS4 reboot (when `0-47,96-143` genuinely was one
    node), and its source CSV records `spec == "baseline"`.

    Without a policy, every weight page lands on whichever node faults it first
    and the rest of the thread team reads cross-node for the model's lifetime.
    `numactl --interleave` binds at FIRST TOUCH only, so a warm-cache A/B cannot
    detect this — which is exactly how the 1.7% result that justified the old
    test was produced.
    """
    offenders: list[str] = []
    for role, cfg in stack_numa.NUMA_CONFIG.items():
        policy = cfg.get("numactl_policy") or cfg.get("numactl_policy_instances")
        if policy:
            continue
        for cpuset, port, _threads in cfg.get("instances", []):
            spanned = _nodes_spanned(cpuset)
            if len(spanned) > 1:
                offenders.append(
                    f"{role}:{port} cpuset={cpuset!r} spans NPS4 nodes "
                    f"{sorted(spanned)} with no numactl policy"
                )
    assert not offenders, "straddling cpuset without a memory policy:\n  " + "\n  ".join(
        offenders
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
