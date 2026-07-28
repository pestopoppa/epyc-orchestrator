"""Tests for the gpu_shadow_lane guarded preflight probe (pure logic only).

The gather layer (rocm-smi / /proc / subprocess) is not exercised; tests feed
synthetic PreflightFacts into the pure evaluators, mirroring the
eval_batch_serving probe's testing style. No process is started or signalled.
"""

from __future__ import annotations

import pytest

from scripts.server import gpu_shadow_lane_preflight as probe
from scripts.server.gpu_shadow_lane import LANE_BINARY_COMMIT, LANE_BINARY_VERSION


def make_facts(**overrides) -> probe.PreflightFacts:
    base = dict(
        flag_enabled=False,
        binary_exists=True,
        binary_version_output=f"version: {LANE_BINARY_VERSION} ({LANE_BINARY_COMMIT})",
        kfd_dev_present=True,
        kfd_topology_present=True,
        vram_total_gib=64.0,
        vram_used_gib=0.5,
        gpu_compute_pids=[],
        live_llama_processes=[],
        model_file_exists=True,
        static_smt_overlaps={},
        errors=[],
    )
    base.update(overrides)
    return probe.PreflightFacts(**base)


def make_plan(**overrides) -> probe.LanePlan:
    base = dict(
        tenant_id="qwen36_27b_q8",
        model_path="/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf",
        model_vram_gib=26.7,
        np_slots=8,
        slot_context_tokens=8192,
        port=18100,
        host_cpuset="184-191",
        dynamic_budget_gib=27.0,
        np_ceiling=16,
        estimated_dynamic_gib=7.2,
        launch_argv=["taskset", "-c", "184-191"],
    )
    base.update(overrides)
    return probe.LanePlan(**base)


# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------
class TestParsers:
    def test_parse_cpu_list_ranges_and_singles(self):
        assert probe.parse_cpu_list("184-191") == set(range(184, 192))
        assert probe.parse_cpu_list("72-95,168-191") == set(range(72, 96)) | set(
            range(168, 192)
        )
        assert probe.parse_cpu_list("3") == {3}
        assert probe.parse_cpu_list("") == set()

    def test_cpuset_overlap(self):
        assert probe.cpuset_overlap("184-191", "72-95,168-191") == set(range(184, 192))
        assert probe.cpuset_overlap("184-191", "0-47,96-143") == set()

    def test_static_smt_overlap_includes_q1b_roles(self):
        overlaps = probe.static_smt_overlap_roles("184-191")
        # NUMA_Q1B (72-95,168-191) hosts these production quarters/lanes.
        assert "vision_escalation" in overlaps
        assert "frontdoor" in overlaps
        assert 8087 in overlaps["vision_escalation"]

    def test_parse_rocm_meminfo(self):
        payload = {
            "card0": {
                "VRAM Total Memory (B)": str(64 * (1 << 30)),
                "VRAM Total Used Memory (B)": str(1 << 30),
            }
        }
        parsed = probe.parse_rocm_meminfo(payload)
        assert parsed is not None
        total, used = parsed
        assert total == pytest.approx(64.0)
        assert used == pytest.approx(1.0)

    def test_parse_rocm_meminfo_malformed(self):
        assert probe.parse_rocm_meminfo(None) is None
        assert probe.parse_rocm_meminfo({"card0": {"foo": "1"}}) is None

    def test_parse_rocm_pids_from_keys(self):
        payload = {"system": {"PID 12345": ["comm", "1", "2"], "PID678": "x"}}
        assert probe.parse_rocm_pids(payload) == [678, 12345]
        assert probe.parse_rocm_pids({"system": {}}) == []
        assert probe.parse_rocm_pids(None) == []

    def test_parse_rocm_pids_nested_values(self):
        payload = {"process": [{"pid": 42}, {"pid": "77"}]}
        assert probe.parse_rocm_pids(payload) == [42, 77]


# ---------------------------------------------------------------------------
# Blocker evaluation
# ---------------------------------------------------------------------------
class TestEvaluate:
    def test_clean_plan_has_no_blockers(self):
        blockers, warnings = probe.evaluate_preflight(make_facts(), make_plan())
        assert blockers == []
        # flag-off is a warning pre-activation, never a blocker by default
        assert any("feature flag is off" in w for w in warnings)

    def test_require_enabled_blocks_when_flag_off(self):
        blockers, _ = probe.evaluate_preflight(
            make_facts(), make_plan(), require_enabled=True
        )
        assert any("feature flag" in b for b in blockers)

    def test_binary_version_mismatch_blocks(self):
        facts = make_facts(binary_version_output="version: 9999 (deadbeef)")
        blockers, _ = probe.evaluate_preflight(facts, make_plan())
        assert any("version mismatch" in b for b in blockers)

    def test_binary_missing_blocks(self):
        blockers, _ = probe.evaluate_preflight(
            make_facts(binary_exists=False), make_plan()
        )
        assert any("binary missing" in b for b in blockers)

    def test_kfd_missing_blocks(self):
        blockers, _ = probe.evaluate_preflight(
            make_facts(kfd_dev_present=False), make_plan()
        )
        assert any("KFD" in b for b in blockers)

    def test_vram_unattestable_blocks(self):
        blockers, _ = probe.evaluate_preflight(
            make_facts(vram_total_gib=None, vram_used_gib=None), make_plan()
        )
        assert any("attest VRAM" in b for b in blockers)

    def test_vram_short_blocks(self):
        # 64 total, 40 used -> 24 free < 26.7 model + 7.2 dynamic
        blockers, _ = probe.evaluate_preflight(
            make_facts(vram_used_gib=40.0), make_plan()
        )
        assert any("VRAM budget short" in b for b in blockers)

    def test_foreign_gpu_pids_block_by_default(self):
        facts = make_facts(gpu_compute_pids=[4242])
        blockers, _ = probe.evaluate_preflight(facts, make_plan())
        assert any("foreign GPU compute PIDs" in b for b in blockers)

    def test_foreign_gpu_pids_downgrade_with_flag(self):
        facts = make_facts(gpu_compute_pids=[4242])
        blockers, warnings = probe.evaluate_preflight(
            facts, make_plan(), allow_existing_gpu_pids=True
        )
        assert not any("foreign GPU" in b for b in blockers)
        assert any("foreign GPU" in w for w in warnings)

    def test_live_affinity_overlap_blocks_by_default(self):
        facts = make_facts(
            live_llama_processes=[
                {"pid": 111, "cpus_allowed_list": "72-95,168-191", "overlap": [184, 185]}
            ]
        )
        blockers, _ = probe.evaluate_preflight(facts, make_plan())
        assert any("overlaps LIVE production" in b for b in blockers)

    def test_live_affinity_overlap_downgrades_with_flag(self):
        facts = make_facts(
            live_llama_processes=[
                {"pid": 111, "cpus_allowed_list": "72-95,168-191", "overlap": [184]}
            ]
        )
        blockers, warnings = probe.evaluate_preflight(
            facts, make_plan(), allow_smt_overlap=True
        )
        assert not any("overlaps LIVE" in b for b in blockers)
        assert any("overlaps LIVE" in w for w in warnings)

    def test_non_overlapping_live_process_is_fine(self):
        facts = make_facts(
            live_llama_processes=[
                {"pid": 222, "cpus_allowed_list": "0-47,96-143", "overlap": []}
            ]
        )
        blockers, _ = probe.evaluate_preflight(facts, make_plan())
        assert blockers == []

    def test_static_overlap_is_warning_only(self):
        facts = make_facts(static_smt_overlaps={"vision_escalation": [8087]})
        blockers, warnings = probe.evaluate_preflight(facts, make_plan())
        assert blockers == []
        assert any("static NUMA_CONFIG co-tenants" in w for w in warnings)

    def test_missing_model_blocks(self):
        blockers, _ = probe.evaluate_preflight(
            make_facts(model_file_exists=False), make_plan()
        )
        assert any("model file missing" in b for b in blockers)

    def test_null_ceiling_blocks(self):
        blockers, _ = probe.evaluate_preflight(
            make_facts(), make_plan(np_ceiling=None)
        )
        assert any("no validated operating point" in b for b in blockers)

    def test_np_above_ceiling_blocks(self):
        blockers, _ = probe.evaluate_preflight(
            make_facts(), make_plan(np_slots=32, np_ceiling=16)
        )
        assert any("exceeds validated ceiling" in b for b in blockers)

    def test_gather_errors_propagate_as_blockers(self):
        blockers, _ = probe.evaluate_preflight(
            make_facts(errors=["rocm-smi: not found"]), make_plan()
        )
        assert "rocm-smi: not found" in blockers


# ---------------------------------------------------------------------------
# CLI defaults (no execution beyond argparse)
# ---------------------------------------------------------------------------
class TestCli:
    def test_defaults_are_plan_only(self):
        args = probe.parse_args([])
        assert args.apply is False
        assert args.require_enabled is False
        assert args.allow_existing_gpu_pids is False
        assert args.allow_smt_overlap is False
        assert args.tenant == "qwen36_27b_q8"
        assert args.budget_profile == "phase2_resident_set"
        assert args.np == 8
        assert args.slot_context == 8192
        assert args.port == 18100

    def test_build_plan_resolves_ceiling_and_estimate(self):
        from scripts.server.gpu_shadow_lane import load_np_ceiling_policy
        from src.features import Features

        policy = load_np_ceiling_policy(feats=Features(gpu_shadow_lane=True))
        args = probe.parse_args(["--np", "8", "--slot-context", "32768"])
        plan = probe.build_plan(policy, args)
        assert plan.np_ceiling == 8
        assert plan.model_vram_gib == pytest.approx(26.7)
        assert plan.estimated_dynamic_gib == pytest.approx(19.2, abs=0.01)
        assert "-np" in plan.launch_argv

    def test_build_plan_rejects_unknown_budget_profile(self):
        from scripts.server.gpu_shadow_lane import load_np_ceiling_policy
        from src.features import Features

        policy = load_np_ceiling_policy(feats=Features(gpu_shadow_lane=True))
        args = probe.parse_args(["--budget-profile", "nope"])
        with pytest.raises(KeyError, match="budget profile"):
            probe.build_plan(policy, args)
