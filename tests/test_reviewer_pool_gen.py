#!/usr/bin/env python3
"""Unit tests for scripts/analysis/reviewer_pool_gen.py (RM-1).

Tests run entirely over a small SYNTHETIC registry fixture written to a
temp path — they never read the live model_registry.yaml.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "analysis" / "reviewer_pool_gen.py"
)
_SPEC = importlib.util.spec_from_file_location("reviewer_pool_gen", _MODULE_PATH)
rpg = importlib.util.module_from_spec(_SPEC)
# Register before exec so @dataclasses.dataclass can resolve cls.__module__.
sys.modules["reviewer_pool_gen"] = rpg
_SPEC.loader.exec_module(rpg)


# --------------------------------------------------------------------------- #
# Synthetic registry fixture
# --------------------------------------------------------------------------- #
def _synthetic_registry() -> dict:
    """A compact registry covering every branch the pruner exercises."""
    return {
        "roles": {
            # A large, fast, high-quality architect (Qwen family).
            "arch_big_qwen": {
                "tier": "B",
                "description": "big qwen architect",
                "model": {"name": "Qwen3.5-122B-A10B", "quant": "Q4_K_M",
                          "size_gb": 69, "architecture": "qwen35moe"},
                "candidate_roles": ["architect", "general"],
                "performance": {"baseline_tps": 12.0, "quality_pct": 88},
                "memory": {"residency": "warm"},
            },
            # A cross-family reviewer (GLM), measured, high quality.
            "rev_glm": {
                "tier": "B",
                "description": "glm reviewer",
                "model": {"name": "GLM-5.2-UD-IQ2_M", "quant": "UD-IQ2_M",
                          "size_gb": 239, "architecture": "glm_moe_dsa"},
                "candidate_roles": ["architect", "reasoning"],
                "performance": {"baseline_tps": 8.0, "quality_score": "2.4/3"},
                "memory": {"residency": "cold"},
            },
            # A cheap grader (small, gemma family).
            "grader_cheap": {
                "tier": "C",
                "description": "cheap grader",
                "model": {"name": "gemma-4-26B-A4B-it", "quant": "Q4_K_M",
                          "size_gb": 16, "architecture": "gemma4"},
                "candidate_roles": ["worker", "general", "coder"],
                "performance": {"baseline_tps": 38.0, "quality_pct": 90},
                "memory": {"residency": "hot"},
            },
            # A model measured BELOW the t/s floor, not forced -> dropped.
            "slow_measured": {
                "tier": "C",
                "description": "too slow, measured",
                "model": {"name": "SlowThing-70B", "quant": "Q4_K_M",
                          "size_gb": 40, "architecture": "dense"},
                "candidate_roles": ["general", "reasoning"],
                "performance": {"baseline_tps": 0.9},
                "memory": {"residency": "cold"},
            },
            # A model with NO measured data -> kept, flagged unmeasured.
            "unmeasured_model": {
                "tier": "B",
                "description": "no perf data",
                "model": {"name": "Mystery-30B", "quant": "Q8_0",
                          "size_gb": 30, "architecture": "dense"},
                "candidate_roles": ["general", "thinking"],
                "memory": {"residency": "cold"},
            },
            # A staged candidate that is BELOW every floor but force-included.
            "bonsai_27b_q1_0": {
                "tier": "C",
                "description": "staged 1-bit, slow + low quality",
                "model": {"name": "Bonsai-27B-Q1_0", "quant": "Q1_0",
                          "size_gb": 3.6, "architecture": "qwen35"},
                "candidate_roles": ["worker", "reasoning", "frontdoor"],
                "performance": {"baseline_tps": 0.5, "quality_pct": 20},
                "memory": {"residency": "cold"},
            },
            # An embedder -> excluded.
            "embedder_bge": {
                "tier": "D",
                "description": "bge embedder",
                "model": {"name": "bge-large-en", "quant": "F16",
                          "size_gb": 1, "architecture": "bert"},
                "candidate_roles": ["embedder"],
                "memory": {"residency": "hot"},
            },
            # A vision model -> excluded.
            "vision_vl": {
                "tier": "B",
                "description": "vision",
                "model": {"name": "Qwen3-VL-8B", "quant": "Q8_0",
                          "size_gb": 8, "architecture": "qwen3vl"},
                "candidate_roles": ["vision", "multimodal"],
                "memory": {"residency": "hot"},
            },
            # A draft-only model -> excluded.
            "draft_tiny": {
                "tier": "D",
                "description": "draft only",
                "model": {"name": "Qwen3-0.6B", "quant": "Q8_0",
                          "size_gb": 1, "architecture": "dense"},
                "candidate_roles": ["draft"],
                "memory": {"residency": "cold"},
            },
            # A deprecated model -> excluded (not forced).
            "old_dep": {
                "tier": "X",
                "deprecated": True,
                "description": "deprecated",
                "model": {"name": "MiniMax-M2.1", "quant": "Q4_K_M",
                          "size_gb": 132, "architecture": "moe"},
                "candidate_roles": ["architect", "general"],
                "memory": {"residency": "cold"},
            },
        }
    }


@pytest.fixture()
def registry_path(tmp_path) -> Path:
    p = tmp_path / "synthetic_registry.yaml"
    p.write_text(yaml.safe_dump(_synthetic_registry(), sort_keys=True))
    return p


@pytest.fixture()
def cfg():
    # Point anchors at fixture keys where possible; leave A3/A4 reviewer keys
    # (qwen35_122b_q4km / glm_52_ud_iq2m) unresolved to exercise that path.
    return rpg.PruneConfig(
        production_architect="arch_big_qwen",
        default_grader="grader_cheap",
        staged_keys=("bonsai_27b_q1_0",),
        production_trio=("grader_cheap",),
        ts_floor=3.0,
        quality_floor=0.60,
    )


def test_default_staged_keys_exclude_bonsai_q1_0():
    assert "bonsai_27b_q1_0" not in rpg.DEFAULT_STAGED_KEYS
    assert "bonsai_27b_q1_0" not in rpg.PruneConfig().staged_keys


def _build(registry_path, cfg, **kw):
    data, sha = rpg.load_registry(str(registry_path))
    return rpg.build_output(data, cfg, str(registry_path), sha, **kw), sha


# --------------------------------------------------------------------------- #
# Exclusions
# --------------------------------------------------------------------------- #
def test_embedder_vision_draft_deprecated_excluded(registry_path, cfg):
    out, _ = _build(registry_path, cfg)
    all_keys = {c["key"] for pool in out["pools"].values() for c in pool}
    for excluded in ("embedder_bge", "vision_vl", "draft_tiny", "old_dep"):
        assert excluded not in all_keys, f"{excluded} should be excluded"
    ec = out["provenance"]["excluded_counts"]
    assert ec.get("embedder") == 1
    assert ec.get("vision_or_multimodal") == 1
    assert ec.get("draft_only") == 1
    assert ec.get("deprecated") == 1


# --------------------------------------------------------------------------- #
# Floors
# --------------------------------------------------------------------------- #
def test_measured_below_ts_floor_dropped(registry_path, cfg):
    out, _ = _build(registry_path, cfg)
    all_keys = {c["key"] for pool in out["pools"].values() for c in pool}
    assert "slow_measured" not in all_keys
    dropped = {d["key"]: d for d in out["floor_dropped"]}
    assert "slow_measured" in dropped
    assert any("below_ts_floor" in r for r in dropped["slow_measured"]["reasons"])


def test_unmeasured_model_kept_and_flagged(registry_path, cfg):
    out, _ = _build(registry_path, cfg)
    cards = {c["key"]: c for pool in out["pools"].values() for c in pool}
    assert "unmeasured_model" in cards
    card = cards["unmeasured_model"]
    assert card["tps_measured"] is False
    assert card["quality_measured"] is False
    assert card["tps"] is None


def test_staged_forced_present_despite_below_floor(registry_path, cfg):
    out, _ = _build(registry_path, cfg)
    cards = {c["key"]: c for pool in out["pools"].values() for c in pool}
    # bonsai is below ts + quality floors yet forced-in on staged list.
    assert "bonsai_27b_q1_0" in cards
    assert cards["bonsai_27b_q1_0"]["forced"] is True
    assert cards["bonsai_27b_q1_0"]["staged"] is True
    # It must NOT appear in the floor_dropped audit list.
    assert "bonsai_27b_q1_0" not in {d["key"] for d in out["floor_dropped"]}


def test_high_ts_floor_still_keeps_forced(registry_path):
    cfg = rpg.PruneConfig(
        production_architect="arch_big_qwen",
        default_grader="grader_cheap",
        staged_keys=("bonsai_27b_q1_0",),
        production_trio=("grader_cheap",),
        ts_floor=100.0,  # brutal floor
    )
    out, _ = _build(registry_path, cfg, pools_only=True)
    cards = {c["key"]: c for pool in out["pools"].values() for c in pool}
    assert "bonsai_27b_q1_0" in cards       # forced survives
    assert "arch_big_qwen" not in cards      # non-forced 12 t/s dropped


# --------------------------------------------------------------------------- #
# Anchors
# --------------------------------------------------------------------------- #
def test_anchor_arms_always_present(registry_path, cfg):
    out, _ = _build(registry_path, cfg)
    ids = [a["arm_id"] for a in out["anchor_arms"]]
    assert ids == ["A0", "A1", "A3", "A4"]


def test_anchor_unresolved_key_flagged(registry_path, cfg):
    out, _ = _build(registry_path, cfg)
    arms = {a["arm_id"]: a for a in out["anchor_arms"]}
    # A4 reviewer default key (glm_52_ud_iq2m) is absent from the fixture.
    assert arms["A4"]["reviewer"]["resolved"] is False
    # A0 gates-only has no reviewer/grader.
    assert arms["A0"]["reviewer"] is None
    assert arms["A0"]["grader"] is None
    # A1 self-review: architect == reviewer, both resolved.
    assert arms["A1"]["architect"]["key"] == arms["A1"]["reviewer"]["key"]
    assert arms["A1"]["reviewer"]["resolved"] is True


def test_anchor_present_even_when_all_pruned(registry_path):
    """Anchors survive a config that prunes every candidate."""
    cfg = rpg.PruneConfig(
        production_architect="arch_big_qwen",
        default_grader="grader_cheap",
        staged_keys=(), production_trio=(),
        ts_floor=1e9, quality_floor=1.0,
    )
    out, _ = _build(registry_path, cfg, pools_only=True)
    ids = [a["arm_id"] for a in out["anchor_arms"]]
    assert ids == ["A0", "A1", "A3", "A4"]


# --------------------------------------------------------------------------- #
# Pairings + cross-family
# --------------------------------------------------------------------------- #
def test_cross_family_flag_correct(registry_path, cfg):
    out, _ = _build(registry_path, cfg)
    pairings = {p["pairing_id"]: p for p in out["pairings"]}
    # arch_big_qwen (qwen) x rev_glm (glm) -> cross-family True
    hit = [p for p in out["pairings"]
           if p["architect"] == "arch_big_qwen" and p["reviewer"] == "rev_glm"]
    assert hit and hit[0]["cross_family_preferred"] is True
    # arch_big_qwen x arch_big_qwen -> same family, self-review
    self_hit = [p for p in out["pairings"]
                if p["architect"] == "arch_big_qwen"
                and p["reviewer"] == "arch_big_qwen"]
    assert self_hit
    assert self_hit[0]["cross_family_preferred"] is False
    assert self_hit[0]["self_review"] is True


def test_require_cross_family_drops_same_family(registry_path, cfg):
    cfg2 = rpg.dataclasses.replace(cfg, require_cross_family=True)
    out, _ = _build(registry_path, cfg2)
    assert all(p["cross_family_preferred"] for p in out["pairings"])
    assert out["provenance"]["n_pairings_dropped"] > 0


def test_coresidency_prune_on_tiny_host(registry_path, cfg):
    # Host so small that GLM(239)+QWen(69) cannot co-reside.
    cfg2 = rpg.dataclasses.replace(cfg, host_ram_gb=100.0,
                                   coresidency_fraction=1.0)
    out, _ = _build(registry_path, cfg2)
    for p in out["pairings"]:
        assert p["coresidency"]["fits"] is True  # only fitting pairs survive
    assert out["provenance"]["n_pairings_dropped"] > 0


def test_sequential_swap_uses_max_footprint(registry_path, cfg):
    cfg2 = rpg.dataclasses.replace(cfg, sequential_swap=True,
                                   host_ram_gb=250.0, coresidency_fraction=1.0)
    out, _ = _build(registry_path, cfg2)
    # GLM alone is 239 < 250, so a GLM-involving pair survives under swap mode
    # even though the co-resident sum would exceed the budget.
    assert any(p["reviewer"] == "rev_glm" for p in out["pairings"])
    for p in out["pairings"]:
        assert p["coresidency"]["mode"] == "sequential"


def test_grader_sweep_expands_grader_dimension(registry_path, cfg):
    single, _ = _build(registry_path, cfg)
    swept_cfg = rpg.dataclasses.replace(cfg, grader_sweep=True)
    swept, _ = _build(registry_path, swept_cfg)
    graders_single = {p["grader"] for p in single["pairings"]}
    graders_swept = {p["grader"] for p in swept["pairings"]}
    assert len(graders_single) == 1
    assert len(graders_swept) >= len(graders_single)


# --------------------------------------------------------------------------- #
# Provenance + determinism
# --------------------------------------------------------------------------- #
def test_provenance_has_hashes(registry_path, cfg):
    out, sha = _build(registry_path, cfg)
    pv = out["provenance"]
    assert pv["registry_sha256"] == sha
    assert len(pv["registry_sha256"]) == 64
    assert len(pv["prune_config_sha256"]) == 64
    assert pv["prune_config"]["ts_floor"] == 3.0
    assert pv["n_roles_scanned"] == 10


def test_deterministic_bytes(registry_path, cfg):
    out1, _ = _build(registry_path, cfg)
    out2, _ = _build(registry_path, cfg)
    assert rpg.dumps(out1) == rpg.dumps(out2)


def test_config_hash_changes_with_knob(registry_path, cfg):
    out1, _ = _build(registry_path, cfg)
    cfg2 = rpg.dataclasses.replace(cfg, ts_floor=9.0)
    out2, _ = _build(registry_path, cfg2)
    assert (out1["provenance"]["prune_config_sha256"]
            != out2["provenance"]["prune_config_sha256"])


# --------------------------------------------------------------------------- #
# Helper units
# --------------------------------------------------------------------------- #
def test_infer_family_arch_and_name():
    assert rpg.infer_family({"model": {"architecture": "glm_moe_dsa"}}) == "glm"
    assert rpg.infer_family({"model": {"architecture": "gemma4"}}) == "gemma"
    # Bonsai is a Qwen derivative -> qwen family (via arch or name).
    assert rpg.infer_family(
        {"model": {"name": "Bonsai-27B-Q1_0", "architecture": "qwen35"}}) == "qwen"
    # Ambiguous arch falls back to name.
    assert rpg.infer_family(
        {"model": {"name": "DeepSeek-R1-Distill-Llama-70B",
                   "architecture": "dense"}}) == "deepseek"
    assert rpg.infer_family(
        {"model": {"name": "Mystery", "architecture": "dense"}}) == "other"


def test_quality_fraction_parsing():
    assert rpg.get_quality_fraction({"performance": {"quality_pct": 93}}) == 0.93
    assert abs(rpg.get_quality_fraction(
        {"performance": {"quality_score": "2.57/3"}}) - 2.57 / 3) < 1e-9
    assert rpg.get_quality_fraction(
        {"performance": {"quality_score": "25/27 (93%)"}}) == 0.93
    assert rpg.get_quality_fraction({"performance": {}}) is None


def test_tps_takes_best_measured():
    assert rpg.get_tps(
        {"performance": {"baseline_tps": 7.0, "mtp_tps": 21.0}}) == 21.0
    assert rpg.get_tps({"performance": {"baseline_tps": None}}) is None
