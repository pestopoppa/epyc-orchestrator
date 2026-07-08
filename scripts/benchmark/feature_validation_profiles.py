"""Feature-validation harness data structures + profile registry.

Extracted from scripts/benchmark/feature_validation.py during the 2026-05-22
Task-D refactor. Holds the dataclasses (TestSpec, FeatureProfile,
MetricSnapshot, ComparisonReport) and _build_profiles() that defines the
feature-by-tier validation matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("feature_validation")


@dataclass
class TestSpec:
    """Single test specification within a feature profile."""

    name: str
    kind: str  # "unit", "replay", "live"
    prompt_manifest: str = ""  # path relative to MANIFESTS_DIR
    pass_criteria: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)




@dataclass
class FeatureProfile:
    """Complete validation profile for one feature flag."""

    name: str
    tier: int
    deps: list[str] = field(default_factory=list)
    offline_tests: list[TestSpec] = field(default_factory=list)
    live_tests: list[TestSpec] = field(default_factory=list)
    description: str = ""




@dataclass
class MetricSnapshot:
    """Metrics captured from a single run."""

    feature: str
    enabled: bool
    timestamp: str = ""
    # Quality
    quality_score: float = 0.0
    routing_accuracy: float = 0.0
    escalation_rate: float = 0.0
    # Performance
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0
    predicted_tps: float = 0.0
    # Memory
    memory_rss_mb: float = 0.0
    memory_delta_mb: float = 0.0
    # Misc
    test_pass_rate: float = 0.0
    prompts_run: int = 0
    errors: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)




@dataclass
class ComparisonReport:
    """Side-by-side baseline vs candidate comparison."""

    feature: str
    baseline: MetricSnapshot | None = None
    candidate: MetricSnapshot | None = None
    quality_delta: float = 0.0
    latency_delta_s: float = 0.0
    tps_delta: float = 0.0
    memory_delta_mb: float = 0.0
    verdict: str = "PENDING"  # PASS / FAIL / BORDERLINE / PENDING


# ── Feature profile registry ────────────────────────────────────────



def _build_profiles() -> dict[str, FeatureProfile]:
    """Build the complete set of feature validation profiles."""
    profiles: dict[str, FeatureProfile] = {}

    # ── Tier 0: Trivial (unit test only) ──
    profiles["accurate_token_counting"] = FeatureProfile(
        name="accurate_token_counting", tier=0,
        description="Use /tokenize for exact counts vs len//4 heuristic",
        offline_tests=[TestSpec("token_accuracy", "unit",
                                pass_criteria={"mean_error_pct": 5.0, "latency_ms": 5.0})],
    )
    profiles["content_cache"] = FeatureProfile(
        name="content_cache", tier=0,
        description="SHA-256 keyed response cache for identical prompts",
        offline_tests=[TestSpec("cache_hit_rate", "unit",
                                pass_criteria={"hit_rate": 1.0})],
    )
    profiles["deferred_tool_results"] = FeatureProfile(
        name="deferred_tool_results", tier=0,
        description="Keep tool outputs out of prompt context by default",
        offline_tests=[TestSpec("prompt_size_reduction", "unit",
                                pass_criteria={"size_reduction_gt": 0})],
    )

    # ── Tier 1: MemRL incremental chain ──
    memrl_chain = [
        ("specialist_routing", "Routing via Q-values"),
        ("plan_review", "Architect review of frontdoor plans"),
        ("architect_delegation", "Architect delegates to specialists"),
        ("parallel_execution", "Wave-based parallel step execution"),
    ]
    cumulative_deps: list[str] = ["memrl"]
    for feat_name, desc in memrl_chain:
        profiles[feat_name] = FeatureProfile(
            name=feat_name, tier=1, deps=list(cumulative_deps),
            description=desc,
            offline_tests=[TestSpec(f"{feat_name}_replay", "replay",
                                    prompt_manifest="memrl_chain.json",
                                    pass_criteria={"quality_ge_baseline": 0.0})],
            live_tests=[TestSpec(f"{feat_name}_live", "live",
                                 prompt_manifest="memrl_chain.json",
                                 pass_criteria={"quality_ge_baseline": 0.0,
                                                "latency_overhead_s": 2.0})],
        )
        cumulative_deps.append(feat_name)

    # ── Tier 2: Independent features ──
    tier2 = {
        "react_mode": ("tool_compliance.json", "ReAct tool loop"),
        "output_formalizer": ("output_format.json", "Format constraint enforcement"),
        "input_formalizer": ("input_formalize.json", "MathSmith-8B preprocessing"),
        "personas": ("personas.json", "Persona-based prompt overlays"),
        "model_fallback": ("model_fallback.json", "Circuit-open fallback"),
        "unified_streaming": ("streaming.json", "Stream adapter correctness"),
        "escalation_compression": ("escalation_compress.json", "LLMLingua-2 compression"),
        "binding_routing": ("binding_routing.json", "Priority routing overrides"),
    }
    for feat_name, (manifest, desc) in tier2.items():
        profiles[feat_name] = FeatureProfile(
            name=feat_name, tier=2, description=desc,
            offline_tests=[TestSpec(f"{feat_name}_unit", "unit",
                                    prompt_manifest=manifest,
                                    pass_criteria={"quality_ge_baseline": 0.0})],
            live_tests=[TestSpec(f"{feat_name}_live", "live",
                                 prompt_manifest=manifest,
                                 pass_criteria={"quality_ge_baseline": 0.0})],
        )

    # ── Tier 3: Safety & infrastructure ──
    tier3 = {
        "side_effect_tracking": "Tool side-effect declarations",
        "resume_tokens": "Crash-recovery continuation tokens",
        "approval_gates": "Human approval at escalation boundaries",
        "structured_tool_output": "ToolOutput envelope wrapping",
        "cascading_tool_policy": "Global→Role→Task permission chain",
        "credential_redaction": "Credential scan regression test",
    }
    # Tier 3 features use general prompts for regression testing —
    # verify enabling doesn't break quality or add unacceptable latency.
    # tool_compliance prompts exercise tool-use paths relevant to safety features.
    tier3_manifest = {
        "side_effect_tracking": "tool_compliance.json",
        "resume_tokens": "general_5.json",
        "approval_gates": "tool_compliance.json",
        "structured_tool_output": "tool_compliance.json",
        "cascading_tool_policy": "tool_compliance.json",
        "credential_redaction": "general_5.json",
    }
    for feat_name, desc in tier3.items():
        deps = []
        if feat_name == "approval_gates":
            deps = ["side_effect_tracking", "resume_tokens"]
        manifest = tier3_manifest.get(feat_name, "general_5.json")
        profiles[feat_name] = FeatureProfile(
            name=feat_name, tier=3, deps=deps, description=desc,
            offline_tests=[TestSpec(f"{feat_name}_unit", "unit",
                                    pass_criteria={"pass_rate": 1.0})],
            live_tests=[TestSpec(f"{feat_name}_live", "live",
                                 prompt_manifest=manifest,
                                 pass_criteria={"quality_ge_baseline": 0.0,
                                                "latency_overhead_s": 5.0})],
        )

    # ── Tier 4: Deferred ──
    for feat_name in ("skillbank", "staged_rewards", "script_interception", "restricted_python"):
        profiles[feat_name] = FeatureProfile(
            name=feat_name, tier=4, description=f"Deferred: {feat_name}",
        )

    return profiles


PROFILES = _build_profiles()


# ── Helpers ──────────────────────────────────────────────────────────


