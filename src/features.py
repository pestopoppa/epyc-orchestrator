#!/usr/bin/env python3
"""Feature flag system for optional orchestration modules.

This module defines all optional features that can be enabled/disabled
independently of core orchestration functionality. Each feature has:
- A clear description of what it does
- Dependencies (what other features/modules it requires)
- Environment variable to enable/disable

Usage:
    from src.features import Features, get_features

    # Get feature flags (reads from environment or config)
    features = get_features()

    # Check if a feature is enabled
    if features.memrl:
        from orchestration.repl_memory import TaskEmbedder
        # ... use MemRL components

    # Create features with explicit flags
    features = Features(memrl=False, tools=True)

Environment Variables:
    ORCHESTRATOR_MEMRL=1         Enable MemRL (learned routing, Q-scoring)
    ORCHESTRATOR_TOOLS=1         Enable tool registry (REPL tools)
    ORCHESTRATOR_SCRIPTS=1       Enable script registry (prepared scripts)
    ORCHESTRATOR_STREAMING=1     Enable SSE streaming endpoints
    ORCHESTRATOR_OPENAI_COMPAT=1 Enable OpenAI-compatible API
    ORCHESTRATOR_REPL=1          Enable REPL execution environment
    ORCHESTRATOR_DEFERRED_TOOL_RESULTS=1  Disable mixin tool-output wrapping

Design Principles:
    1. Core orchestration works with ALL features disabled
    2. Features are opt-in by default in tests, opt-out in production
    3. Each feature can be toggled independently
    4. Dependencies are documented and checked at initialization

Adding New Features:
    1. Add field to Features dataclass with description
    2. Add environment variable check in get_features()
    3. Add dependency documentation if needed
    4. Guard feature code with if features.your_feature:
    5. Add tests for both enabled/disabled states
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from src.env_parsing import env_bool

# Environment variable prefix for all feature flags
ENV_PREFIX = "ORCHESTRATOR_"


@dataclass
class Features:
    """Feature flags for optional orchestration modules.

    All features default to False for test isolation. Production code should
    use get_features() which reads from environment variables.

    Attributes:
        memrl: Memory-based Reinforcement Learning (Phase 4)
            - TaskEmbedder: BGE-large embeddings for task similarity (1024-dim)
            - QScorer: Q-value scoring for escalation decisions
            - HybridRouter: Learned + rule-based routing
            - EpisodicStore: SQLite storage for task memories
            Dependencies: numpy, sqlite3, sentence-transformers (optional)

        tools: Tool Registry for REPL
            - TOOL() function in REPL environment
            - Role-based permission checking
            - Built-in tools (lint, test, search)
            Dependencies: None (tools are pure Python)

        scripts: Script Registry for prepared scripts
            - SCRIPT() function in REPL
            - Semantic search for script matching
            Dependencies: tools feature (scripts can invoke tools)

        streaming: SSE streaming for chat responses
            - /chat/stream endpoint
            - Server-sent events for incremental output
            Dependencies: None

        openai_compat: OpenAI-compatible API endpoints
            - /v1/chat/completions
            - /v1/models
            Dependencies: None

        repl: REPL execution environment
            - Sandboxed Python execution
            - Context-as-variable pattern
            - peek(), grep(), FINAL() built-ins
            Dependencies: None

        caching: Response caching with prefix routing
            - CachingBackend for LLM responses
            - Prefix-based routing to workers
            Dependencies: None

        restricted_python: Use RestrictedPython for REPL sandboxing
            - More battle-tested security model
            - compile_restricted for safer compilation
            - Built-in guards against attribute access exploits
            Dependencies: RestrictedPython>=7.0
    """

    # Phase 4: MemRL (Memory-based Reinforcement Learning)
    memrl: bool = False

    # Tool and Script Registries
    tools: bool = True  # Enable TOOL() in REPL
    scripts: bool = False  # Scripts require script_registry.yaml to exist

    # API Features
    streaming: bool = False
    openai_compat: bool = False

    # Core Features (usually enabled)
    repl: bool = True
    caching: bool = True

    # Phase 2: Structured Delimiters for tool output isolation
    structured_delimiters: bool = True  # Wrap tool outputs with <<<TOOL_OUTPUT>>> delimiters

    # Phase 2: ReAct-style tool loop (direct mode with tool access)
    react_mode: bool = False  # Enable ReAct tool loop for direct-mode prompts

    # Phase 2: Output formalizer (format constraint enforcement)
    output_formalizer: bool = False  # Post-process answers to satisfy format constraints

    # Parallel read-only tool dispatch (ThreadPoolExecutor for independent REPL tools)
    parallel_tools: bool = True  # Dispatch independent read-only tools in parallel

    # Deferred tool result wrapping (keep mixin tool outputs out of prompt-by-default)
    deferred_tool_results: bool = False

    # Escalation prompt compression (LLMLingua-2 BERT for large prompts)
    escalation_compression: bool = False  # Compress prompts on architect escalation

    # Pre-routing optimization
    script_interception: bool = False  # Resolve trivial queries locally without LLM call

    # Security Features
    credential_redaction: bool = True  # Scan tool/REPL output for leaked credentials
    cascading_tool_policy: bool = True  # Layered tool permission chain (Global→Role→Task)
    restricted_python: bool = False  # Use RestrictedPython for REPL (requires library)

    # Phase 3: Specialist routing (MemRL-driven intelligent orchestration)
    specialist_routing: bool = False  # Enable specialist routing (coder, architect) via Q-values

    # Phase 3+: GraphRouter (GNN-based parallel routing signal for cold-start optimization)
    graph_router: bool = False  # Enable bipartite GAT routing predictor

    # Phase 3: Architect plan review (pre-execution plan vetting)
    plan_review: bool = False  # Enable architect review of frontdoor plans before execution

    # Phase 5: Architect delegation (investigate via specialist tools)
    architect_delegation: bool = False  # Architect delegates tool work to faster specialists

    # Phase 7: Parallel step execution (wave-based dependency ordering)
    parallel_execution: bool = False  # Enable wave-based step execution in ProactiveDelegator

    # Phase 8: Persona registry (dynamic prompt specialization)
    personas: bool = False  # Enable persona-based system prompt overlays

    # Phase 9: Staged reward shaping (PARL-inspired explore→exploit annealing)
    staged_rewards: bool = False  # Anneal exploration bonus in Q-value updates

    # MemRL Distillation: offline-trained routing classifier (ColBERT-Zero inspired)
    routing_classifier: bool = False  # Fast MLP routing before FAISS retrieval

    # SkillBank: Experience distillation into structured skills (SkillRL §3.1)
    skillbank: bool = False  # Enable SkillBank skill retrieval + prompt injection

    # Phase 4: Input formalizer (extract formal specs before specialist execution)
    input_formalizer: bool = False  # Preprocess complex prompts via MathSmith-8B

    # Unified streaming: route streaming endpoint through pipeline stages
    unified_streaming: bool = False  # Use stream_adapter.py instead of inline generator

    # Phase 10: Semantic classifiers (externalized keyword matching + MemRL routing)
    semantic_classifiers: bool = True  # Use config-driven classifiers from classifier_config.yaml

    # Generation Monitoring (Phase 6)
    generation_monitor: bool = True  # Enable early failure detection (post-hoc quality check)

    # OpenClaw/Lobster concept integration
    side_effect_tracking: bool = False  # Declare tool side effects for safety reasoning
    structured_tool_output: bool = False  # ToolOutput envelope (human + machine modes)
    model_fallback: bool = False  # Try same-tier alternatives on circuit-open
    content_cache: bool = False  # SHA-256 keyed response cache for LLM calls
    session_compaction: bool = False  # Summarize old context on long conversations
    session_log: bool = False  # Append-only processing journal across REPL turns
    session_scratchpad: bool = False  # Model-extracted semantic insights from session log
    depth_model_overrides: bool = False  # Map nested llm_call depth to cheaper roles
    resume_tokens: bool = False  # Base64url continuation tokens for crash recovery
    approval_gates: bool = False  # Human approval at escalation boundaries
    binding_routing: bool = False  # Priority-ordered routing overrides

    # LangGraph pre-migration: full state snapshots + generalized interrupts
    state_history_snapshots: bool = False  # Full TaskState snapshots each turn
    generalized_interrupts: bool = False  # Pluggable interrupt conditions before REPL

    # LangGraph migration Phase 1: hybrid bridge (run_task dispatches to LG backend)
    langgraph_bridge: bool = False  # Route orchestration through LangGraph instead of pydantic_graph

    # Budget controls (Fast-RLM)
    worker_call_budget: bool = False  # Cap total REPL executions per task
    task_token_budget: bool = False   # Cap cumulative tokens across all turns

    # Pipeline monitoring: model-graded subjective evals
    model_grading: bool = False  # Post-hoc model-graded evals via worker_explore

    # HSD: Hierarchical Self-Speculation
    self_speculation: bool = False  # Self-speculation with layer-exit draft
    hierarchical_speculation: bool = False  # Hierarchical intermediate verification

    # Two-level condensation (Context-Folding Phase 1)
    two_level_condensation: bool = False  # Granular + deep consolidation instead of per-2-turn re-summarization

    # Context-Folding Phase 1+: segment hash dedup cache
    segment_cache_dedup: bool = False  # Hash-based dedup to skip LLM consolidation on repeated blocks

    # Context-Folding Phase 2c: heuristic helpfulness scoring
    helpfulness_scoring: bool = False  # Score segments by recency/overlap/outcome for compaction priority

    # Context-Folding Phase 3a: process reward telemetry
    process_reward_telemetry: bool = False  # Log token_budget_ratio, on_scope, tool_success per turn

    # Context-Folding Phase 3b: role-aware compaction profiles
    role_aware_compaction: bool = False  # Per-role compaction aggressiveness (architect conservative, worker aggressive)

    # Context window management (C2/C3/C1)
    accurate_token_counting: bool = False  # Use llama-server /tokenize for exact token counts
    tool_result_clearing: bool = False  # Clear stale <<<TOOL_OUTPUT>>> blocks from last_output

    # Reasoning length alarm (short-m@k Action 9): retry with conciseness nudge
    reasoning_length_alarm: bool = False  # Cancel + retry when <think> exceeds 1.5× band budget

    # Tool output compression (Phase 2 native): compress verbose output before prompt injection
    tool_output_compression: bool = False  # Compress pytest/git/build output to preserve actionable info only

    # CMV-style output spill (Action 11): write truncated output/error to temp file with peek() pointer
    output_spill_to_file: bool = False  # Spill long REPL output/error to file + retrieval pointer

    # Conversation Management (B-series cherry-picks from Hermes/OpenGauss)
    injection_scanning: bool = False  # B7: Prompt injection scanning on loaded context
    context_compression: bool = False  # B2: Protected-zone context compression
    user_modeling: bool = False  # B1: Cross-session user preference modeling
    session_token_budget: bool = False  # B5: Per-session token budget with compact/stop signals

    # Claude Code Local Integration (CC Local)
    claude_code_mcp_chat: bool = False  # MCP tools for delegating chat to running orchestrator

    # Debug/Development
    mock_mode: bool = True  # Default to mock mode for safety

    def validate(self) -> list[str]:
        """Validate feature dependencies.

        Returns:
            List of validation errors (empty if all valid).
        """
        errors = []

        # Scripts require tools
        if self.scripts and not self.tools:
            errors.append("scripts feature requires tools feature")

        # MemRL-dependent features
        if self.specialist_routing and not self.memrl:
            errors.append("specialist_routing feature requires memrl feature")
        if self.graph_router and not self.specialist_routing:
            errors.append("graph_router feature requires specialist_routing feature")
        if self.plan_review and not self.memrl:
            errors.append("plan_review feature requires memrl feature")
        if self.architect_delegation and not self.memrl:
            errors.append("architect_delegation feature requires memrl feature")
        if self.parallel_execution and not self.architect_delegation:
            errors.append("parallel_execution feature requires architect_delegation feature")
        if self.personas and not self.memrl:
            errors.append("personas feature requires memrl feature")
        if self.staged_rewards and not self.memrl:
            errors.append("staged_rewards feature requires memrl feature")
        if self.skillbank and not self.memrl:
            errors.append("skillbank feature requires memrl feature")

        # Approval gates require resume tokens and side effect tracking
        if self.approval_gates and not self.resume_tokens:
            errors.append("approval_gates feature requires resume_tokens feature")
        if self.approval_gates and not self.side_effect_tracking:
            errors.append("approval_gates feature requires side_effect_tracking feature")

        # Generalized interrupts depend on approval gates and resume tokens
        if self.generalized_interrupts and not self.approval_gates:
            errors.append("generalized_interrupts requires approval_gates")
        if self.generalized_interrupts and not self.resume_tokens:
            errors.append("generalized_interrupts requires resume_tokens")

        # User modeling requires injection scanning for write safety
        if self.user_modeling and not self.injection_scanning:
            errors.append("user_modeling feature requires injection_scanning feature")

        # RestrictedPython requires the library
        if self.restricted_python:
            try:
                import RestrictedPython  # noqa: F401
            except ImportError:
                errors.append(
                    "restricted_python feature requires RestrictedPython library: "
                    "pip install RestrictedPython>=7.0"
                )

        return errors

    def summary(self) -> dict[str, bool]:
        """Get summary of all feature flags.

        Returns:
            Dictionary of feature name -> enabled status.
        """
        return {
            "memrl": self.memrl,
            "tools": self.tools,
            "scripts": self.scripts,
            "streaming": self.streaming,
            "openai_compat": self.openai_compat,
            "repl": self.repl,
            "caching": self.caching,
            "structured_delimiters": self.structured_delimiters,
            "react_mode": self.react_mode,
            "output_formalizer": self.output_formalizer,
            "parallel_tools": self.parallel_tools,
            "deferred_tool_results": self.deferred_tool_results,
            "escalation_compression": self.escalation_compression,
            "script_interception": self.script_interception,
            "credential_redaction": self.credential_redaction,
            "cascading_tool_policy": self.cascading_tool_policy,
            "restricted_python": self.restricted_python,
            "specialist_routing": self.specialist_routing,
            "graph_router": self.graph_router,
            "plan_review": self.plan_review,
            "architect_delegation": self.architect_delegation,
            "parallel_execution": self.parallel_execution,
            "personas": self.personas,
            "staged_rewards": self.staged_rewards,
            "input_formalizer": self.input_formalizer,
            "generation_monitor": self.generation_monitor,
            "semantic_classifiers": self.semantic_classifiers,
            "unified_streaming": self.unified_streaming,
            "side_effect_tracking": self.side_effect_tracking,
            "structured_tool_output": self.structured_tool_output,
            "model_fallback": self.model_fallback,
            "content_cache": self.content_cache,
            "session_compaction": self.session_compaction,
            "session_log": self.session_log,
            "session_scratchpad": self.session_scratchpad,
            "depth_model_overrides": self.depth_model_overrides,
            "resume_tokens": self.resume_tokens,
            "approval_gates": self.approval_gates,
            "binding_routing": self.binding_routing,
            "routing_classifier": self.routing_classifier,
            "skillbank": self.skillbank,
            "worker_call_budget": self.worker_call_budget,
            "task_token_budget": self.task_token_budget,
            "two_level_condensation": self.two_level_condensation,
            "segment_cache_dedup": self.segment_cache_dedup,
            "helpfulness_scoring": self.helpfulness_scoring,
            "process_reward_telemetry": self.process_reward_telemetry,
            "role_aware_compaction": self.role_aware_compaction,
            "accurate_token_counting": self.accurate_token_counting,
            "tool_result_clearing": self.tool_result_clearing,
            "reasoning_length_alarm": self.reasoning_length_alarm,
            "tool_output_compression": self.tool_output_compression,
            "output_spill_to_file": self.output_spill_to_file,
            "model_grading": self.model_grading,
            "self_speculation": self.self_speculation,
            "hierarchical_speculation": self.hierarchical_speculation,
            "state_history_snapshots": self.state_history_snapshots,
            "generalized_interrupts": self.generalized_interrupts,
            "langgraph_bridge": self.langgraph_bridge,
            "injection_scanning": self.injection_scanning,
            "context_compression": self.context_compression,
            "user_modeling": self.user_modeling,
            "session_token_budget": self.session_token_budget,
            "claude_code_mcp_chat": self.claude_code_mcp_chat,
            "mock_mode": self.mock_mode,
        }

    def enabled_features(self) -> list[str]:
        """Get list of enabled feature names.

        Returns:
            List of enabled feature names.
        """
        return [name for name, enabled in self.summary().items() if enabled]


def _feature_flag_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from environment variable.

    Truthy values: 1, true, yes, on (case-insensitive)
    Falsy values: 0, false, no, off (case-insensitive)

    Args:
        name: Environment variable name (without prefix).
        default: Default value if not set.

    Returns:
        Boolean value.
    """
    key = f"{ENV_PREFIX}{name.upper()}"
    return env_bool(key, default)


def get_features(
    *,
    production: bool = False,
    override: dict[str, bool] | None = None,
) -> Features:
    """Get feature flags from environment variables.

    In production mode (production=True), most features default to enabled.
    In test mode (production=False), most features default to disabled.

    Args:
        production: If True, use production defaults (most features on).
        override: Explicit overrides for specific features.

    Returns:
        Features instance with flags set.

    Example:
        # Read from environment
        features = get_features()

        # Production defaults
        features = get_features(production=True)

        # Test with specific features
        features = get_features(override={"memrl": True, "tools": False})
    """
    # Base defaults depend on production vs test
    if production:
        defaults = {
            "memrl": True,
            "tools": True,
            "scripts": True,
            "streaming": True,
            "openai_compat": True,
            "repl": True,
            "caching": True,
            "structured_delimiters": True,  # Low risk, always on
            "react_mode": True,  # Validated: PASS -36.8s latency (2026-02-20)
            "output_formalizer": True,  # Validated: PASS -21.3s latency (2026-02-20)
            "parallel_tools": True,  # Parallel read-only tool dispatch enabled
            "deferred_tool_results": False,  # Keep legacy wrapping unless explicitly enabled
            "escalation_compression": True,  # BORDERLINE +4.8s but enabled per operator decision (2026-02-20)
            "script_interception": False,  # Enable after validation of interception accuracy
            "credential_redaction": True,  # Safety-first: always redact credentials in production
            "cascading_tool_policy": True,  # Validated: PASS -15.3s latency (2026-02-20)
            "restricted_python": False,  # AST blocklist is sufficient; RestrictedPython blocks all imports including safe ones (scipy, numpy)
            "specialist_routing": True,  # Validated: PASS -25.0s latency (2026-02-20)
            "graph_router": False,  # Enable after GAT training and cold-start validation
            "plan_review": True,  # Validated: PASS -24.8s latency (2026-02-20)
            "architect_delegation": True,  # Validated: PASS -24.9s latency (2026-02-20)
            "parallel_execution": True,  # Validated: PASS -25.5s latency (2026-02-20)
            "personas": False,  # Enable after persona quality validation
            "staged_rewards": False,  # Enable after exploration/exploitation validation
            "routing_classifier": False,  # Enable after classifier training and A/B test
            "skillbank": False,  # Enable after distillation pipeline validation
            "input_formalizer": True,  # Validated: PASS -16.2s latency (2026-02-20)
            "generation_monitor": True,  # Early failure detection in production
            "semantic_classifiers": True,  # Config-driven classifiers enabled by default
            "unified_streaming": True,  # Validated: PASS -7.9s latency (2026-02-20)
            "side_effect_tracking": True,  # Validated: PASS -28.3s latency (2026-02-20)
            "structured_tool_output": True,  # Validated: PASS -8.1s latency (2026-02-20)
            "model_fallback": True,  # Validated: PASS -1.5s latency (2026-02-20)
            "content_cache": False,  # Enable after cache correctness validation
            "session_compaction": True,  # Low-risk default: compacts long contexts with clear rollback toggle
            "session_log": True,  # Append-only REPL session journal for multi-turn context
            "session_scratchpad": True,  # Model-extracted semantic insights from session log
            "depth_model_overrides": True,  # Enabled with worker-only + max-depth guardrails
            "resume_tokens": True,  # Validated: PASS -1.1s latency (2026-02-20)
            "approval_gates": True,  # Validated: PASS -20.6s latency (2026-02-20)
            "binding_routing": False,  # Enable after routing regression testing
            "worker_call_budget": True,  # Fast-RLM: cap total REPL executions per task
            "task_token_budget": True,  # Fast-RLM: cap cumulative tokens per task
            "two_level_condensation": False,  # CF Phase 1: enable after quality validation
            "segment_cache_dedup": False,  # CF Phase 1+: enable after two_level_condensation validated
            "helpfulness_scoring": False,  # CF Phase 2c: enable after calibration
            "process_reward_telemetry": False,  # CF Phase 3a: enable after two_level_condensation validated
            "role_aware_compaction": False,  # CF Phase 3b: enable after helpfulness_scoring validated
            "accurate_token_counting": False,  # Enable after /tokenize validation
            "tool_result_clearing": True,  # Enabled for production context pressure relief
            "reasoning_length_alarm": True,  # short-m@k Action 9: retry verbose reasoning
            "tool_output_compression": False,  # Phase 2 native: enable after quality validation
            "output_spill_to_file": True,  # CMV Action 11: spill truncated output/error to file
            "model_grading": False,  # Enable after grading spec validation
            "self_speculation": False,  # HSD: self-speculation with layer-exit draft
            "hierarchical_speculation": False,  # HSD: hierarchical intermediate verification
            "state_history_snapshots": False,  # LangGraph pre-migration: full state snapshots
            "generalized_interrupts": False,  # LangGraph pre-migration: pluggable interrupts
            "langgraph_bridge": False,  # LangGraph Phase 1: off until validated
            "injection_scanning": True,  # B7: low risk, always scan loaded context
            "context_compression": False,  # B2: enable after protected-zone quality validation
            "user_modeling": False,  # B1: enable after profile store + deriver validation
            "session_token_budget": False,  # B5: enable after budget threshold tuning
            "claude_code_mcp_chat": False,  # CC Local: enable after MCP tool validation
            "mock_mode": False,  # Real mode in production
        }
    else:
        defaults = {
            "memrl": False,
            "tools": False,
            "scripts": False,
            "streaming": False,
            "openai_compat": False,
            "repl": True,  # REPL is core functionality
            "caching": False,
            "structured_delimiters": True,  # Low risk, always on
            "react_mode": False,
            "output_formalizer": False,
            "parallel_tools": True,  # Parallel read-only tool dispatch enabled
            "deferred_tool_results": False,
            "escalation_compression": False,  # Disabled in tests by default
            "script_interception": False,  # Disabled in tests by default
            "credential_redaction": True,  # Safety-first: always redact in tests too
            "cascading_tool_policy": False,  # Disabled in tests by default
            "restricted_python": False,  # Use custom sandbox in tests
            "specialist_routing": False,  # Disabled in tests by default
            "graph_router": False,  # Disabled in tests by default
            "plan_review": False,  # Disabled in tests by default
            "architect_delegation": False,  # Disabled in tests by default
            "parallel_execution": False,  # Disabled in tests by default
            "personas": False,  # Disabled in tests by default
            "staged_rewards": False,  # Disabled in tests by default
            "routing_classifier": False,  # Disabled in tests by default
            "skillbank": False,  # Disabled in tests by default
            "input_formalizer": False,  # Disabled in tests by default
            "generation_monitor": False,  # Disabled in tests by default
            "semantic_classifiers": True,  # Config-driven classifiers enabled by default
            "unified_streaming": False,  # Disabled in tests by default
            "side_effect_tracking": False,  # Disabled in tests by default
            "structured_tool_output": False,  # Disabled in tests by default
            "model_fallback": False,  # Disabled in tests by default
            "content_cache": False,  # Disabled in tests by default
            "session_compaction": False,  # Disabled in tests by default
            "session_log": False,  # Disabled in tests by default
            "session_scratchpad": False,  # Disabled in tests by default
            "depth_model_overrides": False,  # Disabled in tests by default
            "resume_tokens": False,  # Disabled in tests by default
            "approval_gates": False,  # Disabled in tests by default
            "binding_routing": False,  # Disabled in tests by default
            "worker_call_budget": False,  # Disabled in tests by default
            "task_token_budget": False,  # Disabled in tests by default
            "two_level_condensation": False,  # CF Phase 1: off in tests
            "segment_cache_dedup": False,  # CF Phase 1+: off in tests
            "helpfulness_scoring": False,  # CF Phase 2c: off in tests
            "process_reward_telemetry": False,  # CF Phase 3a: off in tests
            "role_aware_compaction": False,  # CF Phase 3b: off in tests
            "accurate_token_counting": False,  # Disabled in tests by default
            "tool_result_clearing": False,  # Disabled in tests by default
            "reasoning_length_alarm": False,  # Disabled in tests by default
            "tool_output_compression": False,  # Disabled in tests by default
            "output_spill_to_file": False,  # Disabled in tests by default
            "model_grading": False,  # Disabled in tests by default
            "self_speculation": False,  # HSD: self-speculation with layer-exit draft
            "hierarchical_speculation": False,  # HSD: hierarchical intermediate verification
            "state_history_snapshots": False,  # LangGraph pre-migration: off in tests
            "generalized_interrupts": False,  # LangGraph pre-migration: off in tests
            "langgraph_bridge": False,  # LangGraph Phase 1: off in tests
            "injection_scanning": False,  # B7: off in tests
            "context_compression": False,  # B2: off in tests
            "user_modeling": False,  # B1: off in tests
            "session_token_budget": False,  # B5: off in tests
            "claude_code_mcp_chat": False,  # CC Local: off in tests
            "mock_mode": True,  # Mock mode in tests
        }

    # Read from environment (overrides defaults)
    flags = {
        "memrl": _feature_flag_bool("MEMRL", defaults["memrl"]),
        "tools": _feature_flag_bool("TOOLS", defaults["tools"]),
        "scripts": _feature_flag_bool("SCRIPTS", defaults["scripts"]),
        "streaming": _feature_flag_bool("STREAMING", defaults["streaming"]),
        "openai_compat": _feature_flag_bool("OPENAI_COMPAT", defaults["openai_compat"]),
        "repl": _feature_flag_bool("REPL", defaults["repl"]),
        "caching": _feature_flag_bool("CACHING", defaults["caching"]),
        "structured_delimiters": _feature_flag_bool(
            "STRUCTURED_DELIMITERS", defaults["structured_delimiters"]
        ),
        "react_mode": _feature_flag_bool("REACT_MODE", defaults["react_mode"]),
        "output_formalizer": _feature_flag_bool("OUTPUT_FORMALIZER", defaults["output_formalizer"]),
        "parallel_tools": _feature_flag_bool("PARALLEL_TOOLS", defaults["parallel_tools"]),
        "deferred_tool_results": _feature_flag_bool(
            "DEFERRED_TOOL_RESULTS", defaults["deferred_tool_results"]
        ),
        "escalation_compression": _feature_flag_bool("ESCALATION_COMPRESSION", defaults["escalation_compression"]),
        "script_interception": _feature_flag_bool("SCRIPT_INTERCEPTION", defaults["script_interception"]),
        "credential_redaction": _feature_flag_bool("CREDENTIAL_REDACTION", defaults["credential_redaction"]),
        "cascading_tool_policy": _feature_flag_bool("CASCADING_TOOL_POLICY", defaults["cascading_tool_policy"]),
        "restricted_python": _feature_flag_bool("RESTRICTED_PYTHON", defaults["restricted_python"]),
        "specialist_routing": _feature_flag_bool("SPECIALIST_ROUTING", defaults["specialist_routing"]),
        "graph_router": _feature_flag_bool("GRAPH_ROUTER", defaults["graph_router"]),
        "plan_review": _feature_flag_bool("PLAN_REVIEW", defaults["plan_review"]),
        "architect_delegation": _feature_flag_bool("ARCHITECT_DELEGATION", defaults["architect_delegation"]),
        "parallel_execution": _feature_flag_bool("PARALLEL_EXECUTION", defaults["parallel_execution"]),
        "personas": _feature_flag_bool("PERSONAS", defaults["personas"]),
        "staged_rewards": _feature_flag_bool("STAGED_REWARDS", defaults["staged_rewards"]),
        "routing_classifier": _feature_flag_bool("ROUTING_CLASSIFIER", defaults["routing_classifier"]),
        "skillbank": _feature_flag_bool("SKILLBANK", defaults["skillbank"]),
        "input_formalizer": _feature_flag_bool("INPUT_FORMALIZER", defaults["input_formalizer"]),
        "generation_monitor": _feature_flag_bool("GENERATION_MONITOR", defaults["generation_monitor"]),
        "semantic_classifiers": _feature_flag_bool("SEMANTIC_CLASSIFIERS", defaults["semantic_classifiers"]),
        "unified_streaming": _feature_flag_bool("UNIFIED_STREAMING", defaults["unified_streaming"]),
        "side_effect_tracking": _feature_flag_bool("SIDE_EFFECT_TRACKING", defaults["side_effect_tracking"]),
        "structured_tool_output": _feature_flag_bool(
            "STRUCTURED_TOOL_OUTPUT", defaults["structured_tool_output"]
        ),
        "model_fallback": _feature_flag_bool("MODEL_FALLBACK", defaults["model_fallback"]),
        "content_cache": _feature_flag_bool("CONTENT_CACHE", defaults["content_cache"]),
        "session_compaction": _feature_flag_bool("SESSION_COMPACTION", defaults["session_compaction"]),
        "session_log": _feature_flag_bool("SESSION_LOG", defaults["session_log"]),
        "session_scratchpad": _feature_flag_bool("SESSION_SCRATCHPAD", defaults["session_scratchpad"]),
        "depth_model_overrides": _feature_flag_bool(
            "DEPTH_MODEL_OVERRIDES", defaults["depth_model_overrides"]
        ),
        "resume_tokens": _feature_flag_bool("RESUME_TOKENS", defaults["resume_tokens"]),
        "approval_gates": _feature_flag_bool("APPROVAL_GATES", defaults["approval_gates"]),
        "binding_routing": _feature_flag_bool("BINDING_ROUTING", defaults["binding_routing"]),
        "worker_call_budget": _feature_flag_bool("WORKER_CALL_BUDGET", defaults["worker_call_budget"]),
        "task_token_budget": _feature_flag_bool("TASK_TOKEN_BUDGET", defaults["task_token_budget"]),
        "two_level_condensation": _feature_flag_bool("TWO_LEVEL_CONDENSATION", defaults["two_level_condensation"]),
        "segment_cache_dedup": _feature_flag_bool("SEGMENT_CACHE_DEDUP", defaults["segment_cache_dedup"]),
        "helpfulness_scoring": _feature_flag_bool("HELPFULNESS_SCORING", defaults["helpfulness_scoring"]),
        "process_reward_telemetry": _feature_flag_bool("PROCESS_REWARD_TELEMETRY", defaults["process_reward_telemetry"]),
        "role_aware_compaction": _feature_flag_bool("ROLE_AWARE_COMPACTION", defaults["role_aware_compaction"]),
        "accurate_token_counting": _feature_flag_bool("ACCURATE_TOKEN_COUNTING", defaults["accurate_token_counting"]),
        "tool_result_clearing": _feature_flag_bool("TOOL_RESULT_CLEARING", defaults["tool_result_clearing"]),
        "reasoning_length_alarm": _feature_flag_bool("REASONING_LENGTH_ALARM", defaults["reasoning_length_alarm"]),
        "tool_output_compression": _feature_flag_bool("TOOL_OUTPUT_COMPRESSION", defaults["tool_output_compression"]),
        "output_spill_to_file": _feature_flag_bool("OUTPUT_SPILL_TO_FILE", defaults["output_spill_to_file"]),
        "model_grading": _feature_flag_bool("MODEL_GRADING", defaults["model_grading"]),
        "self_speculation": _feature_flag_bool("SELF_SPECULATION", defaults["self_speculation"]),
        "hierarchical_speculation": _feature_flag_bool("HIERARCHICAL_SPECULATION", defaults["hierarchical_speculation"]),
        "state_history_snapshots": _feature_flag_bool("STATE_HISTORY_SNAPSHOTS", defaults["state_history_snapshots"]),
        "generalized_interrupts": _feature_flag_bool("GENERALIZED_INTERRUPTS", defaults["generalized_interrupts"]),
        "langgraph_bridge": _feature_flag_bool("LANGGRAPH_BRIDGE", defaults["langgraph_bridge"]),
        "injection_scanning": _feature_flag_bool("INJECTION_SCANNING", defaults["injection_scanning"]),
        "context_compression": _feature_flag_bool("CONTEXT_COMPRESSION", defaults["context_compression"]),
        "user_modeling": _feature_flag_bool("USER_MODELING", defaults["user_modeling"]),
        "session_token_budget": _feature_flag_bool("SESSION_TOKEN_BUDGET", defaults["session_token_budget"]),
        "claude_code_mcp_chat": _feature_flag_bool("CLAUDE_CODE_MCP_CHAT", defaults["claude_code_mcp_chat"]),
        "mock_mode": _feature_flag_bool("MOCK_MODE", defaults["mock_mode"]),
    }

    # Apply explicit overrides
    if override:
        flags.update(override)

    return Features(**flags)


# Singleton for global access (lazy-loaded, thread-safe)
_features: Features | None = None
_features_lock = threading.Lock()


def features() -> Features:
    """Get the global Features instance (lazy-loaded from environment).

    Thread-safe via double-checked locking (matches PromptCompressor,
    WorkerPoolManager patterns).

    For most code, use this function:
        from src.features import features
        if features().memrl:
            ...

    Returns:
        Global Features instance.
    """
    global _features
    if _features is None:
        with _features_lock:
            if _features is None:
                _features = get_features()
    return _features


def reset_features() -> None:
    """Reset the global Features instance (useful for tests).

    Call this to re-read feature flags from environment.
    """
    global _features
    with _features_lock:
        _features = None


def set_features(new_features: Features) -> None:
    """Set the global Features instance (useful for tests).

    Args:
        new_features: Features instance to use globally.
    """
    global _features
    with _features_lock:
        _features = new_features
