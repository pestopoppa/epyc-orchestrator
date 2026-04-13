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
from dataclasses import dataclass, field

from src.env_parsing import env_bool

# Environment variable prefix for all feature flags
ENV_PREFIX = "ORCHESTRATOR_"


# ── Declarative Feature Registry ──────────────────────────────────────────
# Single source of truth for feature metadata. Drives summary(), get_features()
# defaults, and env parsing. The Features dataclass fields must stay in sync
# (validated by test_features_registry_consistency).


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative specification for a single feature flag."""
    name: str
    default_test: bool
    default_prod: bool
    env_var: str  # env var suffix (without ORCHESTRATOR_ prefix)
    description: str = ""
    dependencies: tuple[str, ...] = ()


_FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    # Phase 4: MemRL
    FeatureSpec("memrl", False, True, "MEMRL", "Memory-based Reinforcement Learning"),
    # Tool and Script Registries
    FeatureSpec("tools", False, True, "TOOLS", "Tool Registry for REPL"),
    FeatureSpec("scripts", False, True, "SCRIPTS", "Script Registry", ("tools",)),
    # API Features
    FeatureSpec("streaming", False, True, "STREAMING", "SSE streaming endpoints"),
    FeatureSpec("openai_compat", False, True, "OPENAI_COMPAT", "OpenAI-compatible API"),
    # Core Features
    FeatureSpec("repl", True, True, "REPL", "REPL execution environment"),
    FeatureSpec("caching", False, True, "CACHING", "Response caching with prefix routing"),
    # Phase 2: Structured Delimiters
    FeatureSpec("structured_delimiters", True, True, "STRUCTURED_DELIMITERS", "Wrap tool outputs with delimiters"),
    FeatureSpec("react_mode", False, True, "REACT_MODE", "ReAct-style tool loop"),
    FeatureSpec("output_formalizer", False, True, "OUTPUT_FORMALIZER", "Format constraint enforcement"),
    FeatureSpec("parallel_tools", True, True, "PARALLEL_TOOLS", "Parallel read-only tool dispatch"),
    FeatureSpec("deferred_tool_results", False, False, "DEFERRED_TOOL_RESULTS", "Deferred tool result wrapping"),
    FeatureSpec("escalation_compression", False, True, "ESCALATION_COMPRESSION", "LLMLingua-2 BERT for large prompts"),
    FeatureSpec("script_interception", False, False, "SCRIPT_INTERCEPTION", "Resolve trivial queries locally"),
    # Security Features
    FeatureSpec("credential_redaction", True, True, "CREDENTIAL_REDACTION", "Scan for leaked credentials"),
    FeatureSpec("cascading_tool_policy", False, True, "CASCADING_TOOL_POLICY", "Layered tool permission chain"),
    FeatureSpec("restricted_python", False, False, "RESTRICTED_PYTHON", "RestrictedPython for REPL"),
    # Phase 3: Specialist routing
    FeatureSpec("specialist_routing", False, True, "SPECIALIST_ROUTING", "Q-value specialist routing", ("memrl",)),
    FeatureSpec("graph_router", False, False, "GRAPH_ROUTER", "GNN-based parallel routing", ("specialist_routing",)),
    FeatureSpec("plan_review", False, True, "PLAN_REVIEW", "Architect plan review", ("memrl",)),
    FeatureSpec("architect_delegation", False, True, "ARCHITECT_DELEGATION", "Architect delegation", ("memrl",)),
    FeatureSpec("parallel_execution", False, True, "PARALLEL_EXECUTION", "Wave-based step execution", ("architect_delegation",)),
    FeatureSpec("personas", False, False, "PERSONAS", "Persona registry", ("memrl",)),
    FeatureSpec("staged_rewards", False, False, "STAGED_REWARDS", "PARL-inspired annealing", ("memrl",)),
    # MemRL Distillation
    FeatureSpec("routing_classifier", False, False, "ROUTING_CLASSIFIER", "ColBERT-Zero routing classifier"),
    FeatureSpec("skillbank", False, False, "SKILLBANK", "SkillRL experience distillation", ("memrl",)),
    # Phase 4: Input formalizer
    FeatureSpec("input_formalizer", False, True, "INPUT_FORMALIZER", "MathSmith-8B formal spec extraction"),
    # Unified streaming
    FeatureSpec("unified_streaming", False, True, "UNIFIED_STREAMING", "Route streaming through pipeline stages"),
    # Semantic classifiers
    FeatureSpec("semantic_classifiers", True, True, "SEMANTIC_CLASSIFIERS", "Config-driven classifiers"),
    # Generation Monitoring
    FeatureSpec("generation_monitor", False, True, "GENERATION_MONITOR", "Early failure detection"),
    # OpenClaw/Lobster concepts
    FeatureSpec("side_effect_tracking", False, True, "SIDE_EFFECT_TRACKING", "Tool side effect declarations"),
    FeatureSpec("structured_tool_output", False, True, "STRUCTURED_TOOL_OUTPUT", "ToolOutput envelope"),
    FeatureSpec("model_fallback", False, True, "MODEL_FALLBACK", "Same-tier alternatives on circuit-open"),
    FeatureSpec("content_cache", False, False, "CONTENT_CACHE", "SHA-256 keyed response cache"),
    FeatureSpec("session_compaction", False, True, "SESSION_COMPACTION", "Summarize old context"),
    FeatureSpec("session_log", False, True, "SESSION_LOG", "Append-only processing journal"),
    FeatureSpec("session_scratchpad", False, True, "SESSION_SCRATCHPAD", "Model-extracted semantic insights"),
    FeatureSpec("depth_model_overrides", False, True, "DEPTH_MODEL_OVERRIDES", "Map nested depth to cheaper roles"),
    FeatureSpec("resume_tokens", False, True, "RESUME_TOKENS", "Base64url continuation tokens"),
    FeatureSpec("approval_gates", False, True, "APPROVAL_GATES", "Human approval at escalation boundaries", ("resume_tokens", "side_effect_tracking")),
    FeatureSpec("binding_routing", False, False, "BINDING_ROUTING", "Priority-ordered routing overrides"),
    # Budget controls
    FeatureSpec("worker_call_budget", False, True, "WORKER_CALL_BUDGET", "Cap total REPL executions per task"),
    FeatureSpec("task_token_budget", False, True, "TASK_TOKEN_BUDGET", "Cap cumulative tokens per task"),
    # Context-Folding
    FeatureSpec("two_level_condensation", False, False, "TWO_LEVEL_CONDENSATION", "CF Phase 1: granular + deep consolidation"),
    FeatureSpec("segment_cache_dedup", False, False, "SEGMENT_CACHE_DEDUP", "CF Phase 1+: hash-based dedup"),
    FeatureSpec("helpfulness_scoring", False, False, "HELPFULNESS_SCORING", "CF Phase 2c: heuristic helpfulness scoring"),
    FeatureSpec("process_reward_telemetry", False, False, "PROCESS_REWARD_TELEMETRY", "CF Phase 3a: process reward telemetry"),
    FeatureSpec("role_aware_compaction", False, False, "ROLE_AWARE_COMPACTION", "CF Phase 3b: role-aware profiles"),
    # Context window management
    FeatureSpec("accurate_token_counting", False, False, "ACCURATE_TOKEN_COUNTING", "Use llama-server /tokenize"),
    FeatureSpec("tool_result_clearing", False, True, "TOOL_RESULT_CLEARING", "Clear stale tool output blocks"),
    # Reasoning length alarm
    FeatureSpec("reasoning_length_alarm", False, True, "REASONING_LENGTH_ALARM", "Retry verbose reasoning"),
    # Tool output compression
    FeatureSpec("tool_output_compression", False, False, "TOOL_OUTPUT_COMPRESSION", "Compress verbose output"),
    # Output spill
    FeatureSpec("output_spill_to_file", False, True, "OUTPUT_SPILL_TO_FILE", "Spill truncated output to file"),
    # Pipeline monitoring
    FeatureSpec("model_grading", False, False, "MODEL_GRADING", "Post-hoc model-graded evals"),
    # HSD
    FeatureSpec("self_speculation", False, False, "SELF_SPECULATION", "Self-speculation with layer-exit draft"),
    FeatureSpec("hierarchical_speculation", False, False, "HIERARCHICAL_SPECULATION", "Hierarchical intermediate verification"),
    # LangGraph pre-migration
    FeatureSpec("state_history_snapshots", False, False, "STATE_HISTORY_SNAPSHOTS", "Full TaskState snapshots each turn"),
    FeatureSpec("generalized_interrupts", False, False, "GENERALIZED_INTERRUPTS", "Pluggable interrupt conditions", ("approval_gates", "resume_tokens")),
    # LangGraph migration
    FeatureSpec("langgraph_bridge", False, False, "LANGGRAPH_BRIDGE", "LangGraph Phase 1: hybrid bridge"),
    FeatureSpec("langgraph_ingest", False, False, "LANGGRAPH_INGEST", "LangGraph Phase 3: IngestNode"),
    FeatureSpec("langgraph_architect", False, False, "LANGGRAPH_ARCHITECT", "LangGraph Phase 3: ArchitectNode"),
    FeatureSpec("langgraph_architect_coding", False, False, "LANGGRAPH_ARCHITECT_CODING", "LangGraph Phase 3: ArchitectCodingNode"),
    FeatureSpec("langgraph_worker", False, False, "LANGGRAPH_WORKER", "LangGraph Phase 3: WorkerNode"),
    FeatureSpec("langgraph_frontdoor", False, False, "LANGGRAPH_FRONTDOOR", "LangGraph Phase 3: FrontdoorNode"),
    FeatureSpec("langgraph_coder", False, False, "LANGGRAPH_CODER", "LangGraph Phase 3: CoderNode"),
    FeatureSpec("langgraph_coder_escalation", False, False, "LANGGRAPH_CODER_ESCALATION", "LangGraph Phase 3: CoderEscalationNode"),
    # Conversation Management
    FeatureSpec("injection_scanning", False, True, "INJECTION_SCANNING", "B7: prompt injection scanning"),
    FeatureSpec("context_compression", False, False, "CONTEXT_COMPRESSION", "B2: protected-zone compression"),
    FeatureSpec("user_modeling", False, False, "USER_MODELING", "B1: cross-session user preferences", ("injection_scanning",)),
    FeatureSpec("session_token_budget", False, False, "SESSION_TOKEN_BUDGET", "B5: per-session token budget"),
    # Claude Code Local
    FeatureSpec("claude_code_mcp_chat", False, False, "CLAUDE_CODE_MCP_CHAT", "CC Local: MCP chat delegation"),
    # Debug/Development
    FeatureSpec("mock_mode", True, False, "MOCK_MODE", "Mock mode for safety"),
)

# Indexed for fast lookup
_REGISTRY_BY_NAME: dict[str, FeatureSpec] = {s.name: s for s in _FEATURE_REGISTRY}


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

    # LangGraph migration Phase 3: per-node migration flags
    langgraph_ingest: bool = False             # Migrate IngestNode to LangGraph backend
    langgraph_architect: bool = False          # Migrate ArchitectNode to LangGraph backend
    langgraph_architect_coding: bool = False   # Migrate ArchitectCodingNode to LangGraph backend
    langgraph_worker: bool = False             # Migrate WorkerNode to LangGraph backend
    langgraph_frontdoor: bool = False          # Migrate FrontdoorNode to LangGraph backend
    langgraph_coder: bool = False              # Migrate CoderNode to LangGraph backend
    langgraph_coder_escalation: bool = False   # Migrate CoderEscalationNode to LangGraph backend

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
        """Get summary of all feature flags (derived from registry).

        Returns:
            Dictionary of feature name -> enabled status.
        """
        return {spec.name: getattr(self, spec.name) for spec in _FEATURE_REGISTRY}

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

    Defaults, env-var names, and flag inventory are driven by ``_FEATURE_REGISTRY``
    — the single source of truth. Adding a new flag only requires a new
    ``FeatureSpec`` entry (plus the matching dataclass field on ``Features``).

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
    # Derive defaults from registry
    defaults = {
        spec.name: (spec.default_prod if production else spec.default_test)
        for spec in _FEATURE_REGISTRY
    }

    # Read from environment (overrides defaults)
    flags = {
        spec.name: _feature_flag_bool(spec.env_var, defaults[spec.name])
        for spec in _FEATURE_REGISTRY
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
