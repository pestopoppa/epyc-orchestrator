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

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.env_parsing import env_bool

# Environment variable prefixes for feature flags.
#
# ORCHESTRATOR_<FLAG> is the legacy spelling and remains supported for
# compatibility. ORCHESTRATOR_FEATURE_<FLAG> is preferred for stack-managed
# API workers because top-level ORCHESTRATOR_* names can collide with
# structured Pydantic settings such as OrchestratorSettings.repl.
ENV_PREFIX = "ORCHESTRATOR_"
FEATURE_ENV_PREFIX = "ORCHESTRATOR_FEATURE_"
RUNTIME_FLAGS_ENV = "ORCHESTRATOR_RUNTIME_FLAGS_PATH"
RUNTIME_FLAGS_TTL_S = 1.0
# A recovery lifetime for experiment enables, not a measurement threshold.
EXPERIMENT_FLAG_TTL_S = 6 * 60 * 60
logger = logging.getLogger(__name__)


# ── Declarative Feature Registry ──────────────────────────────────────────
# Single source of truth for feature metadata. Drives summary(), get_features()
# defaults, and env parsing. The Features dataclass fields must stay in sync
# (validated by test_features_registry_consistency).


def _env_in(*values: str) -> Callable[[str], bool]:
    """``env_parse`` helper: True when the (stripped, lower-cased) env value is one of ``values``."""
    wanted = frozenset(values)
    return lambda raw: raw.strip().lower() in wanted


def _env_not_in(*values: str) -> Callable[[str], bool]:
    """``env_parse`` helper: False when the (stripped, lower-cased) env value is one of ``values``."""
    blocked = frozenset(values)
    return lambda raw: raw.strip().lower() not in blocked


def _env_positive_int(raw: str) -> bool:
    try:
        return int(raw.strip()) > 0
    except ValueError:
        return False


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative specification for a single feature flag."""
    name: str
    default_test: bool
    default_prod: bool
    env_var: str  # env var suffix (without ORCHESTRATOR_ prefix)
    description: str = ""
    dependencies: tuple[str, ...] = ()
    # Mode-valued env vars (flag-surface, 2026-11-18). Some pre-existing ORCHESTRATOR_* knobs are read
    # straight from os.environ and carry a MODE string (``ORCHESTRATOR_TOOL_SELECTOR=embed``), not a
    # bool. ``env_parse`` turns the raw env string into this flag's bool (default: the 1/0/true/false
    # parser). ``env_on`` / ``env_off`` are the env values the code path sees when a RUNTIME override
    # (``POST /config``) turns the flag on / off; ``env_on`` is used only when the env does not already
    # say "on" (so ``=oneshot`` survives a runtime ON). Empty ``env_on`` and ``env_off`` = the flag is
    # read via ``features()`` and has no env overlay. See ``src/flag_env.py``.
    env_parse: Callable[[str], bool] | None = None
    env_on: str = ""
    env_off: str = ""
    # Full env var name when it is not ORCHESTRATOR_<env_var> (two flags sharing one mode var).
    env_name: str = ""


_FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    # Phase 4: MemRL
    FeatureSpec("memrl", False, True, "MEMRL", "Memory-based Reinforcement Learning"),
    # Tool and Script Registries
    FeatureSpec("tools", False, True, "TOOLS", "Tool Registry for REPL"),
    FeatureSpec("scripts", False, True, "SCRIPTS", "Script Registry", ("tools",)),
    # API Features
    FeatureSpec("streaming", False, True, "STREAMING", "SSE streaming endpoints"),
    FeatureSpec("openai_compat", False, True, "OPENAI_COMPAT", "OpenAI-compatible API"),
    FeatureSpec("voice_turn", False, False, "VOICE_TURN", "First-party session-backed voice-turn SSE route; opt-in until integrated acceptance"),
    # Live speech text front end (CS-8.L4, operator 2026-10-23): Italian only, kill-switch
    # ORCHESTRATOR_TTS_IT_NORMALIZE=0 / ORCHESTRATOR_TTS_IT_RESPELL=0.
    FeatureSpec("tts_it_normalize", True, True, "TTS_IT_NORMALIZE", "Italian TTS text normalizer (numbers, units, pH) on the live voice path"),
    FeatureSpec("tts_it_respell", True, True, "TTS_IT_RESPELL", "Italian TTS stress/proper-noun respelling on the live voice path"),
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
    FeatureSpec("agent_file_injection", False, False, "AGENT_FILE_INJECTION", "Append role agent file (compression operating point) to role prompts"),
    FeatureSpec("restricted_python", False, False, "RESTRICTED_PYTHON", "RestrictedPython for REPL"),
    # Phase 3: Specialist routing
    FeatureSpec("specialist_routing", False, True, "SPECIALIST_ROUTING", "Q-value specialist routing", ("memrl",)),
    FeatureSpec("graph_router", False, False, "GRAPH_ROUTER", "GNN-based parallel routing", ("specialist_routing",)),
    FeatureSpec("ingest_triviality_guard", False, False, "INGEST_TRIVIALITY_GUARD", "Demote trivially-easy short prompts off ingest_long_context", ("specialist_routing",)),
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
    FeatureSpec("observed_dependency_logging", False, False, "OBSERVED_DEPENDENCY_LOGGING", "HIL-15: log resource reads/writes per tool call (observed grade)"),
    FeatureSpec("step_event_logging", False, False, "STEP_EVENT_LOGGING", "HIL-13a: per-step tool events (hash, ok/error, latency, size, truncation, mutating) for the L2 scan"),
    FeatureSpec("replay_recording", False, False, "REPLAY_RECORDING", "HIL-16: record per-turn model input/output and tool outputs content-addressed for the prefix-preserving replay tier (implies step events)"),
    FeatureSpec("exec_containment", True, True, "EXEC_CONTAINMENT", "SEC-EXEC-3/5: run_shell/run_python_code children run under Landlock+seccomp (no sockets, credential-file read-deny, foreign /proc hidden); fail-closed when the host cannot enforce"),
    FeatureSpec("web_fetch_guard", False, False, "WEB_FETCH_GUARD", "SEC-EXEC-3: web_fetch/fetch_docs deny private, loopback, link-local and metadata destinations (redirect- and DNS-rebinding-safe)"),
    FeatureSpec("exec_write_flags", False, False, "EXEC_WRITE_FLAGS", "SEC-EXEC-7: audit records carry write_outside_root (child audit hook for run_python_code, static check for run_shell)"),
    FeatureSpec("model_fallback", False, True, "MODEL_FALLBACK", "Same-tier alternatives on circuit-open"),
    FeatureSpec("content_cache", False, False, "CONTENT_CACHE", "SHA-256 keyed response cache"),
    FeatureSpec("session_compaction", False, True, "SESSION_COMPACTION", "Summarize old context"),
    FeatureSpec("session_log", False, True, "SESSION_LOG", "Append-only processing journal"),
    FeatureSpec("session_log_verbatim", False, False, "SESSION_LOG_VERBATIM", "Verbatim experiment history"),
    FeatureSpec("context_arm_ab", False, False, "CONTEXT_ARM_AB", "Context compression experiment arms", ("session_log_verbatim",)),
    FeatureSpec("session_scratchpad", False, True, "SESSION_SCRATCHPAD", "Model-extracted semantic insights"),
    FeatureSpec("depth_model_overrides", False, True, "DEPTH_MODEL_OVERRIDES", "Map nested depth to cheaper roles"),
    FeatureSpec("resume_tokens", False, True, "RESUME_TOKENS", "Base64url continuation tokens"),
    FeatureSpec("approval_gates", False, True, "APPROVAL_GATES", "Human approval at escalation boundaries", ("resume_tokens", "side_effect_tracking")),
    FeatureSpec("binding_routing", False, False, "BINDING_ROUTING", "Priority-ordered routing overrides"),
    # Budget controls
    FeatureSpec("worker_call_budget", False, True, "WORKER_CALL_BUDGET", "Cap total REPL executions per task"),
    FeatureSpec("task_token_budget", False, True, "TASK_TOKEN_BUDGET", "Cap cumulative tokens per task"),
    FeatureSpec("final_schema_validation", False, False, "FINAL_SCHEMA_VALIDATION", "Validate FINAL() value against caller-supplied JSON Schema; retry-with-error on failure"),
    # Context-Folding
    FeatureSpec("two_level_condensation", False, False, "TWO_LEVEL_CONDENSATION", "CF Phase 1: granular + deep consolidation"),
    FeatureSpec("segment_cache_dedup", False, False, "SEGMENT_CACHE_DEDUP", "CF Phase 1+: hash-based dedup"),
    FeatureSpec("helpfulness_scoring", False, False, "HELPFULNESS_SCORING", "CF Phase 2c: heuristic helpfulness scoring"),
    FeatureSpec("process_reward_telemetry", False, False, "PROCESS_REWARD_TELEMETRY", "CF Phase 3a: process reward telemetry"),
    FeatureSpec("role_aware_compaction", False, False, "ROLE_AWARE_COMPACTION", "CF Phase 3b: role-aware profiles"),
    FeatureSpec("compaction_shadow", False, False, "COMPACTION_SHADOW", "RTG-08 option C: non-mutating compaction shadow; computes role-aware compaction + helpfulness and creates the quality monitor but never writes seg.consolidated. Still requires two_level_condensation to be reached"),
    # Context window management
    FeatureSpec("accurate_token_counting", False, False, "ACCURATE_TOKEN_COUNTING", "Use llama-server /tokenize"),
    FeatureSpec("tool_result_clearing", False, True, "TOOL_RESULT_CLEARING", "Clear stale tool output blocks"),
    # Reasoning length alarm
    FeatureSpec("reasoning_length_alarm", False, True, "REASONING_LENGTH_ALARM", "Retry verbose reasoning"),
    # MF-RR-1: one retry after a native length cutoff only when no executable
    # action/tool call was emitted and the selected backend can honor thinking-off.
    FeatureSpec("reasoning_runaway_retry", False, False, "REASONING_RUNAWAY_RETRY", "MF-RR-1: one concise thinking-off retry after a native length cutoff with no executable action; default off"),
    # Tool output compression
    FeatureSpec("tool_output_compression", False, False, "TOOL_OUTPUT_COMPRESSION", "Compress verbose output"),
    # TOC-SP-6: chronological per-task trajectory artifact (also needs ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR)
    FeatureSpec("trajectory_artifact", False, False, "TRAJECTORY_ARTIFACT", "Write a per-task chronological trajectory-artifact JSONL (TOC-SP-6)"),
    # Output spill
    FeatureSpec("output_spill_to_file", False, True, "OUTPUT_SPILL_TO_FILE", "Spill truncated output to file"),
    FeatureSpec("spill_extractive_summary", False, False, "SPILL_EXTRACTIVE_SUMMARY", "REPL-EMB-3.1: spill summary is verbatim head/tail + top-k task-relevant chunks (no worker call) instead of the 512-token worker summary"),
    # Pipeline monitoring
    FeatureSpec("model_grading", False, False, "MODEL_GRADING", "Post-hoc model-graded evals"),
    # P-BENCH-3 A7: eval-batch serving-class switch. Metadata/shadow support may
    # land separately, but execution routing remains off until this flag is set.
    FeatureSpec("eval_batch_serving", False, False, "EVAL_BATCH_SERVING", "Route eval batches through an evidence-gated batch-serving class"),
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
    FeatureSpec("langgraph_worker", False, False, "LANGGRAPH_WORKER", "LangGraph Phase 3: WorkerNode"),
    FeatureSpec("langgraph_frontdoor", False, False, "LANGGRAPH_FRONTDOOR", "LangGraph Phase 3: FrontdoorNode"),
    FeatureSpec("langgraph_coder", False, False, "LANGGRAPH_CODER", "LangGraph Phase 3: CoderNode"),
    FeatureSpec("langgraph_coder_escalation", False, False, "LANGGRAPH_CODER_ESCALATION", "LangGraph Phase 3: CoderEscalationNode"),
    # Conversation Management
    FeatureSpec("injection_scanning", False, True, "INJECTION_SCANNING", "B7: prompt injection scanning"),
    FeatureSpec("skill_injection_scan", False, False, "SKILL_INJECTION_SCAN", "CT-19b: warn-only injection scan at SkillBank.store / PromptForge failure_context; flags to a ledger, accept/reject unchanged", ("injection_scanning",)),
    FeatureSpec("context_compression", False, False, "CONTEXT_COMPRESSION", "B2: protected-zone compression"),
    FeatureSpec("user_modeling", False, False, "USER_MODELING", "B1: cross-session user preferences", ("injection_scanning",)),
    FeatureSpec("session_token_budget", False, False, "SESSION_TOKEN_BUDGET", "B5: per-session token budget"),
    # Claude Code Local
    FeatureSpec("claude_code_mcp_chat", False, False, "CLAUDE_CODE_MCP_CHAT", "CC Local: MCP chat delegation"),
    # HS-4 P0.2: refuse (422) an agentic-shell /v1 request that lacks x_session_id
    # (OpenCode user-agent or x_tool_mode=client) — a failed session plugin is only
    # logged by OpenCode, so without this it degrades silently. Other clients unaffected.
    FeatureSpec("v1_client_session_guard", True, True, "V1_CLIENT_SESSION_GUARD", "HS-4: require x_session_id from OpenCode/client-tool-mode /v1 requests"),
    # HS-19a stage 1 ("Linked"): record the harness subagent tree on /v1 — the
    # x-parent-session-id header OpenCode already sends (plus HS-16's session-header
    # fallback and the typed x_parent_session_id / x_agent_name body keys) into the
    # inference-tap request_keys and the progress (session) log. RECORD ONLY: model
    # selection is unchanged. Default OFF in test and prod; off = byte-identical.
    FeatureSpec("v1_subagent_link", False, False, "V1_SUBAGENT_LINK", "HS-19a stage 1: record harness subagent parent/child links on /v1 (tap + session log); never changes model selection"),
    # TE-1 (UFH-13) / HS-4 P4 subset: let a /v1 frontdoor answer take /chat's own
    # post-answer escalation hooks (quality escalation to coder_escalation and the
    # MemRL review gate to architect_general), OPT-IN per request via
    # x_escalation=auto|architect_general (absent or off = no escalation), with an
    # escalation receipt in the tap. Default OFF in test and prod. Flag off, or flag
    # on with the key absent, = byte-identical (golden-pinned in
    # test_v1_escalation_off_golden.py), so enabling it changes no unkeyed traffic.
    FeatureSpec("v1_escalation", False, False, "V1_ESCALATION", "TE-1: /v1 frontdoor answers may escalate through /chat's quality-escalation and review-gate hooks, opt-in per request (x_escalation=auto|architect_general; absent/off = none); default off"),
    # RI-23 / OP-69 (a), 2026-09-29: thinking-on roles (live stack priors: --jinja AND
    # acceleration.enable_thinking true — today the :8083 27B's architect_critic /
    # coder_escalation / ingest_long_context) move from /completion to the
    # /v1/chat/completions lane, so their GGUF chat template and registry
    # chat_template_kwargs (enable_thinking, reasoning_effort) take effect and the server
    # splits reasoning into reasoning_content. Read LIVE per request by both the backend
    # router and the orchestrator-side template skip, so the two cannot disagree across a
    # runtime flip. Default OFF in test and prod; flag off = byte-identical (RI-23b A/B).
    # UFH14-LB11.1: the REPL stop sequence "\n```\n" can cut a thinking role's /completion turn
    # inside an unclosed <think>; the extractor drops that text so the turn is wasted. ON: such a
    # turn is continued once (fence restored, no fence stop, output trimmed at the first closing
    # fence after </think>). Default OFF = byte-identical; the diag counter
    # turn_ended_in_unclosed_think is recorded regardless.
    FeatureSpec("repl_think_fence_continue", False, False, "REPL_THINK_FENCE_CONTINUE", "UFH14-LB11.1: continue a REPL turn once when the fence stop sequence cut it inside an unclosed <think> (thinking roles on raw /completion); default off"),
    FeatureSpec("repl_child_lane", False, False, "REPL_CHILD_LANE", "D-RI-CONTAIN-T2: REPL cells run in a warm, killable, Landlock+seccomp+RLIMIT/cgroup-limited CPython child with parent-brokered callbacks (src/repl_environment/child_lane.py); fails closed, never falls back inline; implies repl_tool_arg_validation and the structured exception schema; runtime-toggleable via POST /config; default off", env_on="1", env_off="0"),
    FeatureSpec("repl_tool_arg_validation", False, False, "REPL_TOOL_ARG_VALIDATION", "D-RI-CONTAIN (e)/(a): per-tool argument validation inside every REPL-exposed tool (src/repl_environment/tool_arg_validation.py) and the bounded structured exception schema for tool/REPL errors; tool wrappers are installed when a REPL environment is built; default off", env_on="1", env_off="0"),
    FeatureSpec("monty_repl", False, False, "MONTY_REPL", "D-RI-MONTY-LANE: route REPL cells to a confined pydantic-monty worker with a static screen and CPython fallback (src/repl_environment/monty_adapter.py); default off in test AND prod; hot-swappable via POST /config; fails open to CPython without the `monty` extra"),
    FeatureSpec("thinking_roles_chat_lane", False, False, "THINKING_ROLES_CHAT_LANE", "RI-23: thinking-on roles (jinja + enable_thinking) route through /v1/chat/completions so chat_template_kwargs apply and reasoning_content is split and surfaced; short structured calls (review verdict, plan review/plan JSON) run thinking-off per call; default off pending the RI-23b A/B"),
    FeatureSpec("repl_thinking_budget", False, False, "REPL_THINKING_BUDGET", "Opt-in request and router-selected reasoning/generation budgets on chat turns; default off"),
    # Q38-T10 / Q38-T10.1 (operator ruling 2026-10-20, per-role parity): eval, judge, replay and
    # machine-parsed roles must emit output identical to non-speculative decoding. When ON, those
    # callers (eval_tower llm_judge + rubric judge, model_grader, coherence_judge, the judge_redundancy
    # and routing_replay CLIs) send the per-request llama-server field `speculative.n_max: 0`
    # (src/spec_policy.py). Serving callers never send it. Test default OFF = payloads byte-identical
    # to legacy. Prod default ON since 2026-10-20: the Q38-T10.1 control arm PASSED (per-request n_max 0
    # token-identical to a spec-off server, 20/20 alone and 20/20 beside a speculating slot).
    FeatureSpec("spec_off_bitwise_roles", False, True, "SPEC_OFF_BITWISE_ROLES", "Q38-T10: bitwise-required roles (judges, grader, replay CLIs) send per-request speculative.n_max=0; ON in prod after the Q38-T10.1 control arm passed 2026-10-20 (test default stays off)"),
    # Web research reranking
    FeatureSpec("web_research_rerank", False, False, "WEB_RESEARCH_RERANK", "ColBERT snippet reranking in web_research pipeline"),
    # Routing telemetry
    FeatureSpec("logit_probe", False, False, "LOGIT_PROBE", "First-token logprob capture for learned routing P1.5"),
    # NIB2-45 MindDR deep research mode (three-agent pipeline, feature-flag-gated at pipeline entry)
    FeatureSpec("deep_research_mode", False, False, "DEEP_RESEARCH_MODE", "NIB2-45 MindDR Phase 1: three-agent research pipeline"),
    # intake-607 harness cluster (default-off; shadow/observe only)
    FeatureSpec("ure_uncertainty_shadow_log", False, False, "URE_UNCERTAINTY_SHADOW_LOG", "URE-1: shadow-log routing decision-uncertainty (J10); changes no routing behavior"),
    FeatureSpec("batch_edit_mode", False, False, "BATCH_EDIT_MODE", "BEP-1: coder emits one structured patch set after reasoning instead of interleaved REPL (J8)"),
    FeatureSpec("interleaved_edit_rider", False, False, "INTERLEAVED_EDIT_RIDER", "BEP-2 baseline: per-turn rider instructing coder/architect to apply interleaved file edits (vs answering in prose); experiment-only, default-off"),
    FeatureSpec("dcp_pre_assembly", False, False, "DCP_PRE_ASSEMBLY", "DCP-4 (J7): advisory delegation context pre-assembly — seed the specialist with a budget-bounded code bundle; reactive discovery stays on; default-off"),
    FeatureSpec("dcp_for_consult", False, False, "DCP_FOR_CONSULT", "Internal consults may reuse DCP pre-assembled context; requires dcp_pre_assembly", ("dcp_pre_assembly",)),
    FeatureSpec("review_before_commit_consult", False, False, "REVIEW_BEFORE_COMMIT_CONSULT", "P2 internal interaction: architect_general review_before_commit consult at the edit-transaction seam"),
    FeatureSpec("review_before_commit_targeted_gate", False, False, "REVIEW_BEFORE_COMMIT_TARGETED_GATE", "J17: restrict review_before_commit consults to high-risk edit shapes"),
    # RTE-Prefix (repl-turn-efficiency): render the fixed sections (task,
    # instruction) BEFORE the per-turn-mutating sections (state, context,
    # reference_code) in RootLMPrompt.to_string(), so successive REPL turns
    # share a longer prompt prefix and llama-server KV cache reuse pays off.
    # Default-off: behavioral change to the prompt, needs A/B validation first.
    FeatureSpec("prefix_stable_order", False, False, "PREFIX_STABLE_ORDER", "RTE-Prefix: put fixed prompt sections before per-turn-mutating ones so REPL turns share a longer cache prefix"),
    # RTE-PFX-3: render the root-LM "Turn N" counter as the builder prompt's LAST line
    # instead of the head of the state block, so consecutive REPL turns share the state
    # block (and, in legacy order, the task text) as cache prefix. Default-off: prompt
    # bytes change when on; flag off is byte-identical.
    FeatureSpec("repl_turn_counter_tail", False, False, "REPL_TURN_COUNTER_TAIL", "RTE-PFX-3: move the root-LM Turn N counter from the state-block head to the prompt tail"),
    # CT-21 (DP-G ruling 2026-10-22, "Hold; architect trial"): boundary-only
    # "data, not instructions" sentence directly above the untrusted block in the
    # architect_investigate (evidence bundle) and architect_synthesis (specialist
    # report) prompts only. Worker/REPL roles never change. Default-off; trial only.
    FeatureSpec("architect_untrusted_boundary_sentence", False, False, "ARCHITECT_UNTRUSTED_BOUNDARY_SENTENCE", "CT-21: architect-only boundary sentence above untrusted evidence/report blocks (trial; default off)"),
    # UFH14-B1 F1 (D1): add the derived silent-prefill allowance (serving_params:
    # measured prefill rate of THIS server x THIS prompt) to the HTTP timeout of
    # long-prompt calls — primitives calls with no request deadline, and the
    # passthrough read timeout (raised, never lowered). A deadlined call is
    # clamped back to its deadline regardless. The serving_params record block
    # (allowance, at_risk, doomed) is written whether this is on or off.
    # Default-off in BOTH: it lengthens timeouts (weaker hang detection) and the
    # rate has no live samples yet — flip after a doomed-count window.
    FeatureSpec("derived_prefill_timeout", False, False, "DERIVED_PREFILL_TIMEOUT", "UFH14-B1 F1: add the measured silent-prefill allowance to long-prompt call timeouts (no-deadline primitives calls; passthrough read timeout raised, never lowered)"),
    # UFH14-B1 F2 (D4): wall-time forced-answer turn in the REPL graph. Once the
    # elapsed share of the request budget reaches ORCHESTRATOR_REPL_ANSWER_FORCE_FRAC
    # (default 0.65, F12), the turn prompt demands FINAL(best answer) now and
    # that turn skips session compaction. Default-off in BOTH: prompt change.
    # UFH14-B1: session compaction rewrites TaskState.context, which no turn
    # prompt renders; its worker_general index call therefore reached no model.
    # Off = deterministic index, no LLM call (the bug fix). On = the old LLM
    # index with its input capped to the worker's window, for a future renderer.
    FeatureSpec("session_compaction_llm_index", False, False, "SESSION_COMPACTION_LLM_INDEX", "UFH14-B1: LLM-written compaction index (worker_general, input capped to its window); off = deterministic index, since TaskState.context is not rendered into turn prompts"),
    # OAB-19 (operator decision 2026-11-18): REPL turns keep their full context. Session compaction
    # rewrites TaskState.context, which no REPL turn prompt renders, so on a REPL turn it only produced
    # an externalised file and a rewritten field nobody reads. Retired by default (like the worker
    # summarisation in UFH14-B1). On = the old per-turn compaction call, still reachable. Runtime-toggleable
    # (POST /config) and on the autopilot surface; never pin it observe-only. A prompt that would
    # overflow the window is NOT silently trimmed: the server refuses it and the turn fails with an
    # explicit ContextOverflowError (graph/helpers._execute_turn).
    FeatureSpec("repl_compaction", False, False, "REPL_COMPACTION", "OAB-19: run session compaction on REPL turns (off = retired; REPL turns keep full context, since TaskState.context is not rendered into turn prompts)"),
    FeatureSpec("repl_answer_force", False, False, "REPL_ANSWER_FORCE", "UFH14-B1 F2: wall-time forced-answer turn (FINAL now at a fraction of the request budget; no compaction on that turn)"),
    # REPL harness-robustness guards (taskflow canary: a local 35B scored 0/10 from harness failures alone).
    # All default OFF, runtime-toggleable (POST /config) and on the autopilot flag surface.
    FeatureSpec("repl_syntax_loop_breaker", False, False, "REPL_SYNTAX_LOOP_BREAKER", "Harness guard A: count consecutive SyntaxError REPL turns per task; at 2 append a one-shot 'emit a minimal valid snippet' hint to the parser error, at 3 end the task as a harness failure (harness:syntax_loop) instead of burning turns/escalating"),
    FeatureSpec("repl_plan_narration_nudge", False, False, "REPL_PLAN_NARRATION_NUDGE", "Harness guard B: a prose-only turn that narrates a plan ('I will', 'Let me', 'Next I', 'First,', 'Plan:') is not rescued into a bogus FINAL; it gets a continue-nudge ('emit one python block now') counted by MAX_CONSECUTIVE_NUDGES"),
    FeatureSpec("harness_failure_ratio_guard", False, False, "HARNESS_FAILURE_RATIO_GUARD", "Harness guard C: when REPL+tool failures / turns >= 0.5 over >= 4 turns, mark artifacts harness_suspect=True and refuse tier escalation (the harness, not the model, is failing)"),
    # Prefix-cache layout (hybrid Qwen3.6-35B-A3B cannot --cache-reuse; it rolls back to context checkpoints, so a
    # byte change early in the prompt or a rewrite of earlier turns discards most of the KV cache). All default OFF.
    FeatureSpec("prompt_stable_prefix", False, False, "PROMPT_STABLE_PREFIX", "Prefix-cache layout (a): REPL root prompt renders system/tools/rules/task/instruction first (the stable prefix); the per-request selected-tools block, the previous-turns history and then the volatile state/last-output/turn counter follow, and selected tools are listed sorted by name. Off = legacy layout, byte-identical"),
    FeatureSpec("history_append_only", False, False, "HISTORY_APPEND_ONLY", "Prefix-cache layout (b): REPL turn memory (repl_turn_memory, N>0) keeps every turn until a char budget is hit, then drops the oldest turns once, down to half the budget (instead of sliding the N-turn window every turn); the /v1 B2 history fold (context_compression) only runs past a hard size. Off = C3 sliding window exactly as before"),
    FeatureSpec("tool_output_bound", False, False, "TOOL_OUTPUT_BOUND", "Prefix-cache layout (c): tool/REPL output and error text stored in REPL turn memory is cut to 2000 chars (3:1 head:tail, deterministic marker) when stored. Off = the 4000-char render-time clip only"),
    # intake-614/615 DAR-6 scaffolding (default-off in BOTH test and prod; no production routing until DAR-6.5 A/B clears)
    FeatureSpec("swarm_fanout", False, False, "SWARM_FANOUT", "DAR-6.1: fan high-injection-risk prompts to N>=2 concurrent serves + BT-aggregate (J14). Scaffolding only — default-off until the DAR-6.5 injection-suite A/B clears (handoffs/active/decision-aware-routing.md § DAR-6.5)."),
    # P21.A test-time compute: DeepConf offline confidence-filtered self-consistency (intake-603)
    FeatureSpec("deepconf", False, False, "DEEPCONF", "DeepConf offline confidence-filtered self-consistency"),
    # RD-5 Reviewer decision plane (shadow/enforce split; intake-849 P2). Both default-off
    # in BOTH test and prod. review_decision_shadow has NO dependencies on purpose — it is
    # the DECOUPLED shadow-emission path: plan_review requires memrl (see validate()), but
    # the decision-plane shadow machinery must be able to flow WITHOUT flipping plan_review
    # (and therefore without requiring memrl). Always-on trace emission (TM-3) is independent
    # of both flags; these gate only the extra shadow processing / (blocked) enforcement.
    FeatureSpec("review_decision_shadow", False, False, "REVIEW_DECISION_SHADOW", "RD-5: emit + trace review-plane decisions (verifier precedence, reject-admissibility, warn-only downgrade) WITHOUT acting. Decoupled from plan_review→memrl (no dependencies)."),
    FeatureSpec("review_decision_enforce", False, False, "REVIEW_DECISION_ENFORCE", "RD-5: ACT on review-plane decisions. Default OFF and BLOCKED on the H-LB LB-6 latency/sampling budget gate — do not enable until that gate passes."),
    # gpu-serving-tie-in-program P0-7: role-agnostic GPU-resident shadow lane
    # (MI210). Default-off in BOTH test and prod. Gates only the lane's
    # scaffolding surfaces (np_ceiling policy loader + preflight tooling); the
    # lane serves NO production traffic until the P3-3 three-gates sign-off
    # (shadow-only invariant, program decision D3).
    FeatureSpec("gpu_shadow_lane", False, False, "GPU_SHADOW_LANE", "GPU-resident shadow serving lane (MI210): tenant-as-data scaffolding, shadow-only"),
    # Typed decision plane (TD-1, intake-1473): one-pass choice/score/noul
    # decisions over the llm_call seam. Default-off in BOTH test and prod;
    # TD-1 ships the core only and is NOT wired into any live route — TD-5
    # performs the wiring under this flag.
    FeatureSpec("typed_decisions", False, False, "TYPED_DECISIONS", "TD-1: one-pass typed decision plane (choice/score/noul)"),
    # TD-5: observability-only shadow of the incumbent routing decision. The
    # flag alone is not enough — a log path must also be configured via
    # ORCHESTRATOR_TYPED_DECISIONS_SHADOW_LOG, otherwise submit_shadow no-ops.
    FeatureSpec("typed_decisions_shadow", False, False, "TYPED_DECISIONS_SHADOW", "TD-5: log one typed-decision shadow call beside the incumbent routing decision; never changes routing"),
    # TD-4: closed-set typed selection of tool-call arguments, wired at the
    # REPL tool dispatch chokepoint (src/repl_environment/context.py). The
    # typed arm is fail-open: any decline keeps the model-provided arguments.
    FeatureSpec("typed_decisions_tool_args", False, False, "TYPED_DECISIONS_TOOL_ARGS", "TD-4: closed-set typed selection of tool-call arguments; fail-open to the model-provided arguments"),
    # TD-29.M2: validate-first for the tool-argument shadow. Only meaningful with typed_decisions_tool_args on.
    FeatureSpec("typed_decisions_validate_first", False, False, "TYPED_DECISIONS_VALIDATE_FIRST", "TD-29.M2: validate the original tool kwargs against the lossless schema before any typed repair call; valid = zero repair calls, unsupported constraints refused, shadow only"),
    # UFH-12 REPL-EMB-1.1/1.4: placement-aware pooled embedding client with the
    # neighbour in-flight cap (src/embedding_pool/). Default OFF in BOTH test and
    # prod; with it off the factories return None and no consumer is migrated.
    FeatureSpec("repl_embedding_pool", False, False, "REPL_EMBEDDING_POOL", "UFH-12 REPL-EMB-1.1/1.4: pooled embedding client with placement-aware scheduling and the busy-frontdoor neighbour cap"),
    # ── Flag surface (2026-11-18): codemode / tool-picker / REPL / document knobs that used to be
    # env-only. Registering them here puts them on the autopilot manipulation surface (POST /config,
    # StructuralLab flag trials, the planner flag block) and makes them toggle at runtime: the code
    # paths read them through ``src.flag_env.flag_env()``, which overlays the shared runtime-flag file
    # on ``os.environ``. With no runtime override the env value is returned untouched, so a flag that
    # is not toggled behaves byte-identically. ``env_var`` is the ORIGINAL env name (unchanged).
    FeatureSpec("tool_selector", False, False, "TOOL_SELECTOR", "CM-1 hardened tool picker: narrow the per-request tool list (native /v1 path + REPL prompt); ORCHESTRATOR_TOOL_SELECTOR=embed|oneshot selects the variant (runtime ON = embed); fails open to the full list; embeds through the pooled embedder, so it needs repl_embedding_pool (without it the picker is not wired and every request gets the full list)", ("repl_embedding_pool",), env_parse=_env_in("embed", "oneshot"), env_on="embed", env_off="off"),
    FeatureSpec("trace_navigation_tools", False, False, "TRACE_NAVIGATION_TOOLS", "UTM-B1: model-facing read-only trace navigation (ms.search / ms.expand on the MCP server, ms_search / ms_expand as REPL tools through the shared tool broker); each call is scoped to the caller's own session (default off until measured; the MCP server reads it at startup)"),
    FeatureSpec("trace_navigation_cross_session", False, False, "TRACE_NAVIGATION_CROSS_SESSION", "UTM-B1: allow scope=\"all\" on ms.search / ms.expand to read trace rows from ANY session (privacy setting, default off; needs trace_navigation_tools)", ("trace_navigation_tools",)),
    FeatureSpec("tool_surface_logging", False, False, "TOOL_SURFACE_LOGGING", "UFH14-TOOLS-1: per-turn JSONL of the REPL-global and registry-tool counts each role can see (logs/tool_surface)"),
    FeatureSpec("autopilot_speed_physical_bound", False, False, "SPEED_PHYSICAL_BOUND", "AP-72: SafetyGate marks a trial invalid when its reported speed exceeds the per-model physical ceiling in orchestration/autopilot_speed_ceilings.yaml (measured decode roofline x slots x 1.10); unknown model = no bound (fail-open). Read by scripts/autopilot/safety_gate.py per verdict; AUTOPILOT_SPEED_PHYSICAL_CEILING_TPS, when set, overrides the table", env_name="AUTOPILOT_SPEED_PHYSICAL_BOUND", env_on="1", env_off="0"),
    FeatureSpec("tool_surface_cap", False, False, "TOOL_SURFACE_CAP","UFH14-TOOLS-1: list_tools() shows only the per-role allowlist (orchestration/tool_surface_allowlist.yaml); the rest stay callable and list_tools(deferred=True) shows them"),
    FeatureSpec("tool_selector_shadow", False, False, "TOOL_SELECTOR_SHADOW", "C2 shadow rerank beside the applied picker pick (logs the one-shot decision-model choice, never applied); needs tool_selector; costs one worker call per selection", ("tool_selector",), env_parse=_env_in("oneshot"), env_on="oneshot", env_off="off"),
    FeatureSpec("repl_explicit_final", True, True, "REPL_EXPLICIT_FINAL", "codemode fix 1: a one-line REPL tool/exploration call is never auto-wrapped as FINAL(...) (ORCHESTRATOR_REPL_EXPLICIT_FINAL unset/tools); off = legacy wrap of every single-line expression", env_parse=_env_not_in("0", "false", "off", "no"), env_on="tools", env_off="0"),
    FeatureSpec("repl_explicit_final_strict", False, False, "REPL_EXPLICIT_FINAL_STRICT", "codemode fix 1 strict: only an obvious answer VALUE is auto-wrapped as FINAL (ORCHESTRATOR_REPL_EXPLICIT_FINAL=strict); implies repl_explicit_final", env_parse=_env_in("1", "true", "on", "yes", "strict"), env_name="ORCHESTRATOR_REPL_EXPLICIT_FINAL"),
    FeatureSpec("repl_turn_memory", False, False, "REPL_TURN_MEMORY", "codemode fix 2: last N REPL turns of code + output/error in the root prompt (ORCHESTRATOR_REPL_TURN_MEMORY=N; runtime ON = N 4)", env_parse=_env_positive_int, env_on="4", env_off="0"),
    FeatureSpec("codemode_prompt_v2", False, False, "CODEMODE_PROMPT", "codemode fix 3: v2 code-mode root prompt for task_root-scoped tool requests (ORCHESTRATOR_CODEMODE_PROMPT=v2)", env_parse=_env_in("v2"), env_on="v2", env_off=""),
    FeatureSpec("codemode_stubs", False, False, "CODEMODE_STUBS", "codemode fix 4: typed Python stubs for picked tools, bound as direct REPL functions returning Python objects", env_on="1", env_off="0"),
    FeatureSpec("repl_tool_search", False, False, "REPL_TOOL_SEARCH", "CM-2: search_tools / describe_tool in the REPL", env_on="1", env_off="0"),
    FeatureSpec("structured_read_only", False, False, "STRUCTURED_READ_ONLY", "CM-3: ToolOutput envelope for read-only tools only (write tools keep today's shape); companion of codemode_stubs", env_on="1", env_off="0"),
    FeatureSpec("repl_classify", False, False, "REPL_CLASSIFY", "CM-4: classify / route_classify / risk_bands / q_scores in the REPL (per-session item budget ORCHESTRATOR_REPL_CLASSIFY_BUDGET)", env_on="1", env_off="0"),
    FeatureSpec("repl_handles", False, False, "REPL_HANDLES", "CM-5: store / load / handles (branch-replayed handle store) in the REPL; turns replay recording on by default while on", env_on="1", env_off="0"),
    FeatureSpec("untrusted_tool_text", True, True, "UNTRUSTED_TOOL_TEXT", "SECURITY: label, defang and cap UNTRUSTED tool text before it re-enters model context (ORCHESTRATOR_UNTRUSTED_TOOL_TEXT=0 is the kill switch); autopilot pinned, observe only", env_on="1", env_off="0"),
    FeatureSpec("repl_write_dedupe", True, True, "REPL_WRITE_DEDUPE", "REPL: an identical write call that already succeeded in this request is skipped, not re-run (ORCHESTRATOR_REPL_WRITE_DEDUPE=0 is the kill switch)", env_on="1", env_off="0"),
    FeatureSpec("pdf_route_lighton", False, False, "PDF_ROUTE_LIGHTON", "PDF ingest: needs_ocr documents go to the LightOnOCR-2-1B aux server, the rest to LiteParse (PIP-04 held-out rule); the stack launcher sets ORCHESTRATOR_PDF_ROUTE_LIGHTON=1", env_on="1", env_off="0"),
    FeatureSpec("l3_textrow", False, False, "L3_TEXTROW", "L3 empty-table fallback: text-row table extraction on pages with no parser table (method=textrow); the stack launcher sets ORCHESTRATOR_L3_TEXTROW=1", env_parse=_env_in("1"), env_on="1", env_off="0"),
    # M-27d.1: stamp the request-source label (interactive/autopilot/benchmark/...) on episodic rows so recall
    # composition can be separated by source. Write side only: default off, no retrieval change. Read through
    # src.flag_env.episodic_source_label_on(), so POST /config toggles it live.
    FeatureSpec("episodic_source_label", False, False, "EPISODIC_SOURCE_LABEL", "M-27d.1: stamp request_source (interactive/autopilot/benchmark/repl_experiment) on episodic task-start journal entries and MemRL rows (extra, not embedded); write side only, retrieval unchanged", env_on="1", env_off="0"),
    # M-12k (2026-11-19): the memory-off arm of the frontdoor answer-quality A/B. ON (default) = today's
    # behaviour: the learned HybridRouter (episodic KNN over MemRL memories) picks the initial role, the
    # execution mode, the try-cheap-first Q-value gate and the plan-review Q-value gate. OFF = those four
    # episodic LOOKUPS are skipped (routing falls to the rules classifier / REPL-default mode / no Q gate);
    # episodic WRITES (task-start logging, Q-scoring) are untouched. The same switch exists per request as
    # ChatRequest.episodic_lookups (False = off for that request only; absent/None = follow this flag), so
    # two arms can interleave on one server. Read LIVE per request, so POST /config toggles it with no restart.
    FeatureSpec("episodic_routing_lookups", True, True, "EPISODIC_ROUTING_LOOKUPS", "M-12k: episodic (MemRL/HybridRouter) lookups feed initial-role routing, mode selection and the Q-value gates; off = those lookups are skipped (rules routing), writes unchanged; per-request override ChatRequest.episodic_lookups=false"),
    # Debug/Development
    FeatureSpec("mock_mode", True, False, "MOCK_MODE", "Mock mode for safety"),
)

# Indexed for fast lookup
_REGISTRY_BY_NAME: dict[str, FeatureSpec] = {s.name: s for s in _FEATURE_REGISTRY}


def runtime_flags_path() -> Path:
    """Shared runtime flag override file used by all API workers."""
    override = os.environ.get(RUNTIME_FLAGS_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "orchestration" / "runtime_flags.json"


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_runtime_expiry(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError):
        return None
    try:
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _validate_runtime_flag_ttl(ttl_s: Any) -> float:
    if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)):
        raise ValueError("ttl_s must be a finite number of seconds")
    try:
        ttl = float(ttl_s)
    except OverflowError as exc:
        raise ValueError("ttl_s must be a finite number of seconds") from exc
    if not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("ttl_s must be a finite positive number of seconds")
    return ttl


def _runtime_records(
    path: Path | None = None,
    *,
    as_of: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    path = path or runtime_flags_path()
    as_of = as_of or _utc_now()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("flags", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}
    records: dict[str, dict[str, Any]] = {}
    for name, record in raw.items():
        if name not in _REGISTRY_BY_NAME:
            continue
        if isinstance(record, dict):
            value = _coerce_bool(record.get("value"))
            if value is None:
                continue
            normalized = {
                "value": value,
                "set_by": str(record.get("set_by") or "unknown"),
                "ts": str(record.get("ts") or ""),
            }
            if "expires_at" in record:
                expiry = _parse_runtime_expiry(record.get("expires_at"))
                if expiry is None:
                    logger.warning("Ignoring runtime flag %s with invalid or timezone-naive expiry", name)
                    continue
                if expiry <= as_of:
                    logger.warning("Ignoring expired runtime flag override %s; using environment/registry baseline", name)
                    continue
                normalized["expires_at"] = expiry.isoformat(timespec="seconds")
            records[name] = normalized
        else:
            value = _coerce_bool(record)
            if value is not None:
                records[name] = {"value": value, "set_by": "legacy", "ts": ""}
    return records


def runtime_flag_overrides(
    path: Path | None = None,
    *,
    as_of: datetime | None = None,
) -> dict[str, bool]:
    """Return valid runtime overrides from the shared flag file."""
    return {
        name: bool(record["value"])
        for name, record in _runtime_records(path, as_of=as_of).items()
    }


def write_runtime_flag_overrides(
    overrides: dict[str, bool],
    *,
    set_by: str = "unknown",
    ttl_s: int | float | None = None,
    expires_at: datetime | str | None = None,
    path: Path | None = None,
) -> Path:
    """Atomically write runtime overrides; expiry metadata applies only to enables."""
    if ttl_s is not None and expires_at is not None:
        raise ValueError("specify ttl_s or expires_at, not both")
    path = path or runtime_flags_path()
    records = _runtime_records(path)
    now = _utc_now()
    ts = now.isoformat(timespec="seconds")
    expiry = None
    if ttl_s is not None:
        ttl = _validate_runtime_flag_ttl(ttl_s)
        try:
            expiry = now + timedelta(seconds=ttl)
        except OverflowError as exc:
            raise ValueError("ttl_s exceeds the supported datetime range") from exc
    elif expires_at is not None:
        expiry = _parse_runtime_expiry(expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at)
        if expiry is None or expiry <= now:
            raise ValueError("expires_at must be a future timezone-aware ISO-8601 timestamp")
    for name, value in overrides.items():
        if name not in _REGISTRY_BY_NAME:
            continue
        record = {
            "value": bool(value),
            "set_by": set_by,
            "ts": ts,
        }
        record_expiry = expiry
        if value and record_expiry is None and name == "repl_embedding_pool":
            try:
                record_expiry = now + timedelta(seconds=EXPERIMENT_FLAG_TTL_S)
            except OverflowError as exc:
                raise ValueError("default runtime flag expiry exceeds the supported datetime range") from exc
        if value and record_expiry is not None:
            record["expires_at"] = record_expiry.isoformat(timespec="seconds")
        records[name] = record
    payload = {
        "version": 1,
        "updated_at": ts,
        "flags": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


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
    voice_turn: bool = False  # First-party voice-turn route; opt-in pending integrated acceptance.

    tts_it_normalize: bool = True  # CS-8.L4: Italian-only TTS number/unit/acronym normalizer, live path.
    tts_it_respell: bool = True  # CS-8.L4: Italian-only stress/proper-noun respelling, live path.

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
    agent_file_injection: bool = False  # F2: append registry-selected agent file to role prompts
    restricted_python: bool = False  # Use RestrictedPython for REPL (requires library)

    # Phase 3: Specialist routing (MemRL-driven intelligent orchestration)
    specialist_routing: bool = False  # Enable specialist routing (coder, architect) via Q-values

    # Phase 3+: GraphRouter (GNN-based parallel routing signal for cold-start optimization)
    graph_router: bool = False  # Enable bipartite GAT routing predictor

    # Ingest-triviality guard: keep trivially-easy short prompts off the 80B
    # accuracy/long-context specialist (opt-in; requires specialist_routing)
    ingest_triviality_guard: bool = False  # Demote easy short prompts off ingest_long_context

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
    observed_dependency_logging: bool = False  # HIL-15: per-call read/write resource edges (default off)
    step_event_logging: bool = False  # HIL-13a: per-step tool events for the L2 scan (default off)
    replay_recording: bool = False  # HIL-16: content-addressed per-turn recording for replay (default off)
    exec_containment: bool = True  # SEC-EXEC-3/5: Landlock+seccomp for model-exec children (default ON; ORCHESTRATOR_EXEC_CONTAINMENT=0 is the kill switch)
    web_fetch_guard: bool = False  # SEC-EXEC-3: private-range/link-local/metadata deny for web_fetch + fetch_docs (default off)
    exec_write_flags: bool = False  # SEC-EXEC-7: write_outside_root flag in tool-call audit records (default off)
    model_fallback: bool = False  # Try same-tier alternatives on circuit-open
    content_cache: bool = False  # SHA-256 keyed response cache for LLM calls
    session_compaction: bool = False  # Summarize old context on long conversations
    session_log: bool = False  # Append-only processing journal across REPL turns
    session_log_verbatim: bool = False
    context_arm_ab: bool = False
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
    langgraph_worker: bool = False             # Migrate WorkerNode to LangGraph backend
    langgraph_frontdoor: bool = False          # Migrate FrontdoorNode to LangGraph backend
    langgraph_coder: bool = False              # Migrate CoderNode to LangGraph backend
    langgraph_coder_escalation: bool = False   # Migrate CoderEscalationNode to LangGraph backend

    # Budget controls (Fast-RLM)
    worker_call_budget: bool = False  # Cap total REPL executions per task
    task_token_budget: bool = False   # Cap cumulative tokens across all turns
    final_schema_validation: bool = False  # Validate FINAL() value against caller JSON Schema

    # Pipeline monitoring: model-graded subjective evals
    model_grading: bool = False  # Post-hoc model-graded evals via the live worker-general path

    # P-BENCH-3 A7: opt-in eval-batch serving-class switch.
    eval_batch_serving: bool = False

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
    compaction_shadow: bool = False  # RTG-08 option C: non-mutating shadow of role-aware compaction (needs two_level_condensation)

    # Context window management (C2/C3/C1)
    accurate_token_counting: bool = False  # Use llama-server /tokenize for exact token counts
    tool_result_clearing: bool = False  # Clear stale <<<TOOL_OUTPUT>>> blocks from last_output

    # Reasoning length alarm (short-m@k Action 9): retry with conciseness nudge
    reasoning_length_alarm: bool = False  # Cancel + retry when <think> exceeds 1.5× band budget
    reasoning_runaway_retry: bool = False  # MF-RR-1: retry native length cutoff with no action; default off

    # Tool output compression (Phase 2 native): compress verbose output before prompt injection
    tool_output_compression: bool = False  # Compress pytest/git/build output to preserve actionable info only

    # CMV-style output spill (Action 11): write truncated output/error to temp file with peek() pointer
    output_spill_to_file: bool = False  # Spill long REPL output/error to file + retrieval pointer
    spill_extractive_summary: bool = False  # REPL-EMB-3.1: extractive (no-worker) spill summary (default off)
    trajectory_artifact: bool = False  # TOC-SP-6: per-task trajectory-artifact JSONL writer (needs ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR)

    # Conversation Management (B-series cherry-picks from Hermes/OpenGauss)
    injection_scanning: bool = False  # B7: Prompt injection scanning on loaded context
    skill_injection_scan: bool = False  # CT-19b: warn-only scan at skill store / PromptForge failure_context
    context_compression: bool = False  # B2: Protected-zone context compression
    user_modeling: bool = False  # B1: Cross-session user preference modeling
    session_token_budget: bool = False  # B5: Per-session token budget with compact/stop signals

    # Claude Code Local Integration (CC Local)
    claude_code_mcp_chat: bool = False  # MCP tools for delegating chat to running orchestrator
    v1_client_session_guard: bool = False  # HS-4: 422 on OpenCode/client-mode /v1 requests without x_session_id
    v1_subagent_link: bool = False  # HS-19a stage 1: record /v1 subagent parent links (tap + session log); default off
    v1_escalation: bool = False  # TE-1: opt-in /v1 frontdoor escalation via /chat's post-answer hooks (explicit x_escalation); default off
    thinking_roles_chat_lane: bool = False  # RI-23: thinking-on roles on /v1/chat/completions (kwargs + reasoning_content); default off
    repl_think_fence_continue: bool = False  # UFH14-LB11.1: continue a turn cut inside an unclosed <think> by the fence stop; default off
    repl_child_lane: bool = False  # D-RI-CONTAIN-T2: contained child lane for REPL cells (opt-in; runtime-toggleable); default off
    repl_tool_arg_validation: bool = False  # D-RI-CONTAIN (e)/(a): per-tool argument validation + structured exception schema; default off
    monty_repl: bool = False  # D-RI-MONTY-SLICE: restricted pydantic-monty REPL adapter (opt-in experiment lane); default off
    spec_off_bitwise_roles: bool = False  # Q38-T10: bitwise-required roles send per-request speculative.n_max=0; prod default ON (registry), Q38-T10.1 control arm passed 2026-10-20

    # Web research reranking (ColBERT snippet pre-fetch filtering)
    web_research_rerank: bool = False  # Rerank DDG snippets via ColBERT before page fetch

    # Routing telemetry: capture first-token logprobs for learned routing P1.5
    logit_probe: bool = False  # Log top-64 first-token logprobs from frontdoor to JSONL

    # NIB2-45 MindDR: feature-flag-gated three-agent research pipeline (Planning → DeepSearch → Report).
    # When on, research-like queries (per classifier.is_research_like) route through the MindDR subgraph
    # instead of the direct flow. Phase 1 is prompt-level only; Phase 2 training is GPU-gated.
    deep_research_mode: bool = False

    # intake-607 harness cluster (default-off; shadow/observe only)
    ure_uncertainty_shadow_log: bool = False  # URE-1 (J10): shadow-log routing uncertainty; no behavior change
    batch_edit_mode: bool = False  # BEP-1 (J8): coder emits one structured patch set after reasoning
    interleaved_edit_rider: bool = False  # BEP-2 baseline: per-turn interleaved-edit rider (experiment-only)
    dcp_pre_assembly: bool = False  # DCP-4 (J7): advisory delegation context pre-assembly (seed bundle)
    dcp_for_consult: bool = False  # P2 consult may reuse DCP seed context; requires dcp_pre_assembly
    review_before_commit_consult: bool = False  # P2 edit-transaction consult seam; default-off
    review_before_commit_targeted_gate: bool = False  # J17 targeted high-risk edit-shape consult gate

    # intake-614/615 DAR-6 (default-off; scaffolding only — no production routing until DAR-6.5 A/B clears)
    swarm_fanout: bool = False  # DAR-6.1: fan high-injection-risk prompts to N≥2 concurrent serves + BT-aggregate (J14)
    # P21.A test-time compute: DeepConf offline confidence-filtered self-consistency (intake-603).
    # When on, the caller filters low-confidence reasoning traces (bottom-10% group confidence
    # from per-token top-k logprobs) and takes a confidence-weighted majority vote. Default OFF;
    # build + scoring logic in src/test_time/deepconf.py. Wiring onto N parallel llama-server
    # completions (payload n_probs=K) is P21.A3, gated on the A2 live-server sanity check.
    deepconf: bool = False

    # RD-5 Reviewer decision plane (shadow/enforce split).
    # review_decision_shadow: emit + trace review decisions (verifier precedence,
    #   reject-admissibility, warn-only downgrade) but NEVER act. Deliberately has
    #   NO memrl dependency — this is the decoupled shadow-emission path so shadow
    #   data flows without flipping plan_review (which requires memrl).
    # review_decision_enforce: act on review decisions. Default OFF and BLOCKED on
    #   the H-LB LB-6 latency/sampling budget gate — do not enable until it passes.
    review_decision_shadow: bool = False
    review_decision_enforce: bool = False

    # GPU shadow lane (gpu-serving-tie-in-program P0-7): default OFF everywhere.
    # Gates the np_ceiling policy loader + preflight scaffolding only; no
    # production routing or launch path reads this flag (D3 shadow-only).
    gpu_shadow_lane: bool = False

    # Typed decision plane (TD-1, intake-1473): default OFF everywhere. TD-1
    # ships the core (types/confidence/schema/runner) only; no route reads
    # this flag yet — TD-5 wires the plane in under it.
    typed_decisions: bool = False

    # TD-5: observability-only shadow of the incumbent routing decision.
    # Default OFF; also requires ORCHESTRATOR_TYPED_DECISIONS_SHADOW_LOG to be
    # set — with no configured sink the shadow no-ops even when on.
    typed_decisions_shadow: bool = False

    # TD-4: closed-set typed selection of tool-call arguments at the REPL tool
    # dispatch chokepoint. Default OFF in both test and prod; fail-open — any
    # decline keeps the model-provided arguments.
    typed_decisions_tool_args: bool = False

    # TD-29.M2: validate-first before any typed repair call (shadow). Default OFF.
    typed_decisions_validate_first: bool = False

    # UFH-12 REPL-EMB-1.1/1.4: pooled embedding client (src/embedding_pool/).
    # Default OFF in both test and prod; get_pooled_embedder()/get_sync_embedder()
    # return None while off, so no live path changes until a consumer migrates.
    repl_embedding_pool: bool = False

    # RTE-Prefix (repl-turn-efficiency): render fixed prompt sections (task,
    # instruction) BEFORE per-turn-mutating ones (state, context,
    # reference_code) in RootLMPrompt.to_string(), so successive REPL turns
    # share a longer llama-server KV-cache prefix. Default OFF — prompt
    # behavioral change, pending A/B validation.
    prefix_stable_order: bool = False
    # RTE-PFX-3: root-LM Turn N counter at the prompt tail (default off).
    repl_turn_counter_tail: bool = False
    # CT-21: architect-only untrusted-boundary sentence (default off; trial).
    architect_untrusted_boundary_sentence: bool = False

    # UFH14-B1 F1: derived silent-prefill allowance on long-prompt call timeouts.
    derived_prefill_timeout: bool = False
    # UFH14-B1: LLM-written session-compaction index (off = deterministic).
    session_compaction_llm_index: bool = False
    # OAB-19: session compaction on REPL turns (off = retired, full context kept).
    repl_compaction: bool = False
    # UFH14-B1 F2: wall-time forced-answer turn in the REPL graph.
    repl_answer_force: bool = False
    # REPL harness-robustness guards (default off): syntax-loop breaker, plan-narration nudge, failure-ratio guard.
    repl_syntax_loop_breaker: bool = False
    repl_plan_narration_nudge: bool = False
    harness_failure_ratio_guard: bool = False
    # Prefix-cache layout flags (default off): stable prompt prefix, append-only REPL history, tool-output bound.
    prompt_stable_prefix: bool = False
    history_append_only: bool = False
    tool_output_bound: bool = False


    # Flag surface (2026-11-18): see the registry block above. Defaults mirror the registry.
    tool_selector: bool = False
    tool_selector_shadow: bool = False
    trace_navigation_tools: bool = False  # UTM-B1: ms.search/ms.expand (MCP + REPL), session-scoped (default off)
    trace_navigation_cross_session: bool = False  # UTM-B1: allow scope="all" (default off)
    tool_surface_logging: bool = False  # UFH14-TOOLS-1: per-turn tool-surface counts (default off)
    autopilot_speed_physical_bound: bool = False  # AP-72: per-model physical speed ceiling in SafetyGate (default off)
    tool_surface_cap: bool = False  # UFH14-TOOLS-1: per-role list_tools allowlist, deferred rest (default off)
    repl_explicit_final: bool = True
    repl_explicit_final_strict: bool = False
    repl_turn_memory: bool = False
    codemode_prompt_v2: bool = False
    codemode_stubs: bool = False
    repl_tool_search: bool = False
    structured_read_only: bool = False
    repl_classify: bool = False
    repl_handles: bool = False
    untrusted_tool_text: bool = True
    repl_write_dedupe: bool = True
    pdf_route_lighton: bool = False
    l3_textrow: bool = False
    episodic_source_label: bool = False  # M-27d.1: stamp request_source on episodic rows (write side)
    episodic_routing_lookups: bool = True  # M-12k: False skips episodic lookups in routing (see registry)

    # Debug/Development
    mock_mode: bool = True  # Default to mock mode for safety

    repl_thinking_budget: bool = False  # Request-local caller/router budgets on chat turns; default off

    def validate(self) -> list[str]:
        """Validate feature dependencies.

        Returns:
            List of validation errors (empty if all valid).
        """
        errors = []

        # Scripts require tools
        if self.scripts and not self.tools:
            errors.append("scripts feature requires tools feature")

        # Tool picker shadow rerank only means something beside an applied pick; the picker embeds via the pool
        if self.tool_selector_shadow and not self.tool_selector:
            errors.append("tool_selector_shadow feature requires tool_selector feature")
        if self.tool_selector and not self.repl_embedding_pool:
            errors.append("tool_selector feature requires repl_embedding_pool feature")

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
        if self.skill_injection_scan and not self.injection_scanning:
            errors.append("skill_injection_scan feature requires injection_scanning feature")
        if self.dcp_for_consult and not self.dcp_pre_assembly:
            errors.append("dcp_for_consult feature requires dcp_pre_assembly feature")

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


def _compute_feature_flags(
    *,
    production: bool = False,
    override: dict[str, bool] | None = None,
    runtime_at: datetime | None = None,
) -> tuple[dict[str, bool], dict[str, str]]:
    defaults = {
        spec.name: (spec.default_prod if production else spec.default_test)
        for spec in _FEATURE_REGISTRY
    }
    flags: dict[str, bool] = {}
    sources: dict[str, str] = {}
    for spec in _FEATURE_REGISTRY:
        env_key = spec.env_name or f"{ENV_PREFIX}{spec.env_var}"
        feature_env_key = f"{FEATURE_ENV_PREFIX}{spec.env_var}"
        if env_key in os.environ:
            if spec.env_parse is not None:
                value = spec.env_parse(os.environ[env_key])
            else:
                value = _feature_flag_bool(spec.env_var, defaults[spec.name])
            source = env_key
        elif feature_env_key in os.environ:
            value = env_bool(feature_env_key, defaults[spec.name])
            source = feature_env_key
        else:
            value = defaults[spec.name]
            source = "default_prod" if production else "default_test"
        flags[spec.name] = value
        sources[spec.name] = source

    known_names = set(defaults)
    runtime_path = runtime_flags_path()
    for name, value in runtime_flag_overrides(runtime_path, as_of=runtime_at).items():
        if name not in known_names:
            continue
        flags[name] = value
        sources[name] = f"runtime_file:{runtime_path}"

    if override:
        for name, value in override.items():
            if name not in known_names:
                continue
            flags[name] = bool(value)
            sources[name] = "override"

    return flags, sources


def get_features(
    *,
    production: bool = False,
    override: dict[str, bool] | None = None,
    _runtime_at: datetime | None = None,
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
    flags, _sources = _compute_feature_flags(
        production=production, override=override, runtime_at=_runtime_at
    )
    return Features(**flags)


def feature_sources(*, production: bool = False) -> dict[str, str]:
    """Return the source for each effective feature value."""
    _flags, sources = _compute_feature_flags(production=production)
    return sources


# Singleton for global access (lazy-loaded, thread-safe)
_features: Features | None = None
_features_lock = threading.Lock()
_features_runtime_path: Path | None = None
_features_runtime_mtime: float | None = None
_features_runtime_last_check = 0.0
_features_runtime_next_expiry: float | None = None


def _runtime_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _runtime_next_expiry(path: Path, *, loaded_at: datetime) -> float | None:
    # Inspect the raw records so an expiry that crosses between feature loading
    # and this calculation still schedules an immediate cache refresh.
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw = data.get("flags", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return None
    expiries = [
        _parse_runtime_expiry(record["expires_at"])
        for record in raw.values()
        if isinstance(record, dict) and "expires_at" in record
    ]
    valid = [expiry for expiry in expiries if expiry is not None and expiry > loaded_at]
    if not valid:
        return None
    seconds_left = min((expiry - _utc_now()).total_seconds() for expiry in valid)
    return time.monotonic() + max(0.0, seconds_left)


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
    global _features, _features_runtime_last_check, _features_runtime_mtime, _features_runtime_path
    global _features_runtime_next_expiry
    now = time.monotonic()
    path = runtime_flags_path()
    with _features_lock:
        if _features is None:
            loaded_at = _utc_now()
            _features = get_features(_runtime_at=loaded_at)
            _features_runtime_path = path
            _features_runtime_mtime = _runtime_mtime(path)
            _features_runtime_next_expiry = _runtime_next_expiry(path, loaded_at=loaded_at)
            _features_runtime_last_check = now
            return _features

        if now - _features_runtime_last_check >= RUNTIME_FLAGS_TTL_S:
            mtime = _runtime_mtime(path)
            expired = _features_runtime_next_expiry is not None and now >= _features_runtime_next_expiry
            if path != _features_runtime_path or mtime != _features_runtime_mtime or expired:
                loaded_at = _utc_now()
                _features = get_features(_runtime_at=loaded_at)
                _features_runtime_path = path
                _features_runtime_mtime = mtime
                _features_runtime_next_expiry = _runtime_next_expiry(path, loaded_at=loaded_at)
            _features_runtime_last_check = now
        return _features


def reset_features() -> None:
    """Reset the global Features instance (useful for tests).

    Call this to re-read feature flags from environment.
    """
    global _features, _features_runtime_last_check, _features_runtime_mtime, _features_runtime_path
    global _features_runtime_next_expiry
    with _features_lock:
        _features = None
        _features_runtime_path = None
        _features_runtime_mtime = None
        _features_runtime_last_check = 0.0
        _features_runtime_next_expiry = None


def set_features(new_features: Features) -> None:
    """Set the global Features instance (useful for tests).

    Args:
        new_features: Features instance to use globally.
    """
    global _features, _features_runtime_last_check, _features_runtime_mtime, _features_runtime_path
    global _features_runtime_next_expiry
    with _features_lock:
        loaded_at = _utc_now()
        _features = new_features
        _features_runtime_path = runtime_flags_path()
        _features_runtime_mtime = _runtime_mtime(_features_runtime_path)
        _features_runtime_next_expiry = _runtime_next_expiry(
            _features_runtime_path, loaded_at=loaded_at
        )
        _features_runtime_last_check = time.monotonic()
