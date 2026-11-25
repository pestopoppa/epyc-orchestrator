# Chapter 01: Runtime Environment & Configuration

## Introduction

The orchestration system's runtime environment provides a hierarchical configuration system with feature flags, environment-based overrides, and safety guardrails. This chapter documents the environment setup, configuration architecture, and runtime tuning parameters that control system behavior.

The configuration system uses **pydantic-settings** for type-safe environment variable parsing with fallback to manual parsing when unavailable. Feature flags enable modular system components to be toggled independently for testing and production deployment.

## Feature Flag System

Independent feature flags let us turn orchestration modules on and off without touching code. In test mode everything defaults to off for isolation; in production most flags flip on. The flags live in `src/features.py` and are validated at startup — if you enable `scripts` without `tools`, for example, initialization will complain. The registry has grown substantially since the initial release; the table below shows a representative subset for readability. For the complete current registry (78+ feature specs as of 2026-05-26), see `src/features.py:75-184`.

<details>
<summary>Core feature flags table (representative subset)</summary>

| Flag | Environment Variable | Purpose | Dependencies |
|------|---------------------|---------|--------------|
| `memrl` | `ORCHESTRATOR_MEMRL` | Memory-based RL (TaskEmbedder, QScorer) | numpy, sqlite3, sentence-transformers |
| `tools` | `ORCHESTRATOR_TOOLS` | TOOL() function in REPL | None |
| `scripts` | `ORCHESTRATOR_SCRIPTS` | SCRIPT() function for prepared scripts | `tools` |
| `streaming` | `ORCHESTRATOR_STREAMING` | SSE /chat/stream endpoint | None |
| `openai_compat` | `ORCHESTRATOR_OPENAI_COMPAT` | OpenAI-compatible /v1/* endpoints | None |
| `repl` | `ORCHESTRATOR_REPL` | Python REPL execution environment | None |
| `caching` | `ORCHESTRATOR_CACHING` | LLM response caching | None |
| `restricted_python` | `ORCHESTRATOR_RESTRICTED_PYTHON` | Use RestrictedPython sandbox | RestrictedPython>=7.0 |
| `generation_monitor` | `ORCHESTRATOR_GENERATION_MONITOR` | Early failure detection (Phase 6) | None |
| `mock_mode` | `ORCHESTRATOR_MOCK_MODE` | Mock responses (test safety) | None |
| `cascading_tool_policy` | `ORCHESTRATOR_CASCADING_TOOL_POLICY` | PolicyLayer chain for tool permissions | None |
| `compaction_failed_action_ledger` | `ORCHESTRATOR_COMPACTION_FAILED_ACTION_LEDGER` | HAR-COMPACT-LEDGER-1 (DMC-5 integrity): session compaction keeps every failed tool-call line verbatim in a ledger block and restores pinned pointers (paths, IDs, numbers) byte-exact when the compacted text lost them (default off) | `session_compaction` |
| `session_log` | `ORCHESTRATOR_SESSION_LOG` | Append-only REPL processing journal per task | None |
| `session_scratchpad` | `ORCHESTRATOR_SESSION_SCRATCHPAD` | Model-extracted semantic insights per turn | `session_log` |
| `structured_tool_output` | `ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT` | ToolOutput envelope (human + machine modes) | None |
| `observed_dependency_logging` | `ORCHESTRATOR_OBSERVED_DEPENDENCY_LOGGING` | HIL-15: per-tool-call read/write resource edges, observed grade (default off) | None |
| `tool_surface_logging` | `ORCHESTRATOR_TOOL_SURFACE_LOGGING` | UFH14-TOOLS-1: per-turn JSONL (`logs/tool_surface/`, override `ORCHESTRATOR_TOOL_SURFACE_LOG_DIR`) of REPL-global and registry-tool counts per role (default off) | None |
| `autopilot_speed_physical_bound` | `AUTOPILOT_SPEED_PHYSICAL_BOUND` | AP-72: SafetyGate marks a trial invalid when its reported speed exceeds the per-model ceiling in `orchestration/autopilot_speed_ceilings.yaml` (measured decode roofline x slots x 1.10); unknown model = no bound (default off; `AUTOPILOT_SPEED_PHYSICAL_CEILING_TPS` overrides the table) | None |
| `autopilot_lh_layer_taxonomy` | `AUTOPILOT_LH_LAYER_TAXONOMY` | AP-LH-DISTILL: four-layer harness taxonomy (action_realization / environment_contract / trajectory_regulation / procedural_skill) in the mutation proposer's failure context and the journal row (`eval_details.details.harness_layer`); annotation only (default off) | None |
| `autopilot_lh_trace_replay_precheck` | `AUTOPILOT_LH_TRACE_REPLAY_PRECHECK` | AP-LH-DISTILL: zero-model replay of a `structural_experiment` flag proposal against the failing trace(s) that motivated it; a proposal whose guard fires on none of them is skipped before its eval only when every motivating trace is a whole-trajectory capture (`complete=True`); the current tap-tail trace bank is not, so today the pre-check records `inconclusive` and never rejects (observe-only; default off) | None |
| `autopilot_lh_whole_trajectory_capture` | `AUTOPILOT_LH_WHOLE_TRAJECTORY_CAPTURE` | AP-LH-WHOLE-TRAJECTORY-CAPTURE: failure trials bank a whole-trajectory trace (inference-tap events joined by eval batch id, REPL-tap result blocks joined by task id, no tail cut) as `complete=True` / `source=whole_trajectory` only when completeness is proven, else `complete=False` (cap hit included); writes an id+length header before each REPL result block in `repl_tap.log` (default off) | None |
| `autopilot_alfworld_suite` | `AUTOPILOT_ALFWORLD_SUITE` | AP-ALFWORLD-SUITE: a small rotating ALFWorld slice (default 5 episodes, `train_valid` only, never the sealed `new_std`) on the journaled T1 eval, run in the Life-Harness native ALFWorld venv by a subprocess driver against the served model under test; rows carry split-qualified qids `alfworld:<split>:<index>`, sit in partition `alfworld` outside the scalar quality/speed aggregate, and the T1 `core_id` gains an `_alfw<n>_<hash>` suffix; knobs `AUTOPILOT_ALFWORLD_N`, `_MAX_STEPS`, `_EPISODE_WALL_S`, `_BATCH_WALL_S`, `_ROLE`, `_ENDPOINT` (default off) | None |
| `context_fold_digest` | `ORCHESTRATOR_CONTEXT_FOLD_DIGEST` | CTE-CFAB-A8 (CF-AB-1 arm A8): digest catalog of every turn older than the verbatim tail (one line `T<id> \| tool \| ok/error \| 1-2 sentences \| unfold: peek(...)`), the agent unfolds a turn by running the peek call in its line, catalog capped at 0.3 of `ORCHESTRATOR_CONTEXT_BUDGET_TOKENS`; arms `digest_unfold` and `digest_unfold_compressor` (second-role compressor from `ORCHESTRATOR_CONTEXT_COMPRESSOR_ROLE`, pinned pointers restored for error and unfolded turns) chosen by `ORCHESTRATOR_CONTEXT_ARM`; runs only after TOC-SP-5 shows a binding window (default off) | `context_arm_ab` |
| `trace_navigation_tools` | `ORCHESTRATOR_TRACE_NAVIGATION_TOOLS` | UTM-B1: read-only trace navigation as MCP `ms.search`/`ms.expand` (registered at MCP server startup) and REPL `ms_search`/`ms_expand` (shared tool broker, child lane included); every call is scoped to the caller's own session (default off) | None |
| `trace_navigation_cross_session` | `ORCHESTRATOR_TRACE_NAVIGATION_CROSS_SESSION` | UTM-B1: permits `scope="all"` on those tools to read trace rows from any session (default off) | `trace_navigation_tools` |
| `tool_surface_cap` | `ORCHESTRATOR_TOOL_SURFACE_CAP` | UFH14-TOOLS-1: `list_tools()` shows only the role's allowlist from `orchestration/tool_surface_allowlist.yaml`; the rest stay callable and `list_tools(deferred=True)` shows them (default off; empty allowlist = no cap) | None |
| `usage_root_capture` | `ORCHESTRATOR_USAGE_ROOT_CAPTURE` | AEI-TOKEN-ROOT-CAPTURE: one root row (`request_id`, `prompt_tokens`, `output_tokens_reported`, path `cpu`/`gpu`/`stream`/`unknown`) per completed `/chat` request appended to `usage_roots.jsonl` beside `usage_provenance.jsonl`, the `--root` input of `scripts/analysis/token_span_ratio.py` (default off; off writes nothing) | None |
| `serving_calls_json_schema_capture` | `ORCHESTRATOR_SERVING_CALLS_JSON_SCHEMA_CAPTURE` | SPEC-GRAM-JSON-1 enabler: `json_schema` calls in `serving_calls.jsonl` gain a `json_schema_capture` block (`schema_sha256` into `json_schema_store/<sha>.json` (canonical, sort_keys), `schema_raw_sha256` into `json_schema_store/raw/<raw_sha>.json` (original key order, replay-faithful), `output_text`, `request_class`, `prompt_sha256`, `prompt_body` -> `json_schema_bodies/<record_id>.json`); bodies pruned with log rotation, kill switch unchanged (default off; off = rows byte-identical, no files) | None |
| `chat_finish_reason` | `ORCHESTRATOR_CHAT_FINISH_REASON` | AEI-CHAT-FINISH-REASON: `/chat` response carries `finish_reason` and `truncated` (length, context_limit, context_overflow) from the backend completion_reason; omitted when off (default off) | None |
| `autopilot_alert_push` | `ORCHESTRATOR_AUTOPILOT_ALERT_PUSH` | AutoPilot pushes one deduped, rate-limited alert to `alerts.target_session` (env `ORCHESTRATOR_ALERTS_TARGET_SESSION`, empty = no-op) on health degraded / smoke fail / consecutive_failures threshold (default off) | None |
| `tool_process_cpu_pin` | `ORCHESTRATOR_TOOL_PROCESS_CPU_PIN` | Pin tool/REPL child processes via `os.sched_setaffinity` to `tools.cpu_set` (`ORCHESTRATOR_TOOLS_CPU_SET`, default `84-87`); startup warns if the set overlaps a serving role cpu_shape (default off) | None |
| `step_feedback_in_prompt` | `ORCHESTRATOR_STEP_FEEDBACK_IN_PROMPT` | Observed tool-feedback facts block appended to the next REPL prompt; requires `step_event_logging` (default off) | `step_event_logging` |
| `progress_aware_retry_cap` | `ORCHESTRATOR_PROGRESS_AWARE_RETRY_CAP` | Progressing retries are not charged: multitier per-tier attempt cap and SafetyGate consecutive-failure strikes skip attempts that ended incomplete (infra/timeout/partial) and scored new rows (completed failures always charged); AP-55 seed re-run cap unchanged (default off) | None |
| `empty_output_vacuity_guard` | `ORCHESTRATOR_EMPTY_OUTPUT_VACUITY_GUARD` | Vacuity guard: empty output with no tool calls fails at any elapsed time; batch-edit verify fails when nothing verifiable ran; `FINAL("")` refused; patch-path `#L<n>`/`:<n>` suffixes stripped (default off) | None |
| `test_result_skip_aware` | `ORCHESTRATOR_TEST_RESULT_SKIP_AWARE` | HF-SKIPS: pytest runs in gate_runner / run_tests parse passed/failed/skipped/xfailed and skip reasons from the full output; an all-skipped or zero-passed rc==0 run is inconclusive (cause `skipped`) instead of passed (default off) | None |
| `deploy_ancestry_guard` | `ORCHESTRATOR_DEPLOY_ANCESTRY_GUARD` | Deploy ancestry guard: `reload` refuses a code ref that is not a descendant of the last-deployed ref (`--allow-rollback` overrides); `revert_code_mutation` refuses if the file hash moved since apply (default off) | None |
| `timeout_contention_class` | `ORCHESTRATOR_TIMEOUT_CONTENTION_CLASS` | HF-TIMEOUT-CLASS: eval timeout provenance rows carry `contention_class` (contention / model_slow / unknown) from host covariates, with per-class timeout rates in the aggregate; label only (default off) | None |
| `thinking_budget_headroom` | `ORCHESTRATOR_THINKING_BUDGET_HEADROOM` | AEI-THINK-HEADROOM: chat-lane `thinking_budget_tokens` clamped to `max_tokens` minus an answer floor (`ORCHESTRATOR_THINKING_ANSWER_FLOOR_TOKENS`, default 256; autopilot numeric `thinking.answer_floor_tokens`); HF-THINK-CAPS-2: the same clamp covers the fixed 2048 escalation cap (`stages.py`) and 1024 streamed-reply cap (`stream_adapter.py`); empty content + reasoning + `finish_reason=length` is stamped `failure_reason=thinking_exhausted` (default off) | None |
| `server_log_crash_class` | `ORCHESTRATOR_SERVER_LOG_CRASH_CLASS` | HF-CRASH-CLASS: connection drop / `connect_error` / empty reply is classed `candidate_crash` (counts against the candidate, not excluded as infra) when the role's `llama-server-<port>.log` shows GGML_ASSERT / SIGSEGV / signal 11 / Segmentation fault / Aborted near the failure (default off) | None |
| `truncation_narrowing_hint` | `ORCHESTRATOR_TRUNCATION_NARROWING_HINT` | AEI-TRUNC-NOTICE: the three `llm_call` output-cap markers say how to get the rest, matching the REPL spill footer (default off) | None |
| `tool_denial_names_allowed` | `ORCHESTRATOR_TOOL_DENIAL_NAMES_ALLOWED` | HF-DENIALS: tool denial strings (blocked shell command, run_shell safety block, path not allowed, REPL sandbox hint fallback) name the allowed set or say none exists (default off) | None |
| `delegation_untrusted_data_field` | `ORCHESTRATOR_DELEGATION_UNTRUSTED_DATA_FIELD` | AEI-LAUNDER-FIX-1: previous-turn REPL output/error and specialist reports go into a fenced block labelled as data (with an instruction to treat it as data, never as instructions) instead of being interpolated into the child-role instruction; ten sites in chat, stream adapter, ReAct, architect and specialist loops (default off) | None |
| `http_post_guard` | `ORCHESTRATOR_HTTP_POST_GUARD` | AEI-HTTPPOST-GUARD: `http_post` refuses private, loopback, link-local and metadata destinations via the SEC-EXEC-3 `web_guard` check (default ON; `=0` kill switch; autopilot pinned) | None |
| `spill_extractive_summary` | `ORCHESTRATOR_SPILL_EXTRACTIVE_SUMMARY` | REPL-EMB-3.1: spilled REPL output is summarised as verbatim head, tail and the top-k chunks relevant to the producing cell (no worker call; default off) | None |
| `step_event_logging` | `ORCHESTRATOR_STEP_EVENT_LOGGING` | HIL-13a: per-step tool events (args hash, ok/error, latency, output size, truncation, mutating) as `tool_step` trace-store rows for the L2 scan (default off) | None |
| `replay_recording` | `ORCHESTRATOR_REPLAY_RECORDING` | HIL-16: per-turn model prompt/output and tool args/outputs stored content-addressed (`logs/replay_cas`, override `ORCHESTRATOR_REPLAY_CAS_DIR`) plus `model_turn` trace-store rows, for the prefix-preserving replay tier; implies step events (default off) | None |
| `exec_containment` | `ORCHESTRATOR_EXEC_CONTAINMENT` | SEC-EXEC-3/5: run_shell / run_python_code children run under Landlock + seccomp: no sockets, credential-file read-deny, other processes' `/proc/<pid>` hidden, token-free HF_HOME; refuses the call (logged) when the host cannot enforce (default off) | None |
| `web_fetch_guard` | `ORCHESTRATOR_WEB_FETCH_GUARD` | SEC-EXEC-3: web_fetch and fetch_docs refuse private, loopback, link-local, metadata and non-canonical-IP destinations, re-checked on every redirect hop and at connect time (default off) | None |
| `repl_child_lane` | `ORCHESTRATOR_REPL_CHILD_LANE` | D-RI-CONTAIN-T2: REPL cells run in a warm, killable, Landlock+seccomp+RLIMIT/cgroup-limited CPython child with the FULL REPL tool surface brokered to the parent (writers, TOOL/CALL, llm_call/llm_batch, run_shell, SCRIPT, web, discovery and the codemode helpers all run in the parent through the in-process callables; enumerated from the globals builder in `tool_broker.py`, asserted by test); state crosses via a typed JSON codec plus carried def/class source; fails closed, never inline; implies `repl_tool_arg_validation`; runtime-toggleable via `POST /config` (default off) | None |
| `repl_tool_arg_validation` | `ORCHESTRATOR_REPL_TOOL_ARG_VALIDATION` | D-RI-CONTAIN (e)/(a): per-tool argument validation inside every REPL-exposed tool, a bounded structured exception schema for REPL/tool errors, and bounded regex matching for grep-family tools (default off) | None |
| `exec_write_flags` | `ORCHESTRATOR_EXEC_WRITE_FLAGS` | SEC-EXEC-7: tool-call audit records carry `write_outside_root` (default off) | None |
| `final_schema_validation` | `ORCHESTRATOR_FINAL_SCHEMA_VALIDATION` | Validate FINAL() value against caller-supplied JSON Schema; retry-with-error on failure (2026-05-20) | None |
| `skillbank` | `ORCHESTRATOR_SKILLBANK` | SkillRL experience distillation | `memrl` |
| `worker_call_budget` | `ORCHESTRATOR_WORKER_CALL_BUDGET` | Cap total REPL executions per task (default 30) | None |
| `task_token_budget` | `ORCHESTRATOR_TASK_TOKEN_BUDGET` | Cap cumulative completion tokens per task (default 200K) | None |

</details>

<details>
<summary>Usage and validation patterns</summary>

<details>
<summary>Code: checking and validating feature flags</summary>

```python
from src.features import features

# Check if a feature is enabled
if features().memrl:
    from orchestration.repl_memory import TaskEmbedder
    embedder = TaskEmbedder()

# Get feature summary
enabled = features().enabled_features()
# Returns: ['repl', 'tools', 'caching', 'mock_mode']
```

```python
from src.features import get_features

features = get_features(production=True)
errors = features.validate()

if errors:
    # ['scripts feature requires tools feature']
    raise RuntimeError(f"Invalid configuration: {errors}")
```

</details>
</details>

## Hierarchical Configuration

Configuration is nested — LLM settings, escalation settings, REPL settings, and server settings each live in their own sub-config. Environment variables use double-underscore nesting (`ORCHESTRATOR_LLM__OUTPUT_CAP=4096`) to target specific leaves. The whole thing is backed by pydantic-settings so every value is type-checked and has sensible defaults.

<details>
<summary>Configuration hierarchy and environment variables</summary>

<details>
<summary>Code: environment variable examples</summary>

```bash
# Top-level settings
ORCHESTRATOR_MOCK_MODE=0
ORCHESTRATOR_DEBUG=1

# Nested LLM settings
ORCHESTRATOR_LLM__OUTPUT_CAP=4096
ORCHESTRATOR_LLM__BATCH_PARALLELISM=8
ORCHESTRATOR_LLM__CALL_TIMEOUT=300

# Nested escalation settings
ORCHESTRATOR_ESCALATION__MAX_RETRIES=3
ORCHESTRATOR_ESCALATION__MAX_ESCALATIONS=2

# Nested REPL settings
ORCHESTRATOR_REPL__MAX_OUTPUT_LEN=10000
ORCHESTRATOR_REPL__TIMEOUT_SECONDS=30
```

</details>

### Configuration Hierarchy

```
OrchestratorConfig
├── mock_mode: bool = True
├── debug: bool = False
├── llm: LLMConfig
│   ├── output_cap: int = 8192 (spill-to-file threshold)
│   ├── batch_parallelism: int = 4
│   ├── call_timeout: int = 120
│   ├── max_recursion_depth: int = 5
│   ├── default_prompt_rate: float = 0.50
│   └── default_completion_rate: float = 1.50
├── escalation: EscalationConfig
│   ├── max_retries: int = 2
│   ├── max_escalations: int = 2
│   └── optional_gates: frozenset = {"typecheck", "integration", "shellcheck"}
├── repl: REPLConfig
│   ├── max_output_len: int = 10000
│   ├── timeout_seconds: int = 30
│   ├── forbidden_modules: frozenset = {"os", "sys", "subprocess", ...}
│   └── forbidden_builtins: frozenset = {"eval", "exec", "open", ...}
├── server: ServerConfig
│   ├── default_url: str = "http://localhost:8080"
│   ├── timeout: int = 300
│   ├── num_slots: int = 4
│   ├── connect_timeout: int = 5
│   ├── retry_count: int = 3
│   └── retry_backoff: float = 0.5
├── monitor: MonitorConfig
│   ├── entropy_threshold: float = 2.5
│   ├── repetition_window: int = 50
│   ├── repetition_threshold: float = 0.3
│   └── min_tokens_before_abort: int = 20
├── paths: PathsConfig
│   ├── models_dir: Path = /mnt/raid0/llm/models
│   ├── cache_dir: Path = /mnt/raid0/llm/cache
│   ├── tmp_dir: Path = /mnt/raid0/llm/tmp
│   └── registry_path: Path = .../model_registry.yaml
└── features: FeaturesConfig
    ├── memrl: bool = False
    ├── tools: bool = False
    └── ... (10 feature flags)
```

<details>
<summary>Code: loading configuration at runtime</summary>

```python
from src.config import get_config

# Load from environment (cached)
config = get_config()

# Access nested settings
max_output = config.llm.output_cap
max_retries = config.escalation.max_retries
repl_timeout = config.repl.timeout_seconds

# Reset cache if env changes
from src.config import reset_config
reset_config()
```

</details>
</details>

## Environment Variables

Every cache, temp file, and data directory is redirected to the RAID array via environment variables. This is the project's most important safety measure — the 120GB OS drive has crashed before when large files landed on root. The path checks are enforced by hooks, but the variables should be set in every session regardless.

<details>
<summary>Required environment variables</summary>

<details>
<summary>Code: cache and path redirections</summary>

```bash
# HuggingFace/Transformers caches
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/raid0/llm/cache/huggingface
export HF_DATASETS_CACHE=/mnt/raid0/llm/cache/huggingface/datasets

# Python package cache
export PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip

# System temporary files
export TMPDIR=/mnt/raid0/llm/tmp

# XDG Base Directory Specification
export XDG_CACHE_HOME=/mnt/raid0/llm/epyc-orchestrator/cache
export XDG_DATA_HOME=/mnt/raid0/llm/epyc-orchestrator/share
export XDG_STATE_HOME=/mnt/raid0/llm/epyc-orchestrator/state
```

</details>

### Path Verification (Mandatory)

Before any file write operation:

```bash
# Verify path starts with /mnt/raid0/
[[ "$TARGET_PATH" == /mnt/raid0/* ]] || { echo "ERROR: Path not on RAID!"; exit 1; }
```

**Forbidden paths**: `/home/`, `/tmp/` (except via bind mount), `/var/`, `~/.cache/`, any path not starting with `/mnt/raid0/`.

</details>

### Graph Execution Controls

Environment variables that control REPL turn budgets and token caps. These are set in `orchestrator_stack.py` at startup.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORCHESTRATOR_REPL_TURN_N_TOKENS` | 5000 | Max completion tokens per REPL turn (was 768 prior to 2026-03-03) |
| `ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS` | 5000 | Max tokens for frontdoor non-tool turns |
| `ORCHESTRATOR_WORKER_CALL_BUDGET_CAP` | 30 | Max REPL execute() calls per task |
| `ORCHESTRATOR_TASK_TOKEN_BUDGET_CAP` | 200000 | Max cumulative completion tokens per task |
| `ORCHESTRATOR_CASCADING_TOOL_POLICY` | 1 | Enable PolicyLayer chain (legacy path denies all tools) |

Reviewer control-plane knobs (RPK-SURFACE; autopilot `numeric_swarm` surface `review_plane`, apply keys `delegation.*`, env_restart = API reload; declared in `orchestration/review_plane_knobs.yaml`). These knobs now take effect (RPK-READER): `DelegationConfig` reads them from the environment at construction, clamped to the range below, and `ReviewPlaneKnobs.from_config()` / `ProactiveDelegator` consume them (env_restart: applied on API reload). Defaults preserve current behaviour exactly; the autopilot writes these only while running a trial.

| Variable | Default | Range | Purpose |
|----------|---------|-------|---------|
| `ORCHESTRATOR_DELEGATION_REVIEW_TRIGGER_COMPLEXITY_THRESHOLD` | 0 | 0-3 | Per-subtask review runs only at or above this TaskComplexity rank |
| `ORCHESTRATOR_DELEGATION_MAX_REVIEW_ITERATIONS` | 3 | 1-5 | Review turns per task on the shared iteration ledger |
| `ORCHESTRATOR_DELEGATION_REMINDER_CADENCE` | 0 | 0-50 | Steps between plan-reminder re-injections (0 = off) |
| `ORCHESTRATOR_DELEGATION_REVIEW_MAJORITY_K` | 1 | 1-5 | Grading passes for near-band majority vote |
| `ORCHESTRATOR_DELEGATION_REQUEST_EVIDENCE_ROUND_BUDGET` | 0 | 0-3 | request_evidence follow-up rounds |
| `ORCHESTRATOR_DELEGATION_REVIEW_TOKEN_MULTIPLIER` | 2.0 | 1.0-5.0 | Governed review token multiplier; caps majority-k grading passes at floor(m) |
| `ORCHESTRATOR_DELEGATION_REVIEW_NEAR_BAND_WIDTH` | 0.5 | 0.0-0.5 | Majority-k fan-out only when first-pass score is within this distance of 0.5 (0.5 = every review) |

### Structured Output & Budget Interaction

The REPL turn token budget (`ORCHESTRATOR_REPL_TURN_N_TOKENS`) is shared across regular completion output, the `structured_tool_output` envelope (when enabled), and any `session_log` / `session_scratchpad` summary injection. When both `structured_tool_output` and the session log/scratchpad features are enabled, the effective budget available for code-execution prose may be noticeably lower than the raw token cap suggests.

When `final_schema_validation` is enabled (2026-05-20, default off), validation failures on `FINAL()` values inject a retry-with-error into the next REPL turn. Retries consume completion tokens within the same per-turn budget and decrement the `repl_executions` counter; the loop is bounded by `ORCHESTRATOR_WORKER_CALL_BUDGET_CAP` so there is no new runaway risk. See `handoffs/completed/repl-final-schema-validation.md` for the full design.

### Path-Config Module Refactor (2026-05-22)

The cache redirects above remain canonical. As of 2026-05-22, the environment-variable construction was centralized in `scripts/server/stack_env.py` (`_CANONICAL_OMP_ENV`, `_ROLE_ENV_BLOCKS`) and path constants in `scripts/server/stack_paths.py`. These are implementation-level changes — documented behavior is unchanged — but operator runbooks should reference the new module layout when modifying startup paths.

## OMP & NUMA Runtime Tuning

Every inference command needs the same three-setting prefix: single OMP thread (llama.cpp manages its own parallelism), NUMA interleaving across all 12 memory channels, and 96 physical cores. Getting any of these wrong can halve throughput or worse.

<details>
<summary>Thread and NUMA configuration</summary>

<details>
<summary>Code: standard inference prefix</summary>

```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m model.gguf -t 96 -p "prompt"
```

</details>

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `OMP_NUM_THREADS` | 1 | Disable OpenMP parallelism (llama.cpp handles threading) |
| `-t` | 96 | Use all physical cores (96-core EPYC 9655) |
| `--interleave=all` | Required | Interleave memory across 12 DDR5 channels (~460 GB/s) |

### NUMA Architecture

The EPYC 9655 has **12 memory channels** (1.13TB DDR5-5600 ECC). Using `--interleave=all` ensures memory bandwidth is maximized across all channels rather than binding to a single NUMA node.

**Do NOT use** `numactl --cpunodebind=0` or similar node-specific bindings for large models - this restricts memory bandwidth to 2-4 channels and cuts throughput by 60-75%.

</details>

## Python Environment

The project runs in a dedicated virtual environment called `pace-env`, managed with **uv** for fast installs. Everything from FastAPI to sentence-transformers lives here.

<details>
<summary>Environment setup and packages</summary>

<details>
<summary>Code: activation and installation</summary>

```bash
# Environment name
pace-env

# Activation
source /mnt/raid0/llm/pace-env/bin/activate

# Key packages
# - FastAPI + uvicorn (API server)
# - pydantic + pydantic-settings (config/validation)
# - httpx (async HTTP client)
# - numpy, sentence-transformers (MemRL)
# - RestrictedPython (REPL sandbox)
```

```bash
# Create environment with uv
uv venv /mnt/raid0/llm/pace-env

# Install dependencies
uv pip install -r /mnt/raid0/llm/epyc-orchestrator/requirements.txt
```

</details>
</details>

## Session Initialization

Every session starts with an initialization script that verifies the environment, discovers available models, and checks branch safety. Skipping this step risks running benchmarks on the wrong llama.cpp branch or missing models that have been added since the last session.

<details>
<summary>Initialization steps and verification</summary>

<details>
<summary>Code: session startup commands</summary>

```bash
# Set environment variables
source /mnt/raid0/llm/epyc-orchestrator/scripts/utils/agent_log.sh
agent_session_start "Session purpose"

# Discover models and verify llama.cpp branch
bash /mnt/raid0/llm/epyc-orchestrator/scripts/session/session_init.sh
```

</details>

**Known issue (2026-05-22)**: `scripts/utils/agent_log.sh` has an unmet dependency on `scripts/lib/env.sh` and should be fixed in a separate governance pass. Until then, set environment variables manually or invoke the `stack_env` module directly.

The initialization script:
1. Checks llama.cpp is on `production-consolidated` branch
2. Scans `/mnt/raid0/llm/models/` for GGUF files
3. Validates model registry against discovered models
4. Checks free memory (100GB minimum for tests)
5. Verifies environment variables point to `/mnt/raid0/`

</details>

<details>
<summary>References</summary>

- `src/features.py` - Feature flag system implementation
- `src/config.py` - Hierarchical configuration with pydantic-settings
- `scripts/session/session_init.sh` - Environment initialization
- `scripts/utils/agent_log.sh` - Session logging utilities

</details>

---

*Next: [Chapter 02: Orchestration Architecture](02-orchestration-architecture.md)*
