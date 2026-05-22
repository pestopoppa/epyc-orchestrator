# Orchestration Refactoring Handoff

Status: ready for delegation
Repo: `/workspace/repos/epyc-orchestrator`
Index checked: GitNexus local index is available for this checkout at commit `f5c83d9`

This handoff converts the large-file refactor sweep into small, ordered tasks for junior agents. It also documents the newly required GitNexus setup and verification steps that must happen before any code movement.

## Non-Negotiable GitNexus Workflow

GitNexus must be available before refactoring starts. The local CLI was repaired for this checkout by running:

```bash
NPM_CONFIG_CACHE=/tmp/npm-cache-gitnexus npx -y gitnexus@1.6.5 analyze .
```

The stale duplicate index for `/mnt/raid0/llm/epyc-orchestrator` was removed. The active index is:

```text
Path: /workspace/repos/epyc-orchestrator
Commit: f5c83d9
Stats: 935 files, 39414 symbols, 68272 edges, 300 flows
```

Before starting work, each agent must confirm:

```bash
NPM_CONFIG_CACHE=/tmp/npm-cache-gitnexus npx -y gitnexus@1.6.5 status
NPM_CONFIG_CACHE=/tmp/npm-cache-gitnexus npx -y gitnexus@1.6.5 list
```

Expected `status` output should say the repository is `/workspace/repos/epyc-orchestrator` and `Status: up-to-date`.

Because `epyc-root` is also indexed, all GitNexus calls must include:

```bash
--repo epyc-orchestrator
```

Before editing any function, class, or method, run both:

```bash
NPM_CONFIG_CACHE=/tmp/npm-cache-gitnexus npx -y gitnexus@1.6.5 context --repo epyc-orchestrator --file <path> <symbol>
NPM_CONFIG_CACHE=/tmp/npm-cache-gitnexus npx -y gitnexus@1.6.5 impact --repo epyc-orchestrator --depth 2 <symbol>
```

If a symbol is ambiguous, use the UID returned by `context`:

```bash
NPM_CONFIG_CACHE=/tmp/npm-cache-gitnexus npx -y gitnexus@1.6.5 context --repo epyc-orchestrator --uid '<uid>'
NPM_CONFIG_CACHE=/tmp/npm-cache-gitnexus npx -y gitnexus@1.6.5 impact --repo epyc-orchestrator --depth 2 '<uid>'
```

After each tranche, run:

```bash
NPM_CONFIG_CACHE=/tmp/npm-cache-gitnexus npx -y gitnexus@1.6.5 detect-changes --repo epyc-orchestrator
```

Do not proceed silently on HIGH or CRITICAL impact. Report the blast radius in the work log and get the task lead's confirmation before changing that symbol's behavior or public interface.

## Refactor Strategy

The safest sequence is:

1. Split pure benchmark adapter code while preserving the existing `dataset_adapters.py` import surface.
2. Extract shared seeding helpers from the nearly duplicated v1/v2 routing seed scripts, preserving wrapper functions.
3. Split chat delegation internals behind the existing `chat_delegation.py` facade.
4. Split feature validation harness classes.
5. Defer stateful production systems and graph/session internals to senior-owned tranches.

The main rule is compatibility first. For the first pass, moved code should be re-exported by the original module so existing imports and monkeypatch paths keep working.

## Target Ranking

| Rank | Target | Recommendation | GitNexus Risk | Why |
| --- | --- | --- | --- | --- |
| 1 | `scripts/benchmark/dataset_adapters.py` | Do first | LOW per adapter, HIGH facade/base | Mostly isolated adapter classes; high payoff with low behavior risk if facade remains |
| 2 | `scripts/benchmark/seed_specialist_routing.py` and `_v2.py` | Do second | LOW for `run_batch_3way`, HIGH/CRITICAL for v1 sampling helpers | 97.9 percent similar; clear shared helper extraction |
| 3 | `src/api/routes/chat_delegation.py` | Do third | LOW for internal delegation symbols | Existing `chat_pipeline/delegation_stage.py` already provides boundary |
| 4 | `scripts/benchmark/feature_validation.py` | Good junior task | LOW | Self-contained CLI harness |
| 5 | `src/repl_environment/environment.py` | Senior-owned | HIGH for `_execute_structured` | Affects OpenAI-compatible streaming and REPL execution |
| 6 | `orchestration/repl_memory/episodic_store.py` | Senior-owned | HIGH for store classes | 28 direct importers and persistent data concerns |
| 7 | `src/graph/session_log.py` | Senior-owned | CRITICAL for turn/session functions | Graph execution flow impact |
| 8 | `src/config/models.py` | Defer | MEDIUM | Broad import fanout, already domain-grouped |

## Task A: Split Dataset Adapters

Primary file:

```text
scripts/benchmark/dataset_adapters.py
```

Observed structure:

```text
get_adapter                       line 55
BaseAdapter                       lines 82-165
MMLUAdapter                       lines 171-240
MathAdapter                       lines 246-366
CoderAdapter                      lines 372-483
ThinkingAdapter                   lines 489-601
IFEvalAdapter                     lines 607-705
VLAdapter                         lines 711-752
GaiaAdapter                       lines 758-862
CRUXEvalAdapter                   lines 868-960
BigCodeBenchAdapter               lines 966-1041
GPQAAdapter                       lines 1047-1133
SimpleQAAdapter                   lines 1139-1211
HotpotQAAdapter                   lines 1217-1296
LiveCodeBenchAdapter              lines 1302-1474
DebugBenchAdapter                 lines 1480-1630
USACOAdapter                      lines 1636-1761
```

GitNexus impact notes:

```text
get_adapter: HIGH
  direct callers:
  - scripts/benchmark/seed_specialist_routing.py:_load_from_dataset_adapter
  - scripts/benchmark/seed_specialist_routing_v2.py:_load_from_dataset_adapter
  - scripts/benchmark/question_pool.py:build_pool
  - scripts/benchmark/deprecated/seed_specialist_routing_v1.py:_load_from_dataset_adapter
  affected processes include seed routing mains and autopilot dispatch_action.

BaseAdapter: HIGH
  direct dependents are every adapter subclass plus importer files.

Individual adapters checked: LOW
  MMLUAdapter, MathAdapter, CoderAdapter, LiveCodeBenchAdapter,
  DebugBenchAdapter, USACOAdapter all reported LOW in isolation.
```

New required steps:

1. Create a package:

```text
scripts/benchmark/dataset_adapter_modules/
```

2. Move only `BaseAdapter` first:

```text
dataset_adapter_modules/base.py
```

Keep the class name unchanged. Update imports in the same tranche. Run impact on `BaseAdapter` before moving.

3. Move individual adapter groups in separate commits or reviewable patches:

```text
dataset_adapter_modules/general.py
  MMLUAdapter
  GPQAAdapter
  SimpleQAAdapter
  HotpotQAAdapter

dataset_adapter_modules/math.py
  MathAdapter

dataset_adapter_modules/coding.py
  CoderAdapter
  CRUXEvalAdapter
  BigCodeBenchAdapter
  LiveCodeBenchAdapter
  DebugBenchAdapter
  USACOAdapter

dataset_adapter_modules/reasoning.py
  ThinkingAdapter
  IFEvalAdapter

dataset_adapter_modules/vision_agentic.py
  VLAdapter
  GaiaAdapter
```

4. Add the registry module:

```text
dataset_adapter_modules/registry.py
```

It should define:

```python
ADAPTER_SUITES = {...}
YAML_ONLY_SUITES = {...}
ADAPTER_CLASSES = {
    "general": MMLUAdapter,
    ...
}

def get_adapter(suite: str) -> BaseAdapter | None:
    cls = ADAPTER_CLASSES.get(suite)
    return cls() if cls is not None else None
```

5. Keep `dataset_adapters.py` as a compatibility facade:

```python
from scripts.benchmark.dataset_adapter_modules.base import BaseAdapter
from scripts.benchmark.dataset_adapter_modules.registry import (
    ADAPTER_SUITES,
    YAML_ONLY_SUITES,
    get_adapter,
)
from scripts.benchmark.dataset_adapter_modules.general import (
    MMLUAdapter,
    GPQAAdapter,
    SimpleQAAdapter,
    HotpotQAAdapter,
)
...
```

6. Preserve all public names. Do not rename any adapter class, suite key, prompt field, scoring field, or config field.

7. Keep imports lazy where they already are lazy. In particular, HuggingFace `datasets` imports must remain inside `_ensure_loaded()`.

Verification:

```bash
pytest tests/unit/test_seed_specialist_routing_helpers.py \
       tests/unit/test_seed_specialist_routing_v2_helpers.py \
       tests/unit/test_seeding_consumers.py \
       tests/unit/test_seeding_legacy.py

python - <<'PY'
from scripts.benchmark.dataset_adapters import get_adapter, ADAPTER_SUITES
for suite in sorted(ADAPTER_SUITES):
    adapter = get_adapter(suite)
    assert adapter is not None, suite
print("ok", len(ADAPTER_SUITES))
PY
```

Acceptance criteria:

- `from dataset_adapters import get_adapter, ADAPTER_SUITES` still works when `scripts/benchmark` is on `sys.path`.
- `scripts/benchmark/question_pool.py` does not need behavior changes.
- `dataset_adapters.py` drops from 1761 lines to a facade of imports and constants.
- `gitnexus detect-changes` shows expected adapter symbols only.

## Task B: Extract Shared Seeding Helpers From V1/V2

Primary files:

```text
scripts/benchmark/seed_specialist_routing.py
scripts/benchmark/seed_specialist_routing_v2.py
```

Observed duplication:

```text
line similarity: 0.979
v1 length: 1657 lines
v2 length: 1606 lines
```

Large top-level functions:

```text
main                         v1 lines 862-1607, v2 lines 853-1556
run_batch_3way               v1 lines 347-725,  v2 lines 347-716
sample_unseen_questions      both lines 273-341
print_3way_summary           both around 52 lines
_load_from_yaml              both 40 lines
_build_retrieval_config...   both 31 lines
_load_from_dataset_adapter   both 25 lines
_apply_profile               both 23 lines
```

GitNexus impact notes:

```text
v1 run_batch_3way: LOW
  direct caller: seed_specialist_routing.py:main

v2 run_batch_3way: LOW
  direct caller: seed_specialist_routing_v2.py:main

v1 sample_unseen_questions: CRITICAL
  direct callers include:
  - scripts/autopilot/species/seeder.py:Seeder.run_batch
  - scripts/benchmark/seed_specialist_routing.py:run_batch_3way
  - scripts/benchmark/seeding_legacy.py:run_batch
  affected processes include autopilot dispatch_action and both seeding mains.

v2 sample_unseen_questions: LOW
  direct caller: v2 run_batch_3way

v1 _load_from_dataset_adapter: HIGH
  indirect impact through v1 sample_unseen_questions and autopilot.

v2 _load_from_dataset_adapter: LOW
```

New required steps:

1. Do not delete either script. Treat both as public CLI entrypoints.

2. Create:

```text
scripts/benchmark/seeding_sampling.py
```

Move shared pure loading/sampling logic here first:

```python
load_from_dataset_adapter(...)
load_from_yaml(...)
sample_unseen_questions(...)
```

Keep wrappers in both original files:

```python
def _load_from_dataset_adapter(...):
    return seeding_sampling.load_from_dataset_adapter(...)

def _load_from_yaml(...):
    return seeding_sampling.load_from_yaml(...)

def sample_unseen_questions(...):
    return seeding_sampling.sample_unseen_questions(...)
```

Use wrapper names exactly as they are now. Tests monkeypatch these names directly.

3. Add dependency injection points to the shared helper instead of hard-coding module globals:

```python
debug_prompts_dir: Path
logger: logging.Logger
allow_reseen: bool
use_pool: bool
```

This is required because v1/v2 tests replace `DEBUG_PROMPTS_DIR`, `question_pool`, and loader functions.

4. Create:

```text
scripts/benchmark/seeding_cli_config.py
```

Move shared:

```text
_PROFILE_PRESETS
_apply_profile
_build_retrieval_config_from_args
```

Again, keep wrappers in both original scripts until all tests are updated.

5. Only after sampling and profile extraction pass tests, consider a shared batch runner:

```text
scripts/benchmark/seeding_batch.py
```

Proposed interface:

```python
@dataclass
class BatchOptions:
    allow_question_overrides: bool = False
    emit_prompt_in_progress: bool = False

def run_batch_3way_impl(..., options: BatchOptions) -> list[ThreeWayResult]:
    ...
```

Keep v1/v2 public wrappers:

```python
def run_batch_3way(...):
    return run_batch_3way_impl(..., options=...)
```

Do not start with this step. `run_batch_3way` is large and touches health checks, retry diagnostics, checkpointing, reward injection, progress callbacks, and debugger behavior.

6. Preserve the v1-only `questions_override` behavior. In the current diff, v1 accepts `questions_override`; v2 does not.

7. Preserve the progress callback difference. v2 currently passes prompt text into `on_progress`; v1's annotation still says four args. Before merging callback logic, inspect all TUI/progress call sites.

Verification:

```bash
pytest tests/unit/test_seed_specialist_routing_helpers.py \
       tests/unit/test_seed_specialist_routing_v2_helpers.py \
       tests/unit/test_seed_specialist_routing_main_and_retry.py \
       tests/unit/test_seeding_legacy.py \
       tests/unit/test_seeding_checkpoint.py \
       tests/unit/test_seeding_eval.py \
       tests/unit/test_seeding_injection.py
```

Acceptance criteria:

- Both original CLIs still import and parse.
- Existing monkeypatch paths in tests continue to work.
- Shared helpers have tests of their own before wrappers are removed.
- No behavior changes to seen-question filtering, pool fallback, or YAML fallback.

## Task C: Split Chat Delegation Internals

Primary files:

```text
src/api/routes/chat_delegation.py
src/api/routes/chat_pipeline/delegation_stage.py
```

Existing stage boundary:

```text
src/api/routes/chat_pipeline/delegation_stage.py:_execute_delegated
  calls src/api/routes/chat_delegation.py:_architect_delegated_answer
```

Large symbols in `chat_delegation.py`:

```text
_architect_delegated_answer_inner     lines 1384-1751, 368 lines
_run_specialist_loop                  lines 947-1293, 347 lines
_run_architect_decision               lines 643-797, 155 lines
_parse_architect_decision             lines 440-592, 153 lines
_apply_decision_guards                lines 800-944, 145 lines
DelegationConfig                      lines 62-165, 104 lines
_architect_delegated_answer           lines 1296-1381, 86 lines
```

GitNexus impact notes:

```text
_architect_delegated_answer: LOW
  direct caller: src/api/routes/chat_pipeline/delegation_stage.py:_execute_delegated

_architect_delegated_answer_inner: LOW
_run_specialist_loop: LOW
_run_architect_decision: LOW
_apply_decision_guards: LOW
_parse_architect_decision: LOW
```

Risk caveat:

Tests patch private names in `src.api.routes.chat_delegation`. The first tranche must preserve those names in `chat_delegation.py`, either as imports or wrapper functions.

New required steps:

1. Create:

```text
src/api/routes/chat_delegation_config.py
```

Move:

```text
_VALID_DELEGATE_ROLES
_delegation_local
_get_delegation_depth
DelegationConfig
_delegation_config
_delegation_specialist_turn_token_cap
```

Re-export these from `chat_delegation.py`.

2. Create:

```text
src/api/routes/chat_delegation_reports.py
```

Move:

```text
_trim_block
_store_report_handle
_to_report_handle_text
_build_compact_specialist_prompt
_maybe_summarize_specialist_report
_compress_report_for_loop
```

Keep the public behavior around `store_report()` unchanged.

3. Create:

```text
src/api/routes/chat_delegation_decision.py
```

Move:

```text
_strip_think
_extract_toon_decision
_parse_architect_decision
_architect_decision_token_budget
_architect_compute_token_budget
_classify_failure_reason
_apply_decision_guards
```

Do not change accepted response formats. The parser currently handles TOON, JSON-ish, and fallback text decisions.

4. Create:

```text
src/api/routes/chat_delegation_loop.py
```

Move last:

```text
_run_architect_decision
_run_specialist_loop
_architect_delegated_answer
_architect_delegated_answer_inner
```

This is the behavior-heavy tranche. It should happen only after config/report/decision splits are green.

5. Keep `chat_delegation.py` as facade during the transition:

```python
from .chat_delegation_config import ...
from .chat_delegation_decision import ...
from .chat_delegation_reports import ...
from .chat_delegation_loop import ...
```

6. If tests patch `chat_delegation.REPLEnvironment`, preserve that patch point. Options:

- Leave loop functions in `chat_delegation.py` until tests are migrated.
- Or make `chat_delegation_loop.py` accept `repl_environment_cls` as an injectable dependency, with facade wrappers passing `REPLEnvironment`.

Do not move `REPLEnvironment` imports blindly; tests patch that exact path.

Verification:

```bash
pytest tests/unit/test_architect_delegation.py \
       tests/unit/test_chat_delegation.py \
       tests/unit/test_chat_pipeline_stages.py \
       tests/integration/test_chat_pipeline.py \
       tests/unit/test_api_imports.py
```

Acceptance criteria:

- `_execute_delegated` still imports or calls `_architect_delegated_answer` successfully.
- Tests that patch `src.api.routes.chat_delegation._run_architect_decision`,
  `_run_specialist_loop`, or `REPLEnvironment` still pass.
- `chat_delegation.py` becomes a small facade plus compatibility patch points.

## Task D: Split Feature Validation Harness

Primary file:

```text
scripts/benchmark/feature_validation.py
```

Large symbols:

```text
OfflineValidator          lines 353-855, 503 lines
LiveValidator             lines 878-1078, 201 lines
_build_profiles           lines 120-229, 110 lines
ReportGenerator           lines 1084-1148, 65 lines
main                      lines 1167-1238, 72 lines
```

GitNexus impact notes:

```text
OfflineValidator: LOW
LiveValidator: LOW
ReportGenerator: LOW
_build_profiles: LOW
```

New required steps:

1. Create:

```text
scripts/benchmark/feature_validation_profiles.py
```

Move:

```text
TestSpec
FeatureProfile
MetricSnapshot
ComparisonReport
_build_profiles
```

2. Create:

```text
scripts/benchmark/feature_validation_offline.py
```

Move:

```text
OfflineValidator
```

3. Create:

```text
scripts/benchmark/feature_validation_live.py
```

Move:

```text
LiveValidator
_read_meminfo_mb
_hot_reload_feature
_ensure_stack_running
_verify_health_mid_run
_load_prompt_manifest
_write_incremental
_now_iso
```

Keep helpers near `LiveValidator` unless `OfflineValidator` also uses them.

4. Create:

```text
scripts/benchmark/feature_validation_report.py
```

Move:

```text
ReportGenerator
```

5. Keep `feature_validation.py` as CLI facade:

```text
imports
_parse_args
main
if __name__ == "__main__"
```

Verification:

```bash
pytest tests/unit/test_config_validation.py tests/unit/test_kv_compress_adaptive.py
python scripts/benchmark/feature_validation.py --help
```

If there are no direct feature validation tests, add import smoke tests for each new module and one `_build_profiles()` structure test.

Acceptance criteria:

- CLI help still works.
- Offline and live validators can be imported independently.
- No behavior change to profile names or output JSON shape.

## Task E: REPL Environment Is Senior-Owned

Primary file:

```text
src/repl_environment/environment.py
```

Large symbols:

```text
REPLEnvironment                         lines 108-1358, 1251 lines
REPLEnvironment._execute_structured      lines 702-1043, 342 lines
REPLEnvironment._execute_dependency_aware_chain lines 1045-1251, 207 lines
REPLEnvironment._build_globals           lines 329-476, 148 lines
REPLEnvironment.__init__                 lines 141-282, 142 lines
REPLEnvironment.execute                  lines 1253-1358, 106 lines
```

GitNexus impact notes:

```text
REPLEnvironment: LOW at class import level, but broad import fanout.
_execute_structured: HIGH
  affected processes:
  - src/api/routes/chat_pipeline/stream_adapter.py:_stream_repl
  - src/api/routes/openai_compat.py:generate_stream
  - src/api/routes/openai_compat.py:openai_chat_completions

_execute_dependency_aware_chain: LOW direct, but it sits under _execute_structured.
_build_globals: many direct unit tests and ambiguous method name.
```

New required steps before any split:

1. Add characterization tests for:

```text
structured tool call parsing
dependency-aware chain execution
parallel dispatch behavior
spill output behavior
credential redaction
tool hint output
safe globals and injected helper functions
```

2. Extract only leaf helpers first:

```text
src/repl_environment/globals_builder.py
src/repl_environment/output_spill.py
src/repl_environment/execution_result.py
```

3. Do not move `_execute_structured` until the leaf helpers are isolated and tests are green.

4. If `_build_globals` is moved, preserve the method as:

```python
def _build_globals(self):
    return build_globals(...)
```

Tests:

```bash
pytest tests/unit/test_repl_environment.py \
       tests/unit/test_tool_chaining.py \
       tests/unit/test_allowed_callers.py \
       tests/unit/test_credential_redaction.py \
       tests/integration/test_model_tool_compliance.py \
       tests/integration/test_chat_pipeline.py
```

This is not a junior-first task.

## Task F: Episodic Store Is Senior-Owned

Primary file:

```text
orchestration/repl_memory/episodic_store.py
```

Large symbols:

```text
EpisodicStore                    lines 114-829, 716 lines
GraphEnhancedStore               lines 877-1103, 227 lines
MemoryEntry                      lines 59-111, 53 lines
retrieve_by_similarity           lines 401-492, 92 lines
update_q_value                   lines 494-566, 73 lines
get_all_memories                 lines 605-673, 69 lines
store                            lines 255-319, 65 lines
```

GitNexus impact notes:

```text
EpisodicStore: HIGH
  direct importers: 28
  affected areas include procedure registry, REPL routing, classifier retriever,
  memory seed scripts, graph router scripts, API memrl service, replay engine.

GraphEnhancedStore: HIGH
  similar import fanout.

EpisodicStore.store: process involvement in seed_memory and replay.
retrieve_by_similarity: tied to MemRL retrieval, replay, FAISS tests, graph integration.
get_q_outliers: LOW, only seed_graphs path.
```

New required steps before splitting:

1. Add or refresh round-trip tests for:

```text
store()
store_immediate()
retrieve_by_similarity()
update_q_value()
get_all_memories()
FAISS enabled and disabled modes
GraphEnhancedStore wrapper behavior
```

2. Split leaf data model first:

```text
orchestration/repl_memory/memory_entry.py
```

Move:

```text
MemoryEntry
extract_symptoms
```

Keep re-exports from `episodic_store.py`.

3. Split embedding/index logic second:

```text
orchestration/repl_memory/episodic_index.py
```

Move only methods that wrap `_embedding_store` and FAISS/numpy behavior. Do not alter storage schema in this tranche.

4. Split persistence SQL last:

```text
orchestration/repl_memory/episodic_persistence.py
```

Do not change table names, column names, migration behavior, or default paths.

Tests:

```bash
pytest tests/unit/test_episodic_store.py \
       tests/unit/test_episodic_store_assigned_role.py \
       tests/unit/test_faiss_store.py \
       tests/unit/test_graph_integration.py \
       tests/unit/test_memrl_service.py \
       tests/unit/test_warm_start.py
```

This is not a junior-first task.

## Task G: QScorer Split

Primary file:

```text
orchestration/repl_memory/q_scorer.py
```

Large symbols:

```text
QScorer                              lines 135-867, 733 lines
_score_task                          lines 210-325, 116 lines
_compute_reward                      lines 361-473, 113 lines
score_external_result                lines 770-867, 98 lines
_compute_spo_plus_adjustment         lines 553-648, 96 lines
_compute_contrastive_adjustment      lines 475-551, 77 lines
ClaudeAsJudge                        lines 870-1030, 161 lines
```

GitNexus impact notes:

```text
QScorer: MEDIUM
  direct importers include claude_debugger, q_scorer_runner,
  seed_specialist_routing.py, seed_specialist_routing_v2.py,
  feature_validation.py, retriever.py, memrl.py, replay modules.

_compute_reward: LOW
  direct callers include _score_task and ReplayEngine._replay_step.

_compute_spo_plus_adjustment: LOW
_compute_contrastive_adjustment: LOW
```

New required steps:

1. Start with pure scoring math:

```text
orchestration/repl_memory/q_reward.py
```

Move:

```text
reward computation helpers
contrastive adjustment
SPO+ adjustment
```

Keep methods on `QScorer` as wrappers until tests are migrated.

2. Move judge wrapper separately:

```text
orchestration/repl_memory/q_judge.py
```

Move:

```text
ClaudeAsJudge
```

3. Do not change `QScorer` constructor or `score_pending_tasks()` in first tranche.

Tests:

```bash
pytest tests/unit/test_q_scorer.py \
       tests/unit/test_memrl_service.py \
       tests/unit/test_plan_review.py \
       tests/unit/test_replay_meta_agent_objective.py
```

## Task H: Session Log Is Senior-Owned

Primary file:

```text
src/graph/session_log.py
```

GitNexus impact notes:

```text
TurnRecord: LOW
ConsolidatedSegment: LOW
SegmentCache: LOW

build_turn_record: CRITICAL
append_turn_record: CRITICAL
summarize_session_with_worker: CRITICAL
  affected processes include graph nodes and langgraph nodes:
  coder_escalation_node, architect_coding_node, ingest_node,
  architect_node, coder_node, and src/graph/nodes.py run paths.
```

New required steps:

1. Only split data classes first:

```text
src/graph/session_records.py
```

Move:

```text
TurnRecord
ConsolidatedSegment
SegmentCache
RewardSignals
ScratchpadEntry
```

Re-export from `session_log.py`.

2. Do not move `build_turn_record`, `append_turn_record`, or `summarize_session_with_worker` in a junior tranche.

3. Add import compatibility tests before any split.

Tests:

```bash
pytest tests/unit/test_session_log.py \
       tests/unit/test_session_summary.py \
       tests/unit/test_langgraph_phase2.py \
       tests/unit/test_graph_compaction_budgets.py \
       tests/unit/test_context_compactor.py
```

## Task I: Config Models Should Be Deferred

Primary file:

```text
src/config/models.py
```

GitNexus impact notes:

```text
LLMConfig and OrchestratorConfigData: MEDIUM
  imported through src/config/__init__.py with broad fanout across API,
  backends, tools, graph, runtime, services, sessions, and benchmark scripts.
```

This file is already domain-grouped by dataclass. Do not split it unless there is a config-specific goal. If it is split later, keep `src/config/__init__.py` and `src/config/models.py` as compatibility surfaces for at least one release.

## Task J: Other Later Targets

### `scripts/lib/executor.py`

GitNexus notes:

```text
ServerManager: LOW
Config: LOW
Executor name is ambiguous with src/runtime/executor.py, so use UID:
Class:scripts/lib/executor.py:Executor
```

Possible split:

```text
scripts/lib/executor_paths.py        binary path and registry helpers
scripts/lib/server_manager.py        ServerManager
scripts/lib/executor_config.py       Config and InferenceResult
scripts/lib/inference_executor.py    Executor
```

Run:

```bash
pytest tests/unit/test_executor.py tests/unit/test_executor_extended.py
```

### `src/pipeline_monitor/claude_debugger.py`

GitNexus notes:

```text
ClaudeDebugger: LOW
direct importers are seed routing scripts and deprecated v1.
```

Possible split:

```text
src/pipeline_monitor/debugger_paths.py
src/pipeline_monitor/debugger_parsing.py
src/pipeline_monitor/debugger_replay.py
src/pipeline_monitor/claude_debugger.py
```

Run:

```bash
pytest tests/unit/test_claude_debugger.py \
       tests/unit/test_seed_specialist_routing_main_and_retry.py
```

### `src/session/sqlite_store.py`

GitNexus notes:

```text
SQLiteSessionStore: LOW by class impact, but it is persistent I/O.
```

Possible split:

```text
src/session/sqlite_schema.py
src/session/sqlite_queries.py
src/session/sqlite_embeddings.py
src/session/sqlite_store.py
```

Run:

```bash
pytest tests/unit/test_sqlite_store_extended.py \
       tests/unit/test_session_protocol.py \
       tests/integration/test_frontend_integration.py
```

### `orchestration/procedure_registry.py`

GitNexus notes:

```text
ProcedureRegistry: LOW
direct importers: procedure_scheduler.py and repl_environment/procedure_tools.py
```

Possible split:

```text
orchestration/procedure_models.py
orchestration/procedure_loader.py
orchestration/procedure_executor.py
orchestration/procedure_registry.py
```

Keep `ProcedureRegistry` as the facade.

Run:

```bash
pytest tests/unit/test_procedure_registry.py \
       tests/unit/test_repl_procedure_tools.py
```

## Required Reporting Format For Junior Agents

Each delegated PR or patch should include:

```text
Scope:
- Files changed:
- Symbols moved:
- Compatibility surface preserved:

GitNexus:
- context command(s):
- impact command(s):
- risk level(s):
- direct callers:
- affected processes:
- detect-changes summary:

Tests:
- Commands run:
- Result:
- Known gaps:

Notes:
- Any behavior intentionally changed:
- Any monkeypatch/import path preserved:
```

## Stop Conditions

Stop and ask for review if any of these occur:

- GitNexus reports HIGH or CRITICAL risk for a symbol that was expected to be LOW.
- A public import path would need to be removed.
- A test relies on monkeypatching a private symbol that moved.
- A persistent schema, JSON output shape, prompt dict shape, or CLI option would change.
- `gitnexus detect-changes` reports unexpected execution flows.

