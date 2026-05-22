# Orchestrator Refactor Handoff

This note is a recovery point for the orchestration cleanup work. It is intended
to let another agent continue even if the current session is interrupted.

## Current Repository State

The worktree is intentionally dirty.

Relevant refactor changes already in flight:

- `scripts/server/orchestrator_stack.py`
- `scripts/server/stack_*.py`
- `src/api/routes/dashboard.py`
- `src/api/routes/dashboard_*.py`
- `src/api/routes/dashboard.html`
- `src/api/routes/chat_pipeline/routing.py`
- `src/api/routes/chat_pipeline/routing_decision.py`
- `src/api/services/memrl.py`
- `src/api/services/routing_models.py`
- `scripts/autopilot/config_applicator.py`
- `tests/unit/test_stack_*.py`
- `tests/unit/test_dashboard_helpers.py`
- `tests/unit/test_routing_models.py`
- `tests/unit/test_memrl_service.py`
- earlier autopilot/config changes already present in the same worktree

Unrelated runtime artifacts should be left alone:

- `orchestration/repl_memory/sessions/embeddings.faiss`
- `orchestration/repl_memory/sessions/id_map.npy`
- `scripts/autopilot/failure_blacklist.yaml`
- `scripts/autopilot/short_term_memory.md`

Do not revert those files unless the user explicitly asks.

## What Has Already Been Completed

1. `orchestrator_stack.py` process/PID helpers were extracted behind stable
   wrappers.
2. `orchestrator_stack.py` state persistence was extracted behind stable
   wrappers.
3. Regression tests were added for:
   - listener-only port PID discovery
   - process-tree kill behavior
   - state load/save filtering
   - reload behavior that depends on listener-only cleanup
4. The focused test set passed:
   - `tests/unit/test_stack_processes.py`
   - `tests/unit/test_stack_state.py`
   - `tests/unit/test_orchestrator_stack_reload.py`
   - the earlier autopilot/applicator tests
5. Later stack launcher cleanup reduced `build_server_command()` into named
   helper functions while keeping command construction inside
   `orchestrator_stack.py`.
6. Dashboard backend/static cleanup split the large dashboard route module into
   topology, tap parsing, task correlation, snapshot scanning, and HTML modules
   while preserving route signatures.
7. Routing/MemRL cleanup introduced `src/api/services/routing_models.py` with
   `RoutingModelBundle`, graph-router loading, classifier loading, and
   frontdoor-verifier loading. `ensure_memrl_initialized()` now delegates
   optional routing-model startup to that bundle and remains the compatibility
   facade.
8. Chat routing cleanup introduced
   `src/api/routes/chat_pipeline/routing_decision.py` for initial routing
   selection, factual-risk/difficulty scoring, failure-graph veto, routing
   telemetry metadata, timeout resolution, skill ID extraction, task-type
   derivation, and Trinity shadow classification. `_route_request()` remains the
   public chat-pipeline orchestration function and preserves the legacy
   `routing._classify_and_route` patch point used by tests.
9. Autopilot config application now has a typed `ApplyResult` plus explicit
   `HotSwapApplicator`, `EnvRestartApplicator`, and `KvCompactionApplicator`
   classes. Existing public functions still return dictionaries for
   compatibility with `autopilot.py` and tests.

The live orchestrator API was healthy when last checked. Autopilot was
intentionally killed per user instruction and should remain off unless the user
requests it be restarted.

## Refactor Strategy

Use low-risk extraction first and keep `orchestrator_stack.py` as a
compatibility facade until callers stop depending on its internal helpers.

### Tranche 1: Stack internals

Already done:

- `stack_processes.py`
- `stack_state.py`
- `stack_env.py`
- `stack_health.py`
- `stack_host.py`
- `stack_numa.py`
- `stack_runtime.py`
- `stack_docker.py`
- `stack_checkpoint.py`

### Tranche 2: Launcher cleanup

Completed conservatively:

- keep `build_server_command()` in `orchestrator_stack.py`
- reduce it to a dispatcher over named helpers
- keep command construction close to its large set of launch constants

### Tranche 3: Dashboard cleanup

Completed:

- backend snapshot/state helpers moved into service modules
- task correlation and tap parsing moved into separate helpers
- static HTML/CSS/JS moved out of the route module
- route signatures preserved

### Tranche 4: Routing/MemRL cleanup

Partially completed.

GitNexus marked both `ensure_memrl_initialized()` and `_route_request` as HIGH
risk. The safe portions are now done:

- `RoutingModelBundle` added
- graph-router/classifier/verifier loading isolated in `routing_models.py`
- keep `ensure_memrl_initialized()` as a compatibility facade
- `_route_request` reduced to orchestration while decision helpers live in
  `routing_decision.py`

Do not make behavioral changes to `_route_request` without a dedicated
routing-design pass.

## Remaining Refactor Candidates

**ALL FOUR FURTHER TRANCHES COMPLETE as of 2026-05-22 session 2.** No
mandatory refactor edit remains. The refactor is at a natural stopping point
across stack, autopilot, retriever, and dashboard.

### Tranche 5 — autopilot.py split ✅ DONE 2026-05-22

`autopilot.py`: 1832 → 1151 lines (−37%). Extracted:
- `scripts/autopilot/actions.py` (608): all 14 action handlers as
  `_action_<type>` functions + `dispatch_action()` facade with AP-9 scope
  validation. Each handler takes an `_ActionContext` bundle.
- `scripts/autopilot/controller_io.py` (150): `invoke_controller`,
  `extract_action`, `_unwrap_action`, `validate_single_variable`.
- `scripts/autopilot/state_store.py` (135): `load_state`, `save_state`,
  `load_blacklist`, `check_blacklist`, `append_blacklist`,
  `load_model_signatures`, `format_model_signatures` — all parameterized
  on Path objects.
- `autopilot.py` keeps thin wrappers supplying STATE_PATH/BLACKLIST_PATH
  and the Claude-CLI cwd, plus `_apply_params` lazy lookup so existing
  monkeypatch-based tests keep working.

### Tranche 7 — orchestrator_stack.py split ✅ DONE 2026-05-22

`orchestrator_stack.py`: 2772 → 1325 lines (−52% additional). Extracted:
- `scripts/server/stack_paths.py` (67): `_PATHS`, `_get_paths`,
  `STATE_FILE`, `LLAMA_SERVER`, `_V2_ROLES`, `SLOT_SAVE_DIR`, `_HEALTH_*`
  timeouts. Lives below stack_manifest + stack_commands in the dep graph
  to avoid a cycle.
- `scripts/server/stack_manifest.py` (566): `PORT_MAP`, `ROLE_LAUNCH_META`,
  `HOT_ROLES`, `SERIAL_ROLES`, `NUMA_REPLICA_PORTS`, model paths,
  `ORCHESTRATOR_PROFILES`, `DOCKER_SERVICES`, classification helpers
  (`_filter_by_numa_mode`, `_build_servers_from_classification`),
  `validate_against_registry`, `validate_model_paths`, computed
  `HOT_SERVERS`/`WARM_SERVERS`.
- `scripts/server/stack_commands.py` (998): `cmd_start`, `cmd_stop`,
  `cmd_reload`, `cmd_status` + `_find_pids_on_port` + `_scan_known_ports`.
  Uses lazy proxy functions for symbols still in orchestrator_stack
  (start_server, init_memrl_and_tools, thin wrappers).
- `orchestrator_stack.py`: re-imports all extracted names so
  `from orchestrator_stack import ROLE_LAUNCH_META` (registry_compiler
  fallback per `src/registry/registry_compiler.py:266`) keeps working.
  `main()` imports stack_commands lazily inside the function body to
  avoid the module-load circular import. Module-level `__getattr__`
  exposes `cmd_*` for tests that do
  `from scripts.server import orchestrator_stack as stack; stack.cmd_reload(...)`.

### Tranche 6 — retriever.py split ✅ DONE 2026-05-22

`retriever.py`: 1657 → 881 lines (−47%). Extracted:
- `orchestration/repl_memory/retrieval_config.py` (106): `RetrievalConfig`,
  `RetrievalResult`, `ScoreComponents` dataclasses + `_retr_cfg` helper.
- `orchestration/repl_memory/hybrid_router.py` (713): `HybridRouter` class
  verbatim. `TYPE_CHECKING`-only imports for `TwoPhaseRetriever` /
  `RuleBasedRouter`; no behavioral changes.
- `orchestration/repl_memory/routing_risk.py` (103):
  `is_risk_gate_enforced_for_route`, `guardrail_blocks_gate`,
  `build_{not_enforced,abstain,accept}_response` — pure functions taking
  `config` + `risk_budget_stats` as parameters.
- `orchestration/repl_memory/routing_fast_path.py` (94):
  `compute_robust_confidence`, `apply_confidence_to_results`,
  `effective_confidence_threshold`, `action_prior_prob` — pure functions
  extracted from TwoPhaseRetriever methods.
- `retriever.py`: `_compute_robust_confidence`, `_apply_confidence`,
  `get_effective_confidence_threshold`, `_is_risk_gate_enforced_for_route`,
  `_guardrail_blocks_gate` delegate to the new modules. Re-exports
  `HybridRouter`, `RetrievalConfig`, `RetrievalResult`, `ScoreComponents`
  so existing imports keep working unchanged.

### Tranche 8 — dashboard polish ✅ DONE 2026-05-22

- `src/api/routes/dashboard_tasks.py`: replaced deprecated
  `datetime.utcnow()` with timezone-aware `datetime.now(timezone.utc)`
  (preserves the "Z" suffix in the snapshot header by stripping the tz
  offset).
- `tests/unit/test_dashboard_route_html.py`: route-level coverage — the
  `/dashboard` endpoint returns the extracted `dashboard.html` body
  byte-for-byte.
- Dashboard route handlers do not import any stack launcher internals
  (verified by inspection — only stack-agnostic helper modules are
  imported from `dashboard.py`).

### Cumulative state across Tranches 1-8

| File | Pre-refactor | Post-refactor | Δ |
|---|---|---|---|
| `scripts/server/orchestrator_stack.py` | 3433 | 1325 | −61% |
| `scripts/autopilot/autopilot.py` | 1832 | 1151 | −37% |
| `orchestration/repl_memory/retriever.py` | 1657 | 881 | −47% |
| `src/api/routes/dashboard.py` | 2246 | 863 | −62% |
| **Total reduction in the four targeted files** | **9168** | **4220** | **−54%** |

New sibling modules: 17 (10 stack: paths/manifest/commands/processes/state/env/host/health/docker/numa/checkpoint/runtime, 3 autopilot: actions/controller_io/state_store, 4 retriever: retrieval_config/hybrid_router/routing_risk/routing_fast_path, 5 dashboard: topology/tap/tasks/snapshot/html).

Test coverage: 305 focused tests across all refactor-touched modules
(was 7 at the start of Tranche 1). Full unit suite: 5829 passed, 33
failed (identical baseline failures pre-refactor; +121 new passing tests
vs baseline). `gitnexus detect-changes --scope all` after final re-index:
"No changes detected" — graph is clean.

## Safety Constraints

- Do not touch the unrelated runtime-artifact files listed above.
- Do not change `_route_request` without a routing-specific test plan.
- Do not restart or kill the API unless explicitly asked.
- Keep the live `/health` endpoint intact during refactors.

## Verification Checklist

Run these after each tranche:

1. `python3 -m py_compile` for touched modules.
2. Focused `pytest` on the new modules and any wrapper regression tests.
3. `git diff --check` on the touched files.
4. `gitnexus detect-changes --scope all --repo epyc-orchestrator`
5. If editing existing symbols, run `gitnexus impact` first and record the
   blast radius in the session notes.

## Known Notes

- `gitnexus detect-changes --scope all` currently reports a critical risk only
  because the worktree already contains the separate autopilot changes from the
  earlier fix. That is not caused by the stack/dashboard/routing-model
  extraction itself.
- The stack module still uses the top-level `orchestrator_stack.py` import path
  in at least one registry compiler path. Preserve import compatibility while
  refactoring.
- Latest broad refactor verification: 370 tests passed across stack, dashboard,
  autopilot/applicator, routing/MemRL, chat routing, stream adapter, and chat
  endpoint suites.
