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
- `src/api/services/memrl.py`
- `src/api/services/routing_models.py`
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
risk. The safe portion is now done:

- `RoutingModelBundle` added
- graph-router/classifier/verifier loading isolated in `routing_models.py`
- keep `ensure_memrl_initialized()` as a compatibility facade
- `_route_request` left untouched

Do not refactor `_route_request` further without a dedicated routing-design pass.

## Exact Next Edit

No mandatory refactor edit remains in the current plan.

Recommended next step is review/commit hygiene:

- separate runtime artifacts from source changes
- decide whether to include the earlier autopilot changes in the same commit
- run the focused test suite again before committing

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
