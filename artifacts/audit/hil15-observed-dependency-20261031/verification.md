# HIL-15 observed dependency logging: verification capsule

Date: 2026-10-31 UTC. APP checkout: `epyc-orchestrator`; branch `codex/td29-m4-cache-20261030`; base `2f130ee67` (`origin/main`). This capsule records a model-free source fix and unit-test evidence. It is not an AUC measurement or a parent-task completion claim.

## Source gap and scope

The pre-existing HIL-15 implementation (`f6ebf67f5`) attached `DependencyLog` to `_invoke_tool` and `ExplorationLog.add_event`, but the native combined-operation path calls `ToolRegistry.invoke` directly through `_registry_invoke_observed`. `step_events.py` already documents that bypass. Consequently inner search results could be represented only by a synthetic aggregate event (or be absent); they did not have their own native tool-call provenance.

The fix logs actual registry arguments/results at `_registry_invoke_observed` when `observed_dependency_logging` is enabled, independently of `step_event_logging`, and excludes `batch_web_search` and `search_and_verify` aggregate wrappers from dependency classification. The no-observer fast path is preserved. The existing `observed` grade and resource extraction rules are unchanged. This makes no claim that an observed edge is intervention-validated and does not infer missing resource identifiers.

## Verification

`cpu-tests.log` contains the full command and pytest output for the five requested shared-hook suites. They passed **95/95** in 16.58 s. The run used the canonical APP venv interpreter, this checkout on `PYTHONPATH`, OMP/OPENBLAS=1, prep mutex `prep-locks/32-33.lock`, and allowed SMT siblings 128,129; sampled process-group peak RSS was 273368 KiB, with 900 s deadline and 8-GiB RSS cap. No region-lock, inference, model server, or host-window action was used.

The tests include an eight-case cross-product over both combined builtins, dependency logging on/off, and step-event logging on/off. Each enabled stream contains exactly the native inner calls, never a duplicate aggregate. A separate regression proves a failing telemetry observer cannot replace a successful registry result or mask/rewrite the original registry exception.

`ruff check --ignore E702` passed for the changed Python files; `git diff --check` passed. Plain ruff check reports only three existing E702 semicolon findings in the original test file, and `ruff format --check` would reformat all three existing files; no broad formatting was applied.

Changed-source and proof-file SHA-256 values:

- `src/repl_environment/context.py` — SHA-256 `0c9d95404ddb5b098d56e1b833bc33d4b6e165f5db3ed9b3f28947b51b01afa8`
- `src/trace/dependency_log.py` — SHA-256 `0544660af848c68ff13b7ff1dc9e0ddf4807fadf31757d52334ec87f7a23cab1`
- `tests/unit/test_dependency_log.py` — SHA-256 `48e606628c2fd021dc4b7f319c75ebc7e0785274daca0e767873924292e717a0`
- `artifacts/audit/hil15-observed-dependency-20261031/cpu-tests.log` — SHA-256 `96009fd4cf6d99c97d5ae7d0457c8f24d0cfd129c516281238be63f911ea4670`

## Acceptance boundary

The HIL-15 handoff acceptance also requires a real flag-on trace AUC-over-run-size table with a task-resampled interval. That remains HIL-15a and requires its inference window. This code change does not satisfy or close that parent acceptance.
