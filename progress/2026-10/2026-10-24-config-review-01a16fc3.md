# Config-registry refactor review — 2026-10-24

Reviewed `refactor/config-registry` from `c9ef39de4` in the private worktree
`/mnt/raid0/llm/worktrees/refactor-01a16fc3-config-review`. Parent accepted the result at
`07829696ea504540e569ee78f24354abfdd46f6a`; it is proposed work pending integration.

## Review changes

- `d843eb9c`: serialized the failed first load in `chat_template_kwargs_for_role` with a
  module lock and a double check of loader/backoff state. Before the fix, the 12-thread
  regression recorded 12 loader attempts and 12 warnings; after the fix, all 12 callers
  receive `None` from one attempt. The loader's current ni63 pin was refreshed.
- `07829696`: changed the compat-alias stress test to race 24 threads on one alias per
  fresh subprocess, for all 16 aliases. The per-alias identity, `__name__`, and `__spec__`
  checks remain. This tests the shim first-import race without attributing a separate
  legacy cross-module import cycle to A3-16/A4-08.
- `5886e5ae`: merged the accepted hermetic, hazard, and runtime-store isolation branch at
  `87540dae85d773eb86ffcc286744d2c655b96cd7`, preserving the reviewed registry-loader
  pin during conflict resolution.

## Validation

All commands ran through `scripts/region-lock` for CPU cores 0–95, role `build`, tag
`refactor-config-review-01a16fc3`; each pytest invocation used `CI=true`,
`ORCHESTRATOR_MOCK_MODE=1`, `-p no:cacheprovider`, `-o addopts=""`, and
`TMPDIR=/tmp/refactor-01a16fc3/config-review`.

- `tests/unit/test_registry_chat_template_kwargs.py`: **6 passed** after the fix. The
  concurrency regression failed before the fix (12 concurrent load attempts).
- Compat alias identity and revised same-alias import stress selection:
  **17 passed**.
- Targeted O3 selection from the implementer report (config/registry, OCR, topology,
  shadow worker, alias, and affected route tests): **548 passed, 4 skipped** in 10.58 s.
- Alias-target selection across 25 caller test files: **524 passed** in 9.29 s.
- All seven changed ni63-pinned source files were checked against both `git hash-object`
  and SHA-256; every stored pin matched. `git diff --check` passed. No checkout leaks
  were present after the test runs.
- Not run: the 700-trial pre-fix shim reproducer and the full suite. The implementer also
  reported the `scripts/server/orchestrator_stack.py` ni63 pin as stale at base; it was
  outside this review's edits and remains an O8 pin-design issue.

## Separate baseline issue: mixed-alias cold imports

The original randomized stress in `tests/unit/test_inference_lock_concurrent_import.py`
imports 16 different aliases concurrently. It fails on the reviewed branch and on clean
baseline `origin/main` at full commit
`84f6f6f15f713d21098dbad2b153d32f8cc13624`; it is not an O3 regression. Reproducer:
that test as stored in `5886e5ae:tests/unit/test_inference_lock_concurrent_import.py`, run
against baseline source at `84f6f6f1`. Captured baseline output:
`/tmp/refactor-01a16fc3/config-review/cross-alias-main-84f6f6f1.log`.

Baseline run result: **1 failed, 18 deselected** in 2.14 s. The mixed import order exposed
partial exports in the existing cross-module dependency cycle, including
`src.dispatcher` missing `RegistryLoader` from `src.registry_loader` and `src.executor`
missing `InferenceRequest` from `src.model_server`. The parent will track this separately;
no broad import-graph changes were made in O3.
