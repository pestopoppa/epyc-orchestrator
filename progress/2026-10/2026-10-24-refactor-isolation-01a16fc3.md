# Refactor test-isolation review — 2026-10-24

Implementation accepted by the owning main: `87540dae85d773eb86ffcc286744d2c655b96cd7` on `refactor/test-isolation-reviewed-01a16fc3`, based on `5e1922c14` (`origin/refactor/test-hazards`).

- Accepted the patch-equivalent hermetic harness (`ef7b0d3b`/`be5b4da4`): explicit kernel-store override preserves production defaults. The existing NI63 source pin matches the changed kernel-path source; no additional production files or pins changed.
- Accepted A6-01 (`5884910d`): destructive fence payloads target temporary wiki trees and sinks, retaining the kernel-layer check.
- Accepted A6-02 (`2ca891c0`): temporary UTM output, concrete compaction paths, delegation-report isolation, and ignored-file checkout leak detection. The owning main handles the root NI07 capture environment integration.
- Accepted A6-09 (`5e1922c14`): frozen E8 pre-state and unique REPL/spill/mutation fixtures.
- Accepted A6-M1 (`87540dae`): evaluator-ledger environment pin precedes application imports; trace API positional/keyword defaults and aliases use a fresh database per test, including modules imported during an earlier test. Explicit database destinations are preserved. Host materialized-trace smoke is opt-in, and both runtime-store leak exemptions are removed.

Validation ran under the required CPU-region claim with mock mode, using only targeted tests. The final runtime/trace API suite passed **189 tests with one intentional host-trace skip**. All **seven E8 wrapper cases passed** under the authorized `/tmp/refactor-01a16fc3/test-isolation` scratch root. The three new regressions passed alone and all failed safely on the untouched parent before writes. Ruff, diff checks, and pre-commit hooks passed. No checkout trace database or evaluator ledger remained.

The broader branch check passed 477 tests with two skips; its seven initial E8 failures were the intentional resolved-`/tmp` path guard under the original RAID-only scratch assignment, resolved by the seven-case rerun above. Two mutation-diagnostics tests failed identically on candidate and untouched parent because their `SimpleNamespace` prompt fake lacks `_mutation_exclusion_reason`; this review did not change PromptForge or waive the failures.

Production trace source and evaluator-fingerprinted files remain unchanged. GitNexus reported CRITICAL upstream impact for a proposed trace schema-source change, so that edit stopped and the owning main approved the fixture-only alternative. Existing fixture/hooks/smoke impacts were LOW.

Detailed review evidence: `/mnt/raid0/llm/tmp/refactor-01a16fc3/test-isolation/PROPOSED.md`, with `final-green.log`, `focused.log`, `parent-final.log`, `e8-correct-shape.log`, and `ruff-final.log` alongside it. The temporary `/tmp` subtree was removed after the implementation commit. The main retains this worktree for integration and cleanup. No services, deployments, shared-clone files, or production kernels changed.
