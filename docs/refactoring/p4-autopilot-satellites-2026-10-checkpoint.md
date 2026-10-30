# P4 Autopilot Satellites — 2026-10 checkpoint

Candidate branch: `codex/refactor-autopilot-satellites-01a16fc3`  
Orchestrator baseline: `e1e3e59d72f9be800d7c3ba4cb1cf10fdd4a5c1c`  
Research package: P4 in the root `refactoring-multirepo-2026-10` handoff.

This checkpoint records three scoped P4 behavior fixes. It does not close P4 or authorize
deletions. The research-derived modules and one-shot ratifiers remain untouched pending their
reachability and retention reviews.

## Changes

- **B4-M1 — blacklist runtime path.** `state_store.py` owns the new ignored runtime path
  `orchestration/autopilot_failure_blacklist.yaml`. Reads, legitimate first appends, the
  preflight audit, and the default purge CLI migrate the tracked
  `scripts/autopilot/failure_blacklist.yaml` once when the runtime copy is absent. Migration
  uses a sibling lock and temporary copy followed by `os.replace`; a migration failure raises
  instead of allowing the first append to replace legacy bans. Explicit custom paths retain
  their prior behavior. The tracked legacy YAML remains in place as the migration source.
- **B4-M2 — generator health check.** `gen_system_card.py --check` now checks that generation
  succeeds and includes the required card sections. It reports generator errors and does not
  compare against or write the stale tracked card. Degraded planner guidance now describes
  this check as a generated-card health check.
- **B4-09 — report freshness.** The report CLI writes top-level `generated_at`. The planner
  selects and ages reports using that timestamp, then a timestamp in the filename, then file
  mtime as a compatibility fallback. The overlapping filename globs are collapsed.

## Verification evidence

- Focused tests passed under the reviewed private Python 3.11.17 / pytest 9.0.3 environment:
  - `test_autopilot_state_store.py`: 41 passed.
  - `test_autopilot_preflight_audit.py -k audit_blacklist`: 1 passed, 26 deselected.
  - `test_blacklist_purge_plan.py`: 4 passed.
  - `test_autopilot_system_card.py`: 15 passed.
  - `test_autopilot_actions.py -k model_gate_advisory`: 3 passed, 143 deselected.
- The first preflight attempt failed because its fixture resolved the real P4 runtime path. The
  test now supplies an explicit temporary blacklist path; the corrected focused run passed.
- The first failed attempt created ignored runtime-path artifacts in the private P4 worktree.
  The coordinator recorded their byte hashes before and after the corrected attempt and
  confirmed they were preserved. They were not removed or treated as clean-state evidence.
- `git diff --check` passed. `git diff --exit-code HEAD --` passed for the four P5 immutable
  files, the NI63 fixture/test paths, and the W3e source/test/fixture paths. The candidate
  changes none of those files.

The five focused test invocations are scoped evidence, not a full-suite or full-P4 result.
Targeted Ruff check and Python compilation passed. Targeted `ruff format --check` reports 9 of 10
files would be reformatted, and the exact e1e3 baseline reports the same 9 files. Hunk comparison
found two formatting changes introduced by this patch; both were aligned manually. The normalized
candidate-versus-prior formatter diff then contained only inherited formatting hunks and line-offset
changes. No bulk formatting was applied. The focused tests preceded those two formatting-only edits;
behavior was unchanged.

## Remaining limits

- No P4 deletion was made. B4-02/B4-03 require separate provenance and trust-boundary review;
  B4-13's research-derived modules require an operator wire-or-retain decision.
- This candidate has not been committed, pushed, merged, or deployed. Main review and the
  publication checks remain pending.
- Belief wiring and root-owned handoff/index progress remain with the root owner.
