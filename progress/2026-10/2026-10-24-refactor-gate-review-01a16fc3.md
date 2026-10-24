# P2 autopilot gate review — 2026-10-24

Review checkpoint accepted by the owning main. Source branch: `refactor/autopilot-gate` at `bd97d0d8d`; review branch: `refactor/autopilot-gate-reviewed-01a16fc3`. The accepted test-isolation merge is `f80d82a8d` (reviewed isolation checkpoint `5e392a3c9`). Main accepted the review/migration commit `4069b8ee`; this closeout adds the progress and agent-log shards and removes one confirmed unused import.

## Decision semantics and metric directions

- **Quality:** higher is better. `regression_leg` computes `(quality - same-tier baseline) / baseline`; a sufficiently negative delta is a regression. MAD runs only after the result is at or above baseline.
- **Speed:** tokens/second is higher-is-better throughput and is guarded by a lower bound (`speed < 0.8 × frontdoor_speed`). Cost is represented as `-cost` in the quality/speed/cost/reliability objective tuple, so lower cost is better. Reliability is higher-is-better.
- **B2-03:** signed `z = (new_quality - history_median) / (1.4826 × MAD)`. A fresh gain requires `z > 2`; a result with `z < -2` is classified as `below_recent_level` and excluded from learning. Zero-MAD windows use the existing strict `±0.2` delta boundary. A reproduction is tagged only inside the noise band when the history median is significantly above the same-tier baseline.
- **B2-02:** sequential evaluation upgrades a cached legacy PASS by replacing the MAD leg with the sequential leg, preserving the prior PASS and avoiding a second host-state sample.

The B2-03 regression test fails on isolated base `84f6f6f1` because a result above baseline but far below its recent median receives neither MAD exclusion label. The B2-02 test fails on that base because a recovered host is resampled and turns a throttle-demoted cached PASS into a throughput FAIL. Both pass on the review branch.

## Verification

- Targeted 49-file command, run under `scripts/region-lock` with CPU list `0-95`, role `build`, mock mode: **1,018 passed, 1 skipped**, no failures. This run preceded the final snapshot-migration test addition.
- Updated `tests/unit/test_snapshot_authority_equivalence.py`, after adding migration coverage: **18 passed**. Its new test pins the pre-P2 archive digest, confirms the hash-valid old snapshot is rejected for bounded authority, checks full-replay fallback equality, and confirms a current-format recut verifies as `archive_prefix_match/current` with canonical equality.
- `tests/unit/test_kernel_path_env_overrides.py`: **3 passed**. The `src/registry/kernel_paths.py` fixture pin matches Git blob SHA-1 `fb006e99f7ab8f3706f1371934e51800256988eb` and SHA-256 `3993e914d5436f753a114bd6d92aaf727c75230baa1efe32d28351183b42a379`.
- Ruff passed on all changed Python files after removing the unused `sys` import from `source_clone.py`; inspection confirmed the import was its sole occurrence and removal has no runtime or pin effect. Focused `ruff check`, `git diff --check`, and the source-pin test passed in closeout.
- No full suite, services, inference, deployment, or live journal writes were run.

## Snapshot migration and boundary

On the real 1,372-row `gate_frontier_replay.json` fixture, the pre-P2 archive matches the pinned historic digest after removing the new additive `exclusions.unmeasured_objectives` counter. Current replay derives `count=85`, `max_trial_id=1496`. The old snapshot has a matching payload hash but exact prefix comparison returns `archive_prefix_drift/prefix_invalidated`; bounded authority returns `None`, so callers use full replay. A current-format rebuilt snapshot restores `archive_prefix_match/current` and equals full replay canonically.

The migration requirement is append-only recutting after integration. The live autopilot owner performs it at a trial boundary with autopilot stopped, then verifies before resuming. The documented commands are in `docs/autopilot/pre-p2-journal-snapshot-rebuild.md`; `--force` permits recutting when an existing snapshot already covers the journal, and the live-process guard remains enabled. The command was not run against any live journal.

Native labels: `journal-segment-snapshot-v1` → `journal-segment-snapshot-v2` and `journal-archive-snapshot-v1` → `journal-archive-snapshot-v2`. Representative replay state remains `representative-replay-state-v1`.

## Protected files and integration risk

These evaluator-sensitive files are unchanged by the P2 source diff: `scripts/autopilot/eval_tower.py`, `scripts/autopilot/rubric_scoring.py`, `src/autopilot_core/tier_specs.py`, and `orchestration/eval_registry.yaml`. The NI63 fixture's `SOURCE_PINS` entry for `src/registry/kernel_paths.py` was refreshed and verified against both hashes above.

GitNexus impact was checked before edits: `mad_significance` is LOW (4 impacted); `archive_payload_from_verified_snapshot` is HIGH (6 impacted); `reconstruct_archive_from_journal_rows` is CRITICAL (41 impacted, including preflight-audit and dashboard-Pareto processes). Those existing replay/reconstruction changes remain branch-only and held for owner/operator integration review. No production-readiness claim is made. The new fixture cleanup target was not present in the indexed graph (`UNKNOWN`, 0 impacts); the edit only removes an unused import.

## Closeout scope

Committed only to the review branch: this progress shard, its agent-log shard, and the one-line unused import cleanup. No root handoff, index, source-adapter, or progress files were edited. No main merge or production operation was performed. The review worktree remains for main-owned cleanup.
