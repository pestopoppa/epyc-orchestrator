# MHS-7 source review packet

Task source: ROOT `origin/main` at `1fda0ad840c1a91ea39a9a5f52bf1a59edf0beec`, handoff `handoffs/active/promptforge-mutation-safety-contract.md`, MHS-7 row.
APP base: `eb4561b0247f0a94b34da2851bc7f5a2be184efd`.

## Implemented in the APP lane

- `PromptForge.apply_code_mutation()` now fails closed with `isolated_context_required`; direct live-tree writes are removed.
- `_action_code_mutation()` opens the existing `apply_mutation_isolated()` experiment context, uses `apply_code_mutation_in_context()`, preserves the existing served-content hash capture and all evaluation gates, accepts only after all gates pass, and otherwise relies on context cleanup to restore the recorded preimage. Ordinary exceptions reset the mutation object's accepted flag and propagate after cleanup.
- The in-context apply validates path allowlisting, new-file roots, and current file content against the proposal preimage before overlaying. Stale candidates and collisions return a rejection without applying.
- Experiment acceptance commits through a temporary private Git index, explicit owned-path pathspecs, and normal `git commit` hooks. Commit rejection leaves the decision undecided so context cleanup restores the main-tree preimage. Only accepted paths are refreshed in the shared index after success; unrelated staged entries are not swept or changed.

The existing context intentionally writes the candidate to both its worktree and the main tree for live evaluation. This implements transactional rollback under that established overlay contract; it does not claim candidate invisibility or SIGKILL durability.

## Verification

- Focused suite: 433 passed, 1 pre-existing deprecation warning. Resource receipt and full log: `pytest-focused-review-receipt.json` and `pytest-focused-review.log` in this directory.
- R6 admission: lock `/mnt/raid0/llm/tmp/compute-flags/prep-locks/32-33.lock`; taskset CPUs 32-33 plus SMT 128-129; captured PID 1040500, start ticks 157275356; PID absent after wait; peak process-tree RSS 1,551,859,712 bytes; elapsed 73.12 s; exit 0.
- `ruff check` on the 10 MHS-7-owned source/test files: pass. `git diff --check`: pass.
- GitNexus impact on `apply_code_mutation`: exact/LOW, one production caller. Its indexed tree was 106 commits stale and the APP wrapper refused refresh before Node startup (exit 64: cross-role global region locks not enabled); manual call-site search confirmed only `_action_code_mutation` as production caller.
- `actions.py` AST parity against the pre-trim snapshot confirms the removed formatting churn did not change behavior.

Belief note for parent review: `[no-belief: code-only change; no new measurement]`.

## Final private-index correction (main review)

The commit now runs `git commit -m <message>` against a private index seeded from `HEAD` and populated only by the explicit `git add -A -- <owned paths>`. It no longer uses `--only`, which would re-read the live worktree and could commit a newer edit than the evaluated bytes. After success, `git reset -q HEAD -- <owned paths>` refreshes only those index entries while leaving any post-commit worktree edit unstaged.

Added regression: a normal pre-commit hook changes the main worktree target after private-index staging; the commit contains the evaluated staged bytes, the hook's newer edit stays in the worktree unstaged, and an unrelated staged entry retains its exact index record. Hook rejection rollback coverage remains.

Correction controls passed under the R6 lock: 140 worktree/mutation safety tests, 6 code-mutation action tests, 13 ledger tests, 2 apply/leakage tests, and 1 served-content ordering control (162 total). Receipt/log: `pytest-private-index-correction-receipt.json` and `pytest-private-index-correction.log`. Peak tree RSS 700,133,376 bytes; elapsed 41.29 s; all captured PIDs absent after wait.

Post-correction `ruff check` on the 10 owned source/test files and `git diff --check`: pass. The focused commit hook and post-staging race fixture use temporary Git repositories only.

## Publication identity

The scoped APP source commit is `c91884ee5a38f7b4a2168c76fee1394a867e2e33`, parent `da2fec3e155611118928d0e7b903428bdb6fd0d1`. It contains exactly the 10 MHS-7 source/test files and is published on `origin/main`; `PUSH-VERIFICATION.md` records the serialized-wrapper and native-remote proof. The ROOT handoff checkbox remains MAIN-owned and is not changed by this capsule.
