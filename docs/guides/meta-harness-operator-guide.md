# Meta-Harness Operator Guide (AR-3)

Operational reference for running and monitoring the Meta-Harness optimization tiers inside the AR-3 autopilot loop. For research context and implementation history, see the [meta-harness-optimization handoff](/mnt/raid0/llm/epyc-root/handoffs/active/meta-harness-optimization.md).

## 1. Quick Reference Card

**Tier 1 (Execution Trace Feedback)**: Live inference traces from `inference_tap.log` are fed back into PromptForge's mutation proposals, providing +15 points over score-only feedback (arXiv:2603.28052, Table 3 ablation).

**Tier 2 (Code Mutation Search Space)**: PromptForge can mutate 4 allowlisted Python files in the orchestrator source, with 4-layer validation preventing destructive changes.

### Key File Paths

| File | Path |
|------|------|
| Inference tap log | `/mnt/raid0/llm/tmp/inference_tap.log` |
| Autopilot state | `/mnt/raid0/llm/epyc-orchestrator/orchestration/autopilot_state.json` |
| Autopilot entry point | `scripts/autopilot/autopilot.py` |
| Eval tower (trace capture) | `scripts/autopilot/eval_tower.py` |
| PromptForge (mutations) | `scripts/autopilot/species/prompt_forge.py` |
| Worktree manager | `scripts/autopilot/worktree_manager.py` |

### Allowlisted Mutation Targets (Tier 2)

| # | Relative Path | Purpose |
|---|---------------|---------|
| 1 | `src/prompt_builders/resolver.py` | Prompt resolution logic |
| 2 | `src/escalation.py` | Escalation policy and retry logic |
| 3 | `src/graph/escalation_helpers.py` | Role cycle detection |
| 4 | `src/tool_policy.py` | Tool access control rules |

### Status Commands

```
python autopilot.py status      # Current trial, score, paused state
python autopilot.py report      # Markdown summary of journal entries
python autopilot.py pause       # Halt loop after current trial completes
python autopilot.py resume      # Unpause
python autopilot.py restore     # Restore from checkpoint
python autopilot.py checkpoint  # Save current state snapshot
```

All commands run from `/mnt/raid0/llm/epyc-orchestrator/scripts/autopilot/`.

---

## 2. Tier 1: Execution Trace Feedback

### How traces are produced

The inference tap log at `/mnt/raid0/llm/tmp/inference_tap.log` is written by live inference sessions passing through the orchestrator. It contains ROLE/PROMPT/RESPONSE sections showing how the orchestrator routed and handled each request.

### How traces flow into mutations

1. **Capture**: After each evaluation, `EvalTower.capture_recent_traces(n_lines=50)` (eval_tower.py, line 367) reads the tail of the tap log. It seeks to the last ~8KB of the file and returns the final 50 lines. Returns empty string if the file is missing or unreadable.

2. **Store**: The main loop stores the result in `state["last_traces"]` (autopilot.py, line 1029) after each eval completes.

3. **Inject**: When dispatching a `prompt_mutation` or `code_mutation` action, the dispatcher reads `state["last_traces"]` and prepends it to `failure_context` as a `## Recent Execution Traces` section (autopilot.py, lines 450-456 for prompt mutations, lines 531-533 for code mutations). This context is passed to PromptForge's `propose_mutation()` or `propose_code_mutation()`.

4. **Compose**: The failure_context also includes cross-species insights from the journal, past strategy insights from the strategy store, and per-suite quality scores. The trace section is prepended first, so it appears at the top of the context window.

### Impact

Per the arXiv:2603.28052 ablation (Table 3), full execution traces provide +15 accuracy points over score-only feedback. This is the single largest contributor to Meta-Harness's improvement.

---

## 3. Tier 2: Code Mutation Search Space

### Allowlist enforcement

`CODE_MUTATION_ALLOWLIST` (prompt_forge.py, line 29) is the hard boundary. `propose_code_mutation()` (line 448) raises `ValueError` if the target file is not on the list. Eval, scoring, and safety-gate code are excluded.

### The 4-layer validation

`_validate_code_mutation()` (prompt_forge.py, line 575) runs all four checks before a mutation is accepted:

1. **Syntax** (line 587): `ast.parse(mutated)` -- rejects anything that is not valid Python.
2. **Catastrophic shrinkage** (line 594): If the original has >10 lines and the mutation removes >60% of lines, it is rejected. The threshold is `new_lines < orig_lines * 0.4`.
3. **Public name preservation** (line 603): Extracts all module-level `FunctionDef`, `AsyncFunctionDef`, and `ClassDef` names from both original and mutated ASTs. Any name present in the original but missing from the mutation triggers rejection.
4. **Import test** (line 617): Temporarily writes the mutated code to disk, attempts `importlib.import_module()`, then unconditionally restores the original. Catches circular imports and runtime import errors.

If any layer fails, `mutation.syntax_valid` is set to `False` and the mutation content is replaced with the original (line 497). The dispatcher in autopilot.py checks `syntax_valid` (line 553) and skips the trial entirely if validation failed.

### Mutation lifecycle

```
propose_code_mutation()
  |-- read original from disk
  |-- invoke Claude CLI to generate mutated version
  |-- _validate_code_mutation() (4 layers)
  |-- if invalid: log "Code mutation rejected", return original
  v
apply_code_mutation()
  |-- git add + commit current state as pre-mutation checkpoint
  |-- write mutated content to disk
  |-- git add + commit the mutation
  v
tower.hybrid_eval()
  |-- T0 (10 sentinel questions, fast gate)
  |-- if T0 passes: T1 (50 questions, real signal)
  v
safety gate + simplicity criterion
  |-- if quality drops: revert_code_mutation()
  |--   writes original back, commits the revert
  |-- if >20% size increase for <2% quality gain: revert
  |-- if >50% size decrease: revert (catastrophic shrinkage guard)
  v
accept: swarm.mark_epoch() invalidates stale Optuna trials
```

The controller dispatches `code_mutation` actions in autopilot.py starting at line 516. Context assembly (failure traces, cross-species insights, strategy store) mirrors the prompt_mutation path.

---

## 4. Safety Mechanisms and Trial ~25 Incident

### The incident

During AR-3 run 2, trial ~25 proposed a code mutation on `src/escalation.py` that replaced the 454-line file with a 3-line stub. The orchestrator API went down for 11+ hours because escalation logic was eliminated.

### Five gaps fixed

| # | Gap | Fix | Location |
|---|-----|-----|----------|
| 1 | Syntax-only validation | Deep 4-layer validation (syntax + shrinkage + names + import) | `_validate_code_mutation()`, prompt_forge.py line 575 |
| 2 | No size guard | Catastrophic shrinkage guard: reject if >60% of lines removed | prompt_forge.py line 594; also >50% at dispatch level (autopilot.py line 584) |
| 3 | No revert commits | `revert_code_mutation()` now auto-commits the revert so HEAD is never corrupted | prompt_forge.py line 552 |
| 4 | No isolation | Worktree isolation available via `WorktreeManager` | worktree_manager.py (implemented, not yet wired into dispatch) |
| 5 | No per-trial scope | `apply_code_mutation()` creates a pre-mutation checkpoint commit before writing | prompt_forge.py line 509 |

### Worktree isolation (available, not yet active)

`WorktreeManager` (worktree_manager.py) creates a temporary git worktree per trial. The mutated file is committed in the worktree, then copied to the main repo for live eval. On rejection, the original is restored from the worktree's clean snapshot. The `PromptForge` class exposes `apply_code_mutation_in_context()` (line 418) and `apply_mutation_isolated()` (line 372) for worktree-backed mutations. These are not yet wired into the autopilot dispatch loop but are ready for integration.

Usage pattern:
```
wt = WorktreeManager(project_root)
with wt.experiment("trial_42") as ctx:
    ctx.apply_file(rel_path, mutated_content)  # writes to worktree + main
    result = tower.hybrid_eval()
    if result.quality > baseline:
        ctx.accept("autopilot: improved escalation")
    else:
        ctx.reject()  # auto-restores original in main
```

Auto-rejects if neither `accept()` nor `reject()` is called (safe default, line 174).

---

## 5. Integration: The Feedback Loop

```
  inference_tap.log              autopilot_state.json
        |                               |
        v                               v
  capture_recent_traces(50)      load_state()
  [eval_tower.py:367]            [autopilot.py:152]
        |                               |
        +--------> state["last_traces"] <+
                         |
                         v
              failure_context assembly
              [autopilot.py:450-456]
                         |
          +--- "## Recent Execution Traces"
          |--- "## Cross-Species Insights"   (journal)
          |--- "## Past Strategy Insights"    (strategy_store)
          |--- per-trial failure analysis     (journal.recent_failures)
                         |
                         v
           PromptForge.propose_mutation()     (Tier 1: .md prompts)
           PromptForge.propose_code_mutation() (Tier 2: .py code)
                         |
                         v
                   Claude CLI invocation
                   [prompt_forge.py:179]
                         |
                         v
                  _validate_code_mutation()   (Tier 2 only)
                  [prompt_forge.py:575]
                         |
                         v
               apply_mutation / apply_code_mutation
                         |
                         v
                  tower.hybrid_eval()
                  [eval_tower.py:391]
                         |
                         v
                  safety_gate.check()
                         |
              +----- pass ------+------ fail -----+
              |                                    |
              v                                    v
        journal.record()                  revert_*_mutation()
        strategy_store.record()           (auto-committed)
              |
              v
        swarm.mark_epoch()
        (invalidates stale Optuna trials)
```

### Controller dispatch

The main loop in `_inner_loop()` (autopilot.py, line 779) loads state, builds the controller context, and asks the LLM controller to choose an action. The controller returns an action dict with `type` field. The dispatcher handles `code_mutation` at line 516 with the same context-assembly and eval/revert pattern as `prompt_mutation` at line 419.

---

## 6. Operational Checklist

### Pre-launch

- [ ] Confirm tap log exists and is being written: `ls -la /mnt/raid0/llm/tmp/inference_tap.log`
- [ ] Verify allowlisted file sizes are normal (not 3 lines):
  ```
  wc -l /mnt/raid0/llm/epyc-orchestrator/src/escalation.py
  wc -l /mnt/raid0/llm/epyc-orchestrator/src/prompt_builders/resolver.py
  wc -l /mnt/raid0/llm/epyc-orchestrator/src/graph/escalation_helpers.py
  wc -l /mnt/raid0/llm/epyc-orchestrator/src/tool_policy.py
  ```
- [ ] Check baseline score via `python autopilot.py status`
- [ ] Check trial counter in state file -- know where you are resuming from
- [ ] Confirm orchestrator stack is healthy: `curl http://localhost:8000/health`

### During run

- [ ] Watch for `"Code mutation rejected"` in autopilot logs -- this is the 4-layer validation working correctly
- [ ] Periodically check git log in epyc-orchestrator for revert commits: `git log --oneline -20`
- [ ] Monitor for empty traces -- if `capture_recent_traces` returns empty, Tier 1 feedback is disabled. Check that inference sessions are running and writing to the tap log.
- [ ] Watch for `"catastrophic shrinkage"` or `"missing public names"` log warnings -- these indicate the LLM is proposing destructive mutations

### Emergency intervention

1. **Pause**: `python autopilot.py pause` -- halts after the current trial finishes (checks `state["paused"]` each iteration)
2. **Restore from checkpoint**: `python autopilot.py restore` (uses latest checkpoint) or `python autopilot.py restore --checkpoint /path/to/checkpoint.json`
3. **Manual git recovery**: If a mutation slipped through validation:
   ```
   cd /mnt/raid0/llm/epyc-orchestrator
   git log --oneline -10          # find the pre-mutation checkpoint commit
   git checkout <commit> -- src/escalation.py   # restore specific file
   ```
4. **Full reset**: `python autopilot.py reset-memory` clears short-term memory to start a fresh optimization trajectory
5. **Kill switch**: If the loop is unresponsive, the process holds a file lock at `orchestration/.autopilot.lock` -- killing the process releases it
