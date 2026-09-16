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

`_screen_code_mutation()` (prompt_forge.py; `_validate_code_mutation()` is the `(valid, reason)` wrapper kept for callers) runs all four checks before a mutation is accepted. Since RTG-55 it is **static only** — it never writes into the source tree and never imports or execs the candidate:

1. **Syntax** (line 587): `ast.parse(mutated)` -- rejects anything that is not valid Python.
2. **Catastrophic shrinkage** (line 594): If the original has >10 lines and the mutation removes >60% of lines, it is rejected. The threshold is `new_lines < orig_lines * 0.4`.
3. **Public name preservation** (line 603): Extracts all module-level `FunctionDef`, `AsyncFunctionDef`, and `ClassDef` names from both original and mutated ASTs. Any name present in the original but missing from the mutation triggers rejection.
4. **Static safety screen** (RTG-55 MHS-2, `screen_static_safety()`): an AST denylist over the candidate — module-level statements that do work at import time, imports outside `SAFETY_IMPORT_ALLOWLIST` (first-party roots and capabilities the original file already used are grandfathered), banned calls (`exec`/`eval`/`compile`/`__import__`/`os.system`/`subprocess.*`/…), and dunder attribute access. `new_file` proposals additionally get the strict inertness profile (`SAFETY_STRICT_NODE_DENYLIST`), so MH-9's "default-inert" requirement is compile-time rather than prompt text. The screen also assigns the candidate's `MutationEffect` (MHS-1).

   The AutoMem `schema_evolution` lane prompt asks for exactly the shape that strict profile admits — module constants (a `SCHEMA` dict, `SCHEMA_VERSION`, `ACTIONS`, `CHANNELS`) plus pure public `def`s returning `(ok, reason)`, with no imports, no classes, no underscore-prefixed names and no `raise`/`with`/`while`/`lambda`. `MEMORY_SCHEMA_SHAPE_EXAMPLE` is shipped verbatim inside that prompt and is itself screened by a test, so the prompt and the denylist cannot drift apart silently.

   Until RTG-55 this layer was an **import test** that wrote the model's code over the live repo file and then `importlib.import_module()`-ed it: the candidate's module top level executed unsandboxed in-process, and any concurrent reader of that path saw the candidate on disk.

`MutationEffect` (MHS-1) is a closed, host-normalized enum — `inert`, `constrain`, `expand`, `replace`, `unsafe`, `unknown` — carried on `CodeMutation.effect` and reported in the `apply_code_mutation*()` result. Both apply paths refuse an `unsafe` effect.

**Anti-leakage guard (RTG-55 MHS-3).** Every prompt, code and GEPA mutation is rejected (`safety_valid=False`, reason `eval_instance_leakage: …`) when the text it ADDS names a specific eval instance: an exact pool/core/sentinel id, a member of an id family derived from those ids (`gsm8k_<n>`, `BigCodeBench/<n>`, `hotpot_bridge_<hex>`), a prompt-hash qid, a suite-anchored reference (`gsm8k question 12`), or a generic qualified reference (`sample #12`, `task_id == 17`). The vocabulary is built from the research `question_pool.jsonl` (required), `benchmarks/prompts/core_*.jsonl` and the sentinel YAMLs, and is cached by file identity (~4 s cold). It **fails closed**: if the pool is missing or unreadable, every mutation is rejected with `eval_leakage_vocabulary_unavailable`. `AUTOPILOT_EVAL_ID_VOCAB_SOURCES` (os.pathsep list, all required) overrides the sources.

**ANTI-OVERRIDE risk prior (RTG-55 MHS-4).** `MUTATION_EFFECT_RISK` weights each effect: inert 0.05 < constrain 0.25 < expand 0.50 < replace 0.90 < unknown 1.0 < unsafe ∞. Prompt mutations are classified too (`classify_prompt_effect`: add-only is constrain, any removed/rewritten line is replace). `CodeMutation.effect_risk` / `PromptMutation.effect_risk` carry the weight, `rank_mutations_by_risk()` orders candidates safest-first, and the gate refuses any mutation whose weight reaches `AUTOPILOT_MUTATION_RISK_GATE` (default 1.0, so only unclassified effects are refused; 0.9 is constrain/expand-only; values above 1.0 are ignored). The weights are an ordinal prior, not a calibrated probability. MHS-5 (`scripts/autopilot/species/heldout_effect_corpus.py` → `orchestration/datasets/harness_r1_heldout_effect_corpus.json`, derived labels only, from Harness-R1 @ `411bb548`, Apache-2.0) backs the ordering: all 4 override patches regressed held-out tasks (mean −8.4 pp) while 19 constrain patches averaged +3.9 pp. It also shows constrain is not regression-free: the single worst patch, at −16.9 pp, is a hint-only constrain patch.

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

---

## 7. Mutations all rejected: eval_leakage_vocabulary_unavailable

The RTG-55 MHS-3 leakage guard (`species/prompt_forge.py`) rejects a mutation that names a
specific eval instance. It builds its id vocabulary from the eval data the tower samples from.
If that vocabulary cannot be built, the guard **fails closed**: every prompt, GEPA and code
mutation is rejected with `eval_leakage_vocabulary_unavailable:<error>`, and the trial is skipped.
The loop keeps running, so this state looks like a quiet stall unless you know where to look.

### Symptoms

| Where | What you see |
|---|---|
| Autopilot log, at start | One `ERROR` line: `EVAL-LEAKAGE PREFLIGHT FAILED (<error>) ... Missing/unreadable source(s): <path> ... FIX: ...`. The autopilot still starts. |
| Autopilot log, per mutation | `Prompt mutation failed transfer safety, skipping: eval_leakage_vocabulary_unavailable:...` (GEPA: `rejected by leakage guard`). |
| Autopilot log, after N rejections | One `ERROR` line: `EVAL-LEAKAGE CIRCUIT OPEN — ...`. N defaults to 3 (`AUTOPILOT_LEAKAGE_ALARM_THRESHOLD`). |
| Journal ledger | Rows with `"type": "eval_leakage_guard"` and `event` set to `preflight_failed`, `alarm_raised` or `alarm_cleared`, in `orchestration/autopilot_journal*.jsonl`: `grep -h '"type": "eval_leakage_guard"' orchestration/autopilot_journal*.jsonl` |
| Rejection ledger (AP-53) | `orchestration/autopilot_rejected_mutations.jsonl` rows with `rejecting_gate: "transfer_safety"` and `gate_detail` starting `eval_leakage_vocabulary_unavailable`. |
| Operator alarm | Session-bus alarm key `autopilot-eval-leakage-vocab-unavailable`, severity critical, delivered through `scripts/coordination/alarm_channel.py`. It notifies once, then re-asserts at most every 900 s (`AUTOPILOT_LEAKAGE_ALARM_REASSERT_S`). Check it with `python3 /mnt/raid0/llm/epyc-root/scripts/coordination/alarm_channel.py status`. |
| Dashboard | The autopilot control line shows `MUTATIONS REJECTED: eval-leakage vocabulary unavailable (N consecutive[, alarm raised])` in red; hover for the error, paths and this runbook. The raw data is `/dashboard/api/process_status` → `autopilot_state.eval_leakage_guard`, which is persisted in `autopilot_state.json` under `eval_leakage_guard`. |

### Cause

At least one REQUIRED vocabulary source is missing, unreadable or unparseable, or the sources
contained no ids. The required source is the research question pool:

```
/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl
```

That path is `$EPYC_RESEARCH_ROOT/benchmarks/prompts/question_pool.jsonl`, with
`EPYC_RESEARCH_ROOT` defaulting to `/mnt/raid0/llm/epyc-inference-research`. When
`AUTOPILOT_EVAL_ID_VOCAB_SOURCES` is set (`os.pathsep`-separated), it replaces the default list
and **every** listed file becomes required. Optional sources (`benchmarks/prompts/core_*.jsonl`,
`scripts/autopilot/sentinel_questions.yaml`, `tool_sentinels.yaml`) are skipped when absent.
The error suffix names the failure: `missing_eval_id_source:<path>`,
`eval_id_source_unreadable:<exception>`, `no_eval_ids_found` or `no_eval_id_sources`.

### Check

```bash
ls -la /mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl
sha256sum /mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl
# expected: 1350221880 bytes, 64218c27e07400acf3b10a3cac05a410d5ee67814f353788ab75a19c84dde584
# (the pool pinned in epyc-root artifacts/audit/deterministic-rescore-ledger-20260812.json)
echo "${AUTOPILOT_EVAL_ID_VOCAB_SOURCES:-<unset: default sources>}"
```

To prove that the guard can build its vocabulary, run the same loader offline (zero inference):

```bash
cd /mnt/raid0/llm/epyc-orchestrator && .venv/bin/python -c "
import sys; sys.path[:0] = ['.', 'scripts/autopilot']
from species.prompt_forge import load_eval_id_vocabulary as L
v = L(); print('available' if v.available else 'UNAVAILABLE', v.error, len(v.ids))"
```

### Fix

1. **Restore the byte-identical pool.** `question_pool.jsonl` is **not in git**: it is
   gitignored (`benchmarks/prompts/question_pool*.jsonl`) in `epyc-inference-research`, so
   `git checkout` cannot restore it. Copy it back from a byte-identical copy (verify the sha256
   above) and keep the path exactly as it was.
2. **Or point the guard at a valid copy.** Set
   `AUTOPILOT_EVAL_ID_VOCAB_SOURCES=/path/to/question_pool.jsonl` in the autopilot's environment.
   Every source you list is required. The copy must be the SAME pool the eval tower samples from;
   a different pool gives the guard a vocabulary that does not match the eval set.
3. **Do NOT run `question_pool.py --build`, and do NOT copy
   `pool_rebuild_a3_20260721/question_pool.activated.jsonl` over the live pool**, just to make
   this alarm go away. The pool is the EVAL INSTRUMENT: the activated copy is the pre-2026-07-26
   amendment pool (sha256 `9b433fa7…`), and a rebuild pulls in a registry that has since grown.
   Either one is an eval-instrument change, which is a human-owned measurement boundary. If no
   byte-identical copy exists, regenerating the pool is the operator's instrument transaction
   (`epyc-root/artifacts/operator/e8_quality_pool_regenerator.py`, a deterministic replay from
   the activated pool), not an autopilot fix.

### Restart needed?

- **Default path restored in place: no restart.** The vocabulary cache is keyed by file
  identity `(path, mtime_ns, size)`. A missing file is never cached, so the next mutation
  rebuilds. A file that was present but unparseable is negative-cached for at most 60 s
  (`AUTOPILOT_EVAL_ID_VOCAB_NEG_TTL_S`), and only under its old identity: replacing the file
  changes that identity, so the next mutation rebuilds at once.
- **`AUTOPILOT_EVAL_ID_VOCAB_SOURCES` changed: restart required.** A running process cannot see
  an environment change. Restart through the normal autopilot lifecycle.
- **Cost of the rebuild.** Since 2026-09-16 (`sub-mhs3b`) the same pass also builds the eval
  CONTENT index (below). Measured on the real pool (79,613 rows, 88.9M tokens): about **16 s**
  cold, **+590 MB RSS** kept for the process lifetime (the id vocabulary was +253 MB; the content
  index arrays are 173 MB, 18.5M packed shingles at 8 B each plus a 16 MB sketch), and a
  **~1.37 GB transient peak** during the build. A cached lookup costs about 9 ms for a small
  prompt edit and about 280 ms for a whole 2,700-line file. The harness corpus (prompts plus
  `src/**/*.py`) takes 0.17 s cold and 3 ms warm (a stat per file). The startup preflight pays
  the build once at boot; after a mid-run restore, the first mutation pays it. If the content
  index cannot be built, or a content check raises, the guard fails closed with
  `eval_leakage_vocabulary_unavailable:<...>` (`eval_content_index_missing`,
  `content_check_failed:<exc>`, or `eval_id_source_unreadable:<exc>`) and the same alarm applies.

### Confirm recovery

The circuit closes on the next mutation the guard evaluates, not on the file restore itself.

- Log: `EVAL-LEAKAGE CIRCUIT CLOSED — vocabulary available again after N rejection(s); alarm cleared`.
- Journal: an `eval_leakage_guard` row with `"event": "alarm_cleared"`.
- Alarm: `alarm_channel.py status` no longer lists `autopilot-eval-leakage-vocab-unavailable`,
  and the channel sends one RESOLVED notification.
- Dashboard: the red `MUTATIONS REJECTED` suffix is gone, and `eval_leakage_guard` shows
  `consecutive_unavailable_rejections: 0` and an empty `last_error`.
- New `autopilot_rejected_mutations.jsonl` rows stop carrying `eval_leakage_vocabulary_unavailable`.

If the alarm never fired (fewer than N rejections), only the log and dashboard lines change.

### Reading rejection ledgers: `eval_instance_leakage` false positives

A different reason, `eval_instance_leakage: refs=[...]`, means that the vocabulary WAS available
and the guard matched an instance reference in the text the mutation ADDS. `refs` lists the
matched strings. Before you treat a rejection as real memorisation, check which shape matched:

- **Exact id or id-family member** (`gsm8k_00003`, `BigCodeBench/42`, a 16-hex prompt-hash qid):
  almost always a real leak.
- **Generic instance pattern** (`sample #12`, `question id: 42`, `problem number 3`,
  `task_id == 17`): a real leak in prose. **Known false-positive class:** code-shaped lines. Before
  2026-09-16 the pattern also matched snake_case assignment and mapping lines such as
  `task_index = 0`, `sample_id = 1` and `task_id: 7`, which are ordinary code in a code mutation.
  The pattern now counts a snake_case identifier only as a comparison (`==` / `is`), so those
  three forms no longer match. Comparison forms (`if task_id == 17:`) still match by design.
  Older ledger rows may still carry the assignment and mapping forms; read those as false
  positives.
- **Suite-anchored reference** (`math problem #4`, `gsm8k question 12`): a real leak.

### Reading rejection ledgers: the eval-content refusals (2026-09-15 structural form)

Four more reasons come from the same guard. Each records the matched source, formatted as
`<source file>:<row id>` (for example `question_pool.jsonl:simpleqa_general_00912`), or as
`trace_bank[<entry label>]` together with the traced question ids. All four look only at the text
the mutation ADDS. Text already in the original file, in any current prompt, or anywhere in
`src/**/*.py` is never blamed on a mutation.

| Reason | Fires when | Usually | Known false-positive class |
|---|---|---|---|
| `eval_content_ngram_overlap: source=... ngram="..." shared=N` | The added text shares a verbatim run of 8 or more tokens (case- and punctuation-insensitive) with an eval row, or with a trace in the proposer's context. The 8-gram must contain a rare token, or sit inside a copied run of 15 or more tokens. | A copied question, passage or model response. | Public code idioms or paper titles that also appear in a long-context document (`discriminant = (b * b) - (4 * a * c)`, a `find_class` signature). They are rare, and none are refused once the idiom exists in `src/`. |
| `eval_expected_answer_leakage: source=... answer="..."` | The added text contains a whole expected answer of 1 to 7 tokens. The answer must be name-shaped (upper case, digit, underscore or non-ASCII) or unique. Numbers, placeholders and plain lower-case words are never indexed. | A hard-coded answer, such as `return "has_close_elements"` or a named entity. | Tool or identifier names shared with the BFCL and MBPP suites (`get_time`, `list_directory`, `http_post`, `word_len`), and proper nouns such as `DuckDuckGo` or `Eureka`. These fire only in NEW text that names something `src/` does not already use. |
| `eval_source_identity_leakage: qids=[...] trace=[...] shared_tokens=[...] n=N` | The proposer's context carried a trace (MH-7 contrastive, MH-11 IR or recent traces) that resolves to an eval question, and the added text shares 3 or more rare tokens with it. Paraphrase does not avoid this. | A few-shot example rewritten from a failed eval trace. | Domain vocabulary shared with a trace by coincidence. The question id is always named, so check it against the trace. |
| `eval_suite_special_casing: suite=<name> form=keyed\|phrase\|mention` | The added text keys behaviour on an eval suite. `keyed`: `suite == "math"` or "the math benchmark". `phrase`: "USACO problems". `mention`: a benchmark-shaped name whose count the mutation increases. It fires even when the suite IS in the source context. The AP-33 `domain_mismatched_anchoring` and `misapplied_best_practice` reasons keep precedence. | A benchmark-keyed tactic ("Function tasks (LeetCode/HumanEval): ..."), or code that branches on a suite. | A comment that names a benchmark as an example (`# e.g. HumanEval/MBPP`). Suite names the harness itself uses in 4 or more files (`web_research`, `long_context`, `agentic`) count only in the `keyed` form. |

**Measured false positives (2026-09-16, real pool, offline).** For the whole-file measurements
below, each file was treated as new text, and a file was never excused by its own copy in the
harness.

- **The 33 live mutation targets.**
  - A benign edit to each target: **0/33** refused. This is pinned by
    `test_real_pool_cost_coverage_and_no_false_positives_on_live_targets`.
  - Each target as a whole file: **0/33** content refusals, and 5/33 suite mentions. Those 5 are
    the benchmark names already written into `architect_investigate.md`, `planning_agent.md`,
    `coder_escalation.md`, `chat.py` and `tool_policy.py`.
- **467 `src/` and `docs/` files as whole files:** 34 flagged.
  - Content-refusal false positives: 11/467 (2.4%). With the production harness, the 8 `src/`
    cases are exempt.
  - True positives: 1 real answer reference and 2 real instance references.
  - The other 20 flags are suite mentions: about 14 name a benchmark on purpose, and about 6 are
    harness identifiers (`long_context`, `tool_use`).
- **91 historical autopilot mutation commits:**
  - 0/91 content refusals.
  - 10/91 special-casing refusals. 8 of these were special-casing by the new spec: suite-keyed
    `tool_policy` code, "HumanEval/MBPP" prose, AIME routing, and the corrupted `frontdoor.md`
    commentary.
  - 2/91 (2.2%) were borderline refusals of comments.
- **Detection coverage:** 1,998 of 2,000 verbatim pool prompts were refused.

**Pre-existing leak found by this guard.** `orchestration/prompts/rules.md:67` and
`src/prompt_builders/constants.py` both carry the pool question `simpleqa_general_00912`
("At which university did Jurgen Aschoff study medicine?") verbatim as a worked example. The guard
exempts text that is already present, so this needs an owner's edit.

