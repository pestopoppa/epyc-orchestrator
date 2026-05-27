# Bug report: REPL loop-guard "hard intervention" never fires (BEP-2 multi-file) — 2026-05-27

## One-line
The flag-gated REPL loop-guard's hard intervention never engages because its counter is stored
as an **ad-hoc attribute** (`state._repl_repeat_count`) that does **not survive the graph's
between-turn state snapshot/restore** (which preserves *declared* `TaskState` fields only). The
counter runs but resets to 0 every turn, so it never reaches the fire threshold.

## Goal / context
Pursuing fix-track (A) for BEP-2: the coder loops on read-first multi-file tasks (re-reads / empties
out, never writes+FINALs). A "hard intervention" was added to `_execute_turn`: after N consecutive
non-advancing turns, append a forceful directive (re-inject already-read content + demand a single
write/FINAL). It must FIRE to be tested. It does not fire.

## Symptom (from instrumented traces, `data/bep_sandbox/results-readfix5/traces/*off*.jsonl`)
- `prompt_has_loop_halt = False` on **every** turn of every read-first task.
- `repeat_count_seen = None` on **every** turn (the value the nudge saw = `state._repl_repeat_count`).
- Read-first tasks (t2/t3/t4/t5) run to `turns=8`, `quality_pass=False`. Only t1 (create, no read) passes.

## Expected mechanism (`src/graph/helpers.py`)
- End of `_execute_turn` (~line 971): if loop-guard flag on, `state._repl_repeat_count =
  _loop_guard_noprogress(made_progress, prev)` — increments when the turn made no progress
  (no `file_write_safe`, no `FINAL`), resets on progress.
- Prompt build (~line 734): if `_repl_repeat_count >= 2`, append the "LOOP HALTED" directive.
- Trace (~line 918): records `repeat_count_seen = getattr(state, "_repl_repeat_count", None)`.

## Evidence gathered
1. **Flag is live:** `ORCHESTRATOR_REPL_LOOP_GUARD=1` confirmed in the uvicorn master env
   (`/proc/2611119/environ`). API runs `--workers 6` (orchestrator_stack default).
2. **Worker provably has the bep env:** `ORCHESTRATOR_BEP_TURN_TRACE=1` is set by `bep_ab._restart_api`
   in the **same env dict** via the **same** `os.environ.copy()` propagation (orchestrator_stack.py:976,
   full passthrough, no allowlist). The trace fires + writes → the request-handling worker DOES have
   the bep env → it also has `REPL_LOOP_GUARD` → `_repl_loop_guard_enabled()` returns True →
   **the counter at line 971 runs.** (This rules out "flag absent from worker".)
3. **Declared fields persist; ad-hoc attrs do not:** `state.turns` advances 1→8 and `state.last_output`
   reaches the next prompt (`prompt_has_last_output=True`) — both are **declared** `TaskState` fields.
   `_repl_repeat_count` is an **ad-hoc** `state._x` attribute and reads None every turn.
4. Code is internally consistent (flag fn, counter, nudge, trace all referenced the same attr).

## Hypotheses considered and ruled out
- **auto_wrap_final synthesizes FINAL → made_progress=True every turn → resets counter.** RULED OUT:
  tested the pure fn — `auto_wrap_final('')`, `auto_wrap_final("<peek block>")` do NOT add FINAL.
- **State recreated per turn (dataclasses.replace / new TaskState).** RULED OUT: `state.turns += 1`
  mutates in place (helpers.py:565); graph.py:74-82 token restore is **resume-only** (guarded by
  `if resume_token:`), not per-turn.
- **Early return before the counter for empty turns.** RULED OUT: the only early return before line 971
  is the batch-edit path (`_batch_edit_result is not None`), which is None on the OFF arm.
- **Flag not propagated through `orchestrator_stack reload`.** RULED OUT: launch env is
  `os.environ.copy()` (passthrough), and BEP_TURN_TRACE (same mechanism) demonstrably reached the API.

## Root cause (PROVEN symptom; mechanism NOT established — corrected per operator review 2026-05-27)
**PROVEN:** the ad-hoc counter (`state._repl_repeat_count`) was **not observable at the next
prompt-build boundary** (`repeat_count_seen=None` every turn), so it never reached the `>= 2` threshold
and the hard intervention never appended.

**NOT PROVEN (earlier draft overstated this):** *why* it wasn't observable. The earlier claim "the graph
snapshots/restores state from declared fields, dropping ad-hoc attrs" is **not established** — on the
default path `graph.py:91` passes the **same** state object and `helpers.py:563` mutates it in place, so
an ad-hoc attr *should* persist there; the LangGraph bridge that would drop it is **default-off**. So the
None observation supports "counter absent at prompt-build" but NOT specifically the snapshot story. The
precise cause (counter not set on the live API path? a per-turn state copy not yet found? a path that
does snapshot?) is **unknown**.

**Fix rationale (independent of the unknown mechanism):** storing the counter in a **declared**
`TaskState.repl_noprogress_count` field is robust across same-object execution, snapshotting, and a future
LangGraph conversion — the right fix regardless of which mechanism applies. But it is **NOT yet proven to
resolve the live failure** (see Status). (`_alarm_retried` shares the same ad-hoc-attr fragility.)

## Fix applied (this session, UNVERIFIED — Bash classifier was unavailable to run/verify)
- `src/graph/state.py`: added a **declared** field `repl_noprogress_count: int = 0` to `TaskState`.
- `src/graph/helpers.py`: switched the counter / nudge / trace from the ad-hoc `state._repl_repeat_count`
  to the declared `state.repl_noprogress_count` (so it survives the between-turn snapshot).
- Added a prod-safe diagnostic probe gated on `ORCHESTRATOR_LOOPGUARD_PROBE=1` (logs `enabled`/`count`
  per turn) — off by default so it won't spam J6.

## How to confirm (the decisive check — exactly one A/B)
1. (Optional, removes all doubt) set `ORCHESTRATOR_LOOPGUARD_PROBE=1` in `bep_ab._restart_api` env,
   run `bep_ab.py --reps 1 --host-quiet-confirmed`, grep `logs/orchestrator.log` for `LOOPGUARD-PROBE`:
   expect `enabled=True` and `count` climbing 0→1→2 on a read-first task.
2. Inspect `results-<run>/traces/*t2*off*.jsonl`: expect `repeat_count_seen` to climb and
   `prompt_has_loop_halt=True` once it hits 2 — i.e. the hard intervention now FIRES.
3. Then read the verdict: do read-first tasks (t2/t3/t5) now `quality_pass=True`?
   - **fires + tasks pass** → the advisory nudge was the gap; BEP-2 multi-file unblocked.
   - **fires + tasks still fail** (model gets content + the forceful directive, still can't complete)
     → that is the proof it's coder-model capability, not the harness.

## Repro
`cd epyc-orchestrator; python3 scripts/benchmark/bep_ab.py --reps 1 --max-turns 8 --host-quiet-confirmed
--output data/bep_sandbox/results-X` (reloads API with loop-guard flag; pause J6 first to keep host quiet).
Inspect `data/bep_sandbox/results-X/{results.jsonl,traces/*.jsonl}`.

## Key files / lines
- `src/graph/helpers.py`: `_repl_loop_guard_enabled` (~183), `_loop_guard_noprogress` (~201),
  hard-intervention nudge (~734), counter (~971), trace (~918).
- `src/graph/state.py`: `TaskState.repl_noprogress_count` (new declared field).
- `src/graph/graph.py:74-82`: resume-token restore (resume-only).
- Traces: `data/bep_sandbox/results-readfix5/traces/` (repeat_count_seen=None evidence).

## Status (updated 2026-05-27 post operator review)
- Fix committed `4fe681e`; `py_compile` + 18 unit tests + 2 new multi-turn integration tests
  (`tests/integration/test_execute_turn.py::test_loop_guard_*`) all pass. J6 RESTORED.
- **CAVEAT — do not overclaim:** the new integration test drives `_execute_turn` directly in a loop on the
  SAME `state` object, so an ad-hoc attr would persist there too — it would pass PRE-fix. It proves the
  nudge-injection LOGIC (counter → `LOOP HALTED` at >=2) and matches the requested test shape, but it does
  NOT reproduce the live persistence failure and does NOT prove the declared-field fix resolves the A/B.
- **Decisive close (still outstanding):** one live A/B with `ORCHESTRATOR_LOOPGUARD_PROBE=1` (pause J6).
  The probe logs `enabled`/`count` per turn on the LIVE path → shows whether the counter now ACCUMULATES
  (proves persistence was the cause + the fix works) or still resets (fix is a no-op → mechanism elsewhere).
  Then `prompt_has_loop_halt` + `results.jsonl` show whether the intervention fires and whether read-first
  tasks pass — the binary (A) verdict (advisory-was-the-gap vs coder-capability).
