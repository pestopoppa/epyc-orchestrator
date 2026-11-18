# PREDECLARED_v2 (DRAFT): CODEMODE-AB probe v2

STATUS: DRAFT. Not published, not run, no row exists. It becomes binding only when the operator approves it and the
sha256 of `score_v2.py`, `smoke_check_v2.py` and the harness tree are pinned in `precond.sh` before the FIRST live v2 record
(the v1 rule: the verdict comes ONLY from a sealed scorer). Change nothing after that.

Inputs: `CODEMODE_FAILURE_ANALYSIS.md` (why v1 code mode failed), `SELECTOR_HARDENING.md` (tool-picker design), both in
`/mnt/raid0/llm/tmp/codemode-ab-20261117/analysis/`. v1 records are reused as controls only if kernel, model and spec hashes match.

## Question

- **Q1.** Do the harness fixes (explicit FINAL, prompt v2 with the sandbox contract, wide preview, compact `list_tools`) recover code
  mode to parity with arm A on the static tier? (B1 vs B)
- **Q2.** Does turn memory add success, most of all on the 27B? (B2 vs B1)
- **Q3.** Does the full code-mode stack (typed stubs + hardened picker + name repair) match or beat direct calls that ALSO get the picker,
  at a real token saving on PAIRED SUCCESSES? (C3 vs A_pick, primary)
- **QP.** Does the hardened picker help or harm the DIRECT-CALL path, per model (35B, 27B, Flash-Next)? (A_pick vs A)

## Flags (all default OFF in production; the probe sets them per arm in `arms.ARM_ENV`)

| Flag | Fix |
|---|---|
| `ORCHESTRATOR_REPL_EXPLICIT_FINAL=1` | only an explicit `FINAL(` ends an episode; a one-line tool call is never auto-wrapped |
| `ORCHESTRATOR_CODEMODE_PROMPT=v2` | no STOP-after-CALL, no LeetCode/`run_python_code`, sandbox contract from `security.py`, one batch-loop few-shot |
| `ORCHESTRATOR_REPL_TURN_MEMORY=4` | last 4 turns of code + output/error in the prompt, 4000-char previews, appended after every other section |
| `ORCHESTRATOR_CODEMODE_STUBS=1` + `ORCHESTRATOR_STRUCTURED_READ_ONLY=1` | typed `def` stubs for the picked tools; bound functions return Python objects |
| `ORCHESTRATOR_TOOL_SELECTOR=embed` | hardened C1 picker: core pin, adaptive k 4..12, family cap 3, ladder, 500 ms budget + breaker, memo, frozen pick, server-side `list_tools`/`search_tools`/`describe_tool` |

The probe measures exactly what production would switch on: the arms call the production functions (`auto_wrap_final`, `PromptBuilder`,
`tool_picker.decide`, `select_native_tools_ex`, `execute_server_tool`), not harness re-implementations.

## Arms (same 22 tasks x 3 seeds, same registry, paired on (model, task, seed))

| Arm | What | Turn cap |
|---|---|---|
| A | unchanged v1 arm A (native calls, static list + `list_tools`/`call_tool`) | 40 |
| B | unchanged v1 arm B (control for the harness-fix delta) | 12 |
| **A_pick** | A, but the function list is produced by `select_native_tools_ex` (hardened C1 picker) and the model gets server-executed `list_tools`/`search_tools`/`describe_tool` instead of `call_tool`; a searched/described tool becomes callable next step | 40 |
| **B1** | B + explicit FINAL + prompt v2 + 4000-char preview + compact `list_tools` | 40 |
| **B2** | B1 + turn memory (4 turns) | 40 |
| **C3** | B2 + typed stubs for the picked tools (core pin + adaptive k, picked by the production policy over C1 scores), CM-3 object returns, near-miss tool-name repair (`validate.py`, difflib 0.85) | 40 |

- **Turn cap.** v2 code-mode arms get arm A's cap (40), the operator's call. This differs from the analysis (which kept 12 so a gain
  could not be a cap effect). Consequence: B1 vs B mixes the fixes with the cap (12 -> 40). The cap's own effect is isolated by the
  cap-sensitivity cell below, not by the contrast.
- **Picker is C1.** The embedding picker (BGE through the pooled embedder) is THE picker for A_pick and C3. The analysis proposed the C2
  top-8 for A+P/C3; superseded by the operator: C2 may exist only as an optional background shadow rerank and is NOT an arm here.
- **Discovery cap fix.** In v1 `syn_sku_price` sat at char ~33.8k of the REPL `list_tools()` dump (preview 1500) and ~20.7k of arm A's
  (cap 12000): invisible to both. v2: `list_tools` returns compact `[{name, description[:80]}]` (about 11 kB for 123 tools), arm
  A_pick's tool-result cap is 24000, the REPL preview is 4000 chars for every v2 REPL arm, and the v2 prompt tells the model to print
  names only (about 2.2 kB). `tests/unit/test_codemode_ab_v2.py::test_syn_sku_price_is_visible_inside_every_v2_cap` proves
  it. Arm A and B keep the v1 behaviour (v1 comparability); A_pick is the fair-discovery control.

## Models (parameterised: `run_probe.py --model-key 35b|27b|flash`, one model per row)

| Key | Model | Kernel | Registry role | Why |
|---|---|---|---|---|
| 35b | Qwen3.6-35B-A3B-MTP Q8_0 | CPU | frontdoor | v1 baseline; CPU prefill makes prompt tokens expensive |
| 27b | Qwen3.8-27B Q8_0 | MI210 GPU | architect_critic / coder_escalation | v1 REPL arms floored (0.03), native A scored 0.30: the native picker is the only measurable effect |
| flash | Qwen3.8-Flash-Next UD-IQ4_XS | CPU | architect_general | most heavily quantized: largest expected distractor penalty |

Each model is an independent row on its own private ports and its own results file; verdict and stops are per model. No CPU and GPU rows
are co-resident (the v1 scheduling rule). The Flash-Next serial role has 1 slot: its A_pick rows must not call the picker inline
on its lock (the picker uses the embedder pool, not the model).

## Metrics (every episode record)

As v1 (success by end-state oracle, in/out/cached tokens, selector tokens, wall_s, turns, failed_snippet_rate, param_accuracy,
duplicate_writes, selector recall) plus:

- **per-turn failure-reason enum** from the transcript: `ok, ast_refused, syntax, name_error, tool_arg_error, tool_error, no_code,
  implicit_final, explicit_final, other_error` (`arms.classify_turn`; counts in `turn_failure_reasons`);
- `final_kind` (`explicit | implicit | none`), `max_turns`, `name_repairs`, `transcript_path`, `transcript_turns`;
- **token ratio on PAIRED SUCCESSES only** = the token claim (median ln of total tokens X/Y over pairs where both succeeded);
  the all-episode ratio is descriptive only (a failing episode that stops early is cheap);
- picker (A_pick, C3): selected set, scores top 16 (`picker_events.jsonl`), k, rung (`none|widen|full|core_search|lexical|timeout|
  breaker_open|error`), latency, `called_not_selected`, `proxy_recall`, `recovery_fired`, `breaker_opens`.

Transcripts: every episode writes `transcripts/<unit>.jsonl` (one object per turn: prompt hash, whole prompt for REPL turns and for
turn 0 of A arms, completion, extracted/executed code, output, error, failure reason, final flags, tokens).

## Contrasts (X vs Y, `score_v2.CONTRASTS_V2`)

| Key | Contrast | Isolates | Question |
|---|---|---|---|
| Q1a | B1 vs B | harness fixes (+ cap 12 -> 40) | Q1 |
| Q2 | B2 vs B1 | turn memory | Q2 |
| Q2b | C3 vs B2 | stubs + picker + name repair | |
| **Q3** | **C3 vs A_pick** | **primary** | Q3 |
| Q3a | C3 vs A | secondary, comparable with v1 | |
| **QP** | **A_pick vs A** | picker on the direct-call path | QP |

## Expected effect versus noise ([inference], from the v1 counts; not measured)

- Q1a on 35B: about +0.25 (F1 premature FINAL was 9/14 discordant pairs); >= 9/42 one-way discordant pairs makes the sign test (p <= 0.004)
  resolvable at n_le_20 alone.
- Q2 on 27B: about +0.2 (L, the exploration loop, was 19/19 discordant pairs); resolvable pooled (66 pairs).
- Q3: plausible range -0.1 to +0.2; NOT resolvable below about 0.15 to 0.2 even pooled. A null there reads "parity at lower cost", not a loss.
- QP: +0.1 to +0.3 success for 35B and Flash-Next when the list is much larger than k, 0 to +0.15 for 27B, no change when the list is <= k.
  Selector recall precondition (below) must hold first.
- Noise: 22 tasks x 3 seeds = 66 pairs per model; a pooled success gap under about 0.15 is unresolvable, one N cell under about 0.3
  is not (a sign test needs >= 6 discordant pairs all one way); token and wall ratios resolve 10 percent at 12 pairs.

## Checkpoints and stop rules

Futility-only at interim looks. Formal gates stay undecided per OP-ABSEQ / OP-PQP / OP-ALPHAWIN (`MEASUREMENT.md`); a stop-on-answer is
graded `descriptive_interim`, never a gate claim. A capped episode is a failure; `llm_error` is censored with its pair (as v1);
the verdict comes only from `score_v2.py`, sha pinned first.

**Stage 0 smoke (35B first, 2 tasks x all arms; then once per additional model).** `smoke_check_v2.py OUT --mode smoke` must exit 0.
ABORT the run on any of:
1. any episode without a transcript file, or a transcript whose line count differs from `transcript_turns`;
2. **any B1, B2 or C3 episode that ends `final` without an explicit `FINAL(` in the model's own code** (`final_kind != explicit`): the
   explicit-FINAL fix is not live;
3. a persisted first-turn prompt of B1/B2/C3 that still contains `STOP generating`, `shortestPalindrome`, `run_python_code` or `ALWAYS test`;
4. a v2 REPL arm whose `max_turns` is not 40; a missing per-turn failure-reason enum; an `llm_error` / harness-crash episode;
5. content-based freshness: the harness tree hash is pinned in `precond.sh`; a stale artefact is not evidence.
Additionally, per model: the **selector-recall precondition** (no model needed, only the embedder): replay the 66 task prompts through the
hardened pick offline and require any-of recall >= 0.95 and mean k <= 9; below that A_pick/C3 are not worth the model window.

**Stage 1 (singles + N in {5, 20}), 35B first.**
- Stop-on-answer (descriptive) for Q1 if B1 vs B has >= 6 discordant pairs, all one way.
- FUTILITY stop of the WHOLE v2 if Q1a shows |d_success| < 0.05 on every scored model (`interim.stop_on_futility`): the diagnosis in the
  analysis is wrong; re-run the taxonomy on the transcripts before spending more compute.
- Then the 27B and Flash-Next. FUTILITY on Q2 if B2 vs B1 shows |d| < 0.05 on every scored model's n_le_20 (`interim.drop_b2`):
  drop B2 from later stages.
- QP is evaluated per model with the harm stops below.

**Stage 2 (N = 50).** C3 vs A_pick futility: |d_success| < 0.05 AND |median ln token ratio on paired successes| < ln 1.10
(`interim.c3_vs_apick_futile`) -> stop that comparison (no gap on either axis). Descriptive stop-on-answer when the CI excludes 0 and all
seeds agree.

**Stage 3 (N in {100, 200}).** Only C3, A_pick and A (the token hypothesis lives at large N). Terminal = complete.

**Cap-sensitivity cell.** Because the v2 cap is 40, run a cap-12 cell for B1 and C3 on N <= 20 only when more than 20 percent of C3
episodes end `max_turns`, so the cap's own contribution to Q1a/Q3 is measured instead of assumed.

### Harm stops for the direct-call picker (QP), per model, checked at every look (`score_v2.harm_check`)

HARD (the hardened picker) stops and **reverts to OFF for that model/role** if ANY holds:
- success lower than OFF (A) by >= 0.10 with >= 6 discordant pairs all against A_pick;
- proxy recall `1 - |called_not_selected| / |called|` below 0.90 over >= 50 episodes;
- `recovery_fired` (server tool call or re-pick) above 10 percent of episodes over >= 50 episodes;
- p95 picker latency above 500 ms;
- the circuit breaker opened more than twice.

FUTILITY for a model: pooled |d_success| < 0.05 AND |median ln token ratio on paired successes| < ln 1.10 AND |median ln wall ratio| <
ln 1.10 on n_le_50 means HARD adds nothing measurable there; that is the answer for that model (stop its QP cells).

## Known limitations (declare in the verdict)

- A and B keep v1 behaviour (truncated `list_tools`, 12-turn B) for comparability; A_pick and B1+ are the fair-discovery arms.
- `run_python_code` and shell stay blocked; B1+ no longer advertise them.
- In production `list_dir` returns a dict `{path, files:[{name,type,size}], total}` capped at 100 entries; the probe's `list_dir` returns
  JSON text of names. The v2 prompt's few-shot follows the stubs flag for the shape. Production behaviour at N >= 100 files is therefore
  NOT measured by this harness (the 100-entry cap).
- Fixes 1 to 4 and the picker live in production code paths behind flags; the verdict is about those flags, on this synthetic registry
  (123 tools, 60 same-domain distractors): crowding is a property of the registry, the family cap is calibrated on it.
- Margin `delta`, `k_min`/`k_max` and the 40-tool full-list rung are DEFAULTS, not calibrated: no scores were logged in v1. v2 logs them;
  they are tuned only from logged shadow data, never inside this probe.
- C2 is not an arm. Its optional shadow rerank (off by default) logs `shadow` rows for a later offline comparison.
- Held-out claim grammar: nothing here is a gate claim until OP-ABSEQ / OP-PQP / OP-ALPHAWIN are decided.

## Files (all in this directory, pinned before the first live record)

`arms.py`, `run_probe.py`, `score_v2.py`, `smoke_check_v2.py`, `validate.py`, `models_v2.py`, `world.py`, `mock.py`, `tasks.py`.
Do not publish a row from this draft; do not run it.
