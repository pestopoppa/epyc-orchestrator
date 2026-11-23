# AutoResearch Program — EPYC Orchestrator Optimization

> Strategy document for autonomous experimentation.
> The controller reads this file every trial and follows it.
> Human steers by editing this file. Agent executes experiments autonomously.

---

## Setup

Before each experiment session:

1. Create a run tag: `run_YYYYMMDD_HHMMSS`
2. Read current state:
   - `orchestration/autopilot_state.json` — Pareto archive, trial counter
   - `orchestration/autopilot_journal.tsv` — last 20 trials
   - `orchestration/autopilot_baseline.yaml` — frozen baseline
3. Verify stack health:
   - All configured servers responding (hit `/health` on each port)
   - Debug suite runnable: `python scripts/benchmark/seed_specialist_routing.py --dry-run`
4. Read recent failures:
   - `grep "discard\|crash" orchestration/autopilot_journal.tsv | tail -20`
   - Do NOT re-attempt experiments that match recent failure patterns
5. Create a git branch: `autoresearch/{run_tag}`
6. Commit current state as baseline: `git add -A && git commit -m "baseline: {run_tag}"`

---

## What You Can Modify

**Principle**: You can modify ANYTHING in the orchestrator codebase as long as strict git versioning is in place. Every change is committed before eval, and reverted if quality regresses. Git is the safety net — not file-level permissions.

### Hot-Swap (no restart, immediate effect)
- `orchestration/prompts/*.md` — prompt templates (read on every request)
- Feature flags via `POST http://localhost:8000/config` — runtime toggle
- `orchestration/classifier_config.yaml` — risk thresholds, classification rules
- `orchestration/tool_registry.yaml` — tool registration and wiring
- Q-scorer weights, think-harder thresholds, MemRL retrieval params
- Cheap-first quality threshold and escalation policy parameters

### Code Changes (require API restart for affected component)
- `src/graph/` — REPL loop logic, escalation flow, tool dispatch, early-exit conditions
- `src/api/routes/` — routing logic, pipeline behavior, request handling
- `src/tools/` — tool implementations, tool registration, tool wiring
- `src/classifiers/` — classification logic, risk scoring, routing decisions
- `src/features.py` — feature flag definitions and defaults
- Any bug fix, wiring fix, or behavioral improvement that eval validates

### TOON & Context Transfer
- TOON encoding parameters
- Escalation AND consultation context format and content
- Prompt compression strategies for all model tiers
- Architect consultation → compact plan → live fast execution pathway from the generated system card (explore this — architects can provide compressed high-info plans that get redelegated to current execution roles, not just terminal escalation)

### Specialist Pipelines
- Vision/OCR pipeline configuration
- Embedding model selection
- File extraction parameters

### Guarded: Model Registry & Stack Config

**`orchestration/model_registry.yaml`** and **`scripts/server/orchestrator_stack.py`** are the product of months of isolated benchmarking (single-model throughput, NUMA-aware multi-instance, acceleration flags, quantization selection). Do NOT blindly explore these.

Rules for touching model registry or stack config:
1. **Never change model selection, quantization, or NUMA assignments** without explicit human approval
2. **Never change acceleration flags** — these are already optimized per-model from benchmark data. Treat the generated system card and current benchmark handoffs as the source of truth for which accelerators are allowed on the live stack.
3. **Timeouts and token caps** in model_registry.yaml ARE safe to tune — these are routing parameters, not infrastructure
4. **If swapping a model for a role**: restart ONLY that role's server process, NOT the entire stack. Use `config_applicator.restart_role(role_name)` to minimize downtime.
5. **Instance counts and mlock tiers** are already optimized for the current host. Read live hot/warm status from `orchestration/derived/stack_priors.yaml` and the generated system card; do not explore tier demotions unless a current evidence handoff or human operator explicitly opens that surface.

### Git Safety Protocol (MANDATORY for all changes)

Every modification follows this protocol:
1. **Commit before eval**: `git add <changed files> && git commit -m "trial {N}: {description}"`
2. **Run eval**: T0 evaluation on the change
3. **Keep or revert**: If quality improves → keep. If regresses → `git revert HEAD` immediately.
4. **Checkpoint**: Every 10 trials, tag the current best: `git tag autopilot/best-{trial_id}`
5. **Rollback capability**: Any previous state is recoverable via `git log` and `git checkout`

---

## What You CANNOT Modify (Eval Trust Boundary)

These are the ONLY immutable files — changing them invalidates all experiment results:

- **Evaluation methodology**: `scripts/benchmark/seed_specialist_routing.py`, `debug_scorer.py`, `dataset_adapters.py`, `question_pool.py`
- **Question pool**: `benchmarks/prompts/question_pool.jsonl` (frozen at build time, additions are manual/human-only)
- **Safety gates**: `scripts/autopilot/safety_gate.py` (quality floor, regression guards)
- **Scoring contracts**: 7 scoring methods, per-suite definitions
- **Eval tower**: `scripts/autopilot/eval_tower.py` (measurement instrument)
- **This file** (`program.md`) — only humans edit this

```
CAN MODIFY (with git versioning)     │  CANNOT MODIFY (eval trust boundary)
──────────────────────────────────────┼─────────────────────────────────────────
orchestration/prompts/*.md            │  scripts/benchmark/seed_specialist_routing.py
orchestration/*.yaml (all config)     │  scripts/benchmark/debug_scorer.py
src/**/*.py (all orchestrator code)   │  scripts/benchmark/dataset_adapters.py
scripts/autopilot/species/*.py        │  scripts/benchmark/question_pool.py
scripts/server/orchestrator_stack.py  │  benchmarks/prompts/question_pool.jsonl
                                      │  scripts/autopilot/safety_gate.py
                                      │  scripts/autopilot/eval_tower.py
                                      │  scripts/autopilot/program.md
```

The eval trust boundary ensures that improvements are real: autopilot can change anything about HOW the system works, but not how it's MEASURED.

---

## Goal Metric

**Primary**: Pass rate across all active suites (deterministic scoring, no LLM judge)

```
metric = correct_answers / total_questions
```

Evaluated by sampling uniformly across all active suites (equal questions per suite) to ensure representative coverage. Sample size per trial is configurable via `--sample-size` (default: 15 per suite for T0, full pool for T2).

**Secondary (tracked, used for Pareto optimization)**:
- **Current speed objective**: EvalTower `speed` remains the active Pareto speed axis. In concurrent evals this is aggregate batch t/s; `median_request_tps`, `aggregate_tps`, `eval_concurrency`, and `eval_wall_s` are retained so host throttling and fan-out effects are diagnosable.
- **Task-rate shadow policy**: every eval row also records `task_rate_qph`, `goodput_qph`, and `tokens_per_solved_task`. `task_rate_qph` is questions per eval-wall-hour; `goodput_qph` is quality-scaled solved-task rate; `tokens_per_solved_task` is the planner-visible bloat signal for verbosity/compression work.
- **Escalation and tier telemetry**: escalation rate, route mix, and `cost_tier` remain routing/capacity diagnostics. The current instrument does not compute the old wall-occupancy proxy `sum(tokens_generated[role] / throughput_tps[role])`; do not reason as if that proxy is a live Pareto objective.

**Promotion gate (T2)**: Full evaluation + Claude-as-Judge scoring. Only for experiments that improve primary metric by >=0.5% and hold for 3 consecutive T0 runs.

**Eval-tier exploration — explore T1 AND T2 freely**: You may choose the eval tier per trial. T1 and T2 are first-class, **independently-tracked** frontiers — each keeps its OWN Pareto frontier, hypervolume, and quality baseline, and quality is **never** compared across tiers. Normal experiment trials grade at T1 via the hybrid eval (T0 fast-reject → T1, 50q). To validate on the full benchmark (gpqa / livecodebench / usaco / math / olympiad-grade — ~500q sampled across all 15 suites), emit a deep-eval action: `{"type": "deep_eval", "tier": 2}`. This is a **LOW-RISK action and may be chosen on any trial** — do not reserve it only for promotions. T2 is harder, so its absolute quality runs LOWER than T1; that gap is the honest signal, **not** a regression — never treat a low T2 number as a failure of a T1-good config. Use T2 to confirm T1 wins generalize and to populate the T2 frontier. Aim to interleave periodic `deep_eval` trials so both frontiers stay populated.

**Tool use — optimize MARGINAL USEFULNESS, never "more tools"**: Each trial records, in `eval_details`, tool telemetry: `tool_use_rate`, `mean_tools_used`, `total_tool_calls`, `tool_name_counts`, and the decision-grade signal **`tool_helpfulness` = P(correct | tools used) − P(correct | no tools)** (NaN until each arm has ≥3 questions). **Do NOT treat `tool_use_rate` as a goal** — raising it without raising quality/reliability is a vanity metric, not a win. Use `tool_helpfulness` as a **planner prior**: where it is meaningfully positive for a suite/mode, prefer mutations (prompt / policy / tool-registry) that enable tools there; where it is ≤0, reduce tool pressure or change tool policy. Score every such experiment **only by downstream quality/reliability movement** — tool metrics stay telemetry, never a Pareto objective. (Accounting note: the speed objective counts MODEL-decoded tokens only, exactly once; tool *output* tokens are never counted as throughput. So tool use is neither rewarded nor penalized as "free" or "wasted" tokens — it is judged purely by whether it makes answers correct.)

---

## Logging Format

Append to `orchestration/autopilot_journal.tsv`:

```
trial_id	timestamp	species	metric	status	description
```

- `trial_id`: monotonic integer
- `timestamp`: ISO 8601
- `species`: `numeric` | `prompt` | `structural` | `stack` | `seed`
- `metric`: debug suite pass rate (0.0-1.0)
- `status`: `keep` | `discard` | `crash`
- `description`: one-line summary of what changed

Also append full detail to `orchestration/autopilot_journal.jsonl` with config snapshot, per-suite breakdown, hypothesis, mechanism, and active feature flags.

---

## The Experiment Loop

```
REPEAT FOREVER:
  1. READ current best metric from autopilot_state.json
  2. READ recent failures (last 20 discards) — do NOT retry similar experiments
  3. HYPOTHESIZE: propose ONE change (one variable, one file)
     - State your hypothesis: "Changing X should improve Y because Z"
     - State expected mechanism: "This works because..."
  4. COMMIT the change: git add <changed files> && git commit -m "trial {N}: {description}"
  5. RUN evaluation:
     Sample uniformly from all active suites (equal per-suite representation).
     Use --sample-size to control questions per suite (default 15 for T0).
  6. RECORD result in journal (TSV + JSONL)
  7. DECIDE:
     - IF metric > current_best:
         Update autopilot_state.json with new best
         Log "keep" in journal
         Continue to next experiment
     - IF metric <= current_best:
         git reset HEAD~1  (revert the trial commit)
         Log "discard" in journal with failure analysis
         Continue to next experiment
     - IF crash or timeout:
         git reset HEAD~1
         Log "crash" in journal
         Continue to next experiment
  8. EVERY 10 trials: commit autopilot_state.json and journal updates
  9. NEVER STOP — continue proposing experiments indefinitely
```

---

## Experiment Priorities

Start with highest-expected-impact, lowest-risk experiments. Suggested order (agent may deviate based on findings):

### Tier 1: Prompt Optimization (fast, hot-swap, high signal)
- Frontdoor prompt engineering — conciseness, instruction clarity
- General model prompt efficiency — minimize tokens generated, maximize information-per-token
- TOON compression for escalation AND consultation context — push compression ratios further
- Tool-use instruction formatting — structured vs natural language
- Architect consultation pathway: architect returns compact execution guidance → redelegated to a live execution role from the generated system card (this is consultation, not terminal escalation — explore it)

### Tier 2: Feature Flag Combinations (hot-swap, zero-restart, direct logic changes)
- Skillbank on/off
- BSV-2 accept gate (`autopilot_bsv2_accept_gate`, default off) and its rigor extension (`autopilot_bsv2_accept_rigor`, default off, needs the gate on): accept-path controls, not quality levers. Rigor enforces R1 disjoint accept tasks, R2 fresh parent re-run, R3 sealed test split, R4 per-round decision record (status split, tokens per accepted gain), R5 budgets in rollouts or tokens; it fails closed when the proposal-qid set is unknown. Judge by accepts surviving R1-R5 (replay baseline: 0 of 130), never by score. Do not flip during a search; an operator arms these at a window boundary. **Pinned observe-only (AP-BSV2-RIGOR-PIN):** both are in `PINNED_OBSERVE_ONLY_FLAGS` and absent from `HOT_SWAP_FEATURES`, so the autopilot never proposes or applies them (only the operator, by hand). With both on, the loop populates `state["bsv2_rigor"]` each action (proposal batch = qids of the previous eval, sealed split = hash split of the pool at `AUTOPILOT_BSV2_SEALED_FRACTION` default 0.2, budget from `AUTOPILOT_BSV2_BUDGET_UNIT` default rollouts / `AUTOPILOT_BSV2_BUDGET_LIMIT` default 5000) and the T1 draw excludes proposal and sealed qids (core_id suffix `_bsv2part`), so baseline and candidate arms share one proposal-disjoint accept partition.
- Session log on/off
- Graph router vs classifier-only routing
- Episodic memory configurations (retrieval weights, confidence thresholds)
- Factual risk mode (off/shadow/enforce) — RI-7 A/B test was underpowered, needs larger sample
- REPL contained child lane (`repl_child_lane`, default off) — runtime-toggleable via `POST /config`; runs REPL cells in a warm, killable, resource-limited child (fails closed, never inline). Safety/latency axis, not a quality lever: judge it by REPL cell error and timeout rate and per-cell overhead against the flag-off baseline. The lane brokers the full REPL tool surface from free-form and structured cells alike (every tool runs in the parent through the same budget, write-dedupe, validation and deadline path as in-process), so tool-using arms behave the same either way

### Tier 2b: Codemode / Tool-Picker / REPL / Document Flags (hot-swap via `POST /config`; added 2026-11-18)

These were env-only knobs. Each is now a registered feature flag (`src/features.py`, **original env name and default unchanged**), so it is on the StructuralLab flag-trial surface (`structural_experiment` with `flags`), shows in `/config/attest`, and toggles live with no restart: the code reads it through `src/flag_env.py`, which lets the shared runtime-flag file override `os.environ`, and every worker picks the change up within about a second. A flag nobody toggles reads exactly what the environment says (byte-identical). Integer knobs next to them are Tier 3 (`codemode.*`, `tool_selector.*`, `pdf.*`), applied through `env_restart` because `/config` is bool-only.

Single-variable rule (AP-9): one flag per trial. The REPL flags below apply to the **next REPL a request builds**, so a flip never changes a request already in flight. Always trial paired on task x seed against the flag-off arm, and judge on the end-state quality score first, tokens second.

| Flag | Values | Hypothesis | Metric to watch | Guard |
|---|---|---|---|---|
| `tool_selector` (needs `repl_embedding_pool` first, own trial) | off / on (variant `embed`; env `oneshot` survives a flip) | Narrowing a long tool list to the per-request top-k raises tool-task success and cuts prompt tokens (CODEMODE-AB v1, 35B: top-8 selection took success 0.10 to 0.38; direct native calls beat REPL code mode, 0.43 vs 0.10; the v2 probe in `scripts/benchmark/codemode_ab/PREDECLARED_v2.md` has NOT run) | Task score; prompt tokens on paired successes; **proxy recall** = `proxy_recall` on the `outcome` rows of `logs/tool_selector_events.jsonl` (1 - called_not_selected/called); fail-open rate (`rung` timeout/breaker_open/error) | **Recall alarm:** if rolling `proxy_recall` < 0.90 over >= 30 outcome rows, or the fail-open rate > 20 percent, the next trial reverts `tool_selector` to off. Threshold is seeded from the 0.95 offline-recall precondition in PREDECLARED_v2; revisit once live rows exist. No effect when the list is already <= k (do not read a null there as a loss). The picker needs the pooled embedder: with `repl_embedding_pool` off it silently serves the full list. |
| `tool_selector_shadow` (needs `tool_selector`) | off / on | The C2 one-shot decision-model choice, logged beside the applied pick, shows whether the embedding picker misses tools a model would pick. Observe only: never applied | `shadow` vs `applied` rows in the same events log | Costs one `worker_general` call per selection on a background thread: trial only inside an eval window, never as a standing setting. |
| `repl_explicit_final` | on (production default) / off (legacy) | One-line tool calls such as `list_dir(...)` must not be auto-wrapped as `FINAL(...)` (v1 failure analysis: premature FINAL was 9 of 14 discordant pairs on the 35B) | Premature-FINAL rate (episodes ending on turn 1 with a tool call as the answer); task score | Off restores the legacy wrap: a regression reference arm only, never a standing setting. |
| `repl_explicit_final_strict` | off (default) / on | Wrapping only an obvious answer VALUE removes the remaining false FINALs at the cost of missed implicit finals | Premature-FINAL rate against the no-FINAL (turn-cap) rate; task score | Strict wins over `repl_explicit_final` off. Watch turn-cap exits: a model that never writes `FINAL(` now runs to the cap. |
| `repl_syntax_loop_breaker` | off (default) / on | Ending a task after 3 consecutive SyntaxError turns (hint at 2) stops a model that cannot emit valid Python from burning the turn budget and escalating | Turns per failed task; harness:syntax_loop end rate; task score | A true capability gap ends sooner: watch score on tasks the model used to recover on turn 4+. |
| `repl_plan_narration_nudge` | off (default) / on | A prose-only turn that only narrates a plan ("I will", "Let me", "Plan:") is nudged to emit a python block instead of being rescued into a bogus FINAL after 1 step | Steps per task; premature-FINAL rate; task score | Nudges count toward MAX_CONSECUTIVE_NUDGES (3), then promote to an error; a real prose answer ("The answer is B") is still rescued. |
| `harness_failure_ratio_guard` | off (default) / on | When REPL+tool failures / turns >= 0.5 over >= 4 turns the harness is the likely fault, so escalating the model tier wastes the bigger model | Escalation count; artifacts `harness_suspect`; task score on escalated tasks | Never escalates a task flagged harness_suspect: a genuinely too-weak model on a hard task is not rescued by a tier. |
| `prompt_stable_prefix` | off (default) / on | On the hybrid 35B-A3B llama-server cannot cache-reuse and only rolls back to checkpoints, so bytes that change early in the REPL prompt discard the KV cache. On: system/tools/rules/task/instruction first, then selected tools (sorted), previous turns, then state/last output/turn counter | Prefix-cache hit rate (cache_n / prompt_n); prompt-eval ms per turn; task score | Changes section ORDER for the model (history before state); watch score. Independent of `prefix_stable_order` and `repl_turn_counter_tail` (it does its own ordering). |
| `history_append_only` (needs `repl_turn_memory`) | off (default) / on | Turn memory keeps every turn until a char budget, then drops the oldest once to half the budget, instead of sliding the N-turn window each turn (a slide rewrites everything after the heading) | Prefix-cache hit rate; prompt tokens per turn (grows beyond N turns) | Prompt can hold more than N turns; one full history miss per truncation. Also defers the /v1 B2 fold below a hard size. No effect when turn memory is 0. |
| `tool_output_bound` (pairs with `history_append_only`) | off (default) / on | Turn-memory output/error is cut to 2000 chars at store time (head+tail, fixed marker) so an append-only history stays small | Prompt tokens per turn; repeated-refusal rate; task score | A long output loses its middle: the model must peek()/spill for the rest. |
| `stream_degeneration_guard` | off (default) / on | ~1% of Qwen3.6-35B-A3B requests degenerate into repetition and run to the 4096 cap at near-zero MTP acceptance, burning decode time and drafted-token statistics. On: the streaming chat path asks for per-token timings and closes the stream when rolling MTP acceptance < 0.2 over >= 512 tokens or the word 6-gram repeat ratio >= 0.6 over the last 4000 chars; the call returns a retryable request_error (infra, no tier escalation) | Abort count by reason/role (`src.metrics.stream_degeneration`); capped-run (length) rate; MTP acceptance; task score | A legitimately repetitive long output (tables, lists, code) can trip the repeat detector: watch score on long structured outputs. Streaming chat path only. |
| `tool_call_parse_error_retry` | off (default) / on | Qwen3.6-35B-A3B sometimes emits malformed nested tool-call XML. Fixed at the source by the strict Qwen parser (box PFXC-TOOLPARSE), which answers HTTP 500 `{"error":{"type":"tool_call_parse_error","retryable":true,"raw_excerpt":...}}` (or one SSE error event, no `[DONE]`). On: that error type, or a native tool parser loss from a current server, re-prompts once in the same session with a short correction; a second failure is a model format error (no tier escalation, no model fallback; /v1 clients get an error). Keyed on `error.type`, never the status: any other 5xx stays on the infra path | Retry count by role/outcome `retried_ok`\|`failed` (`src.metrics.tool_call_parse_retry`, dashboard snapshot `tool_call_parse_retry`); tool-call error rate; task score | One extra decode per parse failure. A truncated tool-call opener (length cap) is not retried. Until the strict parser ships only the native-parser-loss trigger fires. |
| `tool_call_arg_leak_guard` | off (default) / on | Belt and braces beside `tool_call_parse_error_retry` (independent flag, same retry seam: with both on a call is re-prompted at most once). A leaked-markup argument still parses as valid JSON (`</parameter></function><parameter=path>` copied into a value). On: parsed native tool-call args are scanned (recursing into dicts/lists) for `<parameter=`, `<function=`, `<tool_call>`, `</tool_call>`, `</parameter>`, `</function>`; a hit re-prompts once with a short correction, a second hit fails as a model format error (no tier escalation, no model fallback; /v1 clients get an error, never leaked markup) | Retry count by role/trigger `arg_leak`/outcome `retried_ok`\|`failed` (`src.metrics.tool_call_parse_retry`); tool-call error rate; task score | One extra decode per leaked call. Args that legitimately quote these markers (e.g. editing a tool-call template file) are rejected: watch the failed count on such tasks. Plain `a < b`, HTML and code pass. |
| `repl_turn_memory` | off / on (N = 4 on a flip; N in 1..8 is the Tier 3 knob `codemode.turn_memory_n`) | Showing the last N turns of code + output/error stops the exploration loop that repeats a refused program (PREDECLARED_v2 Q2: about +0.2 on the 27B, not measured) | Repeated-refusal rate; task score; prompt tokens (grows with N) | Prompt grows by up to 4000 chars per remembered turn: watch the context-budget guard before raising N. |
| `codemode_prompt_v2` | off / on | The v2 root prompt (sandbox contract, no STOP-after-CALL, no LeetCode few-shot) recovers code mode on task_root-scoped tool requests | Task score on task_root requests; `ast_refused` turn rate | Applies ONLY to task_root-scoped tool requests (`PromptConfig(codemode=True)` in `graph/helpers.py`); QA/coding prompts are untouched byte for byte, so a QA-only eval cannot see this flag. |
| `codemode_stubs` + `structured_read_only` | off / on (trial together: stubs depend on the CM-3 envelope) | Typed `def` stubs for the picked tools, returning Python objects, cut parse errors and tokens | Task score; `tool_arg_error` and `name_error` turn rates | `structured_read_only` changes read-only tool results to an envelope on EVERY tool call, not just code mode: confirm consumers tolerate it before a long trial. |
| `repl_tool_search`, `repl_classify`, `repl_handles` | off / on, one per trial | CM-2 search/describe cuts prompt tokens at equal accuracy; CM-4 `classify` labels N items for fewer tokens than in-prompt labelling; CM-5 handles beat `peek` re-reads on large intermediates | Prompt tokens at equal accuracy (CM-2); labelling accuracy vs gold and tokens (CM-4, N in 20/100/500); tokens and success on large-intermediate tasks (CM-5) | **S4 gate:** "no new REPL tools before S4" (`handoffs/active/harness-program.md`). Propose these only after the operator records an exception or S4 closes. Per-session `classify` cap is `ORCHESTRATOR_REPL_CLASSIFY_BUDGET` (2000 items). `repl_handles` also turns replay recording on. |
| `repl_write_dedupe` | on (default) / off | An identical write that already succeeded in this request is skipped, so a regenerated snippet cannot duplicate a `log_append` line or a patch | `duplicate_writes` per episode; task score | Default-on correctness guard: trial the OFF arm only to measure what it prevents, in a scratch task_root. |
| `pdf_route_lighton` | on (live stack; code default off) / off | `needs_ocr` PDFs through LightOnOCR-2-1B and the rest through LiteParse beats the document_formalizer sidecar (held-out PASS row `pip04-holdout-lighton-ocr-20261116-r2`) | PDF extraction quality on the PDF evals; per-document latency | Operator-approved on 2026-11-17 (STACKCHG-OCR-LIGHTON-20261117); the stack launcher sets it. Needs the `ocr_pdf_lighton` aux server on :9003: off is the instant rollback. |
| `l3_textrow` | on (live stack; code default off) / off | Text-row extraction recovers statement rows on pages with no parser table | L3 fact recall on the AGG corpus; table count per document | Operator-approved on 2026-11-17. Evidence so far is a one-slice proof (24 canonical facts on 4 pages), not a corpus measurement: a flip off is cheap, measure before reading a lift. |
| `episodic_routing_lookups` | on (default) / off | Episodic (MemRL) lookups pick the initial role, the execution mode and the Q-value gates; turning them off tests whether memory helps at all (M-12k: it has never been A/B-tested). Writes are unchanged, so the store keeps learning while off | Task score, latency and tokens vs the on arm on the same prompts; role-mix shift; escalation rate | Rules routing is the fallback, so a drop means memory was helping. Do not leave it off in production while a measurement is running: use the per-request `episodic_lookups=false` field to interleave arms. |
| `episodic_source_label` | off (default) / on | Stamps `request_source` (interactive/autopilot/benchmark/repl_experiment) on episodic task-start journal entries and MemRL rows (`extra`, not embedded), so recall composition can be split by source (M-27d.1). Write side only | Share of new rows carrying a label; top-k source composition of fixed recall queries (M-27 probe); NO task-score effect expected | Pure instrumentation: leave it on once enabled, rows written while off stay unlabelled forever. No retrieval filter exists yet, so it cannot change routing. |
| `typed_decisions_validate_first` | off (default) / on | In the TD-4/TD-29 tool-argument SHADOW, validate the original kwargs against the lossless schema first: valid calls cost zero typed repair calls, a violation the closed-set projection cannot express is refused (never dropped), only repair-eligible calls get one typed proposal (TD-29.M2). Needs `typed_decisions_tool_args` on; shadow only, never changes executed arguments | Share of shadow receipts with `selector_reason=original_valid` / `refused_unsupported:*` / a proposal; typed selector calls and tokens per tool call (should fall to the failing share); executed-argument behaviour must be unchanged | No acceptance or enforcement here: that needs an outcome-labelled policy per role/catalogue (TD-29.M2 box). Off is the instant rollback. |
| `monty_repl` | off (default) / on | Routing in-subset REPL cells to a confined Monty worker (Landlock+seccomp, memory/process caps, killable) buys containment at equal task quality and a small latency cost; out-of-subset cells and Monty runtime gaps fall back to CPython by design, so success must not drop | (1) task score vs the off arm (must be equal: any drop is a lane bug, stop and file it); (2) fallback rate = rows with `fallback` set over rows with `screen == "in_subset"` in `logs/monty_lane.jsonl` (those cells run twice); (3) Monty route share and per-cell `total_ms` p50/p95 by route; (4) `lane_unavailable` events | Needs the `monty` extra in the API venv: without it the flag is a logged no-op (fails open). Defers to `ORCHESTRATOR_REPL_CHILD_LANE` when that is on (screen reason `parent:child_lane`). A cell Monty cannot finish within half the REPL deadline is re-run on CPython, so a non-terminating cell costs 1.5x the deadline. |
| `autopilot_lh_layer_taxonomy` | off (default) / on | AP-LH-DISTILL (Life-Harness distillation, HIL-2 item 24): telling the mutation proposer WHICH harness layer the failing traces implicate (`action_realization`, `environment_contract`, `trajectory_regulation`, `procedural_skill`, plus the legacy kinds `prompt`/`routing`/`flag`) and the EXISTING surfaces each layer may edit steers proposals at the failing layer instead of the nearest prompt file. Annotation only: the layer never gates, ranks or rejects a proposal | (1) share of mutation/structural trials whose `eval_details.details.harness_layer.aligned` is true (proposal layer = implicated layer); (2) per-layer keep rate with n per layer, descriptive only (item 24: no layer ranking is assumed); (3) `declared_invalid` count | Meta-flag on the autopilot's own proposer, read live through `src/flag_env.py` in the autopilot process. Needs the contrastive trace bank (`contrastive_trace_bank` in state); with no failing trace there is no diagnosis block and `aligned` is null. Not a quality lever: do not trial it against the task score. |
| `autopilot_lh_trace_replay_precheck` | off (default) / on | AP-LH-DISTILL (HIL-2 item 24 trace replay): replaying a flag proposal with no model call against the failing trace(s) that motivated it, and skipping one whose guard provably would not have fired on any of them, saves the eval a null result would cost | (1) `trace_replay_precheck` statuses over `structural_experiment` trials (`would_fire` / `would_not_fire` / `not_replayable`); (2) journaled `skipped` rows with reason `trace_replay_precheck`; (3) of the proposals that DID run, keep rate by status (a `would_fire` keep rate no better than `not_replayable` means the predicates add nothing) | Only `structural_experiment` enabling ONE flag with a firing predicate is replayable (`repl_syntax_loop_breaker`, `repl_plan_narration_nudge`, `harness_failure_ratio_guard`, `tool_call_parse_error_retry`, `tool_call_arg_leak_guard`, `stream_degeneration_guard`); prompt and code edits record `not_replayable` and proceed. Predicates over-approximate the real guard, and a truncated trace is inconclusive, so a rejection is a proof, never a guess. Skips are `skipped` outcomes (not blacklisted). |

**Pinned, observe only (security): `exec_containment`, `untrusted_tool_text`.** The autopilot never proposes, enables or disables these; `StructuralLab.propose_flag_experiment` refuses any proposal that names them (`PINNED_OBSERVE_ONLY_FLAGS`) and they are absent from `HOT_SWAP_FEATURES`. Observe them in `/config/attest`: a trial result is invalid if either reads off. `exec_containment` is Landlock + seccomp for model-exec children (`ORCHESTRATOR_EXEC_CONTAINMENT=0` is the operator's kill switch); `untrusted_tool_text` labels, defangs and caps untrusted tool text before it re-enters model context (`ORCHESTRATOR_UNTRUSTED_TOOL_TEXT=0`). A score gain bought by switching either off is a security regression, not an optimisation.

### Tier 3: Routing Thresholds (medium risk, medium signal)
- Cheap-first quality threshold tuning
- Think-harder ROI thresholds
- Escalation policy parameters (max_retries, max_escalations)
- MemRL retrieval weights
- Q-scorer `baseline_tps_by_role` alignment with actual measured throughput
- Codemode bounds (`codemode.*`, env_restart, `numeric_swarm` surface `codemode_bounds`): `codemode.max_tool_calls` int [100, 2000] (default 1000; `ORCHESTRATOR_REPL_MAX_TOOL_CALLS`, per-snippet runaway-loop bound, charged on every REPL helper) and `codemode.max_regenerations` int [2, 16] (default 8; `ORCHESTRATOR_REPL_MAX_REGENERATIONS`, task-wide cap on failed-and-regenerated REPL turns). Hypothesis: a tighter bound frees budget from doomed episodes without cutting real batch loops. Watch: share of episodes hitting the cap (a rising cap-hit rate with falling score means the bound is cutting real work), task score, wall time. Guard: the range excludes 0 (= unbounded / guard off) on purpose, and a batch loop of about 200 items x read+write must still fit under `max_tool_calls`. `codemode.turn_memory_n` int [1, 8] is meaningful only while `repl_turn_memory` is on (default N 4).
- Tool-picker knobs (`tool_selector.*`, env_restart, surface `tool_selector_picker`): `k` [4, 16] (default 8), `k_min` [2, 8] (4), `k_max` [6, 24] (12), `family_cap` [1, 6] (3). Meaningful only while `tool_selector` (and `repl_embedding_pool`) is on. Hypothesis: a smaller k saves prompt tokens until recall drops. Watch `proxy_recall` (see the Tier 2b recall alarm) alongside task score; the alarm outranks any token saving.
- PDF LightOn width (`pdf.lighton_concurrency`, env_restart, surface `pdf_lighton`): int [1, 4] (default 4). The `ocr_pdf_lighton` server runs `-np 4`, so a width above 4 only queues pages; the upper bound is the server's slot count. Watch per-document OCR latency and page-error rate.

### Tier 4: Model Selection (requires per-role restart, high impact, use sparingly)
- Per-role model or quantization candidates, only when a current evidence handoff or operator explicitly opens that live role for measurement
- Cascade depth (2-tier vs 3-tier with fast filter)
- **CONSTRAINT**: Only restart the specific role's server, not the full stack. Model selection and quantization have been extensively benchmarked — only explore alternatives with clear hypothesis from quality data.

### Tier 4.5: KV Compression (zero-restart, operational, tunable)

Expected Attention KV compression scores each KV cache entry by predicted future attention importance and evicts the lowest-scoring entries. This extends effective context without restart.

**When to use**: After evaluating long-context question batches (GPQA, multi-hop, agentic) or when the Slot Memory section shows >4000 tokens cached on a production port. Also use proactively to free memory before large batches.

**Endpoint**: `POST http://localhost:{port}/slots/{id}?action=compact`

**Tunable parameters** (autopilot search space):

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `keep_ratio` | float | [0.10, 0.90] | 0.50 | Fraction of KV entries to KEEP. Safe range: [0.50, 0.90]. Below 0.25 format degrades. |
| `scorer` | string | "expected_attention", "knorm" | "expected_attention" | Scoring algorithm. EA is superior; knorm is legacy fallback. |
| `keep_first` | int | [2, 16] | 4 | Sink tokens (never evicted). Protects system prompt. |
| `n_future` | int | [64, 1024] | 128 | Future positions for RoPE averaging. Higher = more stable scores, slower. |
| `use_covariance` | bool | true, false | true | Full EA (with query covariance) vs mean-only. True is more accurate, ~2x slower. |
| `layer_weights` | float[] | each in [0.1, 5.0] | uniform | Per-attention-layer importance weights. Length = number of attention layers. Higher weight = that layer's scores contribute more. **Key tuning surface**: learn per-role weight vectors. |

**Example requests**:
```bash
# Set PORT from the generated controller system card or Slot Memory panel.
PORT="<live port from generated controller system card or Slot Memory>"
curl -X POST "http://localhost:${PORT}/slots/0?action=compact" \
  -H "Content-Type: application/json" \
  -d '{"keep_ratio": 0.5, "scorer": "expected_attention", "keep_first": 4}'
```

**Quality data** (from a legacy frontdoor multi-ratio sweep):
- 90% keep: identical output to baseline
- 75% keep: identical output to baseline
- 50% keep: slight reword, semantically equivalent
- 25% keep: different phrasing, coherent
- 10% keep: format shift, quality cliff onset

**PPL gate** (legacy frontdoor at 50% eviction): ratio = 1.096 (<1.10 threshold).

**Target endpoints**: derive live primary endpoints from the generated stack-priors contract before issuing compaction requests. Do not copy endpoint tables from historical docs, and do not target retired roles or dead ports.

```bash
uv run python scripts/autopilot/gen_system_card.py --stdout
```

Use the generated **Live Stack Roles** table as the controller-facing endpoint
surface. If a code path needs structured values, use shared
`src.registry.stack_priors` helpers rather than adding local YAML parsing to
planner prompts.

**NumericSwarm integration**: Treat `keep_ratio` as a continuous parameter and `layer_weights` as a per-role vector. Evaluate quality post-compact via `tower.hybrid_eval()`. The 4D Pareto archive (quality × speed × -cost × reliability) captures the compression/quality tradeoff naturally — speed improves with compression (fewer KV entries = less memory bandwidth), quality may degrade.

**Constraints**: Only compact idle slots (state != "processing"). Compacting during inference will corrupt the generation. The Slot Memory section in your prompt shows slot state — only target slots marked "idle" with high token counts.

### Tier 5: Stack Topology (requires full restart, already well-optimized)
- Instance counts per role (currently optimized for NUMA 4-way)
- NUMA quarter assignments (currently optimal)
- Acceleration flag combinations (already tuned per-model from isolated benchmarks)
- **NOTE**: The active stack topology is compiled into `orchestration/derived/stack_priors.yaml` and summarized in the generated controller system card. Tier demotion is not an open exploration surface by default. Acceleration flags are the product of extensive isolated benchmarking — do not change without reading the benchmark data in `epyc-inference-research/data/`.
 - **Planner source of truth**: when a prompt, memory, or handoff mentions a role, port, context window, throughput, memory footprint, or tier, treat the generated controller system card as the authoritative live view. Only quote literal values if they match that generated card; otherwise assume the literal is stale and refresh from stack priors before optimizing on it.

### Tier 6: StructuralLab Memory Mutations (hot-swap, NIB2-41 / intake-414 Token Savior)

Two mutation primitives on the strategy store (`orchestration/repl_memory/strategy_store.py`). Both are zero-restart and operate over existing data only — no model or server changes.

- **`mdl_compress_strategies`**: cluster near-duplicate strategies by Jaccard (default threshold 0.60) and promote clusters with MDL compression ratio ≥ 0.20 into a `strategy_conventions` row. Representative = longest insight; member deltas are token-removed insights. Run cadence: after every 50 new strategies.
- **`staleness_invalidate_strategies`**: sha256-scan prompts / classifier config / model registry; when a referenced file's hash changes, bump the Bayesian failure counter (`β_fail += 1`) on each strategy that cites it. Below `α/(α+β_fail) = 0.40` the strategy is quarantined (omitted from default `retrieve()` results); between 0.40 and 0.60 it is flagged `suspected`. Cascade: a quarantined strategy cited in `routing_classifier_meta.json` marks that checkpoint stale. Run cadence: every trial boundary (fast, sha256s only).

Both methods are available via `StructuralLab.mdl_compress_strategies(...)` and `StructuralLab.staleness_invalidate_strategies(...)`. Dry-run supported (`dry_run=True`). See `handoffs/active/meta-harness-optimization.md` § NIB2-41 for the full rationale and schema additions.

---

## Constraints

1. **One variable per experiment**: Change exactly one thing. If you change model AND prompt, you can't attribute the result.
2. **Git versioning mandatory**: Every change committed before eval. Reverted immediately on regression. No uncommitted changes during eval.
3. **No eval gaming**: You cannot modify anything inside the eval trust boundary (scoring, question selection, safety gates, eval tower).
4. **Stack restart budget**: Experiments requiring server restart are expensive (~2-5 min). Batch stack-level changes; prefer hot-swap experiments when possible. When restarting, restart ONLY the affected role's server process.
5. **Simplicity criterion**: Reject improvements that add disproportionate complexity. A 0.1% improvement that doubles prompt length is not worth keeping.
6. **Safety gate compliance**: All experiments must pass the safety gate (quality floor >= 2.0/3.0, no single-suite regression > 0.1, no throughput regression > 20%).
7. **Respect benchmark data**: Model registry values (acceleration flags, quantization, NUMA layout, instance counts) are grounded in isolated benchmark results. Do not explore configurations already proven suboptimal. Read `Known Dead Ends` below.
8. **Code changes require hypothesis**: When modifying `src/`, state the bug or inefficiency being fixed and the expected improvement. "Try random changes and see what sticks" is not acceptable for code modifications.

---

## Validated Decisions (context for mutations)

These findings come from controlled experiments outside autopilot. Mutations that touch these areas should be informed by these results — not blindly undone.

| Area | Decision | Evidence | Date |
|------|----------|----------|------|
| Brevity word limits (Action 12) | **KEEP** static per-format limits in worker prompts | TALE dynamic budget eval: static OAA=-3.48 vs TALE OAA=-5.95. Static simpler + better. TALE matches baseline on math (95%) but hurts general (50%). | 2026-04-11 |
| Tool output compression | **KEEP ON** (default) | Controlled A/B (100q): +4pp REPL overall. Math +25pp, hotpotqa -25pp. Suite-dependent but net positive. | 2026-04-10 |
| Web search in REPL | **EXPLORE** — currently denied for math/coder/thinking/instruction via WS-3 | Omega audit showed 7/10 suites hurt, BUT was run with old "search first" prompt. With fixed prompt (compute-first priority), selective web search may help. WS-3 deny is belt-and-suspenders — the prompt fix alone might suffice. **Autopilot is encouraged to experiment with relaxing WS-3 deny for specific suites** (e.g. enable web for coder, measure impact). Toggle via `NO_WEB_TASK_TYPES` in `tool_policy.py`. | 2026-04-09 |
| TrimR (think block pruning) | Thinking helps GPQA ~6pp, irrelevant on GSM8K (151 tok avg thinking) | Eval on DeepSeek-R1-7B. Difficulty-adaptive: only prune on easy tasks. | 2026-04-09 |
| Context folding free-zone | **L3 (60% target → 82% actual)** is the sweet spot | Compaction sweep: faithfulness stable at 2.9/3 across L1-L4. Retention knee at L3→L4 (2.84→2.21). | 2026-04-11 |
| Summarizer tier | **30B-A3B minimum viable** (3.0/3.0 faith+retain) | 1.5B: 2.55/1.45. 32B: untested (spec decode bug, now fixed). 30B already perfect. | 2026-04-10 |

Entries marked **KEEP** should not be undone without strong counter-evidence. Entries marked **EXPLORE** are open for experimentation — the current setting is a reasonable default but not validated as optimal.

---

## Known Dead Ends (Do NOT Retry)

These have been empirically tested and found non-viable:

### Inference Acceleration (all exhausted for hybrid models)
- **Qwen3.5 hybrid self-acceleration**: ALL 6 approaches exhausted — MoE self-draft (0.50-0.72x), attention-only (0.51x), tree speculation (-53 to -66%), layer-exit (-44-51%), MTP-1 (0.56x). Root cause: 75% Delta Net recurrent layers don't benefit from batching. This is architectural, not tunable.
- **DFlash block diffusion**: C++ verified correct via HF comparison. NOT viable on Q4_K_M (27% per-token acceptance, 1.4% block). AR drafter wins at 36.5 t/s.
- **Lookup table acceleration on Qwen3.5 hybrids**: Disabled since 2026-03-19 (segfault after 1-3 prompts). Do not re-enable without llama.cpp fix. Lookup works on dense models (Coder-32B).
- **Speculation on hybrids**: ANY approach that batches multiple tokens for verification is fundamentally limited — recurrent layers process tokens sequentially regardless of batch size.

### Model Evaluation
- **Nemotron Mamba2**: 69% quality — insufficient for any production role. No deployment.
- **REAP-25B as standalone frontdoor**: Quality gap too large vs 35B-A3B. Only viable as fast-filter in cascade.

### Infrastructure
- **`tool_permissions` in legacy path**: No role has permissions defined. Cascading path (`cascading_tool_policy=True`) is the only viable path.
- **Stack-prior cost/throughput drift**: Q-scorer role costs and throughput priors are generated from `orchestration/derived/stack_priors.yaml`. If a planner prompt, memory, or handoff cites literal model speeds/costs that disagree with the generated system card, treat the literal as stale and run the stack-change pipeline checks before optimizing on it. The same rule applies to any role label, port, or memory class: system card first, literal text second, never the other way around.

---

## Out-of-Action-Space Research Items (Do NOT Propose)

Some open research items in the repo handoffs are tracked but cannot be executed by the action types listed above. Do NOT emit actions for them — they are gated on hardware, external publication, or harnesses outside the eval tower. If you see them in `journal_summary` or `insights_text`, acknowledge in `reasoning` and pick a different action.

| Handoff / item | Why out-of-scope for autopilot actions |
|---|---|
| `agent-world-env-synthesis.md` AW-7 (Endless Terminals released-artifact re-eval on TB-2.0) | TB-2.0 harness not part of eval tower; needs HF dataset + checkpoint pull (~tens of GB) plus a TB-2.0 runner — out-of-band human/external work. |
| `agent-world-env-synthesis.md` AW-8 (env-generation mirror with gemma4-26B-A4B as filter, ~50-100 wall-hr decode-only) | Not a `seed_batch`/`numeric_trial`/`code_mutation` shape; requires a background env-synth job runner, gated on user approval per `feedback_no_concurrent_inference`. |
| `agent-world-env-synthesis.md` AW-9 (PPO consumption of env corpus, ECHO training) | GPU-gated. DGX Spark not acquired (`project_dgx_spark_target`). |
| `gpu-acceleration-path.md` §ECHO 3-gate trigger watch (intake-571) | Pure monitoring; advertised `microsoft/echo-rl` repo is currently 404. Reproduction requires 8×B200 even if repo lands. |
| `internal-kb-rag.md` Mirage K-A/K-F/K-V patterns | Design references for K1–K7 work; lands when the relevant K-task is actively worked, not via autopilot mutation. |
| `hermes-outer-shell.md` Mirage HOS-S/HOS-R patterns | Design references for adapter shim + session replay; out of autopilot's mutation scope. |
| `routing-and-optimization-index.md` P21.A (DeepConf-offline sweep) — intake-603 | **DO NOT WIRE — A2 negative (2026-05-24).** Built + validated on live Qwen3.6: DeepConf's confidence-weighted vote ties plain majority (no accuracy gain) and the confidence signal is anti-correlated with correctness (top-1-confidence 1/4; gap −0.158). Adds N× generation + `n_probs` cost for zero benefit. There is NO DeepConf knob surface and there will not be one — never emit DeepConf trials. Reference impl lives on default-OFF branch `feat/p21a-deepconf` (not merged). |
| `routing-and-optimization-index.md` P21.B (method-selection axis) — intake-601 | **GATED, not yet actionable.** The "which test-time technique" axis (self_consistency first) is a structural build in a dedicated session. Once the axis + flags exist, StructuralLab/PromptForge may optimize the policy — but not before. |
| `optillm-test-time-techniques.md` CoT-decoding / DeepConf-online | Decoder/sampler work in the `epyc-llama` fork (k× BW multiplier; needs a manual speed bench). Outside autopilot's repo entirely — never propose. |

---

## PEAF — Prediction-Error-As-Feature (intake-571 spike, default ON)

**ON by default.** Disable for the next session by exporting `EPYC_AUTOPILOT_PEAF=0` before `python autopilot.py start` if you want a clean baseline A/B period or if controller-behavior drift is suspected.

**When enabled (default)**, the controller prompt is appended with a brief instruction asking you (the controller) to optionally emit a separate fenced block AFTER the `json:autopilot_actions` block:

```json:peaf_prediction
{"quality": 2.40, "speed": 48.0, "cost": 0.05, "reliability": 0.95}
```

This is an OPTIONAL forecast of the trial's four objectives in the same units the journal uses (`quality` in [0,3], `reliability` in [0,1], `speed` in t/s, `cost` per question). Omit the block if you cannot honestly estimate — do not fill with placeholders. The forecast does NOT affect the action's evaluation or the Pareto archive scoring; it is logged as `predicted_objectives` and the L1 distance to actuals is logged as `surprise_score` for offline correlation analysis.

**Cheap-kill criterion**: run `python autopilot.py peaf` periodically. If Pearson r² between surprise and (entry.quality − parent.quality) is < 0.10 over ≥200 predicted trials, the PEAF signal does not correlate with config-quality gradient and the spike is abandoned. If r² ≥ 0.10, surprise is a candidate to promote as a Pareto co-objective in a future PR (NOT autopilot's job — flag in `distill_knowledge` output).

**Why this exists**: ECHO (intake-571, "Terminal Agents Learn World Models for Free") shows that auxiliary environment-prediction loss on GRPO rollouts ~2× baseline performance on Terminal-Bench-2.0. ECHO itself is GPU-gated and not in autopilot's action space; PEAF tests the underlying "prediction error = understanding signal" intuition on EPYC's CPU-only stack at logging-only cost.

---

## Production Flow (optimize this end-to-end)

The full production request path is generated from the live stack and routing
contracts. Do not treat the diagram below as a source of literal role or port
truth; use the generated controller system card for current role names,
endpoints, tiers, and routing availability.

```
cheap/fast path from generated stack truth
  → quality gate from stack priors and routing policy
    → specialist route when policy requires it
    → OR architect consultation → compact plan → redelegate to a live execution role
```

Key insight: **optimize the FULL flow, not individual components.**
- REPL in production includes delegation to fast workers — isolated SELF:repl measurement underestimates its value
- Architect consultation ≠ terminal escalation — architects provide compressed high-info-per-token plans that get executed fast downstream
- The gap between single-role accuracy and oracle best-of-three is the optimization target

---

## Constrained-Creativity Exploration (2026-05-23)

The controller prompt now switches between two exploration fragments based on a
stagnation signal computed at assembly time. Default is the *lean* fragment:
enumerate 3–5 alternatives with one-line reject/accept reasons. When any of
these fire, the *rich* fragment activates:

- `hv_slope_10 < 1e-3` (Pareto frontier not advancing)
- trustworthy trial count `< 5` (low-signal regime)
- last 3 trials share the same `action_type` (exploit lock-in)

The rich fragment asks for N=5 candidates under truth-preserving constraints,
scores each on a 3-axis rubric (info_gain / coherence / usefulness), prefers
fusion of the top-2 when one action can encode the other, and requires the
chosen action's scores to be quoted (not regenerated) from the candidate
table. Tail-sampled under-used action types are passed as *inspiration*, not
as candidates to defend.

Every trial that runs through the controller emits a second fenced block:

```
```json:autopilot_rationale
{"falsifier": "...", "rubric_scores": {"info_gain": 4, "coherence": 5,
 "usefulness": 3, "synthesis_note": "..."}}
```
```

Soft contract — a missing/malformed block logs a warning but does not abort.
The `falsifier` and `rubric_scores` fields land on `JournalEntry`, and the
next planner pass surfaces still-open hypotheses (those with an explicit
falsifier that hasn't been resolved) in the rich-fragment context.

Knobs live at the top of `autopilot.py`: `CREATIVITY_N`, `TAIL_WINDOW`,
`TAIL_SEED_COUNT`, `STAGNATION_HV_EPS`, `STAGNATION_STREAK`.

---

## Cache Flushes (Operator-Initiated) — 2026-05-24

If you need to drop the kernel page cache (e.g., to re-baseline frontdoor
throughput after a multi-day uptime, or before running a controlled bench),
**always** go through the canonical wrapper rather than calling the bare flush
helper:

```
python scripts/autopilot/flush_cache_safely.py
```

The wrapper does three things the bare `sudo /usr/local/sbin/autopilot-flush-cache`
does NOT:

1. Pauses the autopilot via `state.json` (the 2026-05-24 loop fix actually
   honors this now — pre-fix, `autopilot.py pause` was a no-op on a running
   autopilot because state was cached in-memory).
2. NUMA-interleave-rewarms every active role GGUF serially after the flush.
   Without this re-warm, the first non-NUMA-aware re-read pins the model to
   ONE NUMA node and HALVES sustained t/s (per `feedback_drop_caches_numa_eviction`).
3. Restores the pre-flush paused state (if the operator had explicitly paused).

If a trial happens to complete during the flush window, its journal entry is
tagged `bug_corrupted_by = exogenous_cache_flush` via the new
`DeficiencyCategory.EXOGENOUS_CACHE_FLUSH`. The planner's trustworthiness
gate then excludes it from hypothesis chains so the suspect data doesn't
contaminate future decisions.

The safety_gate's in-process host-throttle remediation (`safety_gate.py:256-278`)
uses the same flush+rewarm path automatically when it detects sustained-load
slowdown.

## Cross-role Contention Matrix (2026-05-24)

The orchestrator's cross-role admission gate (`src/scheduling/contention_gate.py`)
queues requests when a new role's decode would catastrophically contend with
another role currently decoding. Source-of-truth: `orchestration/contention_matrix.yaml`.

### When to re-run the matrix

Re-run after ANY of:
- Adding/removing/renaming a role in `scripts/server/stack_numa.py` `NUMA_CONFIG`
- Changing a role's CPU pinning, thread count, or numactl_policy
- Swapping a role's model file or quantization (affects per-token BW demand)
- Upgrading the llama.cpp binary (kernel changes can shift BW characteristics)
- Reboot if it changed BIOS NPS mode

How:
```bash
python scripts/server/contention_matrix.py run
```

Validate without re-running:
```bash
python scripts/server/contention_matrix.py validate
# or
python scripts/validate/check_contention_matrix_fresh.py
```

Both report `MatrixStatus.OK | MISSING | STALE | INVALID`. The CI/pre-commit
script exits with code 2 on MISSING/STALE so a `NUMA_CONFIG` change without a
matrix refresh fails loud.

### Interpreting gate decisions

The gate emits decisions as `PairDecision` enum values. For a request to role X
with role Y currently decoding:

| Pair ratio | Foreground | Background |
|---|---|---|
| ≥ 1.0 | ALLOW (parallel beats sequential) | ALLOW |
| 0.85 ≤ r < 1.0 | ALLOW (mild loss tolerated) | QUEUE |
| < 0.85 | QUEUE (foreground may DEGRADED_ALLOW on SLO) | QUEUE |
| Unknown pair | ALLOW (loud warning) | QUEUE |

The 0.85 floor is `CONTENTION_RATIO_FLOOR` in `src/scheduling/contention.py`.

Background traffic (autopilot seeding, batch evals) always queues on
known-bad or unknown pairs — even just `worker_vision` decoding can hold an
autopilot probe back if the probe targets `frontdoor` (worker_vision's Q0B
overlaps frontdoor's NUMA_NODE0).

### When a request gets denied

If the gate times out (`max_queue_wait_ms` exceeded), the chat route returns
**HTTP 503** with a `Retry-After` header. Callers should back off and retry;
autopilot's seeding loop catches the timeout and skips/defers the probe.

### Dashboard signals

The orchestrator dashboard exposes a "CONTENTION GATE" panel at the top showing:
- `matrix_status` — OK/MISSING/STALE/INVALID with a colored badge
- `active_decodes_by_role` — live region-lock holders
- `contention_blocked_count` — per-pair admission denials since process start
- `contention_wait_seconds` — cumulative queue time
- `contention_timeout_count` — requests that exhausted their budget

A STALE matrix doesn't break the orchestrator; foreground requests fail-open
and background campaigns block (per Phase B of the cross-role-bw-aware-routing
handoff). The operator should still re-run the matrix promptly.

---

## Interaction with Autopilot Infrastructure

This program.md guides autonomous Claude sessions. The existing autopilot infrastructure (`scripts/autopilot/`) provides:

- **EvalTower**: Use for tiered evaluation (T0 quick check, T1 medium, T2 full)
- **SafetyGate**: Must pass before any "keep" decision
- **ParetoArchive**: Record all kept trials for multi-objective analysis
- **ExperimentJournal**: Dual TSV+JSONL logging (use both)
- **ConfigApplicator**: Routes parameter changes to hot-swap or restart

The autopilot species (Seeder, NumericSwarm, PromptForge, StructuralLab) can be invoked as experiment execution methods, but the hypothesis generation and experiment selection is driven by this program.

---

## When to Escalate to Human

Escalation reports appear in `logs/autopilot.log` and the TUI log panel (upper left). The autopilot pauses and waits for human intervention.

- Safety gate triggered 3 times consecutively → pause, log detailed report
- Metric degraded >5% from baseline with no clear cause → pause, log report
- Stack crash that doesn't resolve after revert → pause, log report
- Fundamental architecture question (should we add a new role?) → pause, propose in log
- Any change that would modify files outside the "What You Can Modify" section → STOP
- Model swap or acceleration flag change proposed → STOP, require human approval

---

## Success Criteria

The orchestrator optimization is never "done" — there's always another experiment to try. But key milestones:

1. **Baseline established**: Pass rate measured on current production config ✅ (AR-1: 57.3% direct)
2. **First improvement committed**: At least one experiment improves metric and is kept
3. **10 kept improvements**: Compound gains from multiple small improvements
4. **Feature flags explored**: All flag combinations tested with statistical significance
5. **Routing thresholds tuned**: Cheap-first, think-harder, escalation params optimized
6. **Pareto front populated**: 10+ non-dominated configurations in archive

---

## NEVER STOP

Continue proposing and running experiments. Each small improvement compounds. The git history is your ratchet — every commit is a checkpoint you can return to. When in doubt, try the simplest possible experiment next.
