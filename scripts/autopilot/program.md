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
- Architect consultation → TOON plan → frontdoor/cheap-first fast execution pathway (explore this — architects can provide compressed high-info plans that get redelegated back for fast execution, not just terminal escalation)

### Specialist Pipelines
- Vision/OCR pipeline configuration
- Embedding model selection
- File extraction parameters

### Guarded: Model Registry & Stack Config

**`orchestration/model_registry.yaml`** and **`scripts/server/orchestrator_stack.py`** are the product of months of isolated benchmarking (single-model throughput, NUMA-aware multi-instance, acceleration flags, quantization selection). Do NOT blindly explore these.

Rules for touching model registry or stack config:
1. **Never change model selection, quantization, or NUMA assignments** without explicit human approval
2. **Never change acceleration flags** — these are already optimized per-model from benchmark data (tree speculation NOT viable on hybrids, lookup disabled on Qwen3.5 due to segfault, REAP expert counts tuned per-model)
3. **Timeouts and token caps** in model_registry.yaml ARE safe to tune — these are routing parameters, not infrastructure
4. **If swapping a model for a role**: restart ONLY that role's server process, NOT the entire stack. Use `config_applicator.restart_role(role_name)` to minimize downtime.
5. **Instance counts and mlock tiers** are already optimized — entire stack fits in HOT tier with mlock. Do not explore WARM tier demotions.

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
- **Throughput**: tokens/second per role, weighted by request volume share
- **Escalation rate**: fraction of requests escalated beyond frontdoor (lower is better — escalation costs time and occupies specialist slots)
- **Cost proxy**: per-request cost estimated as `sum(tokens_generated[role] / throughput_tps[role])` across all roles touched. This measures wall-clock slot occupancy — a request that uses an architect for 30s at 5 t/s costs 6x more than a frontdoor request taking 5s at 12.7 t/s. Throughput values come from `autopilot_baseline.yaml` (measured, not assumed).

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
- Architect consultation pathway: architect returns TOON plan → redelegated to frontdoor/cheap-first for fast execution (this is consultation, not terminal escalation — explore it)

### Tier 2: Feature Flag Combinations (hot-swap, zero-restart, direct logic changes)
- Skillbank on/off
- Session log on/off
- Graph router vs classifier-only routing
- Episodic memory configurations (retrieval weights, confidence thresholds)
- Factual risk mode (off/shadow/enforce) — RI-7 A/B test was underpowered, needs larger sample

### Tier 3: Routing Thresholds (medium risk, medium signal)
- Cheap-first quality threshold tuning
- Think-harder ROI thresholds
- Escalation policy parameters (max_retries, max_escalations)
- MemRL retrieval weights
- Q-scorer `baseline_tps_by_role` alignment with actual measured throughput

### Tier 4: Model Selection (requires per-role restart, high impact, use sparingly)
- Frontdoor model candidates (35B-A3B, 30B-A3B)
- Coder escalation model variants (Q4KM vs Q8 vs f16 — different speed/quality tradeoffs)
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
# Standard 50% compression
curl -X POST "http://localhost:8070/slots/0?action=compact" \
  -H "Content-Type: application/json" \
  -d '{"keep_ratio": 0.5, "scorer": "expected_attention", "keep_first": 4}'

# Aggressive with deep-layer emphasis (for coder role)
curl -X POST "http://localhost:8071/slots/0?action=compact" \
  -H "Content-Type: application/json" \
  -d '{"keep_ratio": 0.3, "keep_first": 8, "layer_weights": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.5,1.5,1.5,2.0,2.0,2.0,2.5,3.0]}'
```

**Quality data** (from multi-ratio sweep on Qwen3.5-35B-A3B frontdoor):
- 90% keep: identical output to baseline
- 75% keep: identical output to baseline
- 50% keep: slight reword, semantically equivalent
- 25% keep: different phrasing, coherent
- 10% keep: format shift, quality cliff onset

**PPL gate** (Qwen3.5-35B-A3B at 50% eviction): ratio = 1.096 (<1.10 threshold).

**Target ports** (primary instances only — quarter instances share no state):
- frontdoor: 8070
- coder: 8071
- worker: 8072
- architect_general: 8083
- architect_coding: 8084

**NumericSwarm integration**: Treat `keep_ratio` as a continuous parameter and `layer_weights` as a per-role vector. Evaluate quality post-compact via `tower.hybrid_eval()`. The 4D Pareto archive (quality × speed × -cost × reliability) captures the compression/quality tradeoff naturally — speed improves with compression (fewer KV entries = less memory bandwidth), quality may degrade.

**Constraints**: Only compact idle slots (state != "processing"). Compacting during inference will corrupt the generation. The Slot Memory section in your prompt shows slot state — only target slots marked "idle" with high token counts.

### Tier 5: Stack Topology (requires full restart, already well-optimized)
- Instance counts per role (currently optimized for NUMA 4-way)
- NUMA quarter assignments (currently optimal)
- Acceleration flag combinations (already tuned per-model from isolated benchmarks)
- **NOTE**: Entire stack fits in HOT tier with mlock on 512GB RAM. WARM tier demotion is unnecessary and should not be explored. Acceleration flags are the product of extensive isolated benchmarking — do not change without reading the benchmark data in `epyc-inference-research/data/`.

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
- **Q-scorer frontdoor throughput**: Currently uses 19.6 t/s (moe6+lookup) but lookup is disabled. Actual is 12.7 t/s (moe6-only). This inflates frontdoor cost penalty ~1.5x. Needs correction.

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

The full production request path is:

```
try-cheap-first (Qwen3-Coder-30B-A3B, fastest)
  → frontdoor (Qwen3.5-35B-A3B, quality gate)
    → escalation to specialist (coder_escalation, math, etc.)
    → OR architect consultation → TOON plan → redelegate to frontdoor/specialist
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
