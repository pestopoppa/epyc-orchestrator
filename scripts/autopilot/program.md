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

### Tier 5: Stack Topology (requires full restart, already well-optimized)
- Instance counts per role (currently optimized for NUMA 4-way)
- NUMA quarter assignments (currently optimal)
- Acceleration flag combinations (already tuned per-model from isolated benchmarks)
- **NOTE**: Entire stack fits in HOT tier with mlock on 512GB RAM. WARM tier demotion is unnecessary and should not be explored. Acceleration flags are the product of extensive isolated benchmarking — do not change without reading the benchmark data in `epyc-inference-research/data/`.

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
| Web search in REPL | **DENY for math/coder/thinking/instruction** via WS-3 cascading policy | Omega audit: 7/10 suites REPL hurt accuracy. Root cause: model web-searched instead of reasoning. WS-3 denies web tools for these task types. | 2026-04-09 |
| TrimR (think block pruning) | Thinking helps GPQA ~6pp, irrelevant on GSM8K (151 tok avg thinking) | Eval on DeepSeek-R1-7B. Difficulty-adaptive: only prune on easy tasks. | 2026-04-09 |
| Context folding free-zone | **L3 (60% target → 82% actual)** is the sweet spot | Compaction sweep: faithfulness stable at 2.9/3 across L1-L4. Retention knee at L3→L4 (2.84→2.21). | 2026-04-11 |
| Summarizer tier | **30B-A3B minimum viable** (3.0/3.0 faith+retain) | 1.5B: 2.55/1.45. 32B: untested (spec decode bug, now fixed). 30B already perfect. | 2026-04-10 |

If PromptForge proposes removing word limits, compressing tool instructions, or changing REPL tool policy — check this table first. Mutations that conflict with validated decisions need a stronger hypothesis than "try something different."

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
