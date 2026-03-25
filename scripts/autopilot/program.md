# AutoResearch Program — EPYC Orchestrator Optimization

> Strategy document for autonomous experimentation.
> The agent reads this file at session start and follows it indefinitely.
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

### Orchestrator Configuration (hot-swap safe)
- `orchestration/prompts/*.md` — prompt templates (read on every request, zero-downtime)
- `orchestration/model_registry.yaml` — role assignments, model parameters, timeout thresholds
- Feature flags via `POST http://localhost:8000/config` — runtime toggle, no restart
- Environment variables in `orchestrator_stack.py` — requires stack restart

### Routing Parameters
- `src/classifiers/` config files — risk thresholds, classification rules
- Q-scorer weights in `q_scorer.py` — `baseline_tps_by_role`, cost weights
- Think-harder thresholds — `min_expected_roi`, `cot_roi_threshold`
- MemRL retrieval params — `q_weight`, `min_similarity`, `confidence_threshold`

### Stack Configuration (requires restart)
- Model selection per role in `model_registry.yaml`
- Instance counts in `orchestrator_stack.py`
- NUMA topology (quarter/half-node/full-node assignments)
- Acceleration flags (draft_max, p_split, moe_experts, lookup_n)
- Tier assignments (mlock flags, startup order)

### TOON & Context Transfer
- TOON encoding parameters
- Escalation context format and content
- Prompt compression strategies for generals

### Specialist Pipelines
- Vision/OCR pipeline configuration
- Embedding model selection
- File extraction parameters

---

## What You CANNOT Modify

These are immutable — changing them invalidates all experiment results:

- **Evaluation methodology**: `scripts/benchmark/seed_specialist_routing.py`, `debug_scorer.py`, `dataset_adapters.py`, `question_pool.py`
- **Question pool**: `benchmarks/prompts/question_pool.jsonl` (frozen at build time)
- **Safety gates**: `scripts/autopilot/safety_gate.py` (quality floor, regression guards)
- **Scoring contracts**: 7 scoring methods, per-suite definitions
- **This file** (`program.md`) — only humans edit this
- **Core orchestrator logic**: `src/graph/`, `src/api/`, `src/pipeline_monitor/` (optimization targets the *configuration* of these systems, not their code)

---

## Goal Metric

**Primary**: Debug suite pass rate (deterministic scoring, no LLM judge)

```
metric = correct_answers / total_questions
```

Evaluated on the full debug suite (579 questions, 23 suites). Each experiment gets one number.

**Secondary (tracked but not optimized directly)**:
- Throughput: tokens/second aggregate across all roles
- Escalation rate: fraction of requests escalated beyond frontdoor
- Cost proxy: weighted token count by tier (architect=10x, frontdoor=3x, worker=1x)

**Promotion gate (T2)**: Full 6-suite evaluation + Claude-as-Judge scoring. Only for experiments that improve primary metric by >=0.5% and hold for 3 consecutive T0 runs.

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

Also append full detail to `orchestration/autopilot_journal.jsonl` with config snapshot, per-suite breakdown, hypothesis, and mechanism.

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
  5. RUN debug suite:
     python scripts/benchmark/seed_specialist_routing.py \
       --mode eval --suite debug --questions 579 --output /tmp/autoresearch_trial_{N}.json
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
- TOON compression for escalation context — push compression ratios further
- Tool-use instruction formatting — structured vs natural language

### Tier 2: Routing Thresholds (medium risk, medium signal)
- Cheap-first quality threshold tuning
- Think-harder ROI thresholds
- Escalation policy parameters (max_retries, max_escalations)
- MemRL retrieval weights

### Tier 3: Model Selection (requires restart, high impact)
- Frontdoor model candidates (35B-A3B, 30B-A3B, REAP-25B as fast-filter)
- Coder escalation model (32B Q4KM vs alternatives)
- Worker model selection and instance count
- Cascade depth (2-tier vs 3-tier with fast filter)

### Tier 4: Stack Topology (requires restart, high impact, slow iteration)
- Instance counts per role (1x vs 2x vs 4x)
- NUMA quarter assignments
- HOT vs WARM tier assignments for generals
- Acceleration flag combinations per model

### Tier 5: Feature Flag Combinations
- Skillbank on/off
- Session log on/off
- Graph router vs classifier-only routing
- Episodic memory configurations

---

## Constraints

1. **One variable per experiment**: Change exactly one thing. If you change model AND prompt, you can't attribute the result.
2. **Revert on failure**: Every discard must be cleanly reverted. The best-known config is always recoverable via git.
3. **No eval gaming**: You cannot modify scoring, question selection, or evaluation methodology.
4. **Stack restart budget**: Experiments requiring server restart are expensive (~2-5 min). Batch stack-level changes; prefer hot-swap experiments when possible.
5. **Simplicity criterion**: Reject improvements that add disproportionate complexity. A 0.1% improvement that doubles prompt length is not worth keeping.
6. **Safety gate compliance**: All experiments must pass the safety gate (quality floor >= 2.0/3.0, no single-suite regression > 0.1, no throughput regression > 20%).
7. **Production safety**: Never modify code in `src/` — only configuration, prompts, and parameters.

---

## Known Dead Ends (Do NOT Retry)

These have been empirically tested and found non-viable:

- **Qwen3.5 hybrid self-acceleration**: ALL approaches exhausted — MoE self-draft (0.50-0.72x), attention-only (0.51x), tree speculation (-53 to -66%), layer-exit (-44-51%), MTP (0.56x). Architectural constraint: 75% Delta Net recurrent layers.
- **Lookup table acceleration on Qwen3.5-35B-A3B**: Disabled since 2026-03-19 (segfault). Do not re-enable without llama.cpp fix.
- **`tool_permissions` in legacy path**: No role has permissions defined. Cascading path (`cascading_tool_policy=True`) is the only viable path.

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

- Safety gate triggered 3 times consecutively -> pause and report
- Metric degraded >5% from baseline with no clear cause -> pause and report
- Stack crash that doesn't resolve after revert -> pause and report
- Fundamental architecture question (should we add a new role?) -> pause and propose
- Any change that would modify files outside the "What You Can Modify" section -> STOP

---

## Success Criteria

The orchestrator optimization is never "done" — there's always another experiment to try. But key milestones:

1. **Baseline established**: Debug suite pass rate measured on current production config
2. **First improvement committed**: At least one experiment improves metric and is kept
3. **10 kept improvements**: Compound gains from multiple small improvements
4. **Stack topology explored**: At least 3 different instance-count configurations tested
5. **Model alternatives tested**: At least 2 different models tested per high-traffic role
6. **Pareto front populated**: 10+ non-dominated configurations in archive

---

## NEVER STOP

Continue proposing and running experiments. Each small improvement compounds. The git history is your ratchet — every commit is a checkpoint you can return to. When in doubt, try the simplest possible experiment next.
