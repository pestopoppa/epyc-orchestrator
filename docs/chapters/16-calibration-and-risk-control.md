# Chapter 16: Calibration and Risk Control

## Introduction

This chapter documents calibration-aware routing controls added to the MemRL decision loop. The routing stack now supports robust confidence estimation from neighbor Q-values, calibrated confidence thresholds, conformal-style safety margins for abstain/escalate behavior, and replay-time calibration metrics for ongoing validation.

## Scope

The routing stack now supports:

1. Robust confidence estimation from neighbor Q-values.
2. Calibrated confidence thresholds.
3. Conformal-style safety margin for abstain/escalate behavior.
4. Replay-time calibration metrics for ongoing validation.

## Runtime Controls

These are the knobs you turn to tune how aggressively the risk gate intervenes. The `RetrievalConfig` object holds all of them, and the effective routing threshold is computed as the calibrated (or base) threshold plus the conformal margin. When confidence falls below that threshold under strict gate enforcement, hybrid routing emits `risk_abstain_escalate` and hands off to the configured target role.

<details>
<summary>RetrievalConfig parameters</summary>

`RetrievalConfig` controls:

- `confidence_threshold`
- `calibrated_confidence_threshold`
- `conformal_margin`
- `risk_control_enabled`
- `risk_budget_id`
- `risk_gate_min_samples`
- `risk_abstain_target_role`
- `risk_gate_rollout_ratio`
- `risk_gate_kill_switch`
- `risk_budget_guardrail_min_events`
- `risk_budget_guardrail_max_abstain_rate`
- `confidence_estimator` (`median` or `trimmed_mean`)
- `confidence_trim_ratio`
- `confidence_min_neighbors`

Effective routing threshold:

`effective_threshold = (calibrated or base threshold) + conformal_margin`

When confidence is below this threshold under strict gate enforcement, hybrid routing
emits `risk_abstain_escalate` and routes to `risk_abstain_target_role`.
Gate provenance is logged with:

- `risk_gate_action`
- `risk_gate_reason`
- `risk_budget_id`

</details>

<details>
<summary>Rollout and guardrail controls</summary>

- deterministic rollout sampling by route key (`risk_gate_rollout_ratio`)
- emergency kill switch (`risk_gate_kill_switch`)
- budget guardrail to auto-disable strict gating if abstain rate exceeds configured budget

</details>

## Metrics

Replay emits four calibration metrics that tell you how well-calibrated your confidence estimates actually are. These are computed across the replay engine and its metrics module.

<details>
<summary>Calibration metric definitions</summary>

Replay now emits:

- `ece_global`
- `brier_global`
- `conformal_coverage`
- `conformal_risk`

These are computed in:

- `orchestration/repl_memory/replay/engine.py`
- `orchestration/repl_memory/replay/metrics.py`

</details>

## Operational Workflow

When you want to validate a new calibration config, run replay twice — once with your baseline, once with the candidate settings — then compare metrics. Only promote if both the risk/coverage targets and your utility KPIs pass.

1. Run replay on recent trajectories with baseline config.
2. Run replay with candidate calibration/risk settings.
3. Compare quality/cost/calibration metrics.
4. Promote only if risk/coverage targets and utility KPIs pass.

## Budget Controls (Fast-RLM)

In addition to the confidence-based risk gate, two resource budget controls limit runaway task execution. These are inspired by the Fast-RLM paper's recursion and call-count limits, adapted as hard caps with pressure warnings.

| Budget | State Field | Default Cap | Env Variable | Pressure Warning |
|--------|-------------|-------------|--------------|------------------|
| Worker call budget | `state.repl_executions` | 30 | `ORCHESTRATOR_WORKER_CALL_BUDGET_CAP` | ≤3 remaining |
| Per-task token budget | `state.aggregate_tokens` | 200K | `ORCHESTRATOR_TASK_TOKEN_BUDGET_CAP` | <15% remaining |

Both budgets are checked **before** `_execute_turn()` in all 7 graph node types (`FrontdoorNode`, `CoderNode`, `WorkerNode`, etc.), saving a wasted LLM call when the budget is already exhausted. When exceeded, `_rescue_from_last_output()` attempts to extract a partial answer from prior output before falling through to hard FAIL.

Feature flags: `worker_call_budget` and `task_token_budget` (both production=True, test=False). Enabled in production via `orchestrator_stack.py`.

Pre-existing depth limits: `max_escalations=2`, `detect_role_cycle()`, `max_turns=15` — these remain as complementary safeguards.

## Related Modules

- `orchestration/repl_memory/retriever.py`
- `orchestration/repl_memory/replay/engine.py`
- `scripts/benchmark/seed_specialist_routing.py`
- `src/pipeline_monitor/claude_debugger.py`

## Concept-to-Code Mapping

This table ties each calibration idea back to the actual code that implements it, so you know where to look when debugging or extending the risk control pipeline.

<details>
<summary>Calibration concept mapping table</summary>

| Calibration/Risk Concept | Runtime/Replay Realization | Code Anchors |
|--------------------------|----------------------------|--------------|
| Calibrated confidence thresholding | Base/calibrated threshold with conformal margin | `orchestration/repl_memory/retriever.py` |
| Strict abstain/escalate gate | `risk_abstain_escalate` decision path with telemetry provenance | `orchestration/repl_memory/retriever.py`, `src/api/routes/chat_pipeline/routing.py` |
| Rollout + kill-switch safety | Deterministic rollout ratio and emergency off switch | `orchestration/repl_memory/retriever.py` |
| Budget guardrail | Auto-disable strict gate when abstain budget is violated | `orchestration/repl_memory/retriever.py` |
| Offline calibration validation | ECE/Brier/conformal metrics during replay candidate comparison | `orchestration/repl_memory/replay/engine.py`, `orchestration/repl_memory/replay/metrics.py` |
| Parameterized seeding/eval | Reproducible calibration/risk sweeps via CLI knobs | `scripts/benchmark/seed_specialist_routing.py`, `orchestration/repl_memory/replay/meta_agent.py` |

</details>

## Literature References (From Architecture Review)

<details>
<summary>References and further reading</summary>

Primary references for this chapter:

1. Xue et al. (2025). Conformal Risk-Controlled Routing for Large Language Model. https://openreview.net/forum?id=lLR61sHcS5
2. Tsiourvas, Sun, Perakis (2025). Causal LLM Routing: End-to-End Regret Minimization from Observational Data. https://openreview.net/forum?id=iZC5xoQQkX
3. Ong et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. https://arxiv.org/abs/2406.18665
4. Self-REF (2025). Self-Reflection with Error-based Feedback for confidence and routing improvements. https://icml.cc/virtual/2025/poster/45145
5. Lu et al. (2025). Token-Entropy Conformal Prediction for LLMs (TECP). https://www.mdpi.com/2227-7390/13/20/3351
6. CP-Router (2025). Uncertainty-aware routing between LLM/LRM tiers. https://www.themoonlight.io/en/review/cp-router-an-uncertainty-aware-router-between-llm-and-lrm

Secondary context:

7. Zheng et al. (2024). SGLang / RadixAttention for serving efficiency under long contexts. https://arxiv.org/abs/2312.07104
8. Dai, Yang, Si (2025). S-GRPO for regulated reasoning length. https://arxiv.org/abs/2505.07686

</details>

## Skill Effectiveness Scoring

When SkillBank is active (`ORCHESTRATOR_SKILLBANK=1`), skill retrieval outcomes feed into effectiveness tracking, which in turn influences confidence calibration.

<details>
<summary>OutcomeTracker and effectiveness lifecycle</summary>

### OutcomeTracker

`OutcomeTracker` (`orchestration/repl_memory/skill_evolution.py`) tracks per-skill retrieval outcomes with a rolling effectiveness score:

- Each time a skill is retrieved for a task, the retrieval is recorded
- After task completion, the outcome (success/failure/escalation) is correlated back to retrieved skills
- `effectiveness_score` is updated as a rolling average (initial default: 0.5)

### Lifecycle Thresholds

| Threshold | Value | Action |
|-----------|-------|--------|
| Promotion | effectiveness >= 0.8 | Confidence boosted by +0.1 |
| Stable | 0.3 <= effectiveness < 0.8 | No action |
| Deprecation | confidence < 0.3 | Skill marked `deprecated=True`, excluded from retrieval |
| Hard ceiling | confidence 0.95 | Prevents overconfidence |

### Interaction with Risk Control

Skill effectiveness data provides an additional signal for calibration:
- High-effectiveness skills increase routing confidence for covered task types
- Low-effectiveness or deprecated skills are excluded, preventing confidence inflation from bad memories
- The `min_confidence` parameter in `SkillRetrievalConfig` (default 0.3) acts as a quality gate aligned with the risk control's confidence thresholds

</details>

See [Chapter 15: SkillBank](15-skillbank-experience-distillation.md) for the full evolution mechanism.

## Input-Side Classifiers (`src/classifiers/`)

The conformal risk gate (above) operates on output-side uncertainty — it requires a routing decision and memory retrieval to produce confidence estimates. Two complementary input-side classifiers assess the prompt itself before any model call, using fast regex-only feature extraction.

### Factual Risk Scorer

`src/classifiers/factual_risk.py` (280 lines, 43 tests) scores prompts for hallucination risk based on input features: date/entity questions, citation requests, claim density, uncertainty markers, factual keyword ratio.

```python
@dataclass
class FactualRiskResult:
    risk_score: float           # [0, 1] raw prompt-based risk
    adjusted_risk_score: float  # [0, 1] adjusted for assigned role capability
    risk_band: str              # "low" | "medium" | "high"
    risk_features: dict         # for telemetry
    role_adjustment: float      # tier multiplier (0.6 for 235B, 1.0 for 7B)
```

Mode gate via `classifier_config.yaml`:
- `off`: no computation
- `shadow`: compute and log in `routing_meta`, no routing changes (current)
- `enforce`: wire into cheap-first bypass, plan review gate, escalation policy (future)

Telemetry fields on `RoutingResult`: `factual_risk_score`, `factual_risk_band`, `estimated_cost`. Logged in every `ROUTING_DECISION` event. The `estimated_cost` field (tier_weight × estimated_tokens / 1M) provides relative cost units for Pareto cost dimension tracking.

### Difficulty Signal Classifier

`src/classifiers/difficulty_signal.py` (~230 lines, 30 tests) classifies prompt difficulty using 7 regex features: prompt length, multi-step indicators, constraint count, code presence, math presence, nesting depth, ambiguity markers. Produces a weighted score mapped to bands.

```python
@dataclass
class DifficultyResult:
    difficulty_score: float     # [0, 1]
    adjusted_difficulty_score: float
    difficulty_band: str        # "easy" | "medium" | "hard"
    difficulty_features: dict
```

Band thresholds (configurable in `classifier_config.yaml`): easy < 0.3, medium 0.3-0.6, hard > 0.6.

Currently in `shadow` mode. Band-adaptive token budgets are wired in `_repl_turn_token_cap()` (`src/graph/helpers.py`): when mode is `enforce`, the flat 5000-token REPL cap is replaced with band-specific budgets (easy=1500, medium=3500, hard=7000). The `difficulty_band` propagates from `RoutingResult` → `TaskState.difficulty_band` → the cap function. In shadow mode, the flat cap is used (backward compatible).

Telemetry fields on `RoutingResult`: `difficulty_score`, `difficulty_band`. Also available on `EscalationContext`.

### Relationship to Conformal Risk Gate

These classifiers complement the conformal prediction gate:

| Signal | Type | When | What it measures |
|--------|------|------|------------------|
| Factual risk | Input-side | Before routing | Prompt's hallucination risk |
| Difficulty signal | Input-side | Before routing | Prompt's reasoning complexity |
| Conformal confidence | Output-side | During routing | Model's uncertainty on this task type |

They should not double-gate: if conformal prediction already rejects a routing, factual risk is moot. When both are in enforce mode, factual risk should modulate the conformal threshold (high factual risk → stricter confidence requirement).

### Output Quality Detection

`src/classifiers/quality_detector.py` detects degenerate model output via text heuristics:

1. **N-gram repetition** — trigram unique ratio below threshold (degeneration loops)
2. **Garbled output** — mostly very short lines mixed with long ones
3. **Near-empty output** — content too short after prefix stripping
4. **Think-block loop detection** — 4-gram repetition inside `<think>` blocks (reasoning model failure mode where the model enters a repetitive reasoning loop). Threshold: >15% duplicate 4-grams. Research backing: SEER shows failed outputs are ~1,193 tokens longer than successful ones; repetition within reasoning is a strong failure signal.

Config-driven thresholds via `ChatPipelineConfig`. The quality detector is always active (no mode gate).

---

*Previous: [Chapter 15: SkillBank & Experience Distillation](15-skillbank-experience-distillation.md)* | *Next: [Chapter 17: Programmatic Tool Chaining](17-programmatic-tool-chaining.md)*
