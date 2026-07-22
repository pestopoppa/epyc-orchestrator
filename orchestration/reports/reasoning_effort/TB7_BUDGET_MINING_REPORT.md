# TB-7 — Budget-policy calibration from AutoPilot's live token stats (no new inference)

**Task**: `handoffs/active/reasoning-effort-levels.md` TB-7 (L150).
**Date**: 2026-07-22 · **Script**: `scripts/analysis/effort_tb7_budget_mining.py` · **Raw**: `tb7_budget_mining.json`
**Status**: COMPLETE (mining deliverable). Feeds TB-2 / TB-3; does **not** replace the TB-1 knee sweep.

> OBSERVATION per MEASUREMENT.md — realized-demand mining, no protocol-id. Proposes budget defaults;
> does **not** gate a production budget change without the TB-1 capability cross-check.

---

## What was mined

AutoPilot's journal already records, for a subset of eval questions, the per-question
`tokens_generated` alongside `suite` (task-class), `route` (role→model), and `correct`.
**5,135** such records exist across both shards (`autopilot_journal_1.jsonl` +
`autopilot_journal.jsonl`), spanning **2026-07-14 → 2026-07-16** (a recent, roughly current-era
window). This is a free, production-grounded distribution of "how much budget did each task class
actually consume" — exactly the TB-7 signal. No inference was run.

**Deployable-budget rule**: `budget = ceil(p95(success-tokens) × 1.15 / 512) × 512` — the 95th
percentile of realized demand among *successfully completed* tasks, +15% headroom, rounded to a
512-token quantum. Per TB-7 this is then capped by the TB-6 VRAM/concurrency budget.

## Per-(task-class, model) budget defaults (n ≥ 30 successful samples)

| task-class | route (model) | n | median | p95 | p99 | **budget** |
|---|---|---:|---:|---:|---:|---:|
| mode_advantage_hard | worker_general | 82 | 2048 | 2176 | 2176 | **2560** |
| debugbench | worker_general | 59 | 1822 | 1883 | 2048 | **2560** |
| mode_advantage | frontdoor | 83 | 1959 | 1959 | 2322 | **2560** |
| livecodebench | coder_escalation | 69 | 1586 | 1586 | 1586 | **2048** |
| livecodebench | worker_general | 68 | 850 | 1676 | 3080 | **2048** |
| bigcodebench | frontdoor | 48 | 1566 | 1772 | 1948 | **2048** |
| real_suite_v1 | frontdoor | 35 | 20 | 1747 | 2315 | **2048** |
| coder | frontdoor | 163 | 1084 | 1193 | 1929 | **1536** |
| cruxeval | worker_general | 116 | 1024 | 1024 | 1025 | **1536** |
| instruction_precision | worker_general | 91 | 442 | 906 | 1677 | **1536** |
| math | worker_general | 243 | 412 | 680 | 1302 | **1024** |
| skill_transfer | worker_general | 85 | 500 | 500 | 519 | **1024** |
| gpqa | worker_general | 86 | 1 | 512 | 637 | **1024** |
| real_suite_v1 | worker_general | 104 | 437 | 539 | 1352 | **1024** |
| hotpotqa | worker_general | 154 | 91 | 309 | 390 | **512** |
| agentic | worker_general | 122 | 159 | 307 | 307 | **512** |
| long_context | worker_general | 111 | 34 | 51 | 86 | **512** |
| tool_use | * (frontdoor/worker/architect) | 30–131 | ~25 | ~30 | ~31 | **512** |
| general / thinking / vl | worker_general / worker_vision | 54–249 | 1–2 | 1–45 | 5–86 | **512** |

(Full 31-cell table + per-suite-pooled and per-route-pooled tables in the JSON.)

## Findings

**1. Per-task-class spread is enormous — hypothesis 1 (per-task tuning matters) is supported.**
Realized demand runs from **~1–2 tokens** (`general`, `thinking`, `vl` in this window — answer-only
responses) to **~2,000–2,200 tokens** (`mode_advantage_hard`, `debugbench`, `mode_advantage`,
`livecodebench`, `bigcodebench`). A single stack-wide budget sized for the hard code/math classes
(~2,560) would over-provision the easy 80% of traffic by 100–1000× on the KV/VRAM axis that TB-6
says is the binding constraint. **Budget should be set per task-class (or at least per class-tier),
not one stack-wide number.**

**2. The realized-demand distribution is partially RIGHT-CENSORED at current caps — p95 values are
lower bounds.** Exact-value spikes at **512 (×172), 1024 (×80), 2048 (×63)** are requests hitting a
max_tokens ceiling, not natural completion lengths; only **1** record exceeds 4096. So for the
high-demand code/math suites (`livecodebench` p99 already 3080; `debugbench`, `mode_advantage_hard`
pinned at ~2048) the true knee is **above** what this window observed. These are precisely the
classes that need a **TB-1 sweep** (`olympiadbench_hard`, `max_tokens ∈ {4k…32k}`) to find the real
knee. The easy classes (`general`, `hotpotqa`, `long_context`, `tool_use`) show natural,
uncensored, low demand — a modest budget is safe there with no sweep needed.

**3. "Single mostly-ok setting" (TB-2 / hypothesis 2) — a partial answer from existing data.**
Within a *given model*, a per-model default equal to that model's **highest-demand task-class budget**
captures all its classes at the cost of over-serving its easy ones. For `worker_general` that is
~2,560 (driven by `mode_advantage_hard`/`debugbench`); for `frontdoor` ~2,560 (`mode_advantage`); for
`coder_escalation` ~2,048 (`livecodebench`). So a **per-model** single budget of ~2.5k is "mostly-ok"
for quality but wastes KV — acceptable only if TB-6's VRAM budget permits. A truly cheap policy needs
the task-class dial (finding 1). The censoring (finding 2) means even these per-model numbers are
floors for the code/math-heavy models.

**4. Degenerate tail.** 316 records at 0 tokens and 436 at 1 token (empty / single-char completions)
sit in the easy classes — they pull medians down but do not affect the high-percentile budget. They
are flagged because a 0-token "success" may be a scoring artifact (see
`feedback_parse_failure_rate_is_a_scoring_artifact`), worth a look independent of budgeting.

## Cross-check (trial-level aggregate)

`tokens_per_solved_task` across 486 trials: median **808**, p90 **1,043**, max **1,996**. Consistent
with the mid-range per-task-class cells; the aggregate hides the code/math tail that the per-class
view exposes, which is why the per-class breakdown is the actionable artifact.

## Caveats (binding)

- **Realized demand ≤ true knee.** Percentiles are over *successful* completions; tasks that failed
  *because* they truncated needed more and are invisible here. Every budget above is a **lower bound**
  — do not lower a production ceiling below these, and cross-check the code/math cells against TB-1.
- **Config mixture.** The window mixes whatever `enable_thinking` / prompt-effort was live per trial;
  the very low medians for `thinking`/`general` reflect answer-only configs, not the model's capacity.
- **Per-model INVARIANT** (`reasoning-effort-levels.md` L36): budget is a `(model, quant)` property.
  `route→model` labels here are the *current* binding; a model/quant/kernel swap re-opens the cell.
- **No truncation signal in the journal.** The per-question record has no `truncated`/`finish_reason`,
  so truncation-rate-vs-budget (the actual TB-1 knee) cannot be computed here — only realized demand.

## Recommended next actions (for the handoff owner; not executed here)

- Feed the per-class budgets above into TB-3 (audit live `reasoning_budget` / `max_tokens_multiplier`
  against these floors) — any role whose live ceiling is below its class budget is silent quality loss.
- Run TB-1 only for the **censored** high-demand classes (code/math); the uncensored easy classes are
  already answered by this mining.
