# E-1 — Reasoning-Effort Ladder Design

**Task**: `handoffs/active/reasoning-effort-levels.md` E-1 (L173). Design-only deliverable.
**Date**: 2026-07-22 · **Executable spec**: `scripts/analysis/effort_ladder_spec.py`
**Status**: COMPLETE (design). Unblocks E-2 (measure the curve, per model).

> Design doc. All token numbers are **OBSERVATIONS** (R2a endpoints + TB-7 mined realized demand).
> No level is production-enabled until its per-(model,quant) curve is certified by E-2. Nothing here
> edits the stack; a per-role default is a proposal to the operator (E-4), not an edit.

---

## Design premises (measured, not assumed)

1. **The PROMPT is the accuracy lever, not the token cap.** R2a: same 50 GPQA-Diamond items,
   `enable_thinking=false` both arms — "letter only" 52.0%/537tok vs "reason step by step" 84.0%/2150tok
   = **+32.0pp, p=8.6e-04, 4.0× tokens**. So the ladder's rungs are *prompt conditions*.
2. **A bare `max_tokens` cap is forbidden as the dial.** It truncates mid-derivation and the answer
   scores *wrong* — you lose the quality without ever getting a cheaper answer (handoff L59-68;
   observed on AIME@4096 and CoT@8192). Budget is a **backstop with a graceful close**, never the knob.
3. **Native `<think>` is a safety-capped option, not an accuracy win.** E-6: force-closing `<think>`
   with `--reasoning-budget N --reasoning-budget-message` drives non-termination to 0% and recovers
   accuracy to ≈think-off, but **no capped arm beats think-off** (all p≥0.62). So native thinking
   earns at most a *safe* rung (L4), and only for models whose channel terminates under cap (R2b:
   35B-A3B fails to terminate 48% of the time even at 16k → L4 disabled there).
4. **Per-model INVARIANT** (handoff L36): levels are *defined* generically but *certified* per
   (model, quant). Backstops below are model-agnostic starting points, re-certified by E-2.

## The ladder (5 rungs)

| id | name | prompt condition (abridged — full text in the spec) | think | answer order | backstop | reasoning budget |
|---|---|---|---|---|---:|---:|
| **L0** | answer-only | "ONLY the final answer, no explanation" | off | answer-first | 256 | — |
| **L1** | terse-justify | "answer first, then ≤1 sentence, <40 words" | off | answer-first | 512 | — |
| **L2** | bounded-reasoning | "≤3 short steps, ~120 words, then answer" | off | reasoning-first | 1024 | — |
| **L3** | full-cot | "reason step by step, then final answer" | off | reasoning-first | 4096 | — |
| **L4** | native-think-capped | clean prompt; native channel supplies reasoning | **on** | reasoning-first | 512 | 2048 (force-close) |

L0 and L3 are the two **already-measured endpoints** (they already exist in
`dataset_adapters.py` as `gpqa_diamond` = letter-only and `gpqa_diamond_cot` = full CoT). L1/L2 are
the **candidate intermediate rungs** whose existence E-2 must confirm — if the accuracy/token curve
turns out to be a step (you either reason or you don't), L1/L2 collapse and the "ladder" is really a
binary threshold per role (the core design question, handoff L32).

## Two design decisions worth calling out

**1. Answer-ordering is chosen per rung to make cheap rungs truncation-robust.**
L0/L1 put the **answer first**, so even an aggressive backstop yields a well-formed answer — the
cheap rungs cannot produce the truncation artifact. L2/L3 put **reasoning first** (the reasoning has
to precede the answer to help), so they need a *generous* backstop and, on hitting it, are
**disqualified — not scored wrong** (see below). This is why L3's backstop is 4096, well above the
R2a mean of 2150 and above the TB-7 code/math p99 (~3080, and right-censored at 2048).

**2. Backstops are sized from realized demand (TB-7), with headroom, never as targets.**
From the TB-7 mining (`TB7_BUDGET_MINING_REPORT.md`): realized p95 demand is ~1-50 tok for
saturated classes, ~300-500 for light QA, ~700-1000 for moderate math, ~1500-2200 (censored) for
hard code/math. The rung backstops (256 / 512 / 1024 / 4096) sit above the p95 of the class each rung
targets, so a well-behaved response never reaches them. **The backstop's only job is a clean close.**

## Disqualification rule (encoded in `effort_ladder_spec.disqualify`)

A rung's result on an item is **DROPPED, not scored incorrect**, when it (a) truncates
(`finish_reason ∈ {length, truncated}`), (b) hits the budget backstop, (c) emits empty content, or
(d) produces unparseable output. Scoring these as "wrong" is the exact artifact that has already
biased two measurements (handoff L61-64, L181, and `feedback_parse_failure_rate_is_a_scoring_artifact`).
E-2 must report truncation-rate / parse-failure-rate / empty-rate **per rung** alongside accuracy, and
a rung that truncates is disqualified at that budget, not compared on accuracy.

## How E-2 consumes this

- Import `LADDER` and `render(level, task)` from `effort_ladder_spec.py` so prompt text lives in one
  place. Sweep all 5 rungs on the **already-pinned** GPQA-Diamond items
  (`artifacts/architect-bench-gpu-20260720/questions_gpqa_diamond_cot.json`) → one Pareto frontier
  (accuracy vs mean tokens) **per (model, quant)**.
- For L4, launch with `--reasoning-budget 2048 --reasoning-budget-message "<REASONING_BUDGET_MESSAGE>"`
  and skip the rung entirely for any model whose native channel is unproven under cap.
- Score by **rescue-rate vs the next-cheaper rung** (E-3), not mean accuracy — a rung may only help a
  narrow band of hard items, which mean accuracy hides.

## Interactions (do not tune blind)

- **Budget is a third axis** (TB-4): a tighter rung *reduces* the budget a model needs; the full
  picture is the (effort × budget) grid per model. The backstops here are the budget column for each
  rung; TB-1 refines the true knee for the censored hard classes.
- **Agent-file compression** already changes the prompt per role
  (`agent_file_compression_operating_point`); the effort rung stacks on top of it — E-2 must fix the
  compression point while sweeping effort, or the two confound.
- **Registry indexing**: certified rungs are stored by **model/quant, never role**
  (`feedback_model_not_role_indexing`); a role default (E-4) is then a `(model × rung)` pair, and a
  model/quant/kernel swap invalidates it (E-7).

## Open question E-2 must answer

Is the accuracy/token curve **smooth** (L1, L2 are real, usable dial positions) or a **step** (only
L0 and L3 matter)? R2a only measured the two endpoints. If it is a step, the deliverable to the
operator is a per-role binary (reason / don't) plus the L4 safety cap — still useful, but not a dial.
