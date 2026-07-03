# Stepping-Stone Archive-Authority Ablation Protocol

**Created:** 2026-07-02 · **Source:** intake-772 (Darwin Gödel Machine, arXiv 2505.22954) via
`handoffs/active/autopilot-continuous-optimization.md` → "Research Intake Distillation — DGM + Harness-Evolution".
**Status:** required gate before the stepping-stone lane may be promoted from observe-only to an
authoritative parent-selection signal. **Nothing here runs automatically** — every arm consumes serial
inference and is operator-scheduled under the single-user no-concurrent-inference rule.

## What is being ablated

DGM's central empirical finding is that BOTH (a) self-improvement AND (b) an open-ended keep-all
**stepping-stone archive** are load-bearing — it beats a *no-open-ended-exploration* ablation
(archive-free, mutate-latest-only). We adopt only axis (b): a bounded, diversity-sampled lane of
dominated-but-novel configs surfaced to the planner (`ParetoArchive.stepping_stones{,_text}` →
`autopilot.py` planner-prompt geometry block, gated by `AUTOPILOT_STEPPING_STONES`).

**Explicit non-applicability — the self-improvement axis is permanently OFF for us.** Our meta-loop,
`SafetyGate`, scoring, and sequential verdict are human-authored and outside `program.md`'s mutable
scope + the MEASUREMENT.md human-amendment-only trust boundary. Autopilot cannot rewrite its own
controller. So DGM's *no-self-improvement* ablation has **no toggle** here; only the open-ended-
exploration axis is a live variable. Do not read the DGM "self-rewriting agent" framing as license
for the loop to edit its own code.

## The two arms

| Arm | `AUTOPILOT_STEPPING_STONES` | Planner sees | DGM analog |
|-----|-----------------------------|--------------|------------|
| **A — frontier-only** (control) | `0` | frontier via `summary_text`/`geometry_text` only (pre-2026-07-02 behavior) | *no open-ended exploration* |
| **B — stepping-stone-on** | `1` (default) | frontier **+** the diversity-sampled dominated-but-novel lane | *open-ended stepping stones* |

The lane is prompt CONTEXT only in its current (do-this-first) form: it can shift what the LLM planner
proposes, but it never re-runs a config and never places a point on the frontier — a stone earns a
frontier point solely by winning the existing 4D dominance test. This protocol governs any move BEYOND
that (e.g. wiring the fecundity parent-sampler of Pattern 2 to draw from the lane, or an
`explore_from_trial` re-run action) into an authoritative signal.

## Why `bsv_paired_runner.py` is necessary but not sufficient

`bsv_paired_runner.run_paired_evaluation()` applies a **baseline vs candidate param set** via
`config_applicator.apply_params`, evals both arms on the SAME `EvalTower` core + seed, and emits a
paired report with a `gate_decision`. That is the right tool for a **single-config** change. The
stepping-stone lane is **not** a config param — it is a search-trajectory effect that only manifests
across MANY planner turns. So the ablation is a **segment-level** paired comparison, not a single
paired eval:

1. Run two autopilot SEGMENTS of equal length N (recommend N ≥ 40 frontier-eligible trials each,
   sized against the current per-suite resolution so a real effect clears noise): one with
   `AUTOPILOT_STEPPING_STONES=0`, one with `=1`. Hold everything else fixed (same species budgets,
   same eval tiers, same seeds where the machinery allows, same served fleet).
2. Compare the arms on TWO distinct axes:
   - **Trajectory effect** (the whole point of the lane): hypervolume growth rate
     (`ParetoArchive.hypervolume_slope`), frontier-rate, and local-optimum escapes (new distinct
     frontier regions). ⚠ `paired_stats.py` does NOT cover this — it is a per-question McNemar
     comparator over binary per-qid correctness vectors, not a trajectory-series test. A segment-level
     trajectory statistic (a paired hv-slope / frontier-rate delta between the two arms) is **not yet
     implemented** and must be built for a rigorous adjudication of this axis.
   - **Non-inferiority on quality/rate**: the per-question outcomes the two arms produce are what
     `paired_stats.py` (McNemar) and the sequential verdict's E-quality / E-rate axes consume.
3. Adjudicate with the anytime-valid **sequential verdict** (`safety_gate.py` `_sequential_verdict`,
   `AUTOPILOT_SEQ_VERDICT=1`): require `confirmed` — both the E-quality and E-rate-non-inferiority
   e-values ≥ `confirm_e` — on the non-inferiority axis, AND require a positive trajectory effect
   (hv-slope / frontier-rate) from the segment statistic above, before promoting the lane to
   authoritative. A `refuted`/inconclusive result keeps the lane observe-only (or removes it if it is
   net-negative on hypervolume slope).

`bsv_paired_runner` is still used WITHIN a segment for any single-config change the planner makes; it
is not the segment-level adjudicator. For the segment ablation, `paired_stats.py` + the sequential
verdict adjudicate only the **per-question non-inferiority** axis; the **trajectory effect** axis
(hv-slope / frontier-rate delta between arms) needs a segment-level statistic that is not yet
implemented.

## Acceptance / promotion rule

- **Promote to authoritative** (allow Pattern 2 fecundity sampling to seed from the lane, and/or add
  an `explore_from_trial` re-run action) **only** when the sequential verdict returns `confirmed`
  with the stepping-stone arm non-inferior on quality/rate AND showing a positive hypervolume-slope
  or local-optimum-escape effect.
- **Keep observe-only** on inconclusive results (the lane stays as prompt context; harmless).
- **Retire** (default the flag to `0`) if the stepping-stone arm is confirmed net-NEGATIVE on
  hypervolume slope — i.e. the extra context distracts the planner more than it helps.

## Guardrails

- **Trust boundary:** this protocol does not modify the SafetyGate verdict, the scoring/objective, or
  the sequential-verdict e-process — all human-amendment-only. It only decides whether a *selection*
  (not acceptance) signal is turned on. Any change that would touch the objective (e.g. Pattern 5's
  token cost axis) is a separate operator-approval step with an `instrument_eras.yaml` note.
- **Compute:** both arms are serial inference; schedule them, never run concurrently (no-concurrent-
  inference rule).
- **Measurement:** the DGM SWE-bench/Polyglot numbers that motivated this are external observations,
  not decision-gating — the promotion decision rests entirely on OUR paired, resolution-aware,
  sequentially-verified segment comparison (MEASUREMENT.md).

## Related

- `handoffs/active/autopilot-continuous-optimization.md` → Patterns 1 (stepping-stone lane), 2
  (fecundity parents), 3 (this protocol).
- `research/deep-dives/2026-07-02-cross-model-lora-transfer-cluster.md` (companion intake lineage).
- Code: `pareto_archive.py::ParetoArchive.stepping_stones{,_text}`, `autopilot.py` planner-prompt block
  (`AUTOPILOT_STEPPING_STONES`), `bsv_paired_runner.py`, `paired_stats.py`, `safety_gate.py`
  (`_sequential_verdict`).
