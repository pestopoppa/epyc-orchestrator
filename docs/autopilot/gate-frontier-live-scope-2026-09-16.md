# Gate frontier: live scope and unit routing (2026-09-16)

This decision note covers the paired safety-gate change: the promotion guard now reads the live
frontier, and the rate axis is routed by its unit. The operator decided on 2026-09-16 to build it.
It takes effect at the next autopilot restart.
Code: `scripts/autopilot/safety_gate.py`, `scripts/autopilot/autopilot.py`
(`_install_promotion_guard_scope`), `src/autopilot_core/tier_specs.py` (`rate_axis_unit`).
Tests: `tests/unit/test_gate_frontier_live_scope.py`.
Replay: `scripts/analysis/gate_frontier_replay.py`, with results pinned in
`tests/fixtures/gate_frontier_replay.json`.

## What changed

1. **Promotion evidence reads the live frontier.** Promotion evidence covers the archive-max
   refusal, the "source trial is a same-tier frontier representative" check and the
   reproduced-field override. At startup the autopilot installs a provider that rebuilds the
   frontier through `_journal_archive_payload_for_authority`, the same authority path that seeds
   the live archive. The provider uses the live `pareto_objective_policy`, the live
   `pareto_epoch_ts` / `pareto_exclude_before_ts` fence, and a verified segment snapshot when its
   scope matches. It reads the loop's live state dict on every call. Without a provider (tools,
   tests), the guard falls back to the old legacy all-era replay and logs a WARNING once.
2. **The rate axis is routed by unit.** `promotion_fields_from_objectives(objectives, tier,
   objective_policy)` resolves the unit with `rate_axis_unit(policy)`. Tokens/second goes to
   `speed`. Questions/hour goes to `task_rate_qph`, which `Baseline.update_tier` stores as the new
   `frontdoor_task_rate_qph`. An unknown policy is refused (`None`). The code never guesses.
3. **The throughput floor compares like with like** (D1 below). It remains
   `result.speed < 0.8 × frontdoor_speed`, and both sides are tokens/second.
4. `frontdoor_task_rate_qph` is optional in the state payload and the YAML. A record without it
   loads as `None`, and a baseline that never set it writes a byte-identical payload.

## Decisions

**D1 — the floor stays tokens/second.** The handoff (W3 / W4) keeps EvalTower `speed` as the
host-throttle and throughput diagnostic. The live dominance axis became questions/hour, but no
floor was ever defined in q/h. The seq rate axis already runs an anytime-valid non-inferiority
test on q/h (`rate_noninferiority_z`), so a second, cruder 0.8× q/h floor would duplicate it with
worse statistics. q/h also moves with `n` and with the question mix (the handoff records a 19%
objective gap from a 3% speed difference). A 0.8× hard floor on it would reject trials for
sampling reasons. So `frontdoor_speed` only ever receives a genuine t/s measurement:
- a legacy-policy reproduced median; or
- the promoted trial's own `result.speed` when the guard runs under a rate policy.

That second case is a single measurement, not a median, because a q/h frontier carries no t/s
median. `frontdoor_task_rate_qph` is recorded for visibility, and no gate reads it. Adding a q/h
floor would be a new gate and therefore a separate decision.

**D2 — an empty live-epoch frontier refuses promotion over an existing baseline.** The old
all-era view was never empty after the first trial, so the rule "skip the guard when the archive
is empty" only ever covered a fresh install. Under the live scope the frontier is empty after
every epoch rebase. The stored journal has 0 rows after the current fence
(`pareto_exclude_before_ts` = 2026-08-10T16:23:03Z), so at the next restart all three tiers start
empty. Keeping the skip would have opened single-trial promotions, with no reproduction evidence,
over the operator-reseeded baselines. The guard now refuses instead
(`live-epoch frontier is empty ...`). A tier with no baseline at all still seeds, so bootstrap is
never blocked. The legacy fallback keeps the old skip.

**D3 — the load-path quality ceiling is not epoch-fenced.** `Baseline.load`, `apply_state` and
`_drop_over_archive_max_tiers` still read the unscoped legacy view
(`_pareto_archive_for_safety_guard`). They read the quality axis only, so axis 1's unit never
reaches them. Fencing them would DELETE an operator-reseeded baseline whenever a young live-era
frontier sits below it. The reseed comes from a separate operational run, not a journaled
frontier trial.

## Replay: old path vs new path

The replay runs over the stored journal (1,372 trial rows, both shards, supersessions folded) and
calls `SafetyGate.update_baseline`. Eligibility is held open, there are no era holds, and each
tier's baseline starts empty. The live scope at each row is that row's policy stamp, fenced at the
first row that carried the stamp; legacy rows are unfenced. "Production" ordering is what the loop
does, because the trial is journaled after `update_baseline`. "Archive-first" also shows the
candidate row to the guard.

| run | promoted | not representative | above archive max | too few repros | live frontier empty | no quantum |
|---|---|---|---|---|---|---|
| old / production | 2 | 479 | 18 | – | – | – |
| new / production | 2 | 475 | 21 | – | 1 | – |
| old / archive-first | – | 450 | – | 52 | – | 1 |
| new / archive-first | – | 444 | – | 58 | – | 1 |

The **promotions are identical**: the T1 seed at trial 10 and the T3 seed at trial 1251, both in
the legacy era, where the two paths are equal by construction. That equality is asserted.
**Every difference is a change of refusal reason inside the rate era.** Nothing flips between
promoted and refused.

- Production ordering:
  - 1474 changes from *not representative* to *live frontier empty*. It is the first
    `task_rate_4d_v7` row, so its epoch holds no earlier point (D2).
  - 1477, 1500 and 1501 change from *not representative* to *above archive max*. The young v7
    frontier's best quality is below these candidates, while the all-era legacy frontier was
    above them, so the old path reached the representative check and the new path stops one step
    earlier.
- Archive-first ordering: 1474, 1475, 1477, 1479, 1500 and 1501 change from *not
  representative* to *too few reproductions*. On the young live frontier the candidate itself is
  a frontier point with `n_reproductions = 1`. On the all-era legacy frontier it is dominated.

**Finding (pre-existing, both paths, not changed here).** In production ordering the source trial
is never in the journal when `update_baseline` runs (`journal.record` comes after it). So a clean
trial can only promote when the guard's frontier is empty. Every other candidate is refused as
*not a same-tier frontier representative*: 479 of 499 guard decisions on the old path. The
`update_baseline` docstring assumes archive-first ordering. Since the 2026-06-18
journal-authority cut-over, the guard has not seen the in-memory `archive.update()`. Whether a
clean single trial should ever promote, or only a ≥3-reproduction representative should, is a
promotion-policy question. It is recorded here for the operator; this change does not alter it.

## Why "every later trial fails the floor" cannot occur

- Inside the running gate, `frontdoor_speed` is written in only two places:
  `Baseline.update_tier` (from `result.speed`) and `_reseed_speed_axis_if_held` (also from
  `result.speed`). `load` and `apply_state` only restore a persisted value.
- Under a rate policy, `promotion_fields_from_objectives` never emits `speed`. This is tested for
  every registered rate policy, and an unknown policy returns `None` and refuses the promotion.
  So `result.speed` keeps the trial's EvalTower t/s.
- `test_later_trials_pass_the_throughput_floor_after_a_live_promotion` promotes under
  `task_rate_4d_v7` (q/h 59.49, t/s 14.78), then shows that 14.78, 13.0 and 12.0 t/s pass the
  floor and 10.0 t/s is still caught.
- `test_mutation_old_routing_would_brick_the_floor` re-applies the pre-change routing and shows
  that `frontdoor_speed` becomes 59.49 and an unchanged 14.78 t/s trial fails the floor. The test
  therefore detects the failure it guards against.
