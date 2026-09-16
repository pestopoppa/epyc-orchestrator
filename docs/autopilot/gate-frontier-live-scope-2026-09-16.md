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

**D2 — an empty live-epoch frontier refuses single-trial promotion over an existing baseline.**
The old all-era view was never empty after the first trial, so the rule "skip the guard when the
archive is empty" only ever covered a fresh install. Under the live scope the frontier is empty
after every epoch rebase. The stored journal has 0 rows after the current fence
(`pareto_exclude_before_ts` = 2026-08-10T16:23:03Z), so at the next restart all three tiers start
empty. Keeping the skip would have opened single-trial promotions, with no reproduction evidence,
over the operator-reseeded baselines. This closes that hole; the freeze itself predates it (see
the finding below, 2026-06-18). The original refusal is now replaced by operator rule (b), next.

## Operator decisions, 2026-09-16 (after D2)

**(c) — the real fix: clean trials become frontier representatives.**
- The trial loop fixes the row it will journal (same timestamp, same objective-bearing fields)
  before the promotion decision (`_provisional_trial_row`). It hands that row to
  `update_baseline(pending_journal_rows=...)`, and the guard's live archive includes it.
  The candidate can therefore be its own cluster's representative
  (`last_tid == source_trial_id`), and `n_reproductions` counts it.
- A clean row (SafetyGate passed, no learning exclusion, live axes measured, tier ≥ 1) is
  stamped `eval_details.frontier_admission = "representative"`. It clusters by config
  fingerprint exactly like a trusted within-noise row. One predicate,
  `learning_exclusions.row_is_representative_member`, drives replay, the snapshot tail fold
  (such rows force full replay, which is what the W3 snapshot semantics require), crash-recovery
  re-import and reproduction counting. The live in-memory archive uses the same
  `upsert_representative` admission.
- **Atomicity.** The decision changes only in-memory state. The first persistence
  (`journal.record`) comes before the promotion ledger event and the state save; a test pins
  this ordering. A crash in between loses the in-memory promotion together with the trial, and
  the WAL writes the usual `autopilot_killed_mid_trial` placeholder, which is never a cluster
  member. After `journal.record`, the loop compares the recorded row with the provisional row
  and logs an ERROR on any divergence.
- Outside multitier mode, a within-noise reproduction (`mad_noise` /
  `reproduction_confirmed`) now also reaches `update_baseline`. Without this, the 2nd and later
  runs of an above-baseline config never reached the gate, so its cluster could never promote.
  Multitier mode (the live launcher) already stages that trial and re-checks at `final_t1`, now
  with the pending row.
- No old row is back-filled: stored rows have no stamp and keep per-trial semantics. The switch
  is `AUTOPILOT_CLEAN_TRIAL_REPRESENTATIVES` (on by default; `0` disables it).
- `promotion_rule` (`frontier` | `empty_frontier_repro` | `seed`, plus
  `refused_guard_unavailable`) is logged, returned on `BaselineUpdateResult`, and written to the
  trial row (`eval_details.promotion_rule`) and the `baseline_promotion` event
  (`result_metrics.promotion_rule`).

**(b) — the interim rule until the frontier fills.**
- Scope: the live frontier of the tier holds no member other than the candidate's own config,
  and the tier has a baseline.
- Requirement: the candidate config needs at least N independent live-regime reproductions,
  `AUTOPILOT_EMPTY_FRONTIER_MIN_REPRO` (default 3; a value below 2 is refused and falls back
  to 3).
- What counts as a reproduction (`live_reproductions.py`):
  - a representative-cluster member of the same tier and SERVED-CONFIG identity (see B1
    below; the action hash alone is not enough);
  - inside the live epoch, with its live axes measured;
  - carrying an AP-55 verdict of `COMPARABLE`, `UNVERIFIED` or none (pre-AP-55 rows);
    `NON_COMPARABLE` rows never count.
- The promoted fields are the median over exactly those reproductions, and that median must
  still clear the baseline by one quantum.
- As soon as another config holds a live frontier point, the normal frontier rule applies
  again. Under that rule, `n_reproductions` counts the whole cluster.
- Tier with no baseline: the `seed` rule applies, as before.

**Fail-closed guard (review finding).** If the live provider raises, or the legacy archive cannot
be read, promotion over an existing baseline is refused (`refused_guard_unavailable`). Before
this change the guard was skipped, and a single clean trial promoted. The reviewer reproduced
this with a state that had no epoch parameters. The live archive is built once per
`update_baseline` call, where it used to be built up to 4 times.

**What the next restart does (live state).** T1, T2 and T3 have baselines, and no journal row
falls after the fence. Each tier therefore starts under rule (b): no promotion over those
baselines until a config has 3 comparable reproductions in the new epoch. Rule (c) is what
produces those reproductions. The forward simulation (below) replays the stored tasks/hour
trials as if they arrived after the fence. The first decision runs under rule (b) and is refused
(1 of 3). The T1 frontier then fills (5 points), and every later decision runs under the
frontier rule. None of those replayed trials promotes, because no config in them reproduced 3
times. That is the evidence requirement working, not a freeze: a config reproduced 3 times that
clears the baseline promotes under both rules (`test_no_path_freezes_promotions_permanently`).

**AP-55.** The operator chose mode A (shadow). `AUTOPILOT_AP55_PROMOTION_GATE` defaults to
`shadow`, and any unknown value also means shadow (`ap55_promotion_gate.gate_mode` on
`sub/ap55bc-20260916`). The launcher does not set it. Rule (b) reads only the recorded AP-55
comparability verdict; it does not depend on the gate's mode. Follow-up task AP-55-ARM: after one
AutoPilot run in shadow mode, report how often enforce would have held a promotion, then arm
enforce plus seed re-runs (the operator pre-approved option B for after that review).

**D3 — the load-path quality ceiling is not epoch-fenced.** `Baseline.load`, `apply_state` and
`_drop_over_archive_max_tiers` still read the unscoped legacy view
(`_pareto_archive_for_safety_guard`). They read the quality axis only, so axis 1's unit never
reaches them. Fencing them would DELETE an operator-reseeded baseline whenever a young live-era
frontier sits below it. The reseed comes from a separate operational run, not a journaled
frontier trial.

## Replay: old path vs new path

The replay runs over the stored journal (1,372 trial rows, both shards, supersessions folded) and
calls `SafetyGate.update_baseline` (`scripts/analysis/gate_frontier_replay.py`; goldens are in
`tests/fixtures/gate_frontier_replay.json`). Eligibility is held open, there are no era holds,
and each tier's baseline starts empty. The live scope at each row is that row's policy stamp,
fenced at the first row that carried the stamp; legacy rows are unfenced. In every path the guard
sees the rows BEFORE the candidate, which is what the loop does.

- `old` — legacy all-era replay, the loop before 2026-09-16.
- `live` — live scope, no candidate row (commit `e401549d`).
- `live_c` — live scope with decisions (c) and (b). This is a what-if: candidate rows are
  stamped as the new loop would stamp them, the candidate's own row is handed to the guard, and
  within-noise reproductions are also decided. The stored journal is not modified.

| run | promoted | not representative | above archive max | too few repros (frontier rule) | median < quantum (frontier rule) | rule (b) refusals |
|---|---|---|---|---|---|---|
| old | 2 | 479 | 18 | – | – | – |
| live | 2 | 475 | 21 | – | – | 1 (no candidate row) |
| live_c | 4 | 93 | 1 | 7 | – | 1 (1 of 3 reproductions); plus 441 refused for no served-config identity |

**old vs live.** The promotions are identical: the T1 seed at trial 10 and the T3 seed at trial
1251, both in the legacy era, where the two paths are equal by construction (asserted). The only
4 differences are refusal-reason changes inside the rate era:
- 1474 is the first `task_rate_4d_v7` row, and its epoch holds no earlier point, so it now falls
  to rule (b); without a candidate row that rule refuses.
- 1477, 1500 and 1501 now stop at *above archive max*: the young v7 frontier's best quality is
  below these candidates, while the all-era legacy frontier was above them.

**live vs live_c.** With decisions (c) and (b) and the re-review B1 fix, the what-if yields 4
promotions: the same 3 seeds, plus one frontier-rule promotion. That promotion is trial 755, a
`structural_experiment` whose representative reproduced at least 3 times and cleared the
baseline by a quantum. **Rule (b) promotes nothing.**

The first (c)+(b) cut showed 27 promotions, 22 of them under rule (b). The re-review found them
vacuous: 19 were T1 `seed_batch` runs (trials 18–137, quality 0.0 → 1.9 across a month of config
changes) and 3 were T2 `deep_eval` runs. All of them clustered under one action hash, for example
`{"type": "seed_batch", "n_questions": 10}` → `4289ed22…`, which is not a served config.

**B1 fix — served-config identity.** A row can be reproduction evidence only if its action
names its served-config delta. Today only two kinds of action do that:
- a `structural_experiment` with non-empty `flags`;
- a `numeric_trial` with non-empty resolved `params`.

`action_identity.row_config_identity` computes the identity from that delta, plus the AP-55
infra digest when the row records one. Rule (b) counts only same-identity rows. A candidate
without an identity cannot promote under either rule, though a tier with no baseline can still
seed. Clean rows are stamped as representatives only when they have an identity. Measurement
and request-only actions are refused with "no served-config identity" in the what-if (441
decisions):

| action type | refusals |
|---|---|
| seed_batch | 331 |
| numeric_trial with empty `params` | 72 |
| deep_eval | 18 |
| train_routing_models | 6 |
| code_mutation | 5 |
| prompt_mutation | 3 |
| gepa_optimize | 2 |
| structural_prune | 2 |
| distill_skillbank | 1 |
| rollback | 1 |

**Operator decision (2026-09-16): prompt, code and GEPA mutations are identified by the
sha256 of the mutated file that was served.**
- **Recording.** The mutation handler writes the file, then immediately hashes the file on disk
  and leaves `{"files": {path: sha256}}` in the loop state, before the eval runs. The loop pops
  that record into `eval_details.served_content`, and it also clears the record before every
  dispatch, so a trial can only see its own.
- **Why the sha is not on the action.** A forced re-run copies the stored action, so a sha
  stored there would misidentify the re-run. Keeping it off the action also leaves
  `config_fingerprint` and `action_signature` unchanged: archive representative keys and repeat
  detection stay exactly as they were for old and new rows (tested).
- **Identity.** It is the sorted set of (path, sha) pairs, plus the regime digest (below).
  Two trials reproduce each other only if they served byte-identical content at the same paths
  under the same regime.
- **Old rows.** Rows without a sha stay non-promotable, and nothing is back-filled. The what-if
  replay is unchanged, since no stored row carries a sha.
- **Scope.** This applies to `prompt_mutation`, `code_mutation` and `gepa_optimize`.
  `structural_prune` is not covered and stays non-promotable.
- **Multitier mode.** Multitier staging accepts only numeric and structural candidates
  (`_seq_promotion_replay_blocker`). So under the live launcher, mutation candidates reach a
  promotion decision only outside multitier mode; that policy predates this change.

**Regime part of the identity (verification note, 2026-09-16).** The identity uses
`infra_fingerprint.regime_digest`: the AP-55 digest over the evaluator, kernel, recipe, models
and host components, WITHOUT the orchestrator component (git HEAD plus dirty digest).
- **Why the orchestrator component is dropped.** Every mutation auto-commit, merge or doc commit
  moves HEAD. Keeping it would start a new cluster on each commit, and the 3-reproduction bar
  could never be met.
- **What replaces it.** Orchestrator code and prompts that a trial changed are identified by
  the served-file sha instead.
- **What still splits a cluster.** A kernel, model, recipe, host or evaluator change does.
- **Deviation from the reviewer's list.** The evaluator stays in the regime, although the
  reviewer listed only kernel, model, recipe and host: the scorer is the measurement instrument.
- **Unreadable components.** An unreadable component enters the digest as "unavailable". The
  AP-55 comparability verdict judges such rows separately (UNVERIFIED still counts under rule
  (b), as decided).
- **Where it is recorded.** Rows carry `eval_details.infra_regime_digest`. A row that only
  carries a full fingerprint gets its regime digest derived from `component_digests`.
- **Tests.** The same config across an unrelated orchestrator commit still counts as a
  reproduction; a kernel or model change does not.

**Forward simulation (the next restart).** The stored tasks/hour trials (1472 and later) are
replayed as if they arrived after the live fence, starting from the live baselines
(T1 1.5, T2 1.356, T3 1.275):
- trial 1472 (`structural_experiment`) falls under rule (b) and is refused (1 of 3
  reproductions);
- the T1 frontier then fills (1, 2, …, 5 points) and every later decision uses the frontier rule;
- the `numeric_trial` candidates are refused for too few reproductions or for not being
  representatives, and the two `seed_batch` runs are refused for having no served-config
  identity;
- nothing promotes.

**Re-review B2 — a journal/decision mismatch rolls the promotion back.** Before the decision, the
loop deep-copies `gate.baseline`. After `journal.record` it compares the recorded row with the
row the guard evaluated (`_reconcile_promotion_with_journal`). On any mismatch it restores the
baseline and turns the update into a refusal, so neither the promotion event nor the promoted
`baseline_state` is written.

**Re-review B3 — a crash after `journal.record`.** The row now carries
`promotion_status: pending_commit` or `refused`. The commit record is the `baseline_promotion`
ledger event with that `source_trial_id`, appended together with the final state save. A crash in
between leaves a pending row without an event and an unchanged baseline, which is consistent.
Readers must treat an unconfirmed pending row as NOT promoted, and recovery needs no action. This
was the simpler of the two options.

**Finding (pre-existing on both paths; fixed by decision (c) above).** In production ordering the source trial
is never in the journal when `update_baseline` runs (`journal.record` comes after it). So a clean
trial can only promote when the guard's frontier is empty. Every other candidate is refused as
*not a same-tier frontier representative*: 479 of 499 guard decisions on the old path. The
`update_baseline` docstring assumes archive-first ordering. Since the 2026-06-18
journal-authority cut-over, the guard has not seen the in-memory `archive.update()`. Whether a
clean single trial should ever promote, or only a ≥3-reproduction representative should, is a
promotion-policy question. The operator answered it with decisions (c) and (b).

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
