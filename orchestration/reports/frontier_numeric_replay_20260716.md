# Frontier Numeric Replay — 2026-07-16

Quiet-window replay of the two numeric frontier parameter configs from 2026-06-28 against the current legacy T1 vector.
AutoPilot was stopped; replay used legacy T1 `n=50`, `seed=42`, `AUTOPILOT_W6_AUDIT_BLOCK=0`, and `AUTOPILOT_TOOL_SENTINELS=0`.
Both runs applied env-backed params through API-only orchestrator reload and restored the API afterward.

Important caveat: this is an exact parameter replay, not an exact historical question-vector replay. `legacy_pool_seed_42_n50` is derived from the current question pool. The replay qid vector exactly matches the current EvalTower helper sample for seed 42, but overlaps the historical frontier rows by only 8/50 qids for trial 996 and 7/50 qids for trial 1003.

## Results

| Source trial | Surface | Historical q | Historical speed | Historical r | Replay q | Replay speed | Replay r | Errors | Tokens | Wall s | Artifact |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 996 | `think_harder` | 2.04 | 69.21354521045092 | 0.98 | 1.7399999999999998 | 34.75560694205509 | 0.86 | 7 | 20624 | 593.4006571769714 | `orchestration/reports/frontier_replay_996_20260716T001046Z.json` |
| 1003 | `memrl_retrieval` | 2.10 | 68.35713081738538 | 0.98 | 1.7999999999999998 | 35.309929890347824 | 0.86 | 7 | 21525 | 609.6018900871277 | `orchestration/reports/frontier_replay_1003_20260715T235917Z.json` |

## Interpretation

- The numeric frontier parameter replays validate the user's suspected throughput regression direction under the current EvalTower/vector: the new floor for these configs is about `35 t/s`, roughly half the 2026-06-28 frontier speeds, despite the newer kernel.
- Replay quality and reliability also fell in the raw aggregate, but this must not be read as a clean model-capability regression. The question vector changed, and reliability/error inflation alone is large enough to explain the observed quality delta:
  - Trial 996: historical `34/50` correct with 1 error; replay `29/50` correct with 7 errors.
  - Trial 1003: historical `35/50` correct with 1 error; replay `30/50` correct with 7 errors.
  - The 6 additional replay errors exceed the 5-correct quality drop in both cases.
- A concrete runtime bug contaminated the quality/error side of these replay rows: plan-review reroute patch text was accepted as a backend role, producing errors such as `No backend configured for role 'Complete filtering, add KMeans, handle column count'`. That path is fixed by `bc8d3303` (`Validate plan review reroute targets`), so these replay quality numbers should not be promoted as post-fix quality evidence.
- Structured tap evidence during both runs showed more request starts than the nominal 50 core questions because current EvalTower routing performs auxiliary plan/reformat calls. That protocol/runtime behavior is a likely contributor to the speed drop and should be investigated before letting the planner treat the surface search itself as the primary bottleneck.
- Trial 996's recovered Optuna params include `think_harder.token_budget_min=3808` and `think_harder.token_budget_max=3054`; this inverted budget pair was replayed exactly and should be treated as a historical anomaly, not a sane candidate to promote.

## Root-Cause Follow-up

- The qid mismatch explains why the quality comparison was surprising: the replay used the current `legacy_pool_seed_42_n50` sample, not the June frontier vector. Current EvalTower can reproduce the replay qids exactly with `_sample_scoreable_eval_questions(load_pool(), 50, random.Random(42))`; the historical rows predate the current pool/vector contents.
- The quality drop is dominated by evaluation/runtime failures, not a demonstrated raw-answer collapse. The immediate fixed bug was invalid plan-review reroute targets escaping into `routed_to`; remaining quality investigation should use post-`bc8d3303` rows and paired qid vectors only.
- For future frontier replays, archive and replay the historical `question_results[].qid` vector directly when the question comparison matters. Reusing `legacy_pool_seed_42_n50` after pool changes is only suitable for current-floor probes.

## Non-Numeric Frontier Trials

Trials 998 and 1005 were `seed_batch(n_questions=16)` actions, not pure runtime configs. Replaying them would mutate the seed/question pool and is not deterministic in the same sense as the numeric config replays. They remain journal-authoritative unless the operator explicitly wants a controlled mutating seeding replay.

## Failed Instrument Attempt

An earlier local attempt ran through a current-protocol path with W6/tool-sentinel overlays and produced all-error zero-token rows. That attempt is not decision evidence for frontier performance; it is useful only as an instrument-failure breadcrumb and is summarized here rather than committed as a replay artifact.
