# Frontier Numeric Replay — 2026-07-16

Quiet-window replay of the two exact numeric frontier configs from 2026-06-28.
AutoPilot was stopped; replay used legacy T1 `n=50`, `seed=42`, `AUTOPILOT_W6_AUDIT_BLOCK=0`, and `AUTOPILOT_TOOL_SENTINELS=0`.
Both runs applied env-backed params through API-only orchestrator reload and restored the API afterward.

## Results

| Source trial | Surface | Historical q | Historical speed | Historical r | Replay q | Replay speed | Replay r | Errors | Tokens | Wall s | Artifact |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 996 | `think_harder` | 2.04 | 69.21354521045092 | 0.98 | 1.7399999999999998 | 34.75560694205509 | 0.86 | 7 | 20624 | 593.4006571769714 | `orchestration/reports/frontier_replay_996_20260716T001046Z.json` |
| 1003 | `memrl_retrieval` | 2.10 | 68.35713081738538 | 0.98 | 1.7999999999999998 | 35.309929890347824 | 0.86 | 7 | 21525 | 609.6018900871277 | `orchestration/reports/frontier_replay_1003_20260715T235917Z.json` |

## Interpretation

- The exact numeric frontier replays validate the user's suspected regression direction: the new floor for these configs is about `35 t/s`, roughly half the 2026-06-28 frontier speeds, despite the newer kernel.
- Replay quality and reliability also regressed: both exact configs fell from `r=0.98` to `r=0.86`, with 7 errored questions each.
- Structured tap evidence during both runs showed more request starts than the nominal 50 core questions because current EvalTower routing performs auxiliary plan/reformat calls. That protocol/runtime behavior is a likely contributor to the speed drop and should be investigated before letting the planner treat the surface search itself as the primary bottleneck.
- Trial 996's recovered Optuna params include `think_harder.token_budget_min=3808` and `think_harder.token_budget_max=3054`; this inverted budget pair was replayed exactly and should be treated as a historical anomaly, not a sane candidate to promote.

## Non-Numeric Frontier Trials

Trials 998 and 1005 were `seed_batch(n_questions=16)` actions, not pure runtime configs. Replaying them would mutate the seed/question pool and is not deterministic in the same sense as the numeric config replays. They remain journal-authoritative unless the operator explicitly wants a controlled mutating seeding replay.

## Failed Instrument Attempt

An earlier local attempt ran through a current-protocol path with W6/tool-sentinel overlays and produced all-error zero-token rows. That attempt is not decision evidence for frontier performance; it is useful only as an instrument-failure breadcrumb and is summarized here rather than committed as a replay artifact.
