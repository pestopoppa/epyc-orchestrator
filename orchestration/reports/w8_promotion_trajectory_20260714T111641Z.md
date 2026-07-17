# AutoPilot W8 Promotion Trajectory Report

- Status: progressing
- Latest trial: 1335
- Evidence: snapshots=312, candidates=108, status_counts={'active_recent_replay': 4, 'refuted': 4, 'reverted': 79, 'single_observation': 19, 'stale_accumulating': 2}
- Replay policy: max_attempts=12, stale_trials=12
- Replay eligibility: eligible=['5c7d629d24291a05', '5dea51f15dcb5f75', 'd522d0f44587142f', '289c4fc0fb5a334d', 'd3f28243801548b2'], recent=['5c7d629d24291a05', '5dea51f15dcb5f75', 'd522d0f44587142f'], blocked=103

## Replay Concentration

- active_recent=3, stale_accumulating=2, single_observation=19, top_active_share=0.333333, warning=False

## Terminal Candidate Reasons

- dominant: Suite 'general' regression: -1.800 (threshold: -1.500; n_result=5, n_baseline=2) (26 candidate(s), status=reverted)
- warning: dominant per-suite regression compares against a very small baseline sample

## Open Requirements

- combined_E_below_required
- fresh_promotion_eval_required
- stale_accumulating_candidates_present
- seq_confirmation_required

## Candidate Trajectories

| Candidate | Status | Trials | latest E / required | latest state | k | fresh evals | Replay |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5c7d629d24291a05 | active_recent_replay | 1333,1334 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| 28c8732694945a90 | active_recent_replay | 1330,1331 | 0.91 / 100 | accumulating | 2 | 0 | E_quality_below_replay_floor |
| 5dea51f15dcb5f75 | active_recent_replay | 1328,1329 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| d522d0f44587142f | active_recent_replay | 1323,1324 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| 84a55fd82af865d8 | single_observation | 1335 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| b0e92a1feb661bdf | single_observation | 1332 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| dcd57132e0ab7954 | single_observation | 1327 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| ed8e891e53bd53c5 | single_observation | 1326 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| b7c431794aec6a00 | single_observation | 1322 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| 56f578e2277fab92 | single_observation | 1321 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| 4289ed22147f876b | reverted | 1318,1319,1320 | 0.91 / 100 | accumulating | 3 | 0 | AP-24=revert |
| 3055f1e32fac0316 | reverted | 1305,1315,1316,1317 | 0.91 / 100 | accumulating | 4 | 0 | AP-24=revert |
| efe3c46c7dc3355d | single_observation | 1312 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| d0146d3359d69d91 | single_observation | 1310,1311 | 0.91 / 100 | accumulating | 2 | 0 | E_quality_below_replay_floor |
| 27037f575e24e8e4 | single_observation | 1307,1308,1309 | 0.91 / 100 | accumulating | 3 | 0 | E_quality_below_replay_floor |
| a7ad1966c4f34834 | reverted | 1281,1282,1303 | 0.922206 / 100 | accumulating | 3 | 0 | AP-24=revert |
| ce82d13e9fe48c1e | reverted | 1163,1164,1302 | 0.594835 / 100 | accumulating | 2 | 0 | AP-24=revert |
| 699e86e5c5710e24 | reverted | 1301 | 0.978678 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 44e13d26840a145e | reverted | 1300 | 0.981834 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 5ce6d43ece01f92e | single_observation | 1299 | 0.988317 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| 477ed0d154576baa | reverted | 1298 | 0.984986 / 100 | accumulating | 1 | 0 | AP-24=revert |
| fef82d188a38c3ee | reverted | 1297 | 0.97823 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 8b7afe4f740b89b5 | reverted | 1296 | 0.95384 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 5a67fefef6a4c8c7 | reverted | 1295 | 0.972571 / 100 | accumulating | 1 | 0 | AP-24=revert |
| b738287be98c3372 | reverted | 1132,1133,...,1266,1294 | 0.907896 / 100 | refuted | 11 | 0 | state=refuted |

_Only the latest 25 of 108 candidates are shown._
