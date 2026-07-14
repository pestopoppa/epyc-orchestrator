# AutoPilot W8 Promotion Trajectory Report

- Status: progressing
- Latest trial: 1346
- Evidence: snapshots=322, candidates=113, status_counts={'active_recent_replay': 5, 'refuted': 4, 'reverted': 80, 'single_observation': 20, 'stale_accumulating': 4}
- Replay policy: max_attempts=12, stale_trials=12
- Replay eligibility: eligible=['54b4c2f8902e94f1', '6c31f72dd5c3a4ea', 'e3f0fd48f5c3f7a3', '5c7d629d24291a05', '5dea51f15dcb5f75', 'd522d0f44587142f', '289c4fc0fb5a334d', 'd3f28243801548b2'], recent=['54b4c2f8902e94f1', '6c31f72dd5c3a4ea', 'e3f0fd48f5c3f7a3', '5c7d629d24291a05'], blocked=105

## Replay Concentration

- active_recent=4, stale_accumulating=4, single_observation=20, top_active_share=0.25, warning=False

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
| 54b4c2f8902e94f1 | active_recent_replay | 1344,1345 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| 6c31f72dd5c3a4ea | active_recent_replay | 1342,1343 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| 9bcf7285be17fa66 | active_recent_replay | 1338,1341 | 0.91 / 100 | accumulating | 2 | 0 | E_quality_below_replay_floor |
| e3f0fd48f5c3f7a3 | active_recent_replay | 1339,1340 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| 5c7d629d24291a05 | active_recent_replay | 1333,1334 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| 3055f1e32fac0316 | reverted | 1305,1315,1316,1317,1346 | 0.91 / 100 | accumulating | 5 | 0 | AP-24=revert |
| f705023c4653fe28 | reverted | 1337 | 0.91 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 84a55fd82af865d8 | single_observation | 1335 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| b0e92a1feb661bdf | single_observation | 1332 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| 28c8732694945a90 | single_observation | 1330,1331 | 0.91 / 100 | accumulating | 2 | 0 | E_quality_below_replay_floor |
| 5dea51f15dcb5f75 | stale_accumulating | 1328,1329 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| dcd57132e0ab7954 | single_observation | 1327 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| ed8e891e53bd53c5 | single_observation | 1326 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| d522d0f44587142f | stale_accumulating | 1323,1324 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| b7c431794aec6a00 | single_observation | 1322 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| 56f578e2277fab92 | single_observation | 1321 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| 4289ed22147f876b | reverted | 1318,1319,1320 | 0.91 / 100 | accumulating | 3 | 0 | AP-24=revert |
| efe3c46c7dc3355d | single_observation | 1312 | 0.91 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| d0146d3359d69d91 | single_observation | 1310,1311 | 0.91 / 100 | accumulating | 2 | 0 | E_quality_below_replay_floor |
| 27037f575e24e8e4 | single_observation | 1307,1308,1309 | 0.91 / 100 | accumulating | 3 | 0 | E_quality_below_replay_floor |
| a7ad1966c4f34834 | reverted | 1281,1282,1303 | 0.922206 / 100 | accumulating | 3 | 0 | AP-24=revert |
| ce82d13e9fe48c1e | reverted | 1163,1164,1302 | 0.594835 / 100 | accumulating | 2 | 0 | AP-24=revert |
| 699e86e5c5710e24 | reverted | 1301 | 0.978678 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 44e13d26840a145e | reverted | 1300 | 0.981834 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 5ce6d43ece01f92e | single_observation | 1299 | 0.988317 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |

_Only the latest 25 of 113 candidates are shown._
