# AutoPilot W8 Promotion Trajectory Report

- Status: progressing
- Latest trial: 1095
- Evidence: snapshots=151, candidates=39, status_counts={'active_recent_replay': 2, 'refuted': 5, 'single_observation': 6, 'stale_accumulating': 26}
- Replay policy: max_attempts=12, stale_trials=12

## Open Requirements

- combined_E_below_required
- fresh_promotion_eval_required
- stale_accumulating_candidates_present
- seq_confirmation_required

## Candidate Trajectories

| Candidate | Status | Trials | latest E / required | latest state | k | fresh evals |
|---|---:|---:|---:|---:|---:|---:|
| 1bcc4e9c9920654f | active_recent_replay | 1082,1088,1089,1094,1095 | 0.915816 / 100 | accumulating | 5 | 0 |
| 968b0a9da524bdbe | active_recent_replay | 1085,1086 | 0.932744 / 100 | accumulating | 2 | 0 |
| ebce36c6f74f1c42 | single_observation | 1093 | 0.936907 / 100 | accumulating | 1 | 0 |
| fd831f6b5166db64 | single_observation | 1092 | 0.921454 / 100 | accumulating | 1 | 0 |
| fe97a9439d7e6415 | single_observation | 1091 | 0.933314 / 100 | accumulating | 1 | 0 |
| 659036bed3693757 | single_observation | 1090 | 0.928916 / 100 | accumulating | 1 | 0 |
| 830b0d0b1a7d8239 | single_observation | 1087 | 0.931927 / 100 | accumulating | 1 | 0 |
| ea4334a6c3ddbadc | single_observation | 1083 | 0.935508 / 100 | accumulating | 1 | 0 |
| c92f1da651931104 | stale_accumulating | 1081 | 0.928162 / 100 | accumulating | 1 | 0 |
| 3ede0c44966d6ba3 | stale_accumulating | 909,1080 | 0.915002 / 100 | accumulating | 2 | 0 |
| 58908d2526326e8e | stale_accumulating | 1077 | 0.938485 / 100 | accumulating | 1 | 0 |
| 89489ee206cba50a | stale_accumulating | 1075 | 0.929311 / 100 | accumulating | 1 | 0 |
| cfae6c2537eb9ce1 | stale_accumulating | 1072,1073,1074 | 0.965301 / 100 | accumulating | 3 | 0 |
| d4d10500f99d6bff | stale_accumulating | 1071 | 0.918745 / 100 | accumulating | 1 | 0 |
| 96d3df9f1ef62ddb | stale_accumulating | 1069 | 0.949047 / 100 | accumulating | 1 | 0 |
| 4d98889ebee78449 | stale_accumulating | 1068 | 0.936955 / 100 | accumulating | 1 | 0 |
| 70902e4b665474e7 | refuted | 888,897,...,1065,1067 | 0.556099 / 100 | refuted | 40 | 0 |
| d0f586ba6972353e | stale_accumulating | 1066 | 0.949578 / 100 | accumulating | 1 | 0 |
| 45129cc6ee5bac29 | refuted | 836,896,...,1062,1063 | 0.986055 / 100 | refuted | 11 | 0 |
| d7b907d936a0bf1e | stale_accumulating | 1061 | 0.945287 / 100 | accumulating | 1 | 0 |
| 01a315fc829252a9 | stale_accumulating | 1060 | 0.944731 / 100 | accumulating | 1 | 0 |
| f602b645746a4348 | stale_accumulating | 1059 | 0.916172 / 100 | accumulating | 1 | 0 |
| 79f1a49802521120 | stale_accumulating | 1058 | 0.924814 / 100 | accumulating | 1 | 0 |
| 1b8e8c5e4edb3ee1 | stale_accumulating | 1045,1050,1055,1056 | 0.87391 / 100 | accumulating | 4 | 0 |
| 5460a224cc6e4fa9 | stale_accumulating | 1034,1044,1054 | 0.917071 / 100 | accumulating | 3 | 0 |

_Only the latest 25 of 39 candidates are shown._
