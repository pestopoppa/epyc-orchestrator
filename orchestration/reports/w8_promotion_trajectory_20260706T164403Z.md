# AutoPilot W8 Promotion Trajectory Report

- Status: progressing
- Latest trial: 1232
- Evidence: snapshots=240, candidates=71, status_counts={'active_recent_replay': 2, 'refuted': 3, 'reverted': 58, 'single_observation': 7, 'stale_accumulating': 1}
- Replay policy: max_attempts=12, stale_trials=12
- Replay eligibility: eligible=['80aa44d93a242af5', '289c4fc0fb5a334d', 'd3f28243801548b2'], recent=['80aa44d93a242af5', '289c4fc0fb5a334d'], blocked=68

## Replay Concentration

- active_recent=2, stale_accumulating=1, single_observation=7, top_active_share=0.5, warning=False

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
| 80aa44d93a242af5 | active_recent_replay | 1231,1232 | 0.91 / 100 | accumulating | 2 | 0 | eligible |
| 289c4fc0fb5a334d | active_recent_replay | 1208,1220 | 0.921034 / 100 | accumulating | 2 | 0 | eligible |
| 67ea799d4870b3c7 | reverted | 1228,1229 | 0.708802 / 100 | accumulating | 2 | 0 | AP-24=revert |
| 4b6b454ea4f884fd | reverted | 1168,1169,...,1226,1227 | 0.91 / 100 | accumulating | 4 | 0 | AP-24=revert |
| b738287be98c3372 | reverted | 1132,1133,...,1207,1225 | 0.627218 / 100 | refuted | 8 | 0 | state=refuted |
| d3f28243801548b2 | stale_accumulating | 1209,1219 | 0.921806 / 100 | accumulating | 2 | 0 | eligible |
| 50b4adcde05d3584 | reverted | 1212,1218 | 0.924686 / 100 | accumulating | 2 | 0 | AP-24=revert |
| c21324afa4f61dfc | reverted | 1197,1217 | 0.942984 / 100 | accumulating | 2 | 0 | AP-24=revert |
| fbcdd046d27b4ba1 | reverted | 1211 | 0.926282 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 86b1c9af0038c6db | single_observation | 1156,1210 | 0.91 / 100 | accumulating | 2 | 0 | unreplayable_action=deep_eval |
| 26af68f08590e7ae | single_observation | 1204 | 0.932934 / 100 | accumulating | 1 | 0 | E_quality_below_replay_floor |
| 648f3c6625118043 | reverted | 1203 | 0.946716 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 05e211df220202c8 | reverted | 1200 | 0.926728 / 100 | accumulating | 1 | 0 | AP-24=revert |
| 6d18a71f46f47821 | reverted | 1196 | 0.938412 / 100 | accumulating | 1 | 0 | AP-24=revert |
| f3757bb20213478d | reverted | 1194 | 0.936074 / 100 | accumulating | 1 | 0 | AP-24=revert |
| ca6b08a2ee281708 | reverted | 1191,1192 | 0.930149 / 100 | accumulating | 2 | 0 | AP-24=revert |
| 73dbd3be67ee5757 | reverted | 1187 | 0.96073 / 100 | accumulating | 1 | 0 | AP-24=revert |
| edfe40b648ee8b26 | reverted | 913,1102,1165 | 0.91 / 100 | accumulating | 3 | 0 | AP-24=revert |
| ce82d13e9fe48c1e | reverted | 1163,1164 | 0.91 / 100 | accumulating | 2 | 0 | AP-24=revert |
| 9defa67b5fd62398 | reverted | 931,1103,1157,1158 | 0.617791 / 100 | accumulating | 4 | 0 | AP-24=revert |
| bbf5a966fa036a00 | reverted | 1123,1128,1154 | 0.952707 / 100 | accumulating | 3 | 0 | AP-24=revert |
| 45129cc6ee5bac29 | reverted | 836,896,...,1063,1152 | 0.986055 / 100 | refuted | 12 | 0 | state=refuted |
| 98dc0e060ab1deac | reverted | 1147,1148,1149,1150,1151 | 0.947361 / 100 | accumulating | 5 | 0 | AP-24=revert |
| a5dd4182e654c21e | reverted | 932,1138,1141 | 0.91 / 100 | accumulating | 3 | 0 | AP-24=revert |
| 8291f25da67ace2e | reverted | 939,1139,1140 | 0.639131 / 100 | accumulating | 3 | 0 | AP-24=revert |

_Only the latest 25 of 71 candidates are shown._
