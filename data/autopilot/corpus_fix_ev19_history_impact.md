# EV-19 corpus fix: effect on journaled AutoPilot results (2026-10-21)

History is NOT rescored or altered. Source: `orchestration/autopilot_journal.jsonl` + `_1.jsonl`
(1,390 trials, 2026-05-26..2026-08-09), trials whose `eval_details.per_suite_quality` contains the suite.
Pool revision those results were scored on: the live pool built 2026-07-27 (before `corpus-ev19-20261021`).
Quality scale in the journal is 0-3 per suite.

| suite | trials | date range | what the defect did to those scores |
|---|---|---|---|
| mmlu_pro | 40 | 2026-07-26..2026-08-09 | All predate CJ-8 (2026-09-07), when gold I/J rows were silently scored False. 17.1% of rows (2,053/12,032) could never pass: scores deflated, ceiling ~0.83 of the real value. |
| livecodebench | 912 | 2026-05-27..2026-08-09 | Substring oracle `def ` passed any Python answer (`def solve(): pass` passed 100%). Scores inflated toward the cap (764/912 trials at 3.0). Not a measure of code ability. |
| real_suite_v1 | 361 | 2026-06-28..2026-08-09 | 3/50 rows (0006, 0007, 0034) vacuous pass: at most 6% of the suite, a small upward bias. |
| tulving_episodic | 43 | 2026-07-26..2026-08-09 | NOT defective. Empty-gold rows are genuine controls and score true on "None". Historical scores stand. |

Also outside the journal: the PRB-T4 TALE run (research b4d38ebc) excluded mmlu_pro (ScoringUnavailable).
Other result stores (benchmarks/results, eval-tower captures) were not enumerated.

## Addendum: corpus-ev19b-20261021 (same day, same era E20)

22 `thinking` ARC rows (source labels 1-4, e.g. `arc_NYSEDREGENTS_2008_8_34`) were unscoreable
(gold with no configured choices; the scorer refuses, so they were EXCLUDED, never scored wrong).
Re-derived in `corpus-ev19b-20261021` (relabelled A-D, gold mapped by source position). Effect on
journaled results: at most 22 of 11,214 `thinking` rows (0.2%) were missing from the denominator;
no score was deflated or inflated by them. Historical scores stand. Pool sha
`73a2ed5aa87c38e554c818aff4b8c6dd865621c6134b4abacebfa9070a818f8b` (prior ev19 pool `f8b53663...`).
