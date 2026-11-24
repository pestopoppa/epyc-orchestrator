# Math re-baseline: deterministic scorer replay

- source report: `orchestration/reports/eval_tower_math_rebaseline_E7c` (not modified)
- scorer_semantics_version: `strict_finals_v2_20261124`
- debug_scorer sha256: `3c110586c260844083acf2351561691602dedc33288a1206948d521ad0bec65c`
- no inference: banked answers re-scored by the live scorer

| label | source | n scorable | banked correct | replayed correct | T->F | F->T |
|---|---|---|---|---|---|---|
| ev11-worker_general | gsm8k | 1319 | 1002 | 1270 | 2 | 270 |
| ev11-worker_general | math500 | 500 | 326 | 367 | 1 | 42 |
| ev11-worker_general | **all** | 1819 | 0.7301 | 0.8999 | 3 | 312 |
| ev11-worker_math | gsm8k | 1319 | 999 | 1261 | 3 | 265 |
| ev11-worker_math | math500 | 500 | 271 | 307 | 0 | 36 |
| ev11-worker_math | **all** | 1819 | 0.6982 | 0.8620 | 3 | 301 |

Accuracy is over rows scorable under BOTH semantics (banked bool and replayed bool).
Per-label replayed outcome counts (including exclusions) are in `summary.json`.
