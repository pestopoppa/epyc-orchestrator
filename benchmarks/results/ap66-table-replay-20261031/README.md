# AP-66 offline table replay — finite-table repair and limitations

## Corrected replay

The corrected `replay_exact.py` wraps each sampler with a finite-table repair layer before Optuna records the parameters. All 16 parameters remain categorical and each of the 4,000 stored parameter vectors exactly matches the row that supplied its result; `verify.stdout` checks the table join, source and writer hashes, all native/unknown values, and the same 20 Sobol-initialized row identities across four arms for each seed. Sobol is one-dimensional over table row indices; it does not establish 16-dimensional space-filling. The repair layer constrains the samplers' behavior and its rate and normalized distance are recorded per trial. Hypervolume and constraint outcomes refer only to this deterministic finite table and rule, with no holdout or prospective inference. No production optimizer advantage or acceptance conclusion is claimed.

| Arm | Calls | Mean rule-infeasible share | Mean repair rate | Mean hypervolume |
|---|---:|---:|---:|---:|
| NSGA-II FAIL | 50 | 0.828 | 0.600 | 54,865.521 |
| NSGA-II raw | 50 | 0.828 | 0.600 | 54,865.521 |
| NSGA-II bounded | 50 | 0.828 | 0.600 | 54,865.521 |
| qLogNEHVI | 50 | 0.366 | 0.268 | 41,912.119 |
| NSGA-II FAIL | 100 | 0.782 | 0.800 | 60,742.035 |
| NSGA-II raw | 100 | 0.754 | 0.784 | 60,202.704 |
| NSGA-II bounded | 100 | 0.754 | 0.784 | 60,202.704 |
| qLogNEHVI | 100 | 0.191 | 0.371 | 41,912.119 |

The replay writer invokes `NSGAIISampler` without a population-size argument; the installed Optuna 4.8.0 `samplers/nsgaii/_sampler.py:159` declares the default `population_size=50` (file SHA-256 `1f7b483e7b08791a531abf928a9e577c012e958ff1ef13375d049e6a80dcad61`). Equal 50-call results are consistent with a cold first generation; the observed equality is not proof of its cause. The two constrained NSGA encodings have the same ordering in this one-constraint replay. At 50 calls their mean infeasible share equals FAIL; at 100 it is lower in this table. These values do not establish an optimizer gain outside this repaired lookup. The exact writer SHA-256 is `836cd8db637d514873ef96232fad4a09349041359efa3169480d9f81751ae32d`; its original absolute scratch input path is retained so changing that path changes the bound writer. GP Cholesky-jitter warnings and the pure-Python acquisition fallback are captured in `exact/` logs. The native R6 supervisors and 40 complete result records are also in `exact/`.

The private source candidate under test was `b4ad337e7dd81253f502a42f4e1ea88fe4742352`, with 112 pinned, import-audited tests passing in `validation-v10.stdout` and targeted Ruff passing in `ruff-v10.stdout`. Both include the R6 whole-process-tree supervisor receipt. These are private review artifacts; no publication or live reload occurred.

## Earlier exploratory replay


Input: LLMSYS-HPOBench, [Zenodo 20048594](https://zenodo.org/records/20048594), DOI 10.5281/zenodo.20048594, CC-BY-4.0; creator LLMSYS-HPOBench, published 2026-05-06. The included CSV is `experiment-data/Engine/SGLang/15.0-1.0-16-16-1024/15.0-1.0-16-16-1024.csv`, extracted by HTTP range from the archive. Archive MD5 is `cd0be3c8152845c4c9a2411726d95786`; the extracted CSV's ZIP CRC32 is `9a83355d` and SHA-256 is `6a5020a7c9536148f12f182c6558a95998de0f49e77760928f7984cb3b398b63`. No benchmark software was imported.

All arms use the same per-seed 20 Sobol-selected table indices, 10 seeds, with results at 50 and 100 calls. Objectives are native request throughput (requests/s, higher is better) and negative p99 end-to-end latency (ms, higher is better). Hypervolume uses the two native worst-observed feasible values as reference, with units requests/s × ms. Per the task's synthetic replay rule, context length at least 9216 is called infeasible. This label is a rule imposed for this replay, not a diagnosis of OOM. The 416 rows with that context have unavailable native objectives, retained as null; 55 other rows have measurements.

The 16 independent categorical value sets produce combinations not present in this sparse table. The writer projects each proposal to the nearest observed row, recording the requested values, row ID, and RMS per-knob distance using min-max normalized numeric distances and categorical mismatch. Therefore these are *projected lookup* results; they do not satisfy the stated exact-table-hit acceptance, and do not support a claim of real optimizer advantage or a prospective performance effect. No holdout data or new inference was used. Identical seeds, source hashes, all projected proposals, native measured values and nulls, outcome rule, objective units, and intermediate 50/100 snapshots are recorded in each JSON. `replay_ap66.py` is the exact executed writer and uses its original scratch path for the CSV; recreate that path or adjust it and treat the resulting code as a different source.

| Arm | Calls | Mean rule-infeasible share | Mean snap rate | Mean hypervolume |
|---|---:|---:|---:|---:|
| NSGA-II FAIL | 50 | 0.750 | 0.600 | 55,470.914 |
| NSGA-II raw | 50 | 0.750 | 0.600 | 55,470.914 |
| NSGA-II bounded | 50 | 0.750 | 0.600 | 55,470.914 |
| qLogNEHVI | 50 | 0.414 | 0.406 | 41,912.119 |
| NSGA-II FAIL | 100 | 0.707 | 0.800 | 61,728.840 |
| NSGA-II raw | 100 | 0.702 | 0.800 | 61,592.064 |
| NSGA-II bounded | 100 | 0.702 | 0.800 | 61,592.064 |
| qLogNEHVI | 100 | 0.258 | 0.548 | 45,138.106 |

These summaries describe only this replay; they are not a pass determination. The BoTorch GP emitted numerical Cholesky-jitter warnings and used the pure-Python qLogEHVI kernel fallback. Software: Optuna 4.8.0; optuna-integration 4.8.0; BoTorch 0.18.1; GPyTorch 1.15.2; CPU PyTorch 2.15.0. The two BoTorch supervisor runs sampled peak owned-process-tree RSS of 575,217,664 and 577,073,152 bytes; the 30-run NSGA batch peaked at 157,728,768 bytes. Sampling may miss transients. No live autopilot, inference endpoint, GPU, production source or service was touched.
