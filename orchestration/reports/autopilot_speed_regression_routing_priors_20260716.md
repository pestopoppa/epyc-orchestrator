# AutoPilot Speed Regression - Routing and Model-Speed Audit

Date: 2026-07-16

## Summary

The July 16 AutoPilot aggregate speed drop is not explained by a raw CPU kernel
regression. Production v6 raw CPU speed is stable against the v7 candidate when
the v7 HIP build is forced into a CPU-comparable mode. The actionable regression
was in orchestration policy and priors:

- `xmas_routing.mode` was live in `enforce`.
- The X-MAS winner table is heavily `worker_general` biased.
- The live routing/scoring priors still treated `worker_general` aliases as
  `60.7 t/s`, which is stale versus the June 28 v6/iQK matched artifact and the
  July 16 direct probes.

## Measurements

Direct live server probes, 512 generated tokens, single request per role:

| Path | Result |
|------|--------|
| `worker_full_8072` | `32.15 t/s` |
| `worker_q0_8082` | `17.59 t/s` |
| `worker_q1_8182` | `18.63 t/s` |
| `frontdoor_full_8070` | `24.80 t/s` |

Canonical raw v6 P-BENCH-1 probes:

| Binary | Model | Mode | Result |
|--------|-------|------|--------|
| production v6 `llama-bench` | `gemma-4-26B-A4B-it-ORIG-Q4_K_M` | CPU `tg512`, `GGML_IQK=1` | `23.33 +/- 0.04 t/s` |
| production v6 `llama-bench` | `Qwen3.6-35B-A3B-MTP-Q8_0` | CPU `tg512`, `GGML_IQK=1` | `15.95 +/- 0.09 t/s` |

Existing June 28 worker artifact:

| Artifact | Result |
|----------|--------|
| `worker_general_v6_iqk_parity_20260628_full_port_matched206.json` | `38.46 t/s` with `GGML_IQK=1`; `27.78 t/s` with `GGML_IQK=0` |

V7 candidate observations:

| Binary | Mode | Result | Interpretation |
|--------|------|--------|----------------|
| `llama.cpp-experimental/build-hip/bin/llama-bench` | ROCm default, worker | `85.97 +/- 0.10 t/s` | GPU observation only; not CPU comparable |
| same | `-ngl 0`, ROCm visible, worker | `11.88 +/- 0.23 t/s` | Invalid CPU comparison; HIP backend selected with zero GPU layers |
| same | ROCm hidden, worker CPU path | `23.67 +/- 0.28 t/s` | CPU-comparable; matches v6 raw worker |
| same | ROCm hidden, frontdoor CPU path | `16.24 +/- 0.15 t/s` | CPU-comparable; matches v6 raw frontdoor |

## Routing Evidence

Routing log counts from `logs/progress/*.jsonl`:

| Date | Total | Main route distribution | X-MAS applied |
|------|-------|-------------------------|---------------|
| 2026-06-28 | 4500 | `frontdoor=2522`, `worker_general=204`, `architect_general=220` | none |
| 2026-07-04 | 3885 | `frontdoor=1081`, `worker_general=915`, `architect_general=650` | `499`, mostly worker |
| 2026-07-16 | 2024 | `worker_general=1154`, `frontdoor=263`, `architect_general=99` | `285`, all worker |

On 2026-07-16, X-MAS suggested `worker_general` for `2019/2024` routing
decisions and explicitly applied `285` worker overrides. The learned router was
also already worker-heavy (`831` learned worker routes), so both the explicit
X-MAS policy and stale throughput priors were contributing.

## Changes Applied

- Rolled `orchestration/classifier_config.yaml` X-MAS mode from `enforce` to
  `shadow`.
- Recalibrated `worker_general`, `worker_math`, and `toolrunner` throughput
  priors from `60.7` to `38.46` in the lean registry, research registry,
  generated model descriptors, generated stack priors, q-scorer fallback, and
  seeding reward fallback.
- Regenerated:
  - `orchestration/model_descriptors.yaml`
  - `orchestration/derived/stack_priors.yaml`
- Reloaded only the orchestrator API. AutoPilot remained stopped.

## Verification

- Runtime prior probe:
  - descriptor q-scorer priors: worker aliases `38.46`
  - registry q-scorer priors: worker aliases `38.46`
  - stack prior throughput: worker aliases `38.46`
  - seeding degraded fallback: worker aliases `38.46`
- X-MAS config loader reports `mode='shadow'`; metadata still records suggested
  roles but does not enforce them.
- Focused tests:
  - `.venv/bin/python -m pytest tests/classifiers/test_xmas_routing.py tests/unit/test_q_scorer.py tests/unit/test_seeding_rewards.py -q`
  - Result: `122 passed`
- API reload:
  - `.venv/bin/python scripts/server/orchestrator_stack.py reload orchestrator`
  - Health: `/health` OK, six model backends probed successfully.

## Remaining Work

- Run a clean AutoPilot evidence window with `xmas_routing` in shadow and the
  refreshed worker prior.
- Recompute the frontier/replay route distribution with fixed denominators and
  split by `learned`, `xmas_enforce:*`, `rules`, and `forced`.
- Re-enable X-MAS only after a fresh, larger function-axis table validates both
  quality and current serving latency.
- Treat v7 promotion as still blocked on full production-shaped validation, even
  though the raw CPU checks did not show a v7 CPU regression.
