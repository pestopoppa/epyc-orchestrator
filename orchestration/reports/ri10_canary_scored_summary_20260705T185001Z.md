# RI-10 Canary Scored Response Report

- Schema: `ri10_canary_scored_response_report.v1`
- Status: `ready`
- Rows: `60`
- F1 threshold: `0.8`
- Arm comparison: `ready`

## Buckets

| Bucket | Rows | Scored | Missing | Correct | Accuracy | Mean Token F1 |
|---|---:|---:|---:|---:|---:|---:|
| `arm:enforce` | 30 | 30 | 0 | 3 | 0.1000 | 0.0638 |
| `arm:shadow` | 30 | 30 | 0 | 3 | 0.1000 | 0.0633 |
| `overall` | 60 | 60 | 0 | 6 | 0.1000 | 0.0636 |
| `role:frontdoor` | 20 | 20 | 0 | 4 | 0.2000 | 0.0286 |
| `role:worker_general` | 20 | 20 | 0 | 2 | 0.1000 | 0.1173 |
| `role:worker_vision` | 20 | 20 | 0 | 0 | 0.0000 | 0.0448 |
| `role_arm:frontdoor:enforce` | 10 | 10 | 0 | 2 | 0.2000 | 0.0286 |
| `role_arm:frontdoor:shadow` | 10 | 10 | 0 | 2 | 0.2000 | 0.0286 |
| `role_arm:worker_general:enforce` | 10 | 10 | 0 | 1 | 0.1000 | 0.1173 |
| `role_arm:worker_general:shadow` | 10 | 10 | 0 | 1 | 0.1000 | 0.1173 |
| `role_arm:worker_vision:enforce` | 10 | 10 | 0 | 0 | 0.0000 | 0.0456 |
| `role_arm:worker_vision:shadow` | 10 | 10 | 0 | 0 | 0.0000 | 0.0440 |
