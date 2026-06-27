# P4.6 Role Dropout A/B

Decision: **keep_current_hard_label_training**

| rate | runs | hard RSA | dropout RSA | delta | std delta | best delta | adopt runs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 5 | 0.5296 | 0.5222 | -0.0074 | 0.0108 | +0.0000 | 0 |
| 0.3 | 5 | 0.5296 | 0.5222 | -0.0074 | 0.0108 | +0.0000 | 0 |

## Interpretation

- Role dropout is not promotion-grade under the current P4.5 decision metric.
- Keep current hard-label training for production/staged classifier weights.
- Treat this as a null robustness experiment unless a future LRC architecture exposes available-role masks as model inputs.
