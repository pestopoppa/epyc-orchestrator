# Trinity Shadow Telemetry Report

## Summary

- log files scanned: 6
- routing_decision rows: 8,861
- rows with assigned_role: 8,685 (98.0%)
- rows missing assigned_role: 176
- first role timestamp: 2026-06-12T17:25:43.583775+00:00
- last role timestamp: 2026-06-19T02:05:02.845458+00:00
- observed role-bearing span: 6.361 days
- malformed rows skipped: 0

## Role Distribution

  - `worker`: 7,568 (87.1%)
  - `thinker`: 892 (10.3%)
  - `verifier`: 225 (2.6%)

## Strategy Distribution

  - `learned`: 5,486 (61.9%)
  - `rules`: 1,701 (19.2%)
  - `forced`: 1,246 (14.1%)
  - `vision_input`: 351 (4.0%)
  - `xmas_enforce:learned`: 53 (0.6%)
  - `mock`: 24 (0.3%)

## Decision Source Distribution

  - `learned`: 6,822 (77.0%)
  - `rules`: 1,953 (22.0%)
  - `forced`: 61 (0.7%)
  - `mock`: 24 (0.3%)
  - `vision_input`: 1 (0.0%)

## TR-3.3 / TR-3.4 Verdict

- TR-3.3 collection window: PENDING — observed span 6.361d < required 7.0d
- TR-3.4 non-degenerate distribution: PASS

Interpretation: telemetry persistence is working once `assigned_role` appears in progress rows. Do not promote TR-4/5 until TR-3.3 has a clean production-like window and the distribution remains non-degenerate.
