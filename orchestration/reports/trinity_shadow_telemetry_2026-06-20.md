# Trinity Shadow Telemetry Report

## Summary

- log files scanned: 7
- routing_decision rows: 10,868
- rows with assigned_role: 10,686 (98.3%)
- rows missing assigned_role: 182
- first role timestamp: 2026-06-12T17:25:43.583775+00:00
- last role timestamp: 2026-06-20T13:09:05.887979+00:00
- observed role-bearing span: 7.822 days
- malformed rows skipped: 0

## Role Distribution

  - `worker`: 9,296 (87.0%)
  - `thinker`: 1,116 (10.4%)
  - `verifier`: 274 (2.6%)

## Strategy Distribution

  - `learned`: 5,694 (52.4%)
  - `rules`: 2,896 (26.6%)
  - `forced`: 1,771 (16.3%)
  - `vision_input`: 421 (3.9%)
  - `xmas_enforce:learned`: 53 (0.5%)
  - `mock`: 33 (0.3%)

## Decision Source Distribution

  - `learned`: 7,048 (64.9%)
  - `rules`: 3,703 (34.1%)
  - `forced`: 83 (0.8%)
  - `mock`: 27 (0.2%)
  - `unknown`: 6 (0.1%)
  - `vision_input`: 1 (0.0%)

## TR-3.3 / TR-3.4 Verdict

- TR-3.3 collection window: PASS
- TR-3.4 non-degenerate distribution: PASS

Interpretation: telemetry persistence is working once `assigned_role` appears in progress rows. Do not promote TR-4/5 until TR-3.3 has a clean production-like window and the distribution remains non-degenerate.
