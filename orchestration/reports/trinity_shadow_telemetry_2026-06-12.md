# Trinity Shadow Telemetry Report

## Summary

- log files scanned: 1
- routing_decision rows: 918
- rows with assigned_role: 742 (80.8%)
- rows missing assigned_role: 176
- first role timestamp: 2026-06-12T17:25:43.583775+00:00
- last role timestamp: 2026-06-12T22:23:18.349365+00:00
- observed role-bearing span: 0.207 days
- malformed rows skipped: 0

## Role Distribution

  - `worker`: 608 (81.9%)
  - `thinker`: 104 (14.0%)
  - `verifier`: 30 (4.0%)

## Strategy Distribution

  - `learned`: 523 (57.0%)
  - `forced`: 326 (35.5%)
  - `rules`: 28 (3.1%)
  - `mock`: 24 (2.6%)
  - `vision_input`: 17 (1.9%)

## Decision Source Distribution

  - `learned`: 838 (91.3%)
  - `rules`: 47 (5.1%)
  - `mock`: 24 (2.6%)
  - `forced`: 8 (0.9%)
  - `vision_input`: 1 (0.1%)

## TR-3.3 / TR-3.4 Verdict

- TR-3.3 collection window: PENDING — observed span 0.207d < required 7.0d
- TR-3.4 non-degenerate distribution: PASS

Interpretation: telemetry persistence is working once `assigned_role` appears in progress rows. Do not promote TR-4/5 until TR-3.3 has a clean production-like window and the distribution remains non-degenerate.
