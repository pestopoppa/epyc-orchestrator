# DS-7 Profile Decision

Generated: 2026-07-04T19:40:20Z

Decision: retain `default` as the steady-state/static-prewarm production profile.

## Evidence

Source packet: `orchestration/reports/ds_e1_evidence_packet_20260704T192333Z.{json,md}`

- DS-E1 ready for profile decision: true
- Stack roster: ready; 10 live roles from generated stack priors compiled at `2026-07-04T19:16:46Z`
- DS-5 manifest: ready; manifest and stack-prior compile timestamps both `2026-07-04T19:16:46Z`
- Contention matrix: ready; full topology `5d19b3e4edf6fc27`, measured contention topology `df373c79cc4af06f`, matrix status `ok`
- RI-10 canary: ready; 80 evaluable high-risk rows since telemetry health start, enforce/shadow counts `31/49`, zero sample or balance deficits
- KV-size measurements: ready; all required production role/context rows observed, 10 parsed rows, no failed rows

## Profile

`stack_templates/default.yaml` now records:

- `metadata.ds7_profile: steady_state_static_prewarm`
- `metadata.ds7_decision.status: retain_default`
- `metadata.ds7_decision.ds6_quarter_scheduler: parked_until_static_prewarm_gap`

Validation command:

```bash
python3 scripts/server/orchestrator_stack.py start --stack-profile default --validate-only
```

Observed result:

- status: ok
- roles: 17
- instances: 28
- RAM estimate: 657 GB

## DS-6 Disposition

Keep QuarterScheduler parked. The current DS-E1 packet proves the decision inputs are present, but does not show a material throughput or latency gap caused by static pre-warm. Reopen DS-6 only if future DS-E1-equivalent evidence shows static pre-warm leaves material throughput or latency on the table.

This artifact records a profile decision, not a live stack change.
