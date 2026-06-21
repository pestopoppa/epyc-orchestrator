# Economic Ledger

The economic ledger is a read-only weekly report over existing logs. It tracks
planner cloud spend, local eval wall time, and decision-throughput proxies so
the lab can review cost per research decision instead of optimizing only local
quality and speed.

## Standing Rules

Rules live in `orchestration/economic_rules.yaml`.

- Planner monthly spend: if weekly planner spend projects above
  `planner_monthly_spend_threshold_usd`, raise F3-W3a planner-distill priority
  for operator review. This does not bypass F3-W3a hardware/training gates.
- Operator gate latency: if canonical median gate latency exceeds
  `operator_gate_latency_threshold_days`, invest in the decision-queue surface.
  Current progress markers are proxy-only, so this rule is not evaluated until
  halt/gate/resume events are captured as structured data.

Ledger reviews are decision support only. They do not mutate handoff priority,
start cloud jobs, or promote autonomous behavior.

## Monthly Review Log

- 2026-06 first review:
  `orchestration/reports/economic_review_2026-06.md`. The 2026-06-06 weekly
  ledger triggered the planner-spend rule (`$410.75` projected monthly spend
  versus `$250.00` threshold), while the 2026-06-21 live trailing digest had
  returned to hold (`$192.55 / $250.00`). Outcome: preserve F3-W3a
  planner-distill as economically justified for operator review, but keep it
  HW-gated and do not bypass F3 acceptance gates.
