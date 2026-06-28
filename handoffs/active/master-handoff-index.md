# EPYC Orchestrator Handoff Index

**Purpose**: local coordination for orchestrator-scoped work. Production
authority, cross-repo priority, and operator gates remain owned by
`/mnt/raid0/llm/epyc-root/handoffs/active/master-handoff-index.md`.

## Prioritized Task List

- [ ] **P0: W4/W6 evidence-plane cutover support**. Keep AutoPilot evidence
  collection observable and do not enable authority until
  `scripts/autopilot/restart_readiness_report.py --json --strict
  --require-seq-cutover --require-w6-audit` passes.
- [ ] **P0: v6 NumericSwarm frontier rerun support**. Preserve
  `orchestration/autopilot_state.json.active_instrument_eras` and the current
  `frontier_rerun_required` marker until the expanded surface campaign clears.
- [ ] **P1: X-MAS production-path evidence**. Run repaired
  `incumbent_constrained_cheapfirst_v2` held-out A/B only in an attested quiet
  window; keep `xmas_routing.mode` default-off unless the acceptance report
  promotes it.
- [ ] **P1: DS-E1 dynamic-stack KV evidence**. Run production KV-size
  measurements only in a coordinated clean window; update the evidence packet
  before any profile decision.
- [ ] **P2: Repo-readiness L4 closeout**. Maintain generated docs, health
  checks, security audit, and this task index as real workflow surfaces rather
  than scorer-only placeholders.

## Dependency Graph

- W4/W6 cutover blocks sequential verdict authority, baseline ledger
  authority, and any accept-path gates that depend on the repaired evidence
  plane.
- v6 NumericSwarm frontier rerun blocks trustworthy consolidated
  max-performance guidance for planner numeric surfaces.
- X-MAS enforce depends on a complete winner table plus a passing repaired
  held-out A/B.
- DS-E1 profile decisions depend on direct production KV-size rows.
- Repo-readiness L4 closeout is independent of inference, but changes to
  generated docs or health checks must not perturb active AutoPilot evidence.

## Cross-Cutting Concerns

- Treat active AutoPilot windows as evidence-sensitive. Do not launch extra
  benchmark coordinators into a trial unless the owning runbook explicitly says
  concurrent measurement is allowed.
- Run GitNexus impact before editing production symbols. HIGH/CRITICAL edits
  belong in the main thread with focused tests.
- Keep generated/runtime artifacts out of commits unless the owning handoff
  defines them as durable evidence.
- Stack facts must come from structured registry, attestation, or generated
  artifacts; avoid stale model/port facts in handwritten prompts.

## Reporting Instructions

- Record completed orchestrator work in the owning root handoff and
  `epyc-root/progress/YYYY-MM/YYYY-MM-DD.md`.
- Commit by pathspec in `epyc-orchestrator`, then refresh GitNexus with
  `scripts/gitnexus-analyze.sh`.
- If a task changes acceptance evidence, rerun the relevant report command and
  include the exact blocker/pass summary in the progress entry.

## Key File Locations

- AutoPilot controller: `scripts/autopilot/autopilot.py`
- Strict readiness: `scripts/autopilot/restart_readiness_report.py`
- Phase health: `scripts/autopilot/phase_health_report.py`
- X-MAS reports: `scripts/benchmark/xmas_live_ab.py`
- Stack launch: `scripts/server/orchestrator_stack.py`
- Security audit: `scripts/security/audit_repository.py`
- Session health: `scripts/session/health_check.sh`
