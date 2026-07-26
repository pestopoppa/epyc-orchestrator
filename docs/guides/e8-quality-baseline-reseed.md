# E8 Quality Baseline Evidence

The E8 evidence runner is evidence-only. It does not change the model stack,
model registries, AutoPilot state, baseline values, or the E8 rebaseline hold.
Do not use `eval_batch_serving_evaltower_window.py` for this work.

## Preparation

1. Keep the current frozen-v8 both-mode lineup unchanged and stop AutoPilot.
2. After the E8 numeric rerun reaches 16 completed trials, have the operator review
   `ratify_e8_quality_baseline_protocol_20260726.sh --plan` and explicitly ratify
   either T2=500 (default) or T2=50. The receipt must remain at the canonical
   operator path and its runner hash must match the executable Python source.
3. Confirm the E8 quality hold is still open and run the read-only preflight:

```bash
cd /mnt/raid0/llm/epyc-orchestrator
.venv/bin/python scripts/benchmark/run_e8_quality_baseline_reseed.py --prepare
```

The report must have no blockers. It checks the E8 hold, numeric-rerun count,
operator receipt, current exact 6/6 endpoint groups,
health, AutoPilot absence, the realized 24-port both-NUMA stack, frozen binary
identity, every port's exact `/proc/<pid>/cmdline` and model flags, the
orchestrator-state model path, canonical stat and content identities for every
distinct executable/model artifact, all three repository heads, and hashes the
runner, selection/scoring sources, question-pool data, state, both registries,
runtime facts, compiled stack contract, and live orchestrator state. This
command makes no model calls.

Before either a receipt can be minted or live evaluation can begin, the runner
also compiles every fixed-vector scorer configuration under the active scorer
contract. The current T2=500 source vector fails this gate: both
`real_suite_v1_0043` and `needle_039` declare the zero-capture-group
`exact_match` pattern `\\d+`. This is intentional fail-closed behavior. Do not
alter the frozen pool, selected denominator, or source rows through this
workflow; an operator-approved source/protocol amendment is required before a
new receipt can be considered.

## Evidence Collection

During a clean window, collect the fixed workload exactly once:

```bash
.venv/bin/python scripts/benchmark/run_e8_quality_baseline_reseed.py --execute \
  --output-dir /mnt/raid0/llm/epyc-root/artifacts/operator/e8_quality_baseline_evidence_20260726
```

The runner pins the receipt-authorized T1/T2 vectors before evaluation, forces
generation and scoring concurrency 3, and makes three independent repetitions for
each tier. A durable watcher records exact API 6/6 groups, AutoPilot absence,
exclusive ownership of all 24 listener/PID identities, full cmdlines, role
topology, and immutable file hashes throughout the run. Monitor records are
fully appended before fsync; any failed sample or persistence error aborts
before another repetition.
Every fixed-vector `llm_judge` row has exactly one sealed outcome trace per
repetition: four T1 rows plus 38 T2 rows. A temporary process-local wrapper
captures substring fast paths and network requests, while blank outputs produce
an explicit `blank_fast_failure` trace. Each trace binds its tier, repetition,
ordinal, and qid, and records exact inputs, prompt/request, target/role, raw
response/status, parsed verdict, timestamps, correlation hash, and scorer-source
hashes. The runner independently replays all scores before accepting a repetition.
The runner writes all response ledgers, private scoring vectors, judge traces,
and EvalTower sidecars inside a mode-0700 staging bundle, then publishes only a
successful bundle with create-only `renameat2(RENAME_NOREPLACE)` and
`run_seal.json`.
A nonzero exit or `decision_grade: false` leaves no validator-acceptable evidence;
do not reuse hidden staging artifacts.

## Human Apply Boundary

Evidence is not a baseline write. The operator must review the source hashes and
run the existing validator before the separate human-only atomic apply
transaction is considered:

```bash
bash /mnt/raid0/llm/epyc-root/artifacts/operator/prepare_e8_quality_baseline_reseed_20260726.sh \
  --validate-evidence /mnt/raid0/llm/epyc-root/artifacts/operator/e8_quality_baseline_evidence_20260726/e8_quality_baseline_evidence.json
```

This validator does not apply values either. It is the required handoff point to
the later human-reviewed baseline state transaction. It requires every evidence
artifact to resolve inside the published bundle, matches response rows to the
ratified vector in exact order, reconstructs both vectors from the source-pinned
pool, independently re-scores every answer (replaying sealed judge transcripts
without inference), matches the durable EvalTower sidecar row-for-row, and
recomputes overall and per-suite quality. It also verifies current repository
heads, the frozen `/mnt/raid0/llm/llama.cpp` source branch
`production-consolidated-v8` at
`67a433bf45a8a091d83b4ea0b32ff0735fd51800`, and every runtime artifact's canonical path, device/inode, size,
nanosecond mtime, and SHA-256.
