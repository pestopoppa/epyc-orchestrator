# P4.5 Journal Routing Analysis

Per-suite per-role correctness extracted from autopilot journal.
Temperature τ=2.0 for soft labels.

## Statistically Robust Routing Misses

Suites where a non-frontdoor route's Wilson 95% lower bound exceeds
frontdoor's Wilson 95% upper bound — BOTH with n>=20. These are genuine
routing gains, not sample-size noise. (Naive max-rate scans surface n=1
flukes like simpleqa's 'architect 100%' which is a single lucky draw —
those are excluded here.)

- **cruxeval**: worker_general 87% (n=188, CI_lo=82%) BEATS frontdoor 0% (n=22, CI_hi=15%) — gain +87%
- **cruxeval**: coder_escalation 47% (n=74, CI_lo=36%) BEATS frontdoor 0% (n=22, CI_hi=15%) — gain +47%
- **bigcodebench**: coder_escalation 69% (n=109, CI_lo=60%) BEATS frontdoor 33% (n=165, CI_hi=40%) — gain +36%
- **gpqa**: coder_escalation 65% (n=37, CI_lo=49%) BEATS frontdoor 39% (n=245, CI_hi=45%) — gain +26%
- **general**: architect_general 98% (n=48, CI_lo=89%) BEATS frontdoor 84% (n=591, CI_hi=87%) — gain +14%

**Capability-ceiling suites (NOT routing misses)**: suites where all
routes score similarly low (e.g. simpleqa ~5% across every route) are a
model-capability/benchmark-difficulty ceiling — obscure factual recall
that small quantized local models genuinely cannot do. Re-routing will
not help; only a larger/RAG-augmented model would.

## Per-Suite Per-Role Correctness Table

| Suite | frontdoor | coder_escala | worker_gener | architect_ge | ingest_long_ | worker_visio | N total |
|-------|---|---|---|---|---|---|---------|
| general | 84% | 77% | — | 98% | — | 58% | 1291 |
| hotpotqa | (100%)* | 99% | 64% | — | — | 22% | 1154 |
| simpleqa | 5% | (25%)* | 6% | (100%)* | 14% | 10% | 881 |
| thinking | 97% | (100%)* | — | (100%)* | (100%)* | (100%)* | 758 |
| math | 94% | 100% | 98% | (0%)* | — | (100%)* | 732 |
| livecodebench | 100% | (100%)* | — | 100% | — | (100%)* | 718 |
| debugbench | 80% | — | — | 100% | — | — | 605 |
| vl | — | — | — | — | — | 62% | 601 |
| instruction_precision | 90% | (67%)* | 100% | (100%)* | 100% | 100% | 587 |
| gpqa | 39% | 65% | (0%)* | (100%)* | — | (0%)* | 585 |
| cruxeval | 0% | 47% | 87% | — | — | — | 585 |
| bigcodebench | 33% | 69% | — | 0% | — | (33%)* | 583 |
| skill_transfer | (0%)* | 33% | 35% | — | — | 76% | 583 |
| mode_advantage | 63% | 89% | — | 100% | (0%)* | — | 581 |
| coder | 87% | 82% | (100%)* | 73% | — | — | 580 |
| long_context | 99% | (100%)* | (100%)* | (100%)* | (100%)* | (100%)* | 577 |
| agentic | 71% | — | (100%)* | 94% | (0%)* | (100%)* | 575 |
| mode_advantage_hard | 29% | 0% | (100%)* | — | — | 13% | 574 |
| tool_use | — | — | — | — | — | — | 80 |

\* Fewer than 5 examples — low confidence.

## Routing Recommendations (from soft labels)

| Suite | Recommended role | Confidence (soft label mass) |
|-------|-----------------|------------------------------|
| agentic | architect_general | 22.8% |
| bigcodebench | coder_escalation | 21.4% |
| coder | frontdoor | 20.6% |
| cruxeval | worker_general | 22.7% |
| debugbench | architect_general | 23.1% |
| general | architect_general | 20.5% |
| gpqa | coder_escalation | 21.0% |
| hotpotqa | coder_escalation | 23.0% |
| instruction_precision | worker_general | 19.4% |
| livecodebench | frontdoor | 22.6% |
| long_context | frontdoor | 24.7% |
| math | coder_escalation | 20.9% |
| mode_advantage | architect_general | 21.8% |
| mode_advantage_hard | frontdoor | 18.6% |
| simpleqa | ingest_long_context | 17.4% |
| skill_transfer | worker_vision | 21.4% |
| thinking | frontdoor | 24.5% |
| vl | worker_vision | 21.4% |

## Next Steps for MLP Retraining

The `soft_labels.jsonl` dataset has per-qid soft label distributions.
To complete P4.5 MLP retraining:

1. **Embed question texts** (requires BGE server at port 8090/8091):
   ```
   python3 scripts/graph_router/embed_soft_label_dataset.py \
       --soft-labels orchestration/reports/p45_soft_labels/soft_labels.jsonl \
       --output orchestration/reports/p45_soft_labels/soft_labels_embedded.npz
   ```

2. **Retrain MLP via KL divergence**:
   ```
   python3 scripts/graph_router/train_routing_classifier_kl.py \
       --data orchestration/reports/p45_soft_labels/soft_labels_embedded.npz \
       --output orchestration/repl_memory/routing_classifier_weights_kl.npz
   ```

3. **A/B against hard-label baseline** (val acc gate: ≥1 pp improvement).

Alternatively, apply `suite_priors.json` as label smoothing to the
existing episodic memory training set (if suite-type classification is added).
