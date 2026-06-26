# P4.5 Journal Routing Analysis

Per-suite per-role correctness extracted from autopilot journal.
Temperature τ=2.0 for soft labels.

## Key Findings

Severe routing failures (pass rate <30% for frontdoor with a better alternative):

- **cruxeval**: frontdoor 0.0% → worker_general 87.2% (+87.2%)
- **simpleqa**: frontdoor 4.9% → ingest_long_context 14.3% (+9.4%)
- **mode_advantage_hard**: frontdoor 29.5% → worker_vision 13.3% (+-16.1%)

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
