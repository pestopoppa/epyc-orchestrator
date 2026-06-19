# N9 Routing Rollout Evidence Pack

Generated: 2026-06-19T09:59:40Z
Scope: orchestrator routing-retrain evidence only
Decision boundary: do not enable live routing flags

## Current State

- `handoffs/active/retrain-routing-models.md` is still ACTIVE/PARTIAL.
- The retrain blocker is no longer label volume; the handoff and recent research notes point to BGE re-embed / embedding freshness as the operative unblock path.
- `orchestration/repl_memory/routing_classifier_weights.npz` exists and is staged as the classifier artifact, but production routing remains off.

## Live Artifact Check

- Path: `orchestration/repl_memory/routing_classifier_weights.npz`
- Size: 522,921 bytes
- Timestamp: 2026-06-12 23:49:27 UTC
- Keys:
  - `W1` `(1031, 128)`
  - `b1` `(128,)`
  - `W2` `(128, 64)`
  - `b2` `(64,)`
  - `W3` `(64, 8)`
  - `b3` `(8,)`
  - `_label_map_keys` `(8,)`
  - `_label_map_vals` `(8,)`
  - `_config` `(4,)`
  - `_threshold_keys` `(8,)`
  - `_threshold_vals` `(8,)`

## Script State

- `scripts/maintenance/repair_episodic_embeddings.py`
  - `diagnose()` reports FAISS / `reembedded.npz` health.
  - `run_repair()` performs re-embed + atomic FAISS/id-map rebuild.
  - The script is wired into the startup stack through `scripts/server/stack_commands.py:cmd_start`, so changes here have startup surface area.
- `scripts/graph_router/extract_training_data.py`
  - Default output: `orchestration/repl_memory/training_data.npz`
  - Uses `orchestration/repl_memory/sessions/reembedded.npz` when present.
- `scripts/graph_router/train_routing_classifier.py`
  - Default weights output: `orchestration/repl_memory/routing_classifier_weights.npz`
  - Trains a 2-layer MLP and writes the promoted bundle in place.

## GitNexus Impact

Commands run on 2026-06-19:

```bash
gitnexus status
gitnexus impact Function:scripts/graph_router/extract_training_data.py:extract --direction upstream --repo epyc-orchestrator
gitnexus impact Function:scripts/graph_router/train_routing_classifier.py:train --direction upstream --repo epyc-orchestrator
gitnexus impact Function:scripts/maintenance/repair_episodic_embeddings.py:diagnose --direction upstream --repo epyc-orchestrator
```

Results:

- `extract_training_data.py:extract` -> LOW risk, impactedCount 3, direct caller `main`, downstream import chain reaches `reembed_episodic_store.py`
- `train_routing_classifier.py:train` -> LOW risk, impactedCount 2, direct caller `main`
- `repair_episodic_embeddings.py:diagnose` -> LOW risk, impactedCount 6, reaches `scripts/server/stack_commands.py:cmd_start`

## Validation Performed

```bash
stat -c '%n %s bytes %y' /mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/routing_classifier_weights.npz
cd /mnt/raid0/llm/epyc-orchestrator && .venv/bin/python - <<'PY'
import numpy as np
from pathlib import Path
p = Path('orchestration/repl_memory/routing_classifier_weights.npz')
d = np.load(p, allow_pickle=True)
print('keys', d.files)
for k in d.files:
    arr = d[k]
    if hasattr(arr, 'shape'):
        print(k, arr.shape, getattr(arr, 'dtype', None))
    else:
        print(k, type(arr))
PY
```

## Rollout Gate

- Keep `ORCHESTRATOR_ROUTING_CLASSIFIER=0` unless the operator explicitly approves a clean-window rollout.
- Do not enable `ORCHESTRATOR_GRAPH_ROUTER` or `ORCHESTRATOR_SKILLBANK` from this evidence pass.
- This pass is evidence-only; no retrain, repair, or production flip was executed.
