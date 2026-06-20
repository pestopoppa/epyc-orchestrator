# Real-Task Corpus W2 Harvest Report

- Window: `2026-06-14` through `2026-06-20`
- Source: orchestrator progress JSONL
- Output: `real_tasks.training_eligible.compact.jsonl`
- Emitted training-eligible records: 372
- Duplicate prompt attempts represented: 2609
- Records with outcome: 372 / 372
- Records with wall time: 372 / 372
- Records with token payloads: 0 / 372
- Prompt text included: no

## Gate Status

| Requirement | Status | Evidence |
|---|---|---|
| >=100 real records with class+outcome | pass | 372 training-eligible records emitted |
| 2-week normal-use soak | partial | harvest covers 2026-06-14..2026-06-20 only |
| token completeness | open | 0/372 emitted records include token payloads |

## By Class

| Class | Records |
|---|---:|
| benchmark_eval_measurement | 33 |
| code_change_implementation | 193 |
| debug_root_cause | 17 |
| governance_docs_handoff | 21 |
| ops_deploy_process | 34 |
| planning_architecture_review | 29 |
| research_intake_deep_dive | 45 |

## By Outcome

| Outcome | Records |
|---|---:|
| failure | 79 |
| success | 293 |

## Source Manifest

```json
{
  "by_class": {
    "benchmark_eval_measurement": 33,
    "code_change_implementation": 193,
    "debug_root_cause": 17,
    "governance_docs_handoff": 21,
    "ops_deploy_process": 34,
    "planning_architecture_review": 29,
    "research_intake_deep_dive": 45
  },
  "by_outcome": {
    "failure": 79,
    "success": 293
  },
  "by_source": {
    "orchestrator_progress_jsonl": 372
  },
  "duplicates_collapsed": 2237,
  "synthetic_like": 0,
  "taxonomy_class": 372,
  "training_eligible": 372,
  "written": 372
}
```
