# EV-15e paired distributional consumer

This is an offline planning and readout instrument. It does not call a model,
validate a tokenizer or template, set an acceptance threshold, or emit a belief
claim. Native acquisition still requires the EV-15b position/token contract and
EV-15c/d guards. Existing mathematical readouts are unchanged.

Create a manifest with pinned model and scorer identities and hashes of each
task context and candidate answer. Hashes identify the exact bytes the future
native producer must resolve; they do not prove that producer used those bytes.

```json
{
  "schema": "ev15-comparison-manifest.v1",
  "model_id": "model-revision-and-runtime-pin",
  "scorer_id": "scorer-and-prompt-revision",
  "items": [{
    "item_id": "example-1",
    "context_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "candidates": [
      {"candidate_id": "A", "answer_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
      {"candidate_id": "B", "answer_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
    ]
  }]
}
```

The hashes above are schema examples, not native evidence. Generate the complete
request plan without inference:

```bash
python -m src.typed_decisions.paired_distributional_readout plan manifest.json --output plan.json
```

Every unordered pair gets two explicit requests: AB and BA. Each request ID binds
the item, context, candidate IDs and answer hashes, model/scorer pins, and actual
display order. Candidate-list order determines the reference orientation of the
reported scalar. The plan must not be edited after generation.

The producer supplies a JSON list of capture objects. Each object copies all
seven identity fields of one planned order (`request_id`, `model_id`, `scorer_id`,
`item_id`, `context_sha256`, `left`, `right`) and adds:

* `capture_id`: a distinct locator for that independently observed presentation;
* `probabilities_left`: nine probabilities over scores 1 through 9 for the left candidate;
* `probabilities_right`: nine probabilities over scores 1 through 9 for the right candidate.

These are already validated, normalized candidate distributions, not raw logits,
truncated logprob heads or inferred complements. The producer must retain its
native records and independently establish their admissibility; this consumer
checks the declared identity and probability contract only. Capture IDs prevent
literal reuse of one record for both orders but cannot attest its truth.

```bash
python -m src.typed_decisions.paired_distributional_readout readout plan.json captures.json --method mean --output readout.json
```

All planned presentations must be present exactly once, including both orders of
every pair for every item. Unknown, duplicate, missing or mismatched identities,
reused capture IDs, malformed probabilities and edited plans refuse. Output paths
must be new; refusal leaves unrelated existing outputs intact.

For pair A/B, the consumer averages A-left from AB with A-right from BA, and
B-right from AB with B-left from BA. It then applies the selected existing
distributional readout exactly once to these two averaged distributions. Averaging
preferences or readout results is not equivalent for nonlinear methods. The
report retains aligned distributions, both request IDs, model/scorer identities,
and hashes of the entire input plan and capture set. It reports no winner,
promotion verdict, native-validity finding or measurement grade.

Tests include a nonlinear counterexample where pre-aggregation returns +1 while
post-aggregation would return 0, candidate-order reversal, all eight unchanged
readouts, missing complete pairs, cross-item substitutions and actual CLI refusal
and no-clobber controls. Synthetic fixtures establish software behavior only.
