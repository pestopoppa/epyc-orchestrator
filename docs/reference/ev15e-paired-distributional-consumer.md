# EV-15e paired distributional consumer

This is an offline planning and readout instrument. It does not call a model,
validate a tokenizer or template, or emit a belief claim. Native acquisition
still requires the EV-15b position/token contract and EV-15c validation. The
EV-15d parser records and gates each call's retained score-token mass before
this paired consumer can use its normalized distribution. Existing mathematical
readouts are unchanged.

Create a manifest with pinned model and scorer identities and hashes of each
task context and candidate answer. Hashes identify the exact bytes the future
native producer must resolve; they do not prove that producer used those bytes.

```json
{
  "schema": "ev15-comparison-manifest.v2",
  "model_id": "model-revision-and-runtime-pin",
  "scorer_id": "scorer-and-prompt-revision",
  "score_token_ids": [101, 102, 103, 104, 105, 106, 107, 108, 109],
  "minimum_retained_mass": 0.95,
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
the item, context, candidate IDs and answer hashes, model/scorer pins, actual
display order, ordered K=9 score-token IDs and explicit retained-mass floor. The
plan binds the same ordered IDs and floor, so captures cannot change the score
alphabet, score ordering or floor. Candidate-list order determines the reference
orientation of the reported scalar. The plan must not be edited after generation.

The producer supplies a JSON list of capture objects. Each object copies all
seven identity fields of one planned order (`request_id`, `model_id`, `scorer_id`,
`item_id`, `context_sha256`, `left`, `right`) and adds a distinct `capture_id`.
The EV-15d raw-head form supplies `score_head_left` and `score_head_right`. Each
per-call head has this shape under a v2 plan:

```json
{
  "entries": [
    {"id": 101, "logprob": -2.3},
    {"id": 102, "logprob": -1.7}
  ]
}
```

`score_token_ids` must be resolved for the pinned scorer/tokenizer; this CLI does
not establish that mapping. Each head's `entries` may use a finite `logprob <= 0`
or an explicit finite `prob` in `[0, 1]`, with an integer token `id`. Duplicate
IDs and a head whose total probability exceeds `1 + 1e-6` refuse. Missing score
IDs contribute zero and are listed. The consumer sums matching raw probabilities
first, records `retained_mass` for that call, and compares it with the plan's
explicit floor. It normalizes over the score support only after the call passes
the floor; a zero-mass or under-floor call returns a typed refusal with no
judgment. It never pools head entries or retained mass across calls. A CLI
refusal writes the per-call audit JSON and exits 2 without producing comparisons.

The original v1 manifest/plan contract and request IDs remain unchanged for the
older `probabilities_left` / `probabilities_right` math-only fixtures. A v1
manifest has the original fields (`schema`, `model_id`, `scorer_id`, `items`)
and uses `"schema": "ev15-comparison-manifest.v1"`; it does not bind token IDs
or a mass floor. V1 output remains the original v1 result shape. V1 plans reject
raw score heads, and v2 plans require them, so neither path can silently borrow
the other's warrant. The legacy vector path does not carry retained mass and
never establishes native validity. Capture IDs prevent literal reuse of one
record for both orders but cannot attest a producer's truth.

```bash
python -m src.typed_decisions.paired_distributional_readout readout plan.json captures.json --method mean --output readout.json
```

All planned presentations must be present exactly once, including both orders of
every pair for every item. Unknown, duplicate, missing or mismatched identities,
reused capture IDs, malformed heads and edited plans refuse. Under-floor calls
produce an explicit refusal record retaining each call's mass and missing IDs;
no comparison is emitted. Output paths must be new; refusal leaves unrelated
existing outputs intact.

For pair A/B, the consumer averages A-left from AB with A-right from BA, and
B-right from AB with B-left from BA. It then applies the selected existing
distributional readout exactly once to these two averaged distributions. Each
input head passed its own mass guard before averaging. Averaging preferences or
readout results is not equivalent for nonlinear methods. The report retains
aligned distributions, both request IDs, model/scorer identities, per-call head
audits, and hashes of the entire input plan and capture set. It reports no winner,
promotion verdict, native-validity finding or measurement grade.

Tests include mass-floor refusal, missing score tokens as zero, duplicate and
invalid tokens, separate call-level retained mass, a nonlinear counterexample
where pre-aggregation returns +1 while post-aggregation would return 0,
candidate-order reversal, all eight unchanged readouts, missing complete pairs,
cross-item substitutions and CLI refusal/no-clobber controls. Synthetic fixtures
establish software behavior only.
